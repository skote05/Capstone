#!/usr/bin/env python3
"""
Unified Weed Detection - Annotated GT + Region-based metrics + Noise filtering
- MIN_REGION_SIZE = 500 px (predicted regions below this are ignored)
- Region match rule = any overlap (same as earlier COCO-based code behavior)
- GT extracted from annotated images via color rules + dilation
- Special paddy-only handling:
    * GT == 0 && PRED == 0 => all metrics = 1.0
    * GT == 0 && PRED > 0  => recall = 1.0, precision/f1/iou = 0.0 (pixel accuracy still computed)
"""

import os
import time
import gc
from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ==================== CONFIGURATION ====================

BASE_DIR = Path(__file__).parent
ORIGINAL_FOLDER = BASE_DIR / 'data' / 'original_images'
ANNOTATED_FOLDER = BASE_DIR / 'data' / 'annotated_images'
MODEL_PATH = BASE_DIR / 'data' / 'model' / 'paddy_detection_model_combined.pth'
RESULTS_DIR = BASE_DIR / 'results'

RESULTS_DIR.mkdir(exist_ok=True)

device = torch.device("cpu")
torch.set_num_threads(2)

# image and inference settings
RESIZE_TO = (800, 800)
DETECTION_SCALES = [1.0, 1.25, 1.5, 1.75, 2.0]    # kept as your memory-optimized choice
CONFIDENCE_THRESHOLD = 0.8     # from your memory-optimized script
NMS_THRESHOLD = 0.2              # from your memory-optimized script

# Ground truth dilation (from your memory-optimized script)
GT_DILATION_KERNEL_SIZE = 15
GT_DILATION_ITERATIONS = 3

# NEW / chosen by you
MIN_REGION_SIZE = 500    # predicted regions smaller than this (pixels) are discarded
REGION_IOU_MATCH_REQ = 0.0  # match if any overlap (same as COCO version behavior)

SAVE_INTERVAL = 25

print("="*70)
print("Unified Weed Detection (Annotated GT + Region evaluation + Noise filtering)")
print(f"MIN_REGION_SIZE = {MIN_REGION_SIZE} px, Region match IoU requirement = any overlap")
print("="*70)

# ==================== MODEL LOADING ====================

def load_paddy_model(model_path, num_classes=2):
    print("\nLoading model...")
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    del checkpoint
    gc.collect()
    print("✅ Model loaded\n")
    return model

# ==================== IMAGE / MASK UTILITIES ====================

def resize_to_training_size(image, target_size=RESIZE_TO):
    """Resize with padding to preserve aspect ratio (same behavior as your first script)."""
    h, w = image.shape[:2]
    target_h, target_w = target_size
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas, scale, (x_offset, y_offset)

def extract_shape_from_bbox(image_rgb, bbox, expansion=15):
    """Extract paddy shape from a bbox area (same approach used earlier)."""
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = bbox.astype(int)
    x1, y1 = max(0, x1-expansion), max(0, y1-expansion)
    x2, y2 = min(w, x2+expansion), min(h, y2+expansion)
    roi = image_rgb[y1:y2, x1:x2]
    if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
        return np.zeros((h, w), dtype=np.uint8)
    r, g, b = roi[:,:,0], roi[:,:,1], roi[:,:,2]
    green_mask = ((g > r + 5) & (g > b + 5) & (g > 25)).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_DILATE, kernel)
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = green_mask
    return full_mask

def weed_segmentation(image_rgb, paddy_mask, exg_threshold=20, green_ratio_threshold=1.1):
    """Pixel-based weed segmentation using EXG + green ratio + green dominance (same as your functions)."""
    r = image_rgb[:,:,0].astype(np.float32)
    g = image_rgb[:,:,1].astype(np.float32)
    b = image_rgb[:,:,2].astype(np.float32)
    exg = 2.0 * g - r - b
    exg_mask = (exg > exg_threshold).astype(np.uint8)
    green_ratio = g / (r + b + 1)
    ratio_mask = (green_ratio > green_ratio_threshold).astype(np.uint8)
    green_dom = ((g > r + 5) & (g > b + 5) & (g > 30)).astype(np.uint8)
    veg_mask = ((exg_mask + ratio_mask + green_dom) >= 2).astype(np.uint8) * 255
    weed_mask = veg_mask.copy()
    weed_mask[paddy_mask > 0] = 0
    weed_mask = cv2.morphologyEx(weed_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    weed_mask = cv2.morphologyEx(weed_mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    # cleanup
    del r, g, b, exg, exg_mask, green_ratio, ratio_mask, green_dom, veg_mask
    return weed_mask

def extract_ground_truth_weed(annotated_rgb):
    """
    Convert annotated image (colored labels) into weed mask.
    This follows the color-extraction rules from your memory-optimized script.
    """
    r, g, b = annotated_rgb[:,:,0], annotated_rgb[:,:,1], annotated_rgb[:,:,2]
    # Conservative color rules - tuned to annotated format
    green_mask = ((g > r + 20) & (g > b + 20) & (g > 80)).astype(np.uint8) * 255
    max_ch = np.maximum(np.maximum(r, g), b)
    colored = ((max_ch > 80) & ((r + g + b) > 150)).astype(np.uint8) * 255
    weed_mask = colored.copy()
    weed_mask[green_mask > 0] = 0  # remove paddy-colored regions
    # Dilation/cleaning (dilate annotated - your GT dilation)
    kernel_large = np.ones((GT_DILATION_KERNEL_SIZE, GT_DILATION_KERNEL_SIZE), np.uint8)
    weed_mask = cv2.morphologyEx(weed_mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
    weed_mask = cv2.morphologyEx(weed_mask, cv2.MORPH_DILATE, kernel_large, iterations=GT_DILATION_ITERATIONS)
    kernel_small = np.ones((5, 5), np.uint8)
    weed_mask = cv2.morphologyEx(weed_mask, cv2.MORPH_OPEN, kernel_small)
    weed_mask = cv2.morphologyEx(weed_mask, cv2.MORPH_CLOSE, kernel_small)
    del r, g, b, green_mask, max_ch, colored, kernel_large, kernel_small
    return weed_mask

# ==================== INFERENCE (multi-scale) ====================

@torch.no_grad()
def generate_weed_prediction_multiscale(image_rgb, model):
    """
    Run detector at multiple scales, produce paddy mask, then derive weed mask.
    Predicted boxes labeled '1' are used to make paddy_mask (same assumption as original code).
    """
    h, w = image_rgb.shape[:2]
    all_boxes, all_scores, all_labels = [], [], []
    for scale in DETECTION_SCALES:
        try:
            if scale != 1.0:
                new_h, new_w = int(h * scale), int(w * scale)
                if new_h > 1600 or new_w > 1600:
                    continue
                scaled_img = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                scaled_img = image_rgb
            img_tensor = transforms.ToTensor()(scaled_img).unsqueeze(0).to(device)
            outputs = model(img_tensor)[0]
            boxes = outputs['boxes'].cpu().numpy()
            scores = outputs['scores'].cpu().numpy()
            labels = outputs['labels'].cpu().numpy()
            del img_tensor, outputs
            if scale != 1.0:
                boxes = boxes / scale
                del scaled_img
            keep = scores >= CONFIDENCE_THRESHOLD
            if keep.sum() > 0:
                all_boxes.append(boxes[keep])
                all_scores.append(scores[keep])
                all_labels.append(labels[keep])
        except Exception as e:
            print(f"Error at scale {scale}: {e}")
            continue
    if len(all_boxes) > 0 and sum(len(b) for b in all_boxes) > 0:
        combined_boxes = np.concatenate(all_boxes)
        combined_scores = np.concatenate(all_scores)
        combined_labels = np.concatenate(all_labels)
        keep_indices = nms(
            torch.from_numpy(combined_boxes).float(),
            torch.from_numpy(combined_scores).float(),
            iou_threshold=NMS_THRESHOLD
        ).numpy()
        final_boxes = combined_boxes[keep_indices]
        final_labels = combined_labels[keep_indices]
        del combined_boxes, combined_scores, combined_labels
    else:
        final_boxes = np.array([]); final_labels = np.array([])
    paddy_mask = np.zeros((h, w), dtype=np.uint8)
    if len(final_boxes) > 0:
        for box in final_boxes[final_labels == 1]:
            paddy_mask = cv2.bitwise_or(paddy_mask, extract_shape_from_bbox(image_rgb, box))
    weed_mask = weed_segmentation(image_rgb, paddy_mask)
    del paddy_mask
    gc.collect()
    return weed_mask

# ==================== REGION HELPERS ====================

def filter_predicted_regions_by_size(pred_mask, min_size=MIN_REGION_SIZE):
    """Remove predicted connected components smaller than min_size (in pixels)."""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((pred_mask>0).astype(np.uint8), connectivity=8)
    out_mask = np.zeros_like(pred_mask, dtype=np.uint8)
    kept_labels = []
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_size:
            out_mask[labels == label] = 255
            kept_labels.append(label)
    return out_mask, labels, stats, kept_labels

def get_connected_components(mask):
    """Return labels and stats (connectedComponentsWithStats)."""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((mask>0).astype(np.uint8), connectivity=8)
    return num_labels, labels, stats

def compute_iou_between_masks(mask_a, mask_b):
    """Compute IoU between two binary masks."""
    inter = np.logical_and(mask_a>0, mask_b>0).sum()
    union = np.logical_or(mask_a>0, mask_b>0).sum()
    return (inter / union) if union > 0 else 0.0, inter

# ==================== METRICS (region + pixel) ====================

def calculate_region_and_pixel_metrics(gt_mask, pred_mask, min_region_size=MIN_REGION_SIZE):
    """Return both region-based metrics and pixel metrics for a single image."""
    # Pixel-level metrics first
    gt_bin = (gt_mask > 0).astype(int).flatten()
    pred_bin = (pred_mask > 0).astype(int).flatten()
    # handle special GT=0 cases per your instruction:
    gt_pixels = int(gt_bin.sum())
    pred_pixels = int(pred_bin.sum())
    pixel_accuracy = accuracy_score(gt_bin, pred_bin) if gt_bin.size > 0 else 0.0

    # Special rules when GT has zero weed pixels
    if gt_pixels == 0:
        if pred_pixels == 0:
            # both empty => everything perfect
            return {
                'pixel_accuracy': 1.0,
                'precision': 1.0,
                'recall': 1.0,
                'f1_score': 1.0,
                'iou': 1.0,
                'gt_pixels': 0,
                'pred_pixels': 0,
                # region metrics
                'region_precision': 1.0,
                'region_recall': 1.0,
                'region_f1': 1.0,
                'region_iou': 1.0,
                'gt_regions': 0,
                'pred_regions': 0,
                'matched_gt_regions': 0,
                'matched_pred_regions': 0
            }
        else:
            # GT zero, predictions non-zero => special: recall = 1, others 0 per your instruction
            return {
                'pixel_accuracy': pixel_accuracy,
                'precision': 0.0,
                'recall': 1.0,
                'f1_score': 0.0,
                'iou': 0.0,
                'gt_pixels': 0,
                'pred_pixels': pred_pixels,
                'region_precision': 0.0,
                'region_recall': 1.0,
                'region_f1': 0.0,
                'region_iou': 0.0,
                'gt_regions': 0,
                'pred_regions': None,
                'matched_gt_regions': 0,
                'matched_pred_regions': 0
            }

    # For non-empty GT, do region-based matching
    # Filter predicted small regions
    pred_filtered_mask, pred_labels, pred_stats, kept_labels = filter_predicted_regions_by_size(pred_mask, min_size=min_region_size)
    # GT connected components (do not filter GT as you answered 'no')
    gt_num_labels, gt_labels, gt_stats = get_connected_components(gt_mask)
    gt_regions = gt_num_labels - 1

    pred_num_labels, _, pred_stats2 = get_connected_components(pred_filtered_mask)
    pred_regions = pred_num_labels - 1

    matched_gt = set()
    matched_pred = set()
    ious = []

    # Build GT region masks once (for speed)
    gt_region_masks = {}
    for g_label in range(1, gt_num_labels):
        region_mask = (gt_labels == g_label).astype(np.uint8)
        gt_region_masks[g_label] = region_mask

    # Iterate predicted regions and try to match to a GT region (any overlap -> match)
    for p_label in range(1, pred_num_labels):
        pred_region_mask = (pred_labels == p_label).astype(np.uint8)
        # skip empty
        if pred_region_mask.sum() == 0:
            continue
        # Try all GT regions for any overlap
        for g_label, g_mask in gt_region_masks.items():
            inter = np.logical_and(pred_region_mask > 0, g_mask > 0).sum()
            if inter > 0:
                union = np.logical_or(pred_region_mask > 0, g_mask > 0).sum()
                iou = inter / union if union > 0 else 0.0
                ious.append(iou)
                matched_gt.add(g_label)
                matched_pred.add(p_label)
                # once matched, do not match the same pred to multiple GTs (but allow GT matched to multiple preds if needed)
                break

    num_matched_gt = len(matched_gt)
    num_matched_pred = len(matched_pred)

    # Region-based precision / recall / f1
    region_precision = num_matched_pred / max(pred_regions, 1)
    region_recall = num_matched_gt / max(gt_regions, 1)
    region_f1 = 2 * (region_precision * region_recall) / (region_precision + region_recall + 1e-10)
    region_iou = np.mean(ious) if ious else 0.0

    # Pixel metrics (with sklearn helpers but safe zero_division)
    precision = precision_score(gt_bin, pred_bin, zero_division=0)
    recall = recall_score(gt_bin, pred_bin, zero_division=0)
    f1 = f1_score(gt_bin, pred_bin, zero_division=0)
    inter = np.logical_and(gt_bin, pred_bin).sum()
    union = np.logical_or(gt_bin, pred_bin).sum()
    iou = inter / union if union > 0 else 0.0

    return {
        'pixel_accuracy': pixel_accuracy,
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'iou': float(iou),
        'gt_pixels': int(gt_pixels),
        'pred_pixels': int(pred_pixels),
        'region_precision': float(region_precision),
        'region_recall': float(region_recall),
        'region_f1': float(region_f1),
        'region_iou': float(region_iou),
        'gt_regions': int(gt_regions),
        'pred_regions': int(pred_regions),
        'matched_gt_regions': int(num_matched_gt),
        'matched_pred_regions': int(num_matched_pred)
    }

# ==================== MAIN ====================

def main():
    start_time = time.time()
    try:
        original_files = [f for f in os.listdir(ORIGINAL_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        annotated_files = [f for f in os.listdir(ANNOTATED_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        annotated_map = {os.path.splitext(af)[0].lower(): af for af in annotated_files}
        matching_pairs = [(of, annotated_map[os.path.splitext(of)[0].lower()]) 
                          for of in original_files if os.path.splitext(of)[0].lower() in annotated_map]

        print(f"\nMatched pairs: {len(matching_pairs)}\n")
        if len(matching_pairs) == 0:
            print("No matching files between originals and annotated. Exiting.")
            return

        model = load_paddy_model(MODEL_PATH)

        metrics_list = []
        agg_gt, agg_pred = [], []
        error_count = 0

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = RESULTS_DIR / f'weed_results_unified_{timestamp}.csv'

        for idx, (orig_fname, annot_fname) in enumerate(tqdm(matching_pairs, desc="Processing"), 1):
            try:
                orig_bgr = cv2.imread(str(ORIGINAL_FOLDER / orig_fname))
                annot_bgr = cv2.imread(str(ANNOTATED_FOLDER / annot_fname))
                if orig_bgr is None or annot_bgr is None:
                    error_count += 1
                    continue

                # Resize with padding (keeps aspect and maps GT boxes accordingly)
                orig_resized, scale_o, offset_o = resize_to_training_size(orig_bgr, RESIZE_TO)
                annot_resized, scale_a, offset_a = resize_to_training_size(annot_bgr, RESIZE_TO)

                # Convert to RGB for processing
                orig_rgb = cv2.cvtColor(orig_resized, cv2.COLOR_BGR2RGB)
                annot_rgb = cv2.cvtColor(annot_resized, cv2.COLOR_BGR2RGB)

                # Extract GT and prediction
                gt_weed = extract_ground_truth_weed(annot_rgb)
                pred_weed = generate_weed_prediction_multiscale(orig_rgb, model)

                # Filter predicted small regions (noise removal)
                pred_filtered_mask, _, _, _ = filter_predicted_regions_by_size(pred_weed, min_size=MIN_REGION_SIZE)

                # Calculate metrics (region + pixel)
                m = calculate_region_and_pixel_metrics(gt_weed, pred_filtered_mask, min_region_size=MIN_REGION_SIZE)
                m['filename'] = orig_fname
                metrics_list.append(m)

                # For aggregated overall metrics, flatten and store
                agg_gt.append((gt_weed > 0).astype(np.uint8).flatten())
                agg_pred.append((pred_filtered_mask > 0).astype(np.uint8).flatten())

                # cleanup
                del orig_bgr, annot_bgr, orig_resized, annot_resized, orig_rgb, annot_rgb, gt_weed, pred_weed, pred_filtered_mask
                gc.collect()

                if idx % SAVE_INTERVAL == 0:
                    pd.DataFrame(metrics_list).to_csv(csv_path, index=False)
                    gc.collect()

            except Exception as e:
                print(f"\nError {orig_fname}: {e}")
                error_count += 1
                gc.collect()
                continue

        # Final save and aggregated results
        df = pd.DataFrame(metrics_list)
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Saved: {csv_path.name}")

        if len(agg_gt) > 0:
            gt_all = np.concatenate(agg_gt)
            pred_all = np.concatenate(agg_pred)
            elapsed = time.time() - start_time

            # Overall pixel-level scores
            overall_accuracy = accuracy_score(gt_all, pred_all)
            overall_precision = precision_score(gt_all, pred_all, zero_division=0)
            overall_recall = recall_score(gt_all, pred_all, zero_division=0)
            overall_f1 = f1_score(gt_all, pred_all, zero_division=0)
            inter = np.logical_and(gt_all, pred_all).sum()
            union = np.logical_or(gt_all, pred_all).sum()
            overall_iou = inter / union if union > 0 else 0.0

            print("\n" + "="*60)
            print("OVERALL METRICS")
            print("="*60)
            print(f"Processed: {len(metrics_list)}/{len(matching_pairs)}")
            print(f"Errors: {error_count}")
            print(f"Time: {elapsed/60:.1f} minutes")
            print(f"\nPixel-level:")
            print(f"  Accuracy:  {overall_accuracy:.4f}")
            print(f"  Precision: {overall_precision:.4f}")
            print(f"  Recall:    {overall_recall:.4f}")
            print(f"  F1-Score:  {overall_f1:.4f}")
            print(f"  IoU:       {overall_iou:.4f}")

            print("\nPer-image means (region + pixel):")
            print(f"  Mean Pixel Accuracy: {df['pixel_accuracy'].mean():.4f}")
            if 'precision' in df:
                print(f"  Mean Precision: {df['precision'].mean():.4f}")
                print(f"  Mean Recall:    {df['recall'].mean():.4f}")
                print(f"  Mean F1:        {df['f1_score'].mean():.4f}")
                print(f"  Mean IoU:       {df['iou'].mean():.4f}")
            if 'region_precision' in df:
                print(f"  Mean Region Precision: {df['region_precision'].mean():.4f}")
                print(f"  Mean Region Recall:    {df['region_recall'].mean():.4f}")
                print(f"  Mean Region F1:        {df['region_f1'].mean():.4f}")
                print(f"  Mean Region IoU:       {df['region_iou'].mean():.4f}")

            print("\n✅ COMPLETE")
            print("="*60)

    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback; traceback.print_exc()
        if 'metrics_list' in locals() and len(metrics_list) > 0:
            pd.DataFrame(metrics_list).to_csv(RESULTS_DIR / 'weed_results_partial.csv', index=False)
            print("Partial results saved")

if __name__ == "__main__":
    main()