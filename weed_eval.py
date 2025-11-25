#!/usr/bin/env python3
"""
Weed Detection Test - REGION-BASED METRICS WITH FLOOD FILL
If predicted region overlaps GT region, count ENTIRE regions as matched
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from tqdm import tqdm
import time
from pathlib import Path
import gc

# ==================== CONFIGURATION ====================

BASE_DIR = Path(__file__).parent
ORIGINAL_FOLDER = BASE_DIR / 'data' / 'originalimages_from_dataset1(greaater precision images)'
ANNOTATED_FOLDER = BASE_DIR / 'data' / 'annotated_images'
MODEL_PATH = BASE_DIR / 'data' / 'model' / 'paddy_model_with_test_data.pth'
RESULTS_DIR = BASE_DIR / 'results'

device = torch.device("cpu")
torch.set_num_threads(4)

RESIZE_TO = (800, 800)
DETECTION_SCALES = [1.0, 1.25, 1.5, 2.0]
CONFIDENCE_THRESHOLD = 0.95
NMS_THRESHOLD = 0.3
SAVE_INTERVAL = 50

RESULTS_DIR.mkdir(exist_ok=True)

print("="*60)
print("WEED DETECTION - REGION-BASED FLOOD FILL METRICS")
print("="*60)
print(f"Image size: {RESIZE_TO}")
print(f"Metric: Region overlap (flood fill logic)")
print("="*60)

# ==================== FUNCTIONS ====================

def load_paddy_model(model_path, num_classes=2):
    print("\nLoading model...")
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    print("✅ Model loaded\n")
    return model

def resize_to_training_size(image, target_size=RESIZE_TO):
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

def weed_segmentation(image_rgb, paddy_mask):
    r = image_rgb[:,:,0].astype(np.float32)
    g = image_rgb[:,:,1].astype(np.float32)
    b = image_rgb[:,:,2].astype(np.float32)
    
    exg = 2.0 * g - r - b
    exg_mask = (exg > 20).astype(np.uint8)
    green_ratio = g / (r + b + 1)
    ratio_mask = (green_ratio > 1.1).astype(np.uint8)
    green_dom = ((g > r + 5) & (g > b + 5) & (g > 30)).astype(np.uint8)
    
    veg_mask = ((exg_mask + ratio_mask + green_dom) >= 2).astype(np.uint8) * 255
    weed_mask = veg_mask.copy()
    weed_mask[paddy_mask > 0] = 0
    
    weed_mask = cv2.morphologyEx(weed_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    weed_mask = cv2.morphologyEx(weed_mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    return weed_mask

def extract_ground_truth_weed(annotated_rgb):
    r, g, b = annotated_rgb[:,:,0], annotated_rgb[:,:,1], annotated_rgb[:,:,2]
    green_mask = ((g > r + 20) & (g > b + 20) & (g > 80)).astype(np.uint8) * 255
    max_ch = np.maximum(np.maximum(r, g), b)
    colored = ((max_ch > 80) & ((r + g + b) > 150)).astype(np.uint8) * 255
    weed_mask = colored.copy()
    weed_mask[green_mask > 0] = 0
    return cv2.morphologyEx(weed_mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

@torch.no_grad()
def generate_weed_prediction_multiscale(image_rgb, model):
    h, w = image_rgb.shape[:2]
    all_boxes, all_scores, all_labels = [], [], []
    
    for scale in DETECTION_SCALES:
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
        
        keep = scores >= CONFIDENCE_THRESHOLD
        all_boxes.append(boxes[keep])
        all_scores.append(scores[keep])
        all_labels.append(labels[keep])
    
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
    else:
        final_boxes, final_labels = np.array([]), np.array([])
    
    paddy_mask = np.zeros((h, w), dtype=np.uint8)
    if len(final_boxes) > 0:
        for box in final_boxes[final_labels == 1]:
            paddy_mask = cv2.bitwise_or(paddy_mask, extract_shape_from_bbox(image_rgb, box))
    
    return weed_segmentation(image_rgb, paddy_mask)

def calculate_region_based_metrics(gt_mask, pred_mask):
    """
    FLOOD FILL LOGIC:
    1. Find all connected regions (blobs) in GT and Pred
    2. If ANY pixel overlap exists, count ENTIRE regions as matched
    3. Calculate metrics based on matched regions
    """
    
    # Find connected components (regions) using flood fill
    gt_num_labels, gt_labels, gt_stats, _ = cv2.connectedComponentsWithStats(
        gt_mask.astype(np.uint8), connectivity=8
    )
    
    pred_num_labels, pred_labels, pred_stats, _ = cv2.connectedComponentsWithStats(
        pred_mask.astype(np.uint8), connectivity=8
    )
    
    # Skip background (label 0)
    gt_num_regions = gt_num_labels - 1
    pred_num_regions = pred_num_labels - 1
    
    # Create masks for matched regions
    matched_gt_mask = np.zeros_like(gt_mask)
    matched_pred_mask = np.zeros_like(pred_mask)
    
    matched_gt_labels = set()
    matched_pred_labels = set()
    
    # Check each GT region against each Pred region
    for gt_label in range(1, gt_num_labels):
        gt_region_mask = (gt_labels == gt_label).astype(np.uint8)
        
        for pred_label in range(1, pred_num_labels):
            pred_region_mask = (pred_labels == pred_label).astype(np.uint8)
            
            # Check if ANY overlap exists
            overlap = np.logical_and(gt_region_mask, pred_region_mask).sum()
            
            if overlap > 0:  # ANY overlap = match!
                # Mark ENTIRE regions as matched
                matched_gt_mask[gt_region_mask > 0] = 255
                matched_pred_mask[pred_region_mask > 0] = 255
                matched_gt_labels.add(gt_label)
                matched_pred_labels.add(pred_label)
    
    # Calculate metrics based on matched regions
    num_matched_gt = len(matched_gt_labels)
    num_matched_pred = len(matched_pred_labels)
    
    # Region-level metrics
    region_precision = num_matched_pred / max(pred_num_regions, 1)
    region_recall = num_matched_gt / max(gt_num_regions, 1)
    region_f1 = 2 * (region_precision * region_recall) / (region_precision + region_recall + 1e-10)
    
    # Pixel-level metrics on MATCHED regions only
    matched_gt_pixels = (matched_gt_mask > 0).sum()
    matched_pred_pixels = (matched_pred_mask > 0).sum()
    total_gt_pixels = (gt_mask > 0).sum()
    total_pred_pixels = (pred_mask > 0).sum()
    
    # Overall pixel accuracy using matched regions
    pixel_intersection = np.logical_and(matched_gt_mask, matched_pred_mask).sum()
    pixel_union = np.logical_or(matched_gt_mask, matched_pred_mask).sum()
    pixel_iou = pixel_intersection / pixel_union if pixel_union > 0 else 0
    
    return {
        # Region-based metrics (PRIMARY)
        'region_precision': region_precision,
        'region_recall': region_recall,
        'region_f1': region_f1,
        'region_iou': pixel_iou,  # IoU of matched regions
        
        # Counts
        'gt_regions': gt_num_regions,
        'pred_regions': pred_num_regions,
        'matched_gt_regions': num_matched_gt,
        'matched_pred_regions': num_matched_pred,
        
        # Pixel counts
        'gt_pixels': int(total_gt_pixels),
        'pred_pixels': int(total_pred_pixels),
        'matched_gt_pixels': int(matched_gt_pixels),
        'matched_pred_pixels': int(matched_pred_pixels),
        
        # Coverage
        'gt_coverage': matched_gt_pixels / max(total_gt_pixels, 1),  # % of GT covered
        'pred_coverage': matched_pred_pixels / max(total_pred_pixels, 1)  # % of Pred that matched
    }

# ==================== MAIN ====================

def main():
    start_time = time.time()
    
    original_files = [f for f in os.listdir(ORIGINAL_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    annotated_files = [f for f in os.listdir(ANNOTATED_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    annotated_map = {os.path.splitext(af)[0].lower(): af for af in annotated_files}
    matching_pairs = [(of, annotated_map[os.path.splitext(of)[0].lower()]) 
                      for of in original_files if os.path.splitext(of)[0].lower() in annotated_map]
    
    print(f"\nFound {len(matching_pairs)} matching pairs\n")
    
    model = load_paddy_model(MODEL_PATH)
    
    print("Processing with FLOOD FILL region-based metrics...\n")
    
    metrics_list = []
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f'weed_results_region_based_{timestamp}.csv'
    
    for idx, (orig_fname, annot_fname) in enumerate(tqdm(matching_pairs, desc="Progress"), 1):
        try:
            orig_bgr = cv2.imread(str(ORIGINAL_FOLDER / orig_fname))
            annot_bgr = cv2.imread(str(ANNOTATED_FOLDER / annot_fname))
            
            if orig_bgr is None or annot_bgr is None:
                continue
            
            orig_resized, _, _ = resize_to_training_size(orig_bgr)
            annot_resized, _, _ = resize_to_training_size(annot_bgr)
            
            orig = cv2.cvtColor(orig_resized, cv2.COLOR_BGR2RGB)
            annot = cv2.cvtColor(annot_resized, cv2.COLOR_BGR2RGB)
            
            gt_weed = extract_ground_truth_weed(annot)
            pred_weed = generate_weed_prediction_multiscale(orig, model)
            
            # FLOOD FILL METRICS
            m = calculate_region_based_metrics(gt_weed, pred_weed)
            m['filename'] = orig_fname
            metrics_list.append(m)
            
            if idx % SAVE_INTERVAL == 0:
                pd.DataFrame(metrics_list).to_csv(csv_path, index=False)
                gc.collect()
            
        except Exception as e:
            print(f"\nError on {orig_fname}: {e}")
            continue
    
    df = pd.DataFrame(metrics_list)
    df.to_csv(csv_path, index=False)
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("RESULTS - REGION-BASED FLOOD FILL METRICS")
    print(f"{'='*60}")
    print(f"Processed: {len(metrics_list)} images")
    print(f"Time: {elapsed/60:.1f} minutes\n")
    
    print(f"📊 REGION-BASED METRICS (Primary):")
    print(f"  Mean Precision: {df['region_precision'].mean():.4f}")
    print(f"  Mean Recall:    {df['region_recall'].mean():.4f}")
    print(f"  Mean F1:        {df['region_f1'].mean():.4f}")
    print(f"  Mean IoU:       {df['region_iou'].mean():.4f}")
    
    print(f"\n📈 Region Detection Stats:")
    print(f"  Avg GT regions:      {df['gt_regions'].mean():.1f}")
    print(f"  Avg Pred regions:    {df['pred_regions'].mean():.1f}")
    print(f"  Avg Matched (GT):    {df['matched_gt_regions'].mean():.1f}")
    print(f"  Avg Matched (Pred):  {df['matched_pred_regions'].mean():.1f}")
    
    print(f"\n📏 Coverage:")
    print(f"  GT Coverage:   {df['gt_coverage'].mean():.2%} (% of GT matched)")
    print(f"  Pred Coverage: {df['pred_coverage'].mean():.2%} (% of Pred valid)")
    
    print(f"\n{'='*60}")
    print(f"✅ Saved: {csv_path.name}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
