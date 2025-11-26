#!/usr/bin/env python3
"""
Weed Detection - Mac Version (Updated)
Uses COCO annotations with improved detection from visual analysis
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
import os
import pandas as pd
from tqdm import tqdm
import time
from pathlib import Path
import gc
import json

# ==================== CONFIGURATION ====================

BASE_DIR = Path(__file__).parent
TEST_IMAGES_FOLDER = BASE_DIR / 'data' / 'test'
COCO_ANNOTATION_FILE = BASE_DIR / 'data' / 'test' / '_annotations.coco.json'
MODEL_PATH = BASE_DIR / 'data' / 'model' / 'paddy_model_with_test_data.pth'
RESULTS_DIR = BASE_DIR / 'results'

device = torch.device("cpu")
torch.set_num_threads(4)

# ==================== TUNABLE THRESHOLDS ====================
# Updated based on visual analysis success
RESIZE_TO = (800, 800)
DETECTION_SCALES = [1.0, 1.25, 1.5, 2.0]  # Reduced from [1.0, 1.25, 1.5, 2.0] for speed
CONFIDENCE_THRESHOLD = 0.7    # Increased from 0.5 based on visual results
NMS_THRESHOLD = 0.2

# Weed segmentation thresholds (from visual code)
EXG_THRESHOLD = 20
GREEN_RATIO_THRESHOLD = 1.1
PADDY_EXPANSION = 15           # Expansion around paddy bbox

SAVE_INTERVAL = 50

RESULTS_DIR.mkdir(exist_ok=True)

print("="*60)
print("WEED DETECTION - UPDATED MAC VERSION")
print("="*60)
print(f"🔧 Settings (from visual analysis):")
print(f"  Confidence: {CONFIDENCE_THRESHOLD}")
print(f"  Detection scales: {DETECTION_SCALES}")
print(f"  Image size: {RESIZE_TO}")
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
    """Resize with padding to maintain aspect ratio (from visual code)"""
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


def resize_boxes_to_canvas(boxes, scale, offset):
    """Scale COCO boxes to match resized canvas"""
    x_offset, y_offset = offset
    scaled_boxes = []
    
    for bbox in boxes:
        x, y, w, h = bbox
        # Scale
        x_new = x * scale
        y_new = y * scale
        w_new = w * scale
        h_new = h * scale
        # Add offset
        x_new += x_offset
        y_new += y_offset
        scaled_boxes.append([x_new, y_new, w_new, h_new])
    
    return scaled_boxes


def extract_shape_from_bbox(image_rgb, bbox, expansion=PADDY_EXPANSION):
    """Extract paddy shape from bbox (from visual code)"""
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
    """Weed segmentation (from visual code)"""
    r = image_rgb[:,:,0].astype(np.float32)
    g = image_rgb[:,:,1].astype(np.float32)
    b = image_rgb[:,:,2].astype(np.float32)
    
    # Vegetation detection
    exg = 2.0 * g - r - b
    exg_mask = (exg > EXG_THRESHOLD).astype(np.uint8)
    green_ratio = g / (r + b + 1)
    ratio_mask = (green_ratio > GREEN_RATIO_THRESHOLD).astype(np.uint8)
    green_dom = ((g > r + 5) & (g > b + 5) & (g > 30)).astype(np.uint8)
    
    veg_mask = ((exg_mask + ratio_mask + green_dom) >= 2).astype(np.uint8) * 255
    
    # Remove paddy
    weed_mask = veg_mask.copy()
    weed_mask[paddy_mask > 0] = 0
    
    # Clean up
    weed_mask = cv2.morphologyEx(weed_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    weed_mask = cv2.morphologyEx(weed_mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    
    return weed_mask


@torch.no_grad()
def generate_weed_prediction_multiscale(image_rgb, model):
    """Multi-scale detection (from visual code)"""
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
    
    # Combine and NMS
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
    
    # Extract paddy regions
    paddy_mask = np.zeros((h, w), dtype=np.uint8)
    if len(final_boxes) > 0:
        for box in final_boxes[final_labels == 1]:
            paddy_mask = cv2.bitwise_or(paddy_mask, extract_shape_from_bbox(image_rgb, box))
    
    return weed_segmentation(image_rgb, paddy_mask)


def calculate_region_based_metrics(pred_weed_mask, gt_boxes, image_shape):
    """
    Calculate metrics: predicted weed regions vs GT weed boxes
    Uses flood fill logic from visual code
    """
    # Find predicted weed regions
    pred_num_labels, pred_labels, pred_stats, _ = cv2.connectedComponentsWithStats(
        pred_weed_mask.astype(np.uint8), connectivity=8
    )
    pred_num_regions = pred_num_labels - 1
    
    # Handle empty GT
    if len(gt_boxes) == 0:
        return {
            'region_precision': 0.0,
            'region_recall': 0.0,
            'region_f1': 0.0,
            'region_iou': 0.0,
            'gt_regions': 0,
            'pred_regions': pred_num_regions,
            'matched_gt_regions': 0,
            'matched_pred_regions': 0,
            'gt_pixels': 0,
            'pred_pixels': int((pred_weed_mask > 0).sum()),
            'matched_gt_pixels': 0,
            'matched_pred_pixels': 0,
            'gt_coverage': 0.0,
            'pred_coverage': 0.0
        }
    
    # Track matches
    matched_gt_boxes = set()
    matched_pred_regions = set()
    ious = []
    
    # Check each GT box against each predicted region
    for gt_idx, bbox in enumerate(gt_boxes):
        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        if w <= 0 or h <= 0:
            continue
        
        # Create GT box mask
        gt_box_mask = np.zeros(image_shape[:2], dtype=np.uint8)
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(image_shape[1], x + w), min(image_shape[0], y + h)
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        gt_box_mask[y1:y2, x1:x2] = 255
        
        # Check overlap with each predicted region
        for pred_label in range(1, pred_num_labels):
            pred_region_mask = (pred_labels == pred_label).astype(np.uint8) * 255
            
            # Calculate overlap
            intersection = np.logical_and(pred_region_mask > 0, gt_box_mask > 0).sum()
            
            if intersection > 0:  # Any overlap counts as match
                union = np.logical_or(pred_region_mask > 0, gt_box_mask > 0).sum()
                iou = intersection / union if union > 0 else 0.0
                ious.append(iou)
                
                matched_gt_boxes.add(gt_idx)
                matched_pred_regions.add(pred_label)
    
    # Calculate matched pixel counts
    matched_pred_pixels = 0
    for pred_label in matched_pred_regions:
        matched_pred_pixels += (pred_labels == pred_label).sum()
    
    # GT pixels (total area of all GT boxes)
    gt_pixels = 0
    matched_gt_pixels = 0
    for gt_idx, bbox in enumerate(gt_boxes):
        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        if w > 0 and h > 0:
            box_pixels = w * h
            gt_pixels += box_pixels
            if gt_idx in matched_gt_boxes:
                matched_gt_pixels += box_pixels
    
    pred_pixels = int((pred_weed_mask > 0).sum())
    
    # Calculate metrics
    num_matched_gt = len(matched_gt_boxes)
    num_matched_pred = len(matched_pred_regions)
    
    region_precision = num_matched_pred / max(pred_num_regions, 1)
    region_recall = num_matched_gt / max(len(gt_boxes), 1)
    region_f1 = 2 * (region_precision * region_recall) / (region_precision + region_recall + 1e-10)
    region_iou = np.mean(ious) if ious else 0.0
    
    gt_coverage = matched_gt_pixels / max(gt_pixels, 1)
    pred_coverage = matched_pred_pixels / max(pred_pixels, 1)
    
    return {
        'region_precision': region_precision,
        'region_recall': region_recall,
        'region_f1': region_f1,
        'region_iou': region_iou,
        'gt_regions': len(gt_boxes),
        'pred_regions': pred_num_regions,
        'matched_gt_regions': num_matched_gt,
        'matched_pred_regions': num_matched_pred,
        'gt_pixels': gt_pixels,
        'pred_pixels': pred_pixels,
        'matched_gt_pixels': int(matched_gt_pixels),
        'matched_pred_pixels': int(matched_pred_pixels),
        'gt_coverage': gt_coverage,
        'pred_coverage': pred_coverage
    }


# ==================== MAIN ====================

def main():
    start_time = time.time()
    
    # Load COCO annotations
    print(f"\nLoading COCO annotations...")
    with open(COCO_ANNOTATION_FILE, 'r') as f:
        coco_data = json.load(f)
    
    # Get categories
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    weed_cat_ids = set(categories.keys())
    print(f"Categories: {categories}")
    
    # Build lookups
    image_info_dict = {img['id']: img for img in coco_data['images']}
    image_annotations = {img_id: [] for img_id in image_info_dict.keys()}
    
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        cat_id = ann['category_id']
        if img_id in image_annotations and cat_id in weed_cat_ids:
            image_annotations[img_id].append(ann['bbox'])
    
    print(f"Total images: {len(image_info_dict)}")
    print(f"Total weed boxes: {sum(len(anns) for anns in image_annotations.values())}")
    
    # Load model
    model = load_paddy_model(MODEL_PATH)
    
    print(f"\nProcessing with updated settings...\n")
    
    metrics_list = []
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f'weed_results_{timestamp}.csv'
    
    processed = 0
    for img_id, img_info in tqdm(image_info_dict.items(), desc="Progress"):
        try:
            img_path = TEST_IMAGES_FOLDER / img_info['file_name']
            
            if not img_path.exists():
                continue
            
            # Load image
            image_bgr = cv2.imread(str(img_path))
            if image_bgr is None:
                continue
            
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            gt_boxes_coco = image_annotations[img_id]
            
            # Resize with padding (like visual code)
            resized_rgb, scale, offset = resize_to_training_size(image_rgb, RESIZE_TO)
            
            # Scale GT boxes to resized canvas
            scaled_boxes = resize_boxes_to_canvas(gt_boxes_coco, scale, offset)
            
            # Detect paddy and segment weeds
            pred_weed_mask = generate_weed_prediction_multiscale(resized_rgb, model)
            
            # Calculate metrics
            metrics = calculate_region_based_metrics(
                pred_weed_mask, scaled_boxes, resized_rgb.shape
            )
            
            metrics['filename'] = img_info['file_name']
            metrics_list.append(metrics)
            
            processed += 1
            
            if processed % SAVE_INTERVAL == 0:
                pd.DataFrame(metrics_list).to_csv(csv_path, index=False)
                gc.collect()
            
        except Exception as e:
            print(f"\nError on {img_info.get('file_name', 'unknown')}: {e}")
            continue
    
    # Final save
    df = pd.DataFrame(metrics_list)
    df.to_csv(csv_path, index=False)
    
    elapsed = time.time() - start_time
    
    # Results
    print(f"\n{'='*60}")
    print("RESULTS - UPDATED MAC VERSION")
    print(f"{'='*60}")
    print(f"Processed: {len(metrics_list)} images")
    print(f"Time: {elapsed/60:.1f} minutes\n")
    
    print(f"📊 REGION-BASED METRICS:")
    print(f"  Mean Precision: {df['region_precision'].mean():.4f}")
    print(f"  Mean Recall:    {df['region_recall'].mean():.4f}")
    print(f"  Mean F1:        {df['region_f1'].mean():.4f}")
    print(f"  Mean IoU:       {df['region_iou'].mean():.4f}")
    
    print(f"\n📈 Detection Stats:")
    print(f"  Total GT boxes:        {df['gt_regions'].sum()}")
    print(f"  Total Pred regions:    {df['pred_regions'].sum()}")
    print(f"  Total Matched GT:      {df['matched_gt_regions'].sum()}")
    print(f"  Total Matched Pred:    {df['matched_pred_regions'].sum()}")
    
    print(f"\n📏 Coverage:")
    print(f"  GT Coverage:   {df['gt_coverage'].mean():.2%}")
    print(f"  Pred Coverage: {df['pred_coverage'].mean():.2%}")
    
    print(f"\n✅ Saved: {csv_path.name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
