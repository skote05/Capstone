#!/usr/bin/env python3
"""
Weed Detection with Bounding Box Metrics - Mac Version
Perfect for sparse/dotted annotations
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
from tqdm import tqdm
import time
from pathlib import Path
import gc

# ==================== CONFIGURATION ====================

BASE_DIR = Path(__file__).parent
ORIGINAL_FOLDER = BASE_DIR / 'data' / 'original_images'
ANNOTATED_FOLDER = BASE_DIR / 'data' / 'annotated_images'
MODEL_PATH = BASE_DIR / 'data' / 'model' / 'paddy_detection_model_combined.pth'
RESULTS_DIR = BASE_DIR / 'results'

device = torch.device("cpu")
torch.set_num_threads(4)

RESIZE_TO = (800, 800)
DETECTION_SCALES = [1.0, 1.25, 1.5, 2.0]
CONFIDENCE_THRESHOLD = 0.25
NMS_THRESHOLD = 0.3
SAVE_INTERVAL = 50

# Bounding box settings
IOU_THRESHOLD = 0.5  # 30% overlap = match
MIN_WEED_AREA = 50   # Minimum pixels for a weed region

RESULTS_DIR.mkdir(exist_ok=True)

print("="*60)
print("WEED DETECTION - BOUNDING BOX METRICS")
print("="*60)
print(f"Image resize: {RESIZE_TO}")
print(f"IoU threshold: {IOU_THRESHOLD}")
print(f"Min weed area: {MIN_WEED_AREA} pixels")
print("="*60)

# ==================== FUNCTIONS ====================

def load_paddy_model(model_path, num_classes=2):
    print("\nLoading model...")
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    print("✅ Model loaded\n")
    return model

def resize_to_training_size(image):
    return cv2.resize(image, (800, 800), interpolation=cv2.INTER_LINEAR)

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

def box_iou(box1, box2):
    """Calculate IoU between two bounding boxes [x, y, w, h]"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate intersection
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1+w1, x2+w2)
    yi2 = min(y1+h1, y2+h2)
    
    inter_area = max(0, xi2-xi1) * max(0, yi2-yi1)
    
    # Calculate union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def calculate_bbox_metrics(gt_mask, pred_mask):
    """
    Calculate metrics based on bounding box overlap
    Perfect for sparse annotations
    """
    # Find contours (weed regions)
    gt_contours, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pred_contours, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get bounding boxes
    gt_boxes = [cv2.boundingRect(c) for c in gt_contours if cv2.contourArea(c) > MIN_WEED_AREA]
    pred_boxes = [cv2.boundingRect(c) for c in pred_contours if cv2.contourArea(c) > MIN_WEED_AREA]
    
    # Match GT boxes to predicted boxes
    matched_gt = set()
    matched_pred = set()
    
    for i, gt_box in enumerate(gt_boxes):
        for j, pred_box in enumerate(pred_boxes):
            if box_iou(gt_box, pred_box) > IOU_THRESHOLD:
                matched_gt.add(i)
                matched_pred.add(j)
    
    # Calculate metrics
    num_gt = len(gt_boxes)
    num_pred = len(pred_boxes)
    true_positives = len(matched_gt)
    false_positives = num_pred - len(matched_pred)
    false_negatives = num_gt - len(matched_gt)
    
    precision = true_positives / max(num_pred, 1)
    recall = true_positives / max(num_gt, 1)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    return {
        'bbox_precision': precision,
        'bbox_recall': recall,
        'bbox_f1': f1,
        'gt_boxes': num_gt,
        'pred_boxes': num_pred,
        'matched_boxes': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }

def calculate_combined_metrics(gt_mask, pred_mask):
    """Calculate both pixel-level and bbox metrics"""
    
    # Pixel-level metrics (for reference)
    gt_bin = (gt_mask > 0).astype(int).flatten()
    pred_bin = (pred_mask > 0).astype(int).flatten()
    inter = np.logical_and(gt_bin, pred_bin).sum()
    union = np.logical_or(gt_bin, pred_bin).sum()
    
    pixel_metrics = {
        'pixel_accuracy': accuracy_score(gt_bin, pred_bin),
        'pixel_precision': precision_score(gt_bin, pred_bin, zero_division=0),
        'pixel_recall': recall_score(gt_bin, pred_bin, zero_division=0),
        'pixel_f1': f1_score(gt_bin, pred_bin, zero_division=0),
        'pixel_iou': inter / union if union > 0 else 0,
        'gt_pixels': int(gt_bin.sum()),
        'pred_pixels': int(pred_bin.sum())
    }
    
    # Bounding box metrics (main metric)
    bbox_metrics = calculate_bbox_metrics(gt_mask, pred_mask)
    
    return {**pixel_metrics, **bbox_metrics}

# ==================== MAIN ====================

def main():
    start_time = time.time()
    
    # Get files
    original_files = [f for f in os.listdir(ORIGINAL_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    annotated_files = [f for f in os.listdir(ANNOTATED_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    annotated_map = {os.path.splitext(af)[0].lower(): af for af in annotated_files}
    matching_pairs = [(of, annotated_map[os.path.splitext(of)[0].lower()]) 
                      for of in original_files if os.path.splitext(of)[0].lower() in annotated_map]
    
    print(f"\nMatched pairs: {len(matching_pairs)}\n")
    
    # Load model
    model = load_paddy_model(MODEL_PATH)
    
    print(f"Processing with bounding box metrics...\n")
    
    metrics_list = []
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f'weed_results_bbox_{timestamp}.csv'
    
    for idx, (orig_fname, annot_fname) in enumerate(tqdm(matching_pairs, desc="Processing"), 1):
        try:
            orig_bgr = cv2.imread(str(ORIGINAL_FOLDER / orig_fname))
            annot_bgr = cv2.imread(str(ANNOTATED_FOLDER / annot_fname))
            
            if orig_bgr is None or annot_bgr is None:
                continue
            
            # Resize to 800x800
            orig_resized = resize_to_training_size(orig_bgr)
            annot_resized = resize_to_training_size(annot_bgr)
            
            orig = cv2.cvtColor(orig_resized, cv2.COLOR_BGR2RGB)
            annot = cv2.cvtColor(annot_resized, cv2.COLOR_BGR2RGB)
            
            # Process
            gt_weed = extract_ground_truth_weed(annot)
            pred_weed = generate_weed_prediction_multiscale(orig, model)
            
            # Calculate metrics
            m = calculate_combined_metrics(gt_weed, pred_weed)
            m['filename'] = orig_fname
            metrics_list.append(m)
            
            # Save progress
            if idx % SAVE_INTERVAL == 0:
                pd.DataFrame(metrics_list).to_csv(csv_path, index=False)
                gc.collect()
            
        except Exception as e:
            continue
    
    # Final save
    df = pd.DataFrame(metrics_list)
    df.to_csv(csv_path, index=False)
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("RESULTS - BOUNDING BOX METRICS")
    print(f"{'='*60}")
    print(f"Processed: {len(metrics_list)}/{len(matching_pairs)} images")
    print(f"Time: {elapsed/60:.1f} minutes\n")
    
    # Bounding Box Metrics (PRIMARY)
    print(f"🎯 Bounding Box Metrics (Primary):")
    print(f"  Precision: {df['bbox_precision'].mean():.4f}")
    print(f"  Recall:    {df['bbox_recall'].mean():.4f}")
    print(f"  F1-Score:  {df['bbox_f1'].mean():.4f}")
    print(f"  Avg GT boxes:    {df['gt_boxes'].mean():.1f}")
    print(f"  Avg Pred boxes:  {df['pred_boxes'].mean():.1f}")
    print(f"  Avg Matched:     {df['matched_boxes'].mean():.1f}")
    
    # Pixel Metrics (REFERENCE)
    print(f"\n📊 Pixel Metrics (Reference):")
    print(f"  Precision: {df['pixel_precision'].mean():.4f}")
    print(f"  Recall:    {df['pixel_recall'].mean():.4f}")
    print(f"  F1-Score:  {df['pixel_f1'].mean():.4f}")
    print(f"  IoU:       {df['pixel_iou'].mean():.4f}")
    
    # Summary statistics
    print(f"\n📈 Detection Summary:")
    total_gt = df['gt_boxes'].sum()
    total_matched = df['matched_boxes'].sum()
    total_fp = df['false_positives'].sum()
    total_fn = df['false_negatives'].sum()
    
    print(f"  Total GT weeds:        {int(total_gt)}")
    print(f"  Successfully detected: {int(total_matched)} ({total_matched/total_gt*100:.1f}%)")
    print(f"  False positives:       {int(total_fp)}")
    print(f"  Missed (FN):           {int(total_fn)}")
    
    print(f"\n{'='*60}")
    print(f"Results saved: {csv_path.name}")
    print(f"{'='*60}")
    
    # Save summary
    summary = {
        'bbox_precision': df['bbox_precision'].mean(),
        'bbox_recall': df['bbox_recall'].mean(),
        'bbox_f1': df['bbox_f1'].mean(),
        'pixel_precision': df['pixel_precision'].mean(),
        'pixel_recall': df['pixel_recall'].mean(),
        'pixel_f1': df['pixel_f1'].mean(),
        'total_images': len(metrics_list),
        'total_gt_weeds': int(total_gt),
        'total_detected': int(total_matched),
        'detection_rate': f"{total_matched/total_gt*100:.1f}%"
    }
    
    summary_df = pd.DataFrame([summary])
    summary_path = RESULTS_DIR / f'summary_bbox_{timestamp}.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved: {summary_path.name}")

if __name__ == "__main__":
    main()
