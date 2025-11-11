#!/usr/bin/env python3
"""
Weed Detection Evaluation - Mac Version
Based on working Kaggle code, optimized for Mac M1/M2/M3
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
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

# Mac settings - FORCE CPU (MPS has issues)
device = torch.device("cpu")
torch.set_num_threads(4)  # Optimize for M1/M2/M3

# Memory settings
MAX_IMAGE_SIZE = 1500  # Resize to prevent memory issues
SAVE_INTERVAL = 50     # Save progress every 50 images

RESULTS_DIR.mkdir(exist_ok=True)

print("="*60)
print("WEED DETECTION EVALUATION - MAC OPTIMIZED")
print("="*60)
print(f"Device: {device}")
print(f"CPU Threads: {torch.get_num_threads()}")
print(f"Max Image Size: {MAX_IMAGE_SIZE}px")
print(f"Save Interval: {SAVE_INTERVAL} images")

# ==================== FUNCTIONS (SAME AS KAGGLE) ====================

def load_paddy_model(model_path, num_classes=2):
    """Load model"""
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
    
    with torch.no_grad():
        dummy = torch.randn(1, 3, 800, 800).to(device)
        _ = model(dummy)
        del dummy
    
    print("✅ Model loaded\n")
    return model

def resize_if_needed(image, max_size=MAX_IMAGE_SIZE):
    """Resize large images to save memory"""
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

def extract_shape_from_bbox(image_rgb, bbox, expansion=5):
    """Extract paddy shape from bounding box"""
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = bbox.astype(int)
    
    x1 = max(0, x1 - expansion)
    y1 = max(0, y1 - expansion)
    x2 = min(w, x2 + expansion)
    y2 = min(h, y2 + expansion)
    
    roi = image_rgb[y1:y2, x1:x2]
    if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
        return np.zeros((h, w), dtype=np.uint8)
    
    r, g, b = roi[:,:,0], roi[:,:,1], roi[:,:,2]
    green_mask = ((g > r + 10) & (g > b + 10) & (g > 40)).astype(np.uint8) * 255
    
    kernel = np.ones((3, 3), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = green_mask
    
    return full_mask

def weed_segmentation(image_rgb, paddy_mask):
    """Segment weeds from image"""
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
    
    kernel = np.ones((3, 3), np.uint8)
    weed_mask = cv2.morphologyEx(weed_mask, cv2.MORPH_OPEN, kernel)
    weed_mask = cv2.morphologyEx(weed_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    
    return weed_mask

def extract_ground_truth_weed(annotated_rgb):
    """Extract ground truth weed mask"""
    r, g, b = annotated_rgb[:,:,0], annotated_rgb[:,:,1], annotated_rgb[:,:,2]
    
    green_mask = ((g > r + 20) & (g > b + 20) & (g > 80)).astype(np.uint8) * 255
    max_ch = np.maximum(np.maximum(r, g), b)
    colored = ((max_ch > 80) & ((r + g + b) > 150)).astype(np.uint8) * 255
    
    weed_mask = colored.copy()
    weed_mask[green_mask > 0] = 0
    
    kernel = np.ones((5, 5), np.uint8)
    weed_mask = cv2.morphologyEx(weed_mask, cv2.MORPH_CLOSE, kernel)
    
    return weed_mask

@torch.no_grad()
def generate_weed_prediction(image_rgb, model):
    """Generate weed prediction"""
    h, w = image_rgb.shape[:2]
    img_tensor = transforms.ToTensor()(image_rgb).unsqueeze(0).to(device)
    outputs = model(img_tensor)[0]
    
    boxes = outputs['boxes'].cpu().numpy()
    scores = outputs['scores'].cpu().numpy()
    labels = outputs['labels'].cpu().numpy()
    
    del img_tensor, outputs
    
    keep = scores >= 0.5
    pred_boxes = boxes[keep]
    pred_labels = labels[keep]
    
    paddy_mask = np.zeros((h, w), dtype=np.uint8)
    paddy_boxes = pred_boxes[pred_labels == 1]
    
    for box in paddy_boxes:
        shape_mask = extract_shape_from_bbox(image_rgb, box)
        paddy_mask = cv2.bitwise_or(paddy_mask, shape_mask)
    
    weed_mask = weed_segmentation(image_rgb, paddy_mask)
    
    return weed_mask

def calculate_metrics(gt_mask, pred_mask):
    """Calculate evaluation metrics"""
    gt_bin = (gt_mask > 0).astype(int).flatten()
    pred_bin = (pred_mask > 0).astype(int).flatten()
    
    acc = accuracy_score(gt_bin, pred_bin)
    prec = precision_score(gt_bin, pred_bin, zero_division=0)
    rec = recall_score(gt_bin, pred_bin, zero_division=0)
    f1 = f1_score(gt_bin, pred_bin, zero_division=0)
    
    inter = np.logical_and(gt_bin, pred_bin).sum()
    union = np.logical_or(gt_bin, pred_bin).sum()
    iou = inter / union if union > 0 else 0
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'iou': iou,
        'gt_pixels': int(gt_bin.sum()),
        'pred_pixels': int(pred_bin.sum())
    }

# ==================== MAIN PROCESSING ====================

def main():
    start_time = time.time()
    
    # Get files
    print("\nScanning folders...")
    original_files = [f for f in os.listdir(ORIGINAL_FOLDER) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    annotated_files = [f for f in os.listdir(ANNOTATED_FOLDER) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Original images: {len(original_files)}")
    print(f"Annotated images: {len(annotated_files)}")
    
    # Match files (case-insensitive)
    annotated_map = {os.path.splitext(af)[0].lower(): af for af in annotated_files}
    matching_pairs = [(of, annotated_map[os.path.splitext(of)[0].lower()]) 
                      for of in original_files 
                      if os.path.splitext(of)[0].lower() in annotated_map]
    
    print(f"Matched pairs: {len(matching_pairs)}\n")
    
    if len(matching_pairs) == 0:
        raise ValueError("No matching image pairs found!")
    
    # Load model
    model = load_paddy_model(MODEL_PATH)
    
    # Process images
    print(f"Processing {len(matching_pairs)} images...")
    print("(Estimated: 10-20 minutes on M1 Mac)\n")
    
    metrics_list = []
    agg_gt = []
    agg_pred = []
    error_count = 0
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f'weed_results_{timestamp}.csv'
    
    for idx, (orig_fname, annot_fname) in enumerate(tqdm(matching_pairs, desc="Progress"), 1):
        try:
            # Load images
            orig_path = ORIGINAL_FOLDER / orig_fname
            annot_path = ANNOTATED_FOLDER / annot_fname
            
            orig_bgr = cv2.imread(str(orig_path))
            annot_bgr = cv2.imread(str(annot_path))
            
            if orig_bgr is None or annot_bgr is None:
                error_count += 1
                continue
            
            # Resize for memory efficiency
            orig_bgr = resize_if_needed(orig_bgr)
            annot_bgr = resize_if_needed(annot_bgr)
            
            # Convert to RGB
            orig = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
            annot = cv2.cvtColor(annot_bgr, cv2.COLOR_BGR2RGB)
            
            # Process
            gt_weed = extract_ground_truth_weed(annot)
            pred_weed = generate_weed_prediction(orig, model)
            
            # Calculate metrics
            m = calculate_metrics(gt_weed, pred_weed)
            m['filename'] = orig_fname
            metrics_list.append(m)
            
            # Aggregate for overall metrics
            agg_gt.append((gt_weed > 0).astype(int).flatten())
            agg_pred.append((pred_weed > 0).astype(int).flatten())
            
            # Save progress incrementally
            if idx % SAVE_INTERVAL == 0:
                df_temp = pd.DataFrame(metrics_list)
                df_temp.to_csv(csv_path, index=False)
                gc.collect()
            
        except Exception as e:
            error_count += 1
            continue
    
    elapsed = time.time() - start_time
    
    # Final save
    df = pd.DataFrame(metrics_list)
    df.to_csv(csv_path, index=False)
    
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Processed: {len(metrics_list)} images")
    if error_count > 0:
        print(f"⚠️  Errors: {error_count} images")
    print(f"⏱️  Time: {elapsed/60:.1f} minutes ({elapsed/len(metrics_list):.2f} sec/image)")
    print(f"💾 Results saved: {csv_path}")
    
    # ==================== OVERALL METRICS ====================
    
    print(f"\n{'='*60}")
    print("OVERALL RESULTS")
    print(f"{'='*60}")
    
    gt_all = np.concatenate(agg_gt)
    pred_all = np.concatenate(agg_pred)
    
    acc = accuracy_score(gt_all, pred_all)
    prec = precision_score(gt_all, pred_all, zero_division=0)
    rec = recall_score(gt_all, pred_all, zero_division=0)
    f1 = f1_score(gt_all, pred_all, zero_division=0)
    inter = np.logical_and(gt_all, pred_all).sum()
    union = np.logical_or(gt_all, pred_all).sum()
    iou = inter / union if union > 0 else 0
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"IoU:       {iou:.4f}")
    
    # ==================== STATISTICS ====================
    
    print(f"\n{'='*60}")
    print("STATISTICS")
    print(f"{'='*60}")
    
    df_sorted = df.sort_values('iou', ascending=False)
    
    print(f"Mean IoU:       {df['iou'].mean():.4f} ± {df['iou'].std():.4f}")
    print(f"Median IoU:     {df['iou'].median():.4f}")
    print(f"Mean F1-Score:  {df['f1_score'].mean():.4f} ± {df['f1_score'].std():.4f}")
    print(f"Mean Precision: {df['precision'].mean():.4f} ± {df['precision'].std():.4f}")
    print(f"Mean Recall:    {df['recall'].mean():.4f} ± {df['recall'].std():.4f}")
    
    print(f"\nTop 5 images by IoU:")
    for i, row in df_sorted.head(5).iterrows():
        print(f"  {row['filename']:30s}  IoU: {row['iou']:.4f}  F1: {row['f1_score']:.4f}")
    
    # ==================== CONFUSION MATRIX ====================
    
    cm = confusion_matrix(gt_all, pred_all)
    
    print(f"\n{'='*60}")
    print("CONFUSION MATRIX")
    print(f"{'='*60}")
    print(f"                 Predicted")
    print(f"               No Weed      Weed")
    print(f"Actual No Weed {cm[0,0]:>10,}  {cm[0,1]:>10,}")
    print(f"Actual Weed    {cm[1,0]:>10,}  {cm[1,1]:>10,}")
    
    print(f"\n{'='*60}")
    print("✅ EVALUATION COMPLETE!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
