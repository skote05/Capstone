#!/usr/bin/env python3
"""
Weed Detection Test - MATCHED TO TRAINING
Images resized to 800x800 to match training configuration
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
ORIGINAL_FOLDER = BASE_DIR / 'data' / 'original_images'
ANNOTATED_FOLDER = BASE_DIR / 'data' / 'annotated_images'
MODEL_PATH = BASE_DIR / 'data' / 'model' / 'paddy_detection_model_combined.pth'
RESULTS_DIR = BASE_DIR / 'results'

device = torch.device("cpu")
torch.set_num_threads(4)

# CRITICAL: Match training configuration
RESIZE_TO = (800, 800)  # SAME AS TRAINING!

# Multi-scale settings - scales are now applied to 800x800 base
DETECTION_SCALES = [1.0, 1.25, 1.5, 2.0]  # More conservative
CONFIDENCE_THRESHOLD = 0.25
NMS_THRESHOLD = 0.3

SAVE_INTERVAL = 50

RESULTS_DIR.mkdir(exist_ok=True)

print("="*60)
print("WEED DETECTION - MATCHED TO TRAINING CONFIG")
print("="*60)
print(f"Base image size: {RESIZE_TO} (SAME AS TRAINING)")
print(f"Scales: {DETECTION_SCALES}")
print(f"Confidence: {CONFIDENCE_THRESHOLD}")
print("="*60)

# ==================== FUNCTIONS ====================

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
    print("✅ Model loaded")
    
    # Check if model was trained on 800x800
    if 'image_size' in checkpoint:
        training_size = checkpoint['image_size']
        print(f"   Training image size: {training_size}")
        if training_size != RESIZE_TO:
            print(f"   ⚠️  WARNING: Test size {RESIZE_TO} doesn't match training {training_size}")
    
    return model

def resize_to_training_size(image, target_size=RESIZE_TO):
    """
    Resize image to EXACT training size (800x800)
    This maintains aspect ratio by padding or cropping
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # Calculate scaling to fit within target while maintaining aspect ratio
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create canvas of exact target size
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # Center the resized image on canvas
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    # Return canvas and the offsets for box adjustment
    return canvas, scale, (x_offset, y_offset)

def extract_shape_from_bbox(image_rgb, bbox, expansion=15):
    """Extract paddy shape from bbox"""
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
    green_mask = ((g > r + 5) & (g > b + 5) & (g > 25)).astype(np.uint8) * 255
    
    kernel = np.ones((5, 5), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_DILATE, kernel)
    
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = green_mask
    
    return full_mask

def weed_segmentation(image_rgb, paddy_mask):
    """Segment weeds"""
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
    """Extract ground truth"""
    r, g, b = annotated_rgb[:,:,0], annotated_rgb[:,:,1], annotated_rgb[:,:,2]
    green_mask = ((g > r + 20) & (g > b + 20) & (g > 80)).astype(np.uint8) * 255
    max_ch = np.maximum(np.maximum(r, g), b)
    colored = ((max_ch > 80) & ((r + g + b) > 150)).astype(np.uint8) * 255
    weed_mask = colored.copy()
    weed_mask[green_mask > 0] = 0
    return cv2.morphologyEx(weed_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

@torch.no_grad()
def generate_weed_prediction_multiscale(image_rgb, model):
    """
    Multi-scale detection on 800x800 base image
    Image is ALREADY at training size
    """
    h, w = image_rgb.shape[:2]
    assert h == RESIZE_TO[0] and w == RESIZE_TO[1], f"Image must be {RESIZE_TO}, got {(h, w)}"
    
    all_boxes = []
    all_scores = []
    all_labels = []
    
    for scale in DETECTION_SCALES:
        if scale != 1.0:
            new_h = int(h * scale)
            new_w = int(w * scale)
            
            # Skip if too large (conservative limit for 800x800 base)
            if new_h > 1600 or new_w > 1600:
                continue
            
            scaled_img = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            scaled_img = image_rgb
        
        # Run detection
        img_tensor = transforms.ToTensor()(scaled_img).unsqueeze(0).to(device)
        outputs = model(img_tensor)[0]
        
        boxes = outputs['boxes'].cpu().numpy()
        scores = outputs['scores'].cpu().numpy()
        labels = outputs['labels'].cpu().numpy()
        
        del img_tensor, outputs
        
        # Scale boxes back to 800x800
        if scale != 1.0:
            boxes = boxes / scale
        
        keep = scores >= CONFIDENCE_THRESHOLD
        all_boxes.append(boxes[keep])
        all_scores.append(scores[keep])
        all_labels.append(labels[keep])
    
    # Combine and apply NMS
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
        final_boxes = np.array([])
        final_labels = np.array([])
    
    # Build paddy mask
    paddy_mask = np.zeros((h, w), dtype=np.uint8)
    if len(final_boxes) > 0:
        paddy_boxes = final_boxes[final_labels == 1]
        for box in paddy_boxes:
            shape_mask = extract_shape_from_bbox(image_rgb, box)
            paddy_mask = cv2.bitwise_or(paddy_mask, shape_mask)
    
    weed_mask = weed_segmentation(image_rgb, paddy_mask)
    return weed_mask

def calculate_metrics(gt_mask, pred_mask):
    """Calculate metrics"""
    gt_bin = (gt_mask > 0).astype(int).flatten()
    pred_bin = (pred_mask > 0).astype(int).flatten()
    
    inter = np.logical_and(gt_bin, pred_bin).sum()
    union = np.logical_or(gt_bin, pred_bin).sum()
    
    return {
        'accuracy': accuracy_score(gt_bin, pred_bin),
        'precision': precision_score(gt_bin, pred_bin, zero_division=0),
        'recall': recall_score(gt_bin, pred_bin, zero_division=0),
        'f1_score': f1_score(gt_bin, pred_bin, zero_division=0),
        'iou': inter / union if union > 0 else 0,
        'gt_pixels': int(gt_bin.sum()),
        'pred_pixels': int(pred_bin.sum())
    }

# ==================== MAIN ====================

def main():
    start_time = time.time()
    
    # Get files
    original_files = [f for f in os.listdir(ORIGINAL_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    annotated_files = [f for f in os.listdir(ANNOTATED_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    annotated_map = {os.path.splitext(af)[0].lower(): af for af in annotated_files}
    matching_pairs = [(of, annotated_map[os.path.splitext(of)[0].lower()]) 
                      for of in original_files if os.path.splitext(of)[0].lower() in annotated_map]
    
    print(f"\nFound {len(matching_pairs)} matching pairs")
    
    # Load model
    model = load_paddy_model(MODEL_PATH)
    
    print(f"\nProcessing with training-matched configuration...")
    print(f"All images will be resized to {RESIZE_TO}\n")
    
    metrics_list = []
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f'weed_results_800x800_{timestamp}.csv'
    
    for idx, (orig_fname, annot_fname) in enumerate(tqdm(matching_pairs, desc="Progress"), 1):
        try:
            # Load images
            orig_bgr = cv2.imread(str(ORIGINAL_FOLDER / orig_fname))
            annot_bgr = cv2.imread(str(ANNOTATED_FOLDER / annot_fname))
            
            if orig_bgr is None or annot_bgr is None:
                continue
            
            # CRITICAL: Resize to EXACT training size (800x800)
            orig_resized, _, _ = resize_to_training_size(orig_bgr)
            annot_resized, _, _ = resize_to_training_size(annot_bgr)
            
            # Convert to RGB
            orig = cv2.cvtColor(orig_resized, cv2.COLOR_BGR2RGB)
            annot = cv2.cvtColor(annot_resized, cv2.COLOR_BGR2RGB)
            
            # Process
            gt_weed = extract_ground_truth_weed(annot)
            pred_weed = generate_weed_prediction_multiscale(orig, model)
            
            # Calculate metrics
            m = calculate_metrics(gt_weed, pred_weed)
            m['filename'] = orig_fname
            metrics_list.append(m)
            
            # Save progress
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
    print(f"✅ Processed: {len(metrics_list)} images")
    print(f"⏱️  Time: {elapsed/60:.1f} minutes")
    print(f"💾 Saved: {csv_path.name}")
    print(f"\n📊 Results:")
    print(f"   Mean Precision: {df['precision'].mean():.4f}")
    print(f"   Mean Recall:    {df['recall'].mean():.4f}")
    print(f"   Mean F1:        {df['f1_score'].mean():.4f}")
    print(f"   Mean IoU:       {df['iou'].mean():.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
