#!/usr/bin/env python3
"""
Weed Detection - Mac Memory-Optimized Version
Ground Truth Dilation + Aggressive memory management
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
import sys

# ==================== CONFIGURATION ====================

BASE_DIR = Path(__file__).parent
ORIGINAL_FOLDER = BASE_DIR / 'data' / 'original_images'
ANNOTATED_FOLDER = BASE_DIR / 'data' / 'annotated_images'
MODEL_PATH = BASE_DIR / 'data' / 'model' / 'paddy_detection_model_combined.pth'
RESULTS_DIR = BASE_DIR / 'results'

device = torch.device("cpu")
torch.set_num_threads(2)  # Reduce CPU threads

RESIZE_TO = (800, 800)
DETECTION_SCALES = [1.0, 1.5]  # REDUCED to 2 scales to save memory!
CONFIDENCE_THRESHOLD = 0.25
NMS_THRESHOLD = 0.3
SAVE_INTERVAL = 25  # Save more frequently

# Ground truth dilation
GT_DILATION_KERNEL_SIZE = 15
GT_DILATION_ITERATIONS = 3

RESULTS_DIR.mkdir(exist_ok=True)

print("="*60)
print("WEED DETECTION - MAC MEMORY-OPTIMIZED")
print("="*60)
print(f"Scales: {DETECTION_SCALES} (reduced for memory)")
print(f"GT dilation: {GT_DILATION_KERNEL_SIZE}x{GT_DILATION_KERNEL_SIZE}, {GT_DILATION_ITERATIONS} iterations")
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
    
    # Free memory
    del checkpoint
    gc.collect()
    
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
    
    # Free memory
    del r, g, b, exg, exg_mask, green_ratio, ratio_mask, green_dom, veg_mask
    
    return weed_mask

def extract_ground_truth_weed(annotated_rgb):
    r, g, b = annotated_rgb[:,:,0], annotated_rgb[:,:,1], annotated_rgb[:,:,2]
    
    green_mask = ((g > r + 20) & (g > b + 20) & (g > 80)).astype(np.uint8) * 255
    max_ch = np.maximum(np.maximum(r, g), b)
    colored = ((max_ch > 80) & ((r + g + b) > 150)).astype(np.uint8) * 255
    weed_mask = colored.copy()
    weed_mask[green_mask > 0] = 0
    
    # Morphological dilation
    kernel_large = np.ones((GT_DILATION_KERNEL_SIZE, GT_DILATION_KERNEL_SIZE), np.uint8)
    weed_mask = cv2.morphologyEx(weed_mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
    weed_mask = cv2.morphologyEx(weed_mask, cv2.MORPH_DILATE, kernel_large, iterations=GT_DILATION_ITERATIONS)
    
    kernel_small = np.ones((5, 5), np.uint8)
    weed_mask = cv2.morphologyEx(weed_mask, cv2.MORPH_OPEN, kernel_small)
    weed_mask = cv2.morphologyEx(weed_mask, cv2.MORPH_CLOSE, kernel_small)
    
    # Free memory
    del r, g, b, green_mask, max_ch, colored, kernel_large, kernel_small
    
    return weed_mask

@torch.no_grad()
def generate_weed_prediction_multiscale(image_rgb, model):
    h, w = image_rgb.shape[:2]
    all_boxes, all_scores, all_labels = [], [], []
    
    for scale in DETECTION_SCALES:
        try:
            if scale != 1.0:
                new_h, new_w = int(h * scale), int(w * scale)
                scaled_img = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                scaled_img = image_rgb
            
            img_tensor = transforms.ToTensor()(scaled_img).unsqueeze(0).to(device)
            outputs = model(img_tensor)[0]
            
            boxes = outputs['boxes'].cpu().numpy()
            scores = outputs['scores'].cpu().numpy()
            labels = outputs['labels'].cpu().numpy()
            
            # Immediate cleanup
            del img_tensor, outputs
            if scale != 1.0:
                del scaled_img
            gc.collect()
            
            if scale != 1.0:
                boxes = boxes / scale
            
            keep = scores >= CONFIDENCE_THRESHOLD
            all_boxes.append(boxes[keep])
            all_scores.append(scores[keep])
            all_labels.append(labels[keep])
            
        except Exception as e:
            print(f"\nError at scale {scale}: {e}")
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
        final_boxes, final_labels = np.array([]), np.array([])
    
    paddy_mask = np.zeros((h, w), dtype=np.uint8)
    if len(final_boxes) > 0:
        for box in final_boxes[final_labels == 1]:
            paddy_mask = cv2.bitwise_or(paddy_mask, extract_shape_from_bbox(image_rgb, box))
    
    weed_mask = weed_segmentation(image_rgb, paddy_mask)
    
    del paddy_mask
    gc.collect()
    
    return weed_mask

def calculate_metrics(gt_mask, pred_mask):
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
    
    try:
        original_files = [f for f in os.listdir(ORIGINAL_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        annotated_files = [f for f in os.listdir(ANNOTATED_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        annotated_map = {os.path.splitext(af)[0].lower(): af for af in annotated_files}
        matching_pairs = [(of, annotated_map[os.path.splitext(of)[0].lower()]) 
                          for of in original_files if os.path.splitext(of)[0].lower() in annotated_map]
        
        print(f"\nMatched pairs: {len(matching_pairs)}\n")
        
        model = load_paddy_model(MODEL_PATH)
        
        print(f"Processing (memory-optimized)...\n")
        
        metrics_list = []
        agg_gt, agg_pred = [], []
        error_count = 0
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = RESULTS_DIR / f'weed_results_dilated_{timestamp}.csv'
        
        for idx, (orig_fname, annot_fname) in enumerate(tqdm(matching_pairs, desc="Processing"), 1):
            try:
                orig_bgr = cv2.imread(str(ORIGINAL_FOLDER / orig_fname))
                annot_bgr = cv2.imread(str(ANNOTATED_FOLDER / annot_fname))
                
                if orig_bgr is None or annot_bgr is None:
                    error_count += 1
                    continue
                
                orig_resized = resize_to_training_size(orig_bgr)
                annot_resized = resize_to_training_size(annot_bgr)
                
                # Free immediately
                del orig_bgr, annot_bgr
                
                orig = cv2.cvtColor(orig_resized, cv2.COLOR_BGR2RGB)
                annot = cv2.cvtColor(annot_resized, cv2.COLOR_BGR2RGB)
                
                del orig_resized, annot_resized
                
                gt_weed = extract_ground_truth_weed(annot)
                pred_weed = generate_weed_prediction_multiscale(orig, model)
                
                del orig, annot
                
                m = calculate_metrics(gt_weed, pred_weed)
                m['filename'] = orig_fname
                metrics_list.append(m)
                
                agg_gt.append((gt_weed > 0).astype(int).flatten())
                agg_pred.append((pred_weed > 0).astype(int).flatten())
                
                del gt_weed, pred_weed
                
                # Aggressive cleanup
                if idx % SAVE_INTERVAL == 0:
                    pd.DataFrame(metrics_list).to_csv(csv_path, index=False)
                    gc.collect()
                
            except Exception as e:
                print(f"\nError {orig_fname}: {e}")
                error_count += 1
                gc.collect()
                continue
        
        # FINAL SAVE AND METRICS
        print("\n" + "="*60)
        print("SAVING RESULTS")
        print("="*60)
        
        df = pd.DataFrame(metrics_list)
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved: {csv_path.name}")
        
        if len(agg_gt) > 0:
            gt_all = np.concatenate(agg_gt)
            pred_all = np.concatenate(agg_pred)
            
            elapsed = time.time() - start_time
            
            print(f"\n{'='*60}")
            print("RESULTS")
            print(f"{'='*60}")
            print(f"Processed: {len(metrics_list)}/{len(matching_pairs)}")
            print(f"Errors: {error_count}")
            print(f"Time: {elapsed/60:.1f} minutes")
            
            print(f"\n{'='*60}")
            print("OVERALL METRICS")
            print(f"{'='*60}")
            print(f"  Accuracy:  {accuracy_score(gt_all, pred_all):.4f}")
            print(f"  Precision: {precision_score(gt_all, pred_all, zero_division=0):.4f}")
            print(f"  Recall:    {recall_score(gt_all, pred_all, zero_division=0):.4f}")
            print(f"  F1-Score:  {f1_score(gt_all, pred_all, zero_division=0):.4f}")
            
            inter = np.logical_and(gt_all, pred_all).sum()
            union = np.logical_or(gt_all, pred_all).sum()
            print(f"  IoU:       {(inter/union if union > 0 else 0):.4f}")
            
            print(f"\n{'='*60}")
            print("PER-IMAGE AVERAGES")
            print(f"{'='*60}")
            print(f"  Mean Precision: {df['precision'].mean():.4f}")
            print(f"  Mean Recall:    {df['recall'].mean():.4f}")
            print(f"  Mean F1:        {df['f1_score'].mean():.4f}")
            print(f"  Mean IoU:       {df['iou'].mean():.4f}")
            
            print(f"\n{'='*60}")
            print("✅ COMPLETE")
            print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        if 'metrics_list' in locals() and len(metrics_list) > 0:
            pd.DataFrame(metrics_list).to_csv(RESULTS_DIR / 'weed_results_partial.csv', index=False)
            print("Partial results saved")

if __name__ == "__main__":
    main()
