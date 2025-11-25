#!/usr/bin/env python3
"""
Weed Detection - Correct Flow
1. Detect paddy plants
2. Segment weeds (vegetation - paddy)
3. Check if weed regions fall inside GT weed boxes
4. Calculate metrics
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

RESIZE_TO = (800, 800)
DETECTION_SCALES = [1.0, 1.25, 1.5, 2.0]
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.3
SAVE_INTERVAL = 50

RESULTS_DIR.mkdir(exist_ok=True)

print("="*60)
print("WEED DETECTION - REGION vs BOX OVERLAP")
print("="*60)
print("Flow: Detect Paddy → Segment Weeds → Match with GT Boxes")
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


def resize_image_and_boxes(image, boxes, target_size=RESIZE_TO):
    """Resize image and scale bounding boxes"""
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    scale_x = target_w / w
    scale_y = target_h / h
    
    resized_image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    scaled_boxes = []
    for bbox in boxes:
        x, y, w_box, h_box = bbox
        x_new = x * scale_x
        y_new = y * scale_y
        w_new = w_box * scale_x
        h_new = h_box * scale_y
        scaled_boxes.append([x_new, y_new, w_new, h_new])
    
    return resized_image, scaled_boxes, (scale_x, scale_y)


def extract_shape_from_bbox(image_rgb, bbox, expansion=15):
    """Extract paddy plant shape using green filtering"""
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
    """
    Segment weeds: vegetation detection - paddy regions
    Returns binary mask of weed regions
    """
    r = image_rgb[:,:,0].astype(np.float32)
    g = image_rgb[:,:,1].astype(np.float32)
    b = image_rgb[:,:,2].astype(np.float32)
    
    # Vegetation detection
    exg = 2.0 * g - r - b
    exg_mask = (exg > 20).astype(np.uint8)
    green_ratio = g / (r + b + 1)
    ratio_mask = (green_ratio > 1.1).astype(np.uint8)
    green_dom = ((g > r + 5) & (g > b + 5) & (g > 30)).astype(np.uint8)
    
    veg_mask = ((exg_mask + ratio_mask + green_dom) >= 2).astype(np.uint8) * 255
    
    # Remove paddy regions to get weeds
    weed_mask = veg_mask.copy()
    weed_mask[paddy_mask > 0] = 0
    
    # Clean up
    weed_mask = cv2.morphologyEx(weed_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    weed_mask = cv2.morphologyEx(weed_mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    
    return weed_mask


@torch.no_grad()
def detect_paddy_and_segment_weeds(image_rgb, model):
    """
    Step 1: Detect paddy plants using model
    Step 2: Segment weeds (vegetation - paddy)
    Returns: weed_mask, paddy_mask
    """
    h, w = image_rgb.shape[:2]
    all_boxes, all_scores, all_labels = [], [], []
    
    # Multi-scale detection
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
    
    # Extract paddy regions (class 1)
    paddy_mask = np.zeros((h, w), dtype=np.uint8)
    if len(final_boxes) > 0:
        for box in final_boxes[final_labels == 1]:
            paddy_mask = cv2.bitwise_or(paddy_mask, extract_shape_from_bbox(image_rgb, box))
    
    # Segment weeds (vegetation - paddy)
    weed_mask = weed_segmentation(image_rgb, paddy_mask)
    
    return weed_mask, paddy_mask


def check_region_box_overlap(region_mask, gt_boxes):
    """
    Check if predicted weed regions overlap with GT weed boxes
    
    Args:
        region_mask: Binary mask of predicted weed regions
        gt_boxes: List of GT weed boxes in [x, y, w, h] format
    
    Returns:
        - Number of GT boxes with overlapping predictions
        - Number of predicted regions
        - IoU per GT box
    """
    if len(gt_boxes) == 0:
        # No GT boxes
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(
            region_mask.astype(np.uint8), connectivity=8
        )
        return {
            'gt_boxes_detected': 0,
            'total_gt_boxes': 0,
            'pred_regions': num_labels - 1,
            'iou_per_box': [],
            'avg_iou': 0.0
        }
    
    # Find predicted regions
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        region_mask.astype(np.uint8), connectivity=8
    )
    num_pred_regions = num_labels - 1
    
    # Check each GT box for overlap with predicted weed regions
    gt_detected = []
    ious = []
    
    for bbox in gt_boxes:
        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        if w <= 0 or h <= 0:
            continue
        
        # Create mask for this GT box
        gt_box_mask = np.zeros(region_mask.shape, dtype=np.uint8)
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(region_mask.shape[1], x + w), min(region_mask.shape[0], y + h)
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        gt_box_mask[y1:y2, x1:x2] = 255
        
        # Check overlap with predicted weed regions
        intersection = np.logical_and(region_mask > 0, gt_box_mask > 0).sum()
        union = np.logical_or(region_mask > 0, gt_box_mask > 0).sum()
        
        iou = intersection / union if union > 0 else 0.0
        ious.append(iou)
        
        # Consider detected if any overlap exists
        if intersection > 0:
            gt_detected.append(True)
        else:
            gt_detected.append(False)
    
    return {
        'gt_boxes_detected': sum(gt_detected),
        'total_gt_boxes': len(gt_boxes),
        'pred_regions': num_pred_regions,
        'iou_per_box': ious,
        'avg_iou': np.mean(ious) if ious else 0.0,
        'detection_rate': sum(gt_detected) / len(gt_boxes) if len(gt_boxes) > 0 else 0.0
    }


# ==================== MAIN ====================

def main():
    start_time = time.time()
    
    # Load COCO annotations
    print(f"\nLoading COCO annotations...")
    with open(COCO_ANNOTATION_FILE, 'r') as f:
        coco_data = json.load(f)
    
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    print(f"Categories: {categories}")
    
    # All categories are weed-related
    weed_cat_ids = set(categories.keys())
    
    # Build lookups
    image_info_dict = {img['id']: img for img in coco_data['images']}
    
    # Group weed annotations by image
    image_annotations = {img_id: [] for img_id in image_info_dict.keys()}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        cat_id = ann['category_id']
        if img_id in image_annotations and cat_id in weed_cat_ids:
            image_annotations[img_id].append(ann['bbox'])
    
    total_annotations = sum(len(anns) for anns in image_annotations.values())
    print(f"\nTotal images: {len(image_info_dict)}")
    print(f"Total weed boxes (GT): {total_annotations}")
    
    # Load model
    model = load_paddy_model(MODEL_PATH)
    
    print(f"\nProcessing: Detect Paddy → Segment Weeds → Match GT Boxes\n")
    
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
            
            # Get GT weed boxes
            gt_boxes_coco = image_annotations[img_id]
            
            # Resize image and boxes to match training size
            resized_rgb, scaled_boxes, _ = resize_image_and_boxes(
                image_rgb, gt_boxes_coco, RESIZE_TO
            )
            
            # STEP 1 & 2: Detect paddy and segment weeds
            weed_mask, paddy_mask = detect_paddy_and_segment_weeds(resized_rgb, model)
            
            # STEP 3: Check overlap between weed regions and GT boxes
            overlap_metrics = check_region_box_overlap(weed_mask, scaled_boxes)
            
            # Calculate metrics
            precision = overlap_metrics['gt_boxes_detected'] / max(overlap_metrics['pred_regions'], 1)
            recall = overlap_metrics['detection_rate']
            f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
            
            result = {
                'filename': img_info['file_name'],
                'image_id': img_id,
                'num_gt_boxes': len(gt_boxes_coco),
                'gt_boxes_detected': overlap_metrics['gt_boxes_detected'],
                'pred_weed_regions': overlap_metrics['pred_regions'],
                'detection_rate': overlap_metrics['detection_rate'],
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'avg_iou': overlap_metrics['avg_iou'],
                'weed_pixels': int((weed_mask > 0).sum()),
                'paddy_pixels': int((paddy_mask > 0).sum())
            }
            
            metrics_list.append(result)
            processed += 1
            
            if processed % SAVE_INTERVAL == 0:
                pd.DataFrame(metrics_list).to_csv(csv_path, index=False)
                gc.collect()
            
        except Exception as e:
            print(f"\nError on {img_info['file_name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final save
    df = pd.DataFrame(metrics_list)
    df.to_csv(csv_path, index=False)
    
    elapsed = time.time() - start_time
    
    # Results
    print(f"\n{'='*60}")
    print("RESULTS - WEED REGION vs GT BOX OVERLAP")
    print(f"{'='*60}")
    print(f"Processed: {len(metrics_list)} images")
    print(f"Time: {elapsed/60:.1f} minutes\n")
    
    print(f"📊 DETECTION METRICS:")
    print(f"  Detection Rate (Recall): {df['detection_rate'].mean():.4f}")
    print(f"  Precision:               {df['precision'].mean():.4f}")
    print(f"  F1 Score:                {df['f1_score'].mean():.4f}")
    print(f"  Mean IoU:                {df['avg_iou'].mean():.4f}")
    
    print(f"\n📈 Statistics:")
    print(f"  Total GT weed boxes:      {df['num_gt_boxes'].sum()}")
    print(f"  GT boxes detected:        {df['gt_boxes_detected'].sum()}")
    print(f"  Total pred weed regions:  {df['pred_weed_regions'].sum()}")
    print(f"  Avg GT boxes/image:       {df['num_gt_boxes'].mean():.1f}")
    print(f"  Avg detected/image:       {df['gt_boxes_detected'].mean():.1f}")
    print(f"  Avg weed regions/image:   {df['pred_weed_regions'].mean():.1f}")
    
    print(f"\n✅ Saved: {csv_path.name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
