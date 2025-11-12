#!/usr/bin/env python3
"""
Calculate Overall Metrics from Bounding Box Results CSV
Calculates weighted averages and overall statistics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def calculate_overall_metrics(csv_path):
    """
    Calculate overall metrics from per-image results CSV
    """
    
    # Load results
    print(f"\n{'='*60}")
    print("LOADING RESULTS")
    print(f"{'='*60}")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded: {csv_path}")
        print(f"   Total images: {len(df)}")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        sys.exit(1)
    
    # Check required columns
    required_cols = ['bbox_precision', 'bbox_recall', 'bbox_f1', 
                     'pixel_precision', 'pixel_recall', 'pixel_f1', 'pixel_iou',
                     'gt_boxes', 'pred_boxes', 'matched_boxes']
    
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"❌ Missing columns: {missing}")
        sys.exit(1)
    
    print(f"✅ All required columns found\n")
    
    # Remove rows with no ground truth (can't evaluate)
    df_eval = df[df['gt_boxes'] > 0].copy()
    print(f"Images with GT weeds: {len(df_eval)}/{len(df)}")
    
    if len(df_eval) == 0:
        print("❌ No images with ground truth weeds!")
        sys.exit(1)
    
    # ==================== BOUNDING BOX METRICS ====================
    print(f"\n{'='*60}")
    print("🎯 BOUNDING BOX METRICS (Primary)")
    print(f"{'='*60}")
    
    # Simple averages
    bbox_prec_mean = df_eval['bbox_precision'].mean()
    bbox_rec_mean = df_eval['bbox_recall'].mean()
    bbox_f1_mean = df_eval['bbox_f1'].mean()
    
    print(f"\nSimple Averages:")
    print(f"  Precision: {bbox_prec_mean:.4f} (± {df_eval['bbox_precision'].std():.4f})")
    print(f"  Recall:    {bbox_rec_mean:.4f} (± {df_eval['bbox_recall'].std():.4f})")
    print(f"  F1-Score:  {bbox_f1_mean:.4f} (± {df_eval['bbox_f1'].std():.4f})")
    
    # Weighted averages (weighted by number of GT boxes)
    total_gt = df_eval['gt_boxes'].sum()
    weights = df_eval['gt_boxes'] / total_gt
    
    bbox_prec_weighted = (df_eval['bbox_precision'] * weights).sum()
    bbox_rec_weighted = (df_eval['bbox_recall'] * weights).sum()
    bbox_f1_weighted = (df_eval['bbox_f1'] * weights).sum()
    
    print(f"\nWeighted Averages (by GT boxes):")
    print(f"  Precision: {bbox_prec_weighted:.4f}")
    print(f"  Recall:    {bbox_rec_weighted:.4f}")
    print(f"  F1-Score:  {bbox_f1_weighted:.4f}")
    
    # Overall aggregated metrics
    total_matched = df_eval['matched_boxes'].sum()
    total_pred = df_eval['pred_boxes'].sum()
    total_fp = df_eval['false_positives'].sum()
    total_fn = df_eval['false_negatives'].sum()
    
    overall_prec = total_matched / total_pred if total_pred > 0 else 0
    overall_rec = total_matched / total_gt if total_gt > 0 else 0
    overall_f1 = 2 * (overall_prec * overall_rec) / (overall_prec + overall_rec + 1e-10)
    
    print(f"\nOverall Aggregated:")
    print(f"  Precision: {overall_prec:.4f}")
    print(f"  Recall:    {overall_rec:.4f}")
    print(f"  F1-Score:  {overall_f1:.4f}")
    
    # Detection statistics
    print(f"\nDetection Statistics:")
    print(f"  Total GT weeds:        {int(total_gt)}")
    print(f"  Total predicted:       {int(total_pred)}")
    print(f"  Successfully matched:  {int(total_matched)} ({total_matched/total_gt*100:.1f}%)")
    print(f"  False positives:       {int(total_fp)}")
    print(f"  False negatives:       {int(total_fn)}")
    
    # ==================== PIXEL METRICS ====================
    print(f"\n{'='*60}")
    print("📊 PIXEL METRICS (Reference)")
    print(f"{'='*60}")
    
    # Simple averages
    pixel_prec_mean = df_eval['pixel_precision'].mean()
    pixel_rec_mean = df_eval['pixel_recall'].mean()
    pixel_f1_mean = df_eval['pixel_f1'].mean()
    pixel_iou_mean = df_eval['pixel_iou'].mean()
    pixel_acc_mean = df_eval['pixel_accuracy'].mean()
    
    print(f"\nSimple Averages:")
    print(f"  Accuracy:  {pixel_acc_mean:.4f} (± {df_eval['pixel_accuracy'].std():.4f})")
    print(f"  Precision: {pixel_prec_mean:.4f} (± {df_eval['pixel_precision'].std():.4f})")
    print(f"  Recall:    {pixel_rec_mean:.4f} (± {df_eval['pixel_recall'].std():.4f})")
    print(f"  F1-Score:  {pixel_f1_mean:.4f} (± {df_eval['pixel_f1'].std():.4f})")
    print(f"  IoU:       {pixel_iou_mean:.4f} (± {df_eval['pixel_iou'].std():.4f})")
    
    # Weighted averages (weighted by GT pixels)
    total_gt_pixels = df_eval['gt_pixels'].sum()
    pixel_weights = df_eval['gt_pixels'] / total_gt_pixels
    
    pixel_prec_weighted = (df_eval['pixel_precision'] * pixel_weights).sum()
    pixel_rec_weighted = (df_eval['pixel_recall'] * pixel_weights).sum()
    pixel_f1_weighted = (df_eval['pixel_f1'] * pixel_weights).sum()
    pixel_iou_weighted = (df_eval['pixel_iou'] * pixel_weights).sum()
    
    print(f"\nWeighted Averages (by GT pixels):")
    print(f"  Precision: {pixel_prec_weighted:.4f}")
    print(f"  Recall:    {pixel_rec_weighted:.4f}")
    print(f"  F1-Score:  {pixel_f1_weighted:.4f}")
    print(f"  IoU:       {pixel_iou_weighted:.4f}")
    
    # Pixel statistics
    total_pred_pixels = df_eval['pred_pixels'].sum()
    
    print(f"\nPixel Statistics:")
    print(f"  Total GT pixels:   {int(total_gt_pixels):,}")
    print(f"  Total pred pixels: {int(total_pred_pixels):,}")
    print(f"  Ratio (pred/GT):   {total_pred_pixels/total_gt_pixels:.2f}x")
    
    # ==================== DISTRIBUTION ANALYSIS ====================
    print(f"\n{'='*60}")
    print("📈 DISTRIBUTION ANALYSIS")
    print(f"{'='*60}")
    
    print(f"\nBounding Box F1 Distribution:")
    print(f"  Median:  {df_eval['bbox_f1'].median():.4f}")
    print(f"  Min:     {df_eval['bbox_f1'].min():.4f}")
    print(f"  Max:     {df_eval['bbox_f1'].max():.4f}")
    print(f"  25th %:  {df_eval['bbox_f1'].quantile(0.25):.4f}")
    print(f"  75th %:  {df_eval['bbox_f1'].quantile(0.75):.4f}")
    
    # Count images by performance
    excellent = len(df_eval[df_eval['bbox_f1'] >= 0.8])
    good = len(df_eval[(df_eval['bbox_f1'] >= 0.6) & (df_eval['bbox_f1'] < 0.8)])
    fair = len(df_eval[(df_eval['bbox_f1'] >= 0.4) & (df_eval['bbox_f1'] < 0.6)])
    poor = len(df_eval[df_eval['bbox_f1'] < 0.4])
    
    print(f"\nPerformance Distribution:")
    print(f"  Excellent (F1 ≥ 0.8): {excellent} ({excellent/len(df_eval)*100:.1f}%)")
    print(f"  Good (0.6 ≤ F1 < 0.8): {good} ({good/len(df_eval)*100:.1f}%)")
    print(f"  Fair (0.4 ≤ F1 < 0.6): {fair} ({fair/len(df_eval)*100:.1f}%)")
    print(f"  Poor (F1 < 0.4):       {poor} ({poor/len(df_eval)*100:.1f}%)")
    
    # ==================== TOP/BOTTOM IMAGES ====================
    print(f"\n{'='*60}")
    print("🏆 TOP 5 IMAGES (by bbox F1)")
    print(f"{'='*60}")
    
    top5 = df_eval.nlargest(5, 'bbox_f1')[['filename', 'bbox_f1', 'bbox_precision', 'bbox_recall', 'gt_boxes', 'pred_boxes']]
    for idx, row in top5.iterrows():
        print(f"\n{row['filename']}")
        print(f"  F1: {row['bbox_f1']:.4f}  Prec: {row['bbox_precision']:.4f}  Rec: {row['bbox_recall']:.4f}")
        print(f"  GT boxes: {int(row['gt_boxes'])}  Pred boxes: {int(row['pred_boxes'])}")
    
    print(f"\n{'='*60}")
    print("⚠️  BOTTOM 5 IMAGES (by bbox F1)")
    print(f"{'='*60}")
    
    bottom5 = df_eval.nsmallest(5, 'bbox_f1')[['filename', 'bbox_f1', 'bbox_precision', 'bbox_recall', 'gt_boxes', 'pred_boxes']]
    for idx, row in bottom5.iterrows():
        print(f"\n{row['filename']}")
        print(f"  F1: {row['bbox_f1']:.4f}  Prec: {row['bbox_precision']:.4f}  Rec: {row['bbox_recall']:.4f}")
        print(f"  GT boxes: {int(row['gt_boxes'])}  Pred boxes: {int(row['pred_boxes'])}")
    
    # ==================== SAVE SUMMARY ====================
    summary = {
        # Bbox metrics
        'bbox_precision_mean': bbox_prec_mean,
        'bbox_recall_mean': bbox_rec_mean,
        'bbox_f1_mean': bbox_f1_mean,
        'bbox_precision_weighted': bbox_prec_weighted,
        'bbox_recall_weighted': bbox_rec_weighted,
        'bbox_f1_weighted': bbox_f1_weighted,
        'bbox_precision_overall': overall_prec,
        'bbox_recall_overall': overall_rec,
        'bbox_f1_overall': overall_f1,
        
        # Pixel metrics
        'pixel_accuracy_mean': pixel_acc_mean,
        'pixel_precision_mean': pixel_prec_mean,
        'pixel_recall_mean': pixel_rec_mean,
        'pixel_f1_mean': pixel_f1_mean,
        'pixel_iou_mean': pixel_iou_mean,
        'pixel_precision_weighted': pixel_prec_weighted,
        'pixel_recall_weighted': pixel_rec_weighted,
        'pixel_f1_weighted': pixel_f1_weighted,
        'pixel_iou_weighted': pixel_iou_weighted,
        
        # Counts
        'total_images': len(df_eval),
        'total_gt_boxes': int(total_gt),
        'total_pred_boxes': int(total_pred),
        'total_matched_boxes': int(total_matched),
        'detection_rate': f"{total_matched/total_gt*100:.2f}%"
    }
    
    summary_df = pd.DataFrame([summary])
    output_path = csv_path.replace('.csv', '_summary.csv')
    summary_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"✅ Summary saved: {output_path}")
    print(f"{'='*60}\n")
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate overall metrics from bbox results CSV')
    parser.add_argument('csv_file', type=str, help='Path to results CSV file')
    
    args = parser.parse_args()
    
    if not Path(args.csv_file).exists():
        print(f"❌ File not found: {args.csv_file}")
        sys.exit(1)
    
    calculate_overall_metrics(args.csv_file)
