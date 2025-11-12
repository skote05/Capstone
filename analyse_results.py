#!/usr/bin/env python3
"""
Analyze Weed Detection Results - Calculate Overall Metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ==================== CONFIGURATION ====================

# Path to your saved CSV (CHANGE THIS to your actual path)
csv_path = Path('/Users/shashankkote/Desktop2/Capstone/weed_detection_env/weed_detection/results/weed_results_temp (1).csv')

print("="*60)
print("WEED DETECTION RESULTS ANALYSIS")
print("="*60)

# Load results
print(f"\nLoading: {csv_path.name}")
df = pd.read_csv(csv_path)
print(f"✅ Loaded {len(df)} results\n")

# ==================== OVERALL METRICS ====================

print("="*60)
print("OVERALL RESULTS (Aggregated Across All Images)")
print("="*60)

# Method 1: Weighted average by image size
total_pixels = df['gt_pixels'].sum() + (df['accuracy'] * 0).sum()  # Get total pixels from data

# Calculate overall metrics using weighted averages
weights = df['gt_pixels'] + df['pred_pixels']  # Weight by total pixels in each image
weights = weights / weights.sum()  # Normalize

overall_acc_weighted = (df['accuracy'] * weights).sum()
overall_prec_weighted = (df['precision'] * weights).sum()
overall_rec_weighted = (df['recall'] * weights).sum()
overall_f1_weighted = (df['f1_score'] * weights).sum()
overall_iou_weighted = (df['iou'] * weights).sum()

print(f"\nWeighted Overall Metrics (by image size):")
print(f"  Accuracy:  {overall_acc_weighted:.4f}")
print(f"  Precision: {overall_prec_weighted:.4f}")
print(f"  Recall:    {overall_rec_weighted:.4f}")
print(f"  F1-Score:  {overall_f1_weighted:.4f}")
print(f"  IoU:       {overall_iou_weighted:.4f}")

# Method 2: Simple average
print(f"\nSimple Average Across All Images:")
print(f"  Accuracy:  {df['accuracy'].mean():.4f}")
print(f"  Precision: {df['precision'].mean():.4f}")
print(f"  Recall:    {df['recall'].mean():.4f}")
print(f"  F1-Score:  {df['f1_score'].mean():.4f}")
print(f"  IoU:       {df['iou'].mean():.4f}")

# Method 3: Pixel-level totals (most accurate)
total_gt_pixels = df['gt_pixels'].sum()
total_pred_pixels = df['pred_pixels'].sum()

print(f"\nPixel-Level Statistics:")
print(f"  Total Ground Truth Weed Pixels: {total_gt_pixels:,}")
print(f"  Total Predicted Weed Pixels:    {total_pred_pixels:,}")
if total_gt_pixels > 0:
    coverage = (total_pred_pixels / total_gt_pixels) * 100
    print(f"  Prediction Coverage: {coverage:.2f}% of ground truth")

# ==================== PER-IMAGE STATISTICS ====================

print(f"\n{'='*60}")
print("PER-IMAGE STATISTICS")
print(f"{'='*60}")

print(f"\nIoU Metrics:")
print(f"  Mean:   {df['iou'].mean():.4f}")
print(f"  Median: {df['iou'].median():.4f}")
print(f"  Std:    {df['iou'].std():.4f}")
print(f"  Min:    {df['iou'].min():.4f}")
print(f"  Max:    {df['iou'].max():.4f}")

print(f"\nF1-Score Metrics:")
print(f"  Mean:   {df['f1_score'].mean():.4f}")
print(f"  Median: {df['f1_score'].median():.4f}")
print(f"  Std:    {df['f1_score'].std():.4f}")

print(f"\nPrecision Metrics:")
print(f"  Mean:   {df['precision'].mean():.4f}")
print(f"  Median: {df['precision'].median():.4f}")
print(f"  Std:    {df['precision'].std():.4f}")

print(f"\nRecall Metrics:")
print(f"  Mean:   {df['recall'].mean():.4f}")
print(f"  Median: {df['recall'].median():.4f}")
print(f"  Std:    {df['recall'].std():.4f}")

print(f"\nAccuracy Metrics:")
print(f"  Mean:   {df['accuracy'].mean():.4f}")
print(f"  Median: {df['accuracy'].median():.4f}")
print(f"  Std:    {df['accuracy'].std():.4f}")

# ==================== TOP/BOTTOM PERFORMERS ====================

print(f"\n{'='*60}")
print("TOP 10 IMAGES BY IoU")
print(f"{'='*60}")
df_sorted = df.sort_values('iou', ascending=False)
print(f"{'Rank':<5} {'Filename':<35} {'IoU':<8} {'F1':<8} {'Precision':<10} {'Recall':<8}")
print("-" * 85)
for i, (idx, row) in enumerate(df_sorted.head(10).iterrows(), 1):
    print(f"{i:<5} {row['filename']:<35} {row['iou']:<8.4f} {row['f1_score']:<8.4f} {row['precision']:<10.4f} {row['recall']:<8.4f}")

print(f"\n{'='*60}")
print("BOTTOM 10 IMAGES BY IoU")
print(f"{'='*60}")
print(f"{'Filename':<35} {'IoU':<8} {'F1':<8} {'Precision':<10} {'Recall':<8}")
print("-" * 75)
for idx, row in df_sorted.tail(10).iterrows():
    print(f"{row['filename']:<35} {row['iou']:<8.4f} {row['f1_score']:<8.4f} {row['precision']:<10.4f} {row['recall']:<8.4f}")

# ==================== IoU DISTRIBUTION ====================

print(f"\n{'='*60}")
print("IoU DISTRIBUTION")
print(f"{'='*60}")

bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
iou_dist = pd.cut(df['iou'], bins=bins, include_lowest=True).value_counts().sort_index()

print(f"{'IoU Range':<20} {'Count':<10} {'Percentage':<12} {'Bar'}")
print("-" * 65)
for interval, count in iou_dist.items():
    pct = (count / len(df) * 100)
    bar = '█' * int(pct / 2)  # Scale to fit console
    print(f"{str(interval):<20} {count:<10} {pct:5.1f}%      {bar}")

# ==================== QUALITY CATEGORIES ====================

print(f"\n{'='*60}")
print("PERFORMANCE CATEGORIES")
print(f"{'='*60}")

excellent = len(df[df['iou'] >= 0.8])
good = len(df[(df['iou'] >= 0.6) & (df['iou'] < 0.8)])
fair = len(df[(df['iou'] >= 0.4) & (df['iou'] < 0.6)])
poor = len(df[(df['iou'] >= 0.2) & (df['iou'] < 0.4)])
very_poor = len(df[df['iou'] < 0.2])

print(f"Excellent (IoU ≥ 0.8):   {excellent:4d} images ({excellent/len(df)*100:5.1f}%)")
print(f"Good      (0.6 ≤ IoU < 0.8): {good:4d} images ({good/len(df)*100:5.1f}%)")
print(f"Fair      (0.4 ≤ IoU < 0.6): {fair:4d} images ({fair/len(df)*100:5.1f}%)")
print(f"Poor      (0.2 ≤ IoU < 0.4): {poor:4d} images ({poor/len(df)*100:5.1f}%)")
print(f"Very Poor (IoU < 0.2):   {very_poor:4d} images ({very_poor/len(df)*100:5.1f}%)")

# ==================== SAVE SUMMARY ====================

print(f"\n{'='*60}")
print("SAVING SUMMARY")
print(f"{'='*60}")

summary = {
    'Total Images': len(df),
    'Overall IoU (Weighted)': overall_iou_weighted,
    'Overall F1 (Weighted)': overall_f1_weighted,
    'Overall Precision (Weighted)': overall_prec_weighted,
    'Overall Recall (Weighted)': overall_rec_weighted,
    'Overall Accuracy (Weighted)': overall_acc_weighted,
    'Mean IoU': df['iou'].mean(),
    'Median IoU': df['iou'].median(),
    'Std IoU': df['iou'].std(),
    'Mean F1': df['f1_score'].mean(),
    'Mean Precision': df['precision'].mean(),
    'Mean Recall': df['recall'].mean(),
    'Mean Accuracy': df['accuracy'].mean(),
    'Total GT Pixels': int(total_gt_pixels),
    'Total Pred Pixels': int(total_pred_pixels),
    'Excellent (≥0.8)': excellent,
    'Good (0.6-0.8)': good,
    'Fair (0.4-0.6)': fair,
    'Poor (0.2-0.4)': poor,
    'Very Poor (<0.2)': very_poor
}

summary_df = pd.DataFrame([summary])
summary_path = csv_path.parent / f'summary_{csv_path.stem}.csv'
summary_df.to_csv(summary_path, index=False)

print(f"✅ Summary saved to: {summary_path}")

# Save detailed stats
stats_path = csv_path.parent / f'detailed_stats_{csv_path.stem}.txt'
with open(stats_path, 'w') as f:
    f.write("="*60 + "\n")
    f.write("WEED DETECTION RESULTS - DETAILED STATISTICS\n")
    f.write("="*60 + "\n\n")
    
    f.write("OVERALL WEIGHTED METRICS:\n")
    f.write(f"  IoU:       {overall_iou_weighted:.4f}\n")
    f.write(f"  F1-Score:  {overall_f1_weighted:.4f}\n")
    f.write(f"  Precision: {overall_prec_weighted:.4f}\n")
    f.write(f"  Recall:    {overall_rec_weighted:.4f}\n")
    f.write(f"  Accuracy:  {overall_acc_weighted:.4f}\n\n")
    
    f.write("PER-IMAGE AVERAGES:\n")
    f.write(f"  Mean IoU:  {df['iou'].mean():.4f} ± {df['iou'].std():.4f}\n")
    f.write(f"  Mean F1:   {df['f1_score'].mean():.4f} ± {df['f1_score'].std():.4f}\n\n")
    
    f.write("PIXEL STATISTICS:\n")
    f.write(f"  Total GT Pixels:   {total_gt_pixels:,}\n")
    f.write(f"  Total Pred Pixels: {total_pred_pixels:,}\n")

print(f"✅ Detailed stats saved to: {stats_path}")

print(f"\n{'='*60}")
print("✅ ANALYSIS COMPLETE!")
print(f"{'='*60}")
print(f"\nKey Results:")
print(f"  • Processed {len(df)} images")
print(f"  • Overall Weighted IoU: {overall_iou_weighted:.4f}")
print(f"  • Overall Weighted F1:  {overall_f1_weighted:.4f}")
print(f"  • Mean IoU: {df['iou'].mean():.4f}")
print(f"  • Mean F1:  {df['f1_score'].mean():.4f}")
print(f"{'='*60}")
