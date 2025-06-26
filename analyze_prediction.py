#!/usr/bin/env python3
"""
Analyze the prediction values to understand the low Dice score
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def analyze_prediction():
    """Analyze the verbose test prediction"""
    
    # Load the prediction and ground truth
    pred_path = "verbose_test_output/sub-r048s014_ses-1_verbose_prediction.nii.gz"
    
    test_case = "sub-r048s014_ses-1"
    test_dir = "/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Testing_Split"
    mask_path = f"{test_dir}/Masks/{test_case}_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz"
    
    print("Analyzing prediction values...")
    
    # Load data
    pred_img = nib.load(pred_path)
    pred_data = pred_img.get_fdata()
    
    true_img = nib.load(mask_path)
    true_data = true_img.get_fdata()
    
    print(f"Prediction shape: {pred_data.shape}")
    print(f"Ground truth shape: {true_data.shape}")
    print(f"Prediction range: [{pred_data.min():.6f}, {pred_data.max():.6f}]")
    print(f"Ground truth range: [{true_data.min():.6f}, {true_data.max():.6f}]")
    
    # Check if this is actually the raw prediction or thresholded
    unique_pred = np.unique(pred_data)
    unique_true = np.unique(true_data)
    
    print(f"Unique prediction values: {unique_pred}")
    print(f"Unique ground truth values: {unique_true}")
    
    # Counts
    pred_lesion = np.sum(pred_data > 0)
    true_lesion = np.sum(true_data > 0)
    
    print(f"Predicted lesion voxels: {pred_lesion:,}")
    print(f"True lesion voxels: {true_lesion:,}")
    
    # If this is binary, we need to check what happened to the raw probabilities
    if len(unique_pred) <= 2:
        print("\n❌ This is a binary prediction (already thresholded)")
        print("The issue is that the model's raw probabilities were likely very low")
        print("We need to check raw model output before thresholding")
    else:
        print("\n✅ This contains probability values")
        # Analyze probability distribution
        lesion_probs = pred_data[true_data > 0]
        non_lesion_probs = pred_data[true_data == 0]
        
        print(f"Probabilities in true lesion regions: [{lesion_probs.min():.4f}, {lesion_probs.max():.4f}]")
        print(f"Mean probability in lesion: {lesion_probs.mean():.4f}")
        print(f"Mean probability in non-lesion: {non_lesion_probs.mean():.4f}")
        
        # Test different thresholds
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        print(f"\nDice scores at different thresholds:")
        for thresh in thresholds:
            pred_thresh = (pred_data > thresh).astype(int)
            intersection = np.sum(pred_thresh * true_data)
            union = np.sum(pred_thresh) + np.sum(true_data)
            dice = (2.0 * intersection) / union if union > 0 else 0.0
            pred_count = np.sum(pred_thresh)
            print(f"  Threshold {thresh:.1f}: Dice = {dice:.4f}, Predicted voxels = {pred_count:,}")

if __name__ == "__main__":
    analyze_prediction()
