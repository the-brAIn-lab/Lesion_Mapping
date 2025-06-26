#!/usr/bin/env python3
"""
Simple test of both orientations - minimal dependencies
"""

import os
import sys
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.ndimage import zoom

def setup_tf():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

def simple_test():
    print("Simple Orientation Test")
    print("======================")
    
    setup_tf()
    
    # Load model
    model_path = "/mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota/callbacks/sota_fixed_20250619_063330/best_model.h5"
    
    # Simple custom objects (minimal)
    def simple_loss(y_true, y_pred):
        return tf.keras.losses.binary_crossentropy(y_true, y_pred)
    
    def simple_dice(y_true, y_pred):
        return tf.constant(0.0)  # Dummy metric
    
    custom_objects = {
        'combined_loss': simple_loss,
        'dice_coefficient': simple_dice,
        'binary_dice_coefficient': simple_dice,
        'dice_loss': simple_loss,
        'focal_loss': simple_loss
    }
    
    try:
        model = load_model(model_path, custom_objects=custom_objects, compile=False)
        print(f"✅ Model loaded: {model.count_params():,} parameters")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return
    
    # Load test data
    test_case = "sub-r048s014_ses-1"
    test_dir = "/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Testing_Split"
    image_path = f"{test_dir}/Images/{test_case}_space-MNI152NLin2009aSym_T1w.nii.gz"
    mask_path = f"{test_dir}/Masks/{test_case}_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz"
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # Load and preprocess
    nii_img = nib.load(image_path)
    img_data = nii_img.get_fdata(dtype=np.float32)
    true_mask = nib.load(mask_path).get_fdata().astype(np.uint8)
    
    print(f"Original shape: {img_data.shape}")
    print(f"True lesion voxels: {np.sum(true_mask):,}")
    
    # Simple preprocessing (exactly like training)
    if img_data.shape != (192, 224, 176):
        factors = [192/img_data.shape[0], 224/img_data.shape[1], 176/img_data.shape[2]]
        img_data = zoom(img_data, factors, order=1)
    
    p1, p99 = np.percentile(img_data[img_data > 0], [1, 99])
    img_data = np.clip(img_data, p1, p99)
    img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min() + 1e-8)
    processed = img_data[..., np.newaxis]
    
    # Test both orientations
    orientations = [
        ("Original", processed),
        ("Flipped", np.flip(processed, axis=1))
    ]
    
    for name, data in orientations:
        print(f"\n--- Testing {name} ---")
        
        batch = data[np.newaxis, ...]
        print(f"Batch shape: {batch.shape}")
        
        try:
            # Simple prediction
            with tf.device('/GPU:0'):
                pred = model(batch, training=False)
            
            pred_vol = pred[0, ..., 0].numpy()
            
            # If flipped input, flip back
            if name == "Flipped":
                pred_vol = np.flip(pred_vol, axis=1)
            
            print(f"Prediction range: [{pred_vol.min():.6f}, {pred_vol.max():.6f}]")
            print(f"Prediction mean: {pred_vol.mean():.6f}")
            
            # Test a few thresholds
            for thresh in [0.01, 0.1, 0.3, 0.5]:
                binary = (pred_vol > thresh).astype(np.uint8)
                
                # Simple resize back
                if binary.shape != true_mask.shape:
                    factors = [true_mask.shape[i] / binary.shape[i] for i in range(3)]
                    binary = zoom(binary, factors, order=0)
                    binary = (binary > 0.5).astype(np.uint8)
                
                # Calculate Dice
                intersection = np.sum(binary * true_mask)
                union = np.sum(binary) + np.sum(true_mask)
                dice = (2.0 * intersection) / union if union > 0 else 0.0
                
                pred_voxels = np.sum(binary)
                print(f"  Threshold {thresh:.2f}: Dice={dice:.4f}, Predicted={pred_voxels:,}")
            
            del batch, pred, pred_vol
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            continue
    
    print(f"\n✅ Simple orientation test completed")

if __name__ == "__main__":
    simple_test()
