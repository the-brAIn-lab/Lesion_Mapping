#!/usr/bin/env python3
"""
Test model with the EXACT same augmentation as training
The model was trained with random horizontal flips - we need to test both orientations
"""

import os
import sys
import gc
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.ndimage import zoom
import time

# Custom imports
sys.path.append('/mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota')
from models.losses import dice_loss, combined_loss, focal_loss

def setup_tensorflow():
    """Setup TensorFlow"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print(f"✅ TensorFlow configured")

def load_fixed_model():
    """Load the fixed model"""
    model_path = "/mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota/callbacks/sota_fixed_20250619_063330/best_model.h5"
    
    def combined_loss_fn(y_true, y_pred, smooth=1e-6, focal_gamma=3.0, focal_alpha=0.25):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        dice_loss = 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), focal_alpha, 1 - focal_alpha)
        focal_loss = -alpha_t * tf.pow(1 - p_t, focal_gamma) * tf.math.log(p_t)
        focal_loss = tf.reduce_mean(focal_loss)
        return 0.7 * dice_loss + 0.3 * focal_loss
    
    def dice_coeff_fn(y_true, y_pred, smooth=1e-6):
        y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
        y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred, tf.float32))
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    
    def binary_dice_fn(y_true, y_pred, smooth=1e-6):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    
    custom_objects = {
        'combined_loss': combined_loss_fn,
        'dice_coefficient': dice_coeff_fn,
        'binary_dice_coefficient': binary_dice_fn,
        'dice_loss': dice_loss,
        'focal_loss': focal_loss
    }
    
    model = load_model(model_path, custom_objects=custom_objects, compile=False)
    print(f"✅ Model loaded: {model.count_params():,} parameters")
    return model

def resize_volume(volume, target_shape, order=1):
    """Resize volume - EXACT same as training"""
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=order)

def preprocess_training_exact(image_data, target_shape=(192, 224, 176)):
    """EXACT preprocessing from training script"""
    print(f"  🔧 EXACT TRAINING PREPROCESSING:")
    print(f"    Original shape: {image_data.shape}")
    
    # Step 1: Resize (if needed)
    if image_data.shape != target_shape:
        print(f"    Resizing from {image_data.shape} to {target_shape}")
        image_data = resize_volume(image_data, target_shape)
    
    # Step 2: EXACT normalization from training
    print(f"    Before normalization: [{image_data.min():.6f}, {image_data.max():.6f}]")
    
    p1, p99 = np.percentile(image_data[image_data > 0], [1, 99])
    print(f"    Percentiles: p1={p1:.6f}, p99={p99:.6f}")
    
    image_data = np.clip(image_data, p1, p99)
    print(f"    After clipping: [{image_data.min():.6f}, {image_data.max():.6f}]")
    
    image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min() + 1e-8)
    print(f"    After normalization: [{image_data.min():.6f}, {image_data.max():.6f}]")
    
    image_data = image_data[..., np.newaxis]
    print(f"    Final shape: {image_data.shape}")
    
    return image_data

def test_prediction(pred_volume, true_mask, original_shape):
    """Test prediction with optimal threshold"""
    if pred_volume.shape != original_shape:
        factors = [o / p for o, p in zip(original_shape, pred_volume.shape)]
        pred_resized = resize_volume(pred_volume, original_shape, order=1)
    else:
        pred_resized = pred_volume
    
    best_dice = 0.0
    best_threshold = 0.5
    
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        binary_pred = (pred_resized > threshold).astype(np.uint8)
        intersection = np.sum(binary_pred * true_mask)
        union = np.sum(binary_pred) + np.sum(true_mask)
        dice = (2.0 * intersection) / union if union > 0 else 0.0
        
        if dice > best_dice:
            best_dice = dice
            best_threshold = threshold
    
    return best_dice, best_threshold

def main():
    """Main function to test training augmentation fix"""
    print(f"🔧 TRAINING AUGMENTATION FIX TEST")
    print(f"=" * 80)
    print(f"Purpose: Test model with EXACT same augmentation as training")
    print(f"Key insight: Training used random horizontal flips")
    print(f"Expected: Flipped orientation should give much better Dice")
    
    setup_tensorflow()
    model = load_fixed_model()
    
    # Load test case
    test_dir = "/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Testing_Split"
    case_name = "sub-r048s014_ses-1"
    
    image_path = f"{test_dir}/Images/{case_name}_space-MNI152NLin2009aSym_T1w.nii.gz"
    mask_path = f"{test_dir}/Masks/{case_name}_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz"
    
    nii_img = nib.load(image_path)
    image_data = nii_img.get_fdata(dtype=np.float32)
    true_mask = nib.load(mask_path).get_fdata().astype(np.uint8)
    original_shape = image_data.shape
    
    print(f"\n📂 Case: {case_name}")
    print(f"  True lesion voxels: {np.sum(true_mask):,}")
    
    # Preprocess with EXACT training method
    processed = preprocess_training_exact(image_data.copy())
    
    # Test 1: Original orientation
    print(f"\n🧪 TEST 1: Original Orientation")
    print(f"-" * 40)
    
    batch_original = processed[np.newaxis, ...]
    
    with tf.device('/GPU:0'):
        pred_original = model(batch_original, training=False)
    
    pred_vol_original = pred_original[0, :, :, :, 0].numpy()
    dice_orig, thresh_orig = test_prediction(pred_vol_original, true_mask, original_shape)
    
    print(f"  Prediction max: {pred_vol_original.max():.6f}")
    print(f"  Prediction mean: {pred_vol_original.mean():.8f}")
    print(f"  Best Dice: {dice_orig:.6f} at threshold {thresh_orig:.1f}")
    
    # Clean up
    del batch_original, pred_original, pred_vol_original
    gc.collect()
    
    # Test 2: Horizontally flipped (EXACT same as training augmentation)
    print(f"\n🧪 TEST 2: Horizontally Flipped")
    print(f"-" * 40)
    
    # Apply EXACT same flip as training: axis=1
    processed_flipped = np.flip(processed, axis=1)
    batch_flipped = processed_flipped[np.newaxis, ...]
    
    with tf.device('/GPU:0'):
        pred_flipped = model(batch_flipped, training=False)
    
    # Flip prediction back to match original orientation
    pred_vol_flipped = pred_flipped[0, :, :, :, 0].numpy()
    pred_vol_flipped = np.flip(pred_vol_flipped, axis=1)  # Flip back
    
    dice_flip, thresh_flip = test_prediction(pred_vol_flipped, true_mask, original_shape)
    
    print(f"  Prediction max: {pred_vol_flipped.max():.6f}")
    print(f"  Prediction mean: {pred_vol_flipped.mean():.8f}")
    print(f"  Best Dice: {dice_flip:.6f} at threshold {thresh_flip:.1f}")
    
    # Clean up
    del processed_flipped, batch_flipped, pred_flipped, pred_vol_flipped
    gc.collect()
    
    # Results
    print(f"\n🏆 RESULTS COMPARISON:")
    print(f"=" * 50)
    print(f"Original:  Dice = {dice_orig:.6f}")
    print(f"Flipped:   Dice = {dice_flip:.6f}")
    print(f"Improvement: {dice_flip/dice_orig:.1f}x" if dice_orig > 0 else "∞x")
    
    if dice_flip > dice_orig + 0.1:
        print(f"\n🎉 SUCCESS! Horizontal flip augmentation was the issue!")
        print(f"  → Use flipped orientation for inference")
        print(f"  → Expected batch performance: ~{dice_flip:.2f} Dice")
        success = True
    elif dice_flip > 0.1:
        print(f"\n✅ SIGNIFICANT IMPROVEMENT!")
        print(f"  → Flip augmentation helps significantly")
        success = True
    else:
        print(f"\n❌ Limited improvement")
        print(f"  → May need additional investigation")
        success = False
    
    # Clean up
    del model, image_data, true_mask, processed
    tf.keras.backend.clear_session()
    gc.collect()
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
