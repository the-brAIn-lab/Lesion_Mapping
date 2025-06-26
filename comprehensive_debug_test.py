#!/usr/bin/env python3
"""
Comprehensive debugging test for stroke segmentation model
Tests both orientations and exact preprocessing to identify the core issue
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
from pathlib import Path

# Custom imports
sys.path.append('/mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota')
from models.losses import dice_loss, combined_loss, focal_loss

def setup_tensorflow():
    """Setup TensorFlow for optimal performance"""
    # GPU configuration
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print(f"✅ GPU configured: {gpus[0]}")
        except RuntimeError as e:
            print(f"⚠️ GPU setup error: {e}")
    
    # Mixed precision
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print(f"✅ Mixed precision enabled")

def load_fixed_model():
    """Load the fixed model with proper custom objects"""
    model_path = "/mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota/callbacks/sota_fixed_20250619_063330/best_model.h5"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Define custom loss and metrics exactly as in training
    def combined_loss_fn(y_true, y_pred, smooth=1e-6, focal_gamma=3.0, focal_alpha=0.25):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Dice loss
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        dice_loss = 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
        
        # Focal loss
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
    """Resize volume using scipy zoom - exact training implementation"""
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=order, mode='constant', cval=0)

def preprocess_image_exact(image_data, target_shape=(192, 224, 176)):
    """
    Exact preprocessing pipeline from training
    """
    print(f"  Input shape: {image_data.shape}")
    print(f"  Input range: [{image_data.min():.6f}, {image_data.max():.6f}]")
    print(f"  Input mean: {image_data.mean():.6f}")
    print(f"  Non-zero voxels: {np.count_nonzero(image_data):,}")
    
    # Resize if needed
    if image_data.shape != target_shape:
        print(f"  Resizing from {image_data.shape} to {target_shape}")
        image_data = resize_volume(image_data, target_shape, order=1)
        print(f"  After resize: {image_data.shape}")
    
    # Intensity normalization - exact training method
    print(f"  Before normalization - range: [{image_data.min():.6f}, {image_data.max():.6f}]")
    
    # Get percentiles for clipping (only from non-zero voxels)
    non_zero = image_data[image_data > 0]
    if len(non_zero) > 0:
        p1, p99 = np.percentile(non_zero, [1, 99])
        print(f"  Percentiles: p1={p1:.6f}, p99={p99:.6f}")
        
        # Clip values
        image_data = np.clip(image_data, p1, p99)
        print(f"  After clipping - range: [{image_data.min():.6f}, {image_data.max():.6f}]")
        
        # Min-max normalization
        image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min() + 1e-8)
    else:
        print("  ⚠️ Warning: No non-zero voxels found!")
        image_data = np.zeros_like(image_data)
    
    print(f"  After normalization - range: [{image_data.min():.6f}, {image_data.max():.6f}]")
    print(f"  After normalization - mean: {image_data.mean():.6f}")
    
    # Add channel dimension
    image_data = image_data[..., np.newaxis]
    print(f"  Final shape: {image_data.shape}")
    
    return image_data

def test_orientations(model, image_data, true_mask, case_name):
    """Test both original and flipped orientations"""
    print(f"\n🔄 TESTING BOTH ORIENTATIONS")
    print(f"=" * 50)
    
    results = {}
    
    # Test original orientation
    print(f"\n📍 Testing: Original Orientation")
    print(f"-" * 30)
    
    processed_orig = preprocess_image_exact(image_data.copy())
    batch_orig = processed_orig[np.newaxis, ...]
    
    print(f"  Batch shape: {batch_orig.shape}")
    print(f"  Running prediction...")
    
    start_time = time.time()
    with tf.device('/GPU:0'):
        pred_orig = model(batch_orig, training=False)
    pred_time = time.time() - start_time
    
    # Convert to numpy and test thresholds
    pred_volume_orig = pred_orig[0, :, :, :, 0].numpy()
    dice_orig, threshold_orig = find_best_threshold(pred_volume_orig, true_mask, image_data.shape)
    
    results['original'] = {
        'dice': dice_orig,
        'threshold': threshold_orig,
        'pred_time': pred_time,
        'pred_mean': pred_volume_orig.mean(),
        'pred_std': pred_volume_orig.std(),
        'pred_max': pred_volume_orig.max()
    }
    
    print(f"  ✅ Original: Dice={dice_orig:.4f}, Threshold={threshold_orig:.2f}, Time={pred_time:.1f}s")
    
    # Clean up
    del processed_orig, batch_orig, pred_orig, pred_volume_orig
    gc.collect()
    
    # Test flipped orientation
    print(f"\n📍 Testing: Flipped Orientation")
    print(f"-" * 30)
    
    # Flip along a specific axis (try different axes if needed)
    flipped_data = np.flip(image_data, axis=0)  # Flip along first axis
    processed_flip = preprocess_image_exact(flipped_data.copy())
    batch_flip = processed_flip[np.newaxis, ...]
    
    print(f"  Batch shape: {batch_flip.shape}")
    print(f"  Running prediction...")
    
    start_time = time.time()
    with tf.device('/GPU:0'):
        pred_flip = model(batch_flip, training=False)
    pred_time = time.time() - start_time
    
    # Convert to numpy and flip back
    pred_volume_flip = pred_flip[0, :, :, :, 0].numpy()
    pred_volume_flip = np.flip(pred_volume_flip, axis=0)  # Flip back to match original
    
    dice_flip, threshold_flip = find_best_threshold(pred_volume_flip, true_mask, image_data.shape)
    
    results['flipped'] = {
        'dice': dice_flip,
        'threshold': threshold_flip,
        'pred_time': pred_time,
        'pred_mean': pred_volume_flip.mean(),
        'pred_std': pred_volume_flip.std(),
        'pred_max': pred_volume_flip.max()
    }
    
    print(f"  ✅ Flipped: Dice={dice_flip:.4f}, Threshold={threshold_flip:.2f}, Time={pred_time:.1f}s")
    
    # Clean up
    del flipped_data, processed_flip, batch_flip, pred_flip, pred_volume_flip
    gc.collect()
    
    return results

def find_best_threshold(pred_volume, true_mask, original_shape):
    """Find optimal threshold for best Dice score"""
    # Resize prediction back to original shape if needed
    if pred_volume.shape != original_shape:
        factors = [o / p for o, p in zip(original_shape, pred_volume.shape)]
        pred_resized = resize_volume(pred_volume, original_shape, order=1)
    else:
        pred_resized = pred_volume
    
    best_dice = 0.0
    best_threshold = 0.5
    
    # Test multiple thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    for threshold in thresholds:
        binary_pred = (pred_resized > threshold).astype(np.uint8)
        
        # Calculate Dice
        intersection = np.sum(binary_pred * true_mask)
        union = np.sum(binary_pred) + np.sum(true_mask)
        dice = (2.0 * intersection) / union if union > 0 else 0.0
        
        if dice > best_dice:
            best_dice = dice
            best_threshold = threshold
    
    return best_dice, best_threshold

def analyze_case(case_name, test_dir="/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Testing_Split"):
    """Analyze a single test case comprehensively"""
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE DEBUG ANALYSIS: {case_name}")
    print(f"{'='*70}")
    
    # File paths
    image_path = f"{test_dir}/Images/{case_name}_space-MNI152NLin2009aSym_T1w.nii.gz"
    mask_path = f"{test_dir}/Masks/{case_name}_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz"
    
    # Check file existence
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    
    # Load data
    print(f"📂 Loading data...")
    nii_img = nib.load(image_path)
    image_data = nii_img.get_fdata(dtype=np.float32)
    true_mask = nib.load(mask_path).get_fdata().astype(np.uint8)
    
    print(f"  Image path: {image_path}")
    print(f"  Mask path: {mask_path}")
    print(f"  Original shape: {image_data.shape}")
    print(f"  True lesion voxels: {np.sum(true_mask):,}")
    
    # Setup and load model
    setup_tensorflow()
    model = load_fixed_model()
    
    # Test both orientations
    results = test_orientations(model, image_data, true_mask, case_name)
    
    # Analysis
    print(f"\n🎯 RESULTS ANALYSIS")
    print(f"=" * 40)
    
    orig_dice = results['original']['dice']
    flip_dice = results['flipped']['dice']
    
    print(f"Original orientation: Dice = {orig_dice:.4f}")
    print(f"Flipped orientation:  Dice = {flip_dice:.4f}")
    print(f"Improvement factor:   {flip_dice/orig_dice:.1f}x" if orig_dice > 0 else "∞x")
    
    # Determine the issue
    if flip_dice > orig_dice + 0.2:  # Significant improvement
        print(f"\n🎉 ISSUE IDENTIFIED: ORIENTATION MISMATCH!")
        print(f"  - Model was likely trained with flipped orientation")
        print(f"  - Solution: Use flipped orientation for inference")
        print(f"  - Expected test performance: Dice ~{flip_dice:.2f}")
        solution = "flip_orientation"
    elif max(orig_dice, flip_dice) > 0.3:
        print(f"\n✅ MODEL WORKING: Good performance detected")
        print(f"  - Best orientation: {'Flipped' if flip_dice > orig_dice else 'Original'}")
        print(f"  - Performance looks reasonable")
        solution = "model_working"
    else:
        print(f"\n❌ DEEPER ISSUES: Both orientations perform poorly")
        print(f"  - May need to check preprocessing pipeline")
        print(f"  - Or model architecture/training issues")
        solution = "deeper_investigation"
    
    # Clean up
    del model, image_data, true_mask
    tf.keras.backend.clear_session()
    gc.collect()
    
    return {
        'case': case_name,
        'solution': solution,
        'original_dice': orig_dice,
        'flipped_dice': flip_dice,
        'improvement_factor': flip_dice/orig_dice if orig_dice > 0 else float('inf'),
        'best_orientation': 'flipped' if flip_dice > orig_dice else 'original',
        'best_dice': max(orig_dice, flip_dice)
    }

def main():
    """Main debugging function"""
    print(f"🔍 COMPREHENSIVE STROKE SEGMENTATION DEBUG")
    print(f"=" * 70)
    print(f"Purpose: Identify why model performance is poor (Dice ~0.0)")
    print(f"Strategy: Test both orientations with exact training preprocessing")
    print(f"Expected: One orientation should show much better performance")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test case
    test_case = "sub-r048s014_ses-1"
    
    try:
        result = analyze_case(test_case)
        
        print(f"\n🏁 FINAL CONCLUSION")
        print(f"=" * 50)
        print(f"Test case: {result['case']}")
        print(f"Solution: {result['solution']}")
        print(f"Best performance: {result['best_dice']:.4f} Dice")
        print(f"Best orientation: {result['best_orientation']}")
        
        if result['solution'] == "flip_orientation":
            print(f"\n🎯 NEXT STEPS:")
            print(f"1. Modify inference pipeline to use flipped orientation")
            print(f"2. Test on multiple cases to confirm")
            print(f"3. Run batch testing with corrected orientation")
            return True
        elif result['solution'] == "model_working":
            print(f"\n🎯 NEXT STEPS:")
            print(f"1. Use best orientation for batch testing")
            print(f"2. Should see good overall performance")
            return True
        else:
            print(f"\n🎯 NEXT STEPS:")
            print(f"1. Investigate preprocessing pipeline differences")
            print(f"2. Check training data orientation")
            print(f"3. Verify model architecture")
            return False
            
    except Exception as e:
        print(f"\n❌ DEBUG FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        print(f"\nEnd time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
