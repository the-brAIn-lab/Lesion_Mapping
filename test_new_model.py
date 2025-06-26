#!/usr/bin/env python3
"""
Test the NEW retrained model on both small and large lesions
Compare with the OLD model to verify size bias is fixed
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

def load_models():
    """Load both OLD and NEW models for comparison"""
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
    
    # Load OLD model
    old_model_path = "callbacks/sota_fixed_20250619_063330/best_model.h5"
    old_model = load_model(old_model_path, custom_objects=custom_objects, compile=False)
    print(f"✅ OLD model loaded: {old_model.count_params():,} parameters")
    
    # Load NEW model
    new_model_path = "callbacks/full_retrain_20250622_074312/best_model.h5"
    new_model = load_model(new_model_path, custom_objects=custom_objects, compile=False)
    print(f"✅ NEW model loaded: {new_model.count_params():,} parameters")
    
    return old_model, new_model

def resize_volume(volume, target_shape, order=1):
    """Resize volume"""
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=order, mode='constant', cval=0)

def preprocess_image(image_data, target_shape=(192, 224, 176)):
    """Exact preprocessing from training"""
    if image_data.shape != target_shape:
        image_data = resize_volume(image_data, target_shape, order=1)
    
    # Intensity normalization
    p1, p99 = np.percentile(image_data[image_data > 0], [1, 99])
    image_data = np.clip(image_data, p1, p99)
    image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min() + 1e-8)
    
    return image_data[..., np.newaxis]

def test_model_on_case(model, case_name, test_dir, model_name):
    """Test a model on a specific case"""
    print(f"\n🧪 TESTING {model_name} on {case_name}")
    print(f"-" * 60)
    
    # Load data
    image_path = f"{test_dir}/Images/{case_name}_space-MNI152NLin2009aSym_T1w.nii.gz"
    mask_path = f"{test_dir}/Masks/{case_name}_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz"
    
    image_data = nib.load(image_path).get_fdata(dtype=np.float32)
    true_mask = nib.load(mask_path).get_fdata().astype(np.uint8)
    lesion_size = np.sum(true_mask)
    
    print(f"  Case: {case_name}")
    print(f"  Lesion size: {lesion_size:,} voxels")
    
    # Preprocess
    processed = preprocess_image(image_data.copy())
    batch = processed[np.newaxis, ...]
    
    # Predict
    with tf.device('/GPU:0'):
        prediction = model(batch, training=False)
    
    pred_volume = prediction[0, :, :, :, 0].numpy()
    
    print(f"  Prediction max: {pred_volume.max():.6f}")
    print(f"  Prediction mean: {pred_volume.mean():.8f}")
    
    # Test thresholds for Dice
    best_dice = 0.0
    best_threshold = 0.5
    
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        # Resize prediction to original shape if needed
        if pred_volume.shape != image_data.shape:
            factors = [o / p for o, p in zip(image_data.shape, pred_volume.shape)]
            pred_resized = resize_volume(pred_volume, image_data.shape, order=1)
        else:
            pred_resized = pred_volume
        
        binary_pred = (pred_resized > threshold).astype(np.uint8)
        intersection = np.sum(binary_pred * true_mask)
        union = np.sum(binary_pred) + np.sum(true_mask)
        dice = (2.0 * intersection) / union if union > 0 else 0.0
        
        if dice > best_dice:
            best_dice = dice
            best_threshold = threshold
    
    print(f"  Best Dice: {best_dice:.6f} at threshold {best_threshold:.1f}")
    
    # Clean up
    del image_data, true_mask, processed, batch, prediction, pred_volume
    gc.collect()
    
    return {
        'case': case_name,
        'lesion_size': lesion_size,
        'pred_max': pred_volume.max() if 'pred_volume' in locals() else 0,
        'dice': best_dice,
        'threshold': best_threshold
    }

def main():
    """Main comparison function"""
    print(f"🔍 OLD vs NEW MODEL COMPARISON")
    print(f"=" * 80)
    print(f"Purpose: Verify NEW model fixes size bias")
    print(f"Strategy: Test both models on small and large lesions")
    print(f"Expected: NEW model works on both, OLD only on large")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    setup_tensorflow()
    old_model, new_model = load_models()
    
    test_dir = "/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Testing_Split"
    
    # Test cases: small and large lesions
    test_cases = [
        ("sub-r048s014_ses-1", "SMALL lesion (1,112 voxels)"),
        ("sub-r048s016_ses-1", "LARGE lesion (280,190 voxels)")
    ]
    
    results = {}
    
    for case_name, description in test_cases:
        print(f"\n{'='*80}")
        print(f"TESTING CASE: {description}")
        print(f"{'='*80}")
        
        # Test OLD model
        old_result = test_model_on_case(old_model, case_name, test_dir, "OLD MODEL")
        
        # Clear memory
        tf.keras.backend.clear_session()
        gc.collect()
        
        # Test NEW model
        new_result = test_model_on_case(new_model, case_name, test_dir, "NEW MODEL")
        
        # Clear memory
        tf.keras.backend.clear_session()
        gc.collect()
        
        # Store results
        results[case_name] = {
            'description': description,
            'old': old_result,
            'new': new_result
        }
        
        # Quick comparison
        print(f"\n📊 QUICK COMPARISON:")
        print(f"  OLD model: Max={old_result['pred_max']:.6f}, Dice={old_result['dice']:.6f}")
        print(f"  NEW model: Max={new_result['pred_max']:.6f}, Dice={new_result['dice']:.6f}")
        
        improvement = new_result['dice'] / old_result['dice'] if old_result['dice'] > 0 else float('inf')
        if improvement > 2:
            print(f"  🎉 NEW model: {improvement:.1f}x better!")
        elif new_result['dice'] > old_result['dice']:
            print(f"  ✅ NEW model: Better performance")
        else:
            print(f"  ⚠️ Similar performance")
    
    # Final analysis
    print(f"\n{'='*80}")
    print(f"FINAL ANALYSIS")
    print(f"{'='*80}")
    
    small_case = results["sub-r048s014_ses-1"]
    large_case = results["sub-r048s016_ses-1"]
    
    print(f"SMALL LESION PERFORMANCE:")
    print(f"  OLD model: Dice = {small_case['old']['dice']:.6f}")
    print(f"  NEW model: Dice = {small_case['new']['dice']:.6f}")
    
    print(f"\nLARGE LESION PERFORMANCE:")
    print(f"  OLD model: Dice = {large_case['old']['dice']:.6f}")
    print(f"  NEW model: Dice = {large_case['new']['dice']:.6f}")
    
    # Determine if size bias is fixed
    small_improvement = small_case['new']['dice'] > small_case['old']['dice'] + 0.1
    large_maintained = large_case['new']['dice'] > 0.3  # Should still work on large lesions
    
    if small_improvement and large_maintained:
        print(f"\n🎉 SUCCESS! SIZE BIAS FIXED!")
        print(f"  ✅ NEW model works on small lesions")
        print(f"  ✅ NEW model maintains performance on large lesions")
        print(f"  → Ready for batch testing on all cases")
        success = True
    elif small_improvement:
        print(f"\n✅ PARTIAL SUCCESS!")
        print(f"  ✅ NEW model improved on small lesions")
        print(f"  ⚠️ May need to check large lesion performance")
        success = True
    else:
        print(f"\n❌ LIMITED IMPROVEMENT")
        print(f"  → May need further investigation or different training strategy")
        success = False
    
    # Clean up
    del old_model, new_model
    tf.keras.backend.clear_session()
    gc.collect()
    
    print(f"\nEnd time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
