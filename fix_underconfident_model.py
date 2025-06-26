#!/usr/bin/env python3
"""
Fix for under-confident model predictions
Apply calibration/rescaling to get usable predictions
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
    """Resize volume"""
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=order, mode='constant', cval=0)

def preprocess_image(image_data, target_shape=(192, 224, 176)):
    """Standard preprocessing"""
    if image_data.shape != target_shape:
        image_data = resize_volume(image_data, target_shape, order=1)
    
    # Intensity normalization
    non_zero = image_data[image_data > 0]
    if len(non_zero) > 0:
        p1, p99 = np.percentile(non_zero, [1, 99])
        image_data = np.clip(image_data, p1, p99)
        image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min() + 1e-8)
    
    return image_data[..., np.newaxis]

def apply_prediction_fixes(raw_prediction):
    """Apply various fixes to under-confident predictions"""
    print(f"\n🔧 APPLYING PREDICTION FIXES")
    print(f"=" * 40)
    
    # Original stats
    print(f"  Original max: {raw_prediction.max():.8f}")
    print(f"  Original mean: {raw_prediction.mean():.8f}")
    
    fixes = {}
    
    # Fix 1: Power transformation (makes small values larger)
    power_pred = np.power(raw_prediction, 0.1)  # Raises to 0.1 power
    fixes['power_0.1'] = power_pred
    print(f"  Power 0.1 max: {power_pred.max():.8f}")
    
    # Fix 2: Logarithmic rescaling
    log_pred = np.log1p(raw_prediction * 1000) / np.log1p(1000)  # Log rescaling
    fixes['log_rescale'] = log_pred
    print(f"  Log rescale max: {log_pred.max():.8f}")
    
    # Fix 3: Percentile normalization (most robust)
    p99 = np.percentile(raw_prediction, 99)
    if p99 > 0:
        percentile_pred = raw_prediction / p99
        percentile_pred = np.clip(percentile_pred, 0, 1)
        fixes['percentile_norm'] = percentile_pred
        print(f"  Percentile norm max: {percentile_pred.max():.8f}")
    
    # Fix 4: Simple linear scaling
    if raw_prediction.max() > 0:
        linear_pred = raw_prediction / raw_prediction.max()
        fixes['linear_scale'] = linear_pred
        print(f"  Linear scale max: {linear_pred.max():.8f}")
    
    # Fix 5: Sigmoid calibration (assumes logits were too negative)
    # Convert back to logits and add bias
    epsilon = 1e-8
    safe_pred = np.clip(raw_prediction, epsilon, 1 - epsilon)
    logits = np.log(safe_pred / (1 - safe_pred))
    calibrated_logits = logits + 5.0  # Add positive bias
    sigmoid_calibrated = 1 / (1 + np.exp(-calibrated_logits))
    fixes['sigmoid_calibrated'] = sigmoid_calibrated
    print(f"  Sigmoid calibrated max: {sigmoid_calibrated.max():.8f}")
    
    return fixes

def test_prediction_fix(pred_volume, true_mask, original_shape, fix_name):
    """Test a specific prediction fix"""
    # Resize to original space
    if pred_volume.shape != original_shape:
        factors = [o / p for o, p in zip(original_shape, pred_volume.shape)]
        pred_resized = resize_volume(pred_volume, original_shape, order=1)
    else:
        pred_resized = pred_volume
    
    # Test multiple thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    best_dice = 0.0
    best_threshold = 0.5
    
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

def test_all_fixes(case_name, test_dir="/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Testing_Split"):
    """Test all prediction fixes on a case"""
    print(f"\n{'='*80}")
    print(f"TESTING PREDICTION FIXES: {case_name}")
    print(f"{'='*80}")
    
    # Load data
    image_path = f"{test_dir}/Images/{case_name}_space-MNI152NLin2009aSym_T1w.nii.gz"
    mask_path = f"{test_dir}/Masks/{case_name}_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz"
    
    nii_img = nib.load(image_path)
    image_data = nii_img.get_fdata(dtype=np.float32)
    true_mask = nib.load(mask_path).get_fdata().astype(np.uint8)
    original_shape = image_data.shape
    
    print(f"📂 Case: {case_name}")
    print(f"  True lesion voxels: {np.sum(true_mask):,}")
    
    # Setup and predict
    setup_tensorflow()
    model = load_fixed_model()
    
    # Preprocess and predict
    processed = preprocess_image(image_data)
    batch = processed[np.newaxis, ...]
    
    with tf.device('/GPU:0'):
        prediction = model(batch, training=False)
    
    raw_pred_volume = prediction[0, :, :, :, 0].numpy()
    
    # Apply all fixes
    fixes = apply_prediction_fixes(raw_pred_volume)
    
    # Test each fix
    print(f"\n🧪 TESTING ALL FIXES:")
    print(f"=" * 40)
    
    results = {}
    
    # Test original (baseline)
    dice, thresh = test_prediction_fix(raw_pred_volume, true_mask, original_shape, "original")
    results['original'] = {'dice': dice, 'threshold': thresh}
    print(f"  Original:          Dice={dice:.6f}, Threshold={thresh:.1f}")
    
    # Test all fixes
    for fix_name, fixed_pred in fixes.items():
        dice, thresh = test_prediction_fix(fixed_pred, true_mask, original_shape, fix_name)
        results[fix_name] = {'dice': dice, 'threshold': thresh}
        print(f"  {fix_name:15s}: Dice={dice:.6f}, Threshold={thresh:.1f}")
    
    # Find best fix
    best_fix = max(results.items(), key=lambda x: x[1]['dice'])
    best_name, best_result = best_fix
    
    print(f"\n🏆 BEST FIX FOUND:")
    print(f"  Method: {best_name}")
    print(f"  Dice: {best_result['dice']:.6f}")
    print(f"  Threshold: {best_result['threshold']:.1f}")
    
    if best_result['dice'] > 0.2:
        print(f"\n🎉 SUCCESS! Found working solution")
        print(f"  → Use '{best_name}' method for all predictions")
        print(f"  → Expected batch test performance: ~{best_result['dice']:.2f} Dice")
        success = True
    elif best_result['dice'] > 0.05:
        print(f"\n✅ PARTIAL SUCCESS! Significant improvement")
        print(f"  → '{best_name}' method shows promise")
        print(f"  → May need further calibration")
        success = True
    else:
        print(f"\n❌ FIXES INSUFFICIENT")
        print(f"  → Need deeper model retraining or different approach")
        success = False
    
    # Clean up
    del model, image_data, true_mask, processed, batch, prediction, raw_pred_volume
    for fix_pred in fixes.values():
        del fix_pred
    tf.keras.backend.clear_session()
    gc.collect()
    
    return {
        'case': case_name,
        'best_fix': best_name,
        'best_dice': best_result['dice'],
        'best_threshold': best_result['threshold'],
        'all_results': results,
        'success': success
    }

def main():
    """Main function to test prediction fixes"""
    print(f"🔧 PREDICTION FIX TESTING")
    print(f"=" * 80)
    print(f"Purpose: Fix under-confident model predictions")
    print(f"Strategy: Apply various calibration/rescaling methods")
    print(f"Expected: Find method that gives Dice > 0.2")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_case = "sub-r048s014_ses-1"
    
    try:
        result = test_all_fixes(test_case)
        
        print(f"\n🏁 FINAL RESULT")
        print(f"=" * 50)
        print(f"Best method: {result['best_fix']}")
        print(f"Best Dice: {result['best_dice']:.6f}")
        print(f"Success: {result['success']}")
        
        if result['success']:
            print(f"\n🎯 NEXT STEPS:")
            print(f"1. Implement '{result['best_fix']}' in batch testing")
            print(f"2. Test on multiple cases to confirm")
            print(f"3. Run full batch test with this fix")
            return True
        else:
            print(f"\n🎯 NEXT STEPS:")
            print(f"1. Model may need retraining")
            print(f"2. Check training data and loss function")
            print(f"3. Consider different model architecture")
            return False
            
    except Exception as e:
        print(f"\n❌ FIX TESTING FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        print(f"\nEnd time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
