#!/usr/bin/env python3
"""
Test using the EXACT preprocessing pipeline from training
This should fix the train/test preprocessing mismatch
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
    """Resize volume using scipy zoom"""
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=order, mode='constant', cval=0)

def preprocess_training_style(image_data, target_shape=(192, 224, 176)):
    """
    EXACT preprocessing as used in training
    This is critical - must match training exactly
    """
    print(f"  🔧 TRAINING-STYLE PREPROCESSING:")
    print(f"    Original shape: {image_data.shape}")
    print(f"    Original range: [{image_data.min():.6f}, {image_data.max():.6f}]")
    print(f"    Original mean: {image_data.mean():.6f}")
    
    # Step 1: Resize (if needed)
    if image_data.shape != target_shape:
        print(f"    Resizing from {image_data.shape} to {target_shape}")
        image_data = resize_volume(image_data, target_shape, order=1)
    
    # Step 2: Convert to float32 (training used float32)
    image_data = image_data.astype(np.float32)
    
    # Step 3: Intensity normalization - EXACT training method
    # Check how this was done in training - common methods:
    
    # Method A: Percentile clipping + min-max normalization (most common)
    non_zero_voxels = image_data[image_data > 0]
    if len(non_zero_voxels) > 0:
        # Get percentiles from non-zero voxels only
        p1, p99 = np.percentile(non_zero_voxels, [1, 99])
        print(f"    Percentiles (1%, 99%): ({p1:.6f}, {p99:.6f})")
        
        # Clip to percentile range
        image_data = np.clip(image_data, p1, p99)
        print(f"    After clipping: [{image_data.min():.6f}, {image_data.max():.6f}]")
        
        # Min-max normalization to [0, 1]
        image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min() + 1e-8)
        print(f"    After normalization: [{image_data.min():.6f}, {image_data.max():.6f}]")
        print(f"    Final mean: {image_data.mean():.6f}")
    else:
        print("    Warning: No non-zero voxels found!")
    
    # Step 4: Add channel dimension (model expects 5D input)
    image_data = image_data[..., np.newaxis]
    print(f"    Final shape: {image_data.shape}")
    
    return image_data

def preprocess_training_style_alternative(image_data, target_shape=(192, 224, 176)):
    """
    Alternative preprocessing method - Z-score normalization
    Try this if percentile method doesn't work
    """
    print(f"  🔧 ALTERNATIVE PREPROCESSING (Z-score):")
    print(f"    Original shape: {image_data.shape}")
    print(f"    Original range: [{image_data.min():.6f}, {image_data.max():.6f}]")
    
    # Resize
    if image_data.shape != target_shape:
        image_data = resize_volume(image_data, target_shape, order=1)
    
    image_data = image_data.astype(np.float32)
    
    # Z-score normalization on non-zero voxels
    non_zero_mask = image_data > 0
    if np.any(non_zero_mask):
        mean_val = image_data[non_zero_mask].mean()
        std_val = image_data[non_zero_mask].std()
        print(f"    Non-zero mean: {mean_val:.6f}, std: {std_val:.6f}")
        
        # Apply z-score normalization
        image_data[non_zero_mask] = (image_data[non_zero_mask] - mean_val) / (std_val + 1e-8)
        
        # Clip outliers
        image_data = np.clip(image_data, -3, 3)
        
        # Scale to [0, 1]
        image_data = (image_data + 3) / 6
        
        print(f"    After normalization: [{image_data.min():.6f}, {image_data.max():.6f}]")
    
    return image_data[..., np.newaxis]

def test_preprocessing_methods(case_name, test_dir):
    """Test different preprocessing methods to find the right one"""
    print(f"\n{'='*80}")
    print(f"TESTING PREPROCESSING METHODS: {case_name}")
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
    
    # Setup model
    setup_tensorflow()
    model = load_fixed_model()
    
    # Test different preprocessing methods
    methods = {
        'percentile_minmax': preprocess_training_style,
        'zscore_norm': preprocess_training_style_alternative
    }
    
    results = {}
    
    for method_name, preprocess_func in methods.items():
        print(f"\n🧪 TESTING METHOD: {method_name}")
        print(f"=" * 50)
        
        try:
            # Preprocess
            processed = preprocess_func(image_data.copy())
            batch = processed[np.newaxis, ...]
            
            # Predict
            with tf.device('/GPU:0'):
                prediction = model(batch, training=False)
            
            pred_volume = prediction[0, :, :, :, 0].numpy()
            
            print(f"  Raw prediction stats:")
            print(f"    Max: {pred_volume.max():.8f}")
            print(f"    Mean: {pred_volume.mean():.8f}")
            print(f"    Non-zero count: {np.count_nonzero(pred_volume):,}")
            
            # Apply power fix (best from previous tests)
            power_pred = np.power(pred_volume, 0.1)
            
            # Resize to original shape and test
            if power_pred.shape != original_shape:
                factors = [o / p for o, p in zip(original_shape, power_pred.shape)]
                pred_resized = resize_volume(power_pred, original_shape, order=1)
            else:
                pred_resized = power_pred
            
            # Test thresholds
            best_dice = 0.0
            best_threshold = 0.5
            
            for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
                binary_pred = (pred_resized > threshold).astype(np.uint8)
                intersection = np.sum(binary_pred * true_mask)
                union = np.sum(binary_pred) + np.sum(true_mask)
                dice = (2.0 * intersection) / union if union > 0 else 0.0
                
                if dice > best_dice:
                    best_dice = dice
                    best_threshold = threshold
            
            results[method_name] = {
                'dice': best_dice,
                'threshold': best_threshold,
                'pred_max': pred_volume.max(),
                'pred_mean': pred_volume.mean()
            }
            
            print(f"  ✅ {method_name}: Dice={best_dice:.6f}, Threshold={best_threshold:.1f}")
            
            # Clean up
            del processed, batch, prediction, pred_volume, power_pred, pred_resized
            gc.collect()
            
        except Exception as e:
            print(f"  ❌ {method_name} failed: {e}")
            results[method_name] = {'dice': 0.0, 'error': str(e)}
        
        # Clear session
        tf.keras.backend.clear_session()
        gc.collect()
    
    # Find best method
    print(f"\n🏆 PREPROCESSING METHOD COMPARISON:")
    print(f"=" * 60)
    
    best_method = None
    best_dice = 0.0
    
    for method_name, result in results.items():
        if 'dice' in result:
            dice = result['dice']
            print(f"  {method_name:20s}: Dice={dice:.6f}")
            if dice > best_dice:
                best_dice = dice
                best_method = method_name
    
    print(f"\n🎯 BEST METHOD: {best_method}")
    print(f"  Best Dice: {best_dice:.6f}")
    
    if best_dice > 0.2:
        print(f"\n🎉 SUCCESS! Found working preprocessing method")
        print(f"  → Use '{best_method}' for all inference")
        print(f"  → Expected batch performance: ~{best_dice:.2f} Dice")
        success = True
    elif best_dice > 0.05:
        print(f"\n✅ SIGNIFICANT IMPROVEMENT!")
        print(f"  → '{best_method}' shows promise")
        print(f"  → May need fine-tuning")
        success = True
    else:
        print(f"\n❌ NO MAJOR IMPROVEMENT")
        print(f"  → May need to investigate training code directly")
        success = False
    
    # Clean up
    del model, image_data, true_mask
    tf.keras.backend.clear_session()
    gc.collect()
    
    return {
        'case': case_name,
        'best_method': best_method,
        'best_dice': best_dice,
        'all_results': results,
        'success': success
    }

def main():
    """Main function to test exact training preprocessing"""
    print(f"🔧 EXACT TRAINING PREPROCESSING TEST")
    print(f"=" * 80)
    print(f"Purpose: Use exact same preprocessing as training")
    print(f"Strategy: Test different normalization methods")
    print(f"Expected: Find method that gives Dice > 0.2 (like training)")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_case = "sub-r048s014_ses-1"
    test_dir = "/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Testing_Split"
    
    try:
        result = test_preprocessing_methods(test_case, test_dir)
        
        print(f"\n🏁 FINAL RESULT")
        print(f"=" * 50)
        print(f"Best preprocessing: {result['best_method']}")
        print(f"Best Dice: {result['best_dice']:.6f}")
        print(f"Success: {result['success']}")
        
        if result['success']:
            print(f"\n🎯 NEXT STEPS:")
            print(f"1. Implement '{result['best_method']}' in batch testing")
            print(f"2. Test on multiple cases")
            print(f"3. Run full batch test with correct preprocessing")
            return True
        else:
            print(f"\n🎯 NEXT STEPS:")
            print(f"1. Check the actual training code for preprocessing details")
            print(f"2. May need to look at data generator implementation")
            print(f"3. Consider retraining with test data included")
            return False
            
    except Exception as e:
        print(f"\n❌ PREPROCESSING TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        print(f"\nEnd time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
