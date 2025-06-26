#!/usr/bin/env python3
"""
Test the model on a training case to see if it works there
This will tell us if the issue is train/test mismatch vs fundamental model problems
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

def find_training_cases():
    """Find some training cases to test"""
    training_dir = "/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training_Split"
    
    if not os.path.exists(training_dir):
        print(f"❌ Training directory not found: {training_dir}")
        return []
    
    image_dir = os.path.join(training_dir, "Images")
    if not os.path.exists(image_dir):
        print(f"❌ Training images directory not found: {image_dir}")
        return []
    
    # Get first few training cases
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.nii.gz')])[:5]
    case_names = [f.split('_space-MNI152NLin2009aSym_T1w.nii.gz')[0] for f in image_files]
    
    print(f"✅ Found {len(case_names)} training cases to test")
    return case_names

def test_case(model, case_name, data_dir, is_training=True):
    """Test the model on a single case"""
    # File paths
    image_path = f"{data_dir}/Images/{case_name}_space-MNI152NLin2009aSym_T1w.nii.gz"
    mask_path = f"{data_dir}/Masks/{case_name}_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz"
    
    if not os.path.exists(image_path) or not os.path.exists(mask_path):
        print(f"  ❌ Files not found for {case_name}")
        return None
    
    try:
        # Load data
        nii_img = nib.load(image_path)
        image_data = nii_img.get_fdata(dtype=np.float32)
        true_mask = nib.load(mask_path).get_fdata().astype(np.uint8)
        original_shape = image_data.shape
        
        # Preprocess and predict
        processed = preprocess_image(image_data)
        batch = processed[np.newaxis, ...]
        
        with tf.device('/GPU:0'):
            prediction = model(batch, training=False)
        
        pred_volume = prediction[0, :, :, :, 0].numpy()
        
        # Quick analysis
        pred_max = pred_volume.max()
        pred_mean = pred_volume.mean()
        
        # Test a few thresholds with power fix (best from previous test)
        power_pred = np.power(pred_volume, 0.1)
        
        # Resize to original shape
        if power_pred.shape != original_shape:
            factors = [o / p for o, p in zip(original_shape, power_pred.shape)]
            pred_resized = resize_volume(power_pred, original_shape, order=1)
        else:
            pred_resized = power_pred
        
        # Test thresholds
        best_dice = 0.0
        best_threshold = 0.5
        
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
            binary_pred = (pred_resized > threshold).astype(np.uint8)
            intersection = np.sum(binary_pred * true_mask)
            union = np.sum(binary_pred) + np.sum(true_mask)
            dice = (2.0 * intersection) / union if union > 0 else 0.0
            
            if dice > best_dice:
                best_dice = dice
                best_threshold = threshold
        
        # Clean up
        del image_data, true_mask, processed, batch, prediction, pred_volume, power_pred, pred_resized
        gc.collect()
        
        return {
            'case': case_name,
            'is_training': is_training,
            'pred_max': pred_max,
            'pred_mean': pred_mean,
            'dice': best_dice,
            'threshold': best_threshold,
            'original_shape': original_shape
        }
        
    except Exception as e:
        print(f"  ❌ Error testing {case_name}: {e}")
        return None

def main():
    """Main function to test on training vs test cases"""
    print(f"🔍 TRAINING VS TEST CASE ANALYSIS")
    print(f"=" * 80)
    print(f"Purpose: Check if model works on training data")
    print(f"Strategy: Test same model on training cases vs test cases")
    print(f"Expected: Training cases should work much better")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    setup_tensorflow()
    model = load_fixed_model()
    
    # Test directories
    training_dir = "/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training_Split"
    test_dir = "/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Testing_Split"
    
    print(f"\n📂 TESTING TRAINING CASES")
    print(f"=" * 50)
    
    # Find and test training cases
    training_cases = find_training_cases()
    training_results = []
    
    if training_cases:
        for i, case_name in enumerate(training_cases):
            print(f"  Testing training case {i+1}/{len(training_cases)}: {case_name}")
            result = test_case(model, case_name, training_dir, is_training=True)
            if result:
                training_results.append(result)
                print(f"    Dice: {result['dice']:.6f}, Pred_max: {result['pred_max']:.8f}")
            
            # Clear memory periodically
            if (i + 1) % 2 == 0:
                tf.keras.backend.clear_session()
                gc.collect()
    
    print(f"\n📂 TESTING TEST CASES")
    print(f"=" * 50)
    
    # Test a few test cases for comparison
    test_cases = ["sub-r048s014_ses-1", "sub-r040s011_ses-1", "sub-r034s010_ses-1"]
    test_results = []
    
    for i, case_name in enumerate(test_cases):
        print(f"  Testing test case {i+1}/{len(test_cases)}: {case_name}")
        result = test_case(model, case_name, test_dir, is_training=False)
        if result:
            test_results.append(result)
            print(f"    Dice: {result['dice']:.6f}, Pred_max: {result['pred_max']:.8f}")
        
        # Clear memory
        tf.keras.backend.clear_session()
        gc.collect()
    
    # Analysis
    print(f"\n📊 COMPARATIVE ANALYSIS")
    print(f"=" * 50)
    
    if training_results and test_results:
        # Training stats
        training_dices = [r['dice'] for r in training_results]
        training_pred_maxes = [r['pred_max'] for r in training_results]
        
        # Test stats  
        test_dices = [r['dice'] for r in test_results]
        test_pred_maxes = [r['pred_max'] for r in test_results]
        
        print(f"TRAINING CASES ({len(training_results)} cases):")
        print(f"  Mean Dice: {np.mean(training_dices):.6f}")
        print(f"  Max Dice: {np.max(training_dices):.6f}")
        print(f"  Mean pred_max: {np.mean(training_pred_maxes):.8f}")
        
        print(f"\nTEST CASES ({len(test_results)} cases):")
        print(f"  Mean Dice: {np.mean(test_dices):.6f}")
        print(f"  Max Dice: {np.max(test_dices):.6f}")
        print(f"  Mean pred_max: {np.mean(test_pred_maxes):.8f}")
        
        # Diagnosis
        if np.max(training_dices) > 0.2:
            print(f"\n🎉 MODEL WORKS ON TRAINING DATA!")
            print(f"  → Issue is train/test data mismatch")
            print(f"  → Need to investigate data preprocessing differences")
            success = True
        elif np.max(training_dices) > 0.05:
            print(f"\n⚠️ MODEL PARTIALLY WORKS ON TRAINING DATA")
            print(f"  → Model has some capability but is degraded")
            print(f"  → May need model debugging")
            success = False
        else:
            print(f"\n❌ MODEL FAILS ON TRAINING DATA TOO")
            print(f"  → Fundamental model issue")
            print(f"  → Model may be corrupted or incorrectly loaded")
            success = False
    else:
        print(f"❌ Could not load enough cases for comparison")
        success = False
    
    # Clean up
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    
    print(f"\n🎯 RECOMMENDED NEXT STEPS:")
    if success:
        print(f"1. Investigate training vs test preprocessing differences")
        print(f"2. Check coordinate system alignment")
        print(f"3. Verify same data format and orientation")
        print(f"4. Consider retraining with combined train+test data")
    else:
        print(f"1. Check model file integrity")
        print(f"2. Verify model architecture matches training")
        print(f"3. Consider retraining from scratch")
        print(f"4. Debug training pipeline")
    
    print(f"\nEnd time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
