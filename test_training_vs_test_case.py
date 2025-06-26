#!/usr/bin/env python3
"""
Test the model on a TRAINING case to separate overfitting from alignment issues
If model gets good Dice on training data = overfitting confirmed
If model gets bad Dice on training data = alignment/model issue
"""

import os
import sys
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.ndimage import zoom, center_of_mass
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

def load_model_safe():
    """Load model"""
    model_path = "/mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota/callbacks/sota_20250616_190015/best_model.h5"
    
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
    return model

def resize_volume(volume, target_shape, order=1):
    """Resize volume"""
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=order, mode='constant', cval=0)

def preprocess_exactly_like_training(image_data, target_shape=(192, 224, 176)):
    """Preprocess EXACTLY like training data generator"""
    if image_data.shape != target_shape:
        image_data = resize_volume(image_data, target_shape, order=1)
    
    # Normalize exactly like training
    p1, p99 = np.percentile(image_data[image_data > 0], [1, 99])
    image_data = np.clip(image_data, p1, p99)
    image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min() + 1e-8)
    
    return image_data[..., np.newaxis]

def convert_prediction_chunked(prediction):
    """Convert prediction tensor to numpy in chunks"""
    batch_size, height, width, depth, channels = prediction.shape
    chunk_size = 32
    num_chunks = (depth + chunk_size - 1) // chunk_size
    
    result = np.zeros((height, width, depth), dtype=np.float32)
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, depth)
        chunk_tensor = prediction[0, :, :, start_idx:end_idx, 0]
        result[:, :, start_idx:end_idx] = chunk_tensor.numpy()
        del chunk_tensor
    
    return result

def test_on_training_and_test_cases():
    """Test model on both training and test cases to diagnose the issue"""
    print("=" * 80)
    print("TESTING MODEL ON TRAINING VS TEST CASES")
    print("Hypothesis: If good on training but bad on test → overfitting")
    print("If bad on both → alignment/model architecture issue")
    print("=" * 80)
    
    setup_tensorflow()
    
    # Load model
    print("Loading model...")
    model = load_model_safe()
    
    # Test cases to compare
    test_cases = [
        {
            "name": "Training case",
            "data_dir": "/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training_Split",
            "case_id": "sub-r001s001_ses-1"  # First training case
        },
        {
            "name": "Test case", 
            "data_dir": "/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Testing_Split",
            "case_id": "sub-r048s014_ses-1"  # Our problematic test case
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n{'='*60}")
        print(f"TESTING: {test_case['name']} - {test_case['case_id']}")
        print(f"{'='*60}")
        
        # Paths
        data_dir = test_case['data_dir']
        case_id = test_case['case_id']
        image_path = f"{data_dir}/Images/{case_id}_space-MNI152NLin2009aSym_T1w.nii.gz"
        mask_path = f"{data_dir}/Masks/{case_id}_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz"
        
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"❌ Files not found for {case_id}")
            continue
        
        try:
            # Load data
            nii_img = nib.load(image_path)
            image_data = nii_img.get_fdata(dtype=np.float32)
            true_mask = nib.load(mask_path).get_fdata().astype(np.uint8)
            original_shape = image_data.shape
            
            print(f"Original shape: {original_shape}")
            print(f"True lesion voxels: {np.sum(true_mask):,}")
            
            # Test different configurations to address BOTH issues
            configs = [
                {"flip": False, "mask_order": 1, "name": "Standard"},
                {"flip": True, "mask_order": 1, "name": "With flip"},
                {"flip": False, "mask_order": 0, "name": "Order=0 resize"},
                {"flip": True, "mask_order": 0, "name": "Flip + Order=0"},
            ]
            
            best_result = {"dice": 0.0, "config": "none"}
            
            for config in configs:
                print(f"\n--- Testing {config['name']} ---")
                
                # Preprocess image
                processed = preprocess_exactly_like_training(image_data.copy())
                
                # Apply flip if needed (like training augmentation)
                if config['flip']:
                    processed = np.flip(processed, axis=1)
                
                image_batch = processed[np.newaxis, ...]
                
                # Predict
                with tf.device('/GPU:0'):
                    prediction = model(image_batch, training=False)
                
                # Convert prediction
                pred_volume = convert_prediction_chunked(prediction)
                
                # Flip prediction back if we flipped input
                if config['flip']:
                    pred_volume = np.flip(pred_volume, axis=1)
                
                # Test thresholds
                best_dice_for_config = 0.0
                best_threshold = 0.5
                
                # Test lower thresholds since we know model outputs low probabilities
                thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
                
                for threshold in thresholds:
                    binary_pred_small = (pred_volume > threshold).astype(np.uint8)
                    
                    # Resize to original space
                    if binary_pred_small.shape != original_shape:
                        factors = [o / p for o, p in zip(original_shape, binary_pred_small.shape)]
                        binary_pred = resize_volume(binary_pred_small, original_shape, order=config['mask_order'])
                        binary_pred = (binary_pred > 0.5).astype(np.uint8)
                    else:
                        binary_pred = binary_pred_small
                    
                    # Calculate Dice
                    intersection = np.sum(binary_pred * true_mask)
                    union = np.sum(binary_pred) + np.sum(true_mask)
                    dice = (2.0 * intersection) / union if union > 0 else 0.0
                    
                    if dice > best_dice_for_config:
                        best_dice_for_config = dice
                        best_threshold = threshold
                
                print(f"   Best Dice: {best_dice_for_config:.4f} (threshold: {best_threshold:.3f})")
                
                if best_dice_for_config > best_result["dice"]:
                    best_result = {
                        "dice": best_dice_for_config,
                        "config": config['name'],
                        "threshold": best_threshold
                    }
                
                # Calculate center-of-mass offset for alignment analysis
                if best_dice_for_config > 0.01:  # Only if we found something
                    binary_pred_best = (pred_volume > best_threshold).astype(np.uint8)
                    if binary_pred_best.shape != original_shape:
                        factors = [o / p for o, p in zip(original_shape, binary_pred_best.shape)]
                        binary_pred_best = resize_volume(binary_pred_best, original_shape, order=config['mask_order'])
                        binary_pred_best = (binary_pred_best > 0.5).astype(np.uint8)
                    
                    pred_com = center_of_mass(binary_pred_best)
                    true_com = center_of_mass(true_mask)
                    offset = [pred_com[i] - true_com[i] for i in range(3)]
                    offset_magnitude = np.sqrt(sum(o**2 for o in offset))
                    
                    print(f"   Center-of-mass offset: [{offset[0]:.1f}, {offset[1]:.1f}, {offset[2]:.1f}] (magnitude: {offset_magnitude:.1f})")
                
                # Cleanup
                del processed, image_batch, prediction, pred_volume
                import gc
                gc.collect()
            
            print(f"\n🏆 Best result for {test_case['name']}: {best_result['dice']:.4f} ({best_result['config']})")
            
            result = {
                "case_type": test_case['name'],
                "case_id": case_id,
                "best_dice": best_result['dice'],
                "best_config": best_result['config'],
                "true_voxels": np.sum(true_mask)
            }
            results.append(result)
            
        except Exception as e:
            print(f"❌ Error testing {case_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Analysis
    print(f"\n{'='*80}")
    print("DIAGNOSIS SUMMARY")
    print(f"{'='*80}")
    
    if len(results) >= 2:
        training_dice = next((r['best_dice'] for r in results if r['case_type'] == 'Training case'), 0)
        test_dice = next((r['best_dice'] for r in results if r['case_type'] == 'Test case'), 0)
        
        print(f"Training case performance: {training_dice:.4f}")
        print(f"Test case performance: {test_dice:.4f}")
        print(f"Performance gap: {training_dice - test_dice:.4f}")
        
        if training_dice > 0.3 and test_dice < 0.1:
            print(f"\n🎯 DIAGNOSIS: OVERFITTING CONFIRMED")
            print(f"   - Model performs well on training data")
            print(f"   - Model fails on test data")
            print(f"   - Solution: Retrain with validation split")
        elif training_dice < 0.1 and test_dice < 0.1:
            print(f"\n🧠 DIAGNOSIS: MODEL/ALIGNMENT ISSUE")
            print(f"   - Model fails on both training and test data")
            print(f"   - Possible causes: Architecture too small, alignment issues, broken training")
            print(f"   - Solution: Check model architecture, increase BASE_FILTERS")
        elif training_dice > 0.3 and test_dice > 0.3:
            print(f"\n✅ DIAGNOSIS: ALIGNMENT ISSUE RESOLVED")
            print(f"   - Model works on both training and test data")
            print(f"   - Use the optimal configuration for batch processing")
        else:
            print(f"\n❓ DIAGNOSIS: UNCLEAR")
            print(f"   - Mixed results - need more investigation")
    
    return results

if __name__ == "__main__":
    results = test_on_training_and_test_cases()
    
    if results:
        print(f"\n🎉 Testing completed on {len(results)} cases")
    else:
        print(f"\n❌ Testing failed")
