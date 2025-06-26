#!/usr/bin/env python3
"""
Test the newly trained fixed model
"""

import os
import sys
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

def load_fixed_model():
    """Load the newly trained fixed model"""
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
    
    print(f"Loading fixed model from: {model_path}")
    model = load_model(model_path, custom_objects=custom_objects, compile=False)
    print(f"✅ Fixed model loaded: {model.count_params():,} parameters")
    return model

def resize_volume(volume, target_shape, order=1):
    """Resize volume"""
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=order, mode='constant', cval=0)

def preprocess_image(image_data, target_shape=(192, 224, 176)):
    """Preprocess exactly like training"""
    if image_data.shape != target_shape:
        image_data = resize_volume(image_data, target_shape, order=1)
    
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

def test_fixed_model():
    """Test the fixed model on both training and test cases"""
    print("=" * 70)
    print("TESTING FIXED MODEL (WITH VALIDATION SPLIT)")
    print("=" * 70)
    
    setup_tensorflow()
    
    # Load fixed model
    model = load_fixed_model()
    
    # Test cases
    test_cases = [
        {
            "name": "Training case",
            "data_dir": "/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training_Split",
            "case_id": "sub-r001s001_ses-1",
            "expected": "High Dice (0.6-0.8)"
        },
        {
            "name": "Test case (small lesion)",
            "data_dir": "/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Testing_Split", 
            "case_id": "sub-r048s014_ses-1",
            "expected": "Improved Dice (0.3-0.5)"
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n{'='*50}")
        print(f"TESTING: {test_case['name']} - {test_case['case_id']}")
        print(f"Expected: {test_case['expected']}")
        print(f"{'='*50}")
        
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
            
            # Preprocess
            processed = preprocess_image(image_data)
            image_batch = processed[np.newaxis, ...]
            
            # Predict
            print("Running prediction...")
            start_time = time.time()
            
            with tf.device('/GPU:0'):
                prediction = model(image_batch, training=False)
            
            pred_time = time.time() - start_time
            print(f"Prediction completed in {pred_time:.2f}s")
            
            # Convert prediction
            pred_volume = convert_prediction_chunked(prediction)
            
            print(f"Raw prediction range: [{pred_volume.min():.6f}, {pred_volume.max():.6f}]")
            print(f"Raw prediction mean: {pred_volume.mean():.6f}")
            
            # Test multiple thresholds
            best_dice = 0.0
            best_threshold = 0.5
            best_pred_voxels = 0
            
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            
            print(f"Testing thresholds:")
            for threshold in thresholds:
                binary_pred_small = (pred_volume > threshold).astype(np.uint8)
                
                # Resize to original space
                if binary_pred_small.shape != original_shape:
                    factors = [o / p for o, p in zip(original_shape, binary_pred_small.shape)]
                    binary_pred = resize_volume(binary_pred_small, original_shape, order=0)
                    binary_pred = (binary_pred > 0.5).astype(np.uint8)
                else:
                    binary_pred = binary_pred_small
                
                # Calculate Dice
                intersection = np.sum(binary_pred * true_mask)
                union = np.sum(binary_pred) + np.sum(true_mask)
                dice = (2.0 * intersection) / union if union > 0 else 0.0
                
                pred_voxels = np.sum(binary_pred)
                print(f"  Threshold {threshold:.1f}: Dice={dice:.4f}, Predicted={pred_voxels:6,} voxels")
                
                if dice > best_dice:
                    best_dice = dice
                    best_threshold = threshold
                    best_pred_voxels = pred_voxels
            
            print(f"\n🏆 Best result: Dice={best_dice:.4f} at threshold={best_threshold:.1f}")
            print(f"   Predicted voxels: {best_pred_voxels:,}")
            print(f"   True voxels: {np.sum(true_mask):,}")
            
            # Compare with previous model
            improvement_status = ""
            if test_case['name'] == "Training case":
                if best_dice > 0.6:
                    improvement_status = "✅ Good (as expected)"
                else:
                    improvement_status = "⚠️ Lower than expected"
            else:  # Test case
                if best_dice > 0.3:
                    improvement_status = "🎉 MAJOR IMPROVEMENT from 0.0!"
                elif best_dice > 0.1:
                    improvement_status = "✅ Significant improvement from 0.0"
                else:
                    improvement_status = "❌ Still poor performance"
            
            print(f"   Status: {improvement_status}")
            
            result = {
                "case_type": test_case['name'],
                "case_id": case_id,
                "dice": best_dice,
                "threshold": best_threshold,
                "pred_voxels": best_pred_voxels,
                "true_voxels": np.sum(true_mask),
                "status": improvement_status
            }
            results.append(result)
            
            # Cleanup
            del processed, image_batch, prediction, pred_volume
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"❌ Error testing {case_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*70}")
    print("FIXED MODEL TEST SUMMARY")
    print(f"{'='*70}")
    
    if results:
        for result in results:
            print(f"{result['case_type']:20s}: Dice={result['dice']:.4f} - {result['status']}")
        
        # Overall assessment
        test_dice = next((r['dice'] for r in results if 'Test case' in r['case_type']), 0)
        train_dice = next((r['dice'] for r in results if 'Training case' in r['case_type']), 0)
        
        print(f"\nOverall Assessment:")
        if test_dice > 0.3:
            print(f"🎉 SUCCESS: Fixed model shows major improvement!")
            print(f"   Test performance improved from 0.0 to {test_dice:.4f}")
            print(f"   Ready for batch testing on all 55 cases")
        elif test_dice > 0.1:
            print(f"✅ PROGRESS: Fixed model shows improvement")
            print(f"   Test performance improved from 0.0 to {test_dice:.4f}")
            print(f"   Some overfitting remains but much better")
        else:
            print(f"❌ LIMITED SUCCESS: Still having issues")
            print(f"   May need more regularization or different approach")
        
        print(f"\nNext steps:")
        if test_dice > 0.2:
            print(f"  1. Run batch testing on all 55 test cases")
            print(f"  2. Compare with validation performance (0.47)")
            print(f"  3. Analyze results for further improvements")
        else:
            print(f"  1. Consider more aggressive regularization")
            print(f"  2. Try different augmentation strategies")
            print(f"  3. Investigate other architecture improvements")
    
    return results

if __name__ == "__main__":
    results = test_fixed_model()
    
    if results:
        print(f"\n🎯 Fixed model testing completed!")
    else:
        print(f"\n❌ Testing failed")
