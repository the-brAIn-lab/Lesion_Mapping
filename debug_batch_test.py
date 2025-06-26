#!/usr/bin/env python3
"""
Debug batch test - test first 5 cases to identify issues
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
import traceback

# Custom imports
sys.path.append('/mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota')

def setup_tensorflow():
    """Setup TensorFlow"""
    print("Setting up TensorFlow...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print(f"✅ TensorFlow configured")

def load_model_with_fallback():
    """Load model with fallback custom objects"""
    model_path = "/mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota/callbacks/sota_20250616_190015/best_model.h5"
    
    print(f"Loading model from: {model_path}")
    
    try:
        # Try importing custom losses first
        from models.losses import dice_loss, combined_loss, focal_loss
        print("✅ Custom losses imported successfully")
        
        # Define custom objects inline as backup
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
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        traceback.print_exc()
        return None

def resize_volume(volume, target_shape):
    """Resize volume"""
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=1)

def preprocess_image(image_data, target_shape=(192, 224, 176)):
    """Preprocess image"""
    if image_data.shape != target_shape:
        image_data = resize_volume(image_data, target_shape)
    
    p1, p99 = np.percentile(image_data[image_data > 0], [1, 99])
    image_data = np.clip(image_data, p1, p99)
    image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min() + 1e-8)
    
    return image_data[..., np.newaxis]

def test_single_case_debug(model, case_name, test_dir):
    """Test single case with detailed debugging"""
    print(f"\n--- Testing {case_name} ---")
    
    # Paths
    image_path = f"{test_dir}/Images/{case_name}_space-MNI152NLin2009aSym_T1w.nii.gz"
    mask_path = f"{test_dir}/Masks/{case_name}_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz"
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None
    
    if not os.path.exists(mask_path):
        print(f"❌ Mask not found: {mask_path}")
        return None
    
    try:
        start_time = time.time()
        
        # Load and preprocess
        nii_img = nib.load(image_path)
        image_data = nii_img.get_fdata(dtype=np.float32)
        original_shape = image_data.shape
        
        processed = preprocess_image(image_data, (192, 224, 176))
        image_batch = processed[np.newaxis, ...]
        
        print(f"  Loaded: {original_shape} -> {image_batch.shape}")
        
        # Clean up
        del image_data, processed
        gc.collect()
        
        # Predict
        with tf.device('/GPU:0'):
            prediction = model(image_batch, training=False)
        
        print(f"  Prediction: {prediction.shape}, range: [{prediction.numpy().min():.4f}, {prediction.numpy().max():.4f}]")
        
        # Clean up
        del image_batch
        gc.collect()
        
        # Convert and process with different thresholds
        pred_np = prediction.numpy()
        pred_volume = pred_np[0, ..., 0]
        
        del prediction, pred_np
        gc.collect()
        
        # Load ground truth
        true_mask = nib.load(mask_path).get_fdata().astype(np.uint8)
        true_lesion_voxels = np.sum(true_mask)
        
        print(f"  True lesion voxels: {true_lesion_voxels:,}")
        
        # Test multiple thresholds
        best_dice = 0.0
        best_threshold = 0.5
        
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
            # Apply threshold
            binary_pred_small = (pred_volume > threshold).astype(np.uint8)
            
            # Resize to original space
            if binary_pred_small.shape != original_shape:
                factors = [o / p for o, p in zip(original_shape, binary_pred_small.shape)]
                binary_pred = zoom(binary_pred_small, factors, order=0)
            else:
                binary_pred = binary_pred_small
            
            # Calculate Dice
            intersection = np.sum(binary_pred * true_mask)
            union = np.sum(binary_pred) + np.sum(true_mask)
            dice = (2.0 * intersection) / union if union > 0 else 0.0
            
            pred_voxels = np.sum(binary_pred)
            print(f"    Threshold {threshold:.1f}: Dice={dice:.4f}, Predicted={pred_voxels:,}")
            
            if dice > best_dice:
                best_dice = dice
                best_threshold = threshold
            
            del binary_pred_small, binary_pred
            gc.collect()
        
        processing_time = time.time() - start_time
        
        print(f"  ✅ Best: Threshold={best_threshold:.1f}, Dice={best_dice:.4f}, Time={processing_time:.1f}s")
        
        # Clean up
        del pred_volume, true_mask
        gc.collect()
        
        return {
            'case': case_name,
            'best_dice': best_dice,
            'best_threshold': best_threshold,
            'true_voxels': true_lesion_voxels,
            'processing_time': processing_time
        }
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        traceback.print_exc()
        return None

def main():
    """Debug main function"""
    print("=" * 60)
    print("DEBUG BATCH TEST - FIRST 5 CASES")
    print("=" * 60)
    
    try:
        # Setup
        setup_tensorflow()
        
        # Load model
        model = load_model_with_fallback()
        if model is None:
            print("❌ Cannot proceed without model")
            return
        
        # Get first 5 test cases
        test_dir = "/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Testing_Split"
        image_dir = os.path.join(test_dir, "Images")
        
        if not os.path.exists(image_dir):
            print(f"❌ Test directory not found: {image_dir}")
            return
        
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.nii.gz')])[:5]
        case_names = [f.split('_space-MNI152NLin2009aSym_T1w.nii.gz')[0] for f in image_files]
        
        print(f"Testing first {len(case_names)} cases:")
        for case in case_names:
            print(f"  - {case}")
        
        # Process cases
        results = []
        for i, case_name in enumerate(case_names):
            print(f"\nCase {i+1}/{len(case_names)}: {case_name}")
            
            result = test_single_case_debug(model, case_name, test_dir)
            if result:
                results.append(result)
            
            # Clear session every 2 cases
            if (i + 1) % 2 == 0:
                tf.keras.backend.clear_session()
                gc.collect()
                print(f"  Session cleared after {i+1} cases")
        
        # Summary
        print(f"\n{'='*60}")
        print("DEBUG RESULTS SUMMARY")
        print(f"{'='*60}")
        
        if results:
            for result in results:
                print(f"{result['case']}: Dice={result['best_dice']:.4f} (threshold={result['best_threshold']:.1f})")
            
            mean_dice = np.mean([r['best_dice'] for r in results])
            print(f"\nMean Dice Score: {mean_dice:.4f}")
            print(f"Cases processed: {len(results)}/{len(case_names)}")
        else:
            print("❌ No cases processed successfully")
        
        # Cleanup
        del model
        tf.keras.backend.clear_session()
        gc.collect()
        
        print(f"\n✅ Debug completed")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
