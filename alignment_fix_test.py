#!/usr/bin/env python3
"""
Test alignment fix for misaligned predictions
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
    """Resize volume with specified interpolation order"""
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=order, mode='constant', cval=0)

def preprocess_image_training_style(image_data, target_shape=(192, 224, 176), apply_flip=False):
    """Preprocess exactly like training (with optional flip)"""
    if image_data.shape != target_shape:
        image_data = resize_volume(image_data, target_shape, order=1)
    
    # Normalize like training
    p1, p99 = np.percentile(image_data[image_data > 0], [1, 99])
    image_data = np.clip(image_data, p1, p99)
    image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min() + 1e-8)
    
    # Apply flip augmentation if specified (like training)
    if apply_flip:
        image_data = np.flip(image_data, axis=1)  # Flip along width dimension
    
    return image_data[..., np.newaxis]

def test_alignment_fixes():
    """Test different alignment fixes"""
    print("=" * 70)
    print("TESTING ALIGNMENT FIXES")
    print("=" * 70)
    
    setup_tensorflow()
    
    # Test case
    test_case = "sub-r048s014_ses-1"
    test_dir = "/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Testing_Split"
    image_path = f"{test_dir}/Images/{test_case}_space-MNI152NLin2009aSym_T1w.nii.gz"
    mask_path = f"{test_dir}/Masks/{test_case}_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz"
    
    try:
        # Load model
        print("Loading model...")
        model = load_model_safe()
        
        # Load data
        print("Loading test data...")
        nii_img = nib.load(image_path)
        image_data = nii_img.get_fdata(dtype=np.float32)
        true_mask = nib.load(mask_path).get_fdata().astype(np.uint8)
        original_shape = image_data.shape
        
        print(f"Original image shape: {original_shape}")
        print(f"True lesion voxels: {np.sum(true_mask):,}")
        
        # Test different configurations
        test_configs = [
            {"name": "Original (no flip, order=1)", "flip": False, "mask_order": 1},
            {"name": "With flip (flip, order=1)", "flip": True, "mask_order": 1},
            {"name": "Original (no flip, order=0)", "flip": False, "mask_order": 0},
            {"name": "With flip (flip, order=0)", "flip": True, "mask_order": 0},
        ]
        
        results = []
        
        for config in test_configs:
            print(f"\n{'='*50}")
            print(f"Testing: {config['name']}")
            print(f"{'='*50}")
            
            # Preprocess image
            processed_image = preprocess_image_training_style(
                image_data.copy(), 
                target_shape=(192, 224, 176), 
                apply_flip=config['flip']
            )
            image_batch = processed_image[np.newaxis, ...]
            
            # Run prediction
            with tf.device('/GPU:0'):
                prediction = model(image_batch, training=False)
            
            # Convert prediction (chunked to avoid memory issues)
            pred_chunks = []
            chunk_size = 32
            depth = prediction.shape[3]
            
            for i in range(0, depth, chunk_size):
                end_idx = min(i + chunk_size, depth)
                chunk = prediction[0, :, :, i:end_idx, 0].numpy()
                pred_chunks.append(chunk)
            
            pred_volume = np.concatenate(pred_chunks, axis=2)
            
            # If we applied flip to input, flip prediction back
            if config['flip']:
                pred_volume = np.flip(pred_volume, axis=1)
            
            # Test different thresholds
            best_dice = 0.0
            best_threshold = 0.5
            
            for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
                # Apply threshold
                binary_pred_small = (pred_volume > threshold).astype(np.uint8)
                
                # Resize to original space with specified order
                if binary_pred_small.shape != original_shape:
                    factors = [o / p for o, p in zip(original_shape, binary_pred_small.shape)]
                    binary_pred = resize_volume(binary_pred_small, original_shape, order=config['mask_order'])
                    binary_pred = (binary_pred > 0.5).astype(np.uint8)  # Re-binarize after resize
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
            
            # Calculate center of mass for alignment analysis
            if best_dice > 0:
                binary_pred_best = (pred_volume > best_threshold).astype(np.uint8)
                if binary_pred_best.shape != original_shape:
                    factors = [o / p for o, p in zip(original_shape, binary_pred_best.shape)]
                    binary_pred_best = resize_volume(binary_pred_best, original_shape, order=config['mask_order'])
                    binary_pred_best = (binary_pred_best > 0.5).astype(np.uint8)
                
                # Center of mass analysis
                from scipy.ndimage import center_of_mass
                
                pred_com = center_of_mass(binary_pred_best)
                true_com = center_of_mass(true_mask)
                
                offset = [pred_com[i] - true_com[i] for i in range(3)]
                offset_magnitude = np.sqrt(sum(o**2 for o in offset))
                
                print(f"  Center of mass offset: {offset} (magnitude: {offset_magnitude:.1f} voxels)")
            
            result = {
                'config': config['name'],
                'flip': config['flip'],
                'mask_order': config['mask_order'],
                'best_dice': best_dice,
                'best_threshold': best_threshold
            }
            results.append(result)
            
            print(f"  ✅ Best result: Dice={best_dice:.4f} at threshold={best_threshold:.1f}")
            
            # Clean up
            del processed_image, image_batch, prediction, pred_chunks, pred_volume
            import gc
            gc.collect()
        
        # Summary
        print(f"\n{'='*70}")
        print("ALIGNMENT FIX RESULTS SUMMARY")
        print(f"{'='*70}")
        
        best_result = max(results, key=lambda x: x['best_dice'])
        
        for result in results:
            marker = "🎯" if result == best_result else "  "
            print(f"{marker} {result['config']:30s}: Dice={result['best_dice']:.4f}")
        
        print(f"\n🏆 Best configuration: {best_result['config']}")
        print(f"   Dice Score: {best_result['best_dice']:.4f}")
        print(f"   Uses flip: {best_result['flip']}")
        print(f"   Mask resize order: {best_result['mask_order']}")
        
        if best_result['best_dice'] > 0.3:
            print(f"\n✅ ALIGNMENT ISSUE CONFIRMED AND FIXED!")
            print(f"The model works well when alignment is corrected.")
        else:
            print(f"\n❌ Alignment fixes didn't resolve the issue.")
            print(f"There may be additional problems.")
        
        return results
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = test_alignment_fixes()
    
    if results:
        print(f"\n🎉 Alignment testing completed!")
        print(f"Apply the best configuration to batch processing.")
    else:
        print(f"\n❌ Alignment testing failed.")
