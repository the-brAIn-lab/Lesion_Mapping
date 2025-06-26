#!/usr/bin/env python3
"""
GPU-compatible test script matching training configuration
"""

import os
# Use same GPU allocator as training
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import sys
import gc
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.ndimage import zoom
import time
from pathlib import Path

# Custom imports - same as training script
sys.path.append('/mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota')
from models.losses import dice_loss, combined_loss, focal_loss

def configure_gpu_like_training():
    """Configure GPU exactly like training script"""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # Use same single GPU setup as training
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print(f"✅ GPU configured like training: {gpus[0].name}")
            return True
        else:
            print("❌ No GPUs detected")
            return False
    except Exception as e:
        print(f"❌ GPU configuration failed: {e}")
        return False

def setup_mixed_precision():
    """Setup mixed precision like training"""
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("✅ Mixed precision enabled: mixed_float16")
    except Exception as e:
        print(f"❌ Mixed precision failed: {e}")

def resize_volume_like_training(volume, target_shape):
    """Use same resize function as training"""
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=1)

def preprocess_like_training(image_data, target_shape):
    """Preprocess exactly like training script"""
    print(f"   Original shape: {image_data.shape}")
    
    # Resize like training
    if image_data.shape != target_shape:
        print(f"   Resizing from {image_data.shape} to {target_shape}")
        image_data = resize_volume_like_training(image_data, target_shape)
    
    print(f"   Post-resize shape: {image_data.shape}")
    
    # Normalize like training (percentile clipping + min-max normalization)
    p1, p99 = np.percentile(image_data[image_data > 0], [1, 99])
    image_data = np.clip(image_data, p1, p99)
    image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min() + 1e-8)
    
    print(f"   Normalized range: [{image_data.min():.4f}, {image_data.max():.4f}]")
    
    # Add channel dimension like training
    image_data = image_data[..., np.newaxis]  # (192, 224, 176, 1)
    
    return image_data

def test_gpu_compatible():
    """Test with exact same setup as training"""
    print("=" * 70)
    print("GPU-COMPATIBLE TEST (MATCHING TRAINING SETUP)")
    print("=" * 70)
    
    # Setup like training
    configure_gpu_like_training()
    setup_mixed_precision()
    
    # Paths
    model_path = "/mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota/callbacks/sota_20250616_190015/best_model.h5"
    test_case = "sub-r048s014_ses-1"
    test_dir = "/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Testing_Split"
    image_path = f"{test_dir}/Images/{test_case}_space-MNI152NLin2009aSym_T1w.nii.gz"
    mask_path = f"{test_dir}/Masks/{test_case}_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz"
    
    # Same input shape as training
    target_shape = (192, 224, 176)  # Without channel dimension
    
    try:
        # 1. Load model with custom losses (same as training)
        print(f"\n1. Loading model...")
        
        # Use same custom objects as training
        def combined_loss_compatible(y_true, y_pred, smooth=1e-6, focal_gamma=3.0, focal_alpha=0.25):
            """Same combined loss as training"""
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
        
        def dice_coefficient_compatible(y_true, y_pred, smooth=1e-6):
            """Same dice coefficient as training"""
            y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
            y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred, tf.float32))
            intersection = tf.reduce_sum(y_true_f * y_pred_f)
            return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
        
        def binary_dice_coefficient_compatible(y_true, y_pred, smooth=1e-6):
            """Same binary dice as training"""
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred > 0.5, tf.float32)
            y_true_f = tf.keras.backend.flatten(y_true)
            y_pred_f = tf.keras.backend.flatten(y_pred)
            intersection = tf.reduce_sum(y_true_f * y_pred_f)
            return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
        
        custom_objects = {
            'combined_loss': combined_loss_compatible,
            'dice_coefficient': dice_coefficient_compatible,
            'binary_dice_coefficient': binary_dice_coefficient_compatible,
            'dice_loss': dice_loss,
            'focal_loss': focal_loss
        }
        
        model = load_model(model_path, custom_objects=custom_objects, compile=False)
        print(f"   ✅ Model loaded successfully")
        print(f"   Parameters: {model.count_params():,}")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        
        # 2. Load and preprocess image exactly like training
        print(f"\n2. Loading and preprocessing image...")
        nii_img = nib.load(image_path)
        image_data = nii_img.get_fdata(dtype=np.float32)  # Same dtype as training
        
        # Preprocess exactly like training data generator
        processed_image = preprocess_like_training(image_data, target_shape)
        
        # Create batch (batch_size=1 like training)
        image_batch = processed_image[np.newaxis, ...]  # (1, 192, 224, 176, 1)
        print(f"   Final batch shape: {image_batch.shape}")
        
        # Clean up
        del image_data, processed_image
        gc.collect()
        
        # 3. Run prediction (same way as training validation)
        print(f"\n3. Running prediction...")
        start_time = time.time()
        
        # Use same prediction approach as training script validation
        with tf.device('/GPU:0'):
            prediction = model(image_batch, training=False)
        
        pred_time = time.time() - start_time
        print(f"   ✅ Prediction successful in {pred_time:.2f}s")
        print(f"   Prediction shape: {prediction.shape}")
        print(f"   Prediction range: [{prediction.numpy().min():.4f}, {prediction.numpy().max():.4f}]")
        
        # 4. Postprocess
        print(f"\n4. Postprocessing...")
        pred_volume = prediction[0, ..., 0].numpy()  # Remove batch and channel dims
        
        # Resize back to original space
        original_shape = nii_img.shape
        if pred_volume.shape != original_shape:
            factors = [o / p for o, p in zip(original_shape, pred_volume.shape)]
            pred_volume = zoom(pred_volume, factors, order=1)
        
        # Apply threshold
        binary_pred = (pred_volume > 0.5).astype(np.uint8)
        print(f"   Final shape: {binary_pred.shape}")
        print(f"   Predicted lesion voxels: {np.sum(binary_pred):,}")
        
        # 5. Calculate metrics
        print(f"\n5. Calculating metrics...")
        if os.path.exists(mask_path):
            true_mask = nib.load(mask_path).get_fdata().astype(np.uint8)
            
            # Calculate Dice
            intersection = np.sum(binary_pred * true_mask)
            union = np.sum(binary_pred) + np.sum(true_mask)
            dice = (2.0 * intersection) / union if union > 0 else 0.0
            
            print(f"   True lesion voxels: {np.sum(true_mask):,}")
            print(f"   Dice Score: {dice:.4f}")
        
        # 6. Save results
        print(f"\n6. Saving results...")
        output_dir = "gpu_compatible_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save prediction
        pred_nii = nib.Nifti1Image(binary_pred, nii_img.affine)
        pred_path = os.path.join(output_dir, f"{test_case}_gpu_prediction.nii.gz")
        nib.save(pred_nii, pred_path)
        print(f"   ✅ Saved: {pred_path}")
        
        # Save summary
        summary_path = os.path.join(output_dir, f"{test_case}_gpu_results.txt")
        with open(summary_path, 'w') as f:
            f.write(f"GPU-Compatible Test Results\n")
            f.write(f"{'='*35}\n\n")
            f.write(f"Test Case: {test_case}\n")
            f.write(f"Model: {os.path.basename(model_path)}\n")
            f.write(f"Device: GPU (cuda_malloc_async)\n")
            f.write(f"Mixed Precision: Yes\n")
            f.write(f"Parameters: {model.count_params():,}\n")
            f.write(f"Prediction Time: {pred_time:.2f}s\n")
            f.write(f"Target Shape: {target_shape}\n")
            f.write(f"Final Shape: {binary_pred.shape}\n")
            f.write(f"Predicted Voxels: {np.sum(binary_pred):,}\n")
            if os.path.exists(mask_path):
                f.write(f"True Voxels: {np.sum(true_mask):,}\n")
                f.write(f"Dice Score: {dice:.4f}\n")
        
        print(f"   ✅ Summary: {summary_path}")
        
        # 7. Cleanup
        del model, image_batch, prediction, binary_pred
        if 'true_mask' in locals():
            del true_mask
        tf.keras.backend.clear_session()
        gc.collect()
        
        print(f"\n{'='*70}")
        print(f"✅ GPU TEST SUCCESSFUL!")
        print(f"{'='*70}")
        print(f"Key Results:")
        if 'dice' in locals():
            print(f"  Dice Score: {dice:.4f}")
        print(f"  Prediction Time: {pred_time:.2f}s")
        print(f"  GPU Memory: Working correctly")
        print(f"\n🎉 Ready for batch processing of all 55 test cases!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ GPU test failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Emergency cleanup
        try:
            tf.keras.backend.clear_session()
            gc.collect()
        except:
            pass
        
        return False

if __name__ == "__main__":
    success = test_gpu_compatible()
    
    if success:
        print(f"\n🚀 SUCCESS! GPU works with training-compatible setup")
        print(f"Next step: Run batch testing on all 55 cases")
    else:
        print(f"\n🔧 GPU test failed - check error details above")
