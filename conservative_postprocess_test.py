#!/usr/bin/env python3
"""
Conservative postprocessing to avoid memory issues
"""

import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

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

def configure_gpu():
    """Configure GPU"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    
    # Mixed precision
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

def resize_volume(volume, target_shape):
    """Resize volume"""
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=1)

def preprocess_image(image_data, target_shape):
    """Preprocess like training"""
    if image_data.shape != target_shape:
        image_data = resize_volume(image_data, target_shape)
    
    # Normalize like training
    p1, p99 = np.percentile(image_data[image_data > 0], [1, 99])
    image_data = np.clip(image_data, p1, p99)
    image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min() + 1e-8)
    
    return image_data[..., np.newaxis]

def conservative_test():
    """Test with very conservative memory management"""
    print("=" * 60)
    print("CONSERVATIVE POSTPROCESSING TEST")
    print("=" * 60)
    
    configure_gpu()
    
    # Paths
    model_path = "/mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota/callbacks/sota_20250616_190015/best_model.h5"
    test_case = "sub-r048s014_ses-1"
    test_dir = "/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Testing_Split"
    image_path = f"{test_dir}/Images/{test_case}_space-MNI152NLin2009aSym_T1w.nii.gz"
    mask_path = f"{test_dir}/Masks/{test_case}_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz"
    
    try:
        # 1. Load model
        print("1. Loading model...")
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
        print(f"   ✅ Model loaded: {model.count_params():,} parameters")
        
        # 2. Load and preprocess
        print("2. Loading image...")
        nii_img = nib.load(image_path)
        image_data = nii_img.get_fdata(dtype=np.float32)
        original_shape = image_data.shape
        
        processed = preprocess_image(image_data, (192, 224, 176))
        image_batch = processed[np.newaxis, ...]
        
        print(f"   Input shape: {image_batch.shape}")
        
        # Clean up immediately
        del image_data, processed
        gc.collect()
        
        # 3. Predict
        print("3. Running prediction...")
        with tf.device('/GPU:0'):
            prediction = model(image_batch, training=False)
        
        print(f"   ✅ Prediction done: {prediction.shape}")
        
        # 4. VERY CONSERVATIVE postprocessing
        print("4. Conservative postprocessing...")
        
        # Convert to numpy immediately, slice by slice to save memory
        print("   Converting prediction to numpy...")
        pred_np = prediction.numpy()
        print(f"   Conversion successful: {pred_np.shape}")
        
        # Clear GPU memory immediately
        del prediction, model, image_batch
        tf.keras.backend.clear_session()
        gc.collect()
        print("   GPU memory cleared")
        
        # Extract volume (remove batch and channel dimensions)
        print("   Extracting volume...")
        pred_volume = pred_np[0, ..., 0]  # (192, 224, 176)
        del pred_np
        gc.collect()
        print(f"   Volume extracted: {pred_volume.shape}")
        
        # Apply threshold BEFORE resizing to save memory
        print("   Applying threshold...")
        binary_pred_small = (pred_volume > 0.5).astype(np.uint8)
        del pred_volume
        gc.collect()
        print(f"   Thresholded: {binary_pred_small.shape}")
        
        # Resize back to original shape
        print("   Resizing to original shape...")
        if binary_pred_small.shape != original_shape:
            factors = [o / p for o, p in zip(original_shape, binary_pred_small.shape)]
            print(f"   Resize factors: {factors}")
            binary_pred = zoom(binary_pred_small, factors, order=0)  # Use order=0 for binary
        else:
            binary_pred = binary_pred_small
        
        del binary_pred_small
        gc.collect()
        
        print(f"   Final shape: {binary_pred.shape}")
        print(f"   Predicted lesion voxels: {np.sum(binary_pred):,}")
        
        # 5. Quick metrics
        print("5. Calculating metrics...")
        if os.path.exists(mask_path):
            true_mask = nib.load(mask_path).get_fdata().astype(np.uint8)
            
            intersection = np.sum(binary_pred * true_mask)
            union = np.sum(binary_pred) + np.sum(true_mask)
            dice = (2.0 * intersection) / union if union > 0 else 0.0
            
            print(f"   True lesion voxels: {np.sum(true_mask):,}")
            print(f"   Dice Score: {dice:.4f}")
            
            del true_mask
            gc.collect()
        
        # 6. Save results
        print("6. Saving...")
        output_dir = "conservative_output"
        os.makedirs(output_dir, exist_ok=True)
        
        pred_nii = nib.Nifti1Image(binary_pred, nii_img.affine)
        pred_path = os.path.join(output_dir, f"{test_case}_conservative.nii.gz")
        nib.save(pred_nii, pred_path)
        
        # Save quick summary
        summary_path = os.path.join(output_dir, f"{test_case}_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Conservative Test Results\n")
            f.write(f"========================\n\n")
            f.write(f"Test Case: {test_case}\n")
            f.write(f"Original Shape: {original_shape}\n")
            f.write(f"Model Shape: (192, 224, 176)\n")
            f.write(f"Predicted Voxels: {np.sum(binary_pred):,}\n")
            if 'dice' in locals():
                f.write(f"Dice Score: {dice:.4f}\n")
        
        print(f"   ✅ Saved: {pred_path}")
        print(f"   ✅ Summary: {summary_path}")
        
        # Final cleanup
        del binary_pred, pred_nii
        gc.collect()
        
        print(f"\n{'='*60}")
        print(f"✅ CONSERVATIVE TEST SUCCESSFUL!")
        print(f"{'='*60}")
        if 'dice' in locals():
            print(f"Dice Score: {dice:.4f}")
        print(f"Memory management: Success")
        print(f"Ready for batch processing!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
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
    success = conservative_test()
    
    if success:
        print(f"\n🎉 SUCCESS! Memory management solved")
        print(f"Now we can run batch processing on all 55 cases")
    else:
        print(f"\n❌ Still having memory issues")
