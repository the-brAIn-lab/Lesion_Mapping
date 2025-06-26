#!/usr/bin/env python3
"""
Process prediction in chunks to avoid memory issues
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
    
    p1, p99 = np.percentile(image_data[image_data > 0], [1, 99])
    image_data = np.clip(image_data, p1, p99)
    image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min() + 1e-8)
    
    return image_data[..., np.newaxis]

def process_prediction_in_chunks(prediction_tensor, chunk_size=32):
    """Process prediction tensor in smaller chunks to avoid OOM"""
    print(f"   Processing prediction in chunks of {chunk_size} slices...")
    
    # Get shape
    batch_size, height, width, depth, channels = prediction_tensor.shape
    print(f"   Tensor shape: {prediction_tensor.shape}")
    
    # Initialize output array
    result = np.zeros((height, width, depth), dtype=np.float32)
    
    # Process in chunks along depth dimension
    num_chunks = (depth + chunk_size - 1) // chunk_size
    print(f"   Processing {num_chunks} chunks...")
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, depth)
        
        print(f"   Chunk {i+1}/{num_chunks}: slices {start_idx}-{end_idx}")
        
        # Extract chunk
        chunk_tensor = prediction_tensor[0, :, :, start_idx:end_idx, 0]
        
        # Convert to numpy
        chunk_np = chunk_tensor.numpy()
        
        # Store in result
        result[:, :, start_idx:end_idx] = chunk_np
        
        # Clean up
        del chunk_tensor, chunk_np
        gc.collect()
        
        if i % 5 == 0:  # Progress update every 5 chunks
            print(f"     Processed {i+1}/{num_chunks} chunks")
    
    print(f"   ✅ All chunks processed")
    return result

def chunked_test():
    """Test with chunked prediction processing"""
    print("=" * 60)
    print("CHUNKED PREDICTION PROCESSING TEST")
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
        
        # Clean up
        del image_data, processed
        gc.collect()
        
        # 3. Predict
        print("3. Running prediction...")
        with tf.device('/GPU:0'):
            prediction = model(image_batch, training=False)
        
        print(f"   ✅ Prediction tensor created: {prediction.shape}")
        
        # Clean up input and model immediately
        del image_batch, model
        tf.keras.backend.clear_session()
        gc.collect()
        print("   Model and input cleared from memory")
        
        # 4. Process prediction in chunks
        print("4. Processing prediction in chunks...")
        
        # Process with smaller chunks first
        pred_volume = process_prediction_in_chunks(prediction, chunk_size=16)
        
        # Clear prediction tensor
        del prediction
        gc.collect()
        print("   Prediction tensor cleared")
        
        print(f"   Volume shape: {pred_volume.shape}")
        print(f"   Volume range: [{pred_volume.min():.4f}, {pred_volume.max():.4f}]")
        
        # 5. Apply threshold
        print("5. Applying threshold...")
        binary_pred_small = (pred_volume > 0.5).astype(np.uint8)
        del pred_volume
        gc.collect()
        print(f"   Binary volume: {binary_pred_small.shape}")
        print(f"   Predicted voxels (model space): {np.sum(binary_pred_small):,}")
        
        # 6. Resize to original space  
        print("6. Resizing to original space...")
        if binary_pred_small.shape != original_shape:
            factors = [o / p for o, p in zip(original_shape, binary_pred_small.shape)]
            print(f"   Resize factors: {factors}")
            
            # Resize in chunks if original volume is large
            if np.prod(original_shape) > 50e6:  # If >50M voxels
                print("   Large volume - processing resize in chunks...")
                # Process in smaller sections
                depth_chunks = 4
                chunk_depth = original_shape[2] // depth_chunks
                
                binary_pred = np.zeros(original_shape, dtype=np.uint8)
                
                for i in range(depth_chunks):
                    start_z = i * chunk_depth
                    end_z = (i + 1) * chunk_depth if i < depth_chunks - 1 else original_shape[2]
                    
                    start_z_small = int(start_z / factors[2])
                    end_z_small = int(end_z / factors[2])
                    
                    chunk_small = binary_pred_small[:, :, start_z_small:end_z_small]
                    chunk_factors = [factors[0], factors[1], (end_z - start_z) / (end_z_small - start_z_small)]
                    
                    chunk_resized = zoom(chunk_small, chunk_factors, order=0)
                    binary_pred[:, :, start_z:end_z] = chunk_resized
                    
                    del chunk_small, chunk_resized
                    gc.collect()
                    
                    print(f"     Processed depth chunk {i+1}/{depth_chunks}")
            else:
                binary_pred = zoom(binary_pred_small, factors, order=0)
        else:
            binary_pred = binary_pred_small
        
        del binary_pred_small
        gc.collect()
        
        print(f"   Final shape: {binary_pred.shape}")
        print(f"   Predicted voxels (original space): {np.sum(binary_pred):,}")
        
        # 7. Calculate metrics
        print("7. Calculating metrics...")
        if os.path.exists(mask_path):
            true_mask = nib.load(mask_path).get_fdata().astype(np.uint8)
            
            intersection = np.sum(binary_pred * true_mask)
            union = np.sum(binary_pred) + np.sum(true_mask)
            dice = (2.0 * intersection) / union if union > 0 else 0.0
            
            print(f"   True lesion voxels: {np.sum(true_mask):,}")
            print(f"   Dice Score: {dice:.4f}")
            
            del true_mask
            gc.collect()
        
        # 8. Save results
        print("8. Saving results...")
        output_dir = "chunked_output"
        os.makedirs(output_dir, exist_ok=True)
        
        pred_nii = nib.Nifti1Image(binary_pred, nii_img.affine)
        pred_path = os.path.join(output_dir, f"{test_case}_chunked_prediction.nii.gz")
        nib.save(pred_nii, pred_path)
        
        # Save metrics
        summary_path = os.path.join(output_dir, f"{test_case}_chunked_results.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Chunked Processing Test Results\n")
            f.write(f"==============================\n\n")
            f.write(f"Test Case: {test_case}\n")
            f.write(f"Original Shape: {original_shape}\n")
            f.write(f"Model Shape: (192, 224, 176)\n")
            f.write(f"Processing: Chunked (16 slices)\n")
            f.write(f"Predicted Voxels: {np.sum(binary_pred):,}\n")
            if 'dice' in locals():
                f.write(f"Dice Score: {dice:.4f}\n")
        
        print(f"   ✅ Prediction saved: {pred_path}")
        print(f"   ✅ Results saved: {summary_path}")
        
        # Final cleanup
        del binary_pred, pred_nii
        gc.collect()
        
        print(f"\n{'='*60}")
        print(f"✅ CHUNKED PROCESSING SUCCESSFUL!")
        print(f"{'='*60}")
        if 'dice' in locals():
            print(f"Dice Score: {dice:.4f}")
        print(f"Memory strategy: Chunked processing")
        print(f"🎉 Ready for batch processing!")
        
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
    success = chunked_test()
    
    if success:
        print(f"\n🚀 SUCCESS! Chunked processing works")
        print(f"Memory issue solved - can now batch process all 55 cases")
    else:
        print(f"\n🔧 Need to debug further")
