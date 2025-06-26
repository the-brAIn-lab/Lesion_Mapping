#!/usr/bin/env python3
"""
Extremely verbose single case test to identify exact failure point
"""

import os
import sys
import gc
import psutil
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.ndimage import zoom
import time
import traceback

# Custom imports
sys.path.append('/mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota')
from models.losses import dice_loss, combined_loss, focal_loss

def log_memory_usage(step_name):
    """Log detailed memory usage"""
    print(f"\n{'='*50}")
    print(f"MEMORY CHECK: {step_name}")
    print(f"{'='*50}")
    
    # System memory
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    print(f"System RAM: {memory.used/1e9:.2f}GB / {memory.total/1e9:.2f}GB ({memory.percent:.1f}%)")
    print(f"Available RAM: {memory.available/1e9:.2f}GB")
    print(f"Swap: {swap.used/1e9:.2f}GB / {swap.total/1e9:.2f}GB ({swap.percent:.1f}%)")
    
    # GPU memory
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Try different methods to get GPU memory info
            result = os.popen('nvidia-smi --query-gpu=memory.used,memory.total --format=csv,nounits,noheader').read()
            if result.strip():
                used, total = result.strip().split(', ')
                print(f"GPU Memory: {used}MB / {total}MB ({100*int(used)/int(total):.1f}%)")
            else:
                print("GPU Memory: Info not available")
        except Exception as e:
            print(f"GPU Memory: Error getting info - {e}")
    else:
        print("GPU: Not detected")
    
    print(f"{'='*50}\n")

def safe_step(step_name, func, *args, **kwargs):
    """Execute a step with verbose logging and error handling"""
    print(f"\n🔄 STARTING: {step_name}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    log_memory_usage(f"BEFORE {step_name}")
    
    try:
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        
        print(f"✅ COMPLETED: {step_name} in {elapsed:.2f}s")
        log_memory_usage(f"AFTER {step_name}")
        
        return result
        
    except Exception as e:
        print(f"❌ FAILED: {step_name}")
        print(f"Error: {str(e)}")
        print(f"Traceback:")
        traceback.print_exc()
        log_memory_usage(f"ERROR in {step_name}")
        raise

def setup_tensorflow():
    """Setup TensorFlow with detailed logging"""
    print("Setting up TensorFlow...")
    
    # GPU configuration
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Physical GPUs detected: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
    
    if gpus:
        try:
            # Configure GPU
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print("✅ GPU memory growth enabled")
        except RuntimeError as e:
            print(f"❌ GPU configuration failed: {e}")
    
    # Mixed precision
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("✅ Mixed precision enabled")
    except Exception as e:
        print(f"❌ Mixed precision failed: {e}")
    
    # TensorFlow info
    print(f"TensorFlow version: {tf.__version__}")
    print(f"CUDA available: {tf.test.is_built_with_cuda()}")
    print(f"GPU available: {tf.test.is_gpu_available()}")

def load_model_verbose(model_path):
    """Load model with verbose logging"""
    print(f"Loading model from: {model_path}")
    print(f"File exists: {os.path.exists(model_path)}")
    print(f"File size: {os.path.getsize(model_path)/1e6:.1f}MB")
    
    # Custom objects
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
    
    print("Loading with custom objects...")
    model = load_model(model_path, custom_objects=custom_objects, compile=False)
    
    print(f"✅ Model loaded successfully")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
    print(f"  Parameters: {model.count_params():,}")
    print(f"  Layers: {len(model.layers)}")
    
    return model

def load_image_verbose(image_path):
    """Load image with verbose logging"""
    print(f"Loading image: {image_path}")
    print(f"File exists: {os.path.exists(image_path)}")
    print(f"File size: {os.path.getsize(image_path)/1e6:.1f}MB")
    
    nii_img = nib.load(image_path)
    image_data = nii_img.get_fdata(dtype=np.float32)
    
    print(f"  Original shape: {image_data.shape}")
    print(f"  Original dtype: {image_data.dtype}")
    print(f"  Original range: [{image_data.min():.2f}, {image_data.max():.2f}]")
    print(f"  Memory usage: {image_data.nbytes/1e6:.1f}MB")
    print(f"  Non-zero voxels: {np.count_nonzero(image_data):,}")
    
    return nii_img, image_data

def preprocess_verbose(image_data, target_shape):
    """Preprocess with verbose logging"""
    print(f"Preprocessing to shape: {target_shape}")
    original_shape = image_data.shape
    
    # Resize if needed
    if image_data.shape != target_shape:
        print(f"  Resizing from {original_shape} to {target_shape}")
        factors = [t / s for t, s in zip(target_shape, original_shape)]
        print(f"  Zoom factors: {factors}")
        
        image_data = zoom(image_data, factors, order=1, mode='constant', cval=0)
        print(f"  ✅ Resized to: {image_data.shape}")
        print(f"  Memory after resize: {image_data.nbytes/1e6:.1f}MB")
    
    # Normalize
    print("  Normalizing...")
    p1, p99 = np.percentile(image_data[image_data > 0], [1, 99])
    print(f"  Percentiles (1%, 99%): ({p1:.2f}, {p99:.2f})")
    
    image_data = np.clip(image_data, p1, p99)
    image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min() + 1e-8)
    
    print(f"  Normalized range: [{image_data.min():.4f}, {image_data.max():.4f}]")
    
    # Add channel dimension
    processed = image_data[..., np.newaxis]
    print(f"  Final preprocessed shape: {processed.shape}")
    print(f"  Final memory usage: {processed.nbytes/1e6:.1f}MB")
    
    return processed

def predict_verbose(model, image_batch):
    """Run prediction with verbose logging"""
    print(f"Running prediction...")
    print(f"  Input batch shape: {image_batch.shape}")
    print(f"  Input batch memory: {image_batch.nbytes/1e6:.1f}MB")
    print(f"  Input batch range: [{image_batch.min():.4f}, {image_batch.max():.4f}]")
    
    # Run prediction
    start_time = time.time()
    with tf.device('/GPU:0'):
        prediction = model(image_batch, training=False)
    
    pred_time = time.time() - start_time
    print(f"  ✅ Prediction completed in {pred_time:.2f}s")
    print(f"  Output shape: {prediction.shape}")
    print(f"  Output dtype: {prediction.dtype}")
    
    return prediction

def convert_prediction_verbose(prediction):
    """Convert prediction to numpy with verbose logging"""
    print("Converting prediction to numpy...")
    print(f"  Tensor shape: {prediction.shape}")
    print(f"  Tensor dtype: {prediction.dtype}")
    
    # Convert to numpy
    pred_np = prediction.numpy()
    
    print(f"  ✅ Converted to numpy")
    print(f"  Numpy shape: {pred_np.shape}")
    print(f"  Numpy dtype: {pred_np.dtype}")
    print(f"  Numpy memory: {pred_np.nbytes/1e6:.1f}MB")
    print(f"  Range: [{pred_np.min():.4f}, {pred_np.max():.4f}]")
    
    return pred_np

def postprocess_verbose(pred_np, original_shape, nii_affine):
    """Postprocess with verbose logging"""
    print("Postprocessing prediction...")
    
    # Extract volume
    pred_volume = pred_np[0, ..., 0]  # Remove batch and channel dims
    print(f"  Extracted volume shape: {pred_volume.shape}")
    print(f"  Volume memory: {pred_volume.nbytes/1e6:.1f}MB")
    
    # Apply threshold
    binary_pred_small = (pred_volume > 0.5).astype(np.uint8)
    print(f"  Applied threshold (>0.5)")
    print(f"  Binary volume shape: {binary_pred_small.shape}")
    print(f"  Predicted voxels (model space): {np.sum(binary_pred_small):,}")
    
    # Resize to original
    if binary_pred_small.shape != original_shape:
        print(f"  Resizing from {binary_pred_small.shape} to {original_shape}")
        factors = [o / p for o, p in zip(original_shape, binary_pred_small.shape)]
        print(f"  Resize factors: {factors}")
        
        binary_pred = zoom(binary_pred_small, factors, order=0, mode='constant', cval=0)
        print(f"  ✅ Resized to: {binary_pred.shape}")
    else:
        binary_pred = binary_pred_small
    
    print(f"  Final shape: {binary_pred.shape}")
    print(f"  Final memory: {binary_pred.nbytes/1e6:.1f}MB")
    print(f"  Predicted voxels (original space): {np.sum(binary_pred):,}")
    
    return binary_pred

def calculate_metrics_verbose(binary_pred, mask_path):
    """Calculate metrics with verbose logging"""
    print("Calculating metrics...")
    
    if not os.path.exists(mask_path):
        print(f"  ❌ Mask file not found: {mask_path}")
        return None
    
    print(f"  Loading mask: {mask_path}")
    true_mask = nib.load(mask_path).get_fdata().astype(np.uint8)
    
    print(f"  Mask shape: {true_mask.shape}")
    print(f"  True lesion voxels: {np.sum(true_mask):,}")
    
    # Calculate metrics
    intersection = np.sum(binary_pred * true_mask)
    union = np.sum(binary_pred) + np.sum(true_mask)
    
    dice = (2.0 * intersection) / union if union > 0 else 0.0
    
    tp = np.sum((binary_pred == 1) & (true_mask == 1))
    fp = np.sum((binary_pred == 1) & (true_mask == 0))
    fn = np.sum((binary_pred == 0) & (true_mask == 1))
    tn = np.sum((binary_pred == 0) & (true_mask == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    print(f"  Intersection: {intersection:,}")
    print(f"  Union: {union:,}")
    print(f"  TP: {tp:,}, FP: {fp:,}, FN: {fn:,}, TN: {tn:,}")
    print(f"  ✅ Dice Score: {dice:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    
    return {
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'pred_voxels': np.sum(binary_pred),
        'true_voxels': np.sum(true_mask)
    }

def main():
    """Main verbose test function"""
    print("="*80)
    print("VERBOSE SINGLE CASE TEST - EXACT FAILURE IDENTIFICATION")
    print("="*80)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Initial memory check
    log_memory_usage("INITIAL")
    
    # Paths
    model_path = "/mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota/callbacks/sota_20250616_190015/best_model.h5"
    test_case = "sub-r048s014_ses-1"
    test_dir = "/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Testing_Split"
    image_path = f"{test_dir}/Images/{test_case}_space-MNI152NLin2009aSym_T1w.nii.gz"
    mask_path = f"{test_dir}/Masks/{test_case}_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz"
    
    print(f"\nTest case: {test_case}")
    print(f"Model: {model_path}")
    print(f"Image: {image_path}")
    print(f"Mask: {mask_path}")
    
    try:
        # Step 1: Setup TensorFlow
        safe_step("TensorFlow Setup", setup_tensorflow)
        
        # Step 2: Load model
        model = safe_step("Model Loading", load_model_verbose, model_path)
        
        # Step 3: Load image
        nii_img, image_data = safe_step("Image Loading", load_image_verbose, image_path)
        original_shape = image_data.shape
        
        # Step 4: Preprocess
        processed = safe_step("Preprocessing", preprocess_verbose, image_data, (192, 224, 176))
        
        # Create batch
        image_batch = processed[np.newaxis, ...]
        print(f"Created batch shape: {image_batch.shape}")
        
        # Clean up
        del image_data, processed
        gc.collect()
        print("Cleaned up intermediate data")
        
        # Step 5: Prediction
        prediction = safe_step("Prediction", predict_verbose, model, image_batch)
        
        # Clean up model and input
        del model, image_batch
        tf.keras.backend.clear_session()
        gc.collect()
        print("Cleaned up model and input")
        
        # Step 6: Convert to numpy
        pred_np = safe_step("Numpy Conversion", convert_prediction_verbose, prediction)
        
        # Clean up prediction tensor
        del prediction
        gc.collect()
        print("Cleaned up prediction tensor")
        
        # Step 7: Postprocess
        binary_pred = safe_step("Postprocessing", postprocess_verbose, pred_np, original_shape, nii_img.affine)
        
        # Clean up
        del pred_np
        gc.collect()
        
        # Step 8: Calculate metrics
        metrics = safe_step("Metrics Calculation", calculate_metrics_verbose, binary_pred, mask_path)
        
        # Step 9: Save results
        def save_results():
            output_dir = "verbose_test_output"
            os.makedirs(output_dir, exist_ok=True)
            
            # Save prediction
            pred_nii = nib.Nifti1Image(binary_pred, nii_img.affine)
            pred_path = os.path.join(output_dir, f"{test_case}_verbose_prediction.nii.gz")
            nib.save(pred_nii, pred_path)
            print(f"Saved prediction: {pred_path}")
            
            # Save detailed log
            log_path = os.path.join(output_dir, f"{test_case}_verbose_log.txt")
            with open(log_path, 'w') as f:
                f.write(f"Verbose Test Results\n")
                f.write(f"===================\n\n")
                f.write(f"Test Case: {test_case}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Original Shape: {original_shape}\n")
                f.write(f"Model Input Shape: (192, 224, 176)\n")
                if metrics:
                    for key, value in metrics.items():
                        f.write(f"{key}: {value}\n")
                f.write(f"\nStatus: SUCCESS - All steps completed\n")
            
            print(f"Saved log: {log_path}")
        
        safe_step("Save Results", save_results)
        
        # Final cleanup
        del binary_pred
        gc.collect()
        
        print("\n" + "="*80)
        print("🎉 VERBOSE TEST COMPLETED SUCCESSFULLY!")
        print("="*80)
        if metrics:
            print(f"Dice Score: {metrics['dice']:.4f}")
        print("All steps completed without errors")
        print("Ready for batch processing!")
        
        return True
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"❌ VERBOSE TEST FAILED")
        print(f"{'='*80}")
        print(f"Final error: {str(e)}")
        traceback.print_exc()
        log_memory_usage("FINAL ERROR")
        
        # Emergency cleanup
        try:
            tf.keras.backend.clear_session()
            gc.collect()
        except:
            pass
        
        return False

if __name__ == "__main__":
    success = main()
    print(f"\nTest result: {'SUCCESS' if success else 'FAILED'}")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
