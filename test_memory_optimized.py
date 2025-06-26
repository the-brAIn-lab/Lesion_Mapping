#!/usr/bin/env python3
"""
Memory-optimized test script for SOTA stroke lesion segmentation model
Addresses OOM issues during inference on large 3D volumes
"""

import os
import sys
import gc
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from pathlib import Path
import time
import psutil

# Custom imports
sys.path.append('/mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota')
from models.losses import dice_loss, combined_loss, focal_loss

def setup_gpu_memory():
    """Configure GPU memory growth to prevent OOM"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # For TensorFlow 2.15, use virtual device config for memory limits
            if len(gpus) > 0:
                try:
                    # Try to set memory limit (20GB = 20480MB)
                    tf.config.experimental.set_virtual_device_configuration(
                        gpus[0],
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=20480)]
                    )
                    print(f"GPU memory growth enabled with 20GB limit")
                except RuntimeError:
                    # If virtual device config fails, just use memory growth
                    print(f"GPU memory growth enabled (no limit set)")
            
            print(f"Configured {len(gpus)} GPU(s)")
            return True
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
            print("Continuing without GPU memory configuration...")
            return True  # Continue anyway
    else:
        print("No GPUs found")
        return False

def check_memory_usage():
    """Monitor system memory usage"""
    memory = psutil.virtual_memory()
    print(f"System RAM: {memory.percent:.1f}% used ({memory.used/1e9:.1f}GB/{memory.total/1e9:.1f}GB)")
    
    # Check GPU memory if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # For TF 2.15, use different method to check GPU memory
            gpu_details = tf.config.experimental.get_device_details(gpus[0])
            print(f"GPU: {gpu_details.get('device_name', 'Unknown')}")
        except Exception as e:
            print(f"GPU info unavailable: {e}")
    else:
        print("GPU: Not available")

def preprocess_image_memory_efficient(image_path, target_shape=(192, 224, 176)):
    """
    Memory-efficient image preprocessing with chunked processing
    """
    print(f"Loading image: {image_path}")
    
    # Load image
    nii_img = nib.load(image_path)
    image_data = nii_img.get_fdata().astype(np.float32)
    original_shape = image_data.shape
    print(f"Original shape: {original_shape}")
    
    # Memory check
    check_memory_usage()
    
    # Normalize (in-place to save memory)
    image_data = (image_data - np.mean(image_data)) / (np.std(image_data) + 1e-8)
    
    # Resize efficiently
    zoom_factors = [target_shape[i] / original_shape[i] for i in range(3)]
    print(f"Zoom factors: {zoom_factors}")
    
    # Use order=1 for better quality but still memory efficient
    resized_image = zoom(image_data, zoom_factors, order=1, mode='constant', cval=0)
    
    # Clean up original
    del image_data
    gc.collect()
    
    # Add batch and channel dimensions
    resized_image = resized_image[np.newaxis, ..., np.newaxis]  # (1, 192, 224, 176, 1)
    
    print(f"Preprocessed shape: {resized_image.shape}")
    check_memory_usage()
    
    return resized_image, nii_img.affine, original_shape

def predict_with_memory_management(model, image_batch):
    """
    Run prediction with memory management
    """
    print("Running prediction with memory management...")
    check_memory_usage()
    
    # Clear any existing computations
    tf.keras.backend.clear_session()
    gc.collect()
    
    # Use gradient tape for better memory management
    with tf.device('/GPU:0'):
        # Convert to tensor
        image_tensor = tf.convert_to_tensor(image_batch, dtype=tf.float32)
        
        # Run prediction in no_grad context equivalent
        prediction = model(image_tensor, training=False)
        
        # Convert back to numpy immediately
        pred_np = prediction.numpy()
        
        # Clean up tensors
        del image_tensor, prediction
        tf.keras.backend.clear_session()
        gc.collect()
    
    check_memory_usage()
    return pred_np

def postprocess_prediction(prediction, original_shape, affine):
    """
    Postprocess prediction back to original space
    """
    print("Postprocessing prediction...")
    
    # Remove batch dimension
    pred_volume = prediction[0, ..., 0]  # (192, 224, 176)
    
    # Resize back to original shape
    zoom_factors = [original_shape[i] / pred_volume.shape[i] for i in range(3)]
    resized_pred = zoom(pred_volume, zoom_factors, order=1, mode='constant', cval=0)
    
    # Apply threshold
    binary_pred = (resized_pred > 0.5).astype(np.uint8)
    
    # Clean up
    del pred_volume, resized_pred
    gc.collect()
    
    return binary_pred

def calculate_dice_score(pred_mask, true_mask):
    """Calculate Dice coefficient"""
    intersection = np.sum(pred_mask * true_mask)
    union = np.sum(pred_mask) + np.sum(true_mask)
    
    if union == 0:
        return 1.0 if np.sum(pred_mask) == 0 else 0.0
    
    dice = (2.0 * intersection) / union
    return dice

def test_single_case_optimized(model_path, image_path, mask_path=None, output_dir="test_output"):
    """
    Test single case with memory optimization
    """
    print("=" * 60)
    print("MEMORY-OPTIMIZED SINGLE CASE TEST")
    print("=" * 60)
    
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Setup GPU
    print("\n1. Setting up GPU...")
    if not setup_gpu_memory():
        print("Warning: GPU setup failed, using CPU")
    
    # 2. Load model
    print("\n2. Loading model...")
    check_memory_usage()
    
    custom_objects = {
        'dice_loss': dice_loss,
        'combined_loss': combined_loss,
        'focal_loss': focal_loss
    }
    
    try:
        model = load_model(model_path, custom_objects=custom_objects, compile=False)
        print("   Model loaded successfully!")
        print(f"   Model input shape: {model.input_shape}")
    except Exception as e:
        print(f"   Error loading model: {e}")
        return None
    
    # 3. Preprocess image
    print("\n3. Preprocessing image...")
    try:
        image_batch, affine, original_shape = preprocess_image_memory_efficient(image_path)
    except Exception as e:
        print(f"   Error preprocessing image: {e}")
        return None
    
    # 4. Run prediction
    print("\n4. Running prediction...")
    try:
        prediction = predict_with_memory_management(model, image_batch)
        print(f"   Prediction shape: {prediction.shape}")
        print(f"   Prediction range: [{prediction.min():.4f}, {prediction.max():.4f}]")
    except Exception as e:
        print(f"   Error during prediction: {e}")
        return None
    
    # 5. Postprocess
    print("\n5. Postprocessing...")
    try:
        binary_pred = postprocess_prediction(prediction, original_shape, affine)
        print(f"   Final prediction shape: {binary_pred.shape}")
        print(f"   Predicted lesion voxels: {np.sum(binary_pred)}")
    except Exception as e:
        print(f"   Error postprocessing: {e}")
        return None
    
    # 6. Save results
    print("\n6. Saving results...")
    case_name = Path(image_path).name.split('_space-MNI152NLin2009aSym_T1w.nii.gz')[0]
    
    # Save prediction
    pred_nii = nib.Nifti1Image(binary_pred, affine)
    pred_path = os.path.join(output_dir, f"{case_name}_prediction.nii.gz")
    nib.save(pred_nii, pred_path)
    print(f"   Saved prediction: {pred_path}")
    
    # 7. Calculate metrics if mask available
    if mask_path and os.path.exists(mask_path):
        print("\n7. Calculating metrics...")
        true_mask = nib.load(mask_path).get_fdata().astype(np.uint8)
        dice_score = calculate_dice_score(binary_pred, true_mask)
        print(f"   Dice Score: {dice_score:.4f}")
        print(f"   True lesion voxels: {np.sum(true_mask)}")
        
        # Save metrics
        metrics_path = os.path.join(output_dir, f"{case_name}_metrics.txt")
        with open(metrics_path, 'w') as f:
            f.write(f"Case: {case_name}\n")
            f.write(f"Dice Score: {dice_score:.4f}\n")
            f.write(f"True lesion voxels: {np.sum(true_mask)}\n")
            f.write(f"Predicted lesion voxels: {np.sum(binary_pred)}\n")
            f.write(f"Original shape: {original_shape}\n")
    
    # 8. Final cleanup
    print("\n8. Final cleanup...")
    del model, image_batch, prediction, binary_pred
    tf.keras.backend.clear_session()
    gc.collect()
    check_memory_usage()
    
    print("\n✅ Test completed successfully!")
    return True

def main():
    """Main testing function"""
    # Paths
    model_path = "/mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota/callbacks/sota_20250616_190015/best_model.h5"
    test_dir = "/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Testing_Split"
    
    # Test single case first
    test_case = "sub-r048s014_ses-1"
    image_path = f"{test_dir}/Images/{test_case}_space-MNI152NLin2009aSym_T1w.nii.gz"
    mask_path = f"{test_dir}/Masks/{test_case}_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz"
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return
    
    # Run test
    success = test_single_case_optimized(
        model_path=model_path,
        image_path=image_path,
        mask_path=mask_path,
        output_dir="memory_optimized_test_output"
    )
    
    if success:
        print("\n🎉 Memory-optimized test successful!")
        print("You can now run the full test suite with similar optimizations.")
    else:
        print("\n❌ Test failed. Check memory settings and try again.")

if __name__ == "__main__":
    main()
