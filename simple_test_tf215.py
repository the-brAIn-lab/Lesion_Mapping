#!/usr/bin/env python3
"""
Simple TensorFlow 2.15 compatible test script for stroke lesion segmentation
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
    """Setup TensorFlow with memory growth"""
    print("Setting up TensorFlow...")
    
    # Configure GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU memory growth enabled for {len(gpus)} GPU(s)")
            
            # Print GPU info
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
            
        except RuntimeError as e:
            print(f"⚠️  GPU setup warning: {e}")
            print("Continuing anyway...")
    else:
        print("⚠️  No GPUs detected")
    
    # Set mixed precision policy
    try:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("✅ Mixed precision enabled")
    except:
        print("⚠️  Mixed precision not available")

def load_and_preprocess_image(image_path, target_shape=(192, 224, 176)):
    """Load and preprocess a single image"""
    print(f"Loading image: {os.path.basename(image_path)}")
    
    # Load image
    nii_img = nib.load(image_path)
    image_data = nii_img.get_fdata().astype(np.float32)
    original_shape = image_data.shape
    
    print(f"   Original shape: {original_shape}")
    print(f"   Original range: [{image_data.min():.2f}, {image_data.max():.2f}]")
    
    # Normalize
    mean_val = np.mean(image_data)
    std_val = np.std(image_data)
    image_data = (image_data - mean_val) / (std_val + 1e-8)
    
    print(f"   Normalized range: [{image_data.min():.2f}, {image_data.max():.2f}]")
    
    # Resize to target shape
    zoom_factors = [target_shape[i] / original_shape[i] for i in range(3)]
    print(f"   Zoom factors: {zoom_factors}")
    
    resized_image = zoom(image_data, zoom_factors, order=1, mode='constant', cval=0)
    print(f"   Resized shape: {resized_image.shape}")
    
    # Add batch and channel dimensions
    processed_image = resized_image[np.newaxis, ..., np.newaxis]  # (1, 192, 224, 176, 1)
    
    # Clean up
    del image_data, resized_image
    gc.collect()
    
    return processed_image, nii_img.affine, original_shape

def run_prediction(model, image_batch):
    """Run model prediction with memory management"""
    print("Running prediction...")
    
    # Clear any previous computations
    tf.keras.backend.clear_session()
    gc.collect()
    
    start_time = time.time()
    
    # Run prediction
    with tf.device('/GPU:0'):
        prediction = model.predict(image_batch, batch_size=1, verbose=0)
    
    prediction_time = time.time() - start_time
    print(f"   Prediction completed in {prediction_time:.2f} seconds")
    print(f"   Prediction shape: {prediction.shape}")
    print(f"   Prediction range: [{prediction.min():.4f}, {prediction.max():.4f}]")
    
    # Clean up
    gc.collect()
    
    return prediction

def postprocess_prediction(prediction, original_shape, threshold=0.5):
    """Postprocess prediction back to original space"""
    print("Postprocessing prediction...")
    
    # Remove batch and channel dimensions
    pred_volume = prediction[0, ..., 0]  # (192, 224, 176)
    
    # Resize back to original shape
    zoom_factors = [original_shape[i] / pred_volume.shape[i] for i in range(3)]
    resized_pred = zoom(pred_volume, zoom_factors, order=1, mode='constant', cval=0)
    
    # Apply threshold
    binary_pred = (resized_pred > threshold).astype(np.uint8)
    
    print(f"   Postprocessed shape: {binary_pred.shape}")
    print(f"   Predicted lesion voxels: {np.sum(binary_pred)}")
    
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

def test_single_case():
    """Test a single case end-to-end"""
    print("=" * 70)
    print("STROKE LESION SEGMENTATION - SINGLE CASE TEST")
    print("=" * 70)
    
    # Paths
    model_path = "/mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota/callbacks/sota_20250616_190015/best_model.h5"
    test_case = "sub-r048s014_ses-1"
    test_dir = "/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Testing_Split"
    image_path = f"{test_dir}/Images/{test_case}_space-MNI152NLin2009aSym_T1w.nii.gz"
    mask_path = f"{test_dir}/Masks/{test_case}_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz"
    
    # Verify files exist
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return False
    
    if not os.path.exists(mask_path):
        print(f"❌ Mask not found: {mask_path}")
        return False
    
    print(f"✅ All files found")
    print(f"   Model: {os.path.basename(model_path)}")
    print(f"   Test case: {test_case}")
    
    try:
        # 1. Setup TensorFlow
        setup_tensorflow()
        
        # 2. Load model
        print(f"\n{'='*50}")
        print("LOADING MODEL")
        print(f"{'='*50}")
        
        custom_objects = {
            'dice_loss': dice_loss,
            'combined_loss': combined_loss,
            'focal_loss': focal_loss
        }
        
        model = load_model(model_path, custom_objects=custom_objects, compile=False)
        print(f"✅ Model loaded successfully")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Parameters: {model.count_params():,}")
        
        # 3. Load and preprocess image
        print(f"\n{'='*50}")
        print("PREPROCESSING IMAGE")
        print(f"{'='*50}")
        
        image_batch, affine, original_shape = load_and_preprocess_image(image_path)
        
        # 4. Run prediction
        print(f"\n{'='*50}")
        print("RUNNING PREDICTION")
        print(f"{'='*50}")
        
        prediction = run_prediction(model, image_batch)
        
        # 5. Postprocess
        print(f"\n{'='*50}")
        print("POSTPROCESSING")
        print(f"{'='*50}")
        
        binary_pred = postprocess_prediction(prediction, original_shape)
        
        # 6. Load ground truth and calculate metrics
        print(f"\n{'='*50}")
        print("CALCULATING METRICS")
        print(f"{'='*50}")
        
        true_mask = nib.load(mask_path).get_fdata().astype(np.uint8)
        dice_score = calculate_dice_score(binary_pred, true_mask)
        
        print(f"   True lesion voxels: {np.sum(true_mask):,}")
        print(f"   Predicted lesion voxels: {np.sum(binary_pred):,}")
        print(f"   Dice Score: {dice_score:.4f}")
        
        # 7. Save results
        print(f"\n{'='*50}")
        print("SAVING RESULTS")
        print(f"{'='*50}")
        
        output_dir = "simple_test_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save prediction
        pred_nii = nib.Nifti1Image(binary_pred, affine)
        pred_path = os.path.join(output_dir, f"{test_case}_prediction.nii.gz")
        nib.save(pred_nii, pred_path)
        print(f"✅ Prediction saved: {pred_path}")
        
        # Save metrics
        metrics_path = os.path.join(output_dir, f"{test_case}_results.txt")
        with open(metrics_path, 'w') as f:
            f.write(f"Stroke Lesion Segmentation Results\n")
            f.write(f"{'='*40}\n\n")
            f.write(f"Test Case: {test_case}\n")
            f.write(f"Model: {os.path.basename(model_path)}\n")
            f.write(f"Original Shape: {original_shape}\n")
            f.write(f"True Lesion Voxels: {np.sum(true_mask):,}\n")
            f.write(f"Predicted Lesion Voxels: {np.sum(binary_pred):,}\n")
            f.write(f"Dice Score: {dice_score:.4f}\n")
        
        print(f"✅ Results saved: {metrics_path}")
        
        # 8. Final cleanup
        del model, image_batch, prediction, binary_pred, true_mask
        tf.keras.backend.clear_session()
        gc.collect()
        
        print(f"\n{'='*70}")
        print("✅ SUCCESS! Test completed successfully")
        print(f"Dice Score: {dice_score:.4f}")
        print(f"Results saved in: {output_dir}")
        print(f"{'='*70}")
        
        return True
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ ERROR: {str(e)}")
        print(f"{'='*70}")
        
        # Emergency cleanup
        try:
            tf.keras.backend.clear_session()
            gc.collect()
        except:
            pass
        
        return False

if __name__ == "__main__":
    success = test_single_case()
    if success:
        print("\n🎉 Ready to run full batch testing!")
    else:
        print("\n🔧 Check the error and try again")
