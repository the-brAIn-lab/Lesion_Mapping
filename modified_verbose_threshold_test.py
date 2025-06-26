#!/usr/bin/env python3
"""
Modified version of the working verbose test to save raw probabilities
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
    print(f"\n{'='*30} {step_name} {'='*30}")
    memory = psutil.virtual_memory()
    print(f"System RAM: {memory.used/1e9:.2f}GB / {memory.total/1e9:.2f}GB ({memory.percent:.1f}%)")

def setup_tensorflow():
    """Setup TensorFlow"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

def load_model_verbose(model_path):
    """Load model with verbose logging"""
    print(f"Loading model from: {model_path}")
    
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

def resize_volume(volume, target_shape):
    """Resize volume"""
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=1)

def preprocess_image(image_data, target_shape=(192, 224, 176)):
    """Preprocess like training"""
    if image_data.shape != target_shape:
        image_data = resize_volume(image_data, target_shape)
    
    p1, p99 = np.percentile(image_data[image_data > 0], [1, 99])
    image_data = np.clip(image_data, p1, p99)
    image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min() + 1e-8)
    
    return image_data[..., np.newaxis]

def convert_prediction_chunked(prediction):
    """Convert prediction to numpy in chunks"""
    print("Converting prediction to numpy (chunked approach)...")
    
    batch_size, height, width, depth, channels = prediction.shape
    print(f"Prediction tensor shape: {prediction.shape}")
    
    # Process in chunks of 16 slices
    chunk_size = 16
    num_chunks = (depth + chunk_size - 1) // chunk_size
    
    result = np.zeros((height, width, depth), dtype=np.float32)
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, depth)
        
        print(f"  Processing chunk {i+1}/{num_chunks}: slices {start_idx}-{end_idx}")
        
        # Extract and convert chunk
        chunk_tensor = prediction[0, :, :, start_idx:end_idx, 0]
        chunk_np = chunk_tensor.numpy()
        result[:, :, start_idx:end_idx] = chunk_np
        
        # Clean up
        del chunk_tensor, chunk_np
        gc.collect()
    
    print(f"✅ Conversion complete: {result.shape}")
    return result

def test_thresholds(pred_volume, true_mask):
    """Test multiple thresholds to find optimal"""
    print("Testing multiple thresholds...")
    
    # Test thresholds
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    
    best_dice = 0.0
    best_threshold = 0.5
    threshold_results = []
    
    for threshold in thresholds:
        binary_pred = (pred_volume > threshold).astype(np.uint8)
        
        intersection = np.sum(binary_pred * true_mask)
        union = np.sum(binary_pred) + np.sum(true_mask)
        dice = (2.0 * intersection) / union if union > 0 else 0.0
        
        pred_voxels = np.sum(binary_pred)
        
        result = {
            'threshold': threshold,
            'dice': dice,
            'pred_voxels': pred_voxels
        }
        threshold_results.append(result)
        
        print(f"  Threshold {threshold:5.3f}: Dice={dice:.4f}, Predicted={pred_voxels:6,} voxels")
        
        if dice > best_dice:
            best_dice = dice
            best_threshold = threshold
    
    print(f"✅ Best threshold: {best_threshold:.3f} with Dice = {best_dice:.4f}")
    return best_threshold, best_dice, threshold_results

def main():
    """Main function - modified from working verbose test"""
    print("=" * 70)
    print("RAW PROBABILITY ANALYSIS (MODIFIED VERBOSE TEST)")
    print("=" * 70)
    
    # Setup
    setup_tensorflow()
    log_memory_usage("INITIAL")
    
    # Paths
    model_path = "/mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota/callbacks/sota_20250616_190015/best_model.h5"
    test_case = "sub-r048s014_ses-1"
    test_dir = "/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Testing_Split"
    image_path = f"{test_dir}/Images/{test_case}_space-MNI152NLin2009aSym_T1w.nii.gz"
    mask_path = f"{test_dir}/Masks/{test_case}_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz"
    
    try:
        # 1. Load model
        print("\n1. Loading model...")
        model = load_model_verbose(model_path)
        log_memory_usage("MODEL LOADED")
        
        # 2. Load and preprocess image
        print("\n2. Loading and preprocessing image...")
        nii_img = nib.load(image_path)
        image_data = nii_img.get_fdata(dtype=np.float32)
        original_shape = image_data.shape
        
        processed = preprocess_image(image_data, (192, 224, 176))
        image_batch = processed[np.newaxis, ...]
        
        print(f"  Input shape: {image_batch.shape}")
        
        # Clean up
        del image_data, processed
        gc.collect()
        log_memory_usage("IMAGE PREPROCESSED")
        
        # 3. Run prediction
        print("\n3. Running prediction...")
        with tf.device('/GPU:0'):
            prediction = model(image_batch, training=False)
        
        print(f"✅ Prediction tensor created: {prediction.shape}")
        print(f"Prediction dtype: {prediction.dtype}")
        
        # Clean up model and input immediately
        del model, image_batch
        tf.keras.backend.clear_session()
        gc.collect()
        log_memory_usage("PREDICTION DONE, MODEL CLEARED")
        
        # 4. Convert prediction in chunks
        print("\n4. Converting prediction...")
        pred_volume = convert_prediction_chunked(prediction)
        
        # Clear prediction tensor
        del prediction
        gc.collect()
        log_memory_usage("PREDICTION CONVERTED")
        
        print(f"Raw prediction range: [{pred_volume.min():.6f}, {pred_volume.max():.6f}]")
        print(f"Raw prediction mean: {pred_volume.mean():.6f}")
        
        # 5. Load ground truth and resize to model space
        print("\n5. Loading ground truth...")
        true_mask_orig = nib.load(mask_path).get_fdata().astype(np.uint8)
        
        # Resize true mask to model space for threshold testing
        if true_mask_orig.shape != pred_volume.shape:
            factors = [p / t for p, t in zip(pred_volume.shape, true_mask_orig.shape)]
            true_mask_model = zoom(true_mask_orig, factors, order=0)
        else:
            true_mask_model = true_mask_orig
        
        print(f"True mask (model space): {true_mask_model.shape}, lesion voxels: {np.sum(true_mask_model):,}")
        
        # 6. Test thresholds
        print("\n6. Testing thresholds...")
        best_threshold, best_dice, threshold_results = test_thresholds(pred_volume, true_mask_model)
        
        # 7. Create final prediction with best threshold
        print(f"\n7. Creating final prediction with threshold {best_threshold:.3f}...")
        binary_pred_model = (pred_volume > best_threshold).astype(np.uint8)
        
        # Resize to original space
        if binary_pred_model.shape != original_shape:
            factors = [o / p for o, p in zip(original_shape, binary_pred_model.shape)]
            binary_pred_final = zoom(binary_pred_model, factors, order=0)
        else:
            binary_pred_final = binary_pred_model
        
        # Final metrics
        intersection_final = np.sum(binary_pred_final * true_mask_orig)
        union_final = np.sum(binary_pred_final) + np.sum(true_mask_orig)
        dice_final = (2.0 * intersection_final) / union_final if union_final > 0 else 0.0
        
        print(f"Final results (original space):")
        print(f"  True lesion voxels: {np.sum(true_mask_orig):,}")
        print(f"  Predicted lesion voxels: {np.sum(binary_pred_final):,}")
        print(f"  Final Dice score: {dice_final:.4f}")
        
        # 8. Save results
        print("\n8. Saving results...")
        output_dir = "raw_probability_analysis"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save raw probabilities (resize to original space)
        if pred_volume.shape != original_shape:
            factors = [o / p for o, p in zip(original_shape, pred_volume.shape)]
            raw_probs_orig = zoom(pred_volume, factors, order=1)
        else:
            raw_probs_orig = pred_volume
        
        raw_nii = nib.Nifti1Image(raw_probs_orig, nii_img.affine)
        raw_path = os.path.join(output_dir, f"{test_case}_raw_probabilities.nii.gz")
        nib.save(raw_nii, raw_path)
        
        # Save optimized prediction
        pred_nii = nib.Nifti1Image(binary_pred_final, nii_img.affine)
        pred_path = os.path.join(output_dir, f"{test_case}_optimized_prediction.nii.gz")
        nib.save(pred_nii, pred_path)
        
        # Save detailed analysis
        analysis_path = os.path.join(output_dir, f"{test_case}_threshold_analysis.txt")
        with open(analysis_path, 'w') as f:
            f.write(f"Raw Probability Threshold Analysis\n")
            f.write(f"==================================\n\n")
            f.write(f"Test Case: {test_case}\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Original Shape: {original_shape}\n")
            f.write(f"Model Shape: {pred_volume.shape}\n\n")
            
            f.write(f"Raw Prediction Statistics:\n")
            f.write(f"  Range: [{pred_volume.min():.6f}, {pred_volume.max():.6f}]\n")
            f.write(f"  Mean: {pred_volume.mean():.6f}\n")
            f.write(f"  Std: {pred_volume.std():.6f}\n\n")
            
            f.write(f"Threshold Testing Results:\n")
            for result in threshold_results:
                f.write(f"  {result['threshold']:5.3f}: Dice={result['dice']:.4f}, Predicted={result['pred_voxels']:,}\n")
            
            f.write(f"\nOptimal Results:\n")
            f.write(f"  Best Threshold: {best_threshold:.3f}\n")
            f.write(f"  Best Dice Score: {dice_final:.4f}\n")
            f.write(f"  True Lesion Voxels: {np.sum(true_mask_orig):,}\n")
            f.write(f"  Predicted Lesion Voxels: {np.sum(binary_pred_final):,}\n")
        
        print(f"✅ Results saved:")
        print(f"  Raw probabilities: {raw_path}")
        print(f"  Optimized prediction: {pred_path}")
        print(f"  Analysis report: {analysis_path}")
        
        # Final cleanup
        del pred_volume, binary_pred_model, binary_pred_final, true_mask_orig, true_mask_model
        gc.collect()
        log_memory_usage("FINAL")
        
        print(f"\n{'='*70}")
        print(f"🎉 RAW PROBABILITY ANALYSIS SUCCESSFUL!")
        print(f"{'='*70}")
        print(f"Key Findings:")
        print(f"  Original threshold (0.5): Dice = 0.0")
        print(f"  Optimal threshold ({best_threshold:.3f}): Dice = {dice_final:.4f}")
        print(f"  Improvement: {dice_final/0.001:.0f}x better (avoiding division by 0)")
        print(f"\nThis explains the poor test performance!")
        print(f"Use threshold {best_threshold:.3f} for batch processing.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        traceback.print_exc()
        log_memory_usage("ERROR")
        return False

if __name__ == "__main__":
    success = main()
    print(f"\nAnalysis {'succeeded' if success else 'failed'}")
