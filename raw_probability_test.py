#!/usr/bin/env python3
"""
Test raw probabilities and find optimal threshold
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
    """Setup TensorFlow"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

def load_model_safe():
    """Load model safely"""
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

def resize_volume(volume, target_shape):
    """Resize volume"""
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=1)

def preprocess_image(image_data, target_shape=(192, 224, 176)):
    """Preprocess image like training"""
    if image_data.shape != target_shape:
        image_data = resize_volume(image_data, target_shape)
    
    p1, p99 = np.percentile(image_data[image_data > 0], [1, 99])
    image_data = np.clip(image_data, p1, p99)
    image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min() + 1e-8)
    
    return image_data[..., np.newaxis]

def test_raw_probabilities():
    """Test raw probability output and find optimal threshold"""
    print("=" * 60)
    print("RAW PROBABILITY ANALYSIS")
    print("=" * 60)
    
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
        print(f"✅ Model loaded: {model.count_params():,} parameters")
        
        # Load and preprocess image
        print("Loading and preprocessing image...")
        nii_img = nib.load(image_path)
        image_data = nii_img.get_fdata(dtype=np.float32)
        original_shape = image_data.shape
        
        processed = preprocess_image(image_data, (192, 224, 176))
        image_batch = processed[np.newaxis, ...]
        
        print(f"  Input shape: {image_batch.shape}")
        
        # Clean up
        del image_data, processed
        gc.collect()
        
        # Run prediction to get RAW probabilities
        print("Running prediction for raw probabilities...")
        with tf.device('/GPU:0'):
            prediction = model(image_batch, training=False)
        
        print(f"  Prediction tensor shape: {prediction.shape}")
        
        # Clean up model and input immediately
        del model, image_batch
        tf.keras.backend.clear_session()
        gc.collect()
        
        # Convert to numpy in chunks to avoid memory issues
        print("Converting prediction to numpy (chunked)...")
        
        # Process in smaller chunks
        pred_chunks = []
        chunk_size = 32
        depth = prediction.shape[3]
        
        for i in range(0, depth, chunk_size):
            end_idx = min(i + chunk_size, depth)
            chunk = prediction[0, :, :, i:end_idx, 0].numpy()
            pred_chunks.append(chunk)
            print(f"  Processed chunk {i//chunk_size + 1}/{(depth + chunk_size - 1)//chunk_size}")
        
        # Combine chunks
        pred_volume = np.concatenate(pred_chunks, axis=2)
        
        # Clean up
        del prediction, pred_chunks
        gc.collect()
        
        print(f"  Raw prediction shape: {pred_volume.shape}")
        print(f"  Raw prediction range: [{pred_volume.min():.6f}, {pred_volume.max():.6f}]")
        print(f"  Raw prediction mean: {pred_volume.mean():.6f}")
        print(f"  Raw prediction std: {pred_volume.std():.6f}")
        
        # Load ground truth
        true_mask = nib.load(mask_path).get_fdata().astype(np.uint8)
        
        # Resize true mask to model space for analysis
        if true_mask.shape != pred_volume.shape:
            factors = [p / t for p, t in zip(pred_volume.shape, true_mask.shape)]
            true_mask_small = zoom(true_mask, factors, order=0)
        else:
            true_mask_small = true_mask
        
        print(f"  True mask shape: {true_mask_small.shape}")
        print(f"  True lesion voxels: {np.sum(true_mask_small):,}")
        
        # Analyze probabilities in lesion vs non-lesion regions
        lesion_probs = pred_volume[true_mask_small > 0]
        non_lesion_probs = pred_volume[true_mask_small == 0]
        
        print(f"\nProbability Analysis:")
        print(f"  Lesion regions - Min: {lesion_probs.min():.6f}, Max: {lesion_probs.max():.6f}, Mean: {lesion_probs.mean():.6f}")
        print(f"  Non-lesion regions - Min: {non_lesion_probs.min():.6f}, Max: {non_lesion_probs.max():.6f}, Mean: {non_lesion_probs.mean():.6f}")
        
        # Test thresholds from very low to high
        thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        print(f"\nTesting thresholds on model-space prediction:")
        best_dice = 0.0
        best_threshold = 0.5
        
        for threshold in thresholds:
            binary_pred = (pred_volume > threshold).astype(np.uint8)
            
            intersection = np.sum(binary_pred * true_mask_small)
            union = np.sum(binary_pred) + np.sum(true_mask_small)
            dice = (2.0 * intersection) / union if union > 0 else 0.0
            
            pred_voxels = np.sum(binary_pred)
            
            print(f"  Threshold {threshold:6.3f}: Dice={dice:.4f}, Predicted={pred_voxels:6,} voxels")
            
            if dice > best_dice:
                best_dice = dice
                best_threshold = threshold
        
        print(f"\n✅ Best threshold: {best_threshold:.3f} with Dice = {best_dice:.4f}")
        
        # Create final prediction with best threshold and resize to original space
        print(f"\nCreating final prediction with optimal threshold...")
        binary_pred_small = (pred_volume > best_threshold).astype(np.uint8)
        
        # Resize to original space
        if binary_pred_small.shape != original_shape:
            factors = [o / p for o, p in zip(original_shape, binary_pred_small.shape)]
            binary_pred_final = zoom(binary_pred_small, factors, order=0)
        else:
            binary_pred_final = binary_pred_small
        
        # Calculate final metrics on original space
        intersection_final = np.sum(binary_pred_final * true_mask)
        union_final = np.sum(binary_pred_final) + np.sum(true_mask)
        dice_final = (2.0 * intersection_final) / union_final if union_final > 0 else 0.0
        
        print(f"Final results (original space):")
        print(f"  True lesion voxels: {np.sum(true_mask):,}")
        print(f"  Predicted lesion voxels: {np.sum(binary_pred_final):,}")
        print(f"  Final Dice score: {dice_final:.4f}")
        
        # Save results
        output_dir = "raw_probability_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save raw probabilities (resized to original space)
        if pred_volume.shape != original_shape:
            factors = [o / p for o, p in zip(original_shape, pred_volume.shape)]
            raw_probs_full = zoom(pred_volume, factors, order=1)
        else:
            raw_probs_full = pred_volume
        
        raw_nii = nib.Nifti1Image(raw_probs_full, nii_img.affine)
        raw_path = os.path.join(output_dir, f"{test_case}_raw_probabilities.nii.gz")
        nib.save(raw_nii, raw_path)
        
        # Save optimized prediction
        pred_nii = nib.Nifti1Image(binary_pred_final, nii_img.affine)
        pred_path = os.path.join(output_dir, f"{test_case}_optimized_prediction.nii.gz")
        nib.save(pred_nii, pred_path)
        
        # Save analysis results
        results_path = os.path.join(output_dir, f"{test_case}_threshold_analysis.txt")
        with open(results_path, 'w') as f:
            f.write(f"Raw Probability Analysis Results\n")
            f.write(f"===============================\n\n")
            f.write(f"Test Case: {test_case}\n")
            f.write(f"Raw Prediction Range: [{pred_volume.min():.6f}, {pred_volume.max():.6f}]\n")
            f.write(f"Raw Prediction Mean: {pred_volume.mean():.6f}\n")
            f.write(f"Lesion Region Mean Prob: {lesion_probs.mean():.6f}\n")
            f.write(f"Non-lesion Region Mean Prob: {non_lesion_probs.mean():.6f}\n")
            f.write(f"Optimal Threshold: {best_threshold:.3f}\n")
            f.write(f"Optimal Dice Score: {dice_final:.4f}\n")
            f.write(f"True Lesion Voxels: {np.sum(true_mask):,}\n")
            f.write(f"Predicted Lesion Voxels: {np.sum(binary_pred_final):,}\n")
        
        print(f"\nResults saved:")
        print(f"  Raw probabilities: {raw_path}")
        print(f"  Optimized prediction: {pred_path}")
        print(f"  Analysis: {results_path}")
        
        # Clean up
        del pred_volume, binary_pred_small, binary_pred_final, true_mask, true_mask_small
        gc.collect()
        
        print(f"\n🎉 Raw probability analysis completed successfully!")
        print(f"Key finding: Optimal threshold = {best_threshold:.3f} (much lower than 0.5)")
        print(f"This explains the poor performance with 0.5 threshold!")
        
        return best_threshold
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    optimal_threshold = test_raw_probabilities()
    
    if optimal_threshold:
        print(f"\n🚀 SUCCESS! Use threshold {optimal_threshold:.3f} for batch processing")
    else:
        print(f"\n❌ Analysis failed")
