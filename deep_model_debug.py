#!/usr/bin/env python3
"""
Deep model debugging - investigate what the model is actually predicting
Focus on understanding why both orientations give Dice=0.0
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
        print(f"✅ GPU configured: {gpus[0]}")
    
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print(f"✅ Mixed precision enabled")

def load_fixed_model():
    """Load the fixed model with detailed inspection"""
    model_path = "/mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota/callbacks/sota_fixed_20250619_063330/best_model.h5"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Define custom objects
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
    
    # Inspect model architecture
    print(f"\n📋 MODEL ARCHITECTURE INSPECTION:")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
    print(f"  Number of layers: {len(model.layers)}")
    
    # Check final layer
    final_layer = model.layers[-1]
    print(f"  Final layer: {final_layer.name} ({type(final_layer).__name__})")
    if hasattr(final_layer, 'activation'):
        print(f"  Final activation: {final_layer.activation}")
    
    return model

def resize_volume(volume, target_shape, order=1):
    """Resize volume"""
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=order, mode='constant', cval=0)

def preprocess_image_exact(image_data, target_shape=(192, 224, 176)):
    """Exact preprocessing with detailed logging"""
    print(f"  📊 PREPROCESSING DETAILS:")
    print(f"    Input shape: {image_data.shape}")
    print(f"    Input range: [{image_data.min():.6f}, {image_data.max():.6f}]")
    print(f"    Input mean: {image_data.mean():.6f}")
    print(f"    Non-zero voxels: {np.count_nonzero(image_data):,}")
    
    # Resize if needed
    if image_data.shape != target_shape:
        print(f"    Resizing from {image_data.shape} to {target_shape}")
        image_data = resize_volume(image_data, target_shape, order=1)
    
    # Intensity normalization
    non_zero = image_data[image_data > 0]
    if len(non_zero) > 0:
        p1, p99 = np.percentile(non_zero, [1, 99])
        print(f"    Percentiles: p1={p1:.6f}, p99={p99:.6f}")
        image_data = np.clip(image_data, p1, p99)
        image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min() + 1e-8)
    
    print(f"    Final range: [{image_data.min():.6f}, {image_data.max():.6f}]")
    print(f"    Final mean: {image_data.mean():.6f}")
    
    return image_data[..., np.newaxis]

def analyze_prediction_details(prediction, case_name):
    """Analyze prediction in detail to understand why Dice=0"""
    print(f"\n🔍 DETAILED PREDICTION ANALYSIS")
    print(f"=" * 50)
    
    # Convert to numpy for analysis
    if isinstance(prediction, tf.Tensor):
        pred_array = prediction.numpy()
    else:
        pred_array = prediction
    
    # Remove batch and channel dimensions
    if len(pred_array.shape) == 5:  # (batch, h, w, d, channels)
        pred_volume = pred_array[0, :, :, :, 0]
    elif len(pred_array.shape) == 4:  # (h, w, d, channels)
        pred_volume = pred_array[:, :, :, 0]
    else:  # (h, w, d)
        pred_volume = pred_array
    
    print(f"  Prediction volume shape: {pred_volume.shape}")
    print(f"  Prediction dtype: {pred_volume.dtype}")
    
    # Statistical analysis
    print(f"\n  📈 PREDICTION STATISTICS:")
    print(f"    Min value: {pred_volume.min():.8f}")
    print(f"    Max value: {pred_volume.max():.8f}")
    print(f"    Mean value: {pred_volume.mean():.8f}")
    print(f"    Std value: {pred_volume.std():.8f}")
    print(f"    Median value: {np.median(pred_volume):.8f}")
    
    # Percentile analysis
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\n  📊 PERCENTILE ANALYSIS:")
    for p in percentiles:
        val = np.percentile(pred_volume, p)
        print(f"    {p:2d}th percentile: {val:.8f}")
    
    # Value distribution analysis
    print(f"\n  🎯 VALUE DISTRIBUTION:")
    thresholds = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for thresh in thresholds:
        count = np.sum(pred_volume > thresh)
        percent = count / pred_volume.size * 100
        print(f"    Values > {thresh:.3f}: {count:,} ({percent:.3f}%)")
    
    # Check for common failure modes
    print(f"\n  🚨 FAILURE MODE ANALYSIS:")
    
    # All zeros
    if pred_volume.max() == 0.0:
        print(f"    ❌ ISSUE: All predictions are exactly zero!")
        print(f"       → Model may not be trained properly")
        print(f"       → Check if model weights loaded correctly")
        return "all_zeros"
    
    # All very small values
    elif pred_volume.max() < 1e-6:
        print(f"    ❌ ISSUE: All predictions extremely small (< 1e-6)")
        print(f"       → Model may be under-confident")
        print(f"       → Check final layer activation")
        return "extremely_small"
    
    # All small values
    elif pred_volume.max() < 0.01:
        print(f"    ⚠️ ISSUE: All predictions very small (< 0.01)")
        print(f"       → Model may be very under-confident")
        print(f"       → May need lower thresholds")
        return "very_small"
    
    # Reasonable range but no confident predictions
    elif pred_volume.max() < 0.5:
        print(f"    ⚠️ ISSUE: No confident predictions (max < 0.5)")
        print(f"       → Model uncertain, may need threshold tuning")
        return "low_confidence"
    
    # Reasonable predictions
    else:
        print(f"    ✅ Predictions in reasonable range")
        return "reasonable"

def test_with_different_thresholds(pred_volume, true_mask, original_shape):
    """Test with very low thresholds to see if anything works"""
    print(f"\n🎯 THRESHOLD TESTING")
    print(f"=" * 30)
    
    # Resize prediction back to original shape
    if pred_volume.shape != original_shape:
        factors = [o / p for o, p in zip(original_shape, pred_volume.shape)]
        pred_resized = resize_volume(pred_volume, original_shape, order=1)
    else:
        pred_resized = pred_volume
    
    # Test very low thresholds
    thresholds = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    best_dice = 0.0
    best_threshold = 0.5
    results = []
    
    for threshold in thresholds:
        binary_pred = (pred_resized > threshold).astype(np.uint8)
        
        # Calculate metrics
        intersection = np.sum(binary_pred * true_mask)
        pred_voxels = np.sum(binary_pred)
        true_voxels = np.sum(true_mask)
        union = pred_voxels + true_voxels
        
        dice = (2.0 * intersection) / union if union > 0 else 0.0
        
        if dice > best_dice:
            best_dice = dice
            best_threshold = threshold
        
        results.append({
            'threshold': threshold,
            'dice': dice,
            'intersection': intersection,
            'pred_voxels': pred_voxels,
            'true_voxels': true_voxels
        })
        
        print(f"  Threshold {threshold:6.4f}: Dice={dice:.6f}, Pred={pred_voxels:,}, True={true_voxels:,}, Intersect={intersection:,}")
    
    print(f"\n  🏆 Best result: Threshold={best_threshold:.4f}, Dice={best_dice:.6f}")
    
    return best_dice, best_threshold, results

def deep_analyze_case(case_name, test_dir="/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Testing_Split"):
    """Deep analysis of a single case to understand the failure"""
    print(f"\n{'='*80}")
    print(f"DEEP MODEL DEBUG ANALYSIS: {case_name}")
    print(f"{'='*80}")
    
    # File paths
    image_path = f"{test_dir}/Images/{case_name}_space-MNI152NLin2009aSym_T1w.nii.gz"
    mask_path = f"{test_dir}/Masks/{case_name}_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz"
    
    # Load data
    print(f"📂 Loading data...")
    nii_img = nib.load(image_path)
    image_data = nii_img.get_fdata(dtype=np.float32)
    true_mask = nib.load(mask_path).get_fdata().astype(np.uint8)
    original_shape = image_data.shape
    
    print(f"  Original shape: {original_shape}")
    print(f"  True lesion voxels: {np.sum(true_mask):,}")
    
    # Setup and load model
    setup_tensorflow()
    model = load_fixed_model()
    
    # Preprocess
    print(f"\n🔧 PREPROCESSING:")
    processed = preprocess_image_exact(image_data.copy())
    batch = processed[np.newaxis, ...]
    
    print(f"  Final batch shape: {batch.shape}")
    
    # Run prediction with detailed monitoring
    print(f"\n🚀 RUNNING PREDICTION:")
    start_time = time.time()
    
    with tf.device('/GPU:0'):
        prediction = model(batch, training=False)
    
    pred_time = time.time() - start_time
    print(f"  Prediction completed in {pred_time:.2f} seconds")
    print(f"  Prediction tensor shape: {prediction.shape}")
    print(f"  Prediction tensor dtype: {prediction.dtype}")
    
    # Analyze the raw prediction
    failure_mode = analyze_prediction_details(prediction, case_name)
    
    # Convert to volume for threshold testing
    pred_volume = prediction[0, :, :, :, 0].numpy()
    
    # Test thresholds
    best_dice, best_threshold, threshold_results = test_with_different_thresholds(
        pred_volume, true_mask, original_shape
    )
    
    # Final analysis
    print(f"\n🎯 DIAGNOSTIC CONCLUSION")
    print(f"=" * 40)
    print(f"Failure mode: {failure_mode}")
    print(f"Best achievable Dice: {best_dice:.6f}")
    print(f"Best threshold: {best_threshold:.6f}")
    
    # Recommendations based on failure mode
    if failure_mode == "all_zeros":
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"  1. Check if model file is corrupted")
        print(f"  2. Verify model architecture matches training")
        print(f"  3. Test with a known good input")
        print(f"  4. Check if all layers are properly loaded")
        
    elif failure_mode in ["extremely_small", "very_small"]:
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"  1. Model seems under-confident - check final activation")
        print(f"  2. Try different normalization strategies")
        print(f"  3. Check training/inference preprocessing mismatch")
        print(f"  4. Model may need retraining with better parameters")
        
    elif failure_mode == "low_confidence":
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"  1. Model works but is uncertain")
        print(f"  2. Try post-processing with lower thresholds")
        print(f"  3. Check if training data had similar characteristics")
        print(f"  4. Consider ensemble methods or model calibration")
        
    else:
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"  1. Model predictions look reasonable")
        print(f"  2. Issue may be in post-processing or evaluation")
        print(f"  3. Check threshold selection strategy")
    
    # Clean up
    del model, image_data, true_mask, processed, batch, prediction, pred_volume
    tf.keras.backend.clear_session()
    gc.collect()
    
    return {
        'case': case_name,
        'failure_mode': failure_mode,
        'best_dice': best_dice,
        'best_threshold': best_threshold,
        'prediction_time': pred_time
    }

def main():
    """Main deep debugging function"""
    print(f"🔬 DEEP MODEL DEBUG ANALYSIS")
    print(f"=" * 80)
    print(f"Purpose: Understand why model gives Dice=0.0 for both orientations")
    print(f"Strategy: Detailed prediction analysis and failure mode detection")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test case
    test_case = "sub-r048s014_ses-1"
    
    try:
        result = deep_analyze_case(test_case)
        
        print(f"\n🏁 FINAL DIAGNOSTIC RESULT")
        print(f"=" * 60)
        print(f"Test case: {result['case']}")
        print(f"Failure mode: {result['failure_mode']}")
        print(f"Best achievable Dice: {result['best_dice']:.6f}")
        print(f"Best threshold: {result['best_threshold']:.6f}")
        
        if result['best_dice'] > 0.1:
            print(f"\n✅ PARTIAL SUCCESS: Model can achieve some performance")
            print(f"   → Focus on threshold optimization and post-processing")
            return True
        else:
            print(f"\n❌ FUNDAMENTAL ISSUE: Model not working properly")
            print(f"   → Need to investigate model training/loading issues")
            return False
            
    except Exception as e:
        print(f"\n❌ DEEP DEBUG FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        print(f"\nEnd time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
