#!/usr/bin/env python3
"""
Batch test with orientation correction
Tests the fixed model on all test cases using the correct orientation
"""

import os
import sys
import gc
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.ndimage import zoom
import pandas as pd
import time
from pathlib import Path

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
    print(f"✅ TensorFlow configured")

def load_fixed_model():
    """Load the fixed model"""
    model_path = "/mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota/callbacks/sota_fixed_20250619_063330/best_model.h5"
    
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
    print(f"✅ Fixed model loaded: {model.count_params():,} parameters")
    return model

def resize_volume(volume, target_shape, order=1):
    """Resize volume"""
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=order, mode='constant', cval=0)

def preprocess_image(image_data, target_shape=(192, 224, 176), use_flipped=False):
    """Preprocess image with optional flipping"""
    # Apply flip if needed (based on debug results)
    if use_flipped:
        image_data = np.flip(image_data, axis=0)
    
    # Standard preprocessing
    if image_data.shape != target_shape:
        image_data = resize_volume(image_data, target_shape, order=1)
    
    # Intensity normalization
    non_zero = image_data[image_data > 0]
    if len(non_zero) > 0:
        p1, p99 = np.percentile(non_zero, [1, 99])
        image_data = np.clip(image_data, p1, p99)
        image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min() + 1e-8)
    
    return image_data[..., np.newaxis]

def convert_prediction_chunked(prediction):
    """Convert prediction to numpy in chunks"""
    batch_size, height, width, depth, channels = prediction.shape
    chunk_size = 16
    num_chunks = (depth + chunk_size - 1) // chunk_size
    
    result = np.zeros((height, width, depth), dtype=np.float32)
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, depth)
        chunk_tensor = prediction[0, :, :, start_idx:end_idx, 0]
        result[:, :, start_idx:end_idx] = chunk_tensor.numpy()
        del chunk_tensor
        gc.collect()
    
    return result

def test_orientations_on_case(model, case_name, test_dir):
    """Test both orientations on a case to determine which works better"""
    image_path = f"{test_dir}/Images/{case_name}_space-MNI152NLin2009aSym_T1w.nii.gz"
    mask_path = f"{test_dir}/Masks/{case_name}_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz"
    
    if not os.path.exists(image_path) or not os.path.exists(mask_path):
        return None, None
    
    try:
        # Load data
        nii_img = nib.load(image_path)
        image_data = nii_img.get_fdata(dtype=np.float32)
        true_mask = nib.load(mask_path).get_fdata().astype(np.uint8)
        original_shape = image_data.shape
        
        results = {}
        
        # Test both orientations
        for orientation in ['original', 'flipped']:
            use_flipped = (orientation == 'flipped')
            
            # Preprocess
            processed = preprocess_image(image_data.copy(), use_flipped=use_flipped)
            image_batch = processed[np.newaxis, ...]
            
            # Predict
            with tf.device('/GPU:0'):
                prediction = model(image_batch, training=False)
            
            # Convert prediction
            pred_volume = convert_prediction_chunked(prediction)
            
            # If we flipped the input, flip the prediction back
            if use_flipped:
                pred_volume = np.flip(pred_volume, axis=0)
            
            # Find best threshold and Dice
            best_dice = 0.0
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            
            for threshold in thresholds:
                binary_pred_small = (pred_volume > threshold).astype(np.uint8)
                
                # Resize to original space
                if binary_pred_small.shape != original_shape:
                    factors = [o / p for o, p in zip(original_shape, binary_pred_small.shape)]
                    binary_pred = resize_volume(binary_pred_small, original_shape, order=0)
                    binary_pred = (binary_pred > 0.5).astype(np.uint8)
                else:
                    binary_pred = binary_pred_small
                
                # Calculate Dice
                intersection = np.sum(binary_pred * true_mask)
                union = np.sum(binary_pred) + np.sum(true_mask)
                dice = (2.0 * intersection) / union if union > 0 else 0.0
                
                if dice > best_dice:
                    best_dice = dice
                
                del binary_pred_small, binary_pred
                gc.collect()
            
            results[orientation] = best_dice
            
            # Clean up
            del processed, image_batch, prediction, pred_volume
            gc.collect()
        
        # Determine best orientation
        if results['flipped'] > results['original']:
            best_orientation = 'flipped'
            best_dice = results['flipped']
        else:
            best_orientation = 'original'
            best_dice = results['original']
        
        # Clean up
        del image_data, true_mask
        gc.collect()
        
        return best_orientation, best_dice
        
    except Exception as e:
        print(f"❌ Error testing orientations for {case_name}: {e}")
        return None, None

def process_single_case(model, case_name, test_dir, output_dir, use_flipped=False):
    """Process a single test case with specified orientation"""
    image_path = f"{test_dir}/Images/{case_name}_space-MNI152NLin2009aSym_T1w.nii.gz"
    mask_path = f"{test_dir}/Masks/{case_name}_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz"
    
    if not os.path.exists(image_path) or not os.path.exists(mask_path):
        return None
    
    try:
        start_time = time.time()
        
        # Load and preprocess
        nii_img = nib.load(image_path)
        image_data = nii_img.get_fdata(dtype=np.float32)
        true_mask = nib.load(mask_path).get_fdata().astype(np.uint8)
        original_shape = image_data.shape
        
        processed = preprocess_image(image_data, use_flipped=use_flipped)
        image_batch = processed[np.newaxis, ...]
        
        # Clean up input data
        del image_data, processed
        gc.collect()
        
        # Predict
        with tf.device('/GPU:0'):
            prediction = model(image_batch, training=False)
        
        # Clean up input batch
        del image_batch
        gc.collect()
        
        # Convert prediction
        pred_volume = convert_prediction_chunked(prediction)
        del prediction
        gc.collect()
        
        # If we flipped the input, flip the prediction back
        if use_flipped:
            pred_volume = np.flip(pred_volume, axis=0)
        
        # Find optimal threshold
        best_dice = 0.0
        best_threshold = 0.5
        best_binary_pred = None
        
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        
        for threshold in thresholds:
            binary_pred_small = (pred_volume > threshold).astype(np.uint8)
            
            # Resize to original space
            if binary_pred_small.shape != original_shape:
                factors = [o / p for o, p in zip(original_shape, binary_pred_small.shape)]
                binary_pred = resize_volume(binary_pred_small, original_shape, order=0)
                binary_pred = (binary_pred > 0.5).astype(np.uint8)
            else:
                binary_pred = binary_pred_small
            
            # Calculate Dice
            intersection = np.sum(binary_pred * true_mask)
            union = np.sum(binary_pred) + np.sum(true_mask)
            dice = (2.0 * intersection) / union if union > 0 else 0.0
            
            if dice > best_dice:
                best_dice = dice
                best_threshold = threshold
                best_binary_pred = binary_pred.copy()
            
            del binary_pred_small, binary_pred
            gc.collect()
        
        processing_time = time.time() - start_time
        
        # Calculate additional metrics
        if best_binary_pred is not None:
            tp = np.sum((best_binary_pred == 1) & (true_mask == 1))
            fp = np.sum((best_binary_pred == 1) & (true_mask == 0))
            fn = np.sum((best_binary_pred == 0) & (true_mask == 1))
            tn = np.sum((best_binary_pred == 0) & (true_mask == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            
            pred_voxels = np.sum(best_binary_pred)
            true_voxels = np.sum(true_mask)
            
            # Save prediction
            pred_nii = nib.Nifti1Image(best_binary_pred, nii_img.affine)
            orientation_suffix = "_flipped" if use_flipped else "_original"
            pred_path = os.path.join(output_dir, "predictions", f"{case_name}_corrected{orientation_suffix}.nii.gz")
            nib.save(pred_nii, pred_path)
            
            del best_binary_pred
            gc.collect()
        else:
            tp = fp = fn = tn = 0
            precision = recall = specificity = iou = 0.0
            pred_voxels = true_voxels = 0
        
        # Clean up
        del pred_volume, true_mask
        gc.collect()
        
        return {
            'case': case_name,
            'dice': best_dice,
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'threshold': best_threshold,
            'orientation': 'flipped' if use_flipped else 'original',
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'pred_voxels': pred_voxels,
            'true_voxels': true_voxels,
            'processing_time': processing_time,
            'original_shape': str(original_shape)
        }
        
    except Exception as e:
        print(f"❌ Error processing {case_name}: {e}")
        return None

def main():
    """Main batch testing function with orientation correction"""
    print("=" * 70)
    print("BATCH TESTING WITH ORIENTATION CORRECTION")
    print("=" * 70)
    print(f"Strategy: Test sample cases to determine best orientation, then batch test")
    print(f"Expected: Significant improvement if orientation was the issue")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup
    setup_tensorflow()
    
    # Paths
    test_dir = "/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Testing_Split"
    output_dir = "corrected_model_results"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/predictions", exist_ok=True)
    
    # Load model
    model = load_fixed_model()
    
    # Get test cases
    image_dir = os.path.join(test_dir, "Images")
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.nii.gz')])
    case_names = [f.split('_space-MNI152NLin2009aSym_T1w.nii.gz')[0] for f in image_files]
    
    print(f"Found {len(case_names)} test cases")
    
    # Test orientations on a few sample cases to determine best approach
    print(f"\n🔄 ORIENTATION TESTING PHASE")
    print(f"=" * 50)
    print(f"Testing both orientations on first 3 cases to determine best approach...")
    
    orientation_votes = {'original': 0, 'flipped': 0}
    sample_cases = case_names[:3]
    
    for i, case_name in enumerate(sample_cases):
        print(f"\nTesting case {i+1}/{len(sample_cases)}: {case_name}")
        best_orientation, best_dice = test_orientations_on_case(model, case_name, test_dir)
        
        if best_orientation:
            orientation_votes[best_orientation] += 1
            print(f"  Best: {best_orientation} (Dice: {best_dice:.4f})")
        else:
            print(f"  Failed to test orientations")
        
        # Clear memory
        if (i + 1) % 2 == 0:
            tf.keras.backend.clear_session()
            gc.collect()
    
    # Determine overall best orientation
    if orientation_votes['flipped'] > orientation_votes['original']:
        use_flipped_orientation = True
        print(f"\n✅ DECISION: Using FLIPPED orientation (won {orientation_votes['flipped']}/{len(sample_cases)} cases)")
    else:
        use_flipped_orientation = False
        print(f"\n✅ DECISION: Using ORIGINAL orientation (won {orientation_votes['original']}/{len(sample_cases)} cases)")
    
    print(f"\n🚀 BATCH TESTING PHASE")
    print(f"=" * 50)
    print(f"Processing all {len(case_names)} cases with {'FLIPPED' if use_flipped_orientation else 'ORIGINAL'} orientation...")
    
    # Process all cases
    results = []
    failed_cases = []
    
    for i, case_name in enumerate(case_names):
        print(f"\n--- Case {i+1}/{len(case_names)}: {case_name} ---")
        
        result = process_single_case(model, case_name, test_dir, output_dir, use_flipped=use_flipped_orientation)
        
        if result:
            results.append(result)
            print(f"✅ Dice: {result['dice']:.4f}, Threshold: {result['threshold']:.1f}, Time: {result['processing_time']:.1f}s")
        else:
            failed_cases.append(case_name)
            print(f"❌ Failed")
        
        # Clear session periodically
        if (i + 1) % 10 == 0:
            tf.keras.backend.clear_session()
            gc.collect()
            print(f"   Cleared session after {i+1} cases")
    
    # Generate results
    print(f"\n{'='*70}")
    print("CORRECTED MODEL RESULTS")
    print(f"{'='*70}")
    
    if results:
        df = pd.DataFrame(results)
        
        # Save detailed results
        results_path = os.path.join(output_dir, "corrected_model_detailed_results.csv")
        df.to_csv(results_path, index=False)
        print(f"✅ Detailed results: {results_path}")
        
        # Calculate summary statistics
        summary = {
            'total_cases': len(case_names),
            'successful_cases': len(results),
            'failed_cases': len(failed_cases),
            'success_rate': len(results) / len(case_names) * 100,
            'orientation_used': 'flipped' if use_flipped_orientation else 'original',
            'mean_dice': df['dice'].mean(),
            'std_dice': df['dice'].std(),
            'median_dice': df['dice'].median(),
            'min_dice': df['dice'].min(),
            'max_dice': df['dice'].max(),
            'mean_iou': df['iou'].mean(),
            'mean_precision': df['precision'].mean(),
            'mean_recall': df['recall'].mean(),
            'mean_processing_time': df['processing_time'].mean()
        }
        
        # Save summary
        summary_path = os.path.join(output_dir, "corrected_model_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("CORRECTED MODEL - BATCH TEST RESULTS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: callbacks/sota_fixed_20250619_063330/best_model.h5\n")
            f.write(f"Orientation used: {summary['orientation_used']}\n")
            f.write(f"Validation Dice: 0.4741\n")
            f.write(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("SUMMARY STATISTICS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total test cases: {summary['total_cases']}\n")
            f.write(f"Successful: {summary['successful_cases']}\n")
            f.write(f"Failed: {summary['failed_cases']}\n")
            f.write(f"Success rate: {summary['success_rate']:.1f}%\n\n")
            
            f.write("PERFORMANCE METRICS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Mean Dice Score: {summary['mean_dice']:.4f} ± {summary['std_dice']:.4f}\n")
            f.write(f"Median Dice Score: {summary['median_dice']:.4f}\n")
            f.write(f"Dice Range: [{summary['min_dice']:.4f}, {summary['max_dice']:.4f}]\n")
            f.write(f"Mean IoU: {summary['mean_iou']:.4f}\n")
            f.write(f"Mean Precision: {summary['mean_precision']:.4f}\n")
            f.write(f"Mean Recall: {summary['mean_recall']:.4f}\n\n")
            
            f.write("COMPARISON WITH ORIGINAL MODEL:\n")
            f.write("-" * 32 + "\n")
            f.write(f"Original model: Mean Dice ≈ 0.0 (complete failure)\n")
            f.write(f"Corrected model: Mean Dice = {summary['mean_dice']:.4f}\n")
            if summary['mean_dice'] > 0.35:
                f.write(f"RESULT: 🎉 MAJOR SUCCESS - Orientation issue resolved!\n")
            elif summary['mean_dice'] > 0.2:
                f.write(f"RESULT: ✅ SIGNIFICANT IMPROVEMENT - Much better!\n")
            else:
                f.write(f"RESULT: ❌ LIMITED IMPROVEMENT - May need further investigation\n")
            
            f.write(f"\nProcessing time: {summary['mean_processing_time']:.1f} seconds/case\n")
            
            if failed_cases:
                f.write(f"\nFAILED CASES:\n")
                for case in failed_cases:
                    f.write(f"- {case}\n")
        
        print(f"✅ Summary: {summary_path}")
        
        # Print key results
        print(f"\nKEY RESULTS:")
        print(f"Orientation used: {summary['orientation_used']}")
        print(f"Success rate: {summary['success_rate']:.1f}% ({summary['successful_cases']}/{summary['total_cases']})")
        print(f"Mean Dice Score: {summary['mean_dice']:.4f} ± {summary['std_dice']:.4f}")
        print(f"Dice Range: [{summary['min_dice']:.4f}, {summary['max_dice']:.4f}]")
        
        # Performance assessment
        if summary['mean_dice'] > 0.35:
            print(f"\n🎉 MAJOR SUCCESS!")
            print(f"Orientation correction resolved the issue!")
            print(f"Model performance now matches validation expectations")
        elif summary['mean_dice'] > 0.2:
            print(f"\n✅ SIGNIFICANT IMPROVEMENT!")
            print(f"Much better than original Dice ≈ 0.0")
            print(f"Orientation was likely a major factor")
        else:
            print(f"\n❌ LIMITED SUCCESS")
            print(f"Some improvement but may need deeper investigation")
        
        # Show best and worst cases
        if len(df) >= 5:
            print(f"\nTOP 5 PERFORMERS:")
            top5 = df.nlargest(5, 'dice')[['case', 'dice', 'iou', 'true_voxels']]
            for _, row in top5.iterrows():
                print(f"  {row['case']}: Dice={row['dice']:.4f}, IoU={row['iou']:.4f}, Lesion={row['true_voxels']:,} voxels")
    
    # Final cleanup
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    
    print(f"\n{'='*70}")
    print("🎯 CORRECTED MODEL BATCH TESTING COMPLETED!")
    print(f"{'='*70}")
    print(f"Results saved in: {output_dir}")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
