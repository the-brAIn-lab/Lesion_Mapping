#!/usr/bin/env python3
"""
Batch test all 55 cases in Testing_Split with proper memory management
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
from pathlib import Path
import time
from tqdm import tqdm

# Custom imports
sys.path.append('/mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota')
from models.losses import dice_loss, combined_loss, focal_loss

def setup_tensorflow():
    """Setup TensorFlow like the working verbose test"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print(f"✅ TensorFlow configured - GPU: {len(gpus)}, Mixed Precision: enabled")

def resize_volume(volume, target_shape):
    """Resize volume like training"""
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=1)

def preprocess_image(image_data, target_shape=(192, 224, 176)):
    """Preprocess exactly like training"""
    if image_data.shape != target_shape:
        image_data = resize_volume(image_data, target_shape)
    
    # Normalize like training
    p1, p99 = np.percentile(image_data[image_data > 0], [1, 99])
    image_data = np.clip(image_data, p1, p99)
    image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min() + 1e-8)
    
    return image_data[..., np.newaxis]

def load_model_once():
    """Load model with custom objects"""
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
    print(f"✅ Model loaded: {model.count_params():,} parameters")
    return model

def process_single_case(model, case_name, test_dir, output_dir):
    """Process a single test case"""
    # Paths
    image_path = f"{test_dir}/Images/{case_name}_space-MNI152NLin2009aSym_T1w.nii.gz"
    mask_path = f"{test_dir}/Masks/{case_name}_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz"
    
    if not os.path.exists(image_path) or not os.path.exists(mask_path):
        return None  # Skip missing files
    
    try:
        start_time = time.time()
        
        # 1. Load and preprocess image
        nii_img = nib.load(image_path)
        image_data = nii_img.get_fdata(dtype=np.float32)
        original_shape = image_data.shape
        
        processed = preprocess_image(image_data, (192, 224, 176))
        image_batch = processed[np.newaxis, ...]
        
        # Clean up
        del image_data, processed
        gc.collect()
        
        # 2. Run prediction
        with tf.device('/GPU:0'):
            prediction = model(image_batch, training=False)
        
        # Clean up input
        del image_batch
        gc.collect()
        
        # 3. Convert and postprocess
        pred_np = prediction.numpy()
        del prediction
        gc.collect()
        
        # Extract volume and threshold
        pred_volume = pred_np[0, ..., 0]
        binary_pred_small = (pred_volume > 0.5).astype(np.uint8)
        
        del pred_np, pred_volume
        gc.collect()
        
        # Resize to original space
        if binary_pred_small.shape != original_shape:
            factors = [o / p for o, p in zip(original_shape, binary_pred_small.shape)]
            binary_pred = zoom(binary_pred_small, factors, order=0)
        else:
            binary_pred = binary_pred_small
        
        del binary_pred_small
        gc.collect()
        
        # 4. Calculate metrics
        true_mask = nib.load(mask_path).get_fdata().astype(np.uint8)
        
        intersection = np.sum(binary_pred * true_mask)
        union = np.sum(binary_pred) + np.sum(true_mask)
        dice = (2.0 * intersection) / union if union > 0 else 0.0
        
        # Additional metrics
        tp = np.sum((binary_pred == 1) & (true_mask == 1))
        fp = np.sum((binary_pred == 1) & (true_mask == 0))
        fn = np.sum((binary_pred == 0) & (true_mask == 1))
        tn = np.sum((binary_pred == 0) & (true_mask == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        
        processing_time = time.time() - start_time
        
        # 5. Save prediction
        pred_nii = nib.Nifti1Image(binary_pred, nii_img.affine)
        pred_path = os.path.join(output_dir, "predictions", f"{case_name}_prediction.nii.gz")
        nib.save(pred_nii, pred_path)
        
        # Clean up
        del true_mask, binary_pred, pred_nii
        gc.collect()
        
        # Return results
        return {
            'case': case_name,
            'dice': dice,
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'pred_volume': np.sum(binary_pred),
            'true_volume': np.sum(true_mask),
            'processing_time': processing_time,
            'original_shape': str(original_shape)
        }
        
    except Exception as e:
        print(f"❌ Error processing {case_name}: {e}")
        return None

def main():
    """Main batch processing function"""
    print("=" * 70)
    print("BATCH TESTING - ALL 55 TEST CASES")
    print("=" * 70)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup
    setup_tensorflow()
    
    # Paths
    test_dir = "/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Testing_Split"
    output_dir = "batch_test_results"
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/predictions", exist_ok=True)
    
    # Load model once
    model = load_model_once()
    
    # Get all test cases
    image_dir = os.path.join(test_dir, "Images")
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.nii.gz')])
    
    case_names = [f.split('_space-MNI152NLin2009aSym_T1w.nii.gz')[0] for f in image_files]
    print(f"Found {len(case_names)} test cases")
    
    # Process all cases
    results = []
    failed_cases = []
    
    print(f"\nProcessing cases...")
    for i, case_name in enumerate(tqdm(case_names, desc="Testing")):
        print(f"\n--- Case {i+1}/{len(case_names)}: {case_name} ---")
        
        result = process_single_case(model, case_name, test_dir, output_dir)
        
        if result:
            results.append(result)
            print(f"✅ Dice: {result['dice']:.4f}, Time: {result['processing_time']:.1f}s")
        else:
            failed_cases.append(case_name)
            print(f"❌ Failed")
        
        # Clear session every 10 cases to prevent memory accumulation
        if (i + 1) % 10 == 0:
            tf.keras.backend.clear_session()
            gc.collect()
            print(f"   Cleared session after {i+1} cases")
    
    # Generate results
    print(f"\n{'='*70}")
    print("GENERATING RESULTS")
    print(f"{'='*70}")
    
    if results:
        df = pd.DataFrame(results)
        
        # Save detailed results
        results_path = os.path.join(output_dir, "detailed_results.csv")
        df.to_csv(results_path, index=False)
        print(f"✅ Detailed results: {results_path}")
        
        # Calculate summary statistics
        summary_stats = {
            'total_cases': len(case_names),
            'successful_cases': len(results),
            'failed_cases': len(failed_cases),
            'success_rate': len(results) / len(case_names) * 100,
            'mean_dice': df['dice'].mean(),
            'std_dice': df['dice'].std(),
            'median_dice': df['dice'].median(),
            'min_dice': df['dice'].min(),
            'max_dice': df['dice'].max(),
            'mean_iou': df['iou'].mean(),
            'mean_precision': df['precision'].mean(),
            'mean_recall': df['recall'].mean(),
            'mean_specificity': df['specificity'].mean(),
            'mean_processing_time': df['processing_time'].mean(),
            'total_processing_time': df['processing_time'].sum()
        }
        
        # Save summary
        summary_path = os.path.join(output_dir, "summary_results.txt")
        with open(summary_path, 'w') as f:
            f.write("STROKE LESION SEGMENTATION - BATCH TEST RESULTS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: callbacks/sota_20250616_190015/best_model.h5\n")
            f.write(f"Test Dataset: Atlas 2.0 Testing Split\n")
            f.write(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("SUMMARY STATISTICS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total test cases: {summary_stats['total_cases']}\n")
            f.write(f"Successful: {summary_stats['successful_cases']}\n")
            f.write(f"Failed: {summary_stats['failed_cases']}\n")
            f.write(f"Success rate: {summary_stats['success_rate']:.1f}%\n\n")
            
            f.write("PERFORMANCE METRICS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Mean Dice Score: {summary_stats['mean_dice']:.4f} ± {summary_stats['std_dice']:.4f}\n")
            f.write(f"Median Dice Score: {summary_stats['median_dice']:.4f}\n")
            f.write(f"Dice Range: [{summary_stats['min_dice']:.4f}, {summary_stats['max_dice']:.4f}]\n")
            f.write(f"Mean IoU: {summary_stats['mean_iou']:.4f}\n")
            f.write(f"Mean Precision: {summary_stats['mean_precision']:.4f}\n")
            f.write(f"Mean Recall: {summary_stats['mean_recall']:.4f}\n")
            f.write(f"Mean Specificity: {summary_stats['mean_specificity']:.4f}\n\n")
            
            f.write("TIMING:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Mean processing time: {summary_stats['mean_processing_time']:.1f} seconds/case\n")
            f.write(f"Total processing time: {summary_stats['total_processing_time']/60:.1f} minutes\n\n")
            
            if failed_cases:
                f.write("FAILED CASES:\n")
                f.write("-" * 20 + "\n")
                for case in failed_cases:
                    f.write(f"- {case}\n")
        
        print(f"✅ Summary: {summary_path}")
        
        # Print key results
        print(f"\nKEY RESULTS:")
        print(f"Success rate: {summary_stats['success_rate']:.1f}% ({summary_stats['successful_cases']}/{summary_stats['total_cases']})")
        print(f"Mean Dice Score: {summary_stats['mean_dice']:.4f} ± {summary_stats['std_dice']:.4f}")
        print(f"Dice Range: [{summary_stats['min_dice']:.4f}, {summary_stats['max_dice']:.4f}]")
        print(f"Processing time: {summary_stats['total_processing_time']/60:.1f} minutes total")
        
        # Show best and worst cases
        if len(df) >= 5:
            print(f"\nTOP 5 CASES:")
            top5 = df.nlargest(5, 'dice')[['case', 'dice', 'iou']]
            for _, row in top5.iterrows():
                print(f"  {row['case']}: Dice={row['dice']:.4f}, IoU={row['iou']:.4f}")
            
            print(f"\nWORST 5 CASES:")
            worst5 = df.nsmallest(5, 'dice')[['case', 'dice', 'iou']]
            for _, row in worst5.iterrows():
                print(f"  {row['case']}: Dice={row['dice']:.4f}, IoU={row['iou']:.4f}")
    
    # Final cleanup
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    
    print(f"\n{'='*70}")
    print("🎉 BATCH TESTING COMPLETED!")
    print(f"{'='*70}")
    print(f"Results saved in: {output_dir}")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
