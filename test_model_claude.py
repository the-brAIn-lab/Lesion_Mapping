#!/usr/bin/env python3
"""
Fixed test script with correct input shape (192, 224, 176)
"""

import os
import numpy as np
import tensorflow as tf
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from datetime import datetime
import csv
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH = "callbacks/sota_20250616_190015/best_model.h5"
TEST_DATA_DIR = "/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Testing_Split"
OUTPUT_DIR = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/predictions", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/visualizations", exist_ok=True)

def resize_volume(volume, target_shape):
    """Resize 3D volume"""
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=1)

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Calculate Dice coefficient"""
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred, tf.float32))
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics"""
    # Flatten arrays
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Calculate confusion matrix elements
    tp = np.sum(np.logical_and(y_pred_flat == 1, y_true_flat == 1))
    tn = np.sum(np.logical_and(y_pred_flat == 0, y_true_flat == 0))
    fp = np.sum(np.logical_and(y_pred_flat == 1, y_true_flat == 0))
    fn = np.sum(np.logical_and(y_pred_flat == 0, y_true_flat == 1))
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'jaccard_index': jaccard,
        'specificity': specificity,
        'accuracy': accuracy,
        'true_positive': tp,
        'false_positive': fp,
        'true_negative': tn,
        'false_negative': fn
    }

def combined_loss(y_true, y_pred, smooth=1e-6, focal_gamma=3.0, focal_alpha=0.25):
    """Combined loss function (needed for model loading)"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Dice loss
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice_loss = 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    
    # Focal loss
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    alpha_t = tf.where(tf.equal(y_true, 1), focal_alpha, 1 - focal_alpha)
    focal_loss = -alpha_t * tf.pow(1 - p_t, focal_gamma) * tf.math.log(p_t)
    focal_loss = tf.reduce_mean(focal_loss)
    
    return 0.7 * dice_loss + 0.3 * focal_loss

def load_test_data():
    """Load test images and masks"""
    images_dir = os.path.join(TEST_DATA_DIR, "Images")
    masks_dir = os.path.join(TEST_DATA_DIR, "Masks")
    
    test_pairs = []
    
    for img_file in sorted(os.listdir(images_dir)):
        if img_file.endswith("_T1w.nii.gz"):
            img_id = img_file.replace("_space-MNI152NLin2009aSym_T1w.nii.gz", "")
            mask_file = f"{img_id}_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz"
            
            img_path = os.path.join(images_dir, img_file)
            mask_path = os.path.join(masks_dir, mask_file)
            
            if os.path.exists(mask_path):
                test_pairs.append((img_id, img_path, mask_path))
    
    logger.info(f"Found {len(test_pairs)} test image/mask pairs")
    return test_pairs

def test_model():
    """Test the trained model"""
    
    # Load model
    logger.info(f"Loading model from {MODEL_PATH}")
    
    # Custom objects for loading
    custom_objects = {
        'dice_coefficient': dice_coefficient,
        'binary_dice_coefficient': dice_coefficient,
        'combined_loss': combined_loss
    }
    
    # Load model
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects=custom_objects,
        compile=False
    )
    
    logger.info("Model loaded successfully")
    logger.info(f"Model expects input shape: {model.input_shape}")
    
    # CORRECTED: Use the model's expected shape
    model_shape = (192, 224, 176)  # This is what the model expects!
    
    # Load test data
    test_pairs = load_test_data()
    
    # CSV for results
    csv_path = f"{OUTPUT_DIR}/test_results.csv"
    with open(csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Subject', 'Dice Score', 'Precision', 'Recall', 'F1 Score', 
                           'Jaccard Index', 'Specificity', 'Accuracy'])
    
    all_metrics = []
    
    # Process each test case
    for img_id, img_path, mask_path in test_pairs:
        logger.info(f"Processing {img_id}")
        
        try:
            # Load data
            img_nii = nib.load(img_path)
            mask_nii = nib.load(mask_path)
            
            img_data = img_nii.get_fdata(dtype=np.float32)
            mask_data = mask_nii.get_fdata(dtype=np.float32)
            
            # Store original shape and affine
            original_shape = img_data.shape
            original_affine = img_nii.affine
            
            # Resize to model input shape (192, 224, 176) - CORRECTED!
            img_resized = resize_volume(img_data, model_shape)
            mask_resized = resize_volume(mask_data, model_shape)
            
            # Normalize
            p1, p99 = np.percentile(img_resized[img_resized > 0], [1, 99])
            img_resized = np.clip(img_resized, p1, p99)
            img_resized = (img_resized - img_resized.min()) / (img_resized.max() - img_resized.min() + 1e-8)
            
            # Add batch and channel dimensions
            img_input = img_resized[np.newaxis, ..., np.newaxis]
            
            # Predict
            pred_prob = model.predict(img_input, verbose=0)
            pred_prob = np.squeeze(pred_prob)
            
            # Resize prediction back to original shape
            pred_prob_original = resize_volume(pred_prob, original_shape)
            
            # Apply threshold
            pred_binary = (pred_prob_original > 0.5).astype(np.float32)
            
            # Calculate metrics
            dice_val = dice_coefficient(
                tf.constant(mask_data, dtype=tf.float32),
                tf.constant(pred_prob_original, dtype=tf.float32)
            ).numpy()
            
            metrics = calculate_metrics(mask_data, pred_binary)
            
            # Store results
            all_metrics.append({
                'dice': dice_val,
                **metrics
            })
            
            # Save to CSV
            with open(csv_path, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([
                    img_id,
                    f"{dice_val:.4f}",
                    f"{metrics['precision']:.4f}",
                    f"{metrics['recall']:.4f}",
                    f"{metrics['f1_score']:.4f}",
                    f"{metrics['jaccard_index']:.4f}",
                    f"{metrics['specificity']:.4f}",
                    f"{metrics['accuracy']:.4f}"
                ])
            
            # Save prediction as NIfTI
            pred_nii = nib.Nifti1Image(pred_binary.astype(np.uint8), original_affine)
            nib.save(pred_nii, f"{OUTPUT_DIR}/predictions/{img_id}_prediction.nii.gz")
            
            # Create visualization
            # Middle slices
            ax_idx = original_shape[0] // 2
            sag_idx = original_shape[1] // 2
            cor_idx = original_shape[2] // 2
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'{img_id} - Dice: {dice_val:.4f}')
            
            # Axial views
            axes[0, 0].imshow(img_data[ax_idx, :, :], cmap='gray')
            axes[0, 0].contour(mask_data[ax_idx, :, :], colors='green', levels=[0.5], linewidths=2)
            axes[0, 0].set_title('Axial - Ground Truth')
            axes[0, 0].axis('off')
            
            axes[1, 0].imshow(img_data[ax_idx, :, :], cmap='gray')
            axes[1, 0].contour(pred_binary[ax_idx, :, :], colors='red', levels=[0.5], linewidths=2)
            axes[1, 0].set_title('Axial - Prediction')
            axes[1, 0].axis('off')
            
            # Sagittal views
            axes[0, 1].imshow(img_data[:, sag_idx, :], cmap='gray')
            axes[0, 1].contour(mask_data[:, sag_idx, :], colors='green', levels=[0.5], linewidths=2)
            axes[0, 1].set_title('Sagittal - Ground Truth')
            axes[0, 1].axis('off')
            
            axes[1, 1].imshow(img_data[:, sag_idx, :], cmap='gray')
            axes[1, 1].contour(pred_binary[:, sag_idx, :], colors='red', levels=[0.5], linewidths=2)
            axes[1, 1].set_title('Sagittal - Prediction')
            axes[1, 1].axis('off')
            
            # Coronal views
            axes[0, 2].imshow(img_data[:, :, cor_idx], cmap='gray')
            axes[0, 2].contour(mask_data[:, :, cor_idx], colors='green', levels=[0.5], linewidths=2)
            axes[0, 2].set_title('Coronal - Ground Truth')
            axes[0, 2].axis('off')
            
            axes[1, 2].imshow(img_data[:, :, cor_idx], cmap='gray')
            axes[1, 2].contour(pred_binary[:, :, cor_idx], colors='red', levels=[0.5], linewidths=2)
            axes[1, 2].set_title('Coronal - Prediction')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/visualizations/{img_id}_comparison.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"  Dice: {dice_val:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
            
        except Exception as e:
            logger.error(f"Error processing {img_id}: {e}")
            continue
    
    # Calculate summary statistics
    if all_metrics:
        avg_dice = np.mean([m['dice'] for m in all_metrics])
        avg_precision = np.mean([m['precision'] for m in all_metrics])
        avg_recall = np.mean([m['recall'] for m in all_metrics])
        avg_f1 = np.mean([m['f1_score'] for m in all_metrics])
        
        logger.info("\n=== Test Set Summary ===")
        logger.info(f"Average Dice: {avg_dice:.4f} (+/- {np.std([m['dice'] for m in all_metrics]):.4f})")
        logger.info(f"Average Precision: {avg_precision:.4f}")
        logger.info(f"Average Recall: {avg_recall:.4f}")
        logger.info(f"Average F1 Score: {avg_f1:.4f}")
        
        # Save summary
        with open(f"{OUTPUT_DIR}/summary.txt", 'w') as f:
            f.write("Test Set Evaluation Summary\n")
            f.write("==========================\n\n")
            f.write(f"Number of test cases: {len(all_metrics)}\n")
            f.write(f"Average Dice: {avg_dice:.4f} (+/- {np.std([m['dice'] for m in all_metrics]):.4f})\n")
            f.write(f"Average Precision: {avg_precision:.4f}\n")
            f.write(f"Average Recall: {avg_recall:.4f}\n")
            f.write(f"Average F1 Score: {avg_f1:.4f}\n")

if __name__ == "__main__":
    test_model()
