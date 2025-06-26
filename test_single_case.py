#!/usr/bin/env python3
"""
Test a single case to debug memory issues
"""

import os
import sys
import numpy as np
import tensorflow as tf
import nibabel as nib
from scipy.ndimage import zoom

# Limit GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Use only first GPU
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print(f"Using GPU: {gpus[0]}")
    except RuntimeError as e:
        print(e)

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

def combined_loss(y_true, y_pred, smooth=1e-6, focal_gamma=3.0, focal_alpha=0.25):
    """Combined loss function"""
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

print("\n1. Loading model...")
try:
    custom_objects = {
        'dice_coefficient': dice_coefficient,
        'binary_dice_coefficient': dice_coefficient,
        'combined_loss': combined_loss
    }
    
    model = tf.keras.models.load_model(
        "callbacks/sota_20250616_190015/best_model.h5",
        custom_objects=custom_objects,
        compile=False
    )
    print("✓ Model loaded successfully")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Parameters: {model.count_params():,}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    sys.exit(1)

print("\n2. Loading test data...")
test_dir = "/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Testing_Split"
images_dir = os.path.join(test_dir, "Images")

# Get first test image
test_images = [f for f in os.listdir(images_dir) if f.endswith("_T1w.nii.gz")]
if not test_images:
    print("✗ No test images found!")
    sys.exit(1)

first_image = test_images[0]
img_id = first_image.replace("_space-MNI152NLin2009aSym_T1w.nii.gz", "")
print(f"  Testing with: {img_id}")

# Load image
img_path = os.path.join(images_dir, first_image)
img_nii = nib.load(img_path)
img_data = img_nii.get_fdata(dtype=np.float32)
print(f"  Original shape: {img_data.shape}")

print("\n3. Preprocessing...")
# Resize to model input
model_shape = (192, 224, 176)
img_resized = resize_volume(img_data, model_shape)
print(f"  Resized shape: {img_resized.shape}")

# Normalize
p1, p99 = np.percentile(img_resized[img_resized > 0], [1, 99])
img_resized = np.clip(img_resized, p1, p99)
img_resized = (img_resized - img_resized.min()) / (img_resized.max() - img_resized.min() + 1e-8)

# Add batch and channel dimensions
img_input = img_resized[np.newaxis, ..., np.newaxis]
print(f"  Input shape: {img_input.shape}")

print("\n4. Running prediction...")
try:
    pred = model.predict(img_input, verbose=1)
    print(f"✓ Prediction successful!")
    print(f"  Output shape: {pred.shape}")
    print(f"  Value range: [{pred.min():.4f}, {pred.max():.4f}]")
    
    # Count positive predictions
    pred_binary = (pred > 0.5).astype(np.float32)
    positive_voxels = np.sum(pred_binary)
    total_voxels = pred_binary.size
    print(f"  Positive predictions: {positive_voxels:,} / {total_voxels:,} ({positive_voxels/total_voxels*100:.2f}%)")
    
except Exception as e:
    print(f"✗ Prediction failed: {e}")
    import traceback
    traceback.print_exc()

print("\n5. Saving result...")
# Save prediction
pred_squeezed = np.squeeze(pred)
pred_original = resize_volume(pred_squeezed, img_data.shape)
pred_binary = (pred_original > 0.5).astype(np.uint8)

output_dir = "test_single_output"
os.makedirs(output_dir, exist_ok=True)

pred_nii = nib.Nifti1Image(pred_binary, img_nii.affine)
output_path = os.path.join(output_dir, f"{img_id}_prediction.nii.gz")
nib.save(pred_nii, output_path)
print(f"✓ Saved to: {output_path}")

print("\nTest complete!")
