#!/usr/bin/env python3
import os
import sys
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.ndimage import zoom

print('Testing model on LARGE test lesion...')

# Setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Load model
sys.path.append('/mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota')
from models.losses import dice_loss, combined_loss, focal_loss

def combined_loss_fn(y_true, y_pred, smooth=1e-6, focal_gamma=3.0, focal_alpha=0.25):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice_loss = 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return dice_loss

custom_objects = {'combined_loss': combined_loss_fn, 'dice_loss': dice_loss, 'focal_loss': focal_loss}
model = load_model('callbacks/sota_fixed_20250619_063330/best_model.h5', custom_objects=custom_objects, compile=False)

# Test on large lesion case
case_name = 'sub-r048s016_ses-1'
test_dir = '/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Testing_Split'

image_path = f'{test_dir}/Images/{case_name}_space-MNI152NLin2009aSym_T1w.nii.gz'
mask_path = f'{test_dir}/Masks/{case_name}_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz'

image_data = nib.load(image_path).get_fdata(dtype=np.float32)
true_mask = nib.load(mask_path).get_fdata().astype(np.uint8)

print(f'Case: {case_name}')
print(f'Lesion size: {np.sum(true_mask):,} voxels')

# Quick preprocessing
def resize_volume(volume, target_shape):
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=1)

target_shape = (192, 224, 176)
if image_data.shape != target_shape:
    image_data = resize_volume(image_data, target_shape)

p1, p99 = np.percentile(image_data[image_data > 0], [1, 99])
image_data = np.clip(image_data, p1, p99)
image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min() + 1e-8)
processed = image_data[..., np.newaxis]

# Test original orientation
batch = processed[np.newaxis, ...]
with tf.device('/GPU:0'):
    pred = model(batch, training=False)
pred_vol = pred[0, :, :, :, 0].numpy()

print(f'Prediction max: {pred_vol.max():.6f}')
print(f'Prediction mean: {pred_vol.mean():.8f}')

# Quick dice at different thresholds
for thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
    binary_pred = (pred_vol > thresh).astype(np.uint8)
    intersection = np.sum(binary_pred * true_mask)
    union = np.sum(binary_pred) + np.sum(true_mask)
    dice = (2.0 * intersection) / union if union > 0 else 0.0
    print(f'Threshold {thresh:.1f}: Dice = {dice:.6f}')

if pred_vol.max() > 0.5:
    print('🎉 Model produces confident predictions on large lesions!')
else:
    print('❌ Model still under-confident even on large lesions')
