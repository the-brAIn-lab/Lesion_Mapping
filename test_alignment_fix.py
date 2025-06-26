import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import tensorflow as tf

# Load model (simplified)
model = tf.keras.models.load_model(
    "callbacks/sota_20250616_190015/best_model.h5",
    compile=False
)

# Test case with misalignment
img_id = 'sub-r048s014_ses-1'
img_path = f'/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Testing_Split/Images/{img_id}_space-MNI152NLin2009aSym_T1w.nii.gz'
mask_path = f'/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Testing_Split/Masks/{img_id}_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz'

# Load data
img_nii = nib.load(img_path)
mask_nii = nib.load(mask_path)
img_data = img_nii.get_fdata(dtype=np.float32)
mask_data = mask_nii.get_fdata(dtype=np.float32)

print(f"Original shapes - Image: {img_data.shape}, Mask: {mask_data.shape}")
print(f"Original mask voxels: {np.sum(mask_data > 0):,}")

# Resize with different orders
def resize_volume(volume, target_shape, order=1):
    factors = [t/s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=order)

model_shape = (192, 224, 176)

# Test order=1 (current) vs order=0 (nearest neighbor)
mask_order1 = resize_volume(mask_data, model_shape, order=1)
mask_order0 = resize_volume(mask_data, model_shape, order=0)

print(f"\nAfter resize to {model_shape}:")
print(f"Order=1 mask voxels: {np.sum(mask_order1 > 0.5):,}")
print(f"Order=0 mask voxels: {np.sum(mask_order0 > 0):,}")

# Center of mass comparison
if np.sum(mask_order1 > 0.5) > 0 and np.sum(mask_order0 > 0) > 0:
    com1 = np.array(np.where(mask_order1 > 0.5)).mean(axis=1)
    com0 = np.array(np.where(mask_order0 > 0)).mean(axis=1)
    print(f"Center of mass shift (order=1 vs order=0): {com1 - com0}")

# Normalize and predict
img_resized = resize_volume(img_data, model_shape, order=1)
p1, p99 = np.percentile(img_resized[img_resized > 0], [1, 99])
img_resized = np.clip(img_resized, p1, p99)
img_resized = (img_resized - img_resized.min()) / (img_resized.max() - img_resized.min() + 1e-8)

# Predict
img_input = img_resized[np.newaxis, ..., np.newaxis]
pred = model.predict(img_input, verbose=0)
pred = np.squeeze(pred)

# Resize prediction back with order=0
pred_original = resize_volume(pred, img_data.shape, order=0)
pred_binary = (pred_original > 0.5).astype(np.float32)

print(f"\nPrediction voxels: {np.sum(pred_binary > 0):,}")

# Calculate dice with order=0 resize
def dice_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-6)

dice = dice_score(mask_data, pred_binary)
print(f"Dice score with order=0 resize: {dice:.4f}")

# Save corrected prediction
pred_nii = nib.Nifti1Image(pred_binary.astype(np.uint8), img_nii.affine)
nib.save(pred_nii, f"test_alignment_fix_{img_id}_prediction.nii.gz")
print(f"\nSaved corrected prediction")
