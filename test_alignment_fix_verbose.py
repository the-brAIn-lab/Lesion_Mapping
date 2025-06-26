import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import tensorflow as tf
import sys

print("1. Setting up GPU...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(f"   Using GPU: {gpus[0]}")

print("\n2. Loading model (this may take 1-2 minutes)...")
sys.stdout.flush()

model = tf.keras.models.load_model(
    "callbacks/sota_20250616_190015/best_model.h5",
    compile=False
)
print("   Model loaded successfully!")

print("\n3. Loading test case sub-r048s014_ses-1...")
img_id = 'sub-r048s014_ses-1'
img_path = f'/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Testing_Split/Images/{img_id}_space-MNI152NLin2009aSym_T1w.nii.gz'
mask_path = f'/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Testing_Split/Masks/{img_id}_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz'

img_nii = nib.load(img_path)
mask_nii = nib.load(mask_path)
img_data = img_nii.get_fdata(dtype=np.float32)
mask_data = mask_nii.get_fdata(dtype=np.float32)

print(f"   Image shape: {img_data.shape}")
print(f"   Mask voxels: {np.sum(mask_data > 0):,}")

print("\n4. Testing resize methods...")
def resize_volume(volume, target_shape, order=1):
    factors = [t/s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=order)

model_shape = (192, 224, 176)
mask_order0 = resize_volume(mask_data, model_shape, order=0)
print(f"   Order=0 resized mask voxels: {np.sum(mask_order0 > 0):,}")

print("\n5. Running prediction...")
img_resized = resize_volume(img_data, model_shape, order=1)
p1, p99 = np.percentile(img_resized[img_resized > 0], [1, 99])
img_resized = np.clip(img_resized, p1, p99)
img_resized = (img_resized - img_resized.min()) / (img_resized.max() - img_resized.min() + 1e-8)

img_input = img_resized[np.newaxis, ..., np.newaxis]
pred = model.predict(img_input, verbose=1)
pred = np.squeeze(pred)

print("\n6. Post-processing...")
pred_original = resize_volume(pred, img_data.shape, order=0)
pred_binary = (pred_original > 0.5).astype(np.float32)
print(f"   Prediction voxels: {np.sum(pred_binary > 0):,}")

def dice_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-6)

dice = dice_score(mask_data, pred_binary)
print(f"   Dice score: {dice:.4f}")

if np.sum(pred_binary) > 0:
    pred_com = np.array(np.where(pred_binary > 0)).mean(axis=1)
    mask_com = np.array(np.where(mask_data > 0)).mean(axis=1)
    print(f"   Center of mass offset: {pred_com - mask_com}")

print("\n7. Saving result...")
pred_nii = nib.Nifti1Image(pred_binary.astype(np.uint8), img_nii.affine)
nib.save(pred_nii, f"test_alignment_fix_{img_id}_prediction.nii.gz")
print("   Done!")
