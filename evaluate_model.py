import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

# Define paths
pred_dir = './Predictions'
truth_dir = '/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Testing_Split/Masks'
log_file = './logs/evaluation_metrics.txt'
expected_shape = (192, 224, 176)  # Match test_sota_model.py

# Function to compute Dice coefficient
def dice_coefficient(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1.)

# Function to resize volume
def resize_volume(volume, target_shape, order=1):
    factors = [t/s for t,s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=order)

# Verify directories exist
if not os.path.exists(pred_dir):
    raise FileNotFoundError(f"Prediction directory not found at {pred_dir}")
if not os.path.exists(truth_dir):
    raise FileNotFoundError(f"Test data directory not found at {truth_dir}")

# Initialize metrics
dice_scores = []

# Create log file
os.makedirs(os.path.dirname(log_file), exist_ok=True)
with open(log_file, 'w') as f:
    f.write("Evaluation Metrics\n================\n")

# Process predicted masks
for pred_file in os.listdir(pred_dir):
    if pred_file.endswith('_pred.nii.gz'):
        base_name = pred_file.replace('_pred.nii.gz', '')
        truth_file = f"{base_name}_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz"
        truth_path = os.path.join(truth_dir, truth_file)
        pred_path = os.path.join(pred_dir, pred_file)

        if not os.path.exists(truth_path):
            print(f"  ⚠️ Skipping {base_name} due to missing ground truth: {truth_path}")
            with open(log_file, 'a') as f:
                f.write(f"  ⚠️ Skipping {base_name} due to missing ground truth: {truth_path}\n")
            continue

        # Load masks
        pred_nii = nib.load(pred_path)
        truth_nii = nib.load(truth_path)
        
        pred_data = pred_nii.get_fdata().astype(np.uint8)
        truth_data = truth_nii.get_fdata().astype(np.float32)

        # Resize ground truth to match predicted shape
        truth_data = resize_volume(truth_data, expected_shape, order=1)
        truth_data = (truth_data > 0.5).astype(np.uint8)
        print(f"{base_name}: Truth resized to {truth_data.shape}, Pred shape={pred_data.shape}")

        # Check affine alignment
        if not np.allclose(pred_nii.affine, truth_nii.affine, atol=1e-5):
            print(f"  ⚠️ Affine mismatch for {base_name}: Pred={pred_nii.affine}, Truth={truth_nii.affine}")
            with open(log_file, 'a') as f:
                f.write(f"  ⚠️ Affine mismatch for {base_name}: Pred={pred_nii.affine}, Truth={truth_nii.affine}\n")

        # Compute Dice
        dice = dice_coefficient(truth_data, pred_data)
        dice_scores.append(dice)
        print(f"{base_name}: Binary Dice={dice:.4f}")
        with open(log_file, 'a') as f:
            f.write(f"{base_name}: Binary Dice={dice:.4f}\n")

# Report average
if dice_scores:
    avg_dice = np.mean(dice_scores)
    print(f"\nAverage Binary Dice: {avg_dice:.4f}")
    with open(log_file, 'a') as f:
        f.write(f"\nAverage Binary Dice: {avg_dice:.4f}\n")
else:
    print("\nNo valid Dice scores computed.")
    with open(log_file, 'a') as f:
        f.write("\nNo valid Dice scores computed.\n")
