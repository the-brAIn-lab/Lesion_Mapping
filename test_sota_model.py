import os
import logging
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import tensorflow as tf
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/test_sota.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SOTATester:
    def __init__(self, model_path, test_img_dir, test_mask_dir, output_dir, target_shape=(192, 224, 176)):
        """Initialize tester with paths and parameters."""
        self.model_path = Path(model_path)
        self.test_img_dir = Path(test_img_dir)
        self.test_mask_dir = Path(test_mask_dir)
        self.output_dir = Path(output_dir)
        self.target_shape = target_shape
        self.thresholds = [0.1, 0.3, 0.5]
        self.model = self.load_model()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.key_sample = 'sub-r048s014_ses-1'  # Track key sample
        self.key_results = []  # Store results for key sample

    def load_model(self):
        """Load the trained model."""
        try:
            model = tf.keras.models.load_model(self.model_path, compile=False)
            logger.info(f"Loaded model from {self.model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def resize_volume(self, volume, order=1):
        """Resize volume to target shape."""
        factors = [t / s for t, s in zip(self.target_shape, volume.shape)]
        return zoom(volume, factors, order=order)

    def dice_coefficient(self, y_true, y_pred, smooth=1e-6):
        """Compute Dice coefficient."""
        y_true_f = y_true.flatten().astype(np.float32)
        y_pred_f = y_pred.flatten().astype(np.float32)
        intersection = np.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

    def normalize_image(self, img_data):
        """Apply 1-99 percentile normalization."""
        p1, p99 = np.percentile(img_data[img_data > 0], [1, 99])
        img_data = np.clip(img_data, p1, p99)
        return (img_data - img_data.min()) / (img_data.max() - img_data.min() + 1e-8)

    def process_sample(self, img_path, mask_path, flip_test=False, mask_order=1):
        """Process a single test sample."""
        base_name = img_path.stem.replace('_space-MNI152NLin2009aSym_T1w', '')
        logger.info(f"Processing {base_name}")

        try:
            # Load data
            img_nii = nib.load(img_path)
            mask_nii = nib.load(mask_path)
            img_data = img_nii.get_fdata()
            mask_data = mask_nii.get_fdata()

            # Log original shapes and lesion voxels
            lesion_voxels = int(np.sum(mask_data))
            logger.info(f"{base_name}: Loaded img_shape={img_data.shape}, mask_shape={mask_data.shape}, lesion_voxels={lesion_voxels}")

            # Resize and normalize
            img_data = self.resize_volume(img_data, order=1)
            mask_data = self.resize_volume(mask_data, order=mask_order)
            mask_data = (mask_data > 0.5).astype(np.uint8) if mask_order == 1 else (mask_data > 0).astype(np.uint8)
            img_data = self.normalize_image(img_data)

            # Log post-resize lesion voxels
            post_lesion_voxels = int(np.sum(mask_data))
            logger.info(f"{base_name}: Post-resize lesion_voxels={post_lesion_voxels}")

            # Prepare input
            img_input = img_data[..., np.newaxis]
            pred_mask_float = self.model.predict(img_input[np.newaxis, ...], verbose=0)[0, ..., 0]

            # Log prediction stats
            logger.info(f"{base_name}: Max pred probability={np.max(pred_mask_float):.4f}, Min pred probability={np.min(pred_mask_float):.4f}")

            # Test flipped prediction
            if flip_test:
                img_flipped = np.flip(img_input, axis=1)
                pred_mask_float_flipped = self.model.predict(img_flipped[np.newaxis, ...], verbose=0)[0, ..., 0]
                pred_mask_float_flipped = np.flip(pred_mask_float_flipped, axis=1)
                pred_mask_binary_flipped = (pred_mask_float_flipped > 0.1).astype(np.uint8)
                flipped_dice = self.dice_coefficient(mask_data, pred_mask_binary_flipped)
                logger.info(f"{base_name}: Flipped Binary Dice={flipped_dice:.4f}")
                if base_name == self.key_sample:
                    self.key_results.append(f"Flipped Binary Dice={flipped_dice:.4f}")

            # Compute Dice scores for multiple thresholds
            results = []
            for thresh in self.thresholds:
                pred_mask_binary = (pred_mask_float > thresh).astype(np.uint8)
                soft_dice = self.dice_coefficient(mask_data, pred_mask_float)
                binary_dice = self.dice_coefficient(mask_data, pred_mask_binary)
                logger.info(f"{base_name}: Threshold={thresh}, Soft Dice={soft_dice:.4f}, Binary Dice={binary_dice:.4f}")
                results.append(f"Threshold={thresh}, Soft Dice={soft_dice:.4f}, Binary Dice={binary_dice:.4f}")

            # Save results for key sample
            if base_name == self.key_sample:
                self.key_results.extend(results)
                self.key_results.append(f"Original lesion_voxels={lesion_voxels}, Post-resize lesion_voxels={post_lesion_voxels}")
                self.key_results.append(f"Max pred probability={np.max(pred_mask_float):.4f}, Min pred probability={np.min(pred_mask_float):.4f}")

            # Save prediction
            pred_mask_binary = (pred_mask_float > 0.1).astype(np.uint8)
            pred_nii = nib.Nifti1Image(pred_mask_binary, img_nii.affine)
            output_path = self.output_dir / f"{base_name}_pred.nii.gz"
            nib.save(pred_nii, output_path)
            logger.info(f"Saved predicted mask to {output_path}")

        except Exception as e:
            logger.error(f"Error processing {base_name}: {e}")

    def run(self, flip_test=False, mask_order=1):
        """Run testing on all samples."""
        img_files = sorted(self.test_img_dir.glob('*_T1w.nii.gz'))
        for img_path in img_files:
            base_name = img_path.stem.replace('_space-MNI152NLin2009aSym_T1w', '')
            mask_base_name = base_name.replace('.nii', '')
            mask_path = self.test_mask_dir / f"{mask_base_name}_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz"
            if mask_path.exists():
                self.process_sample(img_path, mask_path, flip_test=flip_test, mask_order=mask_order)
            else:
                logger.warning(f"Skipping {base_name}: Mask not found at {mask_path}")

    def log_key_results(self):
        """Log results for key sample."""
        if self.key_results:
            logger.info(f"Summary for {self.key_sample}:")
            for result in self.key_results:
                logger.info(f"  {result}")

def main():
    tester = SOTATester(
        model_path='/mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota/models/sota_final_20250616_190015.h5',
        test_img_dir='/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Testing_Split/Images',
        test_mask_dir='/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Testing_Split/Masks',
        output_dir='./Predictions'
    )
    # Run with default settings
    tester.run(flip_test=False, mask_order=1)
    tester.log_key_results()
    # Test with flipped predictions
    logger.info("Running flip test...")
    tester.run(flip_test=True, mask_order=1)
    tester.log_key_results()
    # Test with order=0 mask resizing
    logger.info("Running order=0 mask resizing test...")
    tester.run(flip_test=False, mask_order=0)
    tester.log_key_results()

if __name__ == "__main__":
    main()
