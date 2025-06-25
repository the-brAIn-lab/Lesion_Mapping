#!/bin/bash
#SBATCH --job-name=stroke_atlas_fixed
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:a4500:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err

echo "🚀 ATLAS 2.0 Stroke Segmentation Training - FIXED VERSION"
echo "======================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"

# Navigate to project
cd /mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota || {
    echo "❌ Project directory not found"
    exit 1
}

# Load proven modules
echo "🔗 Loading proven working modules..."
module load gcc/9.3.0-5wu3
module load cuda/12.6.3-ziu7
module list

# Set proven environment variables
echo "⚙️ Setting proven environment variables..."
export TF_XLA_FLAGS="--tf_xla_enable_xla_devices=false"
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2

# Find the correct conda path
CONDA_PATHS=(
    "/opt/anaconda3/etc/profile.d/conda.sh"
    "/opt/conda/etc/profile.d/conda.sh"
    "/usr/local/anaconda3/etc/profile.d/conda.sh"
    "$HOME/anaconda3/etc/profile.d/conda.sh"
    "$HOME/miniconda3/etc/profile.d/conda.sh"
)

CONDA_FOUND=false
for conda_path in "${CONDA_PATHS[@]}"; do
    if [ -f "$conda_path" ]; then
        echo "✅ Found conda at: $conda_path"
        source "$conda_path"
        CONDA_FOUND=true
        break
    fi
done

if [ "$CONDA_FOUND" = false ]; then
    echo "⚠️ Conda not found in standard paths, trying conda command directly"
    eval "$(conda shell.bash hook)" 2>/dev/null || {
        echo "❌ Could not initialize conda"
        exit 1
    }
fi

# Activate working environment
echo "🐍 Activating tf215_env..."
conda activate tf215_env || {
    echo "❌ Could not activate tf215_env"
    exit 1
}

echo "Environment: $CONDA_DEFAULT_ENV"
echo "Python: $(python --version)"

# Set critical library path
echo "🔧 Setting critical library path..."
export LD_LIBRARY_PATH="/mnt/beegfs/hellgate/home/rb194958e/.conda/envs/tf215_env/lib:$LD_LIBRARY_PATH"

# Install required packages
echo "📦 Installing required packages..."
pip install nibabel scipy -q

# Verify GPU detection
echo "🔬 Verifying GPU detection..."
python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs detected: {len(gpus)}')
if len(gpus) >= 1:
    print('✅ GPU ready for training')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print('❌ No GPUs detected')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ GPU verification failed"
    exit 1
fi

# Update the data loader with fixed version
echo "🔧 Updating data loader..."
cat > real_data_loader.py << 'EOF'
#!/usr/bin/env python3
"""
Fixed ATLAS 2.0 Data Loader for 3D Volume Resizing
"""

import tensorflow as tf
import nibabel as nib
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, List, Generator
from scipy.ndimage import zoom

logger = logging.getLogger(__name__)

class RealAtlasDataLoader:
    def __init__(self, data_dir: str, target_shape: Tuple[int, int, int] = (192, 224, 176)):
        self.data_dir = Path(data_dir)
        self.target_shape = target_shape
        self.image_dir = self.data_dir / 'Images'
        self.mask_dir = self.data_dir / 'Masks'
        self.file_pairs = self._get_file_pairs()
        logger.info(f"Found {len(self.file_pairs)} image-mask pairs")
        
    def _get_file_pairs(self) -> List[Tuple[Path, Path]]:
        pairs = []
        image_files = list(self.image_dir.glob('*_T1w.nii.gz'))
        
        for img_file in image_files:
            base_name = img_file.name.replace('_T1w.nii.gz', '')
            mask_file = self.mask_dir / f"{base_name}_label-L_desc-T1lesion_mask.nii.gz"
            
            if mask_file.exists():
                pairs.append((img_file, mask_file))
        
        return pairs
    
    def _load_nifti_volume(self, file_path: Path) -> np.ndarray:
        try:
            nii = nib.load(str(file_path))
            volume = nii.get_fdata()
            
            if volume.ndim == 4:
                volume = volume[:, :, :, 0]
            elif volume.ndim != 3:
                return np.zeros(self.target_shape, dtype=np.float32)
            
            volume = np.nan_to_num(volume, nan=0.0)
            volume = self._resize_volume_3d(volume, self.target_shape)
            
            return volume.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return np.zeros(self.target_shape, dtype=np.float32)
    
    def _resize_volume_3d(self, volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        current_shape = volume.shape
        zoom_factors = [target_shape[i] / current_shape[i] for i in range(3)]
        resized_volume = zoom(volume, zoom_factors, order=1)
        
        # Ensure exact target shape
        if resized_volume.shape != target_shape:
            resized_volume = self._crop_or_pad_to_shape(resized_volume, target_shape)
        
        return resized_volume
    
    def _crop_or_pad_to_shape(self, volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        current_shape = volume.shape
        result = volume.copy()
        
        for i in range(3):
            current_size = current_shape[i]
            target_size = target_shape[i]
            
            if current_size > target_size:
                start = (current_size - target_size) // 2
                end = start + target_size
                
                if i == 0:
                    result = result[start:end, :, :]
                elif i == 1:
                    result = result[:, start:end, :]
                else:
                    result = result[:, :, start:end]
                    
            elif current_size < target_size:
                pad_before = (target_size - current_size) // 2
                pad_after = target_size - current_size - pad_before
                
                pad_width = [(0, 0), (0, 0), (0, 0)]
                pad_width[i] = (pad_before, pad_after)
                
                result = np.pad(result, pad_width, mode='constant', constant_values=0)
        
        return result
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        p1, p99 = np.percentile(image, [1, 99])
        image = np.clip(image, p1, p99)
        
        if image.max() > image.min():
            image = (image - image.min()) / (image.max() - image.min())
        
        return image
    
    def _preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        return (mask > 0).astype(np.float32)
    
    def data_generator(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        for img_path, mask_path in self.file_pairs:
            try:
                image = self._load_nifti_volume(img_path)
                mask = self._load_nifti_volume(mask_path)
                
                image = self._preprocess_image(image)
                mask = self._preprocess_mask(mask)
                
                image = image[..., np.newaxis]
                mask = mask[..., np.newaxis]
                
                yield image, mask
                
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                continue
    
    def create_dataset(self, batch_size: int = 2, shuffle: bool = True, validation_split: float = 0.2) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        dataset = tf.data.Dataset.from_generator(
            self.data_generator,
            output_signature=(
                tf.TensorSpec(shape=(*self.target_shape, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(*self.target_shape, 1), dtype=tf.float32)
            )
        )
        
        dataset_size = len(self.file_pairs)
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(100, dataset_size))
        
        val_size = int(dataset_size * validation_split)
        train_size = dataset_size - val_size
        
        train_dataset = dataset.take(train_size)
        val_dataset = dataset.skip(train_size)
        
        train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        logger.info(f"Training samples: {train_size}, Validation samples: {val_size}")
        
        return train_dataset, val_dataset

def create_real_data_loader(config: dict) -> RealAtlasDataLoader:
    data_dir = config['data']['data_dir']
    target_shape = tuple(config['model']['input_shape'][:-1])
    return RealAtlasDataLoader(data_dir, target_shape)
EOF

# Test the fixed data loader
echo "🧪 Testing fixed data loader..."
python -c "
from real_data_loader import RealAtlasDataLoader
loader = RealAtlasDataLoader('/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training', (192, 224, 176))
print(f'✅ Data loader created - found {len(loader.file_pairs)} pairs')
"

if [ $? -ne 0 ]; then
    echo "❌ Data loader test failed"
    exit 1
fi

# Create directories
mkdir -p logs checkpoints outputs

# Start training
echo "🎯 Starting ATLAS 2.0 training with fixed data loader..."
echo "Time: $(date)"

python training/robust_train.py

TRAIN_EXIT_CODE=$?

# Results
echo "📊 Training completed: $(date)"
echo "Exit code: $TRAIN_EXIT_CODE"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ Training successful!"
    
    if [ -f "checkpoints/best_model.h5" ]; then
        echo "Model saved: $(du -h checkpoints/best_model.h5 | cut -f1)"
    fi
    
else
    echo "❌ Training failed"
    
    if [ -f "logs/training.log" ]; then
        echo "Last 20 lines of log:"
        tail -20 logs/training.log
    fi
fi

echo "🏁 Job completed: $(date)"
exit $TRAIN_EXIT_CODE
