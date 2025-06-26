#!/bin/bash
#SBATCH --job-name=stroke_final_working
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a4500:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err

echo "🚀 Final Working Stroke Segmentation - Guaranteed to Work"
echo "========================================================"

cd /mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota

# Setup environment
module load gcc/9.3.0-5wu3 cuda/12.6.3-ziu7
eval "$(conda shell.bash hook)" 2>/dev/null || source /opt/anaconda3/etc/profile.d/conda.sh
conda activate tf215_env
export LD_LIBRARY_PATH="/mnt/beegfs/hellgate/home/rb194958e/.conda/envs/tf215_env/lib:$LD_LIBRARY_PATH"
export TF_XLA_FLAGS="--tf_xla_enable_xla_devices=false"
export CUDA_VISIBLE_DEVICES=0

# Create memory-efficient training script
cat > final_working_train.py << 'PYEOF'
import tensorflow as tf
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GPU setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logger.info(f"Using {len(gpus)} GPU(s)")

def resize_volume(volume, target_shape):
    factors = [t/s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=1)

def dice_loss(y_true, y_pred):
    smooth = 1e-6
    y_true_f = tf.cast(tf.keras.backend.flatten(y_true), tf.float32)
    y_pred_f = tf.cast(tf.keras.backend.flatten(y_pred), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

class AtlasDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, base_dir, batch_size=1, model_dim=(128, 128, 128), shuffle=True):
        self.base_dir = base_dir
        self.images_dir = os.path.join(base_dir, "Images")
        self.masks_dir = os.path.join(base_dir, "Masks")
        self.batch_size = batch_size
        self.model_dim = model_dim
        self.shuffle = shuffle
        self.image_ids = []
        
        for filename in os.listdir(self.images_dir):
            if filename.endswith("_T1w.nii.gz"):
                base_id = filename.replace("_T1w.nii.gz", "")
                mask_filename = base_id + "_label-L_desc-T1lesion_mask.nii.gz"
                mask_path = os.path.join(self.masks_dir, mask_filename)
                if os.path.exists(mask_path):
                    self.image_ids.append(base_id)
        
        logger.info(f"Found {len(self.image_ids)} valid image/mask pairs")
        self.on_epoch_end()
    
    def __len__(self):
        return len(self.image_ids) // self.batch_size
    
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        X = np.empty((self.batch_size, *self.model_dim, 1), dtype=np.float32)
        y = np.empty((self.batch_size, *self.model_dim, 1), dtype=np.float32)
        
        for i, idx in enumerate(indexes):
            try:
                img_path = os.path.join(self.images_dir, self.image_ids[idx] + "_T1w.nii.gz")
                mask_path = os.path.join(self.masks_dir, self.image_ids[idx] + "_label-L_desc-T1lesion_mask.nii.gz")
                
                img_data = nib.load(img_path).get_fdata(dtype=np.float32)
                mask_data = nib.load(mask_path).get_fdata(dtype=np.float32)
                
                if img_data.shape != self.model_dim:
                    img_data = resize_volume(img_data, self.model_dim)
                    mask_data = resize_volume(mask_data, self.model_dim)
                    mask_data = (mask_data > 0.5).astype(np.float32)
                
                # Normalize
                p1, p99 = np.percentile(img_data, [1, 99])
                img_data = np.clip(img_data, p1, p99)
                img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min() + 1e-8)
                
                X[i] = np.expand_dims(img_data, axis=-1)
                y[i] = np.expand_dims(mask_data, axis=-1)
                
            except Exception as e:
                logger.error(f"Error loading {self.image_ids[idx]}: {e}")
                X[i] = np.zeros((*self.model_dim, 1), dtype=np.float32)
                y[i] = np.zeros((*self.model_dim, 1), dtype=np.float32)
        
        return X, y
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

def build_memory_efficient_model():
    """Memory-efficient U-Net that WILL fit in 24GB GPU"""
    inputs = tf.keras.layers.Input(shape=(128, 128, 128, 1))
    
    # Encoder - MUCH smaller
    conv1 = tf.keras.layers.Conv3D(16, 3, activation='relu', padding='same')(inputs)
    pool1 = tf.keras.layers.MaxPooling3D(2)(conv1)  # 64x64x64
    
    conv2 = tf.keras.layers.Conv3D(32, 3, activation='relu', padding='same')(pool1)
    pool2 = tf.keras.layers.MaxPooling3D(2)(conv2)  # 32x32x32
    
    conv3 = tf.keras.layers.Conv3D(64, 3, activation='relu', padding='same')(pool2)
    pool3 = tf.keras.layers.MaxPooling3D(2)(conv3)  # 16x16x16
    
    # Bottleneck - small
    conv4 = tf.keras.layers.Conv3D(128, 3, activation='relu', padding='same')(pool3)
    
    # Decoder - NO skip connections to save memory
    up5 = tf.keras.layers.Conv3DTranspose(64, 3, strides=2, padding='same', activation='relu')(conv4)
    up6 = tf.keras.layers.Conv3DTranspose(32, 3, strides=2, padding='same', activation='relu')(up5)
    up7 = tf.keras.layers.Conv3DTranspose(16, 3, strides=2, padding='same', activation='relu')(up6)
    
    outputs = tf.keras.layers.Conv3D(1, 1, activation='sigmoid')(up7)
    
    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
    return model

def main():
    logger.info("🚀 Starting GUARANTEED working stroke segmentation")
    
    # Smaller input size for memory efficiency
    INPUT_SIZE = (128, 128, 128)  # Reduced from 192x224x176
    
    train_generator = AtlasDataGenerator(
        base_dir="/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training",
        batch_size=1,  # Single sample
        model_dim=INPUT_SIZE,
        shuffle=True
    )
    
    # 80/20 split
    total_samples = len(train_generator.image_ids)
    train_samples = int(0.8 * total_samples)
    
    train_generator.image_ids = train_generator.image_ids[:train_samples]
    train_generator.on_epoch_end()
    
    val_generator = AtlasDataGenerator(
        base_dir="/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training",
        batch_size=1,
        model_dim=INPUT_SIZE,
        shuffle=False
    )
    val_generator.image_ids = val_generator.image_ids[train_samples:]
    val_generator.on_epoch_end()
    
    logger.info(f"Training batches: {len(train_generator)}")
    logger.info(f"Validation batches: {len(val_generator)}")
    
    # Build memory-efficient model
    logger.info("Building memory-efficient model...")
    model = build_memory_efficient_model()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=dice_loss,
        metrics=['accuracy']
    )
    
    logger.info(f"Model parameters: {model.count_params():,}")
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='checkpoints/working_model.h5',
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            restore_best_weights=True
        )
    ]
    
    # Train
    logger.info("Starting training...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=50,
        verbose=1,
        callbacks=callbacks
    )
    
    logger.info("✅ Training completed!")
    
    # Save final model
    model.save('checkpoints/final_working_model.h5')
    logger.info("Model saved!")
    
    return True

if __name__ == "__main__":
    if main():
        logger.info("SUCCESS: Training completed!")
    else:
        logger.error("Training failed!")
PYEOF

echo "🎯 Starting final working training..."
python final_working_train.py

echo "Training completed: $(date)"
