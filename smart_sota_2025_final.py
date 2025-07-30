#!/usr/bin/env python3
"""
SOTA 2025: FINAL PRODUCTION SCRIPT v3
This version implements a powerful combined Focal + Dice loss function,
specifically designed to combat extreme class imbalance and force the model
to learn small, rare targets like stroke lesions.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
import traceback
from pathlib import Path
from datetime import datetime
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras import layers, Model
from scipy.ndimage import zoom, rotate
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Any

# --- 1. CONFIGURATION ---

logger = logging.getLogger(__name__)

class Config:
    """Holds all static configuration for the training run."""
    DATA_DIR: Path = Path("/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training")
    IMAGE_GLOB_PATTERN: str = "*_T1w.nii.gz"
    MASK_GLOB_PATTERN: str = "*_mask.nii.gz"
    
    INPUT_SHAPE: Tuple[int, int, int, int] = (192, 224, 176, 1)
    BASE_FILTERS: int = 22
    
    BATCH_SIZE: int = 4
    EPOCHS: int = 150
    VALIDATION_SPLIT: float = 0.15
    INITIAL_LR: float = 1e-5
    DROPOUT_RATE: float = 0.4

# --- 2. SETUP & ENVIRONMENT ---

def setup_environment(run_timestamp: str) -> Path:
    log_dir = Path("runs") / run_timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger.setLevel(logging.INFO)
    if logger.hasHandlers(): logger.handlers.clear()
        
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(log_dir / 'training.log'); fh.setFormatter(formatter)
    sh = logging.StreamHandler(); sh.setFormatter(formatter)
    logger.addHandler(fh); logger.addHandler(sh)
    
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    logger.info("Mixed precision enabled (mixed_float16).")
    
    return log_dir

def configure_gpu_strategy() -> tf.distribute.Strategy:
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            logger.warning("No GPUs found, using default CPU strategy."); return tf.distribute.get_strategy()
        for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
        strategy = tf.distribute.MirroredStrategy()
        logger.info(f"🚀 MirroredStrategy with {strategy.num_replicas_in_sync} devices.")
        return strategy
    except Exception as e:
        logger.error(f"Could not initialize MirroredStrategy: {e}"); return tf.distribute.get_strategy()

# --- 3. DATA HANDLING & AUGMENTATION ---

def load_dataset_paths(data_dir: Path, img_pattern: str, msk_pattern: str) -> List[Tuple[Path, Path]]:
    image_files = sorted(list(data_dir.rglob(img_pattern)))
    mask_files = sorted(list(data_dir.rglob(msk_pattern)))
    if not image_files or len(image_files) != len(mask_files):
        logger.error("FATAL: Mismatch in data files or no files found."); return []
    logger.info(f"Successfully paired {len(image_files)} samples.")
    return list(zip(image_files, mask_files))

def augment_data(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if np.random.rand() > 0.5:
        flip_axis = np.random.choice([0, 1, 2])
        image, mask = np.flip(image, axis=flip_axis), np.flip(mask, axis=flip_axis)
    if np.random.rand() > 0.5:
        angle = np.random.uniform(-10, 10)
        axes = tuple(np.random.choice([0, 1, 2], size=2, replace=False))
        image = rotate(image, angle, axes=axes, reshape=False, order=1, cval=0.0, prefilter=False)
        mask = rotate(mask, angle, axes=axes, reshape=False, order=0, cval=0.0, prefilter=False)
    return image, mask

class NiftiDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, pairs: List[Tuple[Path, Path]], batch_size: int, target_shape: Tuple[int, ...], is_training: bool = True):
        self.pairs, self.batch_size, self.target_shape, self.is_training = pairs, batch_size, target_shape, is_training
        self.on_epoch_end()
    def __len__(self) -> int: return len(self.pairs) // self.batch_size
    def on_epoch_end(self): self.indexes = np.arange(len(self.pairs)); np.random.shuffle(self.indexes)
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X = np.zeros((self.batch_size, *self.target_shape, 1), dtype=np.float32)
        y = np.zeros((self.batch_size, *self.target_shape, 1), dtype=np.float32)
        for i, idx in enumerate(batch_indexes):
            img_path, msk_path = self.pairs[idx]
            try:
                img = nib.load(img_path).get_fdata(dtype=np.float32)
                msk = nib.load(msk_path).get_fdata(dtype=np.float32)
                if img.shape != self.target_shape:
                    factors = [t / s for t, s in zip(self.target_shape, img.shape)]
                    img = zoom(img, factors, order=1, cval=0.0)
                    msk = zoom(msk, factors, order=0, cval=0.0)
                if self.is_training: img, msk = augment_data(img, msk)
                p1, p99 = np.percentile(img[img > 0], [1, 99]) if np.any(img > 0) else (0, 1)
                img = np.clip(img, p1, p99)
                img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
                X[i,], y[i,] = img[..., None], msk[..., None]
            except Exception as e: logger.error(f"DataGen Error on {img_path}: {e}")
        return X, y

# --- 4. MODEL ARCHITECTURE & NEW LOSS FUNCTION ---

class SOTAConvBlock(layers.Layer):
    """A standard double-convolution block with residual connection and dropout."""
    def __init__(self, filters: int, dropout_rate: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv3D(filters, 3, padding='same', activation='gelu')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv3D(filters, 3, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.dropout = layers.Dropout(dropout_rate)
        self.res_proj = layers.Conv3D(filters, 1, padding='same')
    def call(self, x: tf.Tensor, training: bool = None) -> tf.Tensor:
        residual = x
        x = self.bn1(self.conv1(x), training=training)
        x = self.bn2(self.conv2(x), training=training)
        x = self.dropout(x, training=training)
        if residual.shape[-1] != x.shape[-1]: residual = self.res_proj(residual)
        return tf.nn.gelu(x + residual)

def build_unet_model(input_shape: Tuple[int, ...], base_filters: int, dropout_rate: float) -> Model:
    """Builds the 3D U-Net model."""
    logger.info("🏗️ Building 3D U-Net Model...")
    inputs = layers.Input(shape=input_shape)

    s1 = SOTAConvBlock(base_filters, dropout_rate=dropout_rate, name='s1')(inputs)
    p1 = layers.MaxPooling3D(2)(s1)
    s2 = SOTAConvBlock(base_filters * 2, dropout_rate=dropout_rate, name='s2')(p1)
    p2 = layers.MaxPooling3D(2)(s2)
    s3 = SOTAConvBlock(base_filters * 4, dropout_rate=dropout_rate, name='s3')(p2)
    p3 = layers.MaxPooling3D(2)(s3)
    b = SOTAConvBlock(base_filters * 8, dropout_rate=dropout_rate, name='bottleneck')(p3)

    def decoder_block(x, skip, filters):
        x = layers.Conv3DTranspose(filters, 2, strides=2, padding='same')(x)
        return SOTAConvBlock(filters, dropout_rate=dropout_rate)(layers.Concatenate()([x, skip]))

    d1 = decoder_block(b, s3, base_filters * 4)
    d2 = decoder_block(d1, s2, base_filters * 2)
    d3 = SOTAConvBlock(base_filters, dropout_rate=dropout_rate)(layers.Concatenate()([layers.Conv3DTranspose(base_filters, 2, strides=2, padding='same')(d2), s1]))

    outputs = layers.Conv3D(1, 1, activation='sigmoid', dtype='float32')(d3)
    
    model = Model(inputs, outputs)
    logger.info(f"✅ Model built: {model.count_params():,} parameters")
    return model

# --- 5. TRAINING & EXECUTION ---

def dice_loss(y_true, y_pred, smooth=1e-5):
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true_f, y_pred_f = tf.reshape(y_true, [-1]), tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1.0 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    y_true = tf.cast(y_true, y_pred.dtype)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    p_t = tf.exp(-bce)
    focal = alpha * tf.pow(1 - p_t, gamma) * bce
    return tf.reduce_mean(focal)

def combined_focal_dice_loss(y_true, y_pred):
    """The new combined loss function."""
    return focal_loss(y_true, y_pred) + dice_loss(y_true, y_pred)

def main():
    """Main function to run the entire training pipeline."""
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = setup_environment(run_timestamp)
    logger.info(f"🎉 SOTA 2025: UPGRADED TRAINING RUN (Focal+Dice Loss) - {run_timestamp}")

    try:
        strategy = configure_gpu_strategy()
        all_pairs = load_dataset_paths(Config.DATA_DIR, Config.IMAGE_GLOB_PATTERN, Config.MASK_GLOB_PATTERN)
        if not all_pairs: return
            
        train_pairs, val_pairs = train_test_split(all_pairs, test_size=Config.VALIDATION_SPLIT, random_state=42)
        
        train_gen = NiftiDataGenerator(train_pairs, Config.BATCH_SIZE, Config.INPUT_SHAPE[:-1])
        val_gen = NiftiDataGenerator(val_pairs, Config.BATCH_SIZE, Config.INPUT_SHAPE[:-1], is_training=False)
        
        with strategy.scope():
            model = build_unet_model(Config.INPUT_SHAPE, Config.BASE_FILTERS, Config.DROPOUT_RATE)
            model.compile(
                optimizer=tf.keras.optimizers.AdamW(Config.INITIAL_LR),
                loss=combined_focal_dice_loss, # Using the new combined loss
                metrics=['accuracy', dice_loss] # We can still monitor the pure dice_loss
            )
            
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(str(log_dir / "best_model.keras"), save_best_only=True, monitor='val_dice_loss', mode='min'),
            tf.keras.callbacks.CSVLogger(str(log_dir / "training_log.csv")),
            tf.keras.callbacks.TensorBoard(log_dir=log_dir),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-8),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
        ]
        
        logger.info(f"🚀 Starting training for {Config.EPOCHS} epochs with Focal+Dice loss...")
        model.fit(train_gen, validation_data=val_gen, epochs=Config.EPOCHS, callbacks=callbacks, verbose=1)
        
        logger.info("✅ Training run finished!")
        final_model_path = log_dir / "final_model.keras"
        model.save(final_model_path)
        logger.info(f"💾 Final model saved to: {final_model_path}")
        
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {e}\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()
    logger.info("🎉🎉🎉 SCRIPT EXECUTION FINISHED! 🎉🎉🎉")
