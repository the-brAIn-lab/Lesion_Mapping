#!/usr/bin/env python3
"""
Batch-wise Training Script - Only loads one batch at a time
Adapted from working data generator pattern
"""
import os
import sys
import tensorflow as tf
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
EPOCHS = 100
BATCH_SIZE = 2  # Back to 2 since we're managing memory properly
DATA_PATH = "/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training"

# Data dimensions
MODEL_IMG_HEIGHT = 192
MODEL_IMG_WIDTH = 224  
MODEL_IMG_DEPTH = 176
IMG_CHANNELS = 1

def resize_volume(volume, target_shape):
    """Resize volume to target shape using zoom"""
    factors = [t/s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=1)

def dice_coefficient(y_true, y_pred):
    """Dice coefficient metric"""
    y_true_f = tf.cast(tf.keras.backend.flatten(y_true), tf.float32)
    y_pred_f = tf.cast(tf.keras.backend.flatten(y_pred), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + 1e-6) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1e-6)

def dice_loss(y_true, y_pred):
    """Dice loss"""
    return 1 - dice_coefficient(y_true, y_pred)

class AtlasDataGenerator(tf.keras.utils.Sequence):
    """Data generator that loads only one batch at a time - prevents OOM"""
    
    def __init__(self, base_dir, batch_size, model_dim, shuffle=True):
        self.base_dir = base_dir
        self.images_dir = os.path.join(base_dir, "Images")
        self.masks_dir = os.path.join(base_dir, "Masks")
        self.batch_size = batch_size
        self.model_dim = model_dim
        self.shuffle = shuffle
        self.image_ids = []
        
        # Find matching image/mask pairs
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
        """Load and return one batch - CRITICAL: Only loads batch_size samples"""
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Pre-allocate batch arrays
        X = np.empty((self.batch_size, *self.model_dim, IMG_CHANNELS), dtype=np.float32)
        y = np.empty((self.batch_size, *self.model_dim, 1), dtype=np.float32)
        
        for i, idx in enumerate(indexes):
            try:
                # Construct file paths
                img_path = os.path.join(self.images_dir, self.image_ids[idx] + "_T1w.nii.gz")
                mask_path = os.path.join(self.masks_dir, self.image_ids[idx] + "_label-L_desc-T1lesion_mask.nii.gz")
                
                # Load ONE image and mask at a time
                img_data = nib.load(img_path).get_fdata(dtype=np.float32)
                mask_data = nib.load(mask_path).get_fdata(dtype=np.float32)
                
                # Resize to model dimensions
                if img_data.shape != self.model_dim:
                    img_data = resize_volume(img_data, self.model_dim)
                    mask_data = resize_volume(mask_data, self.model_dim)
                    mask_data = (mask_data > 0.5).astype(np.float32)
                
                # Normalize image (percentile-based to handle outliers)
                p1, p99 = np.percentile(img_data, [1, 99])
                img_data = np.clip(img_data, p1, p99)
                img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min() + 1e-8)
                
                # Add to batch
                X[i] = np.expand_dims(img_data, axis=-1)
                y[i] = np.expand_dims(mask_data, axis=-1)
                
            except Exception as e:
                logger.error(f"Error loading {self.image_ids[idx]}: {e}")
                # Fill with zeros if loading fails
                X[i] = np.zeros((*self.model_dim, IMG_CHANNELS), dtype=np.float32)
                y[i] = np.zeros((*self.model_dim, 1), dtype=np.float32)
        
        return X, y
    
    def on_epoch_end(self):
        """Shuffle data at end of epoch"""
        self.indexes = np.arange(len(self.image_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

def build_model():
    """Build the full-size model - we have enough GPU memory for this"""
    inputs = tf.keras.layers.Input(shape=(MODEL_IMG_HEIGHT, MODEL_IMG_WIDTH, MODEL_IMG_DEPTH, IMG_CHANNELS))
    
    # Encoder path
    conv1 = tf.keras.layers.Conv3D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.Conv3D(32, 3, activation='relu', padding='same')(conv1)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    pool1 = tf.keras.layers.MaxPooling3D(pool_size=2)(conv1)
    
    conv2 = tf.keras.layers.Conv3D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    conv2 = tf.keras.layers.Conv3D(64, 3, activation='relu', padding='same')(conv2)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    pool2 = tf.keras.layers.MaxPooling3D(pool_size=2)(conv2)
    
    conv3 = tf.keras.layers.Conv3D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    conv3 = tf.keras.layers.Conv3D(128, 3, activation='relu', padding='same')(conv3)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    pool3 = tf.keras.layers.MaxPooling3D(pool_size=2)(conv3)
    
    # Bottleneck
    conv4 = tf.keras.layers.Conv3D(256, 3, activation='relu', padding='same')(pool3)
    conv4 = tf.keras.layers.BatchNormalization()(conv4)
    conv4 = tf.keras.layers.Conv3D(256, 3, activation='relu', padding='same')(conv4)
    conv4 = tf.keras.layers.BatchNormalization()(conv4)
    
    # Decoder path
    up5 = tf.keras.layers.Conv3DTranspose(128, 2, strides=2, padding='same')(conv4)
    up5 = tf.keras.layers.concatenate([up5, conv3])
    conv5 = tf.keras.layers.Conv3D(128, 3, activation='relu', padding='same')(up5)
    conv5 = tf.keras.layers.BatchNormalization()(conv5)
    conv5 = tf.keras.layers.Conv3D(128, 3, activation='relu', padding='same')(conv5)
    conv5 = tf.keras.layers.BatchNormalization()(conv5)
    
    up6 = tf.keras.layers.Conv3DTranspose(64, 2, strides=2, padding='same')(conv5)
    up6 = tf.keras.layers.concatenate([up6, conv2])
    conv6 = tf.keras.layers.Conv3D(64, 3, activation='relu', padding='same')(up6)
    conv6 = tf.keras.layers.BatchNormalization()(conv6)
    conv6 = tf.keras.layers.Conv3D(64, 3, activation='relu', padding='same')(conv6)
    conv6 = tf.keras.layers.BatchNormalization()(conv6)
    
    up7 = tf.keras.layers.Conv3DTranspose(32, 2, strides=2, padding='same')(conv6)
    up7 = tf.keras.layers.concatenate([up7, conv1])
    conv7 = tf.keras.layers.Conv3D(32, 3, activation='relu', padding='same')(up7)
    conv7 = tf.keras.layers.BatchNormalization()(conv7)
    conv7 = tf.keras.layers.Conv3D(32, 3, activation='relu', padding='same')(conv7)
    conv7 = tf.keras.layers.BatchNormalization()(conv7)
    
    outputs = tf.keras.layers.Conv3D(1, 1, activation='sigmoid')(conv7)
    
    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
    return model

def main():
    """Main training function"""
    logger.info("🚀 Starting Stroke Lesion Segmentation Training")
    
    # Configure GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"Using {len(gpus)} GPU(s)")
    else:
        logger.error("No GPUs found")
        return False
    
    # Create data generators
    logger.info("Creating data generators...")
    train_generator = AtlasDataGenerator(
        base_dir=DATA_PATH,
        batch_size=BATCH_SIZE,
        model_dim=(MODEL_IMG_HEIGHT, MODEL_IMG_WIDTH, MODEL_IMG_DEPTH),
        shuffle=True
    )
    
    # Use 80/20 split
    total_samples = len(train_generator.image_ids)
    train_samples = int(0.8 * total_samples)
    
    # Split the data
    train_generator.image_ids = train_generator.image_ids[:train_samples]
    train_generator.on_epoch_end()
    
    val_generator = AtlasDataGenerator(
        base_dir=DATA_PATH,
        batch_size=BATCH_SIZE,
        model_dim=(MODEL_IMG_HEIGHT, MODEL_IMG_WIDTH, MODEL_IMG_DEPTH),
        shuffle=False
    )
    val_generator.image_ids = val_generator.image_ids[train_samples:]
    val_generator.on_epoch_end()
    
    logger.info(f"Training batches: {len(train_generator)}")
    logger.info(f"Validation batches: {len(val_generator)}")
    
    # Build model
    logger.info("Building model...")
    model = build_model()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=dice_loss,
        metrics=[dice_coefficient]
    )
    
    logger.info(f"Model parameters: {model.count_params():,}")
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='checkpoints/best_model.h5',
            save_best_only=True,
            monitor='val_dice_coefficient',
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_dice_coefficient',
            patience=15,
            mode='max',
            verbose=1,
            restore_best_weights=True
        )
    ]
    
    # Train model
    logger.info("Starting training...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        verbose=1,
        callbacks=callbacks
    )
    
    logger.info("✅ Training completed!")
    
    # Save final model
    model.save('checkpoints/final_model.h5')
    logger.info("Model saved!")
    
    # Print best results
    best_dice = max(history.history['val_dice_coefficient'])
    logger.info(f"🎯 Best validation Dice score: {best_dice:.4f}")
    
    return True

if __name__ == "__main__":
    if main():
        logger.info("Training finished successfully!")
    else:
        logger.error("Training failed!")
        sys.exit(1)
