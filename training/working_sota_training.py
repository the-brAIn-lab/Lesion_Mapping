#!/usr/bin/env python3
"""
Training script for SOTA model using ATLAS dataset
"""
import tensorflow as tf
import numpy as np
import nibabel as nib
import os
import logging
from pathlib import Path
from scipy.ndimage import zoom
from working_sota_model import build_sota_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable mixed precision
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Configure GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logger.info(f"Using {len(gpus)} GPU(s)")

def resize_volume(volume, target_shape):
    factors = [t/s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=1)

def combined_loss(y_true, y_pred, focal_gamma=3.0, focal_alpha=0.25):
    smooth = 1e-6
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Dice Loss
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice_loss = 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

    # Focal Loss
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    alpha_t = tf.where(tf.equal(y_true, 1), focal_alpha, 1 - focal_alpha)
    focal_loss = -alpha_t * tf.pow(1 - p_t, focal_gamma) * tf.math.log(p_t)
    focal_loss = tf.reduce_mean(focal_loss)

    return 0.7 * dice_loss + 0.3 * focal_loss

def dice_coefficient(y_true, y_pred):
    smooth = 1e-6
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred, tf.float32))
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

class AtlasDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, base_dir, batch_size=1, target_shape=(128,128,128), shuffle=True):
        self.base_dir = base_dir
        self.images_dir = os.path.join(base_dir, "Images")
        self.masks_dir = os.path.join(base_dir, "Masks")
        self.batch_size = batch_size
        self.target_shape = target_shape
        self.shuffle = shuffle
        self.image_ids = []

        for filename in os.listdir(self.images_dir):
            if filename.endswith("_T1w.nii.gz"):
                base_id = filename.replace("_T1w.nii.gz", "")
                mask_path = os.path.join(self.masks_dir, base_id + "_label-L_desc_nn.nii.gz")
                if os.path.exists(mask_path):
                    self.image_ids.append(base_id)
        logger.info(f"Found {len(self.image_ids)} valid image/mask pairs")
        self.on_epoch_end()

    def __len__(self):
        return len(self.image_ids) // self.batch_size

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X = np.empty((self.batch_size, *self.target_shape, 1), dtype=np.float32)
        y = np.empty((self.batch_size, *self.target_shape, 1), dtype=np.float32)

        for i, idx in enumerate(indexes):
            img_path = os.path.join(self.images_dir, self.image_ids[idx] + "_T1w.nii.gz")
            mask_path = os.path.join(self.masks_dir, self.image_ids[idx] + "_label-L_desc_nn.nii.gz")
            img_data = nib.load(img_path).get_fdata(dtype=np.float32)
            mask_data = nib.load(mask_path).get_fdata(dtype=np.float32)

            if img_data.shape != self.target_shape:
                img_data = resize_volume(img_data, self.target_shape)
                mask_data = resize_volume(mask_data, self.target_shape)
                mask_data = (mask_data > 0.5).astype(np.float32)

            p1, p99 = np.percentile(img_data[img_data > 0], [1, 99])
            img_data = np.clip(img_data, p1, p99)
            img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min() + 1e-8)

            X[i] = np.expand_dims(img_data, axis=-1)
            y[i] = np.expand_dims(mask_data, axis=-1)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

def main():
    logger.info("Starting SOTA Training")
    data_dir = "/mnt/beegfs/hellgate/home/rb194958e/Atlas_Train/Training_Split"
    batch_size = 1
    epochs = 64
    
    # Data generators
    train_generator = AtlasDataGenerator(data_dir, batch_size, shuffle=True)
    val_generator = AtlasDataGenerator(data_dir, batch_size, shuffle=False)

    total_samples = len(train_generator.image_ids)
    train_samples = int(0.8 * total_samples)
    train_generator.image_ids = train_generator.image_ids[:train_samples]
    val_generator.image_ids = train_generator.image_ids[train_samples:]
    train_generator.on_epoch_end()
    val_generator.on_epoch_end()

    logger.info(f"Training samples: {len(train_generator.image_ids)}")
    logger.info(f"Validation samples: {len(val_generator.image_ids)}")

    # Model
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = build_sota_model(input_shape=(128, 128, 128, 256))
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        model.compile(
            optimizer=optimizer,
            loss=combined_loss,
            metrics=[dice_coefficient, 'accuracy']
        )
    logger.info(f"Model parameters: {model.count_params():,}")

    # Callbacks
    callbacks_dir = Path(f'logs/callbacks_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    callbacks_dir.mkdir(exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(callbacks_dir / 'best_model.h5'),
            monitor='val_dice_coefficient',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=15,
            monitor='val_loss',
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(str(callbacks_dir / 'training_log.csv'))
    ]

    # Train
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    logger.info("Training completed")

if __name__ == "__main__":
    main()
