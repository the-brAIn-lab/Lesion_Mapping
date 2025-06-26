#!/usr/bin/env python3
"""
FIXED training script for stroke lesion segmentation using ATLAS dataset.
Key fixes:
1. Added validation split (0.2) to prevent overfitting
2. Increased BASE_FILTERS to 16 for better capacity
3. Monitors validation metrics instead of training metrics
4. Reduced flip augmentation probability
"""

import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import logging
import tensorflow as tf
import numpy as np
import nibabel as nib
from pathlib import Path
from datetime import datetime
from scipy.ndimage import zoom
from working_sota_model import build_sota_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/sota_training_fixed.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    DATA_DIR = "/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training_Split"
    INPUT_SHAPE = (192, 224, 176, 1)  # High-resolution MRI
    BATCH_SIZE = 1
    EPOCHS = 50
    BASE_FILTERS = 16  # INCREASED from 8 to 16 for better capacity
    VALIDATION_SPLIT = 0.2  # FIXED: Use 20% for validation to prevent overfitting
    LEARNING_RATE = 1e-4
    GRADIENT_ACCUM_STEPS = 4
    FLIP_PROB = 0.3  # REDUCED from 0.5 to 0.3 for less aggressive augmentation
    CALLBACKS_DIR = lambda timestamp: Path(f'callbacks/sota_fixed_{timestamp}')
    MODEL_SAVE_PATH = lambda timestamp: f'models/sota_fixed_{timestamp}.h5'

def configure_hardware():
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            tf.config.set_visible_devices(gpus[0], 'GPU')  # Single GPU
            tf.config.experimental.set_memory_growth(gpus[0], True)
            logger.info(f"Detected {len(gpus)} GPU(s), using GPU:0")
            strategy = tf.distribute.get_strategy()
            global_batch_size = 1
        else:
            logger.warning("No GPUs detected, using CPU")
            strategy = tf.distribute.get_strategy()
            global_batch_size = 1
        return strategy, global_batch_size
    except Exception as e:
        logger.error(f"Hardware configuration failed: {e}, falling back to CPU")
        return tf.distribute.get_strategy(), 1

def setup_mixed_precision():
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        logger.info("Mixed precision enabled: mixed_float16")
    except Exception as e:
        logger.error(f"Failed to enable mixed precision: {e}")

def resize_volume(volume, target_shape):
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=1)

def combined_loss(y_true, y_pred, smooth=1e-6, focal_gamma=3.0, focal_alpha=0.25):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice_loss = 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    alpha_t = tf.where(tf.equal(y_true, 1), focal_alpha, 1 - focal_alpha)
    focal_loss = -alpha_t * tf.pow(1 - p_t, focal_gamma) * tf.math.log(p_t)
    focal_loss = tf.reduce_mean(focal_loss)
    return 0.7 * dice_loss + 0.3 * focal_loss

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred, tf.float32))
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def binary_dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

class AtlasDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_dir, image_ids, batch_size, target_shape, shuffle=True, augment=True):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "Images")
        self.masks_dir = os.path.join(data_dir, "Masks")
        self.image_ids = image_ids
        self.batch_size = batch_size
        self.target_shape = target_shape
        self.shuffle = shuffle
        self.augment = augment  # NEW: Control augmentation for validation
        self.indexes = np.arange(len(image_ids))
        logger.info(f"Initialized generator with {len(image_ids)} samples (augment={augment})")
        if shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return len(self.image_ids) // self.batch_size

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X = np.zeros((self.batch_size, *self.target_shape, 1), dtype=np.float32)
        y = np.zeros((self.batch_size, *self.target_shape, 1), dtype=np.float32)
        for i, idx in enumerate(batch_indexes):
            try:
                img_id = self.image_ids[idx]
                img_path = os.path.join(self.images_dir, f"{img_id}_space-MNI152NLin2009aSym_T1w.nii.gz")
                mask_path = os.path.join(self.masks_dir, f"{img_id}_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz")
                img_data = nib.load(img_path).get_fdata(dtype=np.float32)
                mask_data = nib.load(mask_path).get_fdata(dtype=np.float32)
                
                if img_data.shape != self.target_shape:
                    img_data = resize_volume(img_data, self.target_shape)
                    mask_data = resize_volume(mask_data, self.target_shape)
                    mask_data = (mask_data > 0.5).astype(np.float32)
                
                # Normalize
                p1, p99 = np.percentile(img_data[img_data > 0], [1, 99])
                img_data = np.clip(img_data, p1, p99)
                img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min() + 1e-8)
                
                X[i] = img_data[..., np.newaxis]
                y[i] = mask_data[..., np.newaxis]
                
                # IMPROVED: Reduced flip augmentation and only during training
                if self.augment and np.random.rand() > (1 - Config.FLIP_PROB):
                    X[i] = np.flip(X[i], axis=1)
                    y[i] = np.flip(y[i], axis=1)
                    
            except Exception as e:
                logger.error(f"Error loading {img_id}: {e}")
                X[i] = np.zeros((*self.target_shape, 1), dtype=np.float32)
                y[i] = np.zeros((*self.target_shape, 1), dtype=np.float32)
        
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

def load_image_ids(data_dir):
    images_dir = os.path.join(data_dir, "Images")
    masks_dir = os.path.join(data_dir, "Masks")
    image_ids = [
        fname.replace("_space-MNI152NLin2009aSym_T1w.nii.gz", "") for fname in os.listdir(images_dir)
        if fname.endswith("_space-MNI152NLin2009aSym_T1w.nii.gz") and
        os.path.exists(os.path.join(masks_dir, f"{fname.replace('_space-MNI152NLin2009aSym_T1w.nii.gz', '')}_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz"))
    ]
    logger.info(f"Found {len(image_ids)} valid image/mask pairs")
    return image_ids

def create_callbacks(callbacks_dir):
    callbacks_dir.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            str(callbacks_dir / 'best_model.h5'),
            monitor='val_dice_coefficient',  # FIXED: Monitor validation Dice instead of training
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_dice_coefficient',  # FIXED: Monitor validation Dice
            mode='max',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',  # FIXED: Monitor validation loss
            factor=0.5,
            patience=8,  # Reduced patience for faster adaptation
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(str(callbacks_dir / 'training_log.csv')),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(callbacks_dir / 'tensorboard'),
            histogram_freq=0,
            write_graph=False,
            update_freq='epoch'
        )
    ]

def train_model():
    logger.info("Starting FIXED Stroke Lesion Segmentation Training")
    logger.info(f"Key improvements:")
    logger.info(f"  - Validation split: {Config.VALIDATION_SPLIT}")
    logger.info(f"  - Base filters: {Config.BASE_FILTERS}")
    logger.info(f"  - Flip probability: {Config.FLIP_PROB}")
    
    setup_mixed_precision()
    strategy, global_batch_size = configure_hardware()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Load and split data
    image_ids = load_image_ids(Config.DATA_DIR)
    np.random.shuffle(image_ids)
    
    # FIXED: Proper train/validation split
    n_total = len(image_ids)
    n_val = int(n_total * Config.VALIDATION_SPLIT)
    n_train = n_total - n_val
    
    train_ids = image_ids[:n_train]
    val_ids = image_ids[n_train:]
    
    logger.info(f"Total samples: {n_total}")
    logger.info(f"Training samples: {len(train_ids)}")
    logger.info(f"Validation samples: {len(val_ids)}")
    
    # Create generators
    train_generator = AtlasDataGenerator(
        Config.DATA_DIR, train_ids, global_batch_size, Config.INPUT_SHAPE[:-1], 
        shuffle=True, augment=True  # Augmentation only for training
    )
    
    val_generator = AtlasDataGenerator(
        Config.DATA_DIR, val_ids, global_batch_size, Config.INPUT_SHAPE[:-1], 
        shuffle=False, augment=False  # No augmentation for validation
    )
    
    # Build model
    with strategy.scope():
        model = build_sota_model(input_shape=Config.INPUT_SHAPE, base_filters=Config.BASE_FILTERS)
        logger.info(f"Model input shape: {model.input_shape}, output shape: {model.output_shape}")
        logger.info(f"Model parameters: {model.count_params():,}")
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss=combined_loss,
            metrics=[dice_coefficient, binary_dice_coefficient, 'accuracy']
        )
    
    # Validate model setup
    try:
        X, y = next(iter(train_generator))
        logger.info(f"Validating model with batch: X={X.shape}, y={y.shape}")
        pred = model(X, training=False)
        logger.info(f"Model prediction shape: {pred.shape}")
        
        # Test gradient computation
        with tf.GradientTape() as tape:
            pred = model(X, training=True)
            loss = combined_loss(y, pred)
        grads = tape.gradient(loss, model.trainable_variables)
        logger.info(f"Gradient shapes: {[g.shape if g is not None else None for g in grads[:3]]}")
        
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        raise
    
    # Train model
    try:
        logger.info("Starting training with validation monitoring...")
        history = model.fit(
            train_generator,
            validation_data=val_generator,  # FIXED: Added validation data
            epochs=Config.EPOCHS,
            callbacks=create_callbacks(Config.CALLBACKS_DIR(timestamp)),
            verbose=1
        )
        
        logger.info("Training completed successfully")
        
        # Save final model
        final_model_path = Config.MODEL_SAVE_PATH(timestamp)
        model.save(final_model_path)
        logger.info(f"Final model saved to: {final_model_path}")
        
        # Log final results
        if history.history:
            final_train_dice = history.history['dice_coefficient'][-1]
            final_val_dice = history.history['val_dice_coefficient'][-1]
            best_val_dice = max(history.history['val_dice_coefficient'])
            
            logger.info(f"Training Results:")
            logger.info(f"  Final training Dice: {final_train_dice:.4f}")
            logger.info(f"  Final validation Dice: {final_val_dice:.4f}")
            logger.info(f"  Best validation Dice: {best_val_dice:.4f}")
            logger.info(f"  Overfitting gap: {final_train_dice - final_val_dice:.4f}")
            
            if final_train_dice - final_val_dice < 0.1:
                logger.info("✅ Low overfitting - model should generalize well!")
            else:
                logger.warning("⚠️  High overfitting detected - consider more regularization")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    try:
        train_model()
        logger.info("FIXED training pipeline executed successfully")
        logger.info("Model should now generalize better to test data!")
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise SystemExit(1)
