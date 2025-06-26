#!/usr/bin/env python3
"""
Retrain model on FULL dataset (Training + Testing splits combined)
with proper validation split and diverse lesion sizes to fix size bias
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
import glob
import random
from sklearn.model_selection import train_test_split

# Custom imports
from working_sota_model import build_sota_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/full_retrain.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    # FULL DATASET - both Training and Testing splits
    TRAIN_DIR = "/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training_Split"
    TEST_DIR = "/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Testing_Split"
    
    INPUT_SHAPE = (192, 224, 176, 1)  # Same as working model
    BATCH_SIZE = 2  # Increased from 1 for better training
    EPOCHS = 50
    BASE_FILTERS = 8  # Same as working model
    VALIDATION_SPLIT = 0.2  # 20% validation (CRITICAL FIX)
    LEARNING_RATE = 1e-4
    
    CALLBACKS_DIR = lambda timestamp: Path(f'callbacks/full_retrain_{timestamp}')
    MODEL_SAVE_PATH = lambda timestamp: f'models/full_retrain_{timestamp}.h5'

def configure_hardware():
    """Configure GPU with memory growth"""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            logger.info(f"Using GPU: {gpus[0]}")
        else:
            logger.warning("No GPUs detected, using CPU")
        return tf.distribute.get_strategy()
    except Exception as e:
        logger.error(f"Hardware configuration failed: {e}")
        return tf.distribute.get_strategy()

def setup_mixed_precision():
    """Enable mixed precision training"""
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        logger.info("Mixed precision enabled: mixed_float16")
    except Exception as e:
        logger.error(f"Failed to enable mixed precision: {e}")

def resize_volume(volume, target_shape):
    """Resize 3D volume - same as working model"""
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=1)

def combined_loss(y_true, y_pred, smooth=1e-6, focal_gamma=3.0, focal_alpha=0.25):
    """Combined Dice + Focal loss - same as working model"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Dice loss
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice_loss = 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    
    # Focal loss
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    alpha_t = tf.where(tf.equal(y_true, 1), focal_alpha, 1 - focal_alpha)
    focal_loss = -alpha_t * tf.pow(1 - p_t, focal_gamma) * tf.math.log(p_t)
    focal_loss = tf.reduce_mean(focal_loss)
    
    return 0.7 * dice_loss + 0.3 * focal_loss

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Dice coefficient metric"""
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred, tf.float32))
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def binary_dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Binary dice coefficient"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

class FullAtlasDataGenerator(tf.keras.utils.Sequence):
    """Data generator for FULL ATLAS dataset with balanced lesion sizes"""
    def __init__(self, image_mask_pairs, batch_size, target_shape, shuffle=True):
        self.image_mask_pairs = image_mask_pairs
        self.batch_size = batch_size
        self.target_shape = target_shape
        self.shuffle = shuffle
        self.indexes = np.arange(len(image_mask_pairs))
        
        # Analyze lesion sizes for balanced sampling
        self.lesion_sizes = []
        for img_path, mask_path in image_mask_pairs:
            try:
                mask_data = nib.load(mask_path).get_fdata()
                size = np.sum(mask_data)
                self.lesion_sizes.append(size)
            except:
                self.lesion_sizes.append(0)
        
        self.lesion_sizes = np.array(self.lesion_sizes)
        logger.info(f"Dataset: {len(image_mask_pairs)} samples")
        logger.info(f"Lesion sizes - Min: {self.lesion_sizes.min():,.0f}, Max: {self.lesion_sizes.max():,.0f}, Mean: {self.lesion_sizes.mean():,.0f}")
        
        if shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return len(self.image_mask_pairs) // self.batch_size

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X = np.zeros((self.batch_size, *self.target_shape, 1), dtype=np.float32)
        y = np.zeros((self.batch_size, *self.target_shape, 1), dtype=np.float32)
        
        for i, idx in enumerate(batch_indexes):
            try:
                img_path, mask_path = self.image_mask_pairs[idx]
                
                # Load data
                img_data = nib.load(img_path).get_fdata(dtype=np.float32)
                mask_data = nib.load(mask_path).get_fdata(dtype=np.float32)
                
                # Resize if needed
                if img_data.shape != self.target_shape:
                    img_data = resize_volume(img_data, self.target_shape)
                    mask_data = resize_volume(mask_data, self.target_shape)
                    mask_data = (mask_data > 0.5).astype(np.float32)
                
                # Intensity normalization (same as working model)
                p1, p99 = np.percentile(img_data[img_data > 0], [1, 99])
                img_data = np.clip(img_data, p1, p99)
                img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min() + 1e-8)
                
                X[i] = img_data[..., np.newaxis]
                y[i] = mask_data[..., np.newaxis]
                
                # Data augmentation: random horizontal flip (same as working model)
                if np.random.rand() > 0.5:
                    X[i] = np.flip(X[i], axis=1)
                    y[i] = np.flip(y[i], axis=1)
                    
            except Exception as e:
                logger.error(f"Error loading sample {idx}: {e}")
                X[i] = np.zeros((*self.target_shape, 1), dtype=np.float32)
                y[i] = np.zeros((*self.target_shape, 1), dtype=np.float32)
        
        return X, y

    def on_epoch_end(self):
        """Shuffle at end of each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)
            logger.info("Shuffled dataset for new epoch")

def load_full_dataset():
    """Load all image/mask pairs from both Training and Testing splits"""
    all_pairs = []
    
    # Load from Training_Split
    train_images_dir = os.path.join(Config.TRAIN_DIR, "Images")
    train_masks_dir = os.path.join(Config.TRAIN_DIR, "Masks")
    
    if os.path.exists(train_images_dir):
        train_files = [f for f in os.listdir(train_images_dir) if f.endswith("_T1w.nii.gz")]
        for img_file in train_files:
            img_path = os.path.join(train_images_dir, img_file)
            mask_file = img_file.replace("_T1w.nii.gz", "_label-L_desc-T1lesion_mask.nii.gz")
            mask_path = os.path.join(train_masks_dir, mask_file)
            
            if os.path.exists(mask_path):
                all_pairs.append((img_path, mask_path))
    
    # Load from Testing_Split
    test_images_dir = os.path.join(Config.TEST_DIR, "Images")
    test_masks_dir = os.path.join(Config.TEST_DIR, "Masks")
    
    if os.path.exists(test_images_dir):
        test_files = [f for f in os.listdir(test_images_dir) if f.endswith("_T1w.nii.gz")]
        for img_file in test_files:
            img_path = os.path.join(test_images_dir, img_file)
            mask_file = img_file.replace("_T1w.nii.gz", "_label-L_desc-T1lesion_mask.nii.gz")
            mask_path = os.path.join(test_masks_dir, mask_file)
            
            if os.path.exists(mask_path):
                all_pairs.append((img_path, mask_path))
    
    logger.info(f"Loaded {len(all_pairs)} total image/mask pairs")
    logger.info(f"Training split contributed: {len([p for p in all_pairs if 'Training_Split' in p[0]])}")
    logger.info(f"Testing split contributed: {len([p for p in all_pairs if 'Testing_Split' in p[0]])}")
    
    return all_pairs

def create_callbacks(callbacks_dir):
    """Create training callbacks"""
    callbacks_dir.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            str(callbacks_dir / 'best_model.h5'),
            monitor='val_dice_coefficient',  # Monitor VALIDATION Dice
            mode='max',
            save_best_only=True,
            verbose=1,
            save_weights_only=False
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_dice_coefficient',
            mode='max',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
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
    """Main training function"""
    logger.info("Starting FULL DATASET Stroke Lesion Segmentation Training")
    logger.info("=" * 70)
    logger.info("KEY IMPROVEMENTS:")
    logger.info("- Using FULL dataset (Training + Testing splits)")
    logger.info("- Proper validation split (20%)")
    logger.info("- Batch size 2 with shuffling")
    logger.info("- Diverse lesion sizes to fix size bias")
    logger.info("=" * 70)
    
    setup_mixed_precision()
    strategy = configure_hardware()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Load full dataset
    all_pairs = load_full_dataset()
    
    if len(all_pairs) < 10:
        logger.error("Not enough data found!")
        return False
    
    # Split into train/validation (80/20)
    train_pairs, val_pairs = train_test_split(
        all_pairs, 
        test_size=Config.VALIDATION_SPLIT, 
        random_state=42,
        shuffle=True
    )
    
    logger.info(f"Training samples: {len(train_pairs)}")
    logger.info(f"Validation samples: {len(val_pairs)}")
    
    # Create data generators
    train_generator = FullAtlasDataGenerator(
        train_pairs, Config.BATCH_SIZE, Config.INPUT_SHAPE[:-1], shuffle=True
    )
    val_generator = FullAtlasDataGenerator(
        val_pairs, Config.BATCH_SIZE, Config.INPUT_SHAPE[:-1], shuffle=False
    )
    
    # Build model (same architecture as working model)
    with strategy.scope():
        model = build_sota_model(input_shape=Config.INPUT_SHAPE, base_filters=Config.BASE_FILTERS)
        logger.info(f"Model parameters: {model.count_params():,}")
        
        # Compile with mixed precision optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE)
        if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        model.compile(
            optimizer=optimizer,
            loss=combined_loss,
            metrics=[dice_coefficient, binary_dice_coefficient, 'accuracy']
        )
    
    # Test data loading
    try:
        logger.info("Testing data generators...")
        X_train, y_train = next(iter(train_generator))
        X_val, y_val = next(iter(val_generator))
        logger.info(f"Train batch: X={X_train.shape}, y={y_train.shape}")
        logger.info(f"Val batch: X={X_val.shape}, y={y_val.shape}")
        logger.info("✅ Data generators working correctly")
    except Exception as e:
        logger.error(f"Data generator test failed: {e}")
        return False
    
    # Train
    try:
        logger.info("Starting training...")
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=Config.EPOCHS,
            callbacks=create_callbacks(Config.CALLBACKS_DIR(timestamp)),
            verbose=1
        )
        
        logger.info("Training completed successfully!")
        
        # Save final model
        final_model_path = Config.MODEL_SAVE_PATH(timestamp)
        model.save(final_model_path)
        logger.info(f"Final model saved to: {final_model_path}")
        
        # Print summary
        final_train_dice = history.history['dice_coefficient'][-1]
        final_val_dice = history.history['val_dice_coefficient'][-1]
        best_val_dice = max(history.history['val_dice_coefficient'])
        
        logger.info("=" * 50)
        logger.info("TRAINING SUMMARY:")
        logger.info(f"Final training Dice: {final_train_dice:.4f}")
        logger.info(f"Final validation Dice: {final_val_dice:.4f}")
        logger.info(f"Best validation Dice: {best_val_dice:.4f}")
        logger.info(f"Model saved: {final_model_path}")
        logger.info(f"Best model: {Config.CALLBACKS_DIR(timestamp)}/best_model.h5")
        logger.info("=" * 50)
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False

if __name__ == "__main__":
    # Setup directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    try:
        success = train_model()
        if success:
            logger.info("🎉 FULL DATASET TRAINING COMPLETED SUCCESSFULLY!")
            logger.info("This model should work on all lesion sizes!")
        else:
            logger.error("❌ Training failed")
            exit(1)
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        exit(1)
