#!/usr/bin/env python3
"""
FINE-TUNE Multi-GPU Advanced SOTA Model from Checkpoint
Resume from 72% Dice checkpoint with anti-overfitting strategies
Target: Improve validation performance from 45% to 55-65%
"""

import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import logging
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import nibabel as nib
from pathlib import Path
from datetime import datetime
from scipy.ndimage import zoom, gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from sklearn.model_selection import train_test_split

# Import working functions
from correct_full_training import load_full_655_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/fine_tune_multi_gpu.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FineTuneConfig:
    DATA_DIR = "/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training"
    
    # CHECKPOINT LOADING
    CHECKPOINT_PATH = "callbacks/multi_gpu_advanced_sota_20250624_044219/best_model.h5"
    LOAD_FROM_CHECKPOINT = True
    
    # Fine-tuning parameters (much more conservative)
    INPUT_SHAPE = (192, 224, 176, 1)
    BATCH_SIZE = 4     # Keep same as original
    
    # FINE-TUNING SPECIFIC SETTINGS
    EPOCHS = 30                    # Shorter run
    INITIAL_LR = 5e-6             # Much lower LR (was 3e-4)
    MIN_LR = 1e-8                 # Lower minimum
    WARMUP_EPOCHS = 0             # No warmup needed
    WEIGHT_DECAY = 1e-4           # L2 regularization
    
    # Anti-overfitting
    EARLY_STOPPING_PATIENCE = 12  # Stop sooner
    LR_REDUCE_PATIENCE = 6        # Reduce LR faster
    VALIDATION_SPLIT = 0.15
    
    # Enhanced regularization
    DROPOUT_AUGMENTATION = True    # Add dropout-like augmentation
    STRONGER_AUGMENTATION = True   # More aggressive augmentation
    MIXUP_ALPHA = 0.2             # Data mixing regularization
    
    CALLBACKS_DIR = lambda timestamp: Path(f'callbacks/fine_tune_{timestamp}')
    MODEL_SAVE_PATH = lambda timestamp: f'models/fine_tuned_{timestamp}.h5'

def configure_multi_gpu_strategy():
    """Configure MirroredStrategy for multi-GPU training"""
    gpus = tf.config.list_physical_devices('GPU')
    logger.info(f"Detected GPUs: {len(gpus)}")
    
    for i, gpu in enumerate(gpus):
        logger.info(f"GPU {i}: {gpu}")
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            logger.warning(f"Could not set memory growth for GPU {i}: {e}")
    
    strategy = tf.distribute.MirroredStrategy()
    logger.info(f"🚀 MirroredStrategy initialized with {strategy.num_replicas_in_sync} devices")
    logger.info(f"🔄 Fine-tuning batch size: {FineTuneConfig.BATCH_SIZE} (global)")
    
    return strategy

def load_model_from_checkpoint(checkpoint_path, strategy):
    """Load the trained model from checkpoint"""
    
    with strategy.scope():
        logger.info(f"🔄 Loading checkpoint from: {checkpoint_path}")
        
        try:
            # Define custom objects that might be in the saved model
            custom_objects = {
                'dice_coefficient': dice_coefficient,
                'binary_dice_coefficient': binary_dice_coefficient,
                'ultimate_loss': ultimate_loss,
                'ultimate_loss_function': ultimate_loss  # Alternative name
            }
            
            model = tf.keras.models.load_model(
                checkpoint_path,
                compile=False,  # We'll recompile with fine-tuning settings
                custom_objects=custom_objects
            )
            
            param_count = model.count_params()
            logger.info(f"✅ Successfully loaded model with {param_count:,} parameters")
            logger.info(f"📊 Model architecture: {model.name}")
            
            # Recompile with fine-tuning optimized settings
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=FineTuneConfig.INITIAL_LR,
                weight_decay=FineTuneConfig.WEIGHT_DECAY
            )
            
            model.compile(
                optimizer=optimizer,
                loss=ultimate_loss,
                metrics=['accuracy', dice_coefficient, binary_dice_coefficient]
            )
            
            logger.info("🔧 Model recompiled with fine-tuning parameters")
            logger.info(f"   Learning rate: {FineTuneConfig.INITIAL_LR}")
            logger.info(f"   Weight decay: {FineTuneConfig.WEIGHT_DECAY}")
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Error loading checkpoint: {e}")
            logger.info("💡 Available checkpoints:")
            import subprocess
            subprocess.run(["find", "callbacks/", "-name", "best_model.h5", "-exec", "ls", "-la", "{}", ";"])
            return None

class EnhancedDataGenerator(tf.keras.utils.Sequence):
    """Enhanced data generator with stronger regularization"""
    
    def __init__(self, image_mask_pairs, batch_size, target_shape, shuffle=True, augment=True, stronger_aug=False):
        self.image_mask_pairs = image_mask_pairs
        self.batch_size = batch_size
        self.target_shape = target_shape
        self.shuffle = shuffle
        self.augment = augment
        self.stronger_aug = stronger_aug
        self.indexes = np.arange(len(image_mask_pairs))
        
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
                
                # Load and process
                img_data = nib.load(img_path).get_fdata(dtype=np.float32)
                mask_data = nib.load(mask_path).get_fdata(dtype=np.float32)
                
                # Resize if needed
                if img_data.shape != self.target_shape:
                    img_data = self.resize_volume(img_data, self.target_shape)
                    mask_data = self.resize_volume(mask_data, self.target_shape)
                    mask_data = (mask_data > 0.5).astype(np.float32)
                
                # Robust normalization
                img_data = self.normalize(img_data)
                
                # Enhanced augmentation for fine-tuning
                if self.augment:
                    img_data, mask_data = self.enhanced_augmentation(img_data, mask_data)
                
                X[i] = img_data[..., np.newaxis]
                y[i] = mask_data[..., np.newaxis]
                
            except Exception as e:
                logger.error(f"Error loading sample {idx}: {e}")
                X[i] = np.zeros((*self.target_shape, 1), dtype=np.float32)
                y[i] = np.zeros((*self.target_shape, 1), dtype=np.float32)
        
        # Apply mixup augmentation
        if self.augment and FineTuneConfig.MIXUP_ALPHA > 0:
            X, y = self.mixup(X, y, FineTuneConfig.MIXUP_ALPHA)
        
        return X, y
    
    def resize_volume(self, volume, target_shape):
        factors = [t / s for t, s in zip(target_shape, volume.shape)]
        return zoom(volume, factors, order=1)
    
    def normalize(self, img):
        p1, p99 = np.percentile(img[img > 0], [0.5, 99.5])
        img = np.clip(img, p1, p99)
        mean = np.mean(img[img > 0])
        std = np.std(img[img > 0])
        img = (img - mean) / (std + 1e-8)
        return (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    def enhanced_augmentation(self, img, mask):
        """Stronger augmentation for better generalization"""
        
        # Spatial augmentations (stronger)
        if np.random.rand() > 0.4:  # More frequent
            img = np.flip(img, axis=1)
            mask = np.flip(mask, axis=1)
        
        if np.random.rand() > 0.6:  # More frequent rotation
            from scipy.ndimage import rotate
            angle = np.random.uniform(-20, 20)  # Stronger rotation
            img = rotate(img, angle, axes=(0, 1), reshape=False, order=1)
            mask = rotate(mask, angle, axes=(0, 1), reshape=False, order=0)
        
        # Enhanced intensity augmentation
        if np.random.rand() > 0.5:  # More frequent
            # Gamma correction
            gamma = np.random.uniform(0.7, 1.3)  # Stronger range
            img = np.power(img, gamma)
            
            # Stronger noise
            noise = np.random.normal(0, 0.03, img.shape)  # Increased noise
            img = np.clip(img + noise, 0, 1)
            
            # Brightness/contrast
            brightness = np.random.uniform(-0.1, 0.1)
            contrast = np.random.uniform(0.8, 1.2)
            img = np.clip(contrast * img + brightness, 0, 1)
        
        # Elastic deformation (new for fine-tuning)
        if np.random.rand() > 0.8 and self.stronger_aug:
            img, mask = self.elastic_deformation(img, mask)
        
        return img, mask
    
    def elastic_deformation(self, img, mask, alpha=100, sigma=10):
        """Apply elastic deformation for stronger augmentation"""
        shape = img.shape
        
        # Generate random displacement fields
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dz = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        
        # Create coordinate arrays
        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]), indexing='ij')
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z + dz, (-1, 1))
        
        # Apply deformation
        img_deformed = map_coordinates(img, indices, order=1, mode='reflect').reshape(shape)
        mask_deformed = map_coordinates(mask, indices, order=0, mode='reflect').reshape(shape)
        
        return img_deformed, mask_deformed
    
    def mixup(self, X, y, alpha):
        """Apply mixup augmentation for regularization"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
            batch_size = X.shape[0]
            index = np.random.permutation(batch_size)
            
            mixed_X = lam * X + (1 - lam) * X[index, :]
            mixed_y = lam * y + (1 - lam) * y[index, :]
            
            return mixed_X, mixed_y
        return X, y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

def fine_tune_schedule(epoch, initial_lr=5e-6, decay_rate=0.92):
    """Learning rate schedule for fine-tuning"""
    return initial_lr * (decay_rate ** epoch)

def create_fine_tuning_callbacks(callbacks_dir):
    """Create callbacks optimized for fine-tuning"""
    callbacks_dir.mkdir(parents=True, exist_ok=True)
    
    return [
        # Aggressive early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_dice_coefficient',
            mode='max',
            patience=FineTuneConfig.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce LR on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_dice_coefficient',
            factor=0.5,
            patience=FineTuneConfig.LR_REDUCE_PATIENCE,
            min_lr=FineTuneConfig.MIN_LR,
            verbose=1,
            mode='max'
        ),
        
        # Save best fine-tuned model
        tf.keras.callbacks.ModelCheckpoint(
            str(callbacks_dir / 'best_model.h5'),
            monitor='val_dice_coefficient',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        
        # Learning rate scheduler
        tf.keras.callbacks.LearningRateScheduler(
            fine_tune_schedule,
            verbose=1
        ),
        
        # CSV logging
        tf.keras.callbacks.CSVLogger(str(callbacks_dir / 'training_log.csv')),
        
        # TensorBoard
        tf.keras.callbacks.TensorBoard(
            log_dir=str(callbacks_dir / 'tensorboard'),
            histogram_freq=0,
            write_graph=False,
            update_freq='epoch'
        )
    ]

# Copy metric functions from original script
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Dice coefficient metric"""
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred, tf.float32))
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def binary_dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Binary dice coefficient metric"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def ultimate_loss(y_true, y_pred, smooth=1e-6):
    """Ultimate loss function combining multiple objectives"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Dice loss
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice_loss = 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    
    # Enhanced Focal loss
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    focal_loss = -tf.reduce_mean(
        0.25 * y_true * tf.pow(1 - y_pred, 3) * tf.math.log(y_pred) +
        0.75 * (1 - y_true) * tf.pow(y_pred, 3) * tf.math.log(1 - y_pred)
    )
    
    # Tversky loss for better recall
    alpha, beta = 0.3, 0.7
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    tversky_loss = 1 - (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    
    # Sensitivity loss (recall)
    sensitivity = tp / (tp + fn + smooth)
    sensitivity_loss = 1 - sensitivity
    
    return 0.3 * dice_loss + 0.3 * focal_loss + 0.25 * tversky_loss + 0.15 * sensitivity_loss

def fine_tune_model():
    """Main fine-tuning function"""
    logger.info("🔄 FINE-TUNING FROM 72% DICE CHECKPOINT")
    logger.info("=" * 60)
    logger.info("Strategy: Load best checkpoint + anti-overfitting")
    logger.info("Target: Improve validation from 45% to 55-65%")
    logger.info("=" * 60)
    
    # Setup
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # Configure multi-GPU strategy
    strategy = configure_multi_gpu_strategy()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Load model from checkpoint
    model = load_model_from_checkpoint(FineTuneConfig.CHECKPOINT_PATH, strategy)
    if model is None:
        logger.error("❌ Failed to load checkpoint. Exiting.")
        return None, None
    
    # Load dataset
    all_pairs = load_full_655_dataset()
    
    # Use same split as original (important for comparison)
    train_pairs, val_pairs = train_test_split(
        all_pairs,
        test_size=FineTuneConfig.VALIDATION_SPLIT,
        random_state=42,  # Same seed as original
        shuffle=True
    )
    
    logger.info(f"Fine-tuning split: {len(train_pairs)} train, {len(val_pairs)} validation")
    
    # Create enhanced data generators
    train_generator = EnhancedDataGenerator(
        train_pairs,
        FineTuneConfig.BATCH_SIZE,
        FineTuneConfig.INPUT_SHAPE[:-1],
        shuffle=True,
        augment=True,
        stronger_aug=FineTuneConfig.STRONGER_AUGMENTATION
    )
    
    val_generator = EnhancedDataGenerator(
        val_pairs,
        FineTuneConfig.BATCH_SIZE,
        FineTuneConfig.INPUT_SHAPE[:-1],
        shuffle=False,
        augment=False,
        stronger_aug=False
    )
    
    # Fine-tune the model
    logger.info("🔥 Starting fine-tuning training...")
    logger.info(f"   Learning rate: {FineTuneConfig.INITIAL_LR}")
    logger.info(f"   Epochs: {FineTuneConfig.EPOCHS}")
    logger.info(f"   Early stopping patience: {FineTuneConfig.EARLY_STOPPING_PATIENCE}")
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=FineTuneConfig.EPOCHS,
        callbacks=create_fine_tuning_callbacks(
            FineTuneConfig.CALLBACKS_DIR(timestamp)
        ),
        verbose=1
    )
    
    # Save final model
    final_model_path = FineTuneConfig.MODEL_SAVE_PATH(timestamp)
    model.save(final_model_path)
    
    # Results
    final_val_dice = history.history['val_dice_coefficient'][-1]
    best_val_dice = max(history.history['val_dice_coefficient'])
    improvement = best_val_dice - 0.459  # Assuming original validation was ~45.9%
    
    logger.info("=" * 60)
    logger.info("🏆 FINE-TUNING COMPLETED!")
    logger.info("=" * 60)
    logger.info(f"Original validation Dice: ~45.9%")
    logger.info(f"Final validation Dice: {final_val_dice:.4f}")
    logger.info(f"Best validation Dice: {best_val_dice:.4f}")
    logger.info(f"Improvement: {improvement:.3f}")
    
    if best_val_dice > 0.65:
        logger.info("🚀 EXCELLENT IMPROVEMENT (>65% validation)!")
    elif best_val_dice > 0.55:
        logger.info("✅ SIGNIFICANT IMPROVEMENT (>55% validation)!")
    elif best_val_dice > 0.50:
        logger.info("👍 GOOD IMPROVEMENT (>50% validation)!")
    
    logger.info(f"Best model: {FineTuneConfig.CALLBACKS_DIR(timestamp)}/best_model.h5")
    logger.info("=" * 60)
    
    return history, model

if __name__ == "__main__":
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    try:
        history, model = fine_tune_model()
        if history and model:
            logger.info("🎉 FINE-TUNING SUCCESS!")
        else:
            logger.error("❌ Fine-tuning failed")
            exit(1)
        
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
