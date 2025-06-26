#!/usr/bin/env python3
"""
CORRECT training script using the FULL 655-image dataset
Uses /mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training/ as you specified
Same architecture as working model but on full dataset with validation split
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
from sklearn.model_selection import train_test_split

# Import the SAME model architecture that worked
from working_sota_model import build_sota_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/correct_full_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    # CORRECT path - the full 655-image dataset you specified
    DATA_DIR = "/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training"
    
    # SAME settings as working model
    INPUT_SHAPE = (192, 224, 176, 1)
    BATCH_SIZE = 2  # As you requested
    EPOCHS = 50
    BASE_FILTERS = 16  # Increased from 8 to get ~5.7M parameters like working model
    VALIDATION_SPLIT = 0.1  # 10% validation as you requested
    LEARNING_RATE = 1e-4
    
    CALLBACKS_DIR = lambda timestamp: Path(f'callbacks/correct_full_{timestamp}')
    MODEL_SAVE_PATH = lambda timestamp: f'models/correct_full_{timestamp}.h5'

def configure_hardware():
    """Configure GPU - same as working model"""
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
    """Enable mixed precision - same as working model"""
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        logger.info("Mixed precision enabled: mixed_float16")
    except Exception as e:
        logger.error(f"Failed to enable mixed precision: {e}")

def resize_volume(volume, target_shape):
    """Resize volume - EXACT same as working model"""
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=1)

def combined_loss(y_true, y_pred, smooth=1e-6, focal_gamma=3.0, focal_alpha=0.25):
    """Combined loss - EXACT same as working model"""
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
    """Dice coefficient - same as working model"""
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred, tf.float32))
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def binary_dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Binary dice coefficient - same as working model"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

class CorrectAtlasDataGenerator(tf.keras.utils.Sequence):
    """Data generator for the FULL 655-image dataset with epoch shuffling"""
    def __init__(self, image_mask_pairs, batch_size, target_shape, shuffle=True):
        self.image_mask_pairs = image_mask_pairs
        self.batch_size = batch_size
        self.target_shape = target_shape
        self.shuffle = shuffle
        self.indexes = np.arange(len(image_mask_pairs))
        
        # Analyze dataset diversity
        logger.info(f"Dataset: {len(image_mask_pairs)} samples")
        
        # Sample lesion sizes to verify diversity
        sample_sizes = []
        for i in range(0, min(50, len(image_mask_pairs)), 5):  # Sample every 5th case
            try:
                _, mask_path = image_mask_pairs[i]
                mask_data = nib.load(mask_path).get_fdata()
                size = np.sum(mask_data)
                sample_sizes.append(size)
            except:
                pass
        
        if sample_sizes:
            sample_sizes = np.array(sample_sizes)
            logger.info(f"Sample lesion sizes - Min: {sample_sizes.min():,.0f}, Max: {sample_sizes.max():,.0f}, Mean: {sample_sizes.mean():,.0f}")
        
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
                
                # Resize if needed - EXACT same as working model
                if img_data.shape != self.target_shape:
                    img_data = resize_volume(img_data, self.target_shape)
                    mask_data = resize_volume(mask_data, self.target_shape)
                    mask_data = (mask_data > 0.5).astype(np.float32)
                
                # Intensity normalization - EXACT same as working model
                p1, p99 = np.percentile(img_data[img_data > 0], [1, 99])
                img_data = np.clip(img_data, p1, p99)
                img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min() + 1e-8)
                
                X[i] = img_data[..., np.newaxis]
                y[i] = mask_data[..., np.newaxis]
                
                # Data augmentation - EXACT same as working model
                if np.random.rand() > 0.5:
                    X[i] = np.flip(X[i], axis=1)
                    y[i] = np.flip(y[i], axis=1)
                    
            except Exception as e:
                logger.error(f"Error loading sample {idx}: {e}")
                X[i] = np.zeros((*self.target_shape, 1), dtype=np.float32)
                y[i] = np.zeros((*self.target_shape, 1), dtype=np.float32)
        
        return X, y

    def on_epoch_end(self):
        """Shuffle at end of each epoch as you requested"""
        if self.shuffle:
            np.random.shuffle(self.indexes)
            logger.info("Shuffled dataset for new epoch")

def load_full_655_dataset():
    """Load all 655 image/mask pairs from the correct directory you specified"""
    data_dir = Config.DATA_DIR
    images_dir = os.path.join(data_dir, "Images")
    masks_dir = os.path.join(data_dir, "Masks")
    
    logger.info(f"Loading dataset from: {data_dir}")
    
    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        raise FileNotFoundError(f"Images or Masks directory not found in {data_dir}")
    
    # Get all image files
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.nii.gz')])
    
    all_pairs = []
    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        
        # Find corresponding mask (handle different naming patterns)
        base_name = img_file.replace('.nii.gz', '')
        
        # Try different mask naming patterns
        possible_mask_names = [
            base_name + '_mask.nii.gz',
            base_name + '_label-L_desc-T1lesion_mask.nii.gz',
            base_name.replace('_T1w', '_label-L_desc-T1lesion_mask') + '.nii.gz',
            # Add more patterns if needed
        ]
        
        mask_path = None
        for mask_name in possible_mask_names:
            potential_path = os.path.join(masks_dir, mask_name)
            if os.path.exists(potential_path):
                mask_path = potential_path
                break
        
        if mask_path:
            all_pairs.append((img_path, mask_path))
        else:
            logger.warning(f"No mask found for {img_file}")
    
    logger.info(f"Loaded {len(all_pairs)} image/mask pairs from full dataset")
    
    if len(all_pairs) != 655:
        logger.warning(f"Expected 655 pairs, got {len(all_pairs)}")
    
    return all_pairs

def create_callbacks(callbacks_dir):
    """Create callbacks - same as working model"""
    callbacks_dir.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            str(callbacks_dir / 'best_model.h5'),
            monitor='val_dice_coefficient',
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
    logger.info("CORRECT FULL DATASET TRAINING")
    logger.info("=" * 70)
    logger.info(f"Dataset: {Config.DATA_DIR}")
    logger.info(f"Expected samples: 655")
    logger.info(f"Batch size: {Config.BATCH_SIZE} (with epoch shuffling)")
    logger.info(f"Validation split: {Config.VALIDATION_SPLIT} (10%)")
    logger.info(f"Architecture: State-of-the-art with {Config.BASE_FILTERS} base filters (~5.7M params)")
    logger.info("=" * 70)
    
    setup_mixed_precision()
    strategy = configure_hardware()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Load the FULL 655-image dataset
    all_pairs = load_full_655_dataset()
    
    if len(all_pairs) < 600:
        logger.error(f"Not enough data found! Expected ~655, got {len(all_pairs)}")
        return False
    
def pre_training_verification(all_pairs, train_pairs, val_pairs):
    """Comprehensive verification before starting training"""
    logger.info("=" * 70)
    logger.info("🔍 PRE-TRAINING VERIFICATION")
    logger.info("=" * 70)
    
    verification_passed = True
    
    # 1. Dataset verification
    logger.info("1. DATASET VERIFICATION:")
    logger.info(f"   Total pairs found: {len(all_pairs)}")
    logger.info(f"   Training pairs: {len(train_pairs)} ({len(train_pairs)/len(all_pairs)*100:.1f}%)")
    logger.info(f"   Validation pairs: {len(val_pairs)} ({len(val_pairs)/len(all_pairs)*100:.1f}%)")
    
    if len(all_pairs) != 655:
        logger.error(f"   ❌ Expected 655 pairs, got {len(all_pairs)}")
        verification_passed = False
    else:
        logger.info("   ✅ Dataset size correct")
    
    # Check for overlap
    train_files = {os.path.basename(p[0]) for p in train_pairs}
    val_files = {os.path.basename(p[0]) for p in val_pairs}
    overlap = train_files.intersection(val_files)
    if overlap:
        logger.error(f"   ❌ {len(overlap)} files in both train and validation!")
        verification_passed = False
    else:
        logger.info("   ✅ No overlap between train/validation")
    
    # 2. Model architecture verification
    logger.info("\n2. MODEL ARCHITECTURE VERIFICATION:")
    try:
        test_model = build_sota_model(input_shape=Config.INPUT_SHAPE, base_filters=Config.BASE_FILTERS)
        param_count = test_model.count_params()
        logger.info(f"   Model parameters: {param_count:,}")
        
        if 5000000 <= param_count <= 6000000:
            logger.info("   ✅ Parameter count in expected range (5M-6M)")
        else:
            logger.warning(f"   ⚠️ Parameter count outside expected range")
            if param_count < 2000000:
                logger.error("   ❌ Model too small!")
                verification_passed = False
        
        # Test forward pass
        test_input = tf.random.normal((1, *Config.INPUT_SHAPE))
        test_output = test_model(test_input, training=False)
        logger.info(f"   ✅ Forward pass successful: {test_input.shape} → {test_output.shape}")
        
        del test_model, test_input, test_output
        tf.keras.backend.clear_session()
        
    except Exception as e:
        logger.error(f"   ❌ Model verification failed: {e}")
        verification_passed = False
    
    # 3. Sample data loading test
    logger.info("\n3. SAMPLE DATA LOADING TEST:")
    try:
        # Test loading 3 samples
        test_indices = [0, len(all_pairs)//2, len(all_pairs)-1]
        sample_sizes = []
        
        for i, idx in enumerate(test_indices):
            img_path, mask_path = all_pairs[idx]
            case_name = os.path.basename(img_path).split('_space')[0]
            
            img_data = nib.load(img_path).get_fdata(dtype=np.float32)
            mask_data = nib.load(mask_path).get_fdata(dtype=np.float32)
            lesion_size = np.sum(mask_data)
            sample_sizes.append(lesion_size)
            
            logger.info(f"   Sample {i+1} ({case_name}): {img_data.shape}, {lesion_size:,.0f} lesion voxels")
            
            if img_data.shape != mask_data.shape:
                logger.error(f"   ❌ Shape mismatch in {case_name}")
                verification_passed = False
        
        # Check lesion size diversity
        sample_sizes = np.array(sample_sizes)
        if sample_sizes.max() / sample_sizes.min() > 10:
            logger.info(f"   ✅ Good lesion size diversity: {sample_sizes.min():,.0f} to {sample_sizes.max():,.0f}")
        else:
            logger.warning(f"   ⚠️ Limited lesion size diversity")
        
    except Exception as e:
        logger.error(f"   ❌ Data loading test failed: {e}")
        verification_passed = False
    
    # 4. Memory estimation
    logger.info("\n4. MEMORY ESTIMATION:")
    try:
        # Estimate memory usage
        batch_size = Config.BATCH_SIZE
        input_size = np.prod(Config.INPUT_SHAPE) * 4 * batch_size  # float32
        output_size = input_size  # same shape
        model_params = param_count * 4  # float32
        
        estimated_gb = (input_size + output_size + model_params * 2) / (1024**3)  # *2 for gradients
        
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   Estimated GPU memory: {estimated_gb:.1f} GB")
        
        if estimated_gb > 20:
            logger.warning(f"   ⚠️ High memory usage, may cause OOM")
        else:
            logger.info("   ✅ Memory usage reasonable")
            
    except Exception as e:
        logger.warning(f"   ⚠️ Memory estimation failed: {e}")
    
    # 5. Time estimation
    logger.info("\n5. TRAINING TIME ESTIMATION:")
    train_batches = len(train_pairs) // Config.BATCH_SIZE
    val_batches = len(val_pairs) // Config.BATCH_SIZE
    seconds_per_batch = 3  # Conservative estimate
    
    time_per_epoch = (train_batches + val_batches) * seconds_per_batch
    total_hours = (time_per_epoch * Config.EPOCHS) / 3600
    
    logger.info(f"   Train batches per epoch: {train_batches}")
    logger.info(f"   Val batches per epoch: {val_batches}")
    logger.info(f"   Estimated time per epoch: {time_per_epoch/60:.1f} minutes")
    logger.info(f"   Estimated total time: {total_hours:.1f} hours")
    
    if total_hours > 20:
        logger.warning(f"   ⚠️ May exceed 24-hour SLURM limit")
    
    # Final verdict
    logger.info("\n" + "=" * 70)
    if verification_passed:
        logger.info("🎉 ALL VERIFICATIONS PASSED!")
        logger.info("✅ Ready to start training")
        logger.info("=" * 70)
        
        # Final confirmation log
        logger.info("FINAL TRAINING CONFIGURATION:")
        logger.info(f"  Dataset: {len(all_pairs)} total cases")
        logger.info(f"  Training: {len(train_pairs)} cases ({len(train_pairs)/len(all_pairs)*100:.1f}%)")
        logger.info(f"  Validation: {len(val_pairs)} cases ({len(val_pairs)/len(all_pairs)*100:.1f}%)")
        logger.info(f"  Model: {param_count:,} parameters")
        logger.info(f"  Batch size: {Config.BATCH_SIZE}")
        logger.info(f"  Input shape: {Config.INPUT_SHAPE}")
        logger.info(f"  Estimated time: {total_hours:.1f} hours")
        logger.info("=" * 70)
        
        return True
    else:
        logger.error("❌ VERIFICATION FAILED!")
        logger.error("🛑 STOPPING BEFORE TRAINING")
        logger.error("Fix the issues above first")
        logger.info("=" * 70)
        return False
    # Split into train/validation (90/10) with random state for reproducibility
    train_pairs, val_pairs = train_test_split(
        all_pairs,
        test_size=Config.VALIDATION_SPLIT,
        random_state=42,
        shuffle=True
    )
    
    # RUN COMPREHENSIVE VERIFICATION BEFORE TRAINING
    if not pre_training_verification(all_pairs, train_pairs, val_pairs):
        logger.error("Pre-training verification failed!")
        return False
    
    # Create data generators with epoch shuffling
    train_generator = CorrectAtlasDataGenerator(
        train_pairs, Config.BATCH_SIZE, Config.INPUT_SHAPE[:-1], shuffle=True
    )
    val_generator = CorrectAtlasDataGenerator(
        val_pairs, Config.BATCH_SIZE, Config.INPUT_SHAPE[:-1], shuffle=False
    )
    
    # Build model with SAME architecture as working model
    with strategy.scope():
        model = build_sota_model(input_shape=Config.INPUT_SHAPE, base_filters=Config.BASE_FILTERS)
        
        # Verify we have the right number of parameters
        param_count = model.count_params()
        logger.info(f"Model parameters: {param_count:,}")
        
        if param_count < 5000000:  # Should be around 5.7M like working model
            logger.warning(f"Parameter count seems low! Expected ~5.7M, got {param_count:,}")
        
        # Compile with same optimizer as working model
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
        logger.info("Starting training on FULL 655-image dataset...")
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
        
        # Print summary
        final_train_dice = history.history['dice_coefficient'][-1]
        final_val_dice = history.history['val_dice_coefficient'][-1]
        best_val_dice = max(history.history['val_dice_coefficient'])
        
        logger.info("=" * 70)
        logger.info("TRAINING SUMMARY:")
        logger.info(f"Dataset used: {len(all_pairs)} samples (FULL dataset)")
        logger.info(f"Training samples: {len(train_pairs)}")
        logger.info(f"Validation samples: {len(val_pairs)}")
        logger.info(f"Model parameters: {param_count:,}")
        logger.info(f"Final training Dice: {final_train_dice:.4f}")
        logger.info(f"Final validation Dice: {final_val_dice:.4f}")
        logger.info(f"Best validation Dice: {best_val_dice:.4f}")
        logger.info(f"Best model: {Config.CALLBACKS_DIR(timestamp)}/best_model.h5")
        logger.info("=" * 70)
        
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
            logger.info("🎉 CORRECT FULL DATASET TRAINING COMPLETED!")
            logger.info("This model was trained on all 655 images with proper validation!")
        else:
            logger.error("❌ Training failed")
            exit(1)
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        exit(1)
