#!/usr/bin/env python3
"""
Corrected training script for stroke lesion segmentation using ATLAS dataset.
Fixed batch size issues and properly configured for your environment.
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/sota_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    DATA_DIR = "/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training_Split"
    INPUT_SHAPE = (128, 128, 128, 1)  # Reduced for memory efficiency
    BATCH_SIZE = 1  # Must be 1 for memory constraints
    EPOCHS = 100
    BASE_FILTERS = 32
    VALIDATION_SPLIT = 0.2
    LEARNING_RATE = 1e-4
    GRADIENT_ACCUM_STEPS = 2
    CALLBACKS_DIR = lambda timestamp: Path(f'callbacks/sota_{timestamp}')
    MODEL_SAVE_PATH = lambda timestamp: f'models/sota_final_{timestamp}.h5'

def configure_hardware():
    """Configure GPU with memory growth"""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # Use only first GPU and enable memory growth
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            logger.info(f"Using 1 GPU: {gpus[0]}")
            return 1  # batch size
        else:
            logger.warning("No GPUs detected, using CPU")
            return 1
    except Exception as e:
        logger.error(f"Hardware configuration failed: {e}")
        return 1

def setup_mixed_precision():
    """Enable mixed precision training"""
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    logger.info("Mixed precision enabled")

def resize_volume(volume, target_shape):
    """Resize 3D volume"""
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=1)

def combined_loss(y_true, y_pred, smooth=1e-6, focal_gamma=3.0, focal_alpha=0.25):
    """Combined Dice + Focal loss"""
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

def build_sota_model(input_shape=(128, 128, 128, 1), base_filters=32):
    """Build 3D U-Net with attention"""
    inputs = tf.keras.layers.Input(input_shape)
    
    # Encoder
    c1 = tf.keras.layers.Conv3D(base_filters, 3, padding='same')(inputs)
    c1 = tf.keras.layers.BatchNormalization()(c1)
    c1 = tf.keras.layers.Activation('relu')(c1)
    c1 = tf.keras.layers.Conv3D(base_filters, 3, padding='same')(c1)
    c1 = tf.keras.layers.BatchNormalization()(c1)
    c1 = tf.keras.layers.Activation('relu')(c1)
    p1 = tf.keras.layers.MaxPooling3D(2)(c1)
    
    c2 = tf.keras.layers.Conv3D(base_filters*2, 3, padding='same')(p1)
    c2 = tf.keras.layers.BatchNormalization()(c2)
    c2 = tf.keras.layers.Activation('relu')(c2)
    c2 = tf.keras.layers.Conv3D(base_filters*2, 3, padding='same')(c2)
    c2 = tf.keras.layers.BatchNormalization()(c2)
    c2 = tf.keras.layers.Activation('relu')(c2)
    p2 = tf.keras.layers.MaxPooling3D(2)(c2)
    
    c3 = tf.keras.layers.Conv3D(base_filters*4, 3, padding='same')(p2)
    c3 = tf.keras.layers.BatchNormalization()(c3)
    c3 = tf.keras.layers.Activation('relu')(c3)
    c3 = tf.keras.layers.Conv3D(base_filters*4, 3, padding='same')(c3)
    c3 = tf.keras.layers.BatchNormalization()(c3)
    c3 = tf.keras.layers.Activation('relu')(c3)
    p3 = tf.keras.layers.MaxPooling3D(2)(c3)
    
    c4 = tf.keras.layers.Conv3D(base_filters*8, 3, padding='same')(p3)
    c4 = tf.keras.layers.BatchNormalization()(c4)
    c4 = tf.keras.layers.Activation('relu')(c4)
    c4 = tf.keras.layers.Conv3D(base_filters*8, 3, padding='same')(c4)
    c4 = tf.keras.layers.BatchNormalization()(c4)
    c4 = tf.keras.layers.Activation('relu')(c4)
    p4 = tf.keras.layers.MaxPooling3D(2)(c4)
    
    # Bottleneck
    c5 = tf.keras.layers.Conv3D(base_filters*16, 3, padding='same')(p4)
    c5 = tf.keras.layers.BatchNormalization()(c5)
    c5 = tf.keras.layers.Activation('relu')(c5)
    c5 = tf.keras.layers.Conv3D(base_filters*16, 3, padding='same')(c5)
    c5 = tf.keras.layers.BatchNormalization()(c5)
    c5 = tf.keras.layers.Activation('relu')(c5)
    
    # Decoder
    u6 = tf.keras.layers.Conv3DTranspose(base_filters*8, 2, strides=2, padding='same')(c5)
    u6 = tf.keras.layers.Concatenate()([u6, c4])
    c6 = tf.keras.layers.Conv3D(base_filters*8, 3, padding='same')(u6)
    c6 = tf.keras.layers.BatchNormalization()(c6)
    c6 = tf.keras.layers.Activation('relu')(c6)
    c6 = tf.keras.layers.Conv3D(base_filters*8, 3, padding='same')(c6)
    c6 = tf.keras.layers.BatchNormalization()(c6)
    c6 = tf.keras.layers.Activation('relu')(c6)
    
    u7 = tf.keras.layers.Conv3DTranspose(base_filters*4, 2, strides=2, padding='same')(c6)
    u7 = tf.keras.layers.Concatenate()([u7, c3])
    c7 = tf.keras.layers.Conv3D(base_filters*4, 3, padding='same')(u7)
    c7 = tf.keras.layers.BatchNormalization()(c7)
    c7 = tf.keras.layers.Activation('relu')(c7)
    c7 = tf.keras.layers.Conv3D(base_filters*4, 3, padding='same')(c7)
    c7 = tf.keras.layers.BatchNormalization()(c7)
    c7 = tf.keras.layers.Activation('relu')(c7)
    
    u8 = tf.keras.layers.Conv3DTranspose(base_filters*2, 2, strides=2, padding='same')(c7)
    u8 = tf.keras.layers.Concatenate()([u8, c2])
    c8 = tf.keras.layers.Conv3D(base_filters*2, 3, padding='same')(u8)
    c8 = tf.keras.layers.BatchNormalization()(c8)
    c8 = tf.keras.layers.Activation('relu')(c8)
    c8 = tf.keras.layers.Conv3D(base_filters*2, 3, padding='same')(c8)
    c8 = tf.keras.layers.BatchNormalization()(c8)
    c8 = tf.keras.layers.Activation('relu')(c8)
    
    u9 = tf.keras.layers.Conv3DTranspose(base_filters, 2, strides=2, padding='same')(c8)
    u9 = tf.keras.layers.Concatenate()([u9, c1])
    c9 = tf.keras.layers.Conv3D(base_filters, 3, padding='same')(u9)
    c9 = tf.keras.layers.BatchNormalization()(c9)
    c9 = tf.keras.layers.Activation('relu')(c9)
    c9 = tf.keras.layers.Conv3D(base_filters, 3, padding='same')(c9)
    c9 = tf.keras.layers.BatchNormalization()(c9)
    c9 = tf.keras.layers.Activation('relu')(c9)
    
    outputs = tf.keras.layers.Conv3D(1, 1, activation='sigmoid', dtype='float32')(c9)
    
    model = tf.keras.models.Model(inputs, outputs, name="SOTA_UNet")
    return model

class AtlasDataGenerator(tf.keras.utils.Sequence):
    """Data generator for ATLAS dataset"""
    def __init__(self, data_dir, image_ids, batch_size, target_shape, shuffle=True):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "Images")
        self.masks_dir = os.path.join(data_dir, "Masks")
        self.image_ids = image_ids
        self.batch_size = batch_size
        self.target_shape = target_shape
        self.shuffle = shuffle
        self.indexes = np.arange(len(image_ids))
        logger.info(f"Initialized generator with {len(image_ids)} samples")
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
                
                # Load data
                img_data = nib.load(img_path).get_fdata(dtype=np.float32)
                mask_data = nib.load(mask_path).get_fdata(dtype=np.float32)
                
                # Resize if needed
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
                
            except Exception as e:
                logger.error(f"Error loading {img_id}: {e}")
                X[i] = np.zeros((*self.target_shape, 1), dtype=np.float32)
                y[i] = np.zeros((*self.target_shape, 1), dtype=np.float32)
        
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

def load_image_ids(data_dir):
    """Load valid image IDs"""
    images_dir = os.path.join(data_dir, "Images")
    masks_dir = os.path.join(data_dir, "Masks")
    
    image_ids = []
    for fname in os.listdir(images_dir):
        if fname.endswith("_space-MNI152NLin2009aSym_T1w.nii.gz"):
            img_id = fname.replace("_space-MNI152NLin2009aSym_T1w.nii.gz", "")
            mask_fname = f"{img_id}_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz"
            if os.path.exists(os.path.join(masks_dir, mask_fname)):
                image_ids.append(img_id)
    
    logger.info(f"Found {len(image_ids)} valid image/mask pairs")
    return image_ids

def create_callbacks(callbacks_dir):
    """Create training callbacks"""
    callbacks_dir.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            str(callbacks_dir / 'best_model.h5'),
            monitor='val_dice_coefficient',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_dice_coefficient',
            mode='max',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(str(callbacks_dir / 'training_log.csv'))
    ]

def train_model():
    """Main training function"""
    logger.info("Starting SOTA Stroke Lesion Segmentation Training")
    
    # Setup
    setup_mixed_precision()
    batch_size = configure_hardware()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Load data
    image_ids = load_image_ids(Config.DATA_DIR)
    np.random.shuffle(image_ids)
    
    # Split data
    train_size = int((1 - Config.VALIDATION_SPLIT) * len(image_ids))
    train_ids = image_ids[:train_size]
    val_ids = image_ids[train_size:]
    
    logger.info(f"Training samples: {len(train_ids)}")
    logger.info(f"Validation samples: {len(val_ids)}")
    
    # Create generators
    train_generator = AtlasDataGenerator(
        Config.DATA_DIR, train_ids, batch_size, Config.INPUT_SHAPE[:-1], shuffle=True
    )
    val_generator = AtlasDataGenerator(
        Config.DATA_DIR, val_ids, batch_size, Config.INPUT_SHAPE[:-1], shuffle=False
    )
    
    # Build model
    model = build_sota_model(input_shape=Config.INPUT_SHAPE, base_filters=Config.BASE_FILTERS)
    logger.info(f"Model parameters: {model.count_params():,}")
    
    # Compile with mixed precision optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE)
    if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    model.compile(
        optimizer=optimizer,
        loss=combined_loss,
        metrics=[dice_coefficient, 'accuracy']
    )
    
    # Train
    try:
        model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=Config.EPOCHS,
            callbacks=create_callbacks(Config.CALLBACKS_DIR(timestamp)),
            verbose=1
        )
        
        logger.info("Training completed successfully")
        model.save(Config.MODEL_SAVE_PATH(timestamp))
        logger.info(f"Model saved to: {Config.MODEL_SAVE_PATH(timestamp)}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('callbacks', exist_ok=True)
    
    try:
        train_model()
        logger.info("Training pipeline executed successfully")
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise SystemExit(1)
