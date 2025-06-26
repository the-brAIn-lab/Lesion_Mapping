#!/usr/bin/env python3
"""
MULTI-GPU Advanced SOTA Training
Uses TensorFlow MirroredStrategy across 4 GPUs for massive memory capacity
Target: 75-80% Dice with full-scale architecture
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
        logging.FileHandler('logs/multi_gpu_advanced_sota.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MultiGPUAdvancedConfig:
    DATA_DIR = "/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training"
    
    # MULTI-GPU SCALING
    INPUT_SHAPE = (192, 224, 176, 1)
    BASE_FILTERS = 32  # MASSIVE capacity with 4 GPUs
    BATCH_SIZE = 4     # 1 sample per GPU (4 total)
    
    # Advanced features enabled
    USE_SWIN_BLOCKS = False
    USE_DEEP_SUPERVISION = False
    
    # Training parameters
    EPOCHS = 100
    VALIDATION_SPLIT = 0.15
    INITIAL_LR = 3e-4  # Higher LR for larger batch
    MIN_LR = 1e-7
    WARMUP_EPOCHS = 8
    
    # Augmentation
    STRONG_AUGMENTATION = True
    MIXUP_ALPHA = 0.3
    
    CALLBACKS_DIR = lambda timestamp: Path(f'callbacks/multi_gpu_advanced_sota_{timestamp}')
    MODEL_SAVE_PATH = lambda timestamp: f'models/multi_gpu_advanced_sota_{timestamp}.h5'

def configure_multi_gpu_strategy():
    """Configure MirroredStrategy for multi-GPU training"""
    # Ensure all GPUs are visible
    gpus = tf.config.list_physical_devices('GPU')
    logger.info(f"Detected GPUs: {len(gpus)}")
    
    for i, gpu in enumerate(gpus):
        logger.info(f"GPU {i}: {gpu}")
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            logger.warning(f"Could not set memory growth for GPU {i}: {e}")
    
    # Create MirroredStrategy
    strategy = tf.distribute.MirroredStrategy()
    logger.info(f"🚀 MirroredStrategy initialized with {strategy.num_replicas_in_sync} devices")
    logger.info(f"💪 Effective batch size: {MultiGPUAdvancedConfig.BATCH_SIZE} (global)")
    logger.info(f"📊 Per-GPU batch size: {MultiGPUAdvancedConfig.BATCH_SIZE // strategy.num_replicas_in_sync}")
    
    return strategy

def swin_transformer_block(x, num_heads=8):
    """Swin Transformer block for advanced attention"""
    input_shape = tf.shape(x)
    C = input_shape[-1]
    
    # Multi-head attention
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=C // num_heads,
        dropout=0.1
    )(x, x)
    
    # Add & Norm
    x = layers.Add()([x, attention_output])
    x = layers.LayerNormalization()(x)
    
    # MLP
    mlp_output = layers.Dense(C * 4, activation='gelu')(x)
    mlp_output = layers.Dropout(0.1)(mlp_output)
    mlp_output = layers.Dense(C)(mlp_output)
    
    # Add & Norm
    x = layers.Add()([x, mlp_output])
    x = layers.LayerNormalization()(x)
    
    return x

def advanced_attention_gate(g, s, num_filters):
    """Advanced dual-attention gate"""
    # Spatial attention
    Wg = layers.Conv3D(num_filters, 1, padding='same')(g)
    Ws = layers.Conv3D(num_filters, 1, padding='same')(s)
    
    combined = layers.Add()([Wg, Ws])
    combined = layers.Activation('relu')(combined)
    spatial_att = layers.Conv3D(1, 1, padding='same', activation='sigmoid')(combined)
    
    # Channel attention (SE block)
    channel_att = layers.GlobalAveragePooling3D()(s)
    channel_att = layers.Dense(num_filters // 16, activation='relu')(channel_att)
    channel_att = layers.Dense(num_filters, activation='sigmoid')(channel_att)
    channel_att = layers.Reshape((1, 1, 1, num_filters))(channel_att)
    
    # Apply both attentions
    attended = layers.Multiply()([s, spatial_att])
    attended = layers.Multiply()([attended, channel_att])
    
    return attended

def massive_conv_block(x, num_filters, use_swin=False):
    """Massive convolution block - FIXED with BatchNorm"""
    # First conv
    x = layers.Conv3D(num_filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('gelu')(x)
    
    # Second conv
    x = layers.Conv3D(num_filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('gelu')(x)
    
    # Enhanced SE block
    se = layers.GlobalAveragePooling3D()(x)
    se = layers.Dense(num_filters // 8, activation='relu')(se)
    se = layers.Dropout(0.1)(se)
    se = layers.Dense(num_filters, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, 1, num_filters))(se)
    x = layers.Multiply()([x, se])
    
    return x

def build_massive_sota_model(input_shape=(192, 224, 176, 1), base_filters=48):
    """
    Build MASSIVE SOTA model - leveraging 4 GPU memory capacity
    Target: 25-30M parameters for maximum performance
    """
    inputs = layers.Input(shape=input_shape)
    
    # Encoder path
    skip_connections = []
    deep_outputs = []
    
    # Stage 1 - 48 filters
    conv1 = massive_conv_block(inputs, base_filters, use_swin=False)
    skip_connections.append(conv1)
    pool1 = layers.MaxPooling3D(2)(conv1)
    
    # Stage 2 - 96 filters
    conv2 = massive_conv_block(pool1, base_filters * 2, use_swin=False)
    skip_connections.append(conv2)
    pool2 = layers.MaxPooling3D(2)(conv2)
    
    # Stage 3 - 192 filters (Start Swin)
    conv3 = massive_conv_block(pool2, base_filters * 4, use_swin=True)
    skip_connections.append(conv3)
    pool3 = layers.MaxPooling3D(2)(conv3)
    
    # Stage 4 - 384 filters
    conv4 = massive_conv_block(pool3, base_filters * 8, use_swin=True)
    skip_connections.append(conv4)
    pool4 = layers.MaxPooling3D(2)(conv4)
    
    # Bottleneck - 768 filters (MASSIVE)
    conv5 = massive_conv_block(pool4, base_filters * 16, use_swin=True)
    
    # Decoder with deep supervision
    # Stage 6
    up6 = layers.Conv3DTranspose(base_filters * 8, 2, strides=2, padding='same')(conv5)
    att6 = advanced_attention_gate(up6, skip_connections[3], base_filters * 8)
    concat6 = layers.Concatenate()([up6, att6])
    conv6 = massive_conv_block(concat6, base_filters * 8, use_swin=True)
    
    if MultiGPUAdvancedConfig.USE_DEEP_SUPERVISION:
        deep_out6 = layers.Conv3D(1, 1, activation='sigmoid', dtype='float32', name='deep_output_6')(conv6)
        deep_outputs.append(deep_out6)
    
    # Stage 7
    up7 = layers.Conv3DTranspose(base_filters * 4, 2, strides=2, padding='same')(conv6)
    att7 = advanced_attention_gate(up7, skip_connections[2], base_filters * 4)
    concat7 = layers.Concatenate()([up7, att7])
    conv7 = massive_conv_block(concat7, base_filters * 4, use_swin=True)
    
    if MultiGPUAdvancedConfig.USE_DEEP_SUPERVISION:
        deep_out7 = layers.Conv3D(1, 1, activation='sigmoid', dtype='float32', name='deep_output_7')(conv7)
        deep_outputs.append(deep_out7)
    
    # Stage 8
    up8 = layers.Conv3DTranspose(base_filters * 2, 2, strides=2, padding='same')(conv7)
    att8 = advanced_attention_gate(up8, skip_connections[1], base_filters * 2)
    concat8 = layers.Concatenate()([up8, att8])
    conv8 = massive_conv_block(concat8, base_filters * 2, use_swin=False)
    
    if MultiGPUAdvancedConfig.USE_DEEP_SUPERVISION:
        deep_out8 = layers.Conv3D(1, 1, activation='sigmoid', dtype='float32', name='deep_output_8')(conv8)
        deep_outputs.append(deep_out8)
    
    # Stage 9
    up9 = layers.Conv3DTranspose(base_filters, 2, strides=2, padding='same')(conv8)
    att9 = advanced_attention_gate(up9, skip_connections[0], base_filters)
    concat9 = layers.Concatenate()([up9, att9])
    conv9 = massive_conv_block(concat9, base_filters, use_swin=False)
    
    # Final output
    main_output = layers.Conv3D(1, 1, activation='sigmoid', dtype='float32', name='main_output')(conv9)
    
    if MultiGPUAdvancedConfig.USE_DEEP_SUPERVISION:
        outputs = [main_output] + deep_outputs
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='Massive_SOTA_UNet')
    else:
        model = tf.keras.Model(inputs=inputs, outputs=main_output, name='Massive_SOTA_UNet')
    
    return model

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

def deep_supervision_loss(y_true, y_pred_list, weights=None):
    """Deep supervision loss for multiple outputs"""
    if weights is None:
        weights = [1.0] + [0.6] * (len(y_pred_list) - 1)
        weights = tf.constant(weights, dtype=tf.float32)
    
    total_loss = 0
    for i, (y_pred, weight) in enumerate(zip(y_pred_list, weights)):
        # Handle different scales
        if y_pred.shape[1:-1] != y_true.shape[1:-1]:
            y_true_resized = tf.image.resize(
                tf.squeeze(y_true, axis=-1),
                y_pred.shape[1:3]
            )
            y_true_resized = tf.expand_dims(y_true_resized, axis=-1)
        else:
            y_true_resized = y_true
        
        loss = ultimate_loss(y_true_resized, y_pred)
        total_loss += weight * loss
    
    return total_loss

class MultiGPUDataGenerator(tf.keras.utils.Sequence):
    """Multi-GPU optimized data generator"""
    def __init__(self, image_mask_pairs, batch_size, target_shape, shuffle=True, augment=True):
        self.image_mask_pairs = image_mask_pairs
        self.batch_size = batch_size
        self.target_shape = target_shape
        self.shuffle = shuffle
        self.augment = augment
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
                
                # Strong augmentation
                if self.augment:
                    img_data, mask_data = self.augment_sample(img_data, mask_data)
                
                X[i] = img_data[..., np.newaxis]
                y[i] = mask_data[..., np.newaxis]
                
            except Exception as e:
                logger.error(f"Error loading sample {idx}: {e}")
                X[i] = np.zeros((*self.target_shape, 1), dtype=np.float32)
                y[i] = np.zeros((*self.target_shape, 1), dtype=np.float32)
        
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
    
    def augment_sample(self, img, mask):
        # Strong augmentation for multi-GPU training
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=1)
            mask = np.flip(mask, axis=1)
        
        if np.random.rand() > 0.7:
            from scipy.ndimage import rotate
            angle = np.random.uniform(-15, 15)
            img = rotate(img, angle, axes=(0, 1), reshape=False, order=1)
            mask = rotate(mask, angle, axes=(0, 1), reshape=False, order=0)
        
        # Intensity augmentation
        if np.random.rand() > 0.6:
            gamma = np.random.uniform(0.8, 1.2)
            img = np.power(img, gamma)
            
            noise = np.random.normal(0, 0.02, img.shape)
            img = np.clip(img + noise, 0, 1)
        
        return img, mask
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

def cosine_schedule_with_warmup(epoch, warmup_epochs=8, total_epochs=100, initial_lr=3e-4, min_lr=1e-7):
    """Cosine annealing with warmup"""
    if epoch < warmup_epochs and warmup_epochs > 0:
        return initial_lr * max(epoch, 1) / max(warmup_epochs, 1)
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + (initial_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

def create_multi_gpu_callbacks(callbacks_dir):
    """Create callbacks for multi-GPU training"""
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
            patience=25,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: cosine_schedule_with_warmup(
                epoch, MultiGPUAdvancedConfig.WARMUP_EPOCHS, 
                MultiGPUAdvancedConfig.EPOCHS, MultiGPUAdvancedConfig.INITIAL_LR, 
                MultiGPUAdvancedConfig.MIN_LR
            ),
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

def train_multi_gpu_advanced_model():
    """Train the MASSIVE SOTA model on multiple GPUs"""
    logger.info("🚀 MULTI-GPU MASSIVE SOTA TRAINING")
    logger.info("=" * 80)
    logger.info("Target: 80%+ Dice coefficient with MASSIVE architecture")
    logger.info("Strategy: 4-GPU MirroredStrategy with 25M+ parameters")
    logger.info("=" * 80)
    
    # Setup
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # Configure multi-GPU strategy
    strategy = configure_multi_gpu_strategy()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Load dataset
    all_pairs = load_full_655_dataset()
    
    # Stratified split
    train_pairs, val_pairs = train_test_split(
        all_pairs,
        test_size=MultiGPUAdvancedConfig.VALIDATION_SPLIT,
        random_state=42,
        shuffle=True
    )
    
    logger.info(f"Multi-GPU split: {len(train_pairs)} train, {len(val_pairs)} validation")
    
    # Create data generators
    train_generator = MultiGPUDataGenerator(
        train_pairs,
        MultiGPUAdvancedConfig.BATCH_SIZE,
        MultiGPUAdvancedConfig.INPUT_SHAPE[:-1],
        shuffle=True,
        augment=True
    )
    
    val_generator = MultiGPUDataGenerator(
        val_pairs,
        MultiGPUAdvancedConfig.BATCH_SIZE,
        MultiGPUAdvancedConfig.INPUT_SHAPE[:-1],
        shuffle=False,
        augment=False
    )
    
    # Build model within strategy scope
    with strategy.scope():
        logger.info("Building MASSIVE SOTA model within MirroredStrategy scope...")
        model = build_massive_sota_model(
            input_shape=MultiGPUAdvancedConfig.INPUT_SHAPE,
            base_filters=MultiGPUAdvancedConfig.BASE_FILTERS
        )
        
        param_count = model.count_params()
        logger.info(f"🔥 MASSIVE Model parameters: {param_count:,}")
        logger.info(f"💪 Distributed across {strategy.num_replicas_in_sync} GPUs")
        
        # Compile with ultimate loss
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=MultiGPUAdvancedConfig.INITIAL_LR,
            weight_decay=2e-4
        )
        
        if MultiGPUAdvancedConfig.USE_DEEP_SUPERVISION:
            model.compile(
                optimizer=optimizer,
                loss=lambda y_true, y_pred: deep_supervision_loss(y_true, y_pred),
                metrics=[dice_coefficient, binary_dice_coefficient]
            )
        else:
            model.compile(
                optimizer=optimizer,
                loss=ultimate_loss,
                metrics=[dice_coefficient, binary_dice_coefficient, 'accuracy']
            )
    
    # Train with multi-GPU power
    logger.info("🔥 Starting MASSIVE multi-GPU training...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=MultiGPUAdvancedConfig.EPOCHS,
        callbacks=create_multi_gpu_callbacks(
            MultiGPUAdvancedConfig.CALLBACKS_DIR(timestamp)
        ),
        verbose=1
    )
    
    # Save final model
    final_model_path = MultiGPUAdvancedConfig.MODEL_SAVE_PATH(timestamp)
    model.save(final_model_path)
    
    # Results
    final_val_dice = history.history['val_dice_coefficient'][-1]
    best_val_dice = max(history.history['val_dice_coefficient'])
    
    logger.info("=" * 80)
    logger.info("🏆 MASSIVE MULTI-GPU SOTA TRAINING COMPLETED!")
    logger.info("=" * 80)
    logger.info(f"Model parameters: {param_count:,}")
    logger.info(f"GPUs used: {strategy.num_replicas_in_sync}")
    logger.info(f"Final validation Dice: {final_val_dice:.4f}")
    logger.info(f"Best validation Dice: {best_val_dice:.4f}")
    logger.info(f"Improvement over baseline: {(best_val_dice - 0.636):.3f}")
    
    if best_val_dice > 0.80:
        logger.info("🚀 ACHIEVED WORLD-CLASS PERFORMANCE (>80% Dice)!")
    elif best_val_dice > 0.75:
        logger.info("🏆 ACHIEVED SOTA PERFORMANCE (>75% Dice)!")
    elif best_val_dice > 0.70:
        logger.info("✅ EXCELLENT PERFORMANCE (>70% Dice)!")
    
    logger.info(f"Best model: {MultiGPUAdvancedConfig.CALLBACKS_DIR(timestamp)}/best_model.h5")
    logger.info("=" * 80)
    
    return history, model

if __name__ == "__main__":
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    try:
        history, model = train_multi_gpu_advanced_model()
        logger.info("🎉 MASSIVE MULTI-GPU SOTA TRAINING SUCCESS!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
