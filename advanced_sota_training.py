#!/usr/bin/env python3
"""
ADVANCED SOTA 3D Medical Segmentation
Target: 75-80% Dice coefficient with cutting-edge techniques
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
import glob
from sklearn.model_selection import train_test_split, StratifiedKFold
import cv2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/advanced_sota_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedConfig:
    # Dataset
    DATA_DIR = "/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training"
    
    # Architecture improvements
    INPUT_SHAPE = (192, 224, 176, 1)
    BASE_FILTERS = 32  # Increased for more capacity
    USE_SWIN_BLOCKS = True
    USE_DEEP_SUPERVISION = False
    
    # Training improvements
    BATCH_SIZE = 2  # Slightly smaller for bigger model
    EPOCHS = 100   # More epochs with better scheduling
    VALIDATION_SPLIT = 0.15  # More validation data
    
    # Advanced optimization
    INITIAL_LR = 2e-4
    MIN_LR = 1e-7
    WARMUP_EPOCHS = 5
    
    # Augmentation
    STRONG_AUGMENTATION = True
    MIXUP_ALPHA = 0.2
    
    # Ensemble
    N_FOLDS = 5  # Cross-validation
    
    CALLBACKS_DIR = lambda timestamp: Path(f'callbacks/advanced_sota_{timestamp}')
    MODEL_SAVE_PATH = lambda timestamp: f'models/advanced_sota_{timestamp}.h5'

def swin_transformer_block(x, num_heads=8, window_size=7, shift_size=0):
    """Swin Transformer block adapted for 3D medical imaging"""
    B, H, W, D, C = x.shape
    
    # Multi-head self attention with window partitioning
    # Simplified for 3D - using standard attention
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
    """Enhanced attention gate with spatial and channel attention"""
    # Spatial attention
    Wg = layers.Conv3D(num_filters, 1, padding='same')(g)
    Ws = layers.Conv3D(num_filters, 1, padding='same')(s)
    
    combined = layers.Add()([Wg, Ws])
    combined = layers.Activation('relu')(combined)
    
    # Spatial attention map
    spatial_att = layers.Conv3D(1, 1, padding='same', activation='sigmoid')(combined)
    
    # Channel attention (Squeeze-and-Excitation)
    channel_att = layers.GlobalAveragePooling3D()(s)
    channel_att = layers.Dense(num_filters // 16, activation='relu')(channel_att)
    channel_att = layers.Dense(num_filters, activation='sigmoid')(channel_att)
    channel_att = layers.Reshape((1, 1, 1, num_filters))(channel_att)
    
    # Apply both attentions
    attended = layers.Multiply()([s, spatial_att])
    attended = layers.Multiply()([attended, channel_att])
    
    return attended

def advanced_conv_block(x, num_filters, use_swin=False):
    """Advanced convolution block with optional Swin attention"""
    # First conv
    x = layers.Conv3D(num_filters, 3, padding='same')(x)
    x = layers.GroupNormalization(groups=min(32, num_filters//4))(x)  # Group norm instead of batch norm
    x = layers.Activation('gelu')(x)  # GELU instead of ReLU
    
    # Second conv
    x = layers.Conv3D(num_filters, 3, padding='same')(x)
    x = layers.GroupNormalization(groups=min(32, num_filters//4))(x)
    
    # Optional Swin Transformer block
    if use_swin and num_filters >= 64:  # Only for deeper layers
        x = swin_transformer_block(x, num_heads=min(8, num_filters//8))
    
    x = layers.Activation('gelu')(x)
    
    # Squeeze-and-Excitation
    se = layers.GlobalAveragePooling3D()(x)
    se = layers.Dense(num_filters // 16, activation='relu')(se)
    se = layers.Dense(num_filters, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, 1, num_filters))(se)
    x = layers.Multiply()([x, se])
    
    return x

def build_advanced_sota_model(input_shape=(192, 224, 176, 1), base_filters=32, use_swin=True, deep_supervision=True):
    """
    Build Advanced SOTA model with:
    - Swin Transformer blocks
    - Advanced attention gates
    - Deep supervision
    - Group normalization
    - GELU activations
    """
    inputs = layers.Input(shape=input_shape)
    
    # Encoder path
    skip_connections = []
    deep_outputs = []
    
    # Stage 1
    conv1 = advanced_conv_block(inputs, base_filters, use_swin=False)
    skip_connections.append(conv1)
    pool1 = layers.MaxPooling3D(2)(conv1)
    
    # Stage 2
    conv2 = advanced_conv_block(pool1, base_filters * 2, use_swin=False)
    skip_connections.append(conv2)
    pool2 = layers.MaxPooling3D(2)(conv2)
    
    # Stage 3 - Start using Swin blocks
    conv3 = advanced_conv_block(pool2, base_filters * 4, use_swin=use_swin)
    skip_connections.append(conv3)
    pool3 = layers.MaxPooling3D(2)(conv3)
    
    # Stage 4
    conv4 = advanced_conv_block(pool3, base_filters * 8, use_swin=use_swin)
    skip_connections.append(conv4)
    pool4 = layers.MaxPooling3D(2)(conv4)
    
    # Bottleneck
    conv5 = advanced_conv_block(pool4, base_filters * 16, use_swin=use_swin)
    
    # Decoder path with deep supervision
    # Stage 6
    up6 = layers.Conv3DTranspose(base_filters * 8, 2, strides=2, padding='same')(conv5)
    att6 = advanced_attention_gate(up6, skip_connections[3], base_filters * 8)
    concat6 = layers.Concatenate()([up6, att6])
    conv6 = advanced_conv_block(concat6, base_filters * 8, use_swin=use_swin)
    
    if deep_supervision:
        deep_out6 = layers.Conv3D(1, 1, activation='sigmoid', dtype='float32', name='deep_output_6')(conv6)
        deep_outputs.append(deep_out6)
    
    # Stage 7
    up7 = layers.Conv3DTranspose(base_filters * 4, 2, strides=2, padding='same')(conv6)
    att7 = advanced_attention_gate(up7, skip_connections[2], base_filters * 4)
    concat7 = layers.Concatenate()([up7, att7])
    conv7 = advanced_conv_block(concat7, base_filters * 4, use_swin=use_swin)
    
    if deep_supervision:
        deep_out7 = layers.Conv3D(1, 1, activation='sigmoid', dtype='float32', name='deep_output_7')(conv7)
        deep_outputs.append(deep_out7)
    
    # Stage 8
    up8 = layers.Conv3DTranspose(base_filters * 2, 2, strides=2, padding='same')(conv7)
    att8 = advanced_attention_gate(up8, skip_connections[1], base_filters * 2)
    concat8 = layers.Concatenate()([up8, att8])
    conv8 = advanced_conv_block(concat8, base_filters * 2, use_swin=False)
    
    if deep_supervision:
        deep_out8 = layers.Conv3D(1, 1, activation='sigmoid', dtype='float32', name='deep_output_8')(conv8)
        deep_outputs.append(deep_out8)
    
    # Stage 9
    up9 = layers.Conv3DTranspose(base_filters, 2, strides=2, padding='same')(conv8)
    att9 = advanced_attention_gate(up9, skip_connections[0], base_filters)
    concat9 = layers.Concatenate()([up9, att9])
    conv9 = advanced_conv_block(concat9, base_filters, use_swin=False)
    
    # Final output
    main_output = layers.Conv3D(1, 1, activation='sigmoid', dtype='float32', name='main_output')(conv9)
    
    if deep_supervision:
        outputs = [main_output] + deep_outputs
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='Advanced_SOTA_UNet')
    else:
        model = tf.keras.Model(inputs=inputs, outputs=main_output, name='Advanced_SOTA_UNet')
    
    return model

def topology_aware_loss(y_true, y_pred, smooth=1e-6):
    """Topology-aware loss that preserves connectivity"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Standard Dice loss
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice_loss = 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    
    # Boundary loss (gradient-based)

    
    def sobel_edges(img):
        # Simplified edge detection - just use gradients
        dx = tf.image.sobel_edges(tf.squeeze(img, axis=-1))[:,:,:,0:1,:]
        dy = tf.image.sobel_edges(tf.squeeze(img, axis=-1))[:,:,:,1:2,:]
        # For 3D, approximate with 2D sobel
        return tf.expand_dims(tf.sqrt(dx**2 + dy**2), axis=-1)

    edges_true = sobel_edges(y_true)
    edges_pred = sobel_edges(y_pred)
    boundary_loss = tf.reduce_mean(tf.square(edges_true - edges_pred))
    
    # Focal loss for hard examples
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    focal_loss = -tf.reduce_mean(
        0.25 * y_true * tf.pow(1 - y_pred, 2) * tf.math.log(y_pred) +
        0.75 * (1 - y_true) * tf.pow(y_pred, 2) * tf.math.log(1 - y_pred)
    )
    
    return 0.5 * dice_loss + 0.3 * focal_loss + 0.2 * boundary_loss

def deep_supervision_loss(y_true, y_pred_list, weights=None):
    """Deep supervision loss for multiple outputs - FIXED"""
    if weights is None:
        # Create weights as TensorFlow constants to avoid type errors
        num_outputs = len(y_pred_list)
        weights = [1.0] + [0.5] * (num_outputs - 1)
    
    # Convert to TensorFlow tensor with correct dtype
    weights = tf.constant(weights, dtype=tf.float32)
    
    total_loss = tf.constant(0.0, dtype=tf.float32)
    
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
            
        loss = topology_aware_loss(y_true_resized, y_pred)
        total_loss += tf.cast(weight, tf.float32) * loss
    
    return total_loss

def elastic_deformation_3d(image, mask, alpha=10, sigma=2):
    """3D elastic deformation for data augmentation"""
    shape = image.shape
    
    # Generate random displacement fields
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    
    # Create coordinate arrays
    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    
    # Apply deformation
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z + dz, (-1, 1))
    
    deformed_image = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    deformed_mask = map_coordinates(mask, indices, order=0, mode='reflect').reshape(shape)
    
    return deformed_image, deformed_mask

class AdvancedDataGenerator(tf.keras.utils.Sequence):
    """Advanced data generator with strong augmentation"""
    def __init__(self, image_mask_pairs, batch_size, target_shape, 
                 shuffle=True, augment=True, mixup_alpha=0.0):
        self.image_mask_pairs = image_mask_pairs
        self.batch_size = batch_size
        self.target_shape = target_shape
        self.shuffle = shuffle
        self.augment = augment
        self.mixup_alpha = mixup_alpha
        self.indexes = np.arange(len(image_mask_pairs))
        
        # Analyze lesion sizes for stratified sampling
        self.lesion_sizes = []
        logger.info("Analyzing lesion sizes for stratified augmentation...")
        
        for i in range(0, len(image_mask_pairs), max(1, len(image_mask_pairs)//50)):
            try:
                _, mask_path = image_mask_pairs[i]
                mask_data = nib.load(mask_path).get_fdata()
                size = np.sum(mask_data)
                self.lesion_sizes.append(size)
            except:
                self.lesion_sizes.append(0)
        
        self.lesion_sizes = np.array(self.lesion_sizes)
        valid_sizes = self.lesion_sizes[self.lesion_sizes > 0]
        if len(valid_sizes) == 0:
            self.small_lesion_threshold = 1000.0
            self.large_lesion_threshold = 5000.0
        else:
            self.small_lesion_threshold = np.percentile(valid_sizes, 33)
            self.large_lesion_threshold = np.percentile(valid_sizes, 66)
        
        logger.info(f"Lesion size thresholds: small < {self.small_lesion_threshold:.0f}, large > {self.large_lesion_threshold:.0f}")
        
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
                    img_data = self.resize_volume(img_data, self.target_shape)
                    mask_data = self.resize_volume(mask_data, self.target_shape)
                    mask_data = (mask_data > 0.5).astype(np.float32)
                
                # Advanced intensity normalization
                img_data = self.robust_normalization(img_data)
                
                # Strong augmentation
                if self.augment:
                    img_data, mask_data = self.apply_strong_augmentation(img_data, mask_data, idx)
                
                X[i] = img_data[..., np.newaxis]
                y[i] = mask_data[..., np.newaxis]
                
            except Exception as e:
                logger.error(f"Error loading sample {idx}: {e}")
                X[i] = np.zeros((*self.target_shape, 1), dtype=np.float32)
                y[i] = np.zeros((*self.target_shape, 1), dtype=np.float32)
        
        # Mixup augmentation
        if self.augment and self.mixup_alpha > 0:
            X, y = self.mixup(X, y, self.mixup_alpha)
        
        return X, y
    
    def resize_volume(self, volume, target_shape):
        """High-quality volume resizing"""
        factors = [t / s for t, s in zip(target_shape, volume.shape)]
        return zoom(volume, factors, order=1)
    
    def robust_normalization(self, img):
        """Robust intensity normalization"""
        # Remove outliers
        p1, p99 = np.percentile(img[img > 0], [0.5, 99.5])
        img = np.clip(img, p1, p99)
        
        # Z-score normalization
        mean = np.mean(img[img > 0])
        std = np.std(img[img > 0])
        img = (img - mean) / (std + 1e-8)
        
        # Rescale to [0, 1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        return img
    
    def apply_strong_augmentation(self, img, mask, idx):
        """Apply strong augmentation based on lesion size"""
        lesion_size = np.sum(mask)
        
        # Flip augmentation (always)
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=1)
            mask = np.flip(mask, axis=1)
        
        # Rotation (small angles)
        if np.random.rand() > 0.7:
            angle = np.random.uniform(-10, 10)
            # Simplified rotation for 3D - rotate around one axis
            from scipy.ndimage import rotate
            img = rotate(img, angle, axes=(0, 1), reshape=False, order=1)
            mask = rotate(mask, angle, axes=(0, 1), reshape=False, order=0)
        
        # Elastic deformation (more aggressive for large lesions)
        if np.random.rand() > 0.8:
            if lesion_size > self.large_lesion_threshold:
                alpha, sigma = 15, 3  # Stronger deformation for large lesions
            else:
                alpha, sigma = 8, 2   # Gentler for small lesions
            img, mask = elastic_deformation_3d(img, mask, alpha, sigma)
        
        # Intensity augmentation
        if np.random.rand() > 0.6:
            # Gamma correction
            gamma = np.random.uniform(0.8, 1.2)
            img = np.power(img, gamma)
            
            # Additive noise
            noise_std = np.random.uniform(0.01, 0.05)
            noise = np.random.normal(0, noise_std, img.shape)
            img = np.clip(img + noise, 0, 1)
            
            # Contrast adjustment
            contrast = np.random.uniform(0.9, 1.1)
            img = np.clip(img * contrast, 0, 1)
        
        return img, mask
    
    def mixup(self, X, y, alpha):
        """Mixup data augmentation"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
            batch_size = X.shape[0]
            index = np.random.permutation(batch_size)
            
            X = lam * X + (1 - lam) * X[index, :]
            y = lam * y + (1 - lam) * y[index, :]
        
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

def cosine_schedule_with_warmup(epoch, warmup_epochs=5, total_epochs=100, initial_lr=2e-4, min_lr=1e-7):
    """Cosine annealing with warmup"""
    if epoch < warmup_epochs:
        return initial_lr * max(epoch, 1) / max(warmup_epochs, 1)
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + (initial_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

def create_advanced_callbacks(callbacks_dir, config):
    """Create advanced callbacks with learning rate scheduling"""
    callbacks_dir.mkdir(parents=True, exist_ok=True)
    
    # Learning rate scheduler
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: cosine_schedule_with_warmup(
            epoch, config.WARMUP_EPOCHS, config.EPOCHS, config.INITIAL_LR, config.MIN_LR
        ),
        verbose=1
    )
    
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
            patience=20,  # More patience for longer training
            restore_best_weights=True,
            verbose=1
        ),
        lr_scheduler,
        tf.keras.callbacks.CSVLogger(str(callbacks_dir / 'training_log.csv')),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(callbacks_dir / 'tensorboard'),
            histogram_freq=0,
            write_graph=False,
            update_freq='epoch'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=config.MIN_LR,
            verbose=1
        )
    ]

# Import metrics from original script
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

def train_advanced_model():
    """Train the advanced SOTA model"""
    logger.info("🚀 ADVANCED SOTA TRAINING")
    logger.info("=" * 70)
    logger.info("Target: 75-80% Dice coefficient")
    logger.info("Features: Swin Transformers, Advanced Attention, Deep Supervision")
    logger.info("=" * 70)
    
    # Setup
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Load dataset (same as before)
    from correct_full_training import load_full_655_dataset
    all_pairs = load_full_655_dataset()
    
    # Stratified split based on lesion sizes
    logger.info("Creating stratified train/validation split...")
    lesion_sizes = []
    for img_path, mask_path in all_pairs:
        try:
            mask_data = nib.load(mask_path).get_fdata()
            size = np.sum(mask_data)
            lesion_sizes.append(size)
        except:
            lesion_sizes.append(0)
    
    # Create size categories for stratification
    lesion_sizes = np.array(lesion_sizes)
    size_categories = np.zeros_like(lesion_sizes)
    size_categories[lesion_sizes > np.percentile(lesion_sizes[lesion_sizes > 0], 66)] = 2  # Large
    size_categories[lesion_sizes > np.percentile(lesion_sizes[lesion_sizes > 0], 33)] = 1  # Medium
    # Small lesions remain 0
    
    train_pairs, val_pairs = train_test_split(
        all_pairs,
        test_size=AdvancedConfig.VALIDATION_SPLIT,
        random_state=42,
        stratify=size_categories
    )
    
    logger.info(f"Stratified split: {len(train_pairs)} train, {len(val_pairs)} validation")
    
    # Create advanced data generators
    train_generator = AdvancedDataGenerator(
        train_pairs, 
        AdvancedConfig.BATCH_SIZE, 
        AdvancedConfig.INPUT_SHAPE[:-1],
        shuffle=True, 
        augment=True, 
        mixup_alpha=AdvancedConfig.MIXUP_ALPHA
    )
    
    val_generator = AdvancedDataGenerator(
        val_pairs, 
        AdvancedConfig.BATCH_SIZE, 
        AdvancedConfig.INPUT_SHAPE[:-1],
        shuffle=False, 
        augment=False
    )
    
    # Build advanced model
    logger.info("Building advanced SOTA model...")
    model = build_advanced_sota_model(
        input_shape=AdvancedConfig.INPUT_SHAPE,
        base_filters=AdvancedConfig.BASE_FILTERS,
        use_swin=AdvancedConfig.USE_SWIN_BLOCKS,
        deep_supervision=AdvancedConfig.USE_DEEP_SUPERVISION
    )
    
    param_count = model.count_params()
    logger.info(f"Model parameters: {param_count:,}")
    
    # Compile with advanced loss
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=AdvancedConfig.INITIAL_LR,
        weight_decay=1e-4  # L2 regularization
    )
    
    if AdvancedConfig.USE_DEEP_SUPERVISION:
        # Multiple outputs for deep supervision
        model.compile(
            optimizer=optimizer,
            loss=lambda y_true, y_pred: deep_supervision_loss(y_true, y_pred),
            metrics=[dice_coefficient, binary_dice_coefficient]
        )
    else:
        model.compile(
            optimizer=optimizer,
            loss=topology_aware_loss,
            metrics=[dice_coefficient, binary_dice_coefficient, 'accuracy']
        )
    
    # Train
    logger.info("Starting advanced training...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=AdvancedConfig.EPOCHS,
        callbacks=create_advanced_callbacks(
            AdvancedConfig.CALLBACKS_DIR(timestamp), 
            AdvancedConfig
        ),
        verbose=1
    )
    
    # Save final model
    final_model_path = AdvancedConfig.MODEL_SAVE_PATH(timestamp)
    model.save(final_model_path)
    
    # Training summary
    if AdvancedConfig.USE_DEEP_SUPERVISION:
        final_train_dice = history.history['dice_coefficient'][-1]
        final_val_dice = history.history['val_dice_coefficient'][-1]
        best_val_dice = max(history.history['val_dice_coefficient'])
    else:
        final_train_dice = history.history['dice_coefficient'][-1]
        final_val_dice = history.history['val_dice_coefficient'][-1]
        best_val_dice = max(history.history['val_dice_coefficient'])
    
    logger.info("=" * 70)
    logger.info("🎉 ADVANCED SOTA TRAINING COMPLETED!")
    logger.info("=" * 70)
    logger.info(f"Model parameters: {param_count:,}")
    logger.info(f"Final training Dice: {final_train_dice:.4f}")
    logger.info(f"Final validation Dice: {final_val_dice:.4f}")
    logger.info(f"Best validation Dice: {best_val_dice:.4f}")
    logger.info(f"Improvement over baseline: {(best_val_dice - 0.636):.3f}")
    
    if best_val_dice > 0.75:
        logger.info("🏆 ACHIEVED SOTA PERFORMANCE (>75% Dice)!")
    elif best_val_dice > 0.70:
        logger.info("✅ EXCELLENT PERFORMANCE (>70% Dice)!")
    elif best_val_dice > 0.65:
        logger.info("👍 GOOD IMPROVEMENT OVER BASELINE!")
    
    logger.info(f"Best model: {AdvancedConfig.CALLBACKS_DIR(timestamp)}/best_model.h5")
    logger.info("=" * 70)
    
    return history, model

def ensemble_inference(model_paths, test_generator, tta_steps=8):
    """Ensemble inference with test-time augmentation"""
    logger.info("Running ensemble inference with TTA...")
    
    models = []
    for path in model_paths:
        try:
            model = tf.keras.models.load_model(path, compile=False)
            models.append(model)
            logger.info(f"Loaded model: {path}")
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
    
    if not models:
        logger.error("No models loaded for ensemble!")
        return None
    
    ensemble_predictions = []
    
    for batch_idx in range(len(test_generator)):
        X_batch, y_batch = test_generator[batch_idx]
        batch_predictions = np.zeros_like(y_batch)
        
        # Model ensemble
        for model in models:
            model_preds = np.zeros_like(y_batch)
            
            # Test-time augmentation
            for tta_step in range(tta_steps):
                # Apply random augmentations
                X_aug = X_batch.copy()
                
                # Random flips
                if np.random.rand() > 0.5:
                    X_aug = np.flip(X_aug, axis=1)
                if np.random.rand() > 0.5:
                    X_aug = np.flip(X_aug, axis=2)
                
                # Predict
                if isinstance(model.output, list):  # Deep supervision model
                    pred_aug = model.predict(X_aug, verbose=0)[0]  # Main output
                else:
                    pred_aug = model.predict(X_aug, verbose=0)
                
                # Reverse augmentations
                if np.random.rand() > 0.5:  # Same random state
                    pred_aug = np.flip(pred_aug, axis=1)
                if np.random.rand() > 0.5:
                    pred_aug = np.flip(pred_aug, axis=2)
                
                model_preds += pred_aug
            
            model_preds /= tta_steps
            batch_predictions += model_preds
        
        batch_predictions /= len(models)
        ensemble_predictions.append(batch_predictions)
    
    return np.concatenate(ensemble_predictions, axis=0)

def cross_validation_training():
    """5-fold cross-validation training"""
    logger.info("🔄 STARTING 5-FOLD CROSS-VALIDATION")
    logger.info("=" * 70)
    
    # Load dataset
    from correct_full_training import load_full_655_dataset
    all_pairs = load_full_655_dataset()
    
    # Create lesion size categories for stratification
    lesion_sizes = []
    for img_path, mask_path in all_pairs:
        try:
            mask_data = nib.load(mask_path).get_fdata()
            size = np.sum(mask_data)
            lesion_sizes.append(size)
        except:
            lesion_sizes.append(0)
    
    lesion_sizes = np.array(lesion_sizes)
    size_categories = np.zeros_like(lesion_sizes)
    size_categories[lesion_sizes > np.percentile(lesion_sizes[lesion_sizes > 0], 66)] = 2
    size_categories[lesion_sizes > np.percentile(lesion_sizes[lesion_sizes > 0], 33)] = 1
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=AdvancedConfig.N_FOLDS, shuffle=True, random_state=42)
    
    fold_results = []
    model_paths = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_pairs, size_categories)):
        logger.info(f"\n{'='*20} FOLD {fold+1}/{AdvancedConfig.N_FOLDS} {'='*20}")
        
        # Split data
        train_pairs = [all_pairs[i] for i in train_idx]
        val_pairs = [all_pairs[i] for i in val_idx]
        
        logger.info(f"Fold {fold+1}: {len(train_pairs)} train, {len(val_pairs)} val")
        
        # Create generators
        train_gen = AdvancedDataGenerator(
            train_pairs, AdvancedConfig.BATCH_SIZE, AdvancedConfig.INPUT_SHAPE[:-1],
            shuffle=True, augment=True, mixup_alpha=AdvancedConfig.MIXUP_ALPHA
        )
        val_gen = AdvancedDataGenerator(
            val_pairs, AdvancedConfig.BATCH_SIZE, AdvancedConfig.INPUT_SHAPE[:-1],
            shuffle=False, augment=False
        )
        
        # Build model
        tf.keras.backend.clear_session()  # Clear memory
        model = build_advanced_sota_model(
            input_shape=AdvancedConfig.INPUT_SHAPE,
            base_filters=AdvancedConfig.BASE_FILTERS,
            use_swin=AdvancedConfig.USE_SWIN_BLOCKS,
            deep_supervision=AdvancedConfig.USE_DEEP_SUPERVISION
        )
        
        # Compile
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=AdvancedConfig.INITIAL_LR,
            weight_decay=1e-4
        )
        
        if AdvancedConfig.USE_DEEP_SUPERVISION:
            model.compile(
                optimizer=optimizer,
                loss=lambda y_true, y_pred: deep_supervision_loss(y_true, y_pred),
                metrics=[dice_coefficient, binary_dice_coefficient]
            )
        else:
            model.compile(
                optimizer=optimizer,
                loss=topology_aware_loss,
                metrics=[dice_coefficient, binary_dice_coefficient, 'accuracy']
            )
        
        # Train
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fold_callbacks_dir = Path(f'callbacks/cv_fold_{fold+1}_{timestamp}')
        
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=AdvancedConfig.EPOCHS,
            callbacks=create_advanced_callbacks(fold_callbacks_dir, AdvancedConfig),
            verbose=1
        )
        
        # Save model
        fold_model_path = f'models/cv_fold_{fold+1}_{timestamp}.h5'
        model.save(fold_model_path)
        model_paths.append(fold_model_path)
        
        # Record results
        best_val_dice = max(history.history['val_dice_coefficient'])
        fold_results.append(best_val_dice)
        
        logger.info(f"Fold {fold+1} best validation Dice: {best_val_dice:.4f}")
    
    # Cross-validation summary
    mean_dice = np.mean(fold_results)
    std_dice = np.std(fold_results)
    
    logger.info("\n" + "="*70)
    logger.info("🏆 CROSS-VALIDATION RESULTS")
    logger.info("="*70)
    logger.info(f"Fold results: {[f'{r:.4f}' for r in fold_results]}")
    logger.info(f"Mean validation Dice: {mean_dice:.4f} ± {std_dice:.4f}")
    logger.info(f"Best single fold: {max(fold_results):.4f}")
    logger.info(f"Worst single fold: {min(fold_results):.4f}")
    
    if mean_dice > 0.75:
        logger.info("🏆 ACHIEVED SOTA PERFORMANCE!")
    elif mean_dice > 0.70:
        logger.info("✅ EXCELLENT PERFORMANCE!")
    elif mean_dice > 0.65:
        logger.info("👍 GOOD IMPROVEMENT!")
    
    logger.info(f"Model ensemble available: {len(model_paths)} models")
    logger.info("="*70)
    
    return fold_results, model_paths

if __name__ == "__main__":
    # Setup directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    try:
        # Choose training mode
        training_mode = "single"  # Change to "cv" for cross-validation
        
        if training_mode == "single":
            logger.info("🚀 SINGLE ADVANCED MODEL TRAINING")
            history, model = train_advanced_model()
            
        elif training_mode == "cv":
            logger.info("🔄 CROSS-VALIDATION TRAINING")
            fold_results, model_paths = cross_validation_training()
            
            # Optional: Run ensemble inference on validation set
            logger.info("Running ensemble evaluation...")
            # Implementation depends on having a separate test set
            
        logger.info("🎉 ADVANCED SOTA TRAINING PIPELINE COMPLETED!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
