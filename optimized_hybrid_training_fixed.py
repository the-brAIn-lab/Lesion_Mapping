#!/usr/bin/env python3
"""
Optimized Hybrid CNN-Transformer Training
Direct implementation using proven working environment
Focus: Get SOTA architecture running with memory optimization
"""

import tensorflow as tf
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import os
import logging
from pathlib import Path
import gc

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable mixed precision training
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Configure GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logger.info(f"Using {len(gpus)} GPU(s)")

def resize_volume(volume, target_shape):
    """Resize 3D volume using scipy zoom"""
    factors = [t/s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=1)

def combined_dice_focal_loss(y_true, y_pred, focal_gamma=3.0, focal_alpha=0.25):
    """Combined loss for extreme class imbalance"""
    smooth = 1e-6
    
    # Cast to float32 for loss computation
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Generalized Dice Loss
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    
    # Class weights (inverse volume)
    w1 = 1.0 / (tf.reduce_sum(y_true_f) + smooth)
    w0 = 1.0 / (tf.reduce_sum(1 - y_true_f) + smooth)
    w_sum = w0 + w1
    w0 = w0 / w_sum
    w1 = w1 / w_sum
    
    # Weighted intersection and union
    intersection = w1 * tf.reduce_sum(y_true_f * y_pred_f) + w0 * tf.reduce_sum((1-y_true_f) * (1-y_pred_f))
    union = w1 * tf.reduce_sum(y_true_f + y_pred_f) + w0 * tf.reduce_sum(2 - y_true_f - y_pred_f)
    dice_loss = 1 - (2. * intersection + smooth) / (union + smooth)
    
    # Focal Loss
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    alpha_t = tf.where(tf.equal(y_true, 1), focal_alpha, 1 - focal_alpha)
    focal_loss = -alpha_t * tf.pow(1 - p_t, focal_gamma) * tf.math.log(p_t)
    focal_loss = tf.reduce_mean(focal_loss)
    
    # Combine losses
    return 0.7 * dice_loss + 0.3 * focal_loss

def dice_coefficient(y_true, y_pred):
    """Dice metric for evaluation"""
    smooth = 1e-6
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred, tf.float32))
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

class MemoryEfficientSEBlock(tf.keras.layers.Layer):
    """Memory-efficient Squeeze-and-Excitation block"""
    def __init__(self, filters, reduction=16, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.pool = tf.keras.layers.GlobalAveragePooling3D()
        self.fc1 = tf.keras.layers.Dense(filters // reduction, activation='relu')
        self.fc2 = tf.keras.layers.Dense(filters, activation='sigmoid')
        
    def call(self, inputs):
        se = self.pool(inputs)
        se = self.fc1(se)
        se = self.fc2(se)
        se = tf.reshape(se, (-1, 1, 1, 1, self.filters))
        return inputs * se

class AttentionGate(tf.keras.layers.Layer):
    """Attention gate for skip connections"""
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.conv_x = tf.keras.layers.Conv3D(filters, 1, strides=2, padding='same')
        self.conv_g = tf.keras.layers.Conv3D(filters, 1, padding='same')
        self.conv_psi = tf.keras.layers.Conv3D(1, 1, padding='same', activation='sigmoid')
        self.conv_out = tf.keras.layers.Conv3D(filters, 1, padding='same')
        self.upsample = tf.keras.layers.UpSampling3D(size=2)
        
    def call(self, x, g):
        # Align dimensions
        x1 = self.conv_x(x)
        g1 = self.conv_g(g)
        
        # Compute attention
        psi = tf.nn.relu(x1 + g1)
        psi = self.conv_psi(psi)
        psi = self.upsample(psi)
        
        # Apply attention
        return self.conv_out(x * psi)

class SimplifiedTransformerBlock(tf.keras.layers.Layer):
    """Simplified 3D transformer block for medical imaging"""
    def __init__(self, embed_dim, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Multi-head attention
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=0.1
        )
        self.norm1 = tf.keras.layers.LayerNormalization()
        
        # Feed-forward
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(embed_dim * 2, activation='gelu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(embed_dim),
            tf.keras.layers.Dropout(0.1)
        ])
        self.norm2 = tf.keras.layers.LayerNormalization()
        
    def call(self, inputs, training=None):
        # Reshape to sequence
        batch_size = tf.shape(inputs)[0]
        h, w, d = inputs.shape[1:4]
        c = inputs.shape[4]
        
        # Flatten spatial dimensions
        x = tf.reshape(inputs, [batch_size, h*w*d, c])
        
        # Self-attention
        attn_out = self.attention(x, x, training=training)
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(x, training=training)
        x = self.norm2(x + ffn_out)
        
        # Reshape back
        x = tf.reshape(x, [batch_size, h, w, d, c])
        return x

def build_hybrid_model(input_shape=(128, 128, 128, 1), base_filters=32, use_transformer=True):
    """Build memory-optimized hybrid CNN-Transformer model"""
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Encoder path with SE blocks
    skip_connections = []
    
    # Block 1
    conv1 = tf.keras.layers.Conv3D(base_filters, 3, padding='same', activation='relu')(inputs)
    conv1 = tf.keras.layers.Conv3D(base_filters, 3, padding='same', activation='relu')(conv1)
    conv1 = MemoryEfficientSEBlock(base_filters)(conv1)
    skip_connections.append(conv1)
    pool1 = tf.keras.layers.MaxPooling3D(2)(conv1)
    
    # Block 2
    conv2 = tf.keras.layers.Conv3D(base_filters*2, 3, padding='same', activation='relu')(pool1)
    conv2 = tf.keras.layers.Conv3D(base_filters*2, 3, padding='same', activation='relu')(conv2)
    conv2 = MemoryEfficientSEBlock(base_filters*2)(conv2)
    skip_connections.append(conv2)
    pool2 = tf.keras.layers.MaxPooling3D(2)(conv2)
    
    # Block 3
    conv3 = tf.keras.layers.Conv3D(base_filters*4, 3, padding='same', activation='relu')(pool2)
    conv3 = tf.keras.layers.Conv3D(base_filters*4, 3, padding='same', activation='relu')(conv3)
    conv3 = MemoryEfficientSEBlock(base_filters*4)(conv3)
    skip_connections.append(conv3)
    pool3 = tf.keras.layers.MaxPooling3D(2)(conv3)
    
    # Bottleneck with optional transformer
    bottleneck = tf.keras.layers.Conv3D(base_filters*8, 3, padding='same', activation='relu')(pool3)
    bottleneck = tf.keras.layers.Conv3D(base_filters*8, 3, padding='same', activation='relu')(bottleneck)
    
    if use_transformer:
        # Add transformer block in bottleneck (most compressed representation)
        bottleneck = SimplifiedTransformerBlock(base_filters*8)(bottleneck)
    
    bottleneck = MemoryEfficientSEBlock(base_filters*8)(bottleneck)
    
    # Decoder path with attention gates
    # Block 4
    up4 = tf.keras.layers.Conv3DTranspose(base_filters*4, 2, strides=2, padding='same')(bottleneck)
    att4 = AttentionGate(base_filters*4)(skip_connections[2], bottleneck)
    concat4 = tf.keras.layers.Concatenate()([up4, att4])
    conv4 = tf.keras.layers.Conv3D(base_filters*4, 3, padding='same', activation='relu')(concat4)
    conv4 = tf.keras.layers.Conv3D(base_filters*4, 3, padding='same', activation='relu')(conv4)
    
    # Block 5
    up5 = tf.keras.layers.Conv3DTranspose(base_filters*2, 2, strides=2, padding='same')(conv4)
    att5 = AttentionGate(base_filters*2)(skip_connections[1], conv4)
    concat5 = tf.keras.layers.Concatenate()([up5, att5])
    conv5 = tf.keras.layers.Conv3D(base_filters*2, 3, padding='same', activation='relu')(concat5)
    conv5 = tf.keras.layers.Conv3D(base_filters*2, 3, padding='same', activation='relu')(conv5)
    
    # Block 6
    up6 = tf.keras.layers.Conv3DTranspose(base_filters, 2, strides=2, padding='same')(conv5)
    att6 = AttentionGate(base_filters)(skip_connections[0], conv5)
    concat6 = tf.keras.layers.Concatenate()([up6, att6])
    conv6 = tf.keras.layers.Conv3D(base_filters, 3, padding='same', activation='relu')(concat6)
    conv6 = tf.keras.layers.Conv3D(base_filters, 3, padding='same', activation='relu')(conv6)
    
    # Output layer (ensure float32 for mixed precision)
    outputs = tf.keras.layers.Conv3D(1, 1, activation='sigmoid', dtype='float32')(conv6)
    
    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
    return model

class OptimizedAtlasDataGenerator(tf.keras.utils.Sequence):
    """Memory-efficient data generator for ATLAS dataset"""
    def __init__(self, base_dir, batch_size=1, model_dim=(128, 128, 128), shuffle=True, augment=False):
        self.base_dir = base_dir
        self.images_dir = os.path.join(base_dir, "Images")
        self.masks_dir = os.path.join(base_dir, "Masks")
        self.batch_size = batch_size
        self.model_dim = model_dim
        self.shuffle = shuffle
        self.augment = augment
        self.image_ids = []
        
        # Find valid pairs
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
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        X = np.empty((self.batch_size, *self.model_dim, 1), dtype=np.float32)
        y = np.empty((self.batch_size, *self.model_dim, 1), dtype=np.float32)
        
        for i, idx in enumerate(indexes):
            try:
                # Load data
                img_path = os.path.join(self.images_dir, self.image_ids[idx] + "_T1w.nii.gz")
                mask_path = os.path.join(self.masks_dir, self.image_ids[idx] + "_label-L_desc-T1lesion_mask.nii.gz")
                
                img_data = nib.load(img_path).get_fdata(dtype=np.float32)
                mask_data = nib.load(mask_path).get_fdata(dtype=np.float32)
                
                # Resize if needed
                if img_data.shape != self.model_dim:
                    img_data = resize_volume(img_data, self.model_dim)
                    mask_data = resize_volume(mask_data, self.model_dim)
                    mask_data = (mask_data > 0.5).astype(np.float32)
                
                # Normalize
                p1, p99 = np.percentile(img_data[img_data > 0], [1, 99])
                img_data = np.clip(img_data, p1, p99)
                img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min() + 1e-8)
                
                # Simple augmentation
                if self.augment and np.random.random() > 0.5:
                    # Random flip
                    if np.random.random() > 0.5:
                        img_data = np.flip(img_data, axis=0)
                        mask_data = np.flip(mask_data, axis=0)
                    
                    # Small rotation (simplified)
                    if np.random.random() > 0.5:
                        angle = np.random.uniform(-10, 10)
                        # Simplified rotation - just transpose for 90 degree rotations
                        if angle > 5:
                            img_data = np.transpose(img_data, (1, 0, 2))
                            mask_data = np.transpose(mask_data, (1, 0, 2))
                
                X[i] = np.expand_dims(img_data, axis=-1)
                y[i] = np.expand_dims(mask_data, axis=-1)
                
            except Exception as e:
                logger.error(f"Error loading {self.image_ids[idx]}: {e}")
                X[i] = np.zeros((*self.model_dim, 1), dtype=np.float32)
                y[i] = np.zeros((*self.model_dim, 1), dtype=np.float32)
        
        return X, y
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

def train_hybrid_model(model_config):
    """Train the hybrid model with progressive complexity"""
    logger.info("🚀 Starting Hybrid CNN-Transformer Training")
    
    # Data configuration
    INPUT_SIZE = tuple(model_config.get('input_size', [128, 128, 128]))
    BATCH_SIZE = model_config.get('batch_size', 1)
    EPOCHS = model_config.get('epochs', 100)
    BASE_FILTERS = model_config.get('base_filters', 32)
    USE_TRANSFORMER = model_config.get('use_transformer', True)
    
    # Create data generators
    train_generator = OptimizedAtlasDataGenerator(
        base_dir="/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training",
        batch_size=BATCH_SIZE,
        model_dim=INPUT_SIZE,
        shuffle=True,
        augment=True
    )
    
    # Split data
    total_samples = len(train_generator.image_ids)
    train_samples = int(0.8 * total_samples)
    
    train_generator.image_ids = train_generator.image_ids[:train_samples]
    train_generator.on_epoch_end()
    
    val_generator = OptimizedAtlasDataGenerator(
        base_dir="/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training",
        batch_size=BATCH_SIZE,
        model_dim=INPUT_SIZE,
        shuffle=False,
        augment=False
    )
    val_generator.image_ids = val_generator.image_ids[train_samples:]
    val_generator.on_epoch_end()
    
    logger.info(f"Training samples: {len(train_generator.image_ids)}")
    logger.info(f"Validation samples: {len(val_generator.image_ids)}")
    
    # Build model
    logger.info(f"Building hybrid model (filters={BASE_FILTERS}, transformer={USE_TRANSFORMER})...")
    model = build_hybrid_model(
        input_shape=(*INPUT_SIZE, 1),
        base_filters=BASE_FILTERS,
        use_transformer=USE_TRANSFORMER
    )
    
    # Create optimizer with mixed precision
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=combined_dice_focal_loss,
        metrics=[dice_coefficient, 'accuracy']
    )
    
    # Log model info
    total_params = model.count_params()
    logger.info(f"Model parameters: {total_params:,}")
    logger.info(f"Estimated memory: ~{total_params * 4 / 1e9:.2f}GB")
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f'checkpoints/hybrid_model_{BASE_FILTERS}f_{"transformer" if USE_TRANSFORMER else "cnn"}.h5',
            save_best_only=True,
            monitor='val_dice_coefficient',
            mode='max',
            verbose=1
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
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            f'logs/training_history_{BASE_FILTERS}f_{"transformer" if USE_TRANSFORMER else "cnn"}.csv'
        )
    ]
    
    # Train model
    logger.info("Starting training...")
    try:
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("✅ Training completed successfully!")
        
        # Save final model
        final_path = f'models/hybrid_final_{BASE_FILTERS}f_{"transformer" if USE_TRANSFORMER else "cnn"}.h5'
        model.save(final_path)
        logger.info(f"Model saved to {final_path}")
        
        # Cleanup memory
        del model
        gc.collect()
        tf.keras.backend.clear_session()
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False

def main():
    """Progressive training from simple to complex"""
    
    # Training configurations in order of complexity
    configs = [
        {
            'name': 'Stage 1: Basic CNN',
            'input_size': [96, 96, 96],
            'batch_size': 2,
            'epochs': 30,
            'base_filters': 16,
            'use_transformer': False
        },
        {
            'name': 'Stage 2: CNN with Attention',
            'input_size': [112, 112, 112],
            'batch_size': 1,
            'epochs': 40,
            'base_filters': 24,
            'use_transformer': False
        },
        {
            'name': 'Stage 3: Small Hybrid',
            'input_size': [128, 128, 128],
            'batch_size': 1,
            'epochs': 50,
            'base_filters': 32,
            'use_transformer': True
        },
        {
            'name': 'Stage 4: Full SOTA',
            'input_size': [144, 144, 144],
            'batch_size': 1,
            'epochs': 100,
            'base_filters': 40,
            'use_transformer': True
        }
    ]
    
    logger.info("🎯 Progressive Training Pipeline")
    logger.info(f"Total stages: {len(configs)}")
    
    for i, config in enumerate(configs):
        logger.info(f"\n{'='*60}")
        logger.info(f"{config['name']}")
        logger.info(f"{'='*60}")
        
        success = train_hybrid_model(config)
        
        if not success:
            logger.warning(f"Stage {i+1} failed, stopping pipeline")
            break
        
        logger.info(f"✅ Stage {i+1} completed successfully")
        
        # Clear memory between stages
        gc.collect()
        tf.keras.backend.clear_session()
    
    logger.info("\n🎉 Progressive training pipeline completed!")

if __name__ == "__main__":
    main()
