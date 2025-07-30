#!/usr/bin/env python3
"""
Simplified SOTA Fallback Model
Removes all potentially problematic components to ensure stable training
Uses only proven, stable components
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
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split

# Import working functions from baseline
from correct_full_training import load_full_655_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/simplified_sota_fallback.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimplifiedConfig:
    DATA_DIR = "/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training"
    INPUT_SHAPE = (192, 224, 176, 1)
    BASE_FILTERS = 32  # Increased from 22 for better capacity
    BATCH_SIZE = 2     # Reduced for stability
    EPOCHS = 60
    VALIDATION_SPLIT = 0.1
    INITIAL_LR = 1e-4  # More conservative learning rate
    
    CALLBACKS_DIR = lambda timestamp: Path(f'callbacks/simplified_sota_fallback_{timestamp}')
    MODEL_SAVE_PATH = lambda timestamp: f'models/simplified_sota_fallback_{timestamp}.h5'

def configure_hardware():
    """Configure hardware with stability focus"""
    logger.info("🔧 Configuring hardware for stability...")
    
    # DISABLE mixed precision to avoid NaN issues
    tf.keras.mixed_precision.set_global_policy('float32')
    logger.info("✅ Using float32 (mixed precision DISABLED)")
    
    gpus = tf.config.list_physical_devices('GPU')
    logger.info(f"Detected GPUs: {len(gpus)}")
    
    for i, gpu in enumerate(gpus):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"✅ Memory growth enabled for GPU {i}")
        except Exception as e:
            logger.warning(f"Could not set memory growth for GPU {i}: {e}")
    
    # Use default strategy for simplicity
    return tf.distribute.get_strategy()

class StableConvBlock(layers.Layer):
    """Stable convolution block with only proven components"""
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        
        # Core convolutions
        self.conv1 = layers.Conv3D(filters, 3, padding='same', name=f'{self.name}_conv1')
        self.bn1 = layers.BatchNormalization(name=f'{self.name}_bn1')
        self.conv2 = layers.Conv3D(filters, 3, padding='same', name=f'{self.name}_conv2')
        self.bn2 = layers.BatchNormalization(name=f'{self.name}_bn2')
        
        # Simple SE attention (proven stable)
        self.se_pool = layers.GlobalAveragePooling3D(name=f'{self.name}_se_pool')
        self.se_dense1 = layers.Dense(max(filters // 16, 1), activation='relu', name=f'{self.name}_se_dense1')
        self.se_dense2 = layers.Dense(filters, activation='sigmoid', name=f'{self.name}_se_dense2')
        self.se_reshape = layers.Reshape((1, 1, 1, filters), name=f'{self.name}_se_reshape')
        
        # Residual projection
        self.residual_projection = layers.Conv3D(filters, 1, padding='same', name=f'{self.name}_residual_proj')
        self.residual_bn = layers.BatchNormalization(name=f'{self.name}_residual_bn')
    
    def call(self, x, training=None):
        residual = x
        
        # First conv path
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)  # Use ReLU instead of GELU for stability
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        # SE attention
        se = self.se_pool(x)
        se = self.se_dense1(se)
        se = self.se_dense2(se)
        se = self.se_reshape(se)
        x = x * se
        
        # Residual connection
        if tf.shape(residual)[-1] != self.filters:
            residual = self.residual_projection(residual)
            residual = self.residual_bn(residual, training=training)
        
        x = x + residual
        x = tf.nn.relu(x)
        
        return x

class SimpleAttentionGate(layers.Layer):
    """Simplified attention gate without boundary awareness"""
    def __init__(self, F_g, F_l, F_int, **kwargs):
        super().__init__(**kwargs)
        
        self.W_g = layers.Conv3D(F_int, 1, padding='same', name=f'{self.name}_Wg')
        self.W_x = layers.Conv3D(F_int, 1, padding='same', name=f'{self.name}_Wx')
        self.psi = layers.Conv3D(1, 1, padding='same', activation='sigmoid', name=f'{self.name}_psi')
        
    def call(self, g, x, training=None):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        combined = tf.nn.relu(g1 + x1)
        psi = self.psi(combined)
        return x * psi

def build_simplified_sota_model(input_shape=(192, 224, 176, 1), base_filters=32):
    """
    Simplified SOTA model with only stable components
    Target: ~5-7M parameters, stable training
    """
    logger.info("🏗️ Building simplified SOTA model...")
    
    inputs = layers.Input(shape=input_shape, name='main_input')
    
    # ENCODER
    skip_connections = []
    
    # Stage 1: 32 filters
    conv1 = StableConvBlock(base_filters, name='stage1')(inputs)
    skip_connections.append(conv1)
    pool1 = layers.MaxPooling3D(2, name='pool1')(conv1)
    
    # Stage 2: 64 filters
    conv2 = StableConvBlock(base_filters * 2, name='stage2')(pool1)
    skip_connections.append(conv2)
    pool2 = layers.MaxPooling3D(2, name='pool2')(conv2)
    
    # Stage 3: 128 filters
    conv3 = StableConvBlock(base_filters * 4, name='stage3')(pool2)
    skip_connections.append(conv3)
    pool3 = layers.MaxPooling3D(2, name='pool3')(conv3)
    
    # Stage 4: 256 filters
    conv4 = StableConvBlock(base_filters * 8, name='stage4')(pool3)
    skip_connections.append(conv4)
    pool4 = layers.MaxPooling3D(2, name='pool4')(conv4)
    
    # BOTTLENECK: 384 filters
    bottleneck = StableConvBlock(base_filters * 12, name='bottleneck')(pool4)
    
    # DECODER
    
    # Stage 5
    up5 = layers.Conv3DTranspose(base_filters * 8, 2, strides=2, padding='same', name='up5')(bottleneck)
    att5 = SimpleAttentionGate(base_filters * 8, base_filters * 8, base_filters * 4, name='att5')(up5, skip_connections[3])
    concat5 = layers.Concatenate(name='concat5')([up5, att5])
    conv5 = StableConvBlock(base_filters * 8, name='decode5')(concat5)
    
    # Stage 6
    up6 = layers.Conv3DTranspose(base_filters * 4, 2, strides=2, padding='same', name='up6')(conv5)
    att6 = SimpleAttentionGate(base_filters * 4, base_filters * 4, base_filters * 2, name='att6')(up6, skip_connections[2])
    concat6 = layers.Concatenate(name='concat6')([up6, att6])
    conv6 = StableConvBlock(base_filters * 4, name='decode6')(concat6)
    
    # Stage 7
    up7 = layers.Conv3DTranspose(base_filters * 2, 2, strides=2, padding='same', name='up7')(conv6)
    att7 = SimpleAttentionGate(base_filters * 2, base_filters * 2, base_filters, name='att7')(up7, skip_connections[1])
    concat7 = layers.Concatenate(name='concat7')([up7, att7])
    conv7 = StableConvBlock(base_filters * 2, name='decode7')(concat7)
    
    # Stage 8
    up8 = layers.Conv3DTranspose(base_filters, 2, strides=2, padding='same', name='up8')(conv7)
    att8 = SimpleAttentionGate(base_filters, base_filters, base_filters // 2, name='att8')(up8, skip_connections[0])
    concat8 = layers.Concatenate(name='concat8')([up8, att8])
    conv8 = StableConvBlock(base_filters, name='decode8')(concat8)
    
    # FINAL OUTPUT
    output = layers.Conv3D(1, 1, activation='sigmoid', dtype='float32', name='segmentation_output')(conv8)
    
    model = tf.keras.Model(inputs=inputs, outputs=output, name='SimplifiedSOTA')
    logger.info(f"✅ Model built successfully: {model.count_params():,} parameters")
    
    return model

def stable_combined_loss(y_true, y_pred, smooth=1e-6):
    """
    Stable loss function - NO boundary loss, NO complex computations
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # 1. Dice loss
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice_loss = 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    
    # 2. Focal loss (simplified)
    alpha = 0.25
    gamma = 2.0
    p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
    focal_loss = -alpha_t * tf.pow(1 - p_t, gamma) * tf.math.log(tf.clip_by_value(p_t, 1e-8, 1.0))
    focal_loss = tf.reduce_mean(focal_loss)
    
    # Simple combination
    return 0.7 * dice_loss + 0.3 * focal_loss

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Dice coefficient metric"""
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred, tf.float32))
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def binary_dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Binary dice coefficient metric"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred_binary)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

class StableDataGenerator(tf.keras.utils.Sequence):
    """Stable data generator without complex augmentation"""
    def __init__(self, image_mask_pairs, batch_size, target_shape, shuffle=True):
        self.image_mask_pairs = image_mask_pairs
        self.batch_size = batch_size
        self.target_shape = target_shape
        self.shuffle = shuffle
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
                
                img_data = nib.load(img_path).get_fdata(dtype=np.float32)
                mask_data = nib.load(mask_path).get_fdata(dtype=np.float32)
                
                if img_data.shape != self.target_shape:
                    img_data = self.resize_volume(img_data, self.target_shape)
                    mask_data = self.resize_volume(mask_data, self.target_shape)
                    mask_data = (mask_data > 0.5).astype(np.float32)
                
                # Simple, stable normalization
                img_data = self.normalize(img_data)
                
                # MINIMAL augmentation - only flip
                if self.shuffle and np.random.rand() > 0.5:
                    img_data = np.flip(img_data, axis=1)
                    mask_data = np.flip(mask_data, axis=1)
                
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
        # Robust normalization
        if img.max() > img.min():
            p1, p99 = np.percentile(img[img > 0], [1, 99])
            img = np.clip(img, p1, p99)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return img
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

def create_stable_callbacks(callbacks_dir):
    """Create stable callbacks"""
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
        tf.keras.callbacks.CSVLogger(str(callbacks_dir / 'training_log.csv'))
    ]

def train_simplified_sota():
    """
    Train simplified SOTA model with stability focus
    """
    logger.info("🚀 SIMPLIFIED SOTA TRAINING - STABILITY FOCUS")
    logger.info("=" * 80)
    logger.info("🔧 CHANGES FOR STABILITY:")
    logger.info("   ❌ NO mixed precision (using float32)")
    logger.info("   ❌ NO boundary-aware loss (using dice + focal)")
    logger.info("   ❌ NO Vision Mamba blocks")
    logger.info("   ❌ NO SAM2 attention")
    logger.info("   ❌ NO complex augmentation")
    logger.info("   ✅ Simple U-Net + SE attention + attention gates")
    logger.info("   ✅ Stable loss function")
    logger.info("   ✅ Conservative learning rate")
    logger.info("   ✅ Reduced batch size")
    logger.info("🎯 Target: 65-70% validation Dice with ZERO crashes")
    logger.info("=" * 80)
    
    # Setup
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    strategy = configure_hardware()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Load dataset
    all_pairs = load_full_655_dataset()
    
    train_pairs, val_pairs = train_test_split(
        all_pairs,
        test_size=SimplifiedConfig.VALIDATION_SPLIT,
        random_state=42,
        shuffle=True
    )
    
    logger.info(f"Dataset split: {len(train_pairs)} train, {len(val_pairs)} validation")
    
    # Create stable data generators
    train_generator = StableDataGenerator(
        train_pairs,
        SimplifiedConfig.BATCH_SIZE,
        SimplifiedConfig.INPUT_SHAPE[:-1],
        shuffle=True
    )
    
    val_generator = StableDataGenerator(
        val_pairs,
        SimplifiedConfig.BATCH_SIZE,
        SimplifiedConfig.INPUT_SHAPE[:-1],
        shuffle=False
    )
    
    # Build model
    with strategy.scope():
        model = build_simplified_sota_model(
            input_shape=SimplifiedConfig.INPUT_SHAPE,
            base_filters=SimplifiedConfig.BASE_FILTERS
        )
        
        param_count = model.count_params()
        logger.info(f"🔥 Model parameters: {param_count:,}")
        
        # Use simple Adam optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=SimplifiedConfig.INITIAL_LR)
        
        model.compile(
            optimizer=optimizer,
            loss=stable_combined_loss,
            metrics=['accuracy', dice_coefficient, binary_dice_coefficient]
        )
    
    # Test data loading
    logger.info("🧪 Testing data generators...")
    try:
        X_train, y_train = next(iter(train_generator))
        X_val, y_val = next(iter(val_generator))
        logger.info(f"✅ Train batch: X={X_train.shape}, y={y_train.shape}")
        logger.info(f"✅ Val batch: X={X_val.shape}, y={y_val.shape}")
        
        # Test forward pass
        test_output = model(X_train[:1], training=False)
        logger.info(f"✅ Forward pass successful: {test_output.shape}")
        
        # Test loss computation
        test_loss = stable_combined_loss(y_train[:1], test_output)
        logger.info(f"✅ Loss computation successful: {test_loss}")
        
    except Exception as e:
        logger.error(f"❌ Data/model test failed: {e}")
        return False
    
    # Train
    try:
        logger.info("🚀 Starting simplified SOTA training...")
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=SimplifiedConfig.EPOCHS,
            callbacks=create_stable_callbacks(
                SimplifiedConfig.CALLBACKS_DIR(timestamp)
            ),
            verbose=1
        )
        
        logger.info("✅ Training completed successfully!")
        
        # Save final model
        final_model_path = SimplifiedConfig.MODEL_SAVE_PATH(timestamp)
        model.save(final_model_path)
        
        # Results analysis
        final_val_dice = history.history['val_dice_coefficient'][-1]
        best_val_dice = max(history.history['val_dice_coefficient'])
        final_train_dice = history.history['dice_coefficient'][-1]
        
        logger.info("=" * 80)
        logger.info("🏆 SIMPLIFIED SOTA TRAINING RESULTS")
        logger.info("=" * 80)
        logger.info(f"📊 Model: {param_count:,} parameters")
        logger.info(f"📈 Final training Dice: {final_train_dice:.4f}")
        logger.info(f"📈 Final validation Dice: {final_val_dice:.4f}")
        logger.info(f"🏆 Best validation Dice: {best_val_dice:.4f}")
        logger.info(f"📊 Train/Val gap: {abs(final_train_dice - final_val_dice):.4f}")
        logger.info("")
        
        if best_val_dice >= 0.70:
            logger.info("🚀 EXCELLENT! 70%+ validation Dice achieved!")
        elif best_val_dice >= 0.65:
            logger.info("✅ SUCCESS! Target 65%+ validation Dice achieved!")
        elif best_val_dice >= 0.50:
            logger.info("👍 GOOD! Significant improvement from broken model!")
        else:
            logger.info("📊 Training completed - investigating further needed")
        
        logger.info(f"💾 Best model: {SimplifiedConfig.CALLBACKS_DIR(timestamp)}/best_model.h5")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    try:
        success = train_simplified_sota()
        if success:
            logger.info("🎉 SIMPLIFIED SOTA TRAINING COMPLETED!")
            logger.info("🚀 Stable training achieved - can now add features incrementally!")
        else:
            logger.error("❌ Even simplified training failed - deeper investigation needed")
            exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        exit(1)
