#!/usr/bin/env python3
"""
Proven Enhanced Model: Baseline + Solid Improvements
NO experimental features - only proven, stable enhancements
Target: 65-70% validation Dice with ZERO risk
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

# Import working functions
from correct_full_training import load_full_655_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/proven_enhanced.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProvenConfig:
    DATA_DIR = "/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training"
    INPUT_SHAPE = (192, 224, 176, 1)
    BASE_FILTERS = 20          # Safe middle ground
    BATCH_SIZE = 4             # Multi-GPU ready
    EPOCHS = 50
    VALIDATION_SPLIT = 0.15
    INITIAL_LR = 1e-4
    
    CALLBACKS_DIR = lambda timestamp: Path(f'callbacks/proven_enhanced_{timestamp}')
    MODEL_SAVE_PATH = lambda timestamp: f'models/proven_enhanced_{timestamp}.h5'

def configure_multi_gpu_strategy():
    """Same proven multi-GPU setup"""
    gpus = tf.config.list_physical_devices('GPU')
    logger.info(f"Detected GPUs: {len(gpus)}")
    
    for i, gpu in enumerate(gpus):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            logger.warning(f"Could not set memory growth for GPU {i}: {e}")
    
    strategy = tf.distribute.MirroredStrategy()
    logger.info(f"🚀 MirroredStrategy with {strategy.num_replicas_in_sync} devices")
    return strategy

def enhanced_conv_block(x, filters, name_prefix="conv"):
    """Enhanced conv block with PROVEN components only"""
    # First conv
    x = layers.Conv3D(filters, 3, padding='same', name=f'{name_prefix}_conv1')(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn1')(x)  # PROVEN
    x = layers.Activation('gelu', name=f'{name_prefix}_gelu1')(x)  # Better than ReLU
    
    # Second conv
    x = layers.Conv3D(filters, 3, padding='same', name=f'{name_prefix}_conv2')(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn2')(x)
    
    # SE attention block (proven effective)
    se = layers.GlobalAveragePooling3D(name=f'{name_prefix}_se_pool')(x)
    se = layers.Dense(filters // 8, activation='relu', name=f'{name_prefix}_se_dense1')(se)
    se = layers.Dense(filters, activation='sigmoid', name=f'{name_prefix}_se_dense2')(se)
    se = layers.Reshape((1, 1, 1, filters), name=f'{name_prefix}_se_reshape')(se)
    x = layers.Multiply(name=f'{name_prefix}_se_apply')([x, se])
    
    x = layers.Activation('gelu', name=f'{name_prefix}_gelu2')(x)
    return x

def proven_attention_gate(g, x, filters, name_prefix="att"):
    """Proven attention gate implementation"""
    W_g = layers.Conv3D(filters, 1, padding='same', name=f'{name_prefix}_Wg')(g)
    W_x = layers.Conv3D(filters, 1, padding='same', name=f'{name_prefix}_Wx')(x)
    
    combined = layers.Add(name=f'{name_prefix}_add')([W_g, W_x])
    combined = layers.Activation('relu', name=f'{name_prefix}_relu')(combined)
    attention = layers.Conv3D(1, 1, padding='same', activation='sigmoid', name=f'{name_prefix}_att')(combined)
    
    return layers.Multiply(name=f'{name_prefix}_apply')([x, attention])

def build_proven_enhanced_model(input_shape=(192, 224, 176, 1), base_filters=20):
    """Build proven enhanced model - zero experimental risk"""
    inputs = layers.Input(shape=input_shape, name='inputs')
    
    # ENCODER - Enhanced conv blocks
    skip_connections = []
    
    # Stage 1: 20 filters
    conv1 = enhanced_conv_block(inputs, base_filters, 'stage1')
    skip_connections.append(conv1)
    pool1 = layers.MaxPooling3D(2, name='pool1')(conv1)
    
    # Stage 2: 40 filters  
    conv2 = enhanced_conv_block(pool1, base_filters * 2, 'stage2')
    skip_connections.append(conv2)
    pool2 = layers.MaxPooling3D(2, name='pool2')(conv2)
    
    # Stage 3: 80 filters
    conv3 = enhanced_conv_block(pool2, base_filters * 4, 'stage3')
    skip_connections.append(conv3)
    pool3 = layers.MaxPooling3D(2, name='pool3')(conv3)
    
    # Stage 4: 160 filters
    conv4 = enhanced_conv_block(pool3, base_filters * 8, 'stage4')
    skip_connections.append(conv4)
    pool4 = layers.MaxPooling3D(2, name='pool4')(conv4)
    
    # BOTTLENECK: 240 filters
    bottleneck = enhanced_conv_block(pool4, base_filters * 12, 'bottleneck')
    
    # DECODER - With proven attention gates
    
    # Stage 5
    up5 = layers.Conv3DTranspose(base_filters * 8, 2, strides=2, padding='same', name='up5')(bottleneck)
    att5 = proven_attention_gate(up5, skip_connections[3], base_filters * 4, 'att5')
    concat5 = layers.Concatenate(name='concat5')([up5, att5])
    conv5 = enhanced_conv_block(concat5, base_filters * 8, 'decode5')
    
    # Stage 6
    up6 = layers.Conv3DTranspose(base_filters * 4, 2, strides=2, padding='same', name='up6')(conv5)
    att6 = proven_attention_gate(up6, skip_connections[2], base_filters * 2, 'att6')
    concat6 = layers.Concatenate(name='concat6')([up6, att6])
    conv6 = enhanced_conv_block(concat6, base_filters * 4, 'decode6')
    
    # Stage 7
    up7 = layers.Conv3DTranspose(base_filters * 2, 2, strides=2, padding='same', name='up7')(conv6)
    att7 = proven_attention_gate(up7, skip_connections[1], base_filters, 'att7')
    concat7 = layers.Concatenate(name='concat7')([up7, att7])
    conv7 = enhanced_conv_block(concat7, base_filters * 2, 'decode7')
    
    # Stage 8
    up8 = layers.Conv3DTranspose(base_filters, 2, strides=2, padding='same', name='up8')(conv7)
    att8 = proven_attention_gate(up8, skip_connections[0], base_filters // 2, 'att8')
    concat8 = layers.Concatenate(name='concat8')([up8, att8])
    conv8 = enhanced_conv_block(concat8, base_filters, 'decode8')
    
    # OUTPUT
    output = layers.Conv3D(1, 1, activation='sigmoid', dtype='float32', name='output')(conv8)
    
    model = tf.keras.Model(inputs=inputs, outputs=output, name='ProvenEnhanced')
    return model

def enhanced_loss(y_true, y_pred, smooth=1e-6):
    """Enhanced loss with proven components"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Dice loss
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice_loss = 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    
    # Focal loss
    focal_loss = -tf.reduce_mean(
        0.25 * y_true * tf.pow(1 - y_pred, 2) * tf.math.log(y_pred + smooth) +
        0.75 * (1 - y_true) * tf.pow(y_pred, 2) * tf.math.log(1 - y_pred + smooth)
    )
    
    # Tversky loss for better recall
    alpha, beta = 0.3, 0.7
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    tversky_loss = 1 - (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    
    return 0.5 * dice_loss + 0.3 * focal_loss + 0.2 * tversky_loss

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

# Use same data generator as baseline (proven working)
from correct_full_training import CorrectAtlasDataGenerator

def create_proven_callbacks(callbacks_dir):
    """Create callbacks with proven settings"""
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

def train_proven_enhanced_model():
    """Train the proven enhanced model"""
    logger.info("🔧 PROVEN ENHANCED MODEL TRAINING")
    logger.info("=" * 60)
    logger.info("Strategy: Baseline + Only proven enhancements")
    logger.info("Target: 65-70% validation Dice (safe improvement)")
    logger.info("=" * 60)
    
    # Setup
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # Multi-GPU strategy
    strategy = configure_multi_gpu_strategy()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Load dataset (same as proven baseline)
    all_pairs = load_full_655_dataset()
    
    train_pairs, val_pairs = train_test_split(
        all_pairs,
        test_size=ProvenConfig.VALIDATION_SPLIT,
        random_state=42,
        shuffle=True
    )
    
    logger.info(f"Dataset: {len(train_pairs)} train, {len(val_pairs)} validation")
    
    # Create data generators (same as baseline)
    train_generator = CorrectAtlasDataGenerator(
        train_pairs, ProvenConfig.BATCH_SIZE, ProvenConfig.INPUT_SHAPE[:-1], shuffle=True
    )
    val_generator = CorrectAtlasDataGenerator(
        val_pairs, ProvenConfig.BATCH_SIZE, ProvenConfig.INPUT_SHAPE[:-1], shuffle=False
    )
    
    # Build model
    with strategy.scope():
        logger.info("Building proven enhanced model...")
        model = build_proven_enhanced_model(
            input_shape=ProvenConfig.INPUT_SHAPE,
            base_filters=ProvenConfig.BASE_FILTERS
        )
        
        param_count = model.count_params()
        logger.info(f"📊 Model parameters: {param_count:,}")
        
        # Compile
        optimizer = tf.keras.optimizers.Adam(learning_rate=ProvenConfig.INITIAL_LR)
        
        model.compile(
            optimizer=optimizer,
            loss=enhanced_loss,
            metrics=['accuracy', dice_coefficient, binary_dice_coefficient]
        )
    
    # Train
    try:
        logger.info("🔥 Starting proven enhanced training...")
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=ProvenConfig.EPOCHS,
            callbacks=create_proven_callbacks(ProvenConfig.CALLBACKS_DIR(timestamp)),
            verbose=1
        )
        
        # Save final model
        final_model_path = ProvenConfig.MODEL_SAVE_PATH(timestamp)
        model.save(final_model_path)
        
        # Results
        best_val_dice = max(history.history['val_dice_coefficient'])
        
        logger.info("=" * 60)
        logger.info("🏆 PROVEN ENHANCED TRAINING COMPLETED!")
        logger.info(f"Parameters: {param_count:,}")
        logger.info(f"Best validation Dice: {best_val_dice:.4f}")
        logger.info(f"Improvement over baseline: {(best_val_dice - 0.636):.3f}")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False

if __name__ == "__main__":
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    success = train_proven_enhanced_model()
    if success:
        logger.info("🎉 PROVEN ENHANCED MODEL SUCCESS!")
    else:
        exit(1)
