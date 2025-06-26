#!/usr/bin/env python3
"""
Robust Training Script for Stroke Lesion Segmentation
Handles both GPU and CPU scenarios with automatic fallback
"""

import os
import sys
import logging
import tensorflow as tf
import numpy as np
from pathlib import Path
import yaml
import argparse
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def configure_gpu():
    """Configure GPU settings with fallback to CPU"""
    try:
        # Check for GPUs
        gpus = tf.config.list_physical_devices('GPU')
        logger.info(f"Detected {len(gpus)} GPU(s)")
        
        if gpus:
            # Configure memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set visible devices
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                visible_devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
                logger.info(f"Using GPU devices: {visible_devices}")
            
            # Test GPU computation
            with tf.device('/GPU:0'):
                test_tensor = tf.constant([1.0, 2.0, 3.0])
                result = tf.reduce_sum(test_tensor)
                logger.info(f"GPU test successful: {result.numpy()}")
            
            return len(gpus)
        else:
            logger.warning("No GPUs detected - using CPU")
            return 0
            
    except Exception as e:
        logger.error(f"GPU configuration failed: {e}")
        logger.warning("Falling back to CPU")
        return 0

def load_config(config_path: str = 'config/default_config.yaml'):
    """Load configuration with GPU-aware adjustments"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using defaults")
        config = get_default_config()
    
    return config

def get_default_config():
    """Default configuration"""
    return {
        'model': {
            'input_shape': [192, 224, 176, 1],
            'base_filters': 32,
            'depth': 5,
            'use_attention': True,
            'use_deep_supervision': True
        },
        'training': {
            'epochs': 100,
            'batch_size_per_gpu': 2,
            'learning_rate': 1e-4,
            'validation_split': 0.2,
            'patience': 15
        },
        'data': {
            'data_dir': '/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training',
            'cache_dir': '/tmp/stroke_cache',
            'num_workers': 4,
            'prefetch': 2
        }
    }

def adjust_config_for_hardware(config, num_gpus):
    """Adjust configuration based on available hardware"""
    # Fix batch size calculation
    batch_size_per_gpu = config['training']['batch_size_per_gpu']
    
    if num_gpus > 0:
        global_batch_size = batch_size_per_gpu * num_gpus
        strategy = tf.distribute.MirroredStrategy()
        logger.info(f"Using MirroredStrategy with {num_gpus} GPUs")
        logger.info(f"Global batch size: {global_batch_size}")
    else:
        # CPU fallback
        global_batch_size = 1  # Use smaller batch size for CPU
        strategy = tf.distribute.get_strategy()  # Default strategy
        logger.info("Using CPU with reduced batch size")
        
        # Adjust other parameters for CPU
        config['training']['batch_size_per_gpu'] = 1
        config['data']['num_workers'] = min(4, os.cpu_count())
        config['model']['base_filters'] = 16  # Reduce model size
    
    config['training']['global_batch_size'] = global_batch_size
    return config, strategy

def create_model(config):
    """Create the Attention U-Net model"""
    input_shape = config['model']['input_shape']
    base_filters = config['model']['base_filters']
    depth = config['model']['depth']
    
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Encoder path
    skip_connections = []
    x = inputs
    
    for i in range(depth):
        filters = base_filters * (2 ** i)
        
        # Convolutional block
        x = tf.keras.layers.Conv3D(filters, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.1)(x)
        
        x = tf.keras.layers.Conv3D(filters, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.1)(x)
        
        # Add attention if enabled
        if config['model']['use_attention'] and i > 0:
            # Simple channel attention
            att = tf.keras.layers.GlobalAveragePooling3D()(x)
            att = tf.keras.layers.Dense(filters // 8, activation='relu')(att)
            att = tf.keras.layers.Dense(filters, activation='sigmoid')(att)
            att = tf.keras.layers.Reshape((1, 1, 1, filters))(att)
            x = tf.keras.layers.Multiply()([x, att])
        
        if i < depth - 1:
            skip_connections.append(x)
            x = tf.keras.layers.MaxPooling3D(2)(x)
    
    # Decoder path
    for i in range(depth - 1):
        filters = base_filters * (2 ** (depth - 2 - i))
        
        # Upsampling
        x = tf.keras.layers.UpSampling3D(2)(x)
        x = tf.keras.layers.Conv3D(filters, 2, padding='same')(x)
        
        # Skip connection
        skip = skip_connections[-(i + 1)]
        
        # Simple attention gate if enabled
        if config['model']['use_attention']:
            # Attention gate
            g = tf.keras.layers.Conv3D(filters, 1)(x)
            s = tf.keras.layers.Conv3D(filters, 1)(skip)
            att = tf.keras.layers.Add()([g, s])
            att = tf.keras.layers.Activation('relu')(att)
            att = tf.keras.layers.Conv3D(1, 1, activation='sigmoid')(att)
            skip = tf.keras.layers.Multiply()([skip, att])
        
        x = tf.keras.layers.Concatenate()([x, skip])
        
        # Convolutional block
        x = tf.keras.layers.Conv3D(filters, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.1)(x)
        
        x = tf.keras.layers.Conv3D(filters, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.1)(x)
    
    # Output
    outputs = tf.keras.layers.Conv3D(2, 1, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def generalized_dice_loss(y_true, y_pred, smooth=1e-6):
    """Generalized Dice Loss for handling class imbalance"""
    # Convert to one-hot if needed
    if len(y_true.shape) == 4:
        y_true = tf.one_hot(tf.cast(y_true[..., 0], tf.int32), 2)
    
    # Calculate weights for each class
    w = 1 / (tf.reduce_sum(y_true, axis=[1, 2, 3])**2 + smooth)
    
    # Calculate intersection and union
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3])
    
    # Weighted Dice coefficient
    dice = (2 * tf.reduce_sum(w * intersection, axis=1) + smooth) / \
           (tf.reduce_sum(w * union, axis=1) + smooth)
    
    return 1 - dice

def focal_loss(y_true, y_pred, alpha=0.25, gamma=3.0):
    """Focal Loss for handling class imbalance"""
    # Convert to one-hot if needed
    if len(y_true.shape) == 4:
        y_true = tf.one_hot(tf.cast(y_true[..., 0], tf.int32), 2)
    
    # Clip predictions to prevent log(0)
    y_pred = tf.clip_by_value(y_pred, 1e-8, 1 - 1e-8)
    
    # Calculate focal loss
    ce = -y_true * tf.math.log(y_pred)
    weight = alpha * y_true + (1 - alpha) * (1 - y_true)
    fl = weight * (1 - y_pred)**gamma * ce
    
    return tf.reduce_mean(fl)

def combined_loss(y_true, y_pred):
    """Combined loss function"""
    dice = generalized_dice_loss(y_true, y_pred)
    focal = focal_loss(y_true, y_pred)
    return 0.6 * dice + 0.4 * focal

def create_real_data_loader(config):
    """Create real ATLAS 2.0 data loader"""
    import sys
    sys.path.append('.')
    from real_data_loader import RealAtlasDataLoader
    
    data_dir = config['data']['data_dir']
    target_shape = tuple(config['model']['input_shape'][:-1])  # Remove channel dim
    
    return RealAtlasDataLoader(data_dir, target_shape)

def train_model():
    """Main training function"""
    logger.info("🚀 Starting Stroke Lesion Segmentation Training")
    
    # Configure hardware
    num_gpus = configure_gpu()
    
    # Load and adjust configuration
    config = load_config()
    config, strategy = adjust_config_for_hardware(config, num_gpus)
    
    logger.info(f"Configuration: {config}")
    
    # Create model within strategy scope
    with strategy.scope():
        model = create_model(config)
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(config['training']['learning_rate']),
            loss=combined_loss,
            metrics=['accuracy']
        )
    
    logger.info(f"Model created with {model.count_params():,} parameters")
    
    # Create real data loader
    logger.info("📊 Creating real ATLAS 2.0 data loader...")
    data_loader = create_real_data_loader(config)
    
    # Create datasets
    train_dataset, val_dataset = data_loader.create_dataset(
        batch_size=config['training']['global_batch_size'],
        validation_split=config['training']['validation_split']
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'checkpoints/best_model.h5',
            save_best_only=True,
            monitor='loss',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            patience=10,
            factor=0.5,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=config['training']['patience'],
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Train model
    try:
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=config['training']['epochs'],
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("✅ Training completed successfully!")
        return model, history
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/default_config.yaml', 
                       help='Configuration file path')
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    try:
        model, history = train_model()
        logger.info("🎯 Training pipeline executed successfully!")
    except Exception as e:
        logger.error(f"💥 Training pipeline failed: {e}")
        sys.exit(1)
