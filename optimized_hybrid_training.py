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
