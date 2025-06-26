#!/usr/bin/env python3
"""
State-of-the-Art 3D U-Net with Attention Gates
Full complexity model for stroke lesion segmentation
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

def attention_gate(g, s, num_filters):
    """
    Attention gate with proper shape handling
    g: gating signal from decoder
    s: skip connection from encoder
    """
    # Ensure both inputs have same spatial dimensions
    Wg = layers.Conv3D(num_filters, kernel_size=1, padding='same')(g)
    Ws = layers.Conv3D(num_filters, kernel_size=1, padding='same')(s)
    
    # Combine features
    psi = layers.Activation('relu')(layers.Add()([Wg, Ws]))
    psi = layers.Conv3D(1, kernel_size=1, padding='same')(psi)
    psi = layers.Activation('sigmoid')(psi)
    
    # Apply attention
    return layers.Multiply()([s, psi])

def conv_block(x, num_filters, use_batchnorm=True):
    """Double convolution block with batch normalization"""
    x = layers.Conv3D(num_filters, 3, padding='same')(x)
    if use_batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv3D(num_filters, 3, padding='same')(x)
    if use_batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    return x

def build_sota_model(input_shape=(128, 128, 128, 1), base_filters=32):
    """
    Build State-of-the-Art 3D U-Net with Attention Gates
    ~22M parameters with base_filters=32
    """
    inputs = layers.Input(shape=input_shape)
    
    # Encoder path with skip connections
    skip_connections = []
    
    # Stage 1
    conv1 = conv_block(inputs, base_filters)
    skip_connections.append(conv1)
    pool1 = layers.MaxPooling3D(pool_size=2)(conv1)
    
    # Stage 2
    conv2 = conv_block(pool1, base_filters * 2)
    skip_connections.append(conv2)
    pool2 = layers.MaxPooling3D(pool_size=2)(conv2)
    
    # Stage 3
    conv3 = conv_block(pool2, base_filters * 4)
    skip_connections.append(conv3)
    pool3 = layers.MaxPooling3D(pool_size=2)(conv3)
    
    # Stage 4
    conv4 = conv_block(pool3, base_filters * 8)
    skip_connections.append(conv4)
    pool4 = layers.MaxPooling3D(pool_size=2)(conv4)
    
    # Bottleneck (Stage 5)
    conv5 = conv_block(pool4, base_filters * 16)
    
    # Decoder path with attention gates
    # Stage 6
    up6 = layers.Conv3DTranspose(base_filters * 8, kernel_size=2, strides=2, padding='same')(conv5)
    att6 = attention_gate(up6, skip_connections[3], base_filters * 8)
    concat6 = layers.Concatenate()([up6, att6])
    conv6 = conv_block(concat6, base_filters * 8)
    
    # Stage 7
    up7 = layers.Conv3DTranspose(base_filters * 4, kernel_size=2, strides=2, padding='same')(conv6)
    att7 = attention_gate(up7, skip_connections[2], base_filters * 4)
    concat7 = layers.Concatenate()([up7, att7])
    conv7 = conv_block(concat7, base_filters * 4)
    
    # Stage 8
    up8 = layers.Conv3DTranspose(base_filters * 2, kernel_size=2, strides=2, padding='same')(conv7)
    att8 = attention_gate(up8, skip_connections[1], base_filters * 2)
    concat8 = layers.Concatenate()([up8, att8])
    conv8 = conv_block(concat8, base_filters * 2)
    
    # Stage 9
    up9 = layers.Conv3DTranspose(base_filters, kernel_size=2, strides=2, padding='same')(conv8)
    att9 = attention_gate(up9, skip_connections[0], base_filters)
    concat9 = layers.Concatenate()([up9, att9])
    conv9 = conv_block(concat9, base_filters)
    
    # Output layer - ensure float32 for mixed precision
    outputs = layers.Conv3D(1, kernel_size=1, activation='sigmoid', dtype='float32')(conv9)
    
    model = Model(inputs=inputs, outputs=outputs, name='SOTA_Attention_UNet')
    
    return model

# SE-ResNet style blocks for even more advanced architecture
def se_block(x, num_filters, reduction_ratio=16):
    """Squeeze-and-Excitation block"""
    # Squeeze
    se = layers.GlobalAveragePooling3D()(x)
    
    # Excitation
    se = layers.Dense(num_filters // reduction_ratio, activation='relu')(se)
    se = layers.Dense(num_filters, activation='sigmoid')(se)
    
    # Reshape and scale
    se = layers.Reshape((1, 1, 1, num_filters))(se)
    x = layers.Multiply()([x, se])
    
    return x

def resnet_block(x, num_filters):
    """ResNet-style block with SE attention"""
    shortcut = x
    
    # Main path
    x = layers.Conv3D(num_filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv3D(num_filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # SE block
    x = se_block(x, num_filters)
    
    # Shortcut connection
    if shortcut.shape[-1] != num_filters:
        shortcut = layers.Conv3D(num_filters, 1, padding='same')(shortcut)
    
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    
    return x

def build_advanced_sota_model(input_shape=(128, 128, 128, 1), base_filters=32):
    """
    Build Advanced SOTA model with ResNet blocks and SE attention
    Even more parameters and complexity
    """
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv3D(base_filters, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Encoder with ResNet blocks
    skip_connections = []
    
    # Stage 1
    x = resnet_block(x, base_filters)
    x = resnet_block(x, base_filters)
    skip_connections.append(x)
    x = layers.MaxPooling3D(2)(x)
    
    # Stage 2
    x = resnet_block(x, base_filters * 2)
    x = resnet_block(x, base_filters * 2)
    skip_connections.append(x)
    x = layers.MaxPooling3D(2)(x)
    
    # Stage 3
    x = resnet_block(x, base_filters * 4)
    x = resnet_block(x, base_filters * 4)
    skip_connections.append(x)
    x = layers.MaxPooling3D(2)(x)
    
    # Stage 4
    x = resnet_block(x, base_filters * 8)
    x = resnet_block(x, base_filters * 8)
    skip_connections.append(x)
    x = layers.MaxPooling3D(2)(x)
    
    # Bottleneck
    x = resnet_block(x, base_filters * 16)
    x = resnet_block(x, base_filters * 16)
    
    # Decoder with attention
    # Stage 6
    x = layers.Conv3DTranspose(base_filters * 8, 2, strides=2, padding='same')(x)
    att = attention_gate(x, skip_connections[3], base_filters * 8)
    x = layers.Concatenate()([x, att])
    x = resnet_block(x, base_filters * 8)
    
    # Stage 7
    x = layers.Conv3DTranspose(base_filters * 4, 2, strides=2, padding='same')(x)
    att = attention_gate(x, skip_connections[2], base_filters * 4)
    x = layers.Concatenate()([x, att])
    x = resnet_block(x, base_filters * 4)
    
    # Stage 8
    x = layers.Conv3DTranspose(base_filters * 2, 2, strides=2, padding='same')(x)
    att = attention_gate(x, skip_connections[1], base_filters * 2)
    x = layers.Concatenate()([x, att])
    x = resnet_block(x, base_filters * 2)
    
    # Stage 9
    x = layers.Conv3DTranspose(base_filters, 2, strides=2, padding='same')(x)
    att = attention_gate(x, skip_connections[0], base_filters)
    x = layers.Concatenate()([x, att])
    x = resnet_block(x, base_filters)
    
    # Output
    outputs = layers.Conv3D(1, 1, activation='sigmoid', dtype='float32')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='Advanced_SOTA_Attention_UNet')
    
    return model

if __name__ == "__main__":
    # Test model creation
    print("Building SOTA Attention U-Net...")
    model = build_sota_model(input_shape=(128, 128, 128, 1), base_filters=32)
    print(f"✓ Model created: {model.name}")
    print(f"✓ Parameters: {model.count_params():,}")
    print(f"✓ Input shape: {model.input_shape}")
    print(f"✓ Output shape: {model.output_shape}")
    
    # Test forward pass
    import numpy as np
    dummy_input = np.random.random((1, 128, 128, 128, 1)).astype(np.float32)
    output = model.predict(dummy_input, verbose=0)
    print(f"✓ Forward pass successful: output shape {output.shape}")
