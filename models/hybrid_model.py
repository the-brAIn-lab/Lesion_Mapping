#!/usr/bin/env python3
"""
State-of-the-Art Hybrid CNN-Transformer Model for Stroke Lesion Segmentation
Combines convolutional networks with transformer components and attention mechanisms
Memory-optimized for RTX 4500 (24GB) with mixed precision training
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Optional, List
import math

# Enable mixed precision for memory efficiency
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

class SE_Block(layers.Layer):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, filters: int, reduction: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.reduction = reduction
        
        self.global_pool = layers.GlobalAveragePooling3D()
        self.dense1 = layers.Dense(filters // reduction, activation='relu')
        self.dense2 = layers.Dense(filters, activation='sigmoid')
        self.multiply = layers.Multiply()
        
    def call(self, inputs):
        se = self.global_pool(inputs)
        se = self.dense1(se)
        se = self.dense2(se)
        se = tf.expand_dims(tf.expand_dims(tf.expand_dims(se, 1), 1), 1)
        return self.multiply([inputs, se])
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "reduction": self.reduction
        })
        return config

class ResNeXtBlock(layers.Layer):
    """ResNeXt block with grouped convolutions and SE attention"""
    def __init__(self, filters: int, cardinality: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.cardinality = cardinality
        
        # Grouped convolutions (simulated with separate conv layers)
        self.conv1 = layers.Conv3D(filters, 1, activation='relu', padding='same')
        self.conv2 = layers.Conv3D(filters, 3, activation='relu', padding='same')
        self.conv3 = layers.Conv3D(filters, 1, padding='same')
        
        self.se_block = SE_Block(filters)
        self.batch_norm = layers.BatchNormalization()
        self.activation = layers.Activation('relu')
        
        # Skip connection
        self.skip_conv = layers.Conv3D(filters, 1, padding='same')
        
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.se_block(x)
        
        # Skip connection with dimension matching
        skip = self.skip_conv(inputs) if inputs.shape[-1] != self.filters else inputs
        
        x = layers.Add()([x, skip])
        x = self.batch_norm(x)
        return self.activation(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "cardinality": self.cardinality
        })
        return config

class MultiHeadSelfAttention3D(layers.Layer):
    """3D Multi-head self-attention optimized for volumetric data"""
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.wq = layers.Dense(embed_dim)
        self.wk = layers.Dense(embed_dim)
        self.wv = layers.Dense(embed_dim)
        self.dense = layers.Dense(embed_dim)
        self.dropout_layer = layers.Dropout(dropout)
        self.layer_norm = layers.LayerNormalization()
        
    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1] * tf.shape(inputs)[2] * tf.shape(inputs)[3]
        
        # Reshape to sequence format
        x = tf.reshape(inputs, [batch_size, seq_len, self.embed_dim])
        
        # Generate Q, K, V
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        
        # Reshape for multi-head attention
        q = tf.reshape(q, [batch_size, seq_len, self.num_heads, self.head_dim])
        k = tf.reshape(k, [batch_size, seq_len, self.num_heads, self.head_dim])
        v = tf.reshape(v, [batch_size, seq_len, self.num_heads, self.head_dim])
        
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])
        
        # Scaled dot-product attention
        attention_scores = tf.matmul(q, k, transpose_b=True)
        attention_scores = attention_scores / tf.sqrt(tf.cast(self.head_dim, tf.float32))
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_weights = self.dropout_layer(attention_weights, training=training)
        
        attention_output = tf.matmul(attention_weights, v)
        attention_output = tf.transpose(attention_output, [0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, [batch_size, seq_len, self.embed_dim])
        
        # Dense projection
        output = self.dense(attention_output)
        output = self.dropout_layer(output, training=training)
        
        # Reshape back to spatial format
        original_shape = tf.shape(inputs)
        output = tf.reshape(output, original_shape)
        
        # Residual connection and layer norm
        output = self.layer_norm(inputs + output)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dropout": self.dropout
        })
        return config

class TransformerBlock(layers.Layer):
    """Complete transformer block with self-attention and FFN"""
    def __init__(self, embed_dim: int, num_heads: int = 8, mlp_ratio: int = 4, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        
        self.attention = MultiHeadSelfAttention3D(embed_dim, num_heads, dropout)
        self.mlp_dim = int(embed_dim * mlp_ratio)
        
        # Feed-forward network
        self.ffn = keras.Sequential([
            layers.Dense(self.mlp_dim, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
            layers.Dropout(dropout)
        ])
        self.layer_norm = layers.LayerNormalization()
        
    def call(self, inputs, training=None):
        # Self-attention
        attn_output = self.attention(inputs, training=training)
        
        # Feed-forward network with residual connection
        ffn_input = attn_output
        ffn_output = self.ffn(attn_output, training=training)
        
        return self.layer_norm(ffn_input + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout
        })
        return config

class AttentionGate(layers.Layer):
    """Attention gate for skip connections"""
    def __init__(self, filters: int, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        
        self.conv_x = layers.Conv3D(filters, 1, strides=2, padding='same')
        self.conv_g = layers.Conv3D(filters, 1, padding='same')
        self.conv_psi = layers.Conv3D(1, 1, padding='same')
        self.conv_out = layers.Conv3D(filters, 1, padding='same')
        
        self.relu = layers.Activation('relu')
        self.sigmoid = layers.Activation('sigmoid')
        self.upsample = layers.UpSampling3D(size=2)
        
    def call(self, x, g):
        # x: skip connection, g: gating signal
        x1 = self.conv_x(x)
        g1 = self.conv_g(g)
        
        psi = self.relu(x1 + g1)
        psi = self.conv_psi(psi)
        psi = self.sigmoid(psi)
        
        # Upsample attention coefficients
        psi = self.upsample(psi)
        
        # Apply attention
        out = x * psi
        out = self.conv_out(out)
        
        return out
    
    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        return config

class HybridCNNTransformer(keras.Model):
    """State-of-the-Art Hybrid CNN-Transformer for Stroke Lesion Segmentation"""
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int, int] = (128, 128, 128, 1),
                 num_classes: int = 1,
                 base_filters: int = 32,
                 depth: int = 4,
                 use_transformer: bool = True,
                 transformer_depth: int = 2,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.base_filters = base_filters
        self.depth = depth
        self.use_transformer = use_transformer
        self.transformer_depth = transformer_depth
        self.num_heads = num_heads
        self.dropout_rate = dropout
        
        # Build encoder
        self.encoder_blocks = []
        self.down_samples = []
        self.transformer_blocks = []
        
        filters = base_filters
        for i in range(depth):
            # ResNeXt block with SE attention
            self.encoder_blocks.append(ResNeXtBlock(filters))
            
            # Add transformer for deeper layers (memory permitting)
            if use_transformer and i >= depth - 2:  # Only in deepest layers
                for _ in range(transformer_depth):
                    self.transformer_blocks.append(TransformerBlock(
                        embed_dim=filters,
                        num_heads=min(num_heads, filters // 8),  # Adaptive heads
                        dropout=dropout
                    ))
            
            # Downsampling (except for the last layer)
            if i < depth - 1:
                self.down_samples.append(layers.MaxPooling3D(2))
            
            filters = min(filters * 2, 512)  # Cap at 512 to save memory
        
        # Bottleneck with transformer
        self.bottleneck = ResNeXtBlock(filters)
        if use_transformer:
            self.bottleneck_transformer = TransformerBlock(
                embed_dim=filters,
                num_heads=min(num_heads, filters // 8),
                dropout=dropout
            )
        
        # Build decoder
        self.decoder_blocks = []
        self.up_samples = []
        self.attention_gates = []
        
        for i in range(depth - 1):
            filters = filters // 2
            
            # Upsampling
            self.up_samples.append(layers.UpSampling3D(2))
            
            # Attention gate for skip connection
            self.attention_gates.append(AttentionGate(filters))
            
            # Decoder block
            self.decoder_blocks.append(ResNeXtBlock(filters))
        
        # Output layers
        self.final_conv = layers.Conv3D(num_classes, 1, activation='sigmoid', 
                                       dtype=tf.float32)  # Ensure float32 output
        
        # Deep supervision outputs (optional)
        self.deep_supervision_outputs = []
        temp_filters = base_filters * (2 ** (depth - 2))
        for i in range(depth - 1):
            self.deep_supervision_outputs.append(
                layers.Conv3D(num_classes, 1, activation='sigmoid', dtype=tf.float32)
            )
            temp_filters = temp_filters // 2
    
    def call(self, inputs, training=None):
        # Store skip connections
        skip_connections = []
        x = inputs
        
        # Encoder path
        transformer_idx = 0
        for i in range(self.depth):
            x = self.encoder_blocks[i](x, training=training)
            
            # Apply transformer in deeper layers
            if self.use_transformer and i >= self.depth - 2:
                for _ in range(self.transformer_depth):
                    x = self.transformer_blocks[transformer_idx](x, training=training)
                    transformer_idx += 1
            
            if i < self.depth - 1:
                skip_connections.append(x)
                x = self.down_samples[i](x)
        
        # Bottleneck
        x = self.bottleneck(x, training=training)
        if self.use_transformer:
            x = self.bottleneck_transformer(x, training=training)
        
        # Decoder path
        deep_outputs = []
        for i in range(self.depth - 1):
            # Upsampling
            x = self.up_samples[i](x)
            
            # Attention gate with skip connection
            skip = skip_connections[-(i + 1)]
            skip = self.attention_gates[i](skip, x)
            
            # Concatenate
            x = layers.Concatenate()([x, skip])
            
            # Decoder block
            x = self.decoder_blocks[i](x, training=training)
            
            # Deep supervision output
            deep_out = self.deep_supervision_outputs[i](x)
            deep_outputs.append(deep_out)
        
        # Final output
        output = self.final_conv(x)
        
        if training:
            return output, deep_outputs
        else:
            return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "input_shape": self.input_shape,
            "num_classes": self.num_classes,
            "base_filters": self.base_filters,
            "depth": self.depth,
            "use_transformer": self.use_transformer,
            "transformer_depth": self.transformer_depth,
            "num_heads": self.num_heads,
            "dropout": self.dropout_rate
        })
        return config

def create_hybrid_model(
    input_shape: Tuple[int, int, int, int] = (128, 128, 128, 1),
    base_filters: int = 32,
    depth: int = 4,
    use_transformer: bool = True,
    memory_efficient: bool = True
) -> HybridCNNTransformer:
    """
    Create hybrid CNN-Transformer model with memory optimization
    
    Args:
        input_shape: Model input shape (H, W, D, C)
        base_filters: Base number of filters (will be doubled each layer)
        depth: Number of encoder/decoder layers
        use_transformer: Whether to use transformer components
        memory_efficient: Enable memory optimization features
    
    Returns:
        Compiled hybrid model
    """
    
    # Adjust parameters for memory efficiency
    if memory_efficient:
        # Reduce transformer usage for memory
        transformer_depth = 1 if use_transformer else 0
        num_heads = min(8, base_filters // 4)
        dropout = 0.1
    else:
        transformer_depth = 2
        num_heads = 8
        dropout = 0.1
    
    model = HybridCNNTransformer(
        input_shape=input_shape,
        num_classes=1,
        base_filters=base_filters,
        depth=depth,
        use_transformer=use_transformer,
        transformer_depth=transformer_depth,
        num_heads=num_heads,
        dropout=dropout
    )
    
    return model

def create_progressive_models():
    """Create models of increasing complexity for progressive training"""
    
    models = {
        'basic': create_hybrid_model(
            base_filters=16,
            depth=3,
            use_transformer=False,
            memory_efficient=True
        ),
        'attention': create_hybrid_model(
            base_filters=24,
            depth=4,
            use_transformer=False,
            memory_efficient=True
        ),
        'hybrid_small': create_hybrid_model(
            base_filters=32,
            depth=4,
            use_transformer=True,
            memory_efficient=True
        ),
        'hybrid_full': create_hybrid_model(
            base_filters=48,
            depth=5,
            use_transformer=True,
            memory_efficient=False
        )
    }
    
    return models

# Example usage and testing
if __name__ == "__main__":
    # Test model creation
    print("🔬 Testing Hybrid CNN-Transformer Model")
    
    # Create different model variants
    models = create_progressive_models()
    
    for name, model in models.items():
        try:
            # Build model with dummy input
            dummy_input = tf.random.normal((1, 128, 128, 128, 1))
            output = model(dummy_input, training=False)
            
            param_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
            
            print(f"\n✅ {name.upper()} Model:")
            print(f"   Parameters: {param_count:,}")
            print(f"   Input shape: {dummy_input.shape}")
            print(f"   Output shape: {output.shape}")
            print(f"   Memory estimate: ~{param_count * 4 / 1e9:.1f}GB")
            
        except Exception as e:
            print(f"❌ {name} failed: {e}")
    
    print(f"\n🎯 Recommended progression:")
    print(f"1. Start with 'basic' (no transformer)")
    print(f"2. Upgrade to 'attention' (SE blocks + attention gates)")
    print(f"3. Add 'hybrid_small' (transformers in bottleneck)")
    print(f"4. Scale to 'hybrid_full' (full SOTA architecture)")
