#!/usr/bin/env python3
"""
Debug SAM2 Attention Issue
"""

import os
import tensorflow as tf
import numpy as np

def test_sam2_attention():
    print("Testing SAM2 Attention issue...")
    
    # Simulate the exact failing scenario from the logs
    # Input: tf.Tensor(shape=(None, 96, 112, 88, 44), dtype=float32)
    batch_size = 2
    height, width, depth = 96, 112, 88
    channels = 44
    
    # Create test input (same shape as your failing case)
    x = tf.random.normal((batch_size, height, width, depth, channels), dtype=tf.float32)
    print(f"Input shape: {x.shape}")
    
    # Test the SAM2InspiredAttention layer
    print("\n--- TESTING SAM2 ATTENTION ---")
    try:
        # Create SAM2 attention layer
        sam2_attention = SAM2InspiredAttention(channels=channels, memory_size=32)
        
        # Test forward pass
        output = sam2_attention(x, training=False)
        print(f"✅ SAM2 attention successful: {x.shape} → {output.shape}")
        
    except Exception as e:
        print(f"❌ SAM2 attention failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Let's debug step by step
        print("\n--- DEBUGGING SAM2 STEP BY STEP ---")
        try:
            sam2_attention = SAM2InspiredAttention(channels=channels, memory_size=32)
            
            # Test individual components
            print("Testing query conv...")
            queries = sam2_attention.query_conv(x)
            print(f"Queries shape: {queries.shape}")
            
            print("Testing key conv...")
            keys = sam2_attention.key_conv(x)
            print(f"Keys shape: {keys.shape}")
            
            print("Testing value conv...")
            values = sam2_attention.value_conv(x)
            print(f"Values shape: {values.shape}")
            
            # Test reshaping
            print("Testing flatten...")
            queries_flat = tf.reshape(queries, [batch_size, -1, channels // 8])
            keys_flat = tf.reshape(keys, [batch_size, -1, channels // 8])
            values_flat = tf.reshape(values, [batch_size, -1, channels])
            
            print(f"Queries flat: {queries_flat.shape}")
            print(f"Keys flat: {keys_flat.shape}")
            print(f"Values flat: {values_flat.shape}")
            
            # Test attention computation
            print("Testing attention scores...")
            attention_scores = tf.matmul(queries_flat, keys_flat, transpose_b=True)
            print(f"Attention scores: {attention_scores.shape}")
            
            # Check if this is where it fails
            print("Testing softmax...")
            attention_weights = tf.nn.softmax(attention_scores, axis=-1)
            print(f"Attention weights: {attention_weights.shape}")
            
            print("Testing matmul with values...")
            attended = tf.matmul(attention_weights, values_flat)
            print(f"Attended: {attended.shape}")
            
            print("✅ All SAM2 components work individually")
            
        except Exception as e2:
            print(f"❌ SAM2 step-by-step failed at: {e2}")
            import traceback
            traceback.print_exc()

class SAM2InspiredAttention(tf.keras.layers.Layer):
    """
    SAM-2 inspired attention with self-sorting memory concepts
    Adapted for medical image segmentation
    """
    def __init__(self, channels, memory_size=32, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.memory_size = memory_size
        
        # Attention components
        self.query_conv = tf.keras.layers.Conv3D(channels // 8, 1)
        self.key_conv = tf.keras.layers.Conv3D(channels // 8, 1)
        self.value_conv = tf.keras.layers.Conv3D(channels, 1)
        
        # Memory bank (simplified)
        self.memory_keys = self.add_weight(
            shape=(memory_size, channels // 8),
            initializer='random_normal',
            trainable=True,
            name='memory_keys'
        )
        
        # Output projection
        self.gamma = self.add_weight(
            shape=(1,),
            initializer='zeros',
            trainable=True,
            name='attention_gamma'
        )
        
    def call(self, x, training=None):
        """
        SAM-2 inspired attention with memory bank
        """
        batch_size, height, width, depth, channels = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3], tf.shape(x)[4]
        
        # Generate queries, keys, values
        queries = self.query_conv(x)  # [B, H, W, D, C//8]
        keys = self.key_conv(x)       # [B, H, W, D, C//8]
        values = self.value_conv(x)   # [B, H, W, D, C]
        
        # Flatten spatial dimensions
        queries_flat = tf.reshape(queries, [batch_size, -1, channels // 8])
        keys_flat = tf.reshape(keys, [batch_size, -1, channels // 8])
        values_flat = tf.reshape(values, [batch_size, -1, channels])
        
        # Attention computation
        attention_scores = tf.matmul(queries_flat, keys_flat, transpose_b=True)
        attention_scores = attention_scores / tf.sqrt(tf.cast(channels // 8, tf.float32))
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        
        # Apply attention
        attended = tf.matmul(attention_weights, values_flat)
        
        # Reshape back to 3D
        attended = tf.reshape(attended, [batch_size, height, width, depth, channels])
        
        # Residual connection with learnable scaling
        return x + self.gamma * attended

if __name__ == "__main__":
    # Configure TensorFlow
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    test_sam2_attention()
    print("\n🔧 SAM2 attention debugging completed!")
