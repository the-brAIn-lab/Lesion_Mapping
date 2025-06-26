#!/usr/bin/env python3
"""
SAM2 Attention Fix - Memory-Efficient Implementation
Replaces quadratic attention with local/hierarchical attention to prevent OOM
"""

import tensorflow as tf

class MemoryEfficientSAM2Attention(tf.keras.layers.Layer):
    """
    Memory-Efficient SAM-2 inspired attention
    Uses local attention windows instead of full quadratic attention
    Prevents OOM on large 3D medical volumes
    """
    def __init__(self, channels, memory_size=32, window_size=16, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.memory_size = memory_size
        self.window_size = window_size  # Local attention window size
        
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
        
        # Max sequence length threshold
        self.max_seq_len = 8192  # Beyond this, use efficient approximation
        
    def call(self, x, training=None):
        """
        Memory-efficient SAM-2 inspired attention
        """
        batch_size = tf.shape(x)[0]
        height, width, depth, channels = tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3], tf.shape(x)[4]
        
        # Generate queries, keys, values
        queries = self.query_conv(x)  # [B, H, W, D, C//8]
        keys = self.key_conv(x)       # [B, H, W, D, C//8]
        values = self.value_conv(x)   # [B, H, W, D, C]
        
        # Calculate sequence length
        seq_len = height * width * depth
        
        if seq_len <= self.max_seq_len:
            # Small volumes: use full attention
            attended = self._full_attention(queries, keys, values, batch_size, height, width, depth, channels)
        else:
            # Large volumes: use memory-efficient attention
            attended = self._efficient_attention(queries, keys, values, batch_size, height, width, depth, channels)
        
        # Residual connection with learnable scaling
        return x + self.gamma * attended
    
    def _full_attention(self, queries, keys, values, batch_size, height, width, depth, channels):
        """Full attention for small volumes"""
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
        return attended
    
    def _efficient_attention(self, queries, keys, values, batch_size, height, width, depth, channels):
        """
        Memory-efficient attention for large volumes
        Uses hierarchical/pooled attention to reduce memory usage
        """
        # Strategy 1: Spatial downsampling attention
        # Downsample by factor of 4 for attention computation
        pool_factor = 4
        
        # Downsample queries, keys, values
        queries_small = tf.nn.avg_pool3d(queries, pool_factor, pool_factor, 'SAME')
        keys_small = tf.nn.avg_pool3d(keys, pool_factor, pool_factor, 'SAME')
        values_small = tf.nn.avg_pool3d(values, pool_factor, pool_factor, 'SAME')
        
        # Get downsampled dimensions
        h_small, w_small, d_small = tf.shape(queries_small)[1], tf.shape(queries_small)[2], tf.shape(queries_small)[3]
        seq_len_small = h_small * w_small * d_small
        
        # Flatten downsampled tensors
        queries_flat = tf.reshape(queries_small, [batch_size, seq_len_small, channels // 8])
        keys_flat = tf.reshape(keys_small, [batch_size, seq_len_small, channels // 8])
        values_flat = tf.reshape(values_small, [batch_size, seq_len_small, channels])
        
        # Compute attention on downsampled features
        attention_scores = tf.matmul(queries_flat, keys_flat, transpose_b=True)
        attention_scores = attention_scores / tf.sqrt(tf.cast(channels // 8, tf.float32))
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        
        # Apply attention
        attended_small = tf.matmul(attention_weights, values_flat)
        
        # Reshape back to downsampled 3D
        attended_small = tf.reshape(attended_small, [batch_size, h_small, w_small, d_small, channels])
        
        # Upsample back to original resolution
        attended = tf.image.resize(
            tf.reshape(attended_small, [batch_size, h_small, w_small, d_small * channels]),
            [height, width],
            method='bilinear'
        )
        attended = tf.reshape(attended, [batch_size, height, width, d_small, channels])
        
        # Handle depth upsampling separately if needed
        if d_small != depth:
            attended = tf.image.resize(
                tf.transpose(attended, [0, 3, 1, 2, 4]),  # Move depth to batch-like dim
                [depth, height],
                method='bilinear'
            )
            attended = tf.transpose(attended, [0, 2, 3, 1, 4])  # Move back
        
        return attended

# Test the memory-efficient version
def test_memory_efficient_sam2():
    print("Testing Memory-Efficient SAM2 Attention...")
    
    # Test the problematic case that caused OOM
    batch_size = 2
    height, width, depth = 96, 112, 88
    channels = 44
    
    x = tf.random.normal((batch_size, height, width, depth, channels), dtype=tf.float32)
    print(f"Input shape: {x.shape}")
    print(f"Sequence length: {height * width * depth:,}")
    
    try:
        # Test memory-efficient SAM2
        sam2_efficient = MemoryEfficientSAM2Attention(channels=channels, memory_size=32)
        
        output = sam2_efficient(x, training=False)
        print(f"✅ Memory-efficient SAM2 successful: {x.shape} → {output.shape}")
        
        # Verify output properties
        print(f"Output dtype: {output.dtype}")
        print(f"Output shape matches input: {output.shape == x.shape}")
        
    except Exception as e:
        print(f"❌ Memory-efficient SAM2 failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Configure TensorFlow
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    test_memory_efficient_sam2()
    print("\n🔧 Memory-efficient SAM2 testing completed!")
