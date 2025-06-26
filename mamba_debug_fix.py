#!/usr/bin/env python3
"""
Vision Mamba Debug Fix - Test the specific dimension issue
"""

import os
import tensorflow as tf
import numpy as np

# Simple test to reproduce and fix the Mamba dimension issue
def test_mamba_dimensions():
    print("Testing Vision Mamba dimension issue...")
    
    # Simulate the exact failing scenario
    batch_size = 2
    height, width, depth = 48, 56, 44
    channels = 88  # This is the failing case
    
    # Create test input (same shape as your failing case)
    x = tf.random.normal((batch_size, height, width, depth, channels), dtype=tf.float16)
    print(f"Input shape: {x.shape}")
    
    # Parameters from your VisionMambaBlock
    d_state = 16
    expand = 2
    d_inner = int(expand * channels)  # 176
    d_conv = 4
    
    print(f"Calculated d_inner: {d_inner}")
    print(f"d_state + d_inner: {d_state + d_inner}")
    
    # Reshape to sequence format (this part works)
    x_flat = tf.reshape(x, [batch_size, height*width*depth, channels])
    print(f"Flattened shape: {x_flat.shape}")
    
    # The problematic projections
    print("\n--- ORIGINAL BUGGY VERSION ---")
    try:
        # This is your original buggy implementation
        x_proj_layer = tf.keras.layers.Dense(d_state + d_inner, use_bias=False)  # 192
        dt_proj_layer = tf.keras.layers.Dense(d_inner, use_bias=False)           # 176
        
        x_proj = x_proj_layer(x_flat)
        dt = dt_proj_layer(x_flat)
        
        print(f"x_proj shape: {x_proj.shape}")
        print(f"dt shape: {dt.shape}")
        
        # This will fail
        result = x_proj * dt
        print("❌ This should have failed!")
        
    except Exception as e:
        print(f"✅ Expected error: {e}")
    
    print("\n--- FIXED VERSION ---")
    try:
        # FIXED: Make dimensions compatible
        # Option 1: Both project to same dimension
        x_proj_layer_fixed = tf.keras.layers.Dense(d_inner, use_bias=False)  # 176
        dt_proj_layer_fixed = tf.keras.layers.Dense(d_inner, use_bias=False) # 176
        
        x_proj_fixed = x_proj_layer_fixed(x_flat)
        dt_fixed = dt_proj_layer_fixed(x_flat)
        
        print(f"x_proj_fixed shape: {x_proj_fixed.shape}")
        print(f"dt_fixed shape: {dt_fixed.shape}")
        
        # This should work
        result_fixed = x_proj_fixed * dt_fixed
        print(f"✅ Fixed result shape: {result_fixed.shape}")
        
        # Test the cumsum operation
        state = tf.cumsum(result_fixed, axis=1)
        print(f"✅ Cumsum result shape: {state.shape}")
        
    except Exception as e:
        print(f"❌ Unexpected error in fix: {e}")
    
    print("\n--- ALTERNATIVE FIX (Split x_proj) ---")
    try:
        # Option 2: Split x_proj into separate state and inner components
        x_proj_layer_alt = tf.keras.layers.Dense(d_state + d_inner, use_bias=False)  # 192
        dt_proj_layer_alt = tf.keras.layers.Dense(d_inner, use_bias=False)           # 176
        
        x_proj_full = x_proj_layer_alt(x_flat)
        dt_alt = dt_proj_layer_alt(x_flat)
        
        # Split x_proj into state and inner parts
        x_state = x_proj_full[:, :, :d_state]      # [?, 118272, 16]
        x_inner = x_proj_full[:, :, d_state:]      # [?, 118272, 176]
        
        print(f"x_state shape: {x_state.shape}")
        print(f"x_inner shape: {x_inner.shape}")
        print(f"dt_alt shape: {dt_alt.shape}")
        
        # Now these are compatible
        result_alt = x_inner * dt_alt
        print(f"✅ Alternative fix result shape: {result_alt.shape}")
        
        # Test the cumsum operation
        state_alt = tf.cumsum(result_alt, axis=1)
        print(f"✅ Alternative cumsum result shape: {state_alt.shape}")
        
    except Exception as e:
        print(f"❌ Unexpected error in alternative fix: {e}")

if __name__ == "__main__":
    # Configure TensorFlow
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    test_mamba_dimensions()
    print("\n🔧 Dimension debugging completed!")
