#!/usr/bin/env python3
"""
Systematic Diagnosis Script for Smart SOTA 2025 Issues
Tests each component individually to identify the root cause
"""

import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import tensorflow as tf
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def test_basic_setup():
    """Test 1: Basic TensorFlow and GPU setup"""
    logger.info("🔧 TEST 1: Basic Setup")
    
    # Check TensorFlow
    logger.info(f"TensorFlow version: {tf.__version__}")
    
    # Check GPUs
    gpus = tf.config.list_physical_devices('GPU')
    logger.info(f"GPUs detected: {len(gpus)}")
    
    # Configure GPU memory
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    # Test basic tensor operations
    try:
        a = tf.random.normal((100, 100))
        b = tf.matmul(a, a)
        logger.info(f"✅ Basic operations work: {b.shape}")
        return True
    except Exception as e:
        logger.error(f"❌ Basic operations failed: {e}")
        return False

def test_mixed_precision():
    """Test 2: Mixed precision stability"""
    logger.info("🔧 TEST 2: Mixed Precision")
    
    # Test without mixed precision
    tf.keras.mixed_precision.set_global_policy('float32')
    
    try:
        x = tf.random.normal((10, 32, 32, 32, 16))
        conv = tf.keras.layers.Conv3D(32, 3, padding='same')
        y = conv(x)
        
        # Test loss computation
        target = tf.random.uniform((10, 32, 32, 32, 32))
        loss = tf.reduce_mean(tf.square(y - target))
        
        logger.info(f"✅ Float32 operations stable: loss={loss:.6f}")
        
        # Now test mixed precision
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        y_fp16 = conv(tf.cast(x, tf.float16))
        loss_fp16 = tf.reduce_mean(tf.square(tf.cast(y_fp16, tf.float32) - target))
        
        if tf.math.is_nan(loss_fp16):
            logger.error("❌ Mixed precision causes NaN")
            return False
        else:
            logger.info(f"✅ Mixed precision stable: loss={loss_fp16:.6f}")
            return True
            
    except Exception as e:
        logger.error(f"❌ Mixed precision test failed: {e}")
        return False

def test_boundary_loss():
    """Test 3: Boundary-aware loss function"""
    logger.info("🔧 TEST 3: Boundary Loss Function")
    
    # Create simple test data
    y_true = tf.random.uniform((2, 32, 32, 32, 1), 0, 1) > 0.5
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.random.uniform((2, 32, 32, 32, 1), 0.1, 0.9)
    
    # Test the boundary loss function from your script
    def test_boundary_loss_function(y_true, y_pred, smooth=1e-6):
        try:
            y_true = tf.cast(y_true, y_pred.dtype)
            smooth = tf.cast(smooth, y_pred.dtype)
            
            # 3D gradient operations
            grad_x = tf.abs(y_true[:, 1:, :, :, :] - y_true[:, :-1, :, :, :])
            grad_y = tf.abs(y_true[:, :, 1:, :, :] - y_true[:, :, :-1, :, :])
            grad_z = tf.abs(y_true[:, :, :, 1:, :] - y_true[:, :, :, :-1, :])
            
            # Pad gradients
            grad_x = tf.pad(grad_x, [[0, 0], [0, 1], [0, 0], [0, 0], [0, 0]], mode='CONSTANT')
            grad_y = tf.pad(grad_y, [[0, 0], [0, 0], [0, 1], [0, 0], [0, 0]], mode='CONSTANT')
            grad_z = tf.pad(grad_z, [[0, 0], [0, 0], [0, 0], [0, 1], [0, 0]], mode='CONSTANT')
            
            # Combine gradients
            boundaries = tf.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + smooth)
            boundary_weight = 1.0 + 3.0 * boundaries
            
            # Binary cross-entropy with boundary weighting
            bce = -(y_true * tf.math.log(y_pred + smooth) + 
                   (1 - y_true) * tf.math.log(1 - y_pred + smooth))
            weighted_bce = bce * boundary_weight
            
            result = tf.reduce_mean(weighted_bce)
            return result
            
        except Exception as e:
            logger.error(f"Boundary loss computation failed: {e}")
            return tf.constant(float('nan'))
    
    loss = test_boundary_loss_function(y_true, y_pred)
    
    if tf.math.is_nan(loss):
        logger.error("❌ Boundary loss produces NaN")
        return False
    else:
        logger.info(f"✅ Boundary loss stable: {loss:.6f}")
        return True

def test_vision_mamba():
    """Test 4: Vision Mamba block"""
    logger.info("🔧 TEST 4: Vision Mamba Block")
    
    try:
        # Simplified Vision Mamba test
        class SimpleVisionMamba(tf.keras.layers.Layer):
            def __init__(self, dim, **kwargs):
                super().__init__(**kwargs)
                self.dim = dim
                self.in_proj = tf.keras.layers.Dense(dim * 2, use_bias=False)
                self.out_proj = tf.keras.layers.Dense(dim, use_bias=False)
                self.norm = tf.keras.layers.LayerNormalization()
                
            def call(self, x):
                B, H, W, D, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3], tf.shape(x)[4]
                x_flat = tf.reshape(x, [B, H*W*D, C])
                
                # Simple projection without complex SSM
                projected = self.in_proj(x_flat)
                x_out, residual = tf.split(projected, 2, axis=-1)
                
                # Simple cumulative sum instead of complex SSM
                state = tf.cumsum(x_out, axis=1)
                
                output = self.out_proj(state)
                output = tf.reshape(output, [B, H, W, D, C])
                
                return self.norm(output + x)
        
        # Test the simplified Mamba
        x = tf.random.normal((2, 16, 16, 16, 64))
        mamba = SimpleVisionMamba(64)
        
        y = mamba(x)
        
        if tf.math.reduce_any(tf.math.is_nan(y)):
            logger.error("❌ Vision Mamba produces NaN")
            return False
        else:
            logger.info(f"✅ Vision Mamba stable: {y.shape}")
            return True
            
    except Exception as e:
        logger.error(f"❌ Vision Mamba test failed: {e}")
        return False

def test_sam2_attention():
    """Test 5: SAM2 attention"""
    logger.info("🔧 TEST 5: SAM2 Attention")
    
    try:
        # Simplified attention test
        x = tf.random.normal((2, 16, 16, 16, 64))
        
        # Simple attention without complex pooling
        queries = tf.keras.layers.Conv3D(8, 1)(x)
        keys = tf.keras.layers.Conv3D(8, 1)(x)
        values = tf.keras.layers.Conv3D(64, 1)(x)
        
        # Reshape for attention
        B, H, W, D, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3], tf.shape(x)[4]
        
        q_flat = tf.reshape(queries, [B, -1, 8])
        k_flat = tf.reshape(keys, [B, -1, 8])
        v_flat = tf.reshape(values, [B, -1, 64])
        
        # Attention computation
        scores = tf.matmul(q_flat, k_flat, transpose_b=True)
        scores = scores / tf.sqrt(tf.cast(8, scores.dtype))
        weights = tf.nn.softmax(scores, axis=-1)
        
        attended = tf.matmul(weights, v_flat)
        attended = tf.reshape(attended, [B, H, W, D, 64])
        
        output = x + 0.1 * attended
        
        if tf.math.reduce_any(tf.math.is_nan(output)):
            logger.error("❌ SAM2 attention produces NaN")
            return False
        else:
            logger.info(f"✅ SAM2 attention stable: {output.shape}")
            return True
            
    except Exception as e:
        logger.error(f"❌ SAM2 attention test failed: {e}")
        return False

def test_learning_rate_schedule():
    """Test 6: Learning rate schedule"""
    logger.info("🔧 TEST 6: Learning Rate Schedule")
    
    try:
        def cosine_schedule_with_warmup(epoch, warmup_epochs=5, total_epochs=60, initial_lr=8e-5, min_lr=1e-7):
            if epoch < warmup_epochs and warmup_epochs > 0:
                return initial_lr * max(epoch, 1) / max(warmup_epochs, 1)
            else:
                effective_epoch = max(0, epoch - warmup_epochs)
                decay_epochs = max(1, total_epochs - warmup_epochs)
                progress = min(1.0, effective_epoch / decay_epochs)
                decay_factor = 0.5 * (1 + np.cos(np.pi * progress * 0.8))
                return min_lr + (initial_lr - min_lr) * max(0.1, decay_factor)
        
        # Test learning rates for different epochs
        lrs = []
        for epoch in range(40):
            lr = cosine_schedule_with_warmup(epoch)
            lrs.append(lr)
            
            if lr <= 0 or np.isnan(lr) or np.isinf(lr):
                logger.error(f"❌ Invalid learning rate at epoch {epoch}: {lr}")
                return False
        
        logger.info(f"✅ Learning rate schedule stable: {lrs[0]:.2e} -> {lrs[-1]:.2e}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Learning rate schedule test failed: {e}")
        return False

def test_simple_baseline():
    """Test 7: Simple baseline model"""
    logger.info("🔧 TEST 7: Simple Baseline Model")
    
    try:
        # Create very simple U-Net
        inputs = tf.keras.layers.Input(shape=(32, 32, 32, 1))
        
        # Encoder
        x1 = tf.keras.layers.Conv3D(16, 3, padding='same', activation='relu')(inputs)
        x1 = tf.keras.layers.BatchNormalization()(x1)
        p1 = tf.keras.layers.MaxPooling3D(2)(x1)
        
        x2 = tf.keras.layers.Conv3D(32, 3, padding='same', activation='relu')(p1)
        x2 = tf.keras.layers.BatchNormalization()(x2)
        p2 = tf.keras.layers.MaxPooling3D(2)(x2)
        
        # Bottleneck
        b = tf.keras.layers.Conv3D(64, 3, padding='same', activation='relu')(p2)
        b = tf.keras.layers.BatchNormalization()(b)
        
        # Decoder
        u1 = tf.keras.layers.Conv3DTranspose(32, 2, strides=2, padding='same')(b)
        u1 = tf.keras.layers.Concatenate()([u1, x2])
        u1 = tf.keras.layers.Conv3D(32, 3, padding='same', activation='relu')(u1)
        
        u2 = tf.keras.layers.Conv3DTranspose(16, 2, strides=2, padding='same')(u1)
        u2 = tf.keras.layers.Concatenate()([u2, x1])
        u2 = tf.keras.layers.Conv3D(16, 3, padding='same', activation='relu')(u2)
        
        outputs = tf.keras.layers.Conv3D(1, 1, activation='sigmoid')(u2)
        
        model = tf.keras.Model(inputs, outputs)
        
        # Test forward pass
        x = tf.random.normal((2, 32, 32, 32, 1))
        y = model(x)
        
        # Test loss computation
        target = tf.random.uniform((2, 32, 32, 32, 1), 0, 1) > 0.5
        target = tf.cast(target, tf.float32)
        
        # Simple dice loss
        def simple_dice_loss(y_true, y_pred):
            y_true_f = tf.keras.backend.flatten(y_true)
            y_pred_f = tf.keras.backend.flatten(y_pred)
            intersection = tf.reduce_sum(y_true_f * y_pred_f)
            return 1 - (2. * intersection + 1e-6) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1e-6)
        
        loss = simple_dice_loss(target, y)
        
        if tf.math.is_nan(loss):
            logger.error("❌ Simple baseline produces NaN")
            return False
        else:
            logger.info(f"✅ Simple baseline stable: loss={loss:.6f}")
            return True
            
    except Exception as e:
        logger.error(f"❌ Simple baseline test failed: {e}")
        return False

def run_all_tests():
    """Run all diagnostic tests"""
    logger.info("🚀 STARTING SYSTEMATIC DIAGNOSIS")
    logger.info("=" * 60)
    
    tests = [
        ("Basic Setup", test_basic_setup),
        ("Mixed Precision", test_mixed_precision),
        ("Boundary Loss", test_boundary_loss),
        ("Vision Mamba", test_vision_mamba),
        ("SAM2 Attention", test_sam2_attention),
        ("Learning Rate Schedule", test_learning_rate_schedule),
        ("Simple Baseline", test_simple_baseline)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("🏁 DIAGNOSIS SUMMARY")
    logger.info("="*60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name:20} : {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    logger.info(f"\nTotal: {passed} passed, {failed} failed")
    
    # Recommendations
    logger.info("\n🎯 RECOMMENDATIONS:")
    
    if not results.get("Mixed Precision", True):
        logger.info("1. 🔧 DISABLE mixed precision - use float32")
    
    if not results.get("Boundary Loss", True):
        logger.info("2. 🔧 REPLACE boundary loss with simple dice loss")
    
    if not results.get("Vision Mamba", True):
        logger.info("3. 🔧 DISABLE Vision Mamba blocks")
    
    if not results.get("SAM2 Attention", True):
        logger.info("4. 🔧 DISABLE SAM2 attention")
    
    if not results.get("Learning Rate Schedule", True):
        logger.info("5. 🔧 USE constant learning rate")
    
    if results.get("Simple Baseline", True) and failed > 0:
        logger.info("6. 🎯 START with simple baseline and add features incrementally")
    
    logger.info("\n🚀 Next step: Create simplified model based on failing components")
    
    return results

if __name__ == "__main__":
    os.makedirs('logs', exist_ok=True)
    run_all_tests()
