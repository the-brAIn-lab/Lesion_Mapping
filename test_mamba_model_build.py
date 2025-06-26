import os
import sys
import tensorflow as tf

# Add current directory to Python path
sys.path.append('/mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota')

# Import your fixed Smart SOTA model
try:
    from smart_sota_2025 import build_smart_sota_2025_model
    print("✅ Successfully imported Smart SOTA 2025 model")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Configure GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Test model building
try:
    print("Building Smart SOTA 2025 model...")
    model = build_smart_sota_2025_model(
        input_shape=(192, 224, 176, 1),
        base_filters=22
    )
    
    param_count = model.count_params()
    print(f"✅ Model built successfully!")
    print(f"Parameters: {param_count:,}")
    
    # Test a forward pass with dummy data
    print("Testing forward pass...")
    dummy_input = tf.random.normal((1, 192, 224, 176, 1), dtype=tf.float16)
    output = model(dummy_input, training=False)
    print(f"✅ Forward pass successful: {dummy_input.shape} → {output.shape}")
    
    print("🎉 MAMBA FIX VERIFICATION SUCCESSFUL!")
    
except Exception as e:
    print(f"❌ Model building/testing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
