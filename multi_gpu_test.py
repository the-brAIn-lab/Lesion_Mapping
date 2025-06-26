# multi_gpu_test.py
import tensorflow as tf
import os
import time

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] TensorFlow version: {tf.__version__}")
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Python executable: {os.sys.executable}")

# Check for GPU devices
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Found {len(gpus)} GPUs:")
    for gpu in gpus:
        print(f"  - {gpu}")
        tf.config.experimental.set_memory_growth(gpu, True) # Avoids allocating all VRAM at once
else:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] No GPUs found by TensorFlow.")
    print("This might mean TensorFlow isn't properly configured or GPUs aren't visible.")
    print("Exiting.")
    exit(1)

# --- Test Multi-GPU with MirroredStrategy ---
# This part will only be effective if multiple GPUs are detected.
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Attempting to use tf.distribute.MirroredStrategy...")
strategy = tf.distribute.MirroredStrategy()
num_gpus_in_strategy = strategy.num_replicas_in_sync
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Number of devices in MirroredStrategy: {num_gpus_in_strategy}")

if num_gpus_in_strategy < len(gpus):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Warning: MirroredStrategy detected {num_gpus_in_strategy} devices, but {len(gpus)} physical GPUs were found. Check your environment/setup.")
elif num_gpus_in_strategy == 1:
     print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Only one device detected by MirroredStrategy. Parallel processing won't occur.")


with strategy.scope():
    # Create a simple model within the strategy scope
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, input_shape=(10,), activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Simple model created and compiled within strategy scope.")

    # Create dummy data
    x = tf.random.normal((100, 10))
    y = tf.random.uniform((100, 1), maxval=2, dtype=tf.int32)

    # Perform a dummy training step
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Performing a dummy training step...")
    try:
        model.fit(x, y, epochs=1, batch_size=32, verbose=0)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Dummy training step successful.")
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error during dummy training: {e}")
        print("This might indicate an issue with the multi-GPU setup.")

# Print device placement for a simple operation
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Checking device placement for a tensor addition...")
with tf.device('/GPU:0'): # Explicitly place on one GPU, but strategy will distribute
    a = tf.constant([1.0, 2.0, 3.0])
    b = tf.constant([4.0, 5.0, 6.0])
    c = a + b
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Result of a + b: {c.numpy()}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Tensor 'c' is on device: {c.device}")

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Multi-GPU test script finished.")
