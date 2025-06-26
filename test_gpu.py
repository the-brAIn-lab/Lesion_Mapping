import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

import tensorflow as tf
import numpy as np

print(f"TensorFlow version: {tf.__version__}")
print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")

# Enable memory growth
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
if gpus:
    print("\nGPU Details:")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")
else:
    print("No GPUs found!")

# Test model building
print("\nTesting model import...")
from models.attention_unet import build_attention_unet
model = build_attention_unet(input_shape=(192, 224, 176, 1))
print(f"Model built successfully! Parameters: {model.count_params():,}")
