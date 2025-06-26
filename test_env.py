import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

# Test imports one by one
print("\nTesting imports:")
try:
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
except Exception as e:
    print(f"❌ NumPy: {e}")

try:
    import matplotlib.pyplot as plt
    print(f"✅ Matplotlib: {plt.matplotlib.__version__}")
except Exception as e:
    print(f"❌ Matplotlib: {e}")

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"❌ PyTorch: {e}")

try:
    import tensorflow as tf
    print(f"✅ TensorFlow: {tf.__version__}")
    print(f"   GPUs: {len(tf.config.list_physical_devices('GPU'))}")
except Exception as e:
    print(f"❌ TensorFlow: {e}")
