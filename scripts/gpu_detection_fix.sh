#!/bin/bash
# gpu_detection_fix.sh - Comprehensive GPU detection fix for stroke_sota environment

echo "🔧 Fixing GPU Detection for Stroke Segmentation"
echo "=============================================="

# Activate environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate stroke_sota

echo "📋 Current Environment Status:"
echo "Python: $(python --version)"
echo "Conda Env: $CONDA_DEFAULT_ENV"

# Check system CUDA
echo -e "\n🖥️ System CUDA Status:"
nvidia-smi
echo "CUDA Version: $(nvcc --version 2>/dev/null || echo 'nvcc not found')"

# Check current TensorFlow
echo -e "\n📦 Current TensorFlow Status:"
python -c "
import tensorflow as tf
print(f'TensorFlow Version: {tf.__version__}')
print(f'Built with CUDA: {tf.test.is_built_with_cuda()}')
print(f'GPU Support: {tf.test.is_gpu_available()}')
print(f'Physical GPUs: {len(tf.config.list_physical_devices(\"GPU\"))}')
"

# Load CUDA modules if available
echo -e "\n🔗 Loading CUDA Modules:"
module load cuda/12.0 2>/dev/null || echo "CUDA module not available"
module load cudnn/8.8.0 2>/dev/null || echo "cuDNN module not available"

# Set environment variables
echo -e "\n⚙️ Setting Environment Variables:"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export CUDA_ROOT="/usr/local/cuda"
export PATH="/usr/local/cuda/bin:$PATH"

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Reinstall TensorFlow with GPU support
echo -e "\n🔄 Reinstalling TensorFlow with GPU Support:"
pip uninstall tensorflow -y
pip install tensorflow[and-cuda]==2.15.1

# Verify installation
echo -e "\n✅ Verification:"
python -c "
import tensorflow as tf
print(f'TensorFlow Version: {tf.__version__}')
print(f'Built with CUDA: {tf.test.is_built_with_cuda()}')
print(f'CUDA Build Info: {tf.sysconfig.get_build_info()}')

try:
    gpus = tf.config.list_physical_devices('GPU')
    print(f'Physical GPUs: {len(gpus)}')
    for i, gpu in enumerate(gpus):
        print(f'  GPU {i}: {gpu}')
        
    # Test GPU computation
    if len(gpus) > 0:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            print(f'GPU Test Successful: {c.numpy()}')
    else:
        print('No GPUs detected - will use CPU')
        
except Exception as e:
    print(f'GPU Test Failed: {e}')
"

# Test memory configuration
echo -e "\n🧠 Testing GPU Memory Configuration:"
python -c "
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print('Memory growth enabled for all GPUs')
    except RuntimeError as e:
        print(f'Memory configuration error: {e}')
else:
    print('No GPUs available for memory configuration')
"

echo -e "\n🎯 Next Steps:"
echo "1. If GPUs detected: Run training with fixed batch size"
echo "2. If no GPUs: Check CUDA installation and paths"
echo "3. Use the fixed training script with fallback logic"
