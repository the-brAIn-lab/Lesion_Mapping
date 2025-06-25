#!/bin/bash
# advanced_gpu_debug.sh - Comprehensive GPU debugging for Hellgate cluster

echo "🔍 Advanced GPU Detection Debugging"
echo "=================================="

# Check if we're on a login node vs compute node
echo "📍 Node Information:"
echo "Hostname: $(hostname)"
echo "Node type: $(if [[ $(hostname) == *login* ]]; then echo 'LOGIN NODE'; else echo 'COMPUTE NODE'; fi)"

if [[ $(hostname) == *login* ]]; then
    echo "⚠️  You're on a LOGIN NODE - GPUs may not be visible here!"
    echo "GPUs are typically only available on compute nodes."
    echo ""
    echo "🎯 Solutions:"
    echo "1. Submit a job to a GPU node: sbatch scripts/gpu_test_job.sh"
    echo "2. Request interactive GPU session: srun --gres=gpu:1 --pty bash"
    echo "3. Run training via SLURM script"
    echo ""
fi

echo "🖥️  System GPU Information:"
echo "nvidia-smi output:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo ""
    echo "nvidia-smi device listing:"
    nvidia-smi -L
else
    echo "nvidia-smi not found - no NVIDIA drivers or on login node"
fi

echo ""
echo "🔗 CUDA Environment:"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-'not set'}"
echo "CUDA_ROOT: ${CUDA_ROOT:-'not set'}"
echo "PATH (CUDA part): $(echo $PATH | tr ':' '\n' | grep -i cuda || echo 'No CUDA in PATH')"
echo "LD_LIBRARY_PATH (CUDA part): $(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -i cuda || echo 'No CUDA in LD_LIBRARY_PATH')"

echo ""
echo "📦 Available CUDA installations:"
find /usr/local -name "cuda*" -type d 2>/dev/null | head -10
find /opt -name "cuda*" -type d 2>/dev/null | head -10

echo ""
echo "🐍 Python Environment:"
echo "Active conda env: ${CONDA_DEFAULT_ENV:-'none'}"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"

echo ""
echo "🧪 TensorFlow CUDA Detection Test:"
python << 'EOF'
import os
print(f"Environment CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

# Check CUDA libraries
print("\nCUDA Library Paths:")
try:
    from tensorflow.python.platform import build_info
    print(f"CUDA version: {build_info.build_info['cuda_version']}")
    print(f"cuDNN version: {build_info.build_info['cudnn_version']}")
except:
    print("Could not get CUDA build info")

# Physical devices
print(f"\nPhysical devices:")
all_devices = tf.config.list_physical_devices()
for device in all_devices:
    print(f"  {device}")

gpus = tf.config.list_physical_devices('GPU')
print(f"\nGPU devices: {len(gpus)}")
for i, gpu in enumerate(gpus):
    print(f"  GPU {i}: {gpu}")

# More detailed GPU check
print(f"\nGPU available (deprecated): {tf.test.is_gpu_available()}")

# Check for CUDA errors
try:
    with tf.device('/GPU:0'):
        a = tf.constant([1.0, 2.0])
        print(f"GPU computation test: {a}")
except Exception as e:
    print(f"GPU test failed: {e}")

# Check CUDA runtime
try:
    print(f"\nCUDA runtime version: {tf.sysconfig.get_build_info()['cuda_version']}")
except:
    print("Could not get CUDA runtime info")
EOF

echo ""
echo "🔧 Potential Fixes:"
echo "1. If on login node: Submit job to GPU node"
echo "2. Load CUDA modules: module load cuda/12.0"
echo "3. Set environment: export CUDA_VISIBLE_DEVICES=0,1,2,3"
echo "4. Check SLURM GPU allocation: echo \$SLURM_GPUS"
