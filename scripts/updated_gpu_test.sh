#!/bin/bash
#SBATCH --job-name=gpu_test_batch
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a4500:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/gpu_test_%j.out
#SBATCH --error=logs/gpu_test_%j.err

echo "🧪 GPU Test on Batch Partition (A4500)"
echo "====================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node: $SLURMD_NODENAME"
echo "GPU allocation: $SLURM_GPUS"
echo "GPU devices: $SLURM_GPUS_ON_NODE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"

# Navigate to project directory
cd /mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota || {
    echo "❌ Could not find project directory"
    exit 1
}

# Check GPU hardware
echo -e "\n🖥️  GPU Hardware Check:"
nvidia-smi
echo ""
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv

# Load CUDA modules using spack system
echo -e "\n🔗 Loading CUDA modules:"
module purge 2>/dev/null

# Try Spack CUDA module based on discovered paths
if module load cuda/12.6.3-gcc-9.3.0 2>/dev/null; then
    echo "✅ Loaded Spack CUDA 12.6.3"
elif module load cuda/12.0 2>/dev/null; then
    echo "✅ Loaded CUDA 12.0"
elif module load cuda 2>/dev/null; then
    echo "✅ Loaded default CUDA"
else
    echo "⚠️  No CUDA module, using system paths"
fi

module list 2>&1

# Set environment variables for A4500
echo -e "\n⚙️  Environment Setup for A4500:"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2

# Use discovered CUDA path
export CUDA_ROOT="/opt/ohpc/pub/spack/opt/spack/linux-rocky8-skylake_avx512/gcc-9.3.0/cuda-12.6.3-ziu74ka3i2glo6q2zt3dy76cnbydqwf4"
export PATH="$CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:$LD_LIBRARY_PATH"

echo "CUDA_ROOT: $CUDA_ROOT"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Activate conda environment
echo -e "\n🐍 Activating conda environment:"
source /opt/anaconda3/etc/profile.d/conda.sh

# Try both environments
if conda activate stroke_sota 2>/dev/null; then
    echo "✅ Activated stroke_sota environment"
elif conda activate tf215_env 2>/dev/null; then
    echo "✅ Activated tf215_env environment"
else
    echo "❌ Could not activate environment"
    exit 1
fi

echo "Active environment: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"

# Test GPU detection with TensorFlow
echo -e "\n🔬 TensorFlow GPU Detection Test:"
python << 'EOF'
import os
import tensorflow as tf
import numpy as np

print(f"🔍 System Info:")
print(f"  Node: {os.uname().nodename}")
print(f"  Job ID: {os.environ.get('SLURM_JOB_ID')}")
print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

print(f"\n📦 TensorFlow Info:")
print(f"  Version: {tf.__version__}")
print(f"  Built with CUDA: {tf.test.is_built_with_cuda()}")

# Check build info
build_info = tf.sysconfig.get_build_info()
print(f"  CUDA Version: {build_info.get('cuda_version', 'unknown')}")
print(f"  cuDNN Version: {build_info.get('cudnn_version', 'unknown')}")

print(f"\n🖥️  Device Detection:")
gpus = tf.config.list_physical_devices('GPU')
print(f"  GPUs detected: {len(gpus)}")

if gpus:
    print("  ✅ SUCCESS: A4500 GPU detected!")
    for i, gpu in enumerate(gpus):
        print(f"    GPU {i}: {gpu}")
    
    # Configure memory growth
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("  ✅ Memory growth enabled")
    except Exception as e:
        print(f"  ⚠️  Memory config warning: {e}")
    
    # Test computation
    print(f"\n🧮 GPU Computation Tests:")
    try:
        with tf.device('/GPU:0'):
            # Basic computation
            a = tf.random.normal((1000, 1000))
            b = tf.random.normal((1000, 1000))
            c = tf.matmul(a, b)
            result = tf.reduce_mean(c)
            print(f"  Matrix ops result: {result.numpy():.4f}")
            
            # 3D convolution test (relevant for our model)
            inputs = tf.random.normal((2, 64, 64, 64, 1))
            conv3d = tf.keras.layers.Conv3D(16, 3, padding='same')
            output = conv3d(inputs)
            print(f"  3D Conv test: {output.shape}")
            
            # Memory test - see how much we can allocate
            try:
                large_tensor = tf.random.normal((10, 128, 128, 128, 32))
                print(f"  Large tensor test: {large_tensor.shape} ✅")
                del large_tensor  # Free memory
            except Exception as e:
                print(f"  Large tensor failed: {e}")
            
            print(f"  ✅ All GPU tests passed!")
        
    except Exception as e:
        print(f"  ❌ GPU computation failed: {e}")
        exit(1)
        
    # Test model creation
    print(f"\n🏗️  Model Creation Test:")
    try:
        # Create a simplified version of our model
        inputs = tf.keras.Input(shape=(32, 32, 32, 1))
        x = tf.keras.layers.Conv3D(32, 3, padding='same', activation='relu')(inputs)
        x = tf.keras.layers.MaxPooling3D(2)(x)
        x = tf.keras.layers.Conv3D(64, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.GlobalAveragePooling3D()(x)
        outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        # Test inference
        test_input = tf.random.normal((1, 32, 32, 32, 1))
        prediction = model(test_input)
        
        print(f"  Model parameters: {model.count_params():,}")
        print(f"  Inference shape: {prediction.shape}")
        print(f"  ✅ Model creation successful!")
        
    except Exception as e:
        print(f"  ❌ Model test failed: {e}")
        exit(1)
        
else:
    print("  ❌ CRITICAL: No GPUs detected!")
    print("  This should not happen on A4500 nodes")
    exit(1)

print(f"\n🎉 ALL TESTS PASSED!")
print(f"A4500 GPU is ready for stroke lesion segmentation training!")
EOF

exit_code=$?

echo -e "\n📊 Final Results:"
if [ $exit_code -eq 0 ]; then
    echo "✅ GPU test SUCCESSFUL on A4500!"
    echo "🚀 Ready to submit full training job"
    echo ""
    echo "Next steps:"
    echo "1. Submit training: sbatch -p batch scripts/slurm_training_batch.sh"
    echo "2. Or try interactive: srun -p batch --gres=gpu:a4500:1 --time=02:00:00 --pty bash"
else
    echo "❌ GPU test failed"
fi

echo -e "\n🏁 Test completed at $(date)"
