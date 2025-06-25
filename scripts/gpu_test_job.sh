#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --account=your_account
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/gpu_test_%j.out
#SBATCH --error=logs/gpu_test_%j.err

echo "🧪 GPU Test Job on Compute Node"
echo "==============================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU allocation: $SLURM_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"

# Navigate to project directory
cd /mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota

# Load modules
echo -e "\n🔗 Loading CUDA modules:"
module purge
module load cuda/12.0 2>/dev/null || echo "CUDA module not available"
module load cudnn/8.8.0 2>/dev/null || echo "cuDNN module not available"
module list

# Set environment
export CUDA_VISIBLE_DEVICES="0"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# Check hardware
echo -e "\n🖥️  Hardware Information:"
nvidia-smi

# Activate environment
echo -e "\n🐍 Activating environment:"
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate stroke_sota
echo "Active environment: $CONDA_DEFAULT_ENV"

# Test GPU detection
echo -e "\n🔬 Testing GPU Detection on Compute Node:"
python << 'EOF'
import tensorflow as tf
import os

print(f"Node hostname: {os.uname().nodename}")
print(f"TensorFlow version: {tf.__version__}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

# List all devices
print(f"\nAll physical devices:")
for device in tf.config.list_physical_devices():
    print(f"  {device}")

# Check GPUs specifically
gpus = tf.config.list_physical_devices('GPU')
print(f"\nGPU devices detected: {len(gpus)}")

if gpus:
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
    
    # Configure memory growth
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ Memory growth configured")
    except Exception as e:
        print(f"⚠️  Memory growth failed: {e}")
    
    # Test computation
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            print(f"✅ GPU computation successful: {c.numpy()}")
    except Exception as e:
        print(f"❌ GPU computation failed: {e}")
        
    # Test model creation
    try:
        inputs = tf.keras.Input(shape=(10, 10, 10, 1))
        x = tf.keras.layers.Conv3D(16, 3, padding='same')(inputs)
        outputs = tf.keras.layers.GlobalAveragePooling3D()(x)
        model = tf.keras.Model(inputs, outputs)
        
        with tf.device('/GPU:0'):
            test_input = tf.random.normal((1, 10, 10, 10, 1))
            output = model(test_input)
            print(f"✅ GPU model test successful: output shape {output.shape}")
    except Exception as e:
        print(f"❌ GPU model test failed: {e}")
        
else:
    print("❌ No GPUs detected on compute node!")
    print("This indicates a deeper configuration issue.")
    
    # Debug information
    print(f"\nDebug info:")
    print(f"SLURM_GPUS: {os.environ.get('SLURM_GPUS', 'not set')}")
    print(f"SLURM_GPUS_ON_NODE: {os.environ.get('SLURM_GPUS_ON_NODE', 'not set')}")
    print(f"GPU_DEVICE_ORDINAL: {os.environ.get('GPU_DEVICE_ORDINAL', 'not set')}")
EOF

echo -e "\n🏁 GPU test completed at $(date)"

# If this is successful, we can proceed with training
if [ -f "training/robust_train.py" ]; then
    echo -e "\n🎯 Training script found - ready for full training job"
else
    echo -e "\n⚠️  Training script not found"
fi

echo "Job completed successfully!"
