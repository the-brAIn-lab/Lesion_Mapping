#!/bin/bash
#SBATCH --job-name=stroke_sota_train
#SBATCH --output=/mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota/logs/train_%j.log
#SBATCH --error=/mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota/logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=384G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a4500:4
#SBATCH --partition=batch
#SBATCH --exclude=hggpu9-12
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rb194958e@umt.edu

echo "=========================================="
echo "Starting Stroke Segmentation Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

cd /mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota
mkdir -p logs

# Load modules
module purge
module load gcc/9.3.0-5wu3
module load cuda/12.6.3-ziu7

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate stroke_sota

# Fix library paths for TensorFlow GPU support
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Set CUDA environment
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Disable XLA to avoid conflicts
export TF_XLA_FLAGS="--tf_xla_enable_xla_devices=false"
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Debug info
echo -e "\n--- Library Paths ---"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "CUDA_HOME: $CUDA_HOME"
echo "CONDA_PREFIX: $CONDA_PREFIX"

echo -e "\n--- GPU Detection ---"
nvidia-smi -L

# Test GPU detection with minimal script
echo -e "\n--- Testing TensorFlow GPU Detection ---"
python -c "
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Show all logs for debugging
import tensorflow as tf
print(f'TensorFlow: {tf.__version__}')
print('Physical devices:')
devices = tf.config.list_physical_devices()
for device in devices:
    print(f'  {device}')
print(f'\\nGPUs found: {len(tf.config.list_physical_devices(\"GPU\"))}')
"

# If GPUs still not detected, try alternative approach
if python -c "import tensorflow as tf; exit(0 if len(tf.config.list_physical_devices('GPU')) > 0 else 1)"; then
    echo "✅ GPUs detected successfully!"
else
    echo "⚠️  GPUs not detected by TensorFlow. Trying alternative configuration..."
    
    # Try loading cudnn separately
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
fi

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/run_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR

echo -e "\n--- Starting Training ---"
echo "Output directory: $OUTPUT_DIR"

# Run training
python training/train.py \
    --data-dir /mnt/beegfs/hellgate/home/rb194958e/Atlas_2 \
    --output-dir $OUTPUT_DIR \
    --auto-config \
    --gpu-memory 24 \
    --deep-supervision \
    --use-tf-data \
    2>&1 | tee $OUTPUT_DIR/training.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo -e "\n✅ Training completed successfully!"
else
    echo -e "\n❌ Training failed with exit code: ${PIPESTATUS[0]}"
    exit 1
fi

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
