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
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rb194958e@umt.edu

echo "=========================================="
echo "Starting Stroke Segmentation Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

# Change to project directory
cd /mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required modules
module purge
module load gcc/9.3.0-5wu3
module load cuda/12.6.3-ziu7

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate stroke_sota

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TF_CPP_MIN_LOG_LEVEL=1
export TF_XLA_FLAGS="--tf_xla_enable_xla_devices=false"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_THREAD_MODE=gpu_private

# Print environment info
echo -e "\n--- Environment Information ---"
python -c "
import tensorflow as tf
import numpy as np
print(f'TensorFlow: {tf.__version__}')
print(f'NumPy: {np.__version__}')
print(f'GPUs Available: {len(tf.config.list_physical_devices(\"GPU\"))}')
for i, gpu in enumerate(tf.config.list_physical_devices('GPU')):
    print(f'  GPU {i}: {gpu.name}')
"

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/run_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR

echo -e "\n--- Starting Training ---"
echo "Output directory: $OUTPUT_DIR"

# Run training with auto-configuration
python training/train.py \
    --data-dir /mnt/beegfs/hellgate/home/rb194958e/Atlas_2 \
    --output-dir $OUTPUT_DIR \
    --auto-config \
    --gpu-memory 24 \
    --deep-supervision \
    --use-tf-data \
    2>&1 | tee $OUTPUT_DIR/training.log

# Check exit status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo -e "\n✅ Training completed successfully!"
    echo "Models saved to: $OUTPUT_DIR"
else
    echo -e "\n❌ Training failed with exit code: ${PIPESTATUS[0]}"
    exit 1
fi

echo -e "\n=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
