#!/bin/bash
#SBATCH --job-name=stroke_sota_train
#SBATCH --output=/mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota/logs/train_%j.log
#SBATCH --error=/mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota/logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a4500:1
#SBATCH --partition=batch
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rb194958e@umt.edu

echo "Starting Stroke Segmentation Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"

cd /mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota

module purge
module load gcc/9.3.0-5wu3
module load cuda/12.6.3-ziu7

eval "$(conda shell.bash hook)"
conda activate stroke_sota

# Use single GPU - this worked before!
export CUDA_VISIBLE_DEVICES=0
export TF_XLA_FLAGS="--tf_xla_enable_xla_devices=false"
export TF_FORCE_GPU_ALLOW_GROWTH=true

echo "Testing GPU detection..."
python -c "import tensorflow as tf; print(f'GPUs: {len(tf.config.list_physical_devices(\"GPU\"))}')"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/run_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR

python training/train.py \
    --data-dir /mnt/beegfs/hellgate/home/rb194958e/Atlas_2 \
    --output-dir $OUTPUT_DIR \
    --auto-config \
    --gpu-memory 24 \
    --deep-supervision \
    --use-tf-data

echo "Job completed at: $(date)"
