#!/bin/bash
#SBATCH --job-name=sota_corrected
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a4500:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/sota_corrected_%j.out
#SBATCH --error=logs/sota_corrected_%j.err

echo "=================================================="
echo "CORRECTED SOTA TRAINING"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"

cd /mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota

# Load modules
module load gcc/9.3.0-5wu3 cuda/12.6.3-ziu7

# Activate environment
eval "$(conda shell.bash hook)" || true
conda activate tf215_env

# Set environment
export LD_LIBRARY_PATH="/mnt/beegfs/hellgate/home/rb194958e/.conda/envs/tf215_env/lib:$LD_LIBRARY_PATH"
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0

# Run training
echo "Starting training..."
python corrected_sota_training.py

echo "Completed: $(date)"
