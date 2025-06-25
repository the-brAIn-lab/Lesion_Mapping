#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --output=test_gpu_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --partition=interactive

cd /mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota

# Load modules
module purge
module load gcc/9.3.0-5wu3
module load cuda/12.6.3-ziu7

# Activate environment
eval "$(conda shell.bash hook)"
conda activate stroke_sota

# Test GPU
python test_gpu.py
