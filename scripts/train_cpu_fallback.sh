#!/bin/bash
#SBATCH --job-name=stroke_cpu_train
#SBATCH --output=/mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota/logs/train_cpu_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=384G
#SBATCH --time=48:00:00
#SBATCH --partition=cpu

cd /mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota
eval "$(conda shell.bash hook)"
conda activate stroke_sota

# Force CPU-only mode
export CUDA_VISIBLE_DEVICES=""

python training/train.py \
    --data-dir /mnt/beegfs/hellgate/home/rb194958e/Atlas_2 \
    --output-dir outputs/cpu_run_$(date +%Y%m%d_%H%M%S) \
    --auto-config \
    --gpu-memory 0 \
    --use-tf-data \
    2>&1
