#!/bin/bash
#SBATCH --job-name=test_large_lesions
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a4500:1
#SBATCH --time=0:20:00
#SBATCH --output=../logs/test_large_lesions_%j.out
#SBATCH --error=../logs/test_large_lesions_%j.err

echo "🔍 TESTING MODEL ON DIFFERENT LESION SIZES"
echo "=========================================="

cd /mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota

# Setup environment  
module load gcc/9.3.0-5wu3 cuda/12.6.3-ziu7
eval "$(conda shell.bash hook)" || true
conda activate tf215_env
export LD_LIBRARY_PATH="/mnt/beegfs/hellgate/home/rb194958e/.conda/envs/tf215_env/lib:$LD_LIBRARY_PATH"
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0
export TF_GPU_ALLOCATOR=cuda_malloc_async

python -u test_large_lesion_case.py
