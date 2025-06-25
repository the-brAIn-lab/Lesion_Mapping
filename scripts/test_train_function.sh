#!/bin/bash
#SBATCH --job-name=test_train_function
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a4500:1
#SBATCH --time=30:00
#SBATCH --output=../logs/test_train_function_%j.out
#SBATCH --error=../logs/test_train_function_%j.err

cd /mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota

# Setup environment
module load gcc/9.3.0-5wu3 cuda/12.6.3-ziu7
eval "$(conda shell.bash hook)" || true
conda activate tf215_env
export LD_LIBRARY_PATH="/mnt/beegfs/hellgate/home/rb194958e/.conda/envs/tf215_env/lib:$LD_LIBRARY_PATH"
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0
export TF_GPU_ALLOCATOR=cuda_malloc_async
export NIBABEL_NIFTI1_QFAC_CHECK=0

echo "Testing train_model function specifically..."
python -u test_train_function.py
