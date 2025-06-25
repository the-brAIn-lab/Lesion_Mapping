#!/bin/bash
#SBATCH --job-name=training_vs_test
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a4500:1
#SBATCH --time=1:00:00
#SBATCH --output=logs/training_vs_test_%j.out
#SBATCH --error=logs/training_vs_test_%j.err

echo "Testing Model on Training vs Test Cases"
echo "======================================="
echo "This will determine if the issue is:"
echo "1. Overfitting (good on training, bad on test)"
echo "2. Alignment (bad on both until fixed)"
echo "3. Model architecture (bad on both even when fixed)"
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"

cd /mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota

# Setup environment
module load gcc/9.3.0-5wu3 cuda/12.6.3-ziu7
eval "$(conda shell.bash hook)" || true
conda activate tf215_env
export LD_LIBRARY_PATH="/mnt/beegfs/hellgate/home/rb194958e/.conda/envs/tf215_env/lib:$LD_LIBRARY_PATH"
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0
export TF_GPU_ALLOCATOR=cuda_malloc_async

echo "Environment configured"
echo "Running training vs test comparison..."

# Run test
python -u test_training_vs_test_case.py

exit_code=$?

echo ""
echo "Testing completed with exit code: $exit_code"
echo "End time: $(date)"

if [ $exit_code -eq 0 ]; then
    echo "✅ COMPARISON TEST SUCCESSFUL"
else
    echo "❌ COMPARISON TEST FAILED"
fi
