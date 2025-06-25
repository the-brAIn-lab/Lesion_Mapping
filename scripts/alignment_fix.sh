#!/bin/bash
#SBATCH --job-name=alignment_fix
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a4500:1
#SBATCH --time=1:00:00
#SBATCH --output=logs/alignment_fix_%j.out
#SBATCH --error=logs/alignment_fix_%j.err

echo "Testing Alignment Fixes for Flip Augmentation Issue"
echo "=================================================="
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

echo ""
echo "Found flip augmentation in training:"
echo "===================================="
grep -A 3 -B 1 "flip" working_sota_training.py

echo ""
echo "Running alignment fix test..."

# Run test
python -u alignment_fix_test.py

exit_code=$?

echo ""
echo "Alignment test completed with exit code: $exit_code"
echo "End time: $(date)"

if [ $exit_code -eq 0 ]; then
    echo "✅ ALIGNMENT TEST SUCCESSFUL"
else
    echo "❌ ALIGNMENT TEST FAILED"
fi
