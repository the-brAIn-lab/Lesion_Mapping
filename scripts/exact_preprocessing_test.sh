#!/bin/bash
#SBATCH --job-name=exact_preprocessing_test
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a4500:1
#SBATCH --time=1:00:00
#SBATCH --output=logs/exact_preprocessing_%j.out
#SBATCH --error=logs/exact_preprocessing_%j.err

echo "Testing Exact Training Preprocessing + Both Orientations"
echo "========================================================"
echo "Expected: Flip augmentation mismatch should be resolved"
echo "  Original orientation: Dice ~0.0 (as before)"
echo "  Flipped orientation: Dice ~0.3-0.5 (major improvement)"
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

echo ""
echo "Environment configured"
echo "Running exact preprocessing test..."

# Run test
python -u exact_training_preprocessing_test.py

exit_code=$?

echo ""
echo "Test completed with exit code: $exit_code"
echo "End time: $(date)"

if [ $exit_code -eq 0 ]; then
    echo "✅ EXACT PREPROCESSING TEST SUCCESSFUL"
    
    # Check if we found a solution
    echo ""
    echo "Key results to look for:"
    echo "- Original orientation Dice score"
    echo "- Flipped orientation Dice score"
    echo "- If flipped orientation Dice > 0.3: PROBLEM SOLVED!"
    
else
    echo "❌ EXACT PREPROCESSING TEST FAILED"
fi

echo ""
echo "Final GPU state:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
