#!/bin/bash
#SBATCH --job-name=test_fixed_model
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a4500:1
#SBATCH --time=1:00:00
#SBATCH --output=logs/test_fixed_model_%j.out
#SBATCH --error=logs/test_fixed_model_%j.err

echo "Testing Fixed Model (5.7M parameters)"
echo "====================================="
echo "Expected improvements:"
echo "  Training case: Dice ~0.65 (similar to training)"
echo "  Test case: Dice ~0.3-0.5 (based on validation 0.47)"
echo "  Overall: MAJOR improvement from previous 0.0"
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
echo "Running fixed model test..."

# Run test
python -u test_fixed_model.py

exit_code=$?

echo ""
echo "Test completed with exit code: $exit_code"
echo "End time: $(date)"

if [ $exit_code -eq 0 ]; then
    echo "✅ FIXED MODEL TEST SUCCESSFUL"
    
    # If successful, check if we should run batch testing
    echo ""
    echo "If results show significant improvement (Dice > 0.3):"
    echo "Next step: Run batch testing on all 55 test cases"
    
else
    echo "❌ FIXED MODEL TEST FAILED"
fi

echo ""
echo "Final GPU state:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
