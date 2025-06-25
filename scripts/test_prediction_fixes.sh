#!/bin/bash
#SBATCH --job-name=test_fixes
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a4500:1
#SBATCH --time=0:15:00
#SBATCH --output=../logs/test_fixes_%j.out
#SBATCH --error=../logs/test_fixes_%j.err

echo "🔧 TESTING PREDICTION FIXES"
echo "=========================="
echo "Purpose: Fix under-confident model (max pred = 0.07)"
echo "Strategy: Test 5 different calibration methods"
echo "Expected: Find method that gives Dice > 0.2"
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo ""

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
echo ""

# Run the fix testing
echo "🚀 Testing all prediction calibration methods..."
python -u fix_underconfident_model.py

exit_code=$?

echo ""
echo "============================================================"
echo "PREDICTION FIX TESTING COMPLETED"
echo "============================================================"
echo "Exit code: $exit_code"
echo "End time: $(date)"

if [ $exit_code -eq 0 ]; then
    echo "✅ FIX TESTING SUCCESSFUL - Solution found!"
    echo ""
    echo "Check the output above for:"
    echo "  - Best calibration method"
    echo "  - Achieved Dice score"
    echo "  - Next steps for implementation"
    
else
    echo "❌ FIX TESTING FAILED"
    echo "May need deeper model investigation"
fi

echo ""
echo "Final GPU status:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
