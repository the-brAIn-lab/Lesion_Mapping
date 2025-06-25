#!/bin/bash
#SBATCH --job-name=comprehensive_debug
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a4500:1
#SBATCH --time=0:30:00
#SBATCH --output=logs/comprehensive_debug_%j.out
#SBATCH --error=logs/comprehensive_debug_%j.err

echo "🔍 COMPREHENSIVE STROKE SEGMENTATION DEBUG"
echo "=========================================="
echo "Purpose: Identify core issue causing Dice ~0.0"
echo "Strategy: Test both orientations systematically"
echo "Expected outcomes:"
echo "  A) Flip fixes issue → Dice jumps to 0.3-0.5"
echo "  B) Model works → Dice 0.3+ in best orientation"  
echo "  C) Deeper issues → Both orientations poor"
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
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
echo ""

# Run comprehensive debug
echo "🚀 Running comprehensive debug analysis..."
python -u comprehensive_debug_test.py

exit_code=$?

echo ""
echo "=" * 60
echo "DEBUG COMPLETED"
echo "=" * 60
echo "Exit code: $exit_code"
echo "End time: $(date)"

if [ $exit_code -eq 0 ]; then
    echo "✅ DEBUG SUCCESSFUL - Issue identified!"
    echo ""
    echo "Check the output above for:"
    echo "  - Original vs Flipped orientation Dice scores"
    echo "  - Solution recommendation"
    echo "  - Next steps"
    echo ""
    echo "If flip orientation fixes the issue:"
    echo "  → Ready for batch testing with corrected orientation"
    echo ""
    echo "If model is working well:"
    echo "  → Proceed with batch testing"
    echo ""
    echo "If both orientations poor:"
    echo "  → Need deeper investigation of preprocessing/training"
    
else
    echo "❌ DEBUG FAILED"
    echo "Check error log for details"
fi

echo ""
echo "Final GPU status:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
