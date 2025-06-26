#!/bin/bash
#SBATCH --job-name=debug_mamba_dims
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=0:15:00
#SBATCH --output=logs/debug_mamba_%j.out
#SBATCH --error=logs/debug_mamba_%j.err

echo "🔍 DEBUGGING VISION MAMBA DIMENSION ISSUE"
echo "========================================="
echo "Purpose: Fix the exact dimension mismatch in your Smart SOTA 2025"
echo "Error: Dimensions 192 and 176 incompatible in SSM computation"
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

echo ""
echo "Environment configured"
echo "Testing Vision Mamba dimension fix..."

# Run the debug test
python -u mamba_debug_fix.py

exit_code=$?

echo ""
echo "Debug test completed with exit code: $exit_code"
echo "End time: $(date)"

if [ $exit_code -eq 0 ]; then
    echo "✅ DIMENSION FIX SUCCESSFUL"
    echo ""
    echo "Next steps:"
    echo "1. Apply the fix to smart_sota_2025.py"
    echo "2. Re-run the full training"
else
    echo "❌ DEBUG TEST FAILED"
fi
