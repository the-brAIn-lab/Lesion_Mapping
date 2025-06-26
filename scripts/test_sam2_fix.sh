#!/bin/bash
#SBATCH --job-name=test_sam2_fix
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=0:15:00
#SBATCH --output=logs/test_sam2_fix_%j.out
#SBATCH --error=logs/test_sam2_fix_%j.err

echo "🔧 TESTING SAM2 MEMORY-EFFICIENT FIX"
echo "===================================="
echo "Purpose: Test memory-efficient SAM2 attention that prevents OOM"
echo "Problem: Original SAM2 tried to create 946K×946K attention matrix (~3.6TB)"
echo "Solution: Use hierarchical/pooled attention to reduce memory"
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
echo "Testing memory-efficient SAM2 attention fix..."

# Run the memory-efficient SAM2 test
python -u sam2_attention_fix.py

exit_code=$?

echo ""
echo "SAM2 fix test completed with exit code: $exit_code"
echo "End time: $(date)"

if [ $exit_code -eq 0 ]; then
    echo "✅ SAM2 MEMORY FIX SUCCESSFUL"
    echo ""
    echo "Next steps:"
    echo "1. Apply the SAM2 fix to smart_sota_2025_fixed.py"
    echo "2. Re-test the full model building"
    echo "3. Run the complete training"
else
    echo "❌ SAM2 FIX STILL HAS ISSUES"
fi
