#!/bin/bash
#SBATCH --job-name=debug_sam2
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=0:15:00
#SBATCH --output=logs/debug_sam2_%j.out
#SBATCH --error=logs/debug_sam2_%j.err

echo "🔍 DEBUGGING SAM2 ATTENTION ISSUE"
echo "================================="
echo "Purpose: Find the exact issue in SAM2InspiredAttention"
echo "Mamba fix worked, now debugging SAM2 component"
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
echo "Testing SAM2 attention component..."

# Run the SAM2 debug test
python -u debug_sam2_attention.py

exit_code=$?

echo ""
echo "SAM2 debug completed with exit code: $exit_code"
echo "End time: $(date)"

if [ $exit_code -eq 0 ]; then
    echo "✅ SAM2 ISSUE IDENTIFIED"
else
    echo "❌ SAM2 DEBUG FAILED"
fi
