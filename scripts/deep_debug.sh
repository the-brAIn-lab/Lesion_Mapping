#!/bin/bash
#SBATCH --job-name=deep_debug
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a4500:1
#SBATCH --time=0:20:00
#SBATCH --output=../logs/deep_debug_%j.out
#SBATCH --error=../logs/deep_debug_%j.err

echo "🔬 DEEP MODEL DEBUG ANALYSIS"
echo "============================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo ""

# Change to the main project directory
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
echo "Current directory: $(pwd)"
echo "Python script exists: $(ls -la deep_model_debug.py 2>/dev/null || echo 'NOT FOUND')"
echo ""

# Run the analysis
echo "🚀 Running deep model debug analysis..."
python -u deep_model_debug.py

exit_code=$?

echo ""
echo "============================================================"
echo "DEEP DEBUG COMPLETED"
echo "============================================================"
echo "Exit code: $exit_code"
echo "End time: $(date)"

if [ $exit_code -eq 0 ]; then
    echo "✅ DEEP DEBUG SUCCESSFUL"
else
    echo "❌ DEEP DEBUG FAILED"
fi
