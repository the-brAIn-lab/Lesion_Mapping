#!/bin/bash
#SBATCH --job-name=diagnostic_test
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a4500:1
#SBATCH --time=0:30:00
#SBATCH --output=logs/diagnostic_test_%j.out
#SBATCH --error=logs/diagnostic_test_%j.err

echo "🔍 SYSTEMATIC DIAGNOSIS OF SMART SOTA 2025 ISSUES"
echo "================================================="
echo "Purpose: Test each component individually to find root cause"
echo "Expected: Identify which components cause NaN values"
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
echo "🧪 Running comprehensive diagnostic tests..."
python -u diagnostic_test.py

exit_code=$?

echo ""
echo "================================================="
echo "DIAGNOSTIC TEST COMPLETED"
echo "================================================="
echo "Exit code: $exit_code"
echo "End time: $(date)"

if [ $exit_code -eq 0 ]; then
    echo "✅ DIAGNOSIS SUCCESSFUL"
    echo ""
    echo "📋 Next steps based on results:"
    echo "1. Check logs above for specific failing components"
    echo "2. If multiple failures: Run simplified fallback model"
    echo "3. If minimal failures: Apply targeted fixes"
    
else
    echo "❌ DIAGNOSIS FAILED"
    echo "Check error logs for details"
fi

echo ""
echo "Final GPU state:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
