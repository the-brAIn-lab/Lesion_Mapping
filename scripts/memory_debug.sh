#!/bin/bash
#SBATCH --job-name=debug_memory
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a4500:1
#SBATCH --time=01:00:00
#SBATCH --output=../logs/debug_memory_%j.out
#SBATCH --error=../logs/debug_memory_%j.err

echo "🔍 MEMORY DEBUG SESSION"
echo "======================"
echo "Finding exact memory failure point in training script"
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
export NIBABEL_NIFTI1_QFAC_CHECK=0

echo "Environment configured"
echo ""

# Show system resources
echo "System resources:"
echo "CPU cores: $(nproc)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "GPU memory:"
nvidia-smi --query-gpu=memory.total,memory.free --format=csv
echo ""

# Install psutil if not available
python -c "import psutil" 2>/dev/null || pip install psutil

echo "🚀 Starting memory debugging..."
python -u debug_memory.py

exit_code=$?

echo ""
echo "============================================================"
echo "MEMORY DEBUG COMPLETED"
echo "============================================================"
echo "Exit code: $exit_code"
echo "End time: $(date)"

if [ $exit_code -eq 0 ]; then
    echo "✅ DEBUG SUCCESSFUL - No memory issues found"
    echo "The problem might be elsewhere in the training script"
else
    echo "❌ DEBUG FAILED - Memory issue identified"
    echo "Check the output above for the exact failure point"
fi

echo ""
echo "Final system status:"
free -h
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
