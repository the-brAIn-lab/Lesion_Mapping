#!/bin/bash
#SBATCH --job-name=batch_test_sota
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a4500:1
#SBATCH --time=8:00:00
#SBATCH --output=logs/batch_test_%j.out
#SBATCH --error=logs/batch_test_%j.err

echo "SOTA Model Batch Testing - All 55 Test Cases"
echo "============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "Expected runtime: ~2-3 hours for 55 cases"

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
echo "Environment configured:"
echo "Python: $(python --version)"
echo "TensorFlow: $(python -c 'import tensorflow as tf; print(tf.__version__)')"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "GPU allocator: $TF_GPU_ALLOCATOR"

echo ""
echo "System resources:"
echo "Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

echo ""
echo "Starting batch processing..."
echo "============================================="

# Run batch test with unbuffered output
python -u batch_test_all_cases.py

exit_code=$?

echo ""
echo "============================================="
echo "Batch testing completed"
echo "Exit code: $exit_code"
echo "End time: $(date)"

if [ $exit_code -eq 0 ]; then
    echo "✅ BATCH TESTING SUCCESSFUL"
    
    # Show summary if results exist
    if [ -f "batch_test_results/summary_results.txt" ]; then
        echo ""
        echo "SUMMARY RESULTS:"
        echo "================"
        cat batch_test_results/summary_results.txt
    fi
    
    if [ -d "batch_test_results" ]; then
        echo ""
        echo "OUTPUT FILES:"
        echo "============="
        ls -lh batch_test_results/
        
        if [ -d "batch_test_results/predictions" ]; then
            echo ""
            echo "Predictions created: $(ls batch_test_results/predictions/ | wc -l) files"
        fi
    fi
else
    echo "❌ BATCH TESTING FAILED"
fi

echo ""
echo "Final system state:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
