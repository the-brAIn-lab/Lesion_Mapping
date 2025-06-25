#!/bin/bash
#SBATCH --job-name=sota_fixed_training
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a4500:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/sota_fixed_training_%j.out
#SBATCH --error=logs/sota_fixed_training_%j.err

echo "FIXED SOTA Training - With Validation Split"
echo "==========================================="
echo "Key improvements:"
echo "  ✅ Validation split: 20% (prevents overfitting)"
echo "  ✅ Increased BASE_FILTERS: 8 → 16 (better capacity)"
echo "  ✅ Monitoring validation metrics (proper early stopping)"
echo "  ✅ Reduced flip augmentation: 50% → 30%"
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "Expected training time: 6-8 hours"

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
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

echo ""
echo "Starting fixed training..."
echo "==========================================="

# Run training with unbuffered output
python -u sota_training_fixed.py

exit_code=$?

echo ""
echo "==========================================="
echo "Training completed with exit code: $exit_code"
echo "End time: $(date)"

if [ $exit_code -eq 0 ]; then
    echo "✅ TRAINING SUCCESSFUL"
    
    # Show latest results
    echo ""
    echo "Latest model files:"
    find callbacks models -name "*fixed*" -type d | head -5
    
    # Show training log summary
    if ls logs/sota_training_fixed.log 1> /dev/null 2>&1; then
        echo ""
        echo "Training summary (last 10 lines):"
        tail -10 logs/sota_training_fixed.log
    fi
    
else
    echo "❌ TRAINING FAILED"
    echo ""
    echo "Check error log for details:"
    tail -20 logs/sota_fixed_training_${SLURM_JOB_ID}.err
fi

echo ""
echo "Final GPU state:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
