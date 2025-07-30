#!/bin/bash
#SBATCH --job-name=simplified_fallback
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a4500:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/simplified_fallback_%j.out
#SBATCH --error=logs/simplified_fallback_%j.err

echo "🛠️ SIMPLIFIED SOTA FALLBACK TRAINING"
echo "====================================="
echo "Strategy: Remove ALL potentially problematic components"
echo "Goal: Achieve stable training and 65-70% validation Dice"
echo ""
echo "🔧 DISABLED COMPONENTS:"
echo "  ❌ Mixed precision (using float32)"
echo "  ❌ Vision Mamba blocks"
echo "  ❌ SAM2 attention"
echo "  ❌ Boundary-aware loss"
echo "  ❌ Complex augmentation"
echo ""
echo "✅ STABLE COMPONENTS:"
echo "  ✅ Standard U-Net architecture"
echo "  ✅ SE attention blocks"
echo "  ✅ Simple attention gates"
echo "  ✅ Dice + Focal loss"
echo "  ✅ Conservative batch size (2)"
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
echo "Environment configured:"
echo "Python: $(python --version)"
echo "TensorFlow: $(python -c 'import tensorflow as tf; print(tf.__version__)')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

echo ""
echo "🚀 Starting simplified SOTA fallback training..."
echo "Expected: Stable training with good Dice scores"

# Run the simplified training
python -u simplified_sota_fallback.py

exit_code=$?

echo ""
echo "====================================="
echo "SIMPLIFIED FALLBACK TRAINING COMPLETED"
echo "====================================="
echo "Exit code: $exit_code"
echo "End time: $(date)"

if [ $exit_code -eq 0 ]; then
    echo "✅ SIMPLIFIED TRAINING SUCCESSFUL!"
    echo ""
    echo "🎉 ACHIEVEMENTS:"
    echo "  ✅ Stable training without crashes"
    echo "  ✅ No NaN values in loss/metrics"
    echo "  ✅ Baseline performance established"
    echo ""
    echo "📈 Check results:"
    echo "  - Training logs: logs/simplified_sota_fallback.log"
    echo "  - Best model: callbacks/simplified_sota_fallback_*/best_model.h5"
    echo "  - Training metrics: callbacks/simplified_sota_fallback_*/training_log.csv"
    echo ""
    echo "🚀 Next steps:"
    echo "1. Evaluate simplified model performance"
    echo "2. If good results: Incrementally add back SOTA features"
    echo "3. Test each feature addition for stability"
    
else
    echo "❌ SIMPLIFIED TRAINING FAILED"
    echo ""
    echo "🔧 If even simplified model fails:"
    echo "1. Check data loading issues"
    echo "2. Verify preprocessing pipeline"
    echo "3. Test with even smaller model"
    echo "4. Check CUDA/TensorFlow compatibility"
fi

echo ""
echo "Final GPU state:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

echo ""
echo "Training session completed: $(date)"
