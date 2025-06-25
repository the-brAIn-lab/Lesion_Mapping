#!/bin/bash
#SBATCH --job-name=fine_tune_multi_gpu
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gpus=4                        # Same 4 GPUs as original
#SBATCH --time=12:00:00                 # Shorter time for fine-tuning
#SBATCH --output=logs/fine_tune_multi_gpu_%j.out
#SBATCH --error=logs/fine_tune_multi_gpu_%j.err

echo "🔄 FINE-TUNING FROM 72% DICE CHECKPOINT"
echo "======================================"
echo "🎯 STRATEGY:"
echo "  📚 Load: callbacks/multi_gpu_advanced_sota_20250624_044219/best_model.h5"
echo "  🔧 Method: Lower LR + Stronger regularization"
echo "  🎯 Target: Validation Dice 45% → 55-65%"
echo "  ⏱️ Duration: 6-12 hours (vs 30-40 for full training)"
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo ""

cd /mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota

# Enhanced environment setup (same as original)
module load gcc/9.3.0-5wu3 cuda/12.6.3-ziu7
eval "$(conda shell.bash hook)" || true
conda activate tf215_env

# Multi-GPU environment variables (same as original)
export LD_LIBRARY_PATH="/mnt/beegfs/hellgate/home/rb194958e/.conda/envs/tf215_env/lib:$LD_LIBRARY_PATH"
export TF_ENABLE_ONEDNN_OPTS=0
export TF_GPU_ALLOCATOR=cuda_malloc_async
export NIBABEL_NIFTI1_QFAC_CHECK=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO

echo "🔍 CHECKPOINT STATUS:"
echo "Target checkpoint: callbacks/multi_gpu_advanced_sota_20250624_044219/best_model.h5"
if [ -f "callbacks/multi_gpu_advanced_sota_20250624_044219/best_model.h5" ]; then
    ls -lah callbacks/multi_gpu_advanced_sota_20250624_044219/best_model.h5
    echo "✅ Checkpoint found and ready to load"
else
    echo "❌ Checkpoint not found!"
    echo "Available checkpoints:"
    find callbacks/ -name "best_model.h5" -exec ls -lah {} \;
fi
echo ""

echo "🔍 MULTI-GPU ENVIRONMENT STATUS:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

echo "🧪 TESTING TENSORFLOW MULTI-GPU:"
python -c "
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')
gpus = tf.config.list_physical_devices('GPU')
print(f'Detected GPUs: {len(gpus)}')
strategy = tf.distribute.MirroredStrategy()
print(f'✅ MirroredStrategy ready with {strategy.num_replicas_in_sync} devices')
"
echo ""

echo "📊 FINE-TUNING CONFIGURATION:"
echo "  🔄 Load checkpoint: 72% training Dice model"
echo "  📉 Learning rate: 5e-6 (down from 3e-4)"
echo "  ⚖️ Weight decay: 1e-4 (L2 regularization)"
echo "  🛑 Early stopping: 12 epochs patience"
echo "  🎲 Enhanced augmentation: Mixup + elastic deformation"
echo "  📈 Target improvement: 10-20% validation Dice"
echo ""

echo "🔄 STARTING FINE-TUNING..."
echo "Expected: Better generalization, reduced overfitting"
echo "Timeline: 6-12 hours for 30 epochs"
echo ""

# Run the fine-tuning
python -u fine_tune_multi_gpu.py

exit_code=$?

echo ""
echo "================================================================"
echo "🏁 FINE-TUNING COMPLETED"
echo "================================================================"
echo "Exit code: $exit_code"
echo "End time: $(date)"
echo ""

if [ $exit_code -eq 0 ]; then
    echo "🎉 FINE-TUNING SUCCESSFUL!"
    echo ""
    echo "🏆 EXPECTED ACHIEVEMENTS:"
    echo "  📈 Improved validation Dice (target: 55-65%)"
    echo "  🎯 Reduced train/validation gap"
    echo "  🧠 Better generalization"
    echo "  💪 More robust predictions"
    echo ""
    echo "🔍 CHECK RESULTS:"
    echo "  📈 Training logs: logs/fine_tune_multi_gpu.log"
    echo "  💾 Best model: callbacks/fine_tune_*/best_model.h5"
    echo "  📊 CSV logs: callbacks/fine_tune_*/training_log.csv"
    echo ""
    echo "📊 PERFORMANCE COMPARISON:"
    echo "  Original training: 72% Dice"
    echo "  Original validation: 45% Dice"
    echo "  Fine-tuned validation: Check logs for improvement!"
    
else
    echo "❌ FINE-TUNING FAILED"
    echo ""
    echo "🔧 DEBUGGING STEPS:"
    echo "  1. Check if checkpoint exists and is loadable"
    echo "  2. Verify custom_objects in model loading"
    echo "  3. Check GPU memory usage"
    echo ""
    echo "💡 FALLBACK OPTIONS:"
    echo "  - Try different checkpoint path"
    echo "  - Use single GPU fine-tuning"
    echo "  - Reduce batch size to 2"
fi

echo ""
echo "📊 FINAL GPU STATUS:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv

echo ""
echo "🔄 FINE-TUNING SESSION COMPLETE!"
echo "================================================================"
