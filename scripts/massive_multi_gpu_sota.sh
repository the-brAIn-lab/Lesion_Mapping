#!/bin/bash
#SBATCH --job-name=massive_multi_gpu_sota
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gpus=4                        # *** 4 GPUs FOR MASSIVE CAPACITY ***
#SBATCH --time=48:00:00
#SBATCH --output=logs/massive_multi_gpu_sota_%j.out
#SBATCH --error=logs/massive_multi_gpu_sota_%j.err

echo "🚀 MASSIVE MULTI-GPU SOTA TRAINING"
echo "=================================="
echo "🔥 UNPRECEDENTED SCALE:"
echo "  💪 4 GPUs × 24GB = 96GB total VRAM"
echo "  🧠 25-30M parameter model"
echo "  🎯 Target: 80%+ Dice coefficient"
echo "  📊 Batch size: 4 (1 per GPU)"
echo "  ⚡ MirroredStrategy distribution"
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo ""

cd /mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota

# Enhanced environment setup
module load gcc/9.3.0-5wu3 cuda/12.6.3-ziu7
eval "$(conda shell.bash hook)" || true
conda activate tf215_env

# Multi-GPU environment variables
export LD_LIBRARY_PATH="/mnt/beegfs/hellgate/home/rb194958e/.conda/envs/tf215_env/lib:$LD_LIBRARY_PATH"
export TF_ENABLE_ONEDNN_OPTS=0
export TF_GPU_ALLOCATOR=cuda_malloc_async
export NIBABEL_NIFTI1_QFAC_CHECK=0
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Multi-GPU specific settings
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Ensure all 4 GPUs are visible
export NCCL_DEBUG=INFO               # For debugging multi-GPU communication

echo "🔍 MULTI-GPU ENVIRONMENT STATUS:"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
echo ""

# Verify all GPUs are available
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

echo "🧪 TESTING TENSORFLOW MULTI-GPU DETECTION:"
python -c "
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')
gpus = tf.config.list_physical_devices('GPU')
print(f'Detected GPUs: {len(gpus)}')
for i, gpu in enumerate(gpus):
    print(f'  GPU {i}: {gpu}')

# Test MirroredStrategy
try:
    strategy = tf.distribute.MirroredStrategy()
    print(f'✅ MirroredStrategy ready with {strategy.num_replicas_in_sync} devices')
except Exception as e:
    print(f'❌ MirroredStrategy failed: {e}')
"
echo ""

echo "🔍 DIAGNOSING PREVIOUS FAILURE:"
if [ -f "logs/advanced_sota_training.log" ]; then
    echo "Previous training log found. Last 20 lines:"
    tail -20 logs/advanced_sota_training.log
    echo ""
fi

echo "🚀 STARTING MASSIVE MULTI-GPU SOTA TRAINING..."
echo "Expected memory usage per GPU: ~20-22GB"
echo "Total model size with 4 GPUs: 25-30M parameters"
echo "Estimated training time: 30-40 hours"
echo ""

# Run the massive multi-GPU training
python -u multi_gpu_advanced_sota.py

exit_code=$?

echo ""
echo "================================================================"
echo "🏁 MASSIVE MULTI-GPU SOTA TRAINING COMPLETED"
echo "================================================================"
echo "Exit code: $exit_code"
echo "End time: $(date)"
echo ""

if [ $exit_code -eq 0 ]; then
    echo "🎉 MASSIVE MULTI-GPU TRAINING SUCCESSFUL!"
    echo ""
    echo "🏆 ACHIEVEMENTS UNLOCKED:"
    echo "  🔥 4-GPU distributed training"
    echo "  💪 25-30M parameter model"
    echo "  🧠 Swin Transformers + Advanced Attention"
    echo "  📐 Deep supervision"
    echo "  🌊 Ultimate loss function"
    echo "  🎲 Advanced augmentation"
    echo ""
    echo "🔍 CHECK RESULTS:"
    echo "  📈 Training logs: logs/multi_gpu_advanced_sota.log"
    echo "  💾 Best model: callbacks/multi_gpu_advanced_sota_*/best_model.h5"
    echo "  📊 TensorBoard: callbacks/multi_gpu_advanced_sota_*/tensorboard/"
    echo ""
    echo "🎯 EXPECTED PERFORMANCE:"
    echo "  Target: 80%+ validation Dice"
    echo "  Baseline improvement: 15-20%"
    echo "  Clinical impact: Revolutionary"
    
else
    echo "❌ MASSIVE MULTI-GPU TRAINING FAILED"
    echo ""
    echo "🔧 DEBUGGING STEPS:"
    echo "  1. Check GPU memory in nvidia-smi"
    echo "  2. Review multi-GPU logs for errors"
    echo "  3. Verify MirroredStrategy setup"
    echo "  4. Check NCCL communication"
    echo ""
    echo "💡 FALLBACK OPTIONS:"
    echo "  - Reduce base_filters from 48 to 32"
    echo "  - Use 2 GPUs instead of 4"
    echo "  - Disable deep supervision temporarily"
fi

echo ""
echo "📊 FINAL GPU STATUS:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv

echo ""
echo "🚀 MASSIVE MULTI-GPU SOTA TRAINING SESSION COMPLETE!"
echo "================================================================"
