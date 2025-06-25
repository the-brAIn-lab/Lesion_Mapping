#!/bin/bash
#SBATCH --job-name=stroke_segmentation_a4500
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:a4500:4
#SBATCH --time=24:00:00
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_email@umt.edu

echo "🚀 Stroke Lesion Segmentation Training on A4500 GPUs"
echo "===================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node: $SLURMD_NODENAME"
echo "GPUs allocated: $SLURM_GPUS (expecting 4x A4500)"
echo "Memory: 256GB"
echo "CPUs: 16 cores"
echo "Time limit: 24 hours"
echo "Started: $(date)"

# Navigate to project
cd /mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota || {
    echo "❌ Project directory not found"
    exit 1
}

# System information
echo -e "\n📊 System Resources:"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "CPUs: $(nproc)"
echo "Storage: $(df -h . | tail -1 | awk '{print $2 " available"}')"

# GPU hardware check
echo -e "\n🖥️  GPU Hardware:"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
echo ""
nvidia-smi

# Load modules
echo -e "\n🔗 Loading CUDA modules:"
module purge
if module load cuda/12.6.3-gcc-9.3.0 2>/dev/null; then
    echo "✅ Loaded Spack CUDA 12.6.3"
else
    echo "⚠️  Using system CUDA"
fi
module list 2>&1

# Environment setup for 4x A4500
echo -e "\n⚙️  Environment Configuration:"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_ROOT="/opt/ohpc/pub/spack/opt/spack/linux-rocky8-skylake_avx512/gcc-9.3.0/cuda-12.6.3-ziu74ka3i2glo6q2zt3dy76cnbydqwf4"
export PATH="$CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:$LD_LIBRARY_PATH"

# Optimizations for A4500
export TF_ENABLE_ONEDNN_OPTS=1
export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=4

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Using 4x NVIDIA A4500 GPUs"

# Activate environment
echo -e "\n🐍 Activating conda environment:"
source /opt/anaconda3/etc/profile.d/conda.sh

if conda activate stroke_sota 2>/dev/null; then
    echo "✅ Using stroke_sota environment"
elif conda activate tf215_env 2>/dev/null; then
    echo "✅ Using tf215_env environment"
else
    echo "❌ Could not activate environment"
    exit 1
fi

echo "Environment: $CONDA_DEFAULT_ENV"
echo "Python: $(python --version)"
echo "TensorFlow: $(python -c 'import tensorflow as tf; print(tf.__version__)')"

# Verify GPU detection
echo -e "\n🔬 GPU Detection Verification:"
python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f'TensorFlow detected {len(gpus)} GPUs')
if len(gpus) == 4:
    print('✅ All 4 A4500 GPUs detected')
    for i, gpu in enumerate(gpus):
        tf.config.experimental.set_memory_growth(gpu, True)
        print(f'  GPU {i}: {gpu}')
else:
    print(f'❌ Expected 4 GPUs, got {len(gpus)}')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ GPU verification failed"
    exit 1
fi

# Create directories
mkdir -p logs checkpoints outputs results

# Training configuration
echo -e "\n📋 Training Configuration:"
echo "Model: Hybrid CNN-Transformer with Attention"
echo "Input size: 192×224×176 (auto-resized from 197×233×189)"
echo "Batch size: 8 (2 per GPU × 4 GPUs)"
echo "Expected training time: 8-12 hours"
echo "Target Dice score: 0.75-0.85"

# Start training
echo -e "\n🎯 Starting Training:"
echo "Command: python training/robust_train.py"
echo "Time: $(date)"

# Monitor GPU usage during training
python training/robust_train.py &
TRAIN_PID=$!

# Background GPU monitoring
(
    while kill -0 $TRAIN_PID 2>/dev/null; do
        echo "$(date): GPU Status:" >> logs/gpu_monitor_${SLURM_JOB_ID}.log
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv >> logs/gpu_monitor_${SLURM_JOB_ID}.log
        echo "" >> logs/gpu_monitor_${SLURM_JOB_ID}.log
        sleep 300  # Every 5 minutes
    done
) &
MONITOR_PID=$!

# Wait for training to complete
wait $TRAIN_PID
TRAIN_EXIT_CODE=$?

# Stop monitoring
kill $MONITOR_PID 2>/dev/null

# Results analysis
echo -e "\n📊 Training Results:"
echo "Training completed at: $(date)"
echo "Exit code: $TRAIN_EXIT_CODE"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    
    # Check for output files
    if [ -f "checkpoints/best_model.h5" ]; then
        echo "Model saved: checkpoints/best_model.h5"
        echo "Model size: $(du -h checkpoints/best_model.h5 | cut -f1)"
    fi
    
    if [ -f "logs/training.log" ]; then
        echo ""
        echo "📈 Final training metrics:"
        tail -20 logs/training.log | grep -E "(loss|dice|accuracy)" || echo "Check logs/training.log for metrics"
    fi
    
    # GPU utilization summary
    if [ -f "logs/gpu_monitor_${SLURM_JOB_ID}.log" ]; then
        echo ""
        echo "🖥️  GPU utilization summary:"
        tail -20 logs/gpu_monitor_${SLURM_JOB_ID}.log
    fi
    
else
    echo "❌ Training failed with exit code: $TRAIN_EXIT_CODE"
    
    echo ""
    echo "🔍 Error Analysis:"
    if [ -f "logs/training.log" ]; then
        echo "Last 30 lines of training log:"
        tail -30 logs/training.log
    fi
    
    echo ""
    echo "System status at failure:"
    nvidia-smi
    df -h
    free -h
fi

# Cleanup
echo -e "\n🧹 Cleanup:"
conda deactivate

# Final summary
echo -e "\n📋 Job Summary:"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "Duration: $SECONDS seconds"
echo "GPU Hardware: 4x NVIDIA A4500"
echo "Status: $([ $TRAIN_EXIT_CODE -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "🎉 Stroke lesion segmentation model training completed!"
    echo "Next steps:"
    echo "1. Evaluate model performance"
    echo "2. Test on validation data"
    echo "3. Generate prediction visualizations"
else
    echo ""
    echo "💡 Troubleshooting suggestions:"
    echo "1. Check logs/training.log for detailed errors"
    echo "2. Verify data paths and accessibility"
    echo "3. Monitor GPU memory usage in logs/gpu_monitor_${SLURM_JOB_ID}.log"
fi

exit $TRAIN_EXIT_CODE
