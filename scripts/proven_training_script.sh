#!/bin/bash
#SBATCH --job-name=stroke_atlas_training
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:a4500:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err

echo "🚀 ATLAS 2.0 Stroke Lesion Segmentation Training"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "Memory: 256GB"
echo "GPU: RTX 4500 Ada Generation"

# Navigate to project
cd /mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota || {
    echo "❌ Project directory not found"
    exit 1
}

# Load the EXACT modules that worked
echo "🔗 Loading proven working modules..."
module load gcc/9.3.0-5wu3
module load cuda/12.6.3-ziu7
module list

# Set proven environment variables
echo "⚙️ Setting proven environment variables..."
export TF_XLA_FLAGS="--tf_xla_enable_xla_devices=false"
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "TF_XLA_FLAGS: $TF_XLA_FLAGS"

# Activate the WORKING environment
echo "🐍 Activating tf215_env (the proven working environment)..."
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate tf215_env

echo "Environment: $CONDA_DEFAULT_ENV"
echo "Python: $(python --version)"

# THE CRITICAL FIX - Set library path for tf215_env
echo "🔧 Setting critical library path..."
export LD_LIBRARY_PATH="/mnt/beegfs/hellgate/home/rb194958e/.conda/envs/tf215_env/lib:$LD_LIBRARY_PATH"
echo "LD_LIBRARY_PATH configured for GPU libraries"

# Install nibabel if needed
echo "📦 Ensuring nibabel is available..."
pip install nibabel -q

# Verify GPU detection
echo "🔬 Verifying GPU detection..."
python -c "
import tensorflow as tf
print(f'TensorFlow: {tf.__version__}')
print(f'Built with CUDA: {tf.test.is_built_with_cuda()}')

gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs detected: {len(gpus)}')

if len(gpus) >= 1:
    print('✅ SUCCESS: GPU libraries found!')
    for i, gpu in enumerate(gpus):
        print(f'  GPU {i}: {gpu}')
        tf.config.experimental.set_memory_growth(gpu, True)
    
    # Test computation
    with tf.device('/GPU:0'):
        test = tf.random.normal((100, 100))
        result = tf.reduce_mean(test)
        print(f'GPU computation test: {result.numpy():.4f}')
        print('🎉 Ready for training!')
else:
    print('❌ No GPUs detected - check environment setup')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ GPU verification failed - exiting"
    exit 1
fi

# Verify data
echo "📊 Verifying ATLAS 2.0 data..."
python -c "
from pathlib import Path
data_dir = Path('/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training')
image_files = list((data_dir / 'Images').glob('*_T1w.nii.gz'))
mask_files = list((data_dir / 'Masks').glob('*_mask.nii.gz'))
print(f'Image files: {len(image_files)}')
print(f'Mask files: {len(mask_files)}')
if len(image_files) > 0:
    print('✅ ATLAS 2.0 data found')
else:
    print('❌ No data found')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Data verification failed - exiting"
    exit 1
fi

# Create directories
mkdir -p logs checkpoints outputs

# Check training script exists
if [ ! -f "training/robust_train.py" ]; then
    echo "❌ Training script not found: training/robust_train.py"
    ls -la training/
    exit 1
fi

if [ ! -f "real_data_loader.py" ]; then
    echo "❌ Data loader not found: real_data_loader.py"
    ls -la *.py
    exit 1
fi

# Start training
echo "🎯 Starting ATLAS 2.0 Training..."
echo "Expected: 655 image-mask pairs"
echo "Target: Dice score > 0.75"
echo "Time: $(date)"

# Monitor GPU usage
nvidia-smi &

# Run training
python training/robust_train.py

TRAIN_EXIT_CODE=$?

# Results
echo "📊 Training Results:"
echo "Completed: $(date)"
echo "Exit code: $TRAIN_EXIT_CODE"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    
    if [ -f "checkpoints/best_model.h5" ]; then
        echo "Model saved: checkpoints/best_model.h5"
        echo "Size: $(du -h checkpoints/best_model.h5 | cut -f1)"
    fi
    
    if [ -f "logs/training.log" ]; then
        echo ""
        echo "📈 Final training metrics:"
        tail -20 logs/training.log | grep -E "(Epoch|loss|dice|val_)" || echo "Check logs/training.log for metrics"
    fi
    
    echo ""
    echo "🎉 ATLAS 2.0 Stroke Lesion Segmentation Model Training Complete!"
    echo "Next steps:"
    echo "1. Evaluate model on test set"
    echo "2. Generate prediction visualizations"
    echo "3. Calculate final performance metrics"
    
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

# Final GPU status
echo ""
echo "🖥️ Final GPU Status:"
nvidia-smi

echo ""
echo "🏁 Job Summary:"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Duration: $SECONDS seconds"
echo "Status: $([ $TRAIN_EXIT_CODE -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
echo "Completed: $(date)"

exit $TRAIN_EXIT_CODE
