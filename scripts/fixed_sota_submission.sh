#!/bin/bash
#SBATCH --job-name=stroke_sota_fixed
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a4500:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/sota_training_%j.out
#SBATCH --error=logs/sota_training_%j.err

echo "🚀 State-of-the-Art Stroke Segmentation - Progressive Training"
echo "============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo ""

# Navigate to project directory
cd /mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota

# Setup proven working environment
echo "📦 Setting up environment..."
module load gcc/9.3.0-5wu3 cuda/12.6.3-ziu7

# Fix conda activation
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate tf215_env

# CRITICAL: Set library path for GPU detection
export LD_LIBRARY_PATH="/mnt/beegfs/hellgate/home/rb194958e/.conda/envs/tf215_env/lib:$LD_LIBRARY_PATH"

# TensorFlow configuration
export TF_XLA_FLAGS="--tf_xla_enable_xla_devices=false"
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=1

# Create necessary directories
mkdir -p logs models checkpoints results callbacks config

# Test environment
echo "🔍 Testing environment..."
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
python -c "import tensorflow as tf; print(f'GPUs: {tf.config.list_physical_devices(\"GPU\")}')"

# Check if required files exist
echo "📁 Checking required files..."
for file in sota_training_pipeline.py config/auto_config.py models/hybrid_model.py models/losses.py; do
    if [ -f "$file" ]; then
        echo "✅ Found: $file"
    else
        echo "❌ Missing: $file"
    fi
done

# If models directory doesn't have required files, check if they're in the root
if [ ! -f "models/hybrid_model.py" ]; then
    echo "⚠️  models/hybrid_model.py not found, checking for hybrid_model.py in root..."
    if [ -f "hybrid_model.py" ]; then
        echo "📋 Found hybrid_model.py in root, creating models directory structure..."
        mkdir -p models
        cp hybrid_model.py models/ 2>/dev/null || echo "Note: hybrid_model.py not in root"
    fi
fi

if [ ! -f "models/losses.py" ]; then
    echo "⚠️  models/losses.py not found, will use built-in losses"
fi

# Run the training with better error handling
echo ""
echo "🎯 Starting progressive training pipeline..."
echo ""

# First, let's test if the pipeline can be imported
python -c "
import sys
sys.path.append('/mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota')
try:
    import sota_training_pipeline
    print('✅ Pipeline module imported successfully')
except Exception as e:
    print(f'❌ Failed to import pipeline: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Import test failed, checking for missing dependencies..."
    
    # Try running with basic optimized training instead
    echo "🔄 Attempting fallback to optimized_hybrid_training.py..."
    python optimized_hybrid_training.py
else
    # Run the full pipeline
    python sota_training_pipeline.py --progressive --config config/auto_generated_config.yaml 2>&1 | tee training_output.log
fi

# Check if training completed
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Training completed successfully!"
    
    # List generated models
    echo "📊 Generated models:"
    ls -la models/*.h5 2>/dev/null || echo "No models found in models/"
    ls -la checkpoints/*.h5 2>/dev/null || echo "No models found in checkpoints/"
    
else
    echo "❌ Training failed!"
    echo "Check logs for details"
fi

echo ""
echo "End Time: $(date)"
echo "============================================================="
