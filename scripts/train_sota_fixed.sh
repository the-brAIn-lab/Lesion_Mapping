#!/bin/bash
#SBATCH --job-name=stroke_sota_fixed
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a4500:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/sota_fixed_%j.out
#SBATCH --error=logs/sota_fixed_%j.err

echo "=================================================="
echo "🚀 FULL STATE-OF-THE-ART STROKE SEGMENTATION (FIXED)"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo ""

# Navigate to project directory
cd /mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota

# Setup environment (proven working configuration)
echo "📦 Setting up environment..."
module load gcc/9.3.0-5wu3 cuda/12.6.3-ziu7

# Fix conda activation - use the method that works
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate tf215_env

# Critical environment variables
export LD_LIBRARY_PATH="/mnt/beegfs/hellgate/home/rb194958e/.conda/envs/tf215_env/lib:$LD_LIBRARY_PATH"
export TF_XLA_FLAGS="--tf_xla_enable_xla_devices=false"
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_ENABLE_ONEDNN_OPTS=0

# Create necessary directories
mkdir -p logs models callbacks results

# Check environment
echo "🔍 Checking environment..."
python -c "
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')
gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs available: {len(gpus)}')
for gpu in gpus:
    print(f'  {gpu}')
"

# Check data directory
echo ""
echo "📁 Checking data directory..."
echo "Training data: /mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training_Split"
ls -la /mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training_Split/

# Count images
echo ""
echo "📊 Dataset statistics:"
echo "Training images: $(ls -1 /mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training_Split/Images/*.nii.gz 2>/dev/null | wc -l)"
echo "Training masks: $(ls -1 /mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training_Split/Masks/*.nii.gz 2>/dev/null | wc -l)"

# Run the fixed training script
echo ""
echo "🚀 Starting Full SOTA Training (Fixed)..."
echo "=================================================="

# Use the fixed version
python fixed_sota_training.py

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Training completed successfully!"
    
    # List generated files
    echo ""
    echo "📁 Generated files:"
    ls -la models/sota_hybrid_final_*.h5 2>/dev/null || echo "No final models found"
    ls -la callbacks/sota_full_*/best_model.h5 2>/dev/null || echo "No best models found"
    
    # Show final results
    latest_summary=$(ls -t callbacks/sota_full_*/training_summary.json 2>/dev/null | head -1)
    if [ -f "$latest_summary" ]; then
        echo ""
        echo "📊 Training Summary:"
        cat "$latest_summary"
    fi
else
    echo ""
    echo "❌ Training failed! Check error logs."
fi

echo ""
echo "End Time: $(date)"
echo "=================================================="
