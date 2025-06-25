#!/bin/bash
#SBATCH --job-name=stroke_sota_progressive
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

# Activate conda environment
eval "$(conda shell.bash hook)" 2>/dev/null || source /opt/anaconda3/etc/profile.d/conda.sh
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
mkdir -p logs models checkpoints results callbacks

# Check GPU availability
echo "🔍 Checking GPU..."
python -c "import tensorflow as tf; print(f'GPUs: {tf.config.list_physical_devices(\"GPU\")}')"

# Generate auto-configuration based on hardware
echo "⚙️ Generating optimal configuration..."
python -c "
import sys
sys.path.append('/mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota')
from config.auto_config import create_auto_config
import yaml

# Create configuration for RTX 4500 (24GB)
auto_config = create_auto_config(
    data_dir='/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training',
    target_memory_gb=20  # Conservative target
)

# Generate and save configuration
config = auto_config.generate_complete_config(dataset_size=655)
with open('config/auto_generated_config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, indent=2)

print('✅ Configuration generated successfully')
print(f'Architecture: {config[\"model\"][\"base_filters\"]} filters, depth {config[\"model\"][\"depth\"]}')
print(f'Transformer: {config[\"model\"][\"use_transformer\"]}')
print(f'Estimated memory: {config[\"meta\"][\"estimated_memory_gb\"]:.1f}GB')
"

# Run progressive training
echo ""
echo "🎯 Starting progressive training pipeline..."
echo "Stage 1: Basic CNN (memory test)"
echo "Stage 2: CNN with attention"
echo "Stage 3: Small hybrid CNN-Transformer"
echo "Stage 4: Full SOTA architecture"
echo ""

# Copy the training script to ensure it's available
cp "$SLURM_SUBMIT_DIR/sota_training_pipeline.py" .

# Run the training
python sota_training_pipeline.py --progressive --config config/auto_generated_config.yaml

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Training completed successfully!"
    
    # Run final evaluation
    echo "📊 Running final evaluation..."
    python sota_training_pipeline.py --evaluate models/stage4_hybrid_full_final.h5
    
    # Archive results
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    tar -czf "results_${TIMESTAMP}.tar.gz" results/ models/ callbacks/
    echo "📦 Results archived: results_${TIMESTAMP}.tar.gz"
else
    echo "❌ Training failed!"
    exit 1
fi

echo ""
echo "End Time: $(date)"
echo "============================================================="
