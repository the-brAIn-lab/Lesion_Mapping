#!/bin/bash
#SBATCH --job-name=stroke_memory_opt
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:a4500:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err

echo "🚀 Memory-Optimized Stroke Segmentation Training"
echo "==============================================="

cd /mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota

# Load modules and setup environment (proven working)
module load gcc/9.3.0-5wu3 cuda/12.6.3-ziu7
source /opt/anaconda3/etc/profile.d/conda.sh 2>/dev/null || eval "$(conda shell.bash hook)"
conda activate tf215_env
export LD_LIBRARY_PATH="/mnt/beegfs/hellgate/home/rb194958e/.conda/envs/tf215_env/lib:$LD_LIBRARY_PATH"
export TF_XLA_FLAGS="--tf_xla_enable_xla_devices=false"
export CUDA_VISIBLE_DEVICES=0

# Create memory-optimized config
cat > config/memory_optimized.yaml << 'EOCONFIG'
model:
  input_shape: [192, 224, 176, 1]
  base_filters: 16  # Reduced from 32
  depth: 4          # Reduced from 5
  use_attention: true
  use_deep_supervision: false  # Disabled to save memory

training:
  epochs: 100
  batch_size_per_gpu: 1  # Reduced from 2
  learning_rate: 0.0001
  validation_split: 0.2
  patience: 15
  global_batch_size: 1

data:
  data_dir: '/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training'
  cache_dir: '/tmp/stroke_cache'
  num_workers: 4
  prefetch: 2
EOCONFIG

echo "📊 Starting memory-optimized training..."
echo "Model: 16 base filters, depth 4, batch size 1"
echo "Expected GPU usage: ~12-16GB (within 24GB limit)"

python training/robust_train.py --config config/memory_optimized.yaml

echo "Training completed: $(date)"
