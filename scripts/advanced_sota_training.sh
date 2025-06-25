#!/bin/bash
#SBATCH --job-name=advanced_sota_training
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=192G
#SBATCH --gres=gpu:a4500:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/advanced_sota_%j.out
#SBATCH --error=logs/advanced_sota_%j.err

echo "🚀 ADVANCED SOTA TRAINING - TARGET: 75-80% DICE"
echo "=================================================="
echo "🔥 CUTTING-EDGE FEATURES:"
echo "  ✨ Swin Transformer blocks"
echo "  🎯 Advanced attention gates (spatial + channel)"
echo "  📐 Deep supervision"
echo "  🌊 Topology-aware loss"
echo "  🎲 Strong augmentation (elastic + mixup)"
echo "  📊 Cosine scheduling with warmup"
echo "  🧠 Group normalization + GELU"
echo "  ⚖️ AdamW with weight decay"
echo ""
echo "📊 CONFIGURATION:"
echo "  Dataset: 655 images (stratified split)"
echo "  Model: ~15M parameters (32 base filters)"
echo "  Training: 100 epochs with advanced scheduling"
echo "  Batch size: 2 (larger model)"
echo "  Validation: 15% (stratified by lesion size)"
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=================================================="
echo ""

cd /mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota

# Enhanced environment setup
module load gcc/9.3.0-5wu3 cuda/12.6.3-ziu7
eval "$(conda shell.bash hook)" || true
conda activate tf215_env

# Advanced environment variables
export LD_LIBRARY_PATH="/mnt/beegfs/hellgate/home/rb194958e/.conda/envs/tf215_env/lib:$LD_LIBRARY_PATH"
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0
export TF_GPU_ALLOCATOR=cuda_malloc_async
export NIBABEL_NIFTI1_QFAC_CHECK=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_LAUNCH_BLOCKING=0

# GPU optimization
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Verify advanced dependencies
echo "🔍 VERIFYING ADVANCED DEPENDENCIES:"
python -c "
import tensorflow as tf
print(f'✅ TensorFlow: {tf.__version__}')
print(f'✅ GPU Available: {tf.config.list_physical_devices(\"GPU\")}')
print(f'✅ Mixed Precision: {tf.keras.mixed_precision.global_policy()}')

import scipy
print(f'✅ SciPy: {scipy.__version__}')

import sklearn
print(f'✅ Scikit-learn: {sklearn.__version__}')

import nibabel as nib
print(f'✅ NiBabel: {nib.__version__}')
"
echo ""

echo "🎯 ADVANCED DATASET ANALYSIS:"
python -c "
import os
data_dir = '/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training'
images = len([f for f in os.listdir(os.path.join(data_dir, 'Images')) if f.endswith('.nii.gz')])
masks = len([f for f in os.listdir(os.path.join(data_dir, 'Masks')) if f.endswith('.nii.gz')])
print(f'📊 Images: {images}')
print(f'📊 Masks: {masks}')
print(f'✅ Dataset ready: {images == masks and images >= 650}')
"
echo ""

echo "🚀 STARTING ADVANCED SOTA TRAINING..."
echo "Target: Breakthrough 75% validation Dice barrier!"
echo ""

# Run the advanced training
python -u advanced_sota_training.py

exit_code=$?

echo ""
echo "================================================================"
echo "🏁 ADVANCED SOTA TRAINING COMPLETED"
echo "================================================================"
echo "Exit code: $exit_code"
echo "End time: $(date)"
echo ""

if [ $exit_code -eq 0 ]; then
    echo "🎉 ADVANCED TRAINING SUCCESSFUL!"
    echo ""
    echo "🏆 ACHIEVEMENTS UNLOCKED:"
    echo "  ✨ Swin Transformer integration"
    echo "  🎯 Multi-scale attention mechanisms"
    echo "  📐 Deep supervision training"
    echo "  🌊 Topology-aware optimization"
    echo "  🎲 Advanced augmentation pipeline"
    echo "  📊 Sophisticated learning rate scheduling"
    echo ""
    echo "🔍 CHECK RESULTS:"
    echo "  📈 Training logs: logs/advanced_sota_training.log"
    echo "  💾 Best model: callbacks/advanced_sota_*/best_model.h5"
    echo "  📊 TensorBoard: callbacks/advanced_sota_*/tensorboard/"
    echo ""
    echo "🎯 NEXT STEPS:"
    echo "  1. Check final validation Dice score"
    echo "  2. Run inference on test cases"
    echo "  3. Consider ensemble methods if needed"
    echo "  4. Deploy for clinical evaluation"
    
else
    echo "❌ ADVANCED TRAINING FAILED"
    echo ""
    echo "🔧 DEBUGGING STEPS:"
    echo "  1. Check GPU memory usage"
    echo "  2. Review training logs for errors"
    echo "  3. Verify dataset integrity"
    echo "  4. Consider reducing batch size"
    echo ""
    echo "💡 FALLBACK OPTIONS:"
    echo "  - Reduce model complexity (base_filters=16)"
    echo "  - Disable Swin blocks temporarily"
    echo "  - Use single GPU precision"
fi

echo ""
echo "📊 FINAL GPU STATUS:"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv

echo ""
echo "🏆 ADVANCED SOTA TRAINING SESSION COMPLETE!"
echo "================================================================"
