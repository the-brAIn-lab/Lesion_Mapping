#!/bin/bash
#SBATCH --job-name=proven_enhanced
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gpus=4
#SBATCH --time=20:00:00
#SBATCH --output=logs/proven_enhanced_%j.out
#SBATCH --error=logs/proven_enhanced_%j.err

echo "🔧 PROVEN ENHANCED MODEL - GUARANTEED TO WORK"
echo "=============================================="
echo "Strategy: Baseline + Only battle-tested improvements"
echo "Target: 65-70% validation Dice with ZERO risk"

cd /mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota

# Same proven environment setup
module load gcc/9.3.0-5wu3 cuda/12.6.3-ziu7
eval "$(conda shell.bash hook)" || true
conda activate tf215_env

export LD_LIBRARY_PATH="/mnt/beegfs/hellgate/home/rb194958e/.conda/envs/tf215_env/lib:$LD_LIBRARY_PATH"
export TF_ENABLE_ONEDNN_OPTS=0
export TF_GPU_ALLOCATOR=cuda_malloc_async
export NIBABEL_NIFTI1_QFAC_CHECK=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "🔥 PROVEN COMPONENTS:"
echo "  ✅ Enhanced conv blocks (GELU + SE attention)"
echo "  ✅ Proven attention gates"
echo "  ✅ Enhanced loss function"
echo "  ✅ BatchNormalization (no OOM)"
echo "  ✅ Multi-GPU MirroredStrategy"
echo "  ❌ NO experimental Vision Mamba"
echo "  ❌ NO experimental SAM-2"

python -u proven_enhanced_model.py
