#!/bin/bash
#SBATCH --job-name=test_sam2_3d_fix
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=0:15:00
#SBATCH --output=logs/test_sam2_3d_fix_%j.out
#SBATCH --error=logs/test_sam2_3d_fix_%j.err

echo "🔧 TESTING SAM2 3D MEMORY-EFFICIENT FIX"
echo "======================================="
echo "Purpose: Test 3D-specific SAM2 attention that works with medical volumes"
echo "Problem: tf.image.resize doesn't work with 5D tensors (3D medical volumes)"
echo "Solution: Use 3D pooling + proper 3D upsampling operations"
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo ""

cd /mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota

# Setup environment
module load gcc/9.3.0-5wu3 cuda/12.6.3-ziu7
eval "$(conda shell.bash hook)" || true
conda activate tf215_env
export LD_LIBRARY_PATH="/mnt/beegfs/hellgate/home/rb194958e/.conda/envs/tf215_env/lib:$LD_LIBRARY_PATH"
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0
export TF_GPU_ALLOCATOR=cuda_malloc_async

echo "Environment configured"
echo ""

echo "🚀 Testing 3D-specific SAM2 fix..."
python -u sam2_3d_fix.py

exit_code=$?

echo ""
echo "SAM2 3D fix test completed with exit code: $exit_code"
echo "End time: $(date)"

if [ $exit_code -eq 0 ]; then
    echo "✅ SAM2 3D FIX SUCCESSFUL"
    echo ""
    echo "Key improvements:"
    echo "  - Works with 5D tensors (3D medical volumes)" 
    echo "  - Uses 3D pooling instead of 2D image operations"
    echo "  - Memory-efficient hierarchical attention"
    echo "  - Pool size 4: 64x memory reduction (946K → 14K sequence length)"
    echo ""
    echo "Next steps:"
    echo "1. Integrate into smart_sota_2025_fixed.py"
    echo "2. Test complete model building"
    echo "3. Run full SOTA training"
else
    echo "❌ SAM2 3D FIX FAILED"
    echo "Check error log for details"
fi

echo ""
echo "Final GPU state:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
