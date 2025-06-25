#!/bin/bash
#SBATCH --job-name=test_new_model
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a4500:1
#SBATCH --time=0:20:00
#SBATCH --output=../logs/test_new_model_%j.out
#SBATCH --error=../logs/test_new_model_%j.err

echo "🔍 OLD vs NEW MODEL COMPARISON"
echo "=============================="
echo "Purpose: Test if retraining fixed the size bias"
echo "Strategy: Compare OLD and NEW models on small + large lesions"
echo ""
echo "Expected results:"
echo "  SMALL LESION:"
echo "    OLD model: Dice ≈ 0.0 (size bias failure)"
echo "    NEW model: Dice > 0.3 (size bias fixed!)"
echo ""
echo "  LARGE LESION:"  
echo "    OLD model: Dice > 0.5 (worked before)"
echo "    NEW model: Dice > 0.5 (should still work)"
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

# Verify models exist
echo "Checking model files..."
echo "OLD model: $(ls -la callbacks/sota_fixed_20250619_063330/best_model.h5 2>/dev/null || echo 'NOT FOUND')"
echo "NEW model: $(ls -la callbacks/full_retrain_20250622_074312/best_model.h5 2>/dev/null || echo 'NOT FOUND')"
echo ""

# Run comparison
echo "🚀 Running OLD vs NEW model comparison..."
python -u test_new_model.py

exit_code=$?

echo ""
echo "============================================================"
echo "MODEL COMPARISON COMPLETED"
echo "============================================================"
echo "Exit code: $exit_code"
echo "End time: $(date)"

if [ $exit_code -eq 0 ]; then
    echo "✅ COMPARISON SUCCESSFUL!"
    echo ""
    echo "Key results to check:"
    echo "  - Did NEW model improve on small lesions?"
    echo "  - Did NEW model maintain performance on large lesions?"
    echo "  - Is size bias fixed?"
    
else
    echo "❌ COMPARISON FAILED"
    echo "Check error log for details"
fi

echo ""
echo "Final GPU status:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
