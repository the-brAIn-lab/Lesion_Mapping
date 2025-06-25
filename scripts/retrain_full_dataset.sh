#!/bin/bash
#SBATCH --job-name=retrain_full
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a4500:1
#SBATCH --time=24:00:00
#SBATCH --output=../logs/retrain_full_%j.out
#SBATCH --error=../logs/retrain_full_%j.err

echo "🔄 RETRAINING ON FULL DATASET"
echo "============================="
echo "Purpose: Fix size bias by training on diverse lesion sizes"
echo "Key improvements:"
echo "  - FULL dataset (Training + Testing splits combined)"
echo "  - Proper validation split (20%)"
echo "  - Batch size 2 with epoch shuffling"
echo "  - Same architecture as working model"
echo ""
echo "Expected results:"
echo "  - Works on both large AND small lesions"
echo "  - Better generalization"
echo "  - Validation Dice 0.4-0.6"
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
export NIBABEL_NIFTI1_QFAC_CHECK=0

echo "Environment configured"
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
echo ""

# Check data availability
echo "Checking data availability..."
echo "Training split: $(ls /mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training_Split/Images/*.nii.gz | wc -l) images"
echo "Testing split: $(ls /mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Testing_Split/Images/*.nii.gz | wc -l) images"
echo ""

# Run retraining
echo "🚀 Starting full dataset retraining..."
python -u retrain_full_dataset.py

exit_code=$?

echo ""
echo "============================================================"
echo "FULL DATASET RETRAINING COMPLETED"
echo "============================================================"
echo "Exit code: $exit_code"
echo "End time: $(date)"

if [ $exit_code -eq 0 ]; then
    echo "✅ RETRAINING SUCCESSFUL!"
    echo ""
    echo "New model should work on:"
    echo "  - Large lesions (like original)"
    echo "  - Small lesions (previously failed)"
    echo "  - All test cases"
    echo ""
    echo "Next steps:"
    echo "1. Test new model on small lesion case"
    echo "2. Run batch testing on all test cases"
    echo "3. Compare with original model performance"
    
    # Show where the model was saved
    echo ""
    echo "Model locations:"
    ls -la callbacks/full_retrain_*/best_model.h5 2>/dev/null || echo "Check logs for model path"
    ls -la models/full_retrain_*.h5 2>/dev/null || echo "Check logs for model path"
    
else
    echo "❌ RETRAINING FAILED"
    echo "Check error log and training logs for details"
fi

echo ""
echo "Final GPU status:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
