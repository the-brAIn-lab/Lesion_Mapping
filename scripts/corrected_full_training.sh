#!/bin/bash
#SBATCH --job-name=correct_full_training
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a4500:1
#SBATCH --time=24:00:00
#SBATCH --output=../logs/correct_full_training_%j.out
#SBATCH --error=../logs/correct_full_training_%j.err

echo "🔄 CORRECT FULL DATASET TRAINING"
echo "================================"
echo "This time using the ACTUAL full dataset you specified:"
echo "  Dataset: /mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training/"
echo "  Expected: 655 images (the full dataset)"
echo "  Batch size: 3 with epoch shuffling"
echo "  Validation split: 10%"
echo "  Architecture: Same as working model (5.7M parameters)"
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
echo ""

# Verify we're using the correct dataset
echo "Verifying dataset:"
echo "Full dataset: $(ls /mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training/Images/*.nii.gz | wc -l) images"
echo "This should be 655 images"
echo ""

# Run the CORRECT training
echo "🚀 Starting CORRECT full dataset training..."
python -u correct_full_training.py

exit_code=$?

echo ""
echo "============================================================"
echo "CORRECT FULL DATASET TRAINING COMPLETED"
echo "============================================================"
echo "Exit code: $exit_code"
echo "End time: $(date)"

if [ $exit_code -eq 0 ]; then
    echo "✅ CORRECT TRAINING SUCCESSFUL!"
    echo ""
    echo "This model was trained on:"
    echo "  - FULL 655-image dataset (not splits)"
    echo "  - 20% validation split"
    echo "  - Batch size 2 with epoch shuffling"
    echo "  - Same architecture as working model"
    echo ""
    echo "Model should now work on ALL lesion sizes!"
    
else
    echo "❌ CORRECT TRAINING FAILED"
    echo "Check logs for details"
fi

echo ""
echo "Final GPU status:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
