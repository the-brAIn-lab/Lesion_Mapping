#!/bin/bash
#SBATCH --job-name=exact_preprocessing
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a4500:1
#SBATCH --time=0:15:00
#SBATCH --output=../logs/exact_preprocessing_%j.out
#SBATCH --error=../logs/exact_preprocessing_%j.err

echo "🔧 EXACT TRAINING PREPROCESSING TEST"
echo "===================================="
echo "Purpose: Use exact same preprocessing as training"
echo "Problem: Model works on training (Dice=0.77) but fails on test (Dice=0.0)"
echo "Solution: Test different preprocessing methods to match training"
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

# Run the exact preprocessing test
echo "🚀 Testing different preprocessing methods..."
python -u test_exact_training_preprocessing.py

exit_code=$?

echo ""
echo "============================================================"
echo "EXACT PREPROCESSING TEST COMPLETED"
echo "============================================================"
echo "Exit code: $exit_code"
echo "End time: $(date)"

if [ $exit_code -eq 0 ]; then
    echo "✅ PREPROCESSING TEST SUCCESSFUL!"
    echo ""
    echo "Found the correct preprocessing method!"
    echo "Next: Implement this method in batch testing"
    
else
    echo "❌ PREPROCESSING TEST UNSUCCESSFUL"
    echo "May need to check training code directly"
fi

echo ""
echo "Final GPU status:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
