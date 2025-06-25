#!/bin/bash
#SBATCH --job-name=test_training
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a4500:1
#SBATCH --time=0:20:00
#SBATCH --output=../logs/test_training_%j.out
#SBATCH --error=../logs/test_training_%j.err

echo "🔍 TRAINING VS TEST CASE ANALYSIS"
echo "================================="
echo "Purpose: Check if model works on training data"
echo "Question: Does the model work on cases it was trained on?"
echo "Expected outcomes:"
echo "  A) Works on training → Train/test data mismatch issue"
echo "  B) Fails on training → Fundamental model problem"
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

# Run the training vs test comparison
echo "🚀 Testing model on training cases vs test cases..."
python -u test_on_training_case.py

exit_code=$?

echo ""
echo "============================================================"
echo "TRAINING VS TEST ANALYSIS COMPLETED"
echo "============================================================"
echo "Exit code: $exit_code"
echo "End time: $(date)"

if [ $exit_code -eq 0 ]; then
    echo "✅ ANALYSIS SUCCESSFUL - Root cause identified!"
    echo ""
    echo "Check the output above for:"
    echo "  - Training case performance vs test case performance"
    echo "  - Diagnosis of the core issue"
    echo "  - Specific next steps"
    
else
    echo "❌ ANALYSIS REVEALED FUNDAMENTAL MODEL ISSUES"
    echo "Model may need complete retraining"
fi

echo ""
echo "Final GPU status:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
