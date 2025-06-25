#!/bin/bash
#SBATCH --job-name=raw_prob_test
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a4500:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/raw_prob_%j.out
#SBATCH --error=logs/raw_prob_%j.err

echo "Raw Probability Analysis"
echo "======================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"

cd /mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota

# Setup environment (same as working verbose test)
module load gcc/9.3.0-5wu3 cuda/12.6.3-ziu7
eval "$(conda shell.bash hook)" || true
conda activate tf215_env
export LD_LIBRARY_PATH="/mnt/beegfs/hellgate/home/rb194958e/.conda/envs/tf215_env/lib:$LD_LIBRARY_PATH"
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0
export TF_GPU_ALLOCATOR=cuda_malloc_async

echo ""
echo "Environment:"
echo "Python: $(python --version)"
echo "TensorFlow: $(python -c 'import tensorflow as tf; print(tf.__version__)')"

echo ""
echo "Running raw probability analysis..."

# Run with unbuffered output
python -u raw_probability_test.py

exit_code=$?

echo ""
echo "Analysis completed with exit code: $exit_code"
echo "End time: $(date)"

if [ $exit_code -eq 0 ]; then
    echo "✅ SUCCESS"
    
    if [ -d "raw_probability_output" ]; then
        echo ""
        echo "Output files:"
        ls -lh raw_probability_output/
    fi
else
    echo "❌ FAILED"
fi
