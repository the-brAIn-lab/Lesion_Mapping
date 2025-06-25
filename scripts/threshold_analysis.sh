#!/bin/bash
#SBATCH --job-name=threshold_analysis
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a4500:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/threshold_analysis_%j.out
#SBATCH --error=logs/threshold_analysis_%j.err

echo "Threshold Analysis - Finding Optimal Threshold"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"

cd /mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota

# Setup environment (exact same as working verbose test)
module load gcc/9.3.0-5wu3 cuda/12.6.3-ziu7
eval "$(conda shell.bash hook)" || true
conda activate tf215_env
export LD_LIBRARY_PATH="/mnt/beegfs/hellgate/home/rb194958e/.conda/envs/tf215_env/lib:$LD_LIBRARY_PATH"
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0
export TF_GPU_ALLOCATOR=cuda_malloc_async

echo ""
echo "Environment setup complete"
echo "Python: $(python --version)"
echo "TensorFlow: $(python -c 'import tensorflow as tf; print(tf.__version__)')"

echo ""
echo "Running threshold analysis..."
echo "=============================================="

# Run with unbuffered output
python -u modified_verbose_threshold_test.py

exit_code=$?

echo ""
echo "=============================================="
echo "Threshold analysis completed"
echo "Exit code: $exit_code"
echo "End time: $(date)"

if [ $exit_code -eq 0 ]; then
    echo "✅ ANALYSIS SUCCESSFUL"
    
    if [ -d "raw_probability_analysis" ]; then
        echo ""
        echo "Output files created:"
        ls -lh raw_probability_analysis/
        
        echo ""
        echo "Analysis results:"
        if [ -f "raw_probability_analysis/sub-r048s014_ses-1_threshold_analysis.txt" ]; then
            echo "--- Threshold Analysis Report ---"
            cat raw_probability_analysis/sub-r048s014_ses-1_threshold_analysis.txt
        fi
    fi
else
    echo "❌ ANALYSIS FAILED"
fi

echo ""
echo "Final GPU state:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
