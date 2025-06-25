#!/bin/bash
#SBATCH --job-name=verbose_test_sota
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a4500:1
#SBATCH --time=4:00:00
#SBATCH --output=logs/verbose_test_%j.out
#SBATCH --error=logs/verbose_test_%j.err

echo "Verbose SOTA Model Test - Exact Failure Detection"
echo "================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "GPUs: $SLURM_GPUS"
echo "Start: $(date)"
echo "User: $(whoami)"
echo "Working dir: $(pwd)"

# Show system info
echo ""
echo "SYSTEM INFORMATION:"
echo "==================="
echo "Hostname: $(hostname)"
echo "Kernel: $(uname -r)"
echo "CPU info:"
lscpu | head -20
echo ""
echo "Memory info:"
free -h
echo ""
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

cd /mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota

# Setup environment with verbose output
echo "ENVIRONMENT SETUP:"
echo "=================="
echo "Loading modules..."
module load gcc/9.3.0-5wu3 cuda/12.6.3-ziu7
echo "GCC version: $(gcc --version | head -1)"
echo "CUDA version: $(nvcc --version | grep release)"

echo ""
echo "Setting up conda..."
eval "$(conda shell.bash hook)" || true
conda activate tf215_env

echo "Python version: $(python --version)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# Set environment variables
export LD_LIBRARY_PATH="/mnt/beegfs/hellgate/home/rb194958e/.conda/envs/tf215_env/lib:$LD_LIBRARY_PATH"
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0
export TF_GPU_ALLOCATOR=cuda_malloc_async

echo ""
echo "Environment variables:"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "TF_ENABLE_ONEDNN_OPTS=$TF_ENABLE_ONEDNN_OPTS"
echo "TF_GPU_ALLOCATOR=$TF_GPU_ALLOCATOR"
echo "LD_LIBRARY_PATH (first part): $(echo $LD_LIBRARY_PATH | cut -d: -f1)"

# Check critical files
echo ""
echo "FILE CHECKS:"
echo "============"
echo "Model file:"
ls -lh callbacks/sota_20250616_190015/best_model.h5
echo ""
echo "Test image:"
ls -lh /mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Testing_Split/Images/sub-r048s014_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz
echo ""
echo "Test mask:"
ls -lh /mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Testing_Split/Masks/sub-r048s014_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz

# Check Python packages
echo ""
echo "PYTHON PACKAGES:"
echo "================"
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import numpy as np; print(f'NumPy: {np.__version__}')"
python -c "import scipy; print(f'SciPy: {scipy.__version__}')"
python -c "import nibabel as nib; print(f'NiBabel: {nib.__version__}')"

echo ""
echo "RUNNING VERBOSE TEST:"
echo "===================="
echo "Start time: $(date)"

# Run the verbose test with output buffering disabled
python -u verbose_single_test.py 2>&1

test_exit_code=$?

echo ""
echo "COMPLETION STATUS:"
echo "=================="
echo "Exit code: $test_exit_code"
echo "End time: $(date)"

if [ $test_exit_code -eq 0 ]; then
    echo "✅ TEST COMPLETED SUCCESSFULLY"
else
    echo "❌ TEST FAILED WITH EXIT CODE: $test_exit_code"
fi

# Final system state
echo ""
echo "FINAL SYSTEM STATE:"
echo "=================="
echo "Memory usage:"
free -h
echo ""
echo "GPU usage:"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv

# Check if any output files were created
echo ""
echo "OUTPUT FILES:"
echo "============="
if [ -d "verbose_test_output" ]; then
    echo "Output directory contents:"
    ls -lh verbose_test_output/
else
    echo "No output directory created"
fi

echo ""
echo "Job completed: $(date)"
