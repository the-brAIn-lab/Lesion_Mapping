#!/bin/bash
#SBATCH --job-name=smart_sota_2025_debug
#SBATCH --partition=interactive
#SBATCH --gres=gpu:a4500:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=24:00:00  
#SBATCH --output=logs/smart_sota_%j.out
#SBATCH --error=logs/smart_sota_%j.err

# Set stricter memory limits
ulimit -v $((280 * 1024 * 1024))  # 280GB hard limit (20GB buffer)

# ========================
# 1. ENHANCED RESOURCE VERIFICATION
# ========================
echo "=== SLURM RESOURCE CHECK ==="
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOBID"
echo "Start time: $(date)"
echo "Allocated GPUs: $SLURM_GPUS"
echo "Allocated CPUs: $SLURM_CPUS_ON_NODE"
echo "Allocated memory: $SLURM_MEM_PER_NODE MB"
echo "Available memory: $(free -h)"
nvidia-smi -L
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv

# ========================
# 2. MEMORY MONITORING SETUP
# ========================
# Start background memory monitoring
monitor_memory() {
    while true; do
        echo "$(date): CPU Memory: $(free -h | grep '^Mem:' | awk '{print $3"/"$2}')"
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd',' - | sed 's/^/GPU Memory: /'
        sleep 30
    done
}
monitor_memory > logs/memory_usage_${SLURM_JOBID}.log &
MONITOR_PID=$!

# Function to cleanup on exit
cleanup() {
    echo "Cleaning up at: $(date)"
    kill $MONITOR_PID 2>/dev/null
    wait $MONITOR_PID 2>/dev/null
}
trap cleanup EXIT

# ========================
# 3. ENVIRONMENT SETUP
# ========================
echo "=== ENVIRONMENT SETUP ==="
module purge
module load gcc/9.3.0-5wu3 cuda/12.6.3-ziu7

# Set CUDA environment
export CUDA_VISIBLE_DEVICES=$(echo $SLURM_JOB_GPUS | sed 's/[^0-9,]//g')
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=1

# Suppress CUDA warnings (they're harmless)
export TF_CPP_MIN_LOG_LEVEL=2

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Python environment
eval "$(conda shell.bash hook)"
conda activate tf215_env

# ========================
# 4. PYTHON ENVIRONMENT TEST
# ========================
echo "=== PYTHON ENVIRONMENT TEST ==="
python -c "
import sys, psutil, tensorflow as tf
print(f'Python: {sys.version}')
print(f'TensorFlow: {tf.__version__}')
print(f'Initial CPU Memory: {psutil.virtual_memory().used/1024**3:.2f}GB')
gpus = tf.config.list_physical_devices('GPU')
print(f'Visible GPUs: {len(gpus)}')
for i, gpu in enumerate(gpus):
    tf.config.experimental.set_memory_growth(gpu, True)
    print(f'GPU {i}: {gpu}')
print('✅ Environment check passed')
"

if [ $? -ne 0 ]; then
    echo "❌ Environment test failed"
    exit 1
fi

# ========================
# 5. DATA DIRECTORY CHECK
# ========================
echo "=== DATA DIRECTORY CHECK ==="
DATA_DIR="/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training"
if [ -d "$DATA_DIR" ]; then
    echo "✅ Data directory exists: $DATA_DIR"
    echo "Images directory:"
    ls -la "$DATA_DIR/Images/" 2>/dev/null | head -5
    echo "Masks directory:"
    ls -la "$DATA_DIR/Masks/" 2>/dev/null | head -5
    
    # Count files
    IMG_COUNT=$(find "$DATA_DIR/Images/" -name "*.nii.gz" 2>/dev/null | wc -l)
    MASK_COUNT=$(find "$DATA_DIR/Masks/" -name "*.nii.gz" 2>/dev/null | wc -l)
    echo "Found $IMG_COUNT image files and $MASK_COUNT mask files"
    
    if [ $IMG_COUNT -eq 0 ] || [ $MASK_COUNT -eq 0 ]; then
        echo "❌ No data files found! Training cannot proceed."
        echo "Image patterns to check: *_T1w.nii.gz, *_t1.nii.gz, *T1w.nii.gz, *t1w.nii.gz"
        echo "Mask patterns to check: *_mask.nii.gz, *_lesion.nii.gz, *_label-L*.nii.gz"
        exit 1
    fi
else
    echo "❌ Data directory not found: $DATA_DIR"
    exit 1
fi

# ========================
# 6. SCRIPT EXISTENCE CHECK
# ========================
echo "=== SCRIPT CHECK ==="
if [ -f "smart_sota_2025_claude.py" ]; then
    echo "✅ Training script found"
    echo "Script size: $(du -h smart_sota_2025_claude.py)"
    echo "Script permissions: $(ls -la smart_sota_2025_claude.py)"
else
    echo "❌ Training script not found: smart_sota_2025_claude.py"
    echo "Current directory: $(pwd)"
    echo "Available Python files:"
    ls -la *.py 2>/dev/null || echo "No Python files found"
    exit 1
fi

# ========================
# 7. PRE-TRAINING MEMORY CHECK
# ========================
echo "=== PRE-TRAINING MEMORY CHECK ==="
free -h
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

# ========================
# 8. MAIN TRAINING RUN WITH ENHANCED ERROR HANDLING
# ========================
echo "=== STARTING TRAINING ==="
echo "Training start time: $(date)"

# Run with verbose output and capture exit code
set -o pipefail
python -u smart_sota_2025_claude.py 2>&1 | tee logs/training_output_${SLURM_JOBID}.log
PYTHON_EXIT_CODE=${PIPESTATUS[0]}

echo "Training end time: $(date)"
echo "Python exit code: $PYTHON_EXIT_CODE"

# ========================
# 9. IMMEDIATE POST-RUN ANALYSIS
# ========================
echo "=== IMMEDIATE POST-RUN ANALYSIS ==="

# Check if training actually started
echo "Checking if training started..."
if grep -q "🚀 Starting training" logs/training_output_${SLURM_JOBID}.log; then
    echo "✅ Training initialization detected"
else
    echo "❌ Training may not have started properly"
fi

# Check for specific error patterns
echo "Checking for common errors..."
if grep -q "FileNotFoundError" logs/training_output_${SLURM_JOBID}.log; then
    echo "❌ File not found error detected"
    grep "FileNotFoundError" logs/training_output_${SLURM_JOBID}.log
fi

if grep -q "CUDA_ERROR_OUT_OF_MEMORY\|ResourceExhaustedError" logs/training_output_${SLURM_JOBID}.log; then
    echo "❌ GPU memory error detected"
    grep -A5 -B5 "CUDA_ERROR_OUT_OF_MEMORY\|ResourceExhaustedError" logs/training_output_${SLURM_JOBID}.log
fi

if grep -q "ImportError\|ModuleNotFoundError" logs/training_output_${SLURM_JOBID}.log; then
    echo "❌ Import error detected"
    grep "ImportError\|ModuleNotFoundError" logs/training_output_${SLURM_JOBID}.log
fi

# Check for successful epochs
EPOCH_COUNT=$(grep -c "Epoch.*:" logs/training_output_${SLURM_JOBID}.log || echo "0")
echo "Number of training epochs detected: $EPOCH_COUNT"

# ========================
# 10. DETAILED ERROR ANALYSIS
# ========================
echo "=== DETAILED ERROR ANALYSIS ==="

if [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo "❌ Training failed with exit code: $PYTHON_EXIT_CODE"
    
    # Show last 100 lines of output for debugging
    echo "Last 100 lines of training output:"
    tail -100 logs/training_output_${SLURM_JOBID}.log
    
    echo "Last 50 lines of error log:"
    tail -50 logs/smart_sota_${SLURM_JOBID}.err
    
    echo "Checking for specific error patterns:"
    grep -i "error\|exception\|failed\|traceback" logs/smart_sota_${SLURM_JOBID}.err | tail -20
    
elif [ $EPOCH_COUNT -eq 0 ]; then
    echo "⚠️ Training completed but no epochs detected - possible early exit"
    echo "Last 50 lines of output:"
    tail -50 logs/training_output_${SLURM_JOBID}.log
    
else
    echo "✅ Training appears to have run successfully with $EPOCH_COUNT epochs"
fi

# ========================
# 11. SYSTEM RESOURCE ANALYSIS
# ========================
echo "=== FINAL SYSTEM STATE ==="
echo "Final memory usage:"
free -h
nvidia-smi --query-gpu=memory.used --format=csv

# Check for OOM in system logs
echo "Checking for OOM events:"
dmesg | grep -i "out of memory\|oom\|killed process" | tail -10

# SLURM memory accounting
echo "SLURM memory accounting:"
sacct -j $SLURM_JOBID --format=JobID,ReqMem,MaxRSS,Elapsed,State -n

# ========================
# 12. FILE OUTPUT CHECK
# ========================
echo "=== OUTPUT FILE CHECK ==="
echo "Checking for generated files..."

# Look for model files
if [ -d "models" ]; then
    echo "Model directory contents:"
    find models -name "*.keras" -o -name "*.h5" -o -name "*.json" | head -10
fi

# Look for log files
if [ -d "logs" ]; then
    echo "Log files generated:"
    ls -la logs/ | grep $SLURM_JOBID
fi

# Look for any crash reports
if [ -f "detailed_crash_report.json" ]; then
    echo "⚠️ Crash report found:"
    head -20 detailed_crash_report.json
fi

# ========================
# 13. SUMMARY REPORT
# ========================
echo "=== FINAL SUMMARY ==="
ELAPSED_TIME=$(sacct -j $SLURM_JOBID --format=Elapsed -n | head -1 | tr -d ' ')
echo "Job duration: $ELAPSED_TIME"

if [ $PYTHON_EXIT_CODE -eq 0 ] && [ $EPOCH_COUNT -gt 0 ]; then
    echo "✅ Training completed successfully with $EPOCH_COUNT epochs"
elif [ $PYTHON_EXIT_CODE -eq 0 ] && [ $EPOCH_COUNT -eq 0 ]; then
    echo "⚠️ Script completed but no training epochs detected - investigate logs"
else
    echo "❌ Training failed - check error logs above"
fi

echo "Job completed at: $(date)"
