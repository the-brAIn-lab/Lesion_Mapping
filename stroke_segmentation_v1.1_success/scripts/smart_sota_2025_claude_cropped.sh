#!/bin/bash
#SBATCH --job-name=smart_sota_cropped_2025
#SBATCH --partition=interactive
#SBATCH --gres=gpu:a4500:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=24:00:00  
#SBATCH --output=logs/smart_sota_cropped_%j.out
#SBATCH --error=logs/smart_sota_cropped_%j.err

# Set stricter memory limits
ulimit -v $((280 * 1024 * 1024))  # 280GB hard limit (20GB buffer)

# ========================
# 1. ENHANCED RESOURCE VERIFICATION
# ========================
echo "=== SLURM RESOURCE CHECK (CROPPED DATASET) ==="
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
monitor_memory > logs/memory_usage_cropped_${SLURM_JOBID}.log &
MONITOR_PID=$!

# Function to cleanup on exit
cleanup() {
    echo "Cleaning up cropped training at: $(date)"
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
# 5. CROPPED DATA DIRECTORY CHECK
# ========================
echo "=== CROPPED DATA DIRECTORY CHECK ==="
CROPPED_DATA_DIR="/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training/Cropped_128_Combined"
if [ -d "$CROPPED_DATA_DIR" ]; then
    echo "✅ Cropped data directory exists: $CROPPED_DATA_DIR"
    echo "Directory structure:"
    ls -la "$CROPPED_DATA_DIR/" 2>/dev/null | head -10
    
    # Count all .nii.gz files
    TOTAL_NII_COUNT=$(find "$CROPPED_DATA_DIR/" -name "*.nii.gz" 2>/dev/null | wc -l)
    echo "Total .nii.gz files found: $TOTAL_NII_COUNT"
    
    # Count images and masks separately
    IMG_COUNT=$(find "$CROPPED_DATA_DIR/" -name "*_T1w_cropped128.nii.gz" 2>/dev/null | wc -l)
    MASK_COUNT=$(find "$CROPPED_DATA_DIR/" -name "*_mask_cropped128.nii.gz" 2>/dev/null | wc -l)
    echo "Found $IMG_COUNT cropped image files"
    echo "Found $MASK_COUNT cropped mask files"
    
    if [ $TOTAL_NII_COUNT -eq 0 ]; then
        echo "❌ No .nii.gz files found! Training cannot proceed."
        echo "Expected files like: *_T1w_cropped128.nii.gz and *_mask_cropped128.nii.gz"
        exit 1
    else
        echo "✅ Cropped dataset appears to contain $TOTAL_NII_COUNT data files"
        echo "Expected: 1310 files (655 images + 655 masks)"
    fi
    
    # Show a few sample files for verification
    echo "Sample files found:"
    find "$CROPPED_DATA_DIR/" -name "*_T1w_cropped128.nii.gz" 2>/dev/null | head -3 | while read file; do
        echo "  $(basename "$file")"
    done
    find "$CROPPED_DATA_DIR/" -name "*_mask_cropped128.nii.gz" 2>/dev/null | head -3 | while read file; do
        echo "  $(basename "$file")"
    done
    
else
    echo "❌ Cropped data directory not found: $CROPPED_DATA_DIR"
    echo "Available directories in Training:"
    ls -la "/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training/" 2>/dev/null || echo "Training directory not accessible"
    exit 1
fi

# ========================
# 6. CROPPED TRAINING SCRIPT CHECK
# ========================
echo "=== CROPPED TRAINING SCRIPT CHECK ==="
SCRIPT_NAME="smart_sota_2025_claude_cropped.py"
if [ -f "$SCRIPT_NAME" ]; then
    echo "✅ Cropped training script found: $SCRIPT_NAME"
    echo "Script size: $(du -h $SCRIPT_NAME)"
    echo "Script permissions: $(ls -la $SCRIPT_NAME)"
    
    # Quick syntax check
    echo "Running Python syntax check..."
    python -m py_compile "$SCRIPT_NAME"
    if [ $? -eq 0 ]; then
        echo "✅ Python syntax check passed"
    else
        echo "❌ Python syntax check failed"
        exit 1
    fi
    
else
    echo "❌ Cropped training script not found: $SCRIPT_NAME"
    echo "Current directory: $(pwd)"
    echo "Available Python files:"
    ls -la *.py 2>/dev/null || echo "No Python files found"
    echo ""
    echo "Expected script name: $SCRIPT_NAME"
    echo "Please ensure the cropped training script is named correctly and present"
    exit 1
fi

# ========================
# 7. PRE-TRAINING MEMORY CHECK
# ========================
echo "=== PRE-TRAINING MEMORY CHECK ==="
free -h
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

# ========================
# 8. MAIN CROPPED DATASET TRAINING RUN
# ========================
echo "=== STARTING CROPPED DATASET TRAINING ==="
echo "Training start time: $(date)"
echo "Dataset: Cropped_655_for_training"
echo "Expected advantages: Smaller memory footprint, faster training, potentially higher batch size"

# Run with verbose output and capture exit code
set -o pipefail
python -u "$SCRIPT_NAME" 2>&1 | tee logs/training_output_cropped_${SLURM_JOBID}.log
PYTHON_EXIT_CODE=${PIPESTATUS[0]}

echo "Cropped dataset training end time: $(date)"
echo "Python exit code: $PYTHON_EXIT_CODE"

# ========================
# 9. IMMEDIATE POST-RUN ANALYSIS
# ========================
echo "=== IMMEDIATE POST-RUN ANALYSIS (CROPPED) ==="

# Check if training actually started
echo "Checking if cropped training started..."
if grep -q "🚀 Starting cropped dataset training" logs/training_output_cropped_${SLURM_JOBID}.log; then
    echo "✅ Cropped training initialization detected"
elif grep -q "🚀 Starting training" logs/training_output_cropped_${SLURM_JOBID}.log; then
    echo "✅ Training initialization detected (generic)"
else
    echo "❌ Training may not have started properly"
fi

# Check for data shape detection
echo "Checking for data shape auto-detection..."
if grep -q "💡 Optimal INPUT_SHAPE" logs/training_output_cropped_${SLURM_JOBID}.log; then
    echo "✅ Data shape auto-detection successful"
    grep "💡 Optimal INPUT_SHAPE" logs/training_output_cropped_${SLURM_JOBID}.log
elif grep -q "🔄 Updated config.INPUT_SHAPE" logs/training_output_cropped_${SLURM_JOBID}.log; then
    echo "✅ Input shape updated automatically"
    grep "🔄 Updated config.INPUT_SHAPE" logs/training_output_cropped_${SLURM_JOBID}.log
else
    echo "⚠️ No input shape auto-detection found"
fi

# Check for specific error patterns
echo "Checking for common errors..."
if grep -q "FileNotFoundError" logs/training_output_cropped_${SLURM_JOBID}.log; then
    echo "❌ File not found error detected"
    grep "FileNotFoundError" logs/training_output_cropped_${SLURM_JOBID}.log
fi

if grep -q "CUDA_ERROR_OUT_OF_MEMORY\|ResourceExhaustedError" logs/training_output_cropped_${SLURM_JOBID}.log; then
    echo "❌ GPU memory error detected"
    grep -A5 -B5 "CUDA_ERROR_OUT_OF_MEMORY\|ResourceExhaustedError" logs/training_output_cropped_${SLURM_JOBID}.log
fi

if grep -q "ImportError\|ModuleNotFoundError" logs/training_output_cropped_${SLURM_JOBID}.log; then
    echo "❌ Import error detected"
    grep "ImportError\|ModuleNotFoundError" logs/training_output_cropped_${SLURM_JOBID}.log
fi

# Check for successful epochs
EPOCH_COUNT=$(grep -c "Epoch.*:" logs/training_output_cropped_${SLURM_JOBID}.log || echo "0")
echo "Number of training epochs detected: $EPOCH_COUNT"

# Check for memory advantages of cropped data
echo "Checking for memory advantages..."
if grep -q "Increased since cropped data is smaller" logs/training_output_cropped_${SLURM_JOBID}.log; then
    echo "✅ Batch size optimization detected"
fi

# ========================
# 10. DETAILED ERROR ANALYSIS
# ========================
echo "=== DETAILED ERROR ANALYSIS (CROPPED) ==="

if [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo "❌ Cropped training failed with exit code: $PYTHON_EXIT_CODE"
    
    # Show last 100 lines of output for debugging
    echo "Last 100 lines of cropped training output:"
    tail -100 logs/training_output_cropped_${SLURM_JOBID}.log
    
    echo "Last 50 lines of error log:"
    tail -50 logs/smart_sota_cropped_${SLURM_JOBID}.err
    
    echo "Checking for specific error patterns:"
    grep -i "error\|exception\|failed\|traceback" logs/smart_sota_cropped_${SLURM_JOBID}.err | tail -20
    
elif [ $EPOCH_COUNT -eq 0 ]; then
    echo "⚠️ Cropped training completed but no epochs detected - possible early exit"
    echo "Last 50 lines of output:"
    tail -50 logs/training_output_cropped_${SLURM_JOBID}.log
    
else
    echo "✅ Cropped training appears to have run successfully with $EPOCH_COUNT epochs"
fi

# ========================
# 11. SYSTEM RESOURCE ANALYSIS
# ========================
echo "=== FINAL SYSTEM STATE (CROPPED) ==="
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
# 12. CROPPED MODEL OUTPUT CHECK
# ========================
echo "=== CROPPED MODEL OUTPUT CHECK ==="
echo "Checking for generated files..."

# Look for cropped model files
if [ -d "models/cropped_production" ]; then
    echo "Cropped model directory contents:"
    find models/cropped_production -name "*.keras" -o -name "*.h5" -o -name "*.json" | head -10
    echo "Model files found:"
    ls -la models/cropped_production/ | grep -E '\.(keras|h5|json)$'
fi

# Look for regular model files (fallback)
if [ -d "models" ]; then
    echo "General model directory contents:"
    find models -name "*cropped*" -o -name "*.keras" -o -name "*.h5" | head -10
fi

# Look for log files
if [ -d "logs" ]; then
    echo "Log files generated for cropped training:"
    ls -la logs/ | grep -E "(cropped|$SLURM_JOBID)"
fi

# Look for any crash reports
if [ -f "detailed_crash_report_cropped.json" ]; then
    echo "⚠️ Cropped training crash report found:"
    head -20 detailed_crash_report_cropped.json
fi

# ========================
# 13. CROPPED TRAINING SUMMARY REPORT
# ========================
echo "=== CROPPED TRAINING SUMMARY ==="
ELAPSED_TIME=$(sacct -j $SLURM_JOBID --format=Elapsed -n | head -1 | tr -d ' ')
echo "Job duration: $ELAPSED_TIME"
echo "Dataset: Cropped_655_for_training"

if [ $PYTHON_EXIT_CODE -eq 0 ] && [ $EPOCH_COUNT -gt 0 ]; then
    echo "✅ Cropped dataset training completed successfully with $EPOCH_COUNT epochs"
    echo "🎉 Advantages of cropped dataset likely realized:"
    echo "   - Smaller memory footprint"
    echo "   - Faster training iterations"
    echo "   - Potentially higher batch sizes"
elif [ $PYTHON_EXIT_CODE -eq 0 ] && [ $EPOCH_COUNT -eq 0 ]; then
    echo "⚠️ Script completed but no training epochs detected - investigate logs"
else
    echo "❌ Cropped training failed - check error logs above"
    echo "💡 Possible next steps:"
    echo "   - Check if cropped data directory structure is correct"
    echo "   - Verify .nii.gz files are valid"
    echo "   - Review data inspection output in logs"
fi

echo "Job completed at: $(date)"
echo "Cropped dataset location: $CROPPED_DATA_DIR"
