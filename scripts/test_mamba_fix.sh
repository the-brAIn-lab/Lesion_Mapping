#!/bin/bash
#SBATCH --job-name=test_mamba_fix
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=0:30:00
#SBATCH --output=logs/test_mamba_fix_%j.out
#SBATCH --error=logs/test_mamba_fix_%j.err

echo "🔧 TESTING VISION MAMBA FIX"
echo "============================"
echo "Purpose: Test if the fixed Vision Mamba can build a model without errors"
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"

cd /mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota

# Setup environment
module load gcc/9.3.0-5wu3 cuda/12.6.3-ziu7
eval "$(conda shell.bash hook)" || true
conda activate tf215_env
export LD_LIBRARY_PATH="/mnt/beegfs/hellgate/home/rb194958e/.conda/envs/tf215_env/lib:$LD_LIBRARY_PATH"
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0

echo ""
echo "Testing if Smart SOTA 2025 can build model after Mamba fix..."

# Create a test script to just build the model (not train)
cat > test_mamba_model_build.py << 'EOF'
import os
import sys
import tensorflow as tf

# Add current directory to Python path
sys.path.append('/mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota')

# Import your fixed Smart SOTA model
try:
    from smart_sota_2025 import build_smart_sota_2025_model
    print("✅ Successfully imported Smart SOTA 2025 model")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Configure GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Test model building
try:
    print("Building Smart SOTA 2025 model...")
    model = build_smart_sota_2025_model(
        input_shape=(192, 224, 176, 1),
        base_filters=22
    )
    
    param_count = model.count_params()
    print(f"✅ Model built successfully!")
    print(f"Parameters: {param_count:,}")
    
    # Test a forward pass with dummy data
    print("Testing forward pass...")
    dummy_input = tf.random.normal((1, 192, 224, 176, 1), dtype=tf.float16)
    output = model(dummy_input, training=False)
    print(f"✅ Forward pass successful: {dummy_input.shape} → {output.shape}")
    
    print("🎉 MAMBA FIX VERIFICATION SUCCESSFUL!")
    
except Exception as e:
    print(f"❌ Model building/testing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

# Run the test
python -u test_mamba_model_build.py

exit_code=$?

echo ""
echo "Test completed with exit code: $exit_code"
echo "End time: $(date)"

if [ $exit_code -eq 0 ]; then
    echo "✅ MAMBA FIX VERIFICATION SUCCESSFUL"
    echo ""
    echo "Your Smart SOTA 2025 should now work!"
    echo "Next step: Run the full training with the fixed model"
else
    echo "❌ MAMBA FIX STILL HAS ISSUES"
    echo "Check the error log for details"
fi
