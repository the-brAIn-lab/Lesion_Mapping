#!/bin/bash
#SBATCH --job-name=debug_multi_gpu
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/debug_multi_gpu_%j.out
#SBATCH --error=logs/debug_multi_gpu_%j.err

echo "🔍 DEBUGGING MULTI-GPU SCRIPT IMPORT ISSUES"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo ""

cd /mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota

# Setup environment
module load gcc/9.3.0-5wu3 cuda/12.6.3-ziu7
eval "$(conda shell.bash hook)" || true
conda activate tf215_env
export LD_LIBRARY_PATH="/mnt/beegfs/hellgate/home/rb194958e/.conda/envs/tf215_env/lib:$LD_LIBRARY_PATH"
export TF_ENABLE_ONEDNN_OPTS=0

echo "✅ Environment activated"
echo ""

echo "🧪 TESTING BASIC IMPORTS:"
python -c "
import sys
print(f'Python version: {sys.version}')

try:
    import tensorflow as tf
    print(f'✅ TensorFlow: {tf.__version__}')
except Exception as e:
    print(f'❌ TensorFlow error: {e}')

try:
    from tensorflow.keras import layers, Model
    print('✅ Keras layers import OK')
except Exception as e:
    print(f'❌ Keras layers error: {e}')

try:
    from sklearn.model_selection import train_test_split
    print('✅ sklearn import OK')
except Exception as e:
    print(f'❌ sklearn error: {e}')

try:
    import numpy as np
    print('✅ numpy import OK')
except Exception as e:
    print(f'❌ numpy error: {e}')

try:
    from correct_full_training import load_full_655_dataset
    print('✅ baseline script import OK')
except Exception as e:
    print(f'❌ baseline script error: {e}')
    import traceback
    traceback.print_exc()
"

echo ""
echo "🔍 TESTING MULTI-GPU SCRIPT IMPORT:"
python -c "
try:
    print('Attempting to import multi_gpu_advanced_sota...')
    import multi_gpu_advanced_sota
    print('✅ Multi-GPU script imports successfully!')
    
    # Test key functions
    print('Testing key functions...')
    from multi_gpu_advanced_sota import configure_multi_gpu_strategy
    print('✅ configure_multi_gpu_strategy import OK')
    
    from multi_gpu_advanced_sota import build_massive_sota_model
    print('✅ build_massive_sota_model import OK')
    
    from multi_gpu_advanced_sota import ultimate_loss
    print('✅ ultimate_loss import OK')
    
except Exception as e:
    print(f'❌ Multi-GPU script import error: {e}')
    import traceback
    traceback.print_exc()
"

echo ""
echo "🔧 CHECKING FOR SPECIFIC ERRORS:"
echo "Checking if deep_supervision_loss is still being called..."
grep -n "deep_supervision_loss" multi_gpu_advanced_sota.py

echo ""
echo "Checking USE_DEEP_SUPERVISION settings..."
grep -n "USE_DEEP_SUPERVISION" multi_gpu_advanced_sota.py

echo ""
echo "🧪 TESTING MODEL BUILDING:"
python -c "
try:
    import tensorflow as tf
    from multi_gpu_advanced_sota import build_massive_sota_model, MultiGPUAdvancedConfig
    
    print('Testing model building with small size...')
    model = build_massive_sota_model(
        input_shape=(32, 32, 32, 1), 
        base_filters=8  # Very small for testing
    )
    print(f'✅ Model builds successfully: {model.count_params():,} parameters')
    
except Exception as e:
    print(f'❌ Model building error: {e}')
    import traceback
    traceback.print_exc()
"

echo ""
echo "🔍 TESTING TENSORFLOW DISTRIBUTION:"
python -c "
try:
    import tensorflow as tf
    
    print('Testing MirroredStrategy...')
    strategy = tf.distribute.MirroredStrategy()
    print(f'✅ MirroredStrategy works: {strategy.num_replicas_in_sync} devices')
    
except Exception as e:
    print(f'❌ MirroredStrategy error: {e}')
    import traceback
    traceback.print_exc()
"

echo ""
echo "📁 CHECKING FILE EXISTENCE:"
echo "Checking if all required files exist..."
ls -la multi_gpu_advanced_sota.py
ls -la correct_full_training.py
ls -la scripts/massive_multi_gpu_sota.sh

echo ""
echo "🔍 FINAL DIAGNOSIS:"
echo "If we got here without errors, the script should work."
echo "If there were errors above, those need to be fixed."

echo ""
echo "============================================"
echo "Debug completed: $(date)"
echo "Check the output above for specific errors."
echo "============================================"
