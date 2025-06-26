#!/bin/bash
# Activation script for stroke_sota environment

echo "🐍 Activating stroke_sota environment..."
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate stroke_sota

# Set library paths for GPU support
export LD_LIBRARY_PATH="/mnt/beegfs/hellgate/home/rb194958e/.conda/envs/stroke_sota/lib:$LD_LIBRARY_PATH"

# Verify environment
python -c "
import tensorflow as tf
print(f'Environment ready - TensorFlow {tf.__version__}')
gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs available: {len(gpus)}')
"

echo "✅ Environment activated and ready!"
