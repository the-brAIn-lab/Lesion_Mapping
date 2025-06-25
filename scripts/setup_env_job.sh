#!/bin/bash
#SBATCH --job-name=setup_tf_env
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/setup_env_%j.out
#SBATCH --error=logs/setup_env_%j.err

echo "🔧 Setting up TensorFlow Environment for Stroke Segmentation"
echo "=========================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Memory: 128GB"
echo "Time: $(date)"

# Navigate to project
cd /mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota

# Remove existing broken environment if it exists
echo "🗑️  Removing existing stroke_sota environment..."
conda env remove -n stroke_sota -y 2>/dev/null || echo "Environment didn't exist"

# Create fresh environment
echo "🆕 Creating new stroke_sota environment..."
conda create -n stroke_sota python=3.10 -y

# Activate environment
echo "🐍 Activating environment..."
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate stroke_sota

# Install CUDA and cuDNN libraries via conda first
echo "🔗 Installing CUDA and cuDNN libraries..."
conda install -n stroke_sota \
    cudnn=8.9 \
    cuda-toolkit=12.2 \
    numpy scipy scikit-image matplotlib h5py nibabel \
    -c nvidia \
    -c conda-forge \
    -y

# Install TensorFlow via pip (this ensures compatibility)
echo "📦 Installing TensorFlow..."
pip install tensorflow==2.15.1

# Install other required packages
echo "📚 Installing additional packages..."
pip install \
    einops \
    tqdm \
    pyyaml \
    opencv-python \
    scikit-learn

# Verify installation
echo "✅ Verifying installation..."
python -c "
import sys
print(f'Python: {sys.version}')

try:
    import tensorflow as tf
    print(f'TensorFlow: {tf.__version__}')
    print(f'Built with CUDA: {tf.test.is_built_with_cuda()}')
except ImportError as e:
    print(f'TensorFlow import failed: {e}')

try:
    import numpy as np
    import scipy
    import nibabel
    import einops
    print('✅ All packages imported successfully')
except ImportError as e:
    print(f'Package import failed: {e}')
"

# Test GPU detection (if on GPU node)
echo "🔬 Testing GPU detection..."
export LD_LIBRARY_PATH="/mnt/beegfs/hellgate/home/rb194958e/.conda/envs/stroke_sota/lib:$LD_LIBRARY_PATH"

python -c "
import tensorflow as tf
import os

print(f'LD_LIBRARY_PATH set: {bool(os.environ.get(\"LD_LIBRARY_PATH\"))}')

try:
    gpus = tf.config.list_physical_devices('GPU')
    print(f'GPUs detected: {len(gpus)}')
    
    if gpus:
        print('✅ GPU detection successful!')
        for i, gpu in enumerate(gpus):
            print(f'  GPU {i}: {gpu}')
    else:
        print('⚠️  No GPUs on this node (normal for CPU nodes)')
        
except Exception as e:
    print(f'GPU test error: {e}')
"

# Create environment activation script
echo "📄 Creating activation script..."
cat > activate_stroke_env.sh << 'EOF'
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
EOF

chmod +x activate_stroke_env.sh

echo ""
echo "🎉 Environment setup completed!"
echo ""
echo "📋 Summary:"
echo "- Environment: stroke_sota"
echo "- Python: 3.10"
echo "- TensorFlow: 2.15.1 with CUDA support"
echo "- Libraries: CUDA 12.2, cuDNN 8.9"
echo ""
echo "🚀 To use this environment:"
echo "1. source activate_stroke_env.sh"
echo "2. Or manually: conda activate stroke_sota && export LD_LIBRARY_PATH=..."
echo ""
echo "✅ Ready for GPU training!"

echo "Setup completed at: $(date)"
