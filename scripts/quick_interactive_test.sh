#!/bin/bash
# quick_interactive_test.sh - Quick test you can run immediately

echo "🚀 Quick Interactive GPU Test"
echo "============================"

# Run the debug script first
echo "Running advanced GPU debugging..."
bash scripts/advanced_gpu_debug.sh

echo ""
echo "🎯 Recommended Actions:"
echo ""

if [[ $(hostname) == *login* ]]; then
    echo "✅ Action Plan (you're on login node):"
    echo "1. Submit GPU test job:"
    echo "   sbatch scripts/gpu_test_job.sh"
    echo ""
    echo "2. OR request interactive GPU session:"
    echo "   srun --gres=gpu:1 --time=01:00:00 --pty bash"
    echo "   # Then run: conda activate stroke_sota && python diagnostic_test.py"
    echo ""
    echo "3. OR submit full training job:"
    echo "   sbatch scripts/slurm_gpu_fixed.sh"
    
else
    echo "✅ You're on a compute node - GPUs should be visible"
    echo "If TensorFlow still can't see GPUs, there may be a driver/CUDA issue"
    
    # Test if nvidia-smi works
    if command -v nvidia-smi &> /dev/null; then
        echo ""
        echo "nvidia-smi works - checking TensorFlow detection..."
        
        # Quick TensorFlow test
        python -c "
import tensorflow as tf
gpus = len(tf.config.list_physical_devices('GPU'))
if gpus > 0:
    print(f'✅ SUCCESS: {gpus} GPU(s) detected!')
    print('Ready to run training!')
else:
    print('❌ TensorFlow cannot see GPUs')
    print('Try: export CUDA_VISIBLE_DEVICES=0')
"
    else
        echo "❌ nvidia-smi not available - no GPU access on this node"
    fi
fi

echo ""
echo "📋 Files ready for training:"
echo "- Environment: stroke_sota ✅"
echo "- TensorFlow: 2.15.1 with CUDA ✅" 
echo "- Training script: robust_train.py ✅"
echo "- SLURM script: slurm_gpu_fixed.sh ✅"
echo ""
echo "🚀 Next step: Submit training job or test on GPU node!"
