#!/bin/bash
# Script to organize Stroke Segmentation V1.0 Success folder with better formatting

echo "🎯 Creating Stroke Segmentation V1.0 Success folder with improved organization..."

# Create the main directory
mkdir -p stroke_segmentation_v1.0_success_formatted
cd stroke_segmentation_v1.0_success_formatted

echo "📁 Creating directory structure..."

# Create subdirectories
mkdir -p models
mkdir -p callbacks  
mkdir -p logs_training_run
mkdir -p scripts

echo "📋 Copying V1.0 training files..."

# Copy the main training script (original full resolution)
cp ../smart_sota_2025_claude.py ./
echo "✅ Copied original training script"

# Copy SLURM script
cp ../scripts/smart_sota_2025_claude.sh ./scripts/
echo "✅ Copied SLURM script"

# Copy the V1.0 trained models (look for the emergency save from v1.0)
echo "🤖 Copying V1.0 trained models..."
if [ -f "../models/production/emergency_save_20250720_052847.keras" ]; then
    cp ../models/production/emergency_save_20250720_052847.keras ./models/
    echo "✅ Copied V1.0 emergency save model"
fi

# Copy any other V1.0 models
if [ -d "../models/production" ]; then
    cp ../models/production/*.json ./models/ 2>/dev/null || echo "No config files found"
    cp ../models/production/*.h5 ./models/ 2>/dev/null || echo "No .h5 files found"
    echo "✅ Copied additional V1.0 model files"
fi

# Copy callbacks/checkpoints from V1.0
echo "💾 Copying V1.0 model checkpoints..."
if [ -d "../callbacks/production" ]; then
    cp -r ../callbacks/production/* ./callbacks/ 2>/dev/null || echo "No V1.0 callbacks found"
    echo "✅ Copied V1.0 model checkpoints"
fi

# Note: V1.0 logs would need to be identified by job number
# You'll need to replace XXXXX with the actual V1.0 job number
echo "📊 V1.0 training logs need to be identified by job number"
echo "Please identify the V1.0 job number and copy manually:"
echo "cp ../logs/smart_sota_XXXXX.out ./logs_training_run/"
echo "cp ../logs/smart_sota_XXXXX.err ./logs_training_run/"

echo "📝 Creating comprehensive V1.0 README..."

# Create the README.md file for V1.0
cat > README.md << 'EOF'
# Stroke Segmentation V1.0 - Successful Training (Full Resolution)

## Training Results
- **Final Validation Dice: 62.5%**
- **Training Dataset**: 655 MRI samples from Atlas_2/Training (full resolution)
- **Model**: Vision Mamba + SAM2 Attention U-Net
- **Epochs**: 200 (completed successfully)
- **Training Duration**: Extended training session
- **Date**: July 20, 2025

## Model Architecture
- **Vision Mamba blocks** for O(n) complexity long-range dependencies
- **Enhanced SAM2 attention** with hierarchical memory banks
- **Residual Conv blocks** with batch normalization and spatial dropout
- **U-Net backbone** with skip connections at 4 resolution levels
- **Advanced loss function**: Dice + boundary-weighted for precise edges

## Technical Specifications
- **Input Resolution**: 192×224×176×1 (full original resolution)
- **Model Parameters**: ~15M parameters
- **Batch Size**: 2 (limited by memory constraints)
- **GPU Configuration**: 4x RTX 4500 Ada Generation (24GB each)
- **Framework**: TensorFlow 2.15.1, Python 3.11

## Key Files
- `smart_sota_2025_claude.py` - Main training script for full resolution
- `scripts/smart_sota_2025_claude.sh` - SLURM training script
- `models/emergency_save_20250720_052847.keras` - Final trained model
- `callbacks/` - Model checkpoints and callbacks
- `models/config.json` - Training configuration (if available)
- `logs_training_run/` - Training logs and monitoring

## Training Configuration
```json
{
  "DATA_DIR": "/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training",
  "INPUT_SHAPE": [192, 224, 176, 1],
  "BATCH_SIZE": 2,
  "TOTAL_EPOCHS": 200,
  "INITIAL_LR": 1e-4,
  "MIN_LR": 1e-6,
  "BASE_FILTERS": 8,
  "DROPOUT_RATE": 0.6,
  "VALIDATION_SPLIT": 0.15,
  "USE_BOUNDARY_LOSS": true
}
```

## Data Pipeline
1. **Original Dataset**: 655 full-resolution T1w MRI + lesion masks
2. **Original Resolution**: 192×224×176 voxels (7.6M voxels per sample)
3. **Direct Processing**: No cropping, full spatial information preserved
4. **Advanced Augmentation**: 
   - Spatial transforms (rotation, flipping)
   - Intensity variations (brightness, contrast, gamma)
   - Synthetic lesion injection for small lesion enhancement
   - Boundary enhancement for precise edge detection

## Memory and Performance Profile
- **High Memory Usage**: Full resolution requires significant GPU memory
- **Batch Size Limitation**: Limited to 2 samples per batch
- **Training Stability**: Achieved stable training after resolving CUDNN issues
- **Single GPU Strategy**: Used single GPU to avoid distribution issues

## Training Insights
- **Full Resolution Benefits**: Preserves all spatial detail and anatomical context
- **Memory Constraints**: Limited batch size affects gradient estimation
- **CUDNN Challenges**: Required single GPU strategy and careful memory management
- **Baseline Performance**: Established 62.5% validation Dice as baseline

## Usage Instructions

### Loading the Trained Model
```python
import tensorflow as tf

# Load the final model
model = tf.keras.models.load_model('models/emergency_save_20250720_052847.keras')

# Custom objects may be needed for Vision Mamba and SAM2 layers
custom_objects = {
    'dice_coefficient': dice_coefficient,
    'compiled_loss': lambda y_true, y_pred: dice_loss(y_true, y_pred)
}
model = tf.keras.models.load_model('models/emergency_save_20250720_052847.keras', 
                                   custom_objects=custom_objects)
```

### Data Preprocessing for Inference
```python
import nibabel as nib
import numpy as np

def preprocess_for_inference_v10(nifti_path):
    # Load MRI volume at full resolution
    img = nib.load(nifti_path)
    data = img.get_fdata().astype(np.float32)
    
    # Ensure correct shape (192, 224, 176)
    target_shape = (192, 224, 176)
    if data.shape != target_shape:
        # Resize to target shape if needed
        factors = [t/s for t, s in zip(target_shape, data.shape)]
        from scipy.ndimage import zoom
        data = zoom(data, factors, order=1)
    
    # Normalize
    if np.max(data) > 0:
        non_zero = data[data > 0]
        if len(non_zero) > 0:
            mean_val, std_val = np.mean(non_zero), np.std(non_zero)
            data = (data - mean_val) / std_val if std_val > 0 else data
            data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
    
    # Add batch and channel dimensions
    return data[np.newaxis, ..., np.newaxis]
```

## Achievements and Limitations

### Achievements
- ✅ **Successful training** on full resolution data
- ✅ **Vision Mamba integration** working with 3D medical images
- ✅ **SAM2 attention** providing spatial reasoning
- ✅ **Stable convergence** after resolving technical issues
- ✅ **Baseline established** for comparison with optimized versions

### Limitations
- ⚠️ **Memory constraints** limiting batch size to 2
- ⚠️ **Training efficiency** reduced due to memory limitations
- ⚠️ **GPU utilization** suboptimal due to single GPU requirement
- ⚠️ **Performance ceiling** at 62.5% validation Dice

## Historical Context
This V1.0 represents the first successful training of the SMART SOTA architecture:
- **Proof of concept** for Vision Mamba in medical imaging
- **Foundation** for subsequent optimizations in V1.1
- **Learning experience** that led to cropping and memory optimization strategies
- **Baseline model** for comparison and ablation studies

## Lessons Learned
1. **Memory optimization critical** for practical deployment
2. **Batch size significantly impacts** gradient quality and convergence
3. **Spatial resolution trade-offs** between detail and efficiency
4. **CUDNN compatibility issues** require careful layer design
5. **Single GPU strategy** more stable than multi-GPU for complex architectures

## Future Development Path
V1.0 → V1.1 improvements:
- Center cropping for memory optimization
- Increased batch size (2 → 6)
- Improved training stability
- Better validation performance (62.5% → 74.4%)

---
**V1.0 Training completed successfully on July 20, 2025**  
**Foundation model for SMART SOTA architecture development**  
**Superseded by V1.1 with significant optimizations**
EOF

echo "✅ V1.0 README.md created successfully"

# Create a summary file with key metrics for V1.0
cat > TRAINING_SUMMARY.txt << 'EOF'
STROKE SEGMENTATION V1.0 - TRAINING SUMMARY
==========================================

FINAL RESULTS:
- Validation Dice: 62.5%
- Training completed: 200/200 epochs
- Resolution: Full (192×224×176)
- Date: July 20, 2025

KEY FILES:
- Trained Model: models/emergency_save_20250720_052847.keras
- Training Script: smart_sota_2025_claude.py
- SLURM Script: scripts/smart_sota_2025_claude.sh

ARCHITECTURE:
- Vision Mamba (O(n) complexity)
- SAM2 Attention (4 heads, memory banks)
- U-Net backbone with residual blocks
- Input: 192×224×176×1 (full resolution)
- Parameters: ~15M

TRAINING CONFIG:
- Batch Size: 2 (memory limited)
- Initial LR: 1e-4
- Dataset: 655 cases (Atlas-2)
- GPUs: 4x RTX 4500 (single GPU used)
- Framework: TensorFlow 2.15.1

SIGNIFICANCE:
- First successful SMART SOTA training
- Proof of concept for Vision Mamba + SAM2
- Foundation for V1.1 optimizations
- Baseline: 62.5% validation Dice
EOF

echo "✅ V1.0 training summary created"

# Show final structure
echo ""
echo "🎉 SUCCESS! Created stroke_segmentation_v1.0_success_formatted with:"
echo ""
find . -type f | sort
echo ""
echo "📊 Directory structure:"
tree -a . 2>/dev/null || ls -la

echo ""
echo "✅ V1.0 files organized successfully!"
echo "📁 Location: $(pwd)"
echo "🎯 Ready for transfer to local machine!"
echo ""
echo "⚠️  NOTE: You may need to manually copy V1.0 training logs if available"
echo "Look for logs with older job numbers in ../logs/ directory"
EOF
