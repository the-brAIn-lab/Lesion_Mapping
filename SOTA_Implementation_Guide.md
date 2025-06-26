# State-of-the-Art Stroke Lesion Segmentation Implementation Guide

## 🎯 Overview

This guide explains how to implement and train a state-of-the-art hybrid CNN-Transformer model for stroke lesion segmentation using the ATLAS 2.0 dataset on the University of Montana's Hellgate cluster.

## 🏗️ Architecture Overview

### Model Components

1. **SE-ResNeXt Blocks**: Convolutional blocks with Squeeze-and-Excitation attention
2. **Attention Gates**: Learnable skip connections that focus on relevant features
3. **Transformer Blocks**: Self-attention mechanisms for global context
4. **Multi-Scale Deep Supervision**: Auxiliary losses at different scales
5. **Combined Loss Function**: Generalized Dice + Focal + Boundary losses

### Progressive Training Strategy

We use a 4-stage progressive approach to build up to the full SOTA model:

| Stage | Architecture | Parameters | Input Size | Memory |
|-------|--------------|------------|------------|---------|
| 1 | Basic CNN | ~500K | 96³ | ~6GB |
| 2 | CNN + Attention | ~2M | 112³ | ~10GB |
| 3 | Small Hybrid | ~8M | 128³ | ~15GB |
| 4 | Full SOTA | ~22M | 144³ | ~20GB |

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Load modules (CRITICAL - exact versions)
module load gcc/9.3.0-5wu3 cuda/12.6.3-ziu7

# Activate environment
conda activate tf215_env

# Set library path (ESSENTIAL for GPU detection)
export LD_LIBRARY_PATH="/mnt/beegfs/hellgate/home/rb194958e/.conda/envs/tf215_env/lib:$LD_LIBRARY_PATH"

# TensorFlow settings
export TF_XLA_FLAGS="--tf_xla_enable_xla_devices=false"
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

### 2. Submit Training Job

```bash
# Progressive training (recommended)
sbatch scripts/sota_submission_script.sh

# Or direct training of specific stage
sbatch --wrap="python optimized_hybrid_training.py"
```

### 3. Monitor Training

```bash
# Watch training progress
tail -f logs/sota_training_*.out

# Check GPU usage
squeue -u $USER

# Monitor memory usage
watch -n 10 'tail -20 callbacks/*/memory_usage.json'
```

## 📊 Data Preparation

The ATLAS 2.0 dataset structure:
```
Atlas_2/
├── Training/
│   ├── Images/
│   │   └── *_T1w.nii.gz
│   └── Masks/
│       └── *_label-L_desc-T1lesion_mask.nii.gz
└── Testing/
    └── Images/
        └── *_T1w.nii.gz
```

### Data Characteristics
- Original size: 197×233×189
- Lesion volume: 0.02%-1.5% of brain
- Format: NIfTI compressed (.nii.gz)

## 🔧 Memory Optimization Techniques

### 1. Mixed Precision Training
```python
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
```

### 2. Gradient Checkpointing
Enabled automatically when memory < 16GB

### 3. Progressive Input Sizing
Start with smaller inputs and gradually increase:
- Stage 1: 96³ → Stage 4: 144³

### 4. Efficient Data Loading
- Batch-wise loading with tf.keras.utils.Sequence
- On-the-fly resizing with scipy.ndimage.zoom
- Memory-mapped file reading

### 5. Model Architecture Optimization
- Bottleneck design with reduced channels
- Grouped convolutions in ResNeXt blocks
- Adaptive number of transformer heads

## 🏃 Running Different Configurations

### Option 1: Full Progressive Pipeline
```python
python sota_training_pipeline.py --progressive
```

### Option 2: Specific Stage Training
```python
python sota_training_pipeline.py --stage stage3_hybrid_small
```

### Option 3: Direct Optimized Training
```python
python optimized_hybrid_training.py
```

### Option 4: Custom Configuration
```python
from config.auto_config import create_auto_config

# Generate config for your hardware
auto_config = create_auto_config(target_memory_gb=20)
config = auto_config.generate_complete_config(dataset_size=655)

# Save and use
with open('custom_config.yaml', 'w') as f:
    yaml.dump(config, f)

python sota_training_pipeline.py --config custom_config.yaml
```

## 📈 Expected Results

### Performance Metrics
- **Dice Score**: 0.75-0.85 (SOTA range)
- **Sensitivity**: >0.80 (critical for clinical use)
- **Specificity**: >0.99 (avoid false positives)

### Training Time
- Stage 1: ~2-3 hours
- Stage 2: ~4-5 hours
- Stage 3: ~6-8 hours
- Stage 4: ~10-12 hours

### Model Sizes
- Basic CNN: ~2MB
- CNN + Attention: ~8MB
- Small Hybrid: ~30MB
- Full SOTA: ~90MB

## 🐛 Troubleshooting

### GPU Not Detected
```bash
# Check GPU visibility
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# If empty, ensure library path is set:
export LD_LIBRARY_PATH="/mnt/beegfs/hellgate/home/rb194958e/.conda/envs/tf215_env/lib:$LD_LIBRARY_PATH"
```

### Out of Memory (OOM)
1. Reduce batch size to 1
2. Decrease input size (e.g., 128³ → 112³)
3. Reduce base_filters (e.g., 32 → 24)
4. Disable transformer blocks
5. Enable gradient checkpointing

### Slow Training
1. Check data loading bottleneck
2. Increase num_workers in data loader
3. Use prefetch_factor=2
4. Ensure SSD storage for data

### Poor Dice Scores
1. Check class imbalance handling
2. Adjust focal loss parameters (gamma, alpha)
3. Increase augmentation strength
4. Use longer training with patience

## 📊 Evaluation and Inference

### Model Evaluation
```python
python sota_training_pipeline.py --evaluate models/stage4_hybrid_full_final.h5
```

### Generate Test Predictions
```python
from inference import generate_test_predictions

# Load best model
model_path = "models/stage4_hybrid_full_final.h5"

# Generate predictions for test set
generate_test_predictions(
    model_path=model_path,
    test_dir="/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Testing",
    output_dir="predictions/test_set"
)
```

### Visualize Results
```python
from visualization import plot_segmentation_results

# Compare prediction with ground truth
plot_segmentation_results(
    image_path="test_image.nii.gz",
    prediction_path="prediction.nii.gz",
    ground_truth_path="ground_truth.nii.gz",
    save_path="visualization.png"
)
```

## 🔄 Transfer Learning

To use trained weights for other datasets:

```python
# Load pretrained model
base_model = tf.keras.models.load_model("models/stage4_hybrid_full_final.h5")

# Freeze early layers
for layer in base_model.layers[:20]:
    layer.trainable = False

# Add task-specific head if needed
# ...

# Fine-tune on new data
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # Lower learning rate
    loss=combined_dice_focal_loss,
    metrics=[dice_coefficient]
)
```

## 📚 Key Innovations

1. **Hybrid Architecture**: Combines local CNN features with global transformer attention
2. **Progressive Training**: Gradual complexity increase prevents overfitting
3. **Advanced Loss Functions**: Handles extreme class imbalance effectively
4. **Memory Optimization**: Enables large models on limited GPU memory
5. **Attention Mechanisms**: SE blocks + attention gates + self-attention

## 🎯 Best Practices

1. **Always start with the basic model** to verify setup
2. **Monitor GPU memory** throughout training
3. **Use validation set** for hyperparameter tuning
4. **Save checkpoints** at each stage
5. **Log all experiments** for reproducibility
6. **Gradually increase complexity** based on results

## 📞 Support

For issues or questions:
1. Check logs in `logs/` directory
2. Review error messages in `.err` files
3. Verify environment setup
4. Consult this guide's troubleshooting section

Remember: The key to success is starting simple and progressively increasing complexity while monitoring performance and memory usage!
