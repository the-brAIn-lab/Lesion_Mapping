#!/bin/bash
#SBATCH --job-name=stroke_sota_full
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a4500:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/sota_full_%j.out
#SBATCH --error=logs/sota_full_%j.err

echo "=================================================="
echo "🚀 FULL STATE-OF-THE-ART STROKE SEGMENTATION"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start Time: $(date)"
echo ""

# Navigate to project directory
cd /mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota

# Setup environment (proven working configuration)
echo "📦 Setting up environment..."
module load gcc/9.3.0-5wu3 cuda/12.6.3-ziu7

# Activate conda environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate tf215_env

# Critical environment variables
export LD_LIBRARY_PATH="/mnt/beegfs/hellgate/home/rb194958e/.conda/envs/tf215_env/lib:$LD_LIBRARY_PATH"
export TF_XLA_FLAGS="--tf_xla_enable_xla_devices=false"
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_ENABLE_ONEDNN_OPTS=0

# Create necessary directories
mkdir -p logs models callbacks results

# Check environment
echo "🔍 Checking environment..."
python -c "
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')
gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs available: {len(gpus)}')
for gpu in gpus:
    print(f'  {gpu}')
"

# Check data directory
echo ""
echo "📁 Checking data directory..."
echo "Training data: /mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training_Split"
ls -la /mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training_Split/

# Count images
echo ""
echo "📊 Dataset statistics:"
echo "Training images: $(ls -1 /mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training_Split/Images/*.nii.gz | wc -l)"
echo "Training masks: $(ls -1 /mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training_Split/Masks/*.nii.gz | wc -l)"

# Save the training script
echo ""
echo "💾 Saving training script..."
cat > sota_full_training.py << 'EOFSCRIPT'
#!/usr/bin/env python3
"""
Full State-of-the-Art Hybrid CNN-Transformer Training
Uses the complete architecture from hybrid_model.py
Trains on Training_Split (600 images)
"""

import os
import sys
import tensorflow as tf
import numpy as np
import logging
from pathlib import Path
import yaml
import json
from datetime import datetime

# Add project directory to path
sys.path.append('/mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota')

# Import the full SOTA model and components
from models.hybrid_model import HybridCNNTransformer, create_hybrid_model
from models.losses import combined_loss, dice_coefficient, sensitivity, specificity
from real_data_loader import RealAtlasDataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enable mixed precision
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Configure GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logger.info(f"Found {len(gpus)} GPU(s): {gpus}")

def train_full_sota_model():
    """Train the complete SOTA hybrid CNN-Transformer model"""
    
    logger.info("="*80)
    logger.info("🚀 FULL STATE-OF-THE-ART TRAINING")
    logger.info("="*80)
    
    # Configuration for full SOTA model
    config = {
        'data_dir': '/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training_Split',
        'input_shape': (128, 128, 128, 1),  # Start with 128³ for memory efficiency
        'base_filters': 32,
        'depth': 4,
        'use_transformer': True,
        'transformer_depth': 2,
        'num_heads': 8,
        'dropout': 0.1,
        'batch_size': 1,
        'epochs': 100,
        'learning_rate': 1e-4,
        'validation_split': 0.2
    }
    
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Create data loader for Training_Split
    logger.info(f"\n📁 Loading data from: {config['data_dir']}")
    data_loader = RealAtlasDataLoader(
        data_dir=config['data_dir'],
        target_shape=config['input_shape'][:-1]
    )
    
    # Create datasets
    train_dataset, val_dataset = data_loader.create_dataset(
        batch_size=config['batch_size'],
        shuffle=True,
        validation_split=config['validation_split']
    )
    
    # Calculate steps
    total_samples = len(data_loader.file_pairs)
    train_samples = int(total_samples * (1 - config['validation_split']))
    val_samples = total_samples - train_samples
    steps_per_epoch = train_samples // config['batch_size']
    validation_steps = val_samples // config['batch_size']
    
    logger.info(f"\n📊 Dataset Statistics:")
    logger.info(f"  Total samples: {total_samples}")
    logger.info(f"  Training samples: {train_samples}")
    logger.info(f"  Validation samples: {val_samples}")
    logger.info(f"  Steps per epoch: {steps_per_epoch}")
    
    # Build the full SOTA model
    logger.info("\n🏗️ Building Full SOTA Hybrid CNN-Transformer Model...")
    
    model = HybridCNNTransformer(
        input_shape=config['input_shape'],
        num_classes=1,
        base_filters=config['base_filters'],
        depth=config['depth'],
        use_transformer=config['use_transformer'],
        transformer_depth=config['transformer_depth'],
        num_heads=config['num_heads'],
        dropout=config['dropout']
    )
    
    # Build model with dummy input to initialize
    dummy_input = tf.random.normal((1, *config['input_shape']))
    _ = model(dummy_input, training=False)
    
    # Count parameters
    total_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    logger.info(f"\n🔢 Model Statistics:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Estimated memory: ~{total_params * 4 / 1e9:.2f}GB")
    
    # Create loss function
    loss_fn = combined_loss(
        dice_weight=1.0,
        focal_weight=1.0,
        boundary_weight=0.5,
        use_generalized_dice=True,
        focal_alpha=0.25,
        focal_gamma=3.0
    )
    
    # Create optimizer with mixed precision
    initial_lr = config['learning_rate']
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[dice_coefficient, sensitivity, specificity, 'accuracy']
    )
    
    # Create callbacks
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    callbacks_dir = Path(f'callbacks/sota_full_{timestamp}')
    callbacks_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        # Save best model based on validation dice
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(callbacks_dir / 'best_model.h5'),
            monitor='val_dice_coefficient',
            mode='max',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_dice_coefficient',
            mode='max',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        ),
        
        # CSV logger
        tf.keras.callbacks.CSVLogger(
            str(callbacks_dir / 'training_history.csv')
        ),
        
        # TensorBoard
        tf.keras.callbacks.TensorBoard(
            log_dir=str(callbacks_dir / 'tensorboard'),
            histogram_freq=0,
            write_graph=False,
            update_freq='epoch'
        )
    ]
    
    # Custom learning rate schedule
    def lr_schedule(epoch):
        """Cosine annealing with warm restarts"""
        if epoch < 10:
            # Warmup
            return initial_lr * (epoch + 1) / 10
        else:
            # Cosine annealing
            progress = (epoch - 10) / (config['epochs'] - 10)
            return initial_lr * 0.5 * (1 + np.cos(np.pi * progress))
    
    callbacks.append(
        tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)
    )
    
    # Train model
    logger.info("\n🎯 Starting Training...")
    logger.info(f"Callbacks directory: {callbacks_dir}")
    
    try:
        history = model.fit(
            train_dataset,
            epochs=config['epochs'],
            steps_per_epoch=steps_per_epoch,
            validation_data=val_dataset,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("\n✅ Training completed successfully!")
        
        # Save final model
        final_model_path = f'models/sota_hybrid_final_{timestamp}.h5'
        model.save(final_model_path)
        logger.info(f"Final model saved to: {final_model_path}")
        
        # Save configuration
        config_path = callbacks_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Save training summary
        summary_path = callbacks_dir / 'training_summary.json'
        final_metrics = {
            'final_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'final_dice': float(history.history['dice_coefficient'][-1]),
            'final_val_dice': float(history.history['val_dice_coefficient'][-1]),
            'best_val_dice': float(max(history.history['val_dice_coefficient'])),
            'total_epochs': len(history.history['loss']),
            'total_parameters': int(total_params)
        }
        
        with open(summary_path, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        logger.info(f"\n📊 Final Results:")
        logger.info(f"  Best Validation Dice: {final_metrics['best_val_dice']:.4f}")
        logger.info(f"  Final Validation Dice: {final_metrics['final_val_dice']:.4f}")
        logger.info(f"  Final Training Dice: {final_metrics['final_dice']:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run the training
    success = train_full_sota_model()
    
    if success:
        logger.info("\n🎉 SOTA training completed successfully!")
        sys.exit(0)
    else:
        logger.error("\n❌ SOTA training failed!")
        sys.exit(1)
EOFSCRIPT

# Make the script executable
chmod +x sota_full_training.py

# Run the training
echo ""
echo "🚀 Starting Full SOTA Training..."
echo "=================================================="
python sota_full_training.py

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Training completed successfully!"
    
    # List generated files
    echo ""
    echo "📁 Generated files:"
    ls -la models/sota_hybrid_final_*.h5 2>/dev/null
    ls -la callbacks/sota_full_*/best_model.h5 2>/dev/null
    
    # Show final results
    if [ -f callbacks/sota_full_*/training_summary.json ]; then
        echo ""
        echo "📊 Training Summary:"
        cat callbacks/sota_full_*/training_summary.json
    fi
else
    echo ""
    echo "❌ Training failed! Check error logs."
fi

echo ""
echo "End Time: $(date)"
echo "=================================================="
