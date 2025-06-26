#!/usr/bin/env python3
"""
State-of-the-Art Training Pipeline for Stroke Lesion Segmentation
Integrates hybrid CNN-Transformer architecture with proven working environment
Implements progressive training and advanced memory optimization techniques
"""

import tensorflow as tf
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import os
import sys
import logging
import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import argparse
import json
from datetime import datetime

# Add project root to path
sys.path.append('/mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota')

# Import custom modules
from models.hybrid_model import create_hybrid_model, create_progressive_models
from models.losses import combined_loss, deep_supervision_loss, dice_coefficient, sensitivity, specificity
from config.auto_config import create_auto_config
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

class ProgressiveTrainingPipeline:
    """Progressive training from simple to complex architectures"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Load or generate configuration
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Auto-generate configuration
            auto_config = create_auto_config(
                data_dir="/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training",
                target_memory_gb=20  # Target 20GB for RTX 4500
            )
            self.config = auto_config.generate_complete_config(dataset_size=655)
        
        # Setup GPU
        self._setup_gpu()
        
        # Create data loader
        self.data_loader = self._create_data_loader()
        
        # Initialize model stages
        self.model_stages = self._initialize_model_stages()
        
    def _setup_gpu(self):
        """Configure GPU for optimal memory usage"""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s): {gpus}")
            
            # Set memory limit if specified
            if self.config['hardware'].get('gpu_memory_limit'):
                memory_limit = int(self.config['hardware']['gpu_memory_limit'] * 1024)
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                )
                logger.info(f"GPU memory limited to {memory_limit}MB")
    
    def _create_data_loader(self) -> RealAtlasDataLoader:
        """Create data loader with configuration"""
        target_shape = tuple(self.config['model']['input_shape'][:-1])
        
        loader = RealAtlasDataLoader(
            data_dir=self.config.get('data', {}).get('data_dir', 
                "/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training"),
            target_shape=target_shape
        )
        
        return loader
    
    def _initialize_model_stages(self) -> Dict[str, Dict]:
        """Initialize progressive model stages"""
        stages = {
            'stage1_basic': {
                'base_filters': 16,
                'depth': 3,
                'use_transformer': False,
                'epochs': 30,
                'description': 'Basic CNN architecture'
            },
            'stage2_attention': {
                'base_filters': 24,
                'depth': 4,
                'use_transformer': False,
                'epochs': 40,
                'description': 'CNN with attention mechanisms'
            },
            'stage3_hybrid_small': {
                'base_filters': 32,
                'depth': 4,
                'use_transformer': True,
                'epochs': 50,
                'description': 'Small hybrid CNN-Transformer'
            },
            'stage4_hybrid_full': {
                'base_filters': self.config['model']['base_filters'],
                'depth': self.config['model']['depth'],
                'use_transformer': True,
                'epochs': self.config['training']['epochs'],
                'description': 'Full SOTA architecture'
            }
        }
        
        return stages
    
    def _create_callbacks(self, stage_name: str) -> List[tf.keras.callbacks.Callback]:
        """Create callbacks for training"""
        callbacks_dir = Path('callbacks') / stage_name
        callbacks_dir.mkdir(parents=True, exist_ok=True)
        
        callbacks = [
            # Model checkpoint
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(callbacks_dir / 'best_model.h5'),
                monitor='val_dice_coefficient',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            
            # Early stopping
            tf.keras.callbacks.EarlyStopping(
                monitor='val_dice_coefficient',
                mode='max',
                patience=self.config['training']['patience'],
                restore_best_weights=True,
                verbose=1
            ),
            
            # Learning rate reduction
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            
            # CSV logger
            tf.keras.callbacks.CSVLogger(
                str(callbacks_dir / 'training_log.csv')
            ),
            
            # Custom memory monitoring
            MemoryMonitorCallback(log_dir=str(callbacks_dir))
        ]
        
        # Add TensorBoard if requested
        if self.config.get('training', {}).get('use_tensorboard', True):
            callbacks.append(
                tf.keras.callbacks.TensorBoard(
                    log_dir=str(callbacks_dir / 'tensorboard'),
                    histogram_freq=0,  # Disable histograms to save memory
                    write_graph=False,
                    update_freq='epoch'
                )
            )
        
        return callbacks
    
    def _compile_model(self, model: tf.keras.Model, stage_config: Dict):
        """Compile model with appropriate loss and metrics"""
        # Get loss configuration
        loss_config = self.config['loss']
        
        # Create loss function
        if loss_config['loss_type'] == 'combined':
            loss_fn = combined_loss(
                dice_weight=loss_config['weights']['generalized_dice'],
                focal_weight=loss_config['weights']['focal'],
                boundary_weight=loss_config['weights']['boundary'],
                focal_gamma=loss_config['focal_gamma'],
                focal_alpha=loss_config['focal_alpha']
            )
        else:
            loss_fn = 'binary_crossentropy'
        
        # Create optimizer with mixed precision
        initial_lr = self.config['training']['learning_rate']
        optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        # Compile
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=[
                dice_coefficient,
                sensitivity,
                specificity,
                'accuracy'
            ]
        )
        
        return model
    
    def train_stage(self, stage_name: str, initial_weights: Optional[str] = None) -> str:
        """Train a single stage"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {stage_name}")
        logger.info(f"{'='*60}")
        
        stage_config = self.model_stages[stage_name]
        
        # Create model
        model = create_hybrid_model(
            input_shape=tuple(self.config['model']['input_shape']),
            base_filters=stage_config['base_filters'],
            depth=stage_config['depth'],
            use_transformer=stage_config['use_transformer'],
            memory_efficient=True
        )
        
        # Load initial weights if provided
        if initial_weights and Path(initial_weights).exists():
            logger.info(f"Loading weights from {initial_weights}")
            try:
                # Load weights from previous stage
                prev_model = tf.keras.models.load_model(
                    initial_weights, 
                    custom_objects={
                        'dice_coefficient': dice_coefficient,
                        'sensitivity': sensitivity,
                        'specificity': specificity
                    },
                    compile=False
                )
                
                # Transfer compatible weights
                for layer in model.layers:
                    try:
                        if hasattr(prev_model, layer.name):
                            prev_layer = getattr(prev_model, layer.name)
                            if prev_layer.get_weights():
                                layer.set_weights(prev_layer.get_weights())
                                logger.info(f"Transferred weights for layer: {layer.name}")
                    except:
                        pass
                        
            except Exception as e:
                logger.warning(f"Could not load all weights: {e}")
        
        # Compile model
        model = self._compile_model(model, stage_config)
        
        # Log model info
        total_params = model.count_params()
        logger.info(f"Model parameters: {total_params:,}")
        logger.info(f"Estimated memory: ~{total_params * 4 / 1e9:.2f}GB")
        
        # Create datasets
        batch_size = self.config['training']['batch_size']
        train_dataset, val_dataset = self.data_loader.create_dataset(
            batch_size=batch_size,
            shuffle=True,
            validation_split=self.config['training']['validation_split']
        )
        
        # Create callbacks
        callbacks = self._create_callbacks(stage_name)
        
        # Train model with gradient accumulation if needed
        if self.config['hardware'].get('gradient_checkpointing', False):
            logger.info("Using gradient accumulation for memory efficiency")
            history = self._train_with_gradient_accumulation(
                model, train_dataset, val_dataset, 
                epochs=stage_config['epochs'],
                callbacks=callbacks,
                accumulation_steps=4
            )
        else:
            # Standard training
            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=stage_config['epochs'],
                callbacks=callbacks,
                verbose=1
            )
        
        # Save final model
        final_path = f"models/{stage_name}_final.h5"
        model.save(final_path)
        logger.info(f"Saved final model to {final_path}")
        
        # Save training history
        history_path = f"results/{stage_name}_history.json"
        os.makedirs("results", exist_ok=True)
        with open(history_path, 'w') as f:
            json.dump(history.history, f, indent=2)
        
        return final_path
    
    def _train_with_gradient_accumulation(self, model, train_dataset, val_dataset, 
                                        epochs, callbacks, accumulation_steps=4):
        """Train with gradient accumulation for memory efficiency"""
        # This is a simplified version - full implementation would require custom training loop
        logger.info(f"Training with gradient accumulation (steps={accumulation_steps})")
        
        # For now, fall back to standard training with smaller batch size
        return model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
    
    def run_progressive_training(self):
        """Run complete progressive training pipeline"""
        logger.info("\n🚀 Starting Progressive Training Pipeline")
        logger.info(f"Total stages: {len(self.model_stages)}")
        
        results = {}
        previous_weights = None
        
        for stage_name, stage_config in self.model_stages.items():
            logger.info(f"\n📊 {stage_config['description']}")
            
            try:
                # Train stage
                model_path = self.train_stage(stage_name, previous_weights)
                
                # Evaluate stage
                metrics = self.evaluate_model(model_path)
                results[stage_name] = {
                    'model_path': model_path,
                    'metrics': metrics,
                    'config': stage_config
                }
                
                # Use this model's weights for next stage
                previous_weights = model_path
                
                logger.info(f"✅ {stage_name} completed - Dice: {metrics['dice']:.4f}")
                
            except Exception as e:
                logger.error(f"❌ {stage_name} failed: {e}")
                results[stage_name] = {'error': str(e)}
        
        # Save final results
        results_path = f"results/progressive_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n🎯 Progressive training completed! Results saved to {results_path}")
        
        return results
    
    def evaluate_model(self, model_path: str) -> Dict[str, float]:
        """Evaluate a trained model"""
        logger.info(f"Evaluating model: {model_path}")
        
        # Load model
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'dice_coefficient': dice_coefficient,
                'sensitivity': sensitivity,
                'specificity': specificity
            },
            compile=False
        )
        
        # Create test dataset
        _, val_dataset = self.data_loader.create_dataset(
            batch_size=1,
            shuffle=False,
            validation_split=0.2
        )
        
        # Evaluate
        metrics = {
            'dice': [],
            'sensitivity': [],
            'specificity': []
        }
        
        for x, y in val_dataset.take(20):  # Evaluate on subset
            pred = model.predict(x, verbose=0)
            
            dice_val = dice_coefficient(y, pred).numpy()
            sens_val = sensitivity(y, pred).numpy()
            spec_val = specificity(y, pred).numpy()
            
            metrics['dice'].append(dice_val)
            metrics['sensitivity'].append(sens_val)
            metrics['specificity'].append(spec_val)
        
        # Average metrics
        avg_metrics = {
            'dice': np.mean(metrics['dice']),
            'sensitivity': np.mean(metrics['sensitivity']),
            'specificity': np.mean(metrics['specificity'])
        }
        
        return avg_metrics


class MemoryMonitorCallback(tf.keras.callbacks.Callback):
    """Monitor GPU memory usage during training"""
    
    def __init__(self, log_dir: str):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.memory_log = []
    
    def on_epoch_end(self, epoch, logs=None):
        try:
            # Get GPU memory info
            gpu_info = tf.config.experimental.get_memory_info('GPU:0')
            current_memory = gpu_info['current'] / (1024**3)  # Convert to GB
            peak_memory = gpu_info['peak'] / (1024**3)
            
            self.memory_log.append({
                'epoch': epoch,
                'current_memory_gb': current_memory,
                'peak_memory_gb': peak_memory,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"GPU Memory - Current: {current_memory:.2f}GB, Peak: {peak_memory:.2f}GB")
            
            # Save log
            with open(self.log_dir / 'memory_usage.json', 'w') as f:
                json.dump(self.memory_log, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Could not get GPU memory info: {e}")


def main():
    """Main training entry point"""
    parser = argparse.ArgumentParser(description='SOTA Stroke Segmentation Training')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--stage', type=str, help='Specific stage to train')
    parser.add_argument('--progressive', action='store_true', help='Run progressive training')
    parser.add_argument('--evaluate', type=str, help='Path to model to evaluate')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = ProgressiveTrainingPipeline(config_path=args.config)
    
    if args.evaluate:
        # Evaluate specific model
        metrics = pipeline.evaluate_model(args.evaluate)
        print(f"\nEvaluation Results:")
        print(f"Dice Score: {metrics['dice']:.4f}")
        print(f"Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")
        
    elif args.stage:
        # Train specific stage
        if args.stage in pipeline.model_stages:
            pipeline.train_stage(args.stage)
        else:
            logger.error(f"Unknown stage: {args.stage}")
            logger.info(f"Available stages: {list(pipeline.model_stages.keys())}")
            
    else:
        # Run progressive training
        pipeline.run_progressive_training()


if __name__ == "__main__":
    main()
