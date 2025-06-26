#!/usr/bin/env python3
"""
Auto-Configuration System for State-of-the-Art Stroke Lesion Segmentation
Automatically adapts model architecture and training parameters based on:
- Available GPU memory and count
- Dataset characteristics
- Hardware capabilities
- Target performance vs memory trade-offs
"""

import os
import psutil
import tensorflow as tf
import numpy as np
from typing import Dict, Tuple, Any, Optional, List
import yaml
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class SOTAAutoConfig:
    """Auto-configuration for state-of-the-art hybrid CNN-Transformer model"""
    
    def __init__(self, data_stats: Optional[Dict] = None, target_memory_gb: Optional[float] = None):
        self.data_stats = data_stats or {}
        self.target_memory_gb = target_memory_gb
        
        # Hardware detection
        self.gpu_memory_gb = self._get_gpu_memory()
        self.gpu_count = self._get_gpu_count()
        self.cpu_count = psutil.cpu_count()
        self.ram_gb = psutil.virtual_memory().total / (1024**3)
        
        # Auto-determine target memory if not specified
        if target_memory_gb is None:
            self.target_memory_gb = self.gpu_memory_gb * 0.8  # Use 80% of available
        else:
            self.target_memory_gb = min(target_memory_gb, self.gpu_memory_gb * 0.9)
        
        logger.info(f"AutoConfig initialized: {self.gpu_count} GPUs, "
                   f"{self.gpu_memory_gb:.1f}GB VRAM, target: {self.target_memory_gb:.1f}GB")
    
    def _get_gpu_memory(self) -> float:
        """Get GPU memory in GB"""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                # Try to get actual memory info
                try:
                    gpu_details = tf.config.experimental.get_device_details(gpus[0])
                    if 'device_name' in gpu_details:
                        device_name = gpu_details['device_name'].lower()
                        # Known GPU memory mappings
                        gpu_memory_map = {
                            'rtx 4090': 24.0, 'rtx 4080': 16.0, 'rtx 4070': 12.0,
                            'rtx 4500': 24.0, 'a100': 40.0, 'v100': 16.0,
                            'rtx 3090': 24.0, 'rtx 3080': 10.0, 'a40': 48.0,
                            'a4500': 20.0, 'rtx 2080': 8.0
                        }
                        for gpu_name, memory in gpu_memory_map.items():
                            if gpu_name in device_name:
                                return memory
                except:
                    pass
                
                # Fallback to conservative estimate
                return 12.0
            return 0.0
        except:
            return 8.0  # Conservative fallback
    
    def _get_gpu_count(self) -> int:
        """Get number of available GPUs"""
        try:
            return len(tf.config.list_physical_devices('GPU'))
        except:
            return 0
    
    def estimate_model_memory(self, config: Dict) -> float:
        """Estimate model memory usage in GB"""
        base_filters = config.get('base_filters', 32)
        depth = config.get('depth', 4)
        input_shape = config.get('input_shape', [128, 128, 128, 1])
        use_transformer = config.get('use_transformer', False)
        batch_size = config.get('batch_size', 1)
        
        # Estimate parameters
        total_params = 0
        filters = base_filters
        
        # Encoder parameters
        for i in range(depth):
            # ResNeXt block parameters
            block_params = (
                filters * filters +  # 1x1 conv
                filters * filters * 27 +  # 3x3x3 conv
                filters * filters +  # 1x1 conv
                filters * (filters // 16) * 2  # SE block
            )
            total_params += block_params
            
            # Transformer parameters (if enabled)
            if use_transformer and i >= depth - 2:
                transformer_params = (
                    filters * filters * 4 +  # QKV projections
                    filters * filters * 4 +  # MLP
                    filters * 2  # Layer norms
                )
                total_params += transformer_params
            
            filters = min(filters * 2, 512)
        
        # Decoder parameters (similar to encoder)
        decoder_params = total_params * 0.7  # Approximate
        total_params += decoder_params
        
        # Memory estimation
        param_memory = total_params * 4 / (1024**3)  # 4 bytes per float32 parameter
        
        # Activation memory (major component for 3D)
        voxels = np.prod(input_shape[:-1])
        activation_memory = voxels * base_filters * depth * batch_size * 4 / (1024**3)
        
        # Gradients (same size as parameters)
        gradient_memory = param_memory
        
        # Optimizer states (Adam uses 2x parameter memory)
        optimizer_memory = param_memory * 2
        
        # Buffer memory
        buffer_memory = 2.0  # GB
        
        total_memory = param_memory + activation_memory + gradient_memory + optimizer_memory + buffer_memory
        
        return total_memory
    
    def auto_determine_architecture(self) -> Dict[str, Any]:
        """Automatically determine the best architecture for available hardware"""
        
        architectures = {
            'basic': {
                'base_filters': 16,
                'depth': 3,
                'use_transformer': False,
                'transformer_depth': 0,
                'num_heads': 0,
                'description': 'Basic CNN-only model'
            },
            'attention': {
                'base_filters': 24,
                'depth': 4,
                'use_transformer': False,
                'transformer_depth': 0,
                'num_heads': 0,
                'description': 'CNN with SE blocks and attention gates'
            },
            'hybrid_small': {
                'base_filters': 32,
                'depth': 4,
                'use_transformer': True,
                'transformer_depth': 1,
                'num_heads': 4,
                'description': 'Hybrid CNN-Transformer (memory efficient)'
            },
            'hybrid_medium': {
                'base_filters': 40,
                'depth': 4,
                'use_transformer': True,
                'transformer_depth': 2,
                'num_heads': 8,
                'description': 'Hybrid CNN-Transformer (balanced)'
            },
            'hybrid_large': {
                'base_filters': 48,
                'depth': 5,
                'use_transformer': True,
                'transformer_depth': 2,
                'num_heads': 8,
                'description': 'Full SOTA architecture'
            }
        }
        
        # Test each architecture for memory compatibility
        compatible_archs = []
        
        for name, arch in architectures.items():
            test_config = {
                **arch,
                'input_shape': self._determine_input_shape(),
                'batch_size': self._determine_batch_size()
            }
            
            estimated_memory = self.estimate_model_memory(test_config)
            
            if estimated_memory <= self.target_memory_gb:
                compatible_archs.append({
                    'name': name,
                    'config': arch,
                    'estimated_memory': estimated_memory,
                    'memory_efficiency': self.target_memory_gb / estimated_memory
                })
        
        if not compatible_archs:
            logger.warning("No architecture fits in available memory, using basic")
            return architectures['basic']
        
        # Select the most complex architecture that fits
        best_arch = max(compatible_archs, key=lambda x: x['config']['base_filters'] * 
                       (x['config']['depth'] + x['config']['transformer_depth']))
        
        logger.info(f"Selected architecture: {best_arch['name']} "
                   f"(~{best_arch['estimated_memory']:.1f}GB)")
        
        return best_arch['config']
    
    def _determine_input_shape(self) -> List[int]:
        """Determine optimal input shape based on memory and data"""
        data_shape = self.data_stats.get('original_shape', [197, 233, 189])
        
        # Memory-based sizing
        if self.target_memory_gb >= 20:
            target_size = [160, 192, 160]
        elif self.target_memory_gb >= 16:
            target_size = [144, 176, 144]
        elif self.target_memory_gb >= 12:
            target_size = [128, 160, 128]
        elif self.target_memory_gb >= 8:
            target_size = [112, 144, 112]
        else:
            target_size = [96, 128, 96]
        
        # Adjust to be close to original aspect ratio
        original_ratios = [data_shape[i] / min(data_shape) for i in range(3)]
        base_size = min(target_size)
        
        adjusted_shape = [int(base_size * ratio) for ratio in original_ratios]
        
        # Ensure multiples of 16 for efficient processing
        adjusted_shape = [max(96, (s // 16) * 16) for s in adjusted_shape]
        
        # Add channel dimension
        adjusted_shape.append(1)
        
        logger.info(f"Input shape: {data_shape} → {adjusted_shape[:-1]}")
        
        return adjusted_shape
    
    def _determine_batch_size(self) -> int:
        """Determine optimal batch size"""
        if self.target_memory_gb >= 20:
            batch_size = 4 if self.gpu_count > 1 else 2
        elif self.target_memory_gb >= 16:
            batch_size = 2
        elif self.target_memory_gb >= 12:
            batch_size = 2
        else:
            batch_size = 1
        
        # Adjust for multi-GPU
        if self.gpu_count > 1:
            batch_size = max(2, batch_size)
        
        return batch_size
    
    def auto_determine_training_config(self, dataset_size: int = 600) -> Dict[str, Any]:
        """Determine training configuration"""
        
        # Base configuration
        config = {
            'epochs': 100,
            'patience': 20,
            'learning_rate': 1e-4,
            'validation_split': 0.2,
            'use_mixed_precision': True,
            'gradient_checkpointing': self.target_memory_gb < 16,
        }
        
        # Adjust based on dataset size
        if dataset_size < 200:
            config.update({
                'epochs': 150,
                'patience': 25,
                'learning_rate': 5e-5,
                'use_data_augmentation': True,
                'augmentation_strength': 0.8
            })
        elif dataset_size < 500:
            config.update({
                'epochs': 120,
                'patience': 20,
                'learning_rate': 1e-4,
                'use_data_augmentation': True,
                'augmentation_strength': 0.6
            })
        else:
            config.update({
                'epochs': 100,
                'patience': 15,
                'learning_rate': 1e-4,
                'use_data_augmentation': True,
                'augmentation_strength': 0.4
            })
        
        # Learning rate schedule
        config['lr_schedule'] = self._determine_lr_schedule(dataset_size)
        
        return config
    
    def _determine_lr_schedule(self, dataset_size: int) -> Dict[str, Any]:
        """Determine learning rate schedule"""
        if dataset_size < 200:
            return {
                'type': 'cosine_decay',
                'warmup_epochs': 10,
                'decay_steps': 100,
                'alpha': 0.1
            }
        elif dataset_size < 500:
            return {
                'type': 'reduce_on_plateau',
                'factor': 0.5,
                'patience': 8,
                'min_lr': 1e-7
            }
        else:
            return {
                'type': 'polynomial_decay',
                'decay_steps': 80,
                'end_learning_rate': 1e-6,
                'power': 0.9
            }
    
    def auto_determine_loss_config(self) -> Dict[str, Any]:
        """Determine loss function configuration"""
        lesion_stats = self.data_stats.get('lesion_stats', {})
        avg_lesion_volume = lesion_stats.get('avg_volume_percentage', 0.5)
        
        if avg_lesion_volume < 0.1:  # Very small lesions
            return {
                'loss_type': 'combined',
                'weights': {
                    'generalized_dice': 0.4,
                    'focal': 0.4,
                    'boundary': 0.2
                },
                'focal_gamma': 3.0,
                'focal_alpha': 0.25,
                'boundary_weight': 1.0
            }
        elif avg_lesion_volume < 1.0:  # Medium lesions
            return {
                'loss_type': 'combined',
                'weights': {
                    'generalized_dice': 0.5,
                    'focal': 0.3,
                    'boundary': 0.2
                },
                'focal_gamma': 2.0,
                'focal_alpha': 0.25,
                'boundary_weight': 0.5
            }
        else:  # Large lesions
            return {
                'loss_type': 'combined',
                'weights': {
                    'generalized_dice': 0.6,
                    'focal': 0.3,
                    'boundary': 0.1
                },
                'focal_gamma': 2.0,
                'focal_alpha': 0.25,
                'boundary_weight': 0.3
            }
    
    def auto_determine_augmentation_config(self) -> Dict[str, Any]:
        """Determine data augmentation configuration"""
        lesion_stats = self.data_stats.get('lesion_stats', {})
        avg_lesion_volume = lesion_stats.get('avg_volume_percentage', 0.5)
        
        base_config = {
            'spatial_prob': 0.8,
            'intensity_prob': 0.7,
            'noise_prob': 0.3,
            'elastic_prob': 0.4,
        }
        
        if avg_lesion_volume < 0.1:  # Small lesions - be careful with spatial augmentation
            base_config.update({
                'rotation_range': (-10, 10),
                'scaling_range': (0.9, 1.1),
                'translation_range': (-5, 5),
                'elastic_deformation': 0.1,
                'spatial_prob': 0.6
            })
        elif avg_lesion_volume < 1.0:  # Medium lesions
            base_config.update({
                'rotation_range': (-15, 15),
                'scaling_range': (0.85, 1.15),
                'translation_range': (-10, 10),
                'elastic_deformation': 0.2
            })
        else:  # Large lesions - can handle more aggressive augmentation
            base_config.update({
                'rotation_range': (-20, 20),
                'scaling_range': (0.8, 1.2),
                'translation_range': (-15, 15),
                'elastic_deformation': 0.3
            })
        
        # Intensity augmentation
        base_config.update({
            'brightness_range': (-0.1, 0.1),
            'contrast_range': (0.9, 1.1),
            'gamma_range': (0.8, 1.2),
            'noise_std': 0.05
        })
        
        return base_config
    
    def generate_complete_config(self, 
                               dataset_size: int = 600,
                               save_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate complete auto-configuration"""
        
        # Architecture configuration
        architecture = self.auto_determine_architecture()
        input_shape = self._determine_input_shape()
        batch_size = self._determine_batch_size()
        
        # Complete configuration
        config = {
            'meta': {
                'generated_by': 'SOTAAutoConfig',
                'gpu_count': self.gpu_count,
                'gpu_memory_gb': self.gpu_memory_gb,
                'target_memory_gb': self.target_memory_gb,
                'dataset_size': dataset_size
            },
            'model': {
                **architecture,
                'input_shape': input_shape,
                'num_classes': 1,
                'memory_efficient': self.target_memory_gb < 16
            },
            'training': {
                **self.auto_determine_training_config(dataset_size),
                'batch_size': batch_size,
                'global_batch_size': batch_size * max(1, self.gpu_count)
            },
            'loss': self.auto_determine_loss_config(),
            'augmentation': self.auto_determine_augmentation_config(),
            'hardware': {
                'use_mixed_precision': True,
                'gradient_checkpointing': self.target_memory_gb < 16,
                'use_multi_gpu': self.gpu_count > 1,
                'num_workers': min(8, self.cpu_count),
                'prefetch_factor': 2
            }
        }
        
        # Validate configuration
        estimated_memory = self.estimate_model_memory(config['model'])
        config['meta']['estimated_memory_gb'] = estimated_memory
        
        if estimated_memory > self.target_memory_gb:
            logger.warning(f"Estimated memory ({estimated_memory:.1f}GB) exceeds target "
                          f"({self.target_memory_gb:.1f}GB)")
        
        # Save configuration
        if save_path:
            with open(save_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {save_path}")
        
        return config
    
    def get_progressive_configs(self, dataset_size: int = 600) -> Dict[str, Dict]:
        """Get progressive training configurations"""
        
        # Base architecture progression
        architectures = ['basic', 'attention', 'hybrid_small', 'hybrid_medium', 'hybrid_large']
        
        configs = {}
        for arch_name in architectures:
            # Temporarily set target memory to test this architecture
            original_target = self.target_memory_gb
            
            # Adjust target based on architecture complexity
            if arch_name == 'basic':
                self.target_memory_gb = min(8, self.gpu_memory_gb * 0.5)
            elif arch_name == 'attention':
                self.target_memory_gb = min(12, self.gpu_memory_gb * 0.6)
            elif arch_name == 'hybrid_small':
                self.target_memory_gb = min(16, self.gpu_memory_gb * 0.7)
            elif arch_name == 'hybrid_medium':
                self.target_memory_gb = min(20, self.gpu_memory_gb * 0.8)
            else:  # hybrid_large
                self.target_memory_gb = self.gpu_memory_gb * 0.9
            
            config = self.generate_complete_config(dataset_size)
            config['meta']['architecture_level'] = arch_name
            
            configs[arch_name] = config
            
            # Restore original target
            self.target_memory_gb = original_target
        
        return configs

# Factory functions for easy usage
def create_auto_config(data_dir: str = None, 
                      target_memory_gb: Optional[float] = None) -> SOTAAutoConfig:
    """Create auto-configuration instance with optional data analysis"""
    
    data_stats = {}
    
    if data_dir:
        data_stats = analyze_dataset(data_dir)
    
    return SOTAAutoConfig(data_stats=data_stats, target_memory_gb=target_memory_gb)

def analyze_dataset(data_dir: str) -> Dict[str, Any]:
    """Analyze dataset characteristics for auto-configuration"""
    import nibabel as nib
    
    data_path = Path(data_dir)
    image_dir = data_path / 'Images'
    mask_dir = data_path / 'Masks'
    
    stats = {
        'total_samples': 0,
        'original_shape': None,
        'lesion_stats': {
            'volumes': [],
            'avg_volume_percentage': 0,
            'min_volume_percentage': 0,
            'max_volume_percentage': 0
        }
    }
    
    try:
        image_files = list(image_dir.glob('*_T1w.nii.gz'))
        stats['total_samples'] = len(image_files)
        
        if image_files:
            # Sample a few files for analysis
            sample_files = image_files[:min(10, len(image_files))]
            
            shapes = []
            lesion_volumes = []
            
            for img_file in sample_files:
                # Load image
                img = nib.load(img_file)
                shapes.append(img.shape)
                
                # Load corresponding mask
                base_name = img_file.name.replace('_T1w.nii.gz', '')
                mask_file = mask_dir / f"{base_name}_label-L_desc-T1lesion_mask.nii.gz"
                
                if mask_file.exists():
                    mask = nib.load(mask_file)
                    mask_data = mask.get_fdata()
                    
                    total_voxels = np.prod(mask_data.shape)
                    lesion_voxels = np.sum(mask_data > 0)
                    lesion_percentage = (lesion_voxels / total_voxels) * 100
                    
                    lesion_volumes.append(lesion_percentage)
            
            # Calculate statistics
            if shapes:
                stats['original_shape'] = list(shapes[0][:3])  # Take first shape
            
            if lesion_volumes:
                stats['lesion_stats'].update({
                    'volumes': lesion_volumes,
                    'avg_volume_percentage': np.mean(lesion_volumes),
                    'min_volume_percentage': np.min(lesion_volumes),
                    'max_volume_percentage': np.max(lesion_volumes)
                })
    
    except Exception as e:
        logger.warning(f"Could not analyze dataset: {e}")
    
    return stats

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example 1: Basic auto-configuration
    auto_config = create_auto_config(target_memory_gb=20)
    config = auto_config.generate_complete_config(dataset_size=600)
    
    print("🎯 Auto-Generated Configuration:")
    print(f"Architecture: {config['model']['base_filters']} filters, depth {config['model']['depth']}")
    print(f"Transformer: {config['model']['use_transformer']}")
    print(f"Input shape: {config['model']['input_shape'][:-1]}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Estimated memory: {config['meta']['estimated_memory_gb']:.1f}GB")
    
    # Example 2: Progressive configurations
    progressive_configs = auto_config.get_progressive_configs()
    
    print(f"\n📈 Progressive Training Options:")
    for name, cfg in progressive_configs.items():
        print(f"{name:15}: {cfg['meta']['estimated_memory_gb']:.1f}GB, "
              f"{cfg['model']['base_filters']} filters, "
              f"Transformer: {cfg['model']['use_transformer']}")
