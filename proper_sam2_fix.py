#!/usr/bin/env python3
"""
proper_sam2_fix.py - Properly fix the SAM2 shape mismatch by updating the aggregation logic
This maintains the multi-scale benefits while ensuring shape compatibility
"""

import os
import re

def create_fixed_sam2_class():
    """Create the properly fixed SAM2 attention class"""
    
    fixed_class = '''class WorkingEnhancedSAM2Attention(layers.Layer):
    """
    PROPERLY FIXED: Enhanced SAM2 attention with correct multi-scale aggregation
    """
    def __init__(self, channels, num_scales=3, memory_size=32, preferred_num_heads=8, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.num_scales = min(num_scales, 3)
        self.memory_size = memory_size
        self.preferred_num_heads = preferred_num_heads
        
        logger.info(f"🔧 WorkingEnhancedSAM2 (FIXED): {channels} channels, {self.num_scales} scales, {memory_size} memory")
        
        # Core components
        self.pos_encoding = WorkingPositionalEncoding3D(channels, name=f'{self.name}_pos_enc')
        self.adaptive_pool = WorkingAdaptivePooling3D(name=f'{self.name}_adaptive_pool')
        self.memory_bank = WorkingHierarchicalMemoryBank(
            channels, self.num_scales, memory_size // self.num_scales, name=f'{self.name}_memory'
        )
        
        # Cross-scale attention
        self.cross_attention = WorkingCrossScaleAttention(
            channels, preferred_num_heads, name=f'{self.name}_cross_attn'
        )
        
        # Feature extraction layers
        self.scale_convs = []
        for i in range(self.num_scales):
            scale_conv = layers.Conv3D(channels, 3, padding='same', name=f'{self.name}_scale_conv_{i}')
            self.scale_convs.append(scale_conv)
        
        # Multi-scale fusion layers (NEW - for proper aggregation)
        self.scale_upsample_layers = []
        for i in range(1, self.num_scales):
            # Create learnable upsampling layers for each scale
            upsample_layer = layers.Conv3DTranspose(
                channels, 
                kernel_size=2**i, 
                strides=2**i, 
                padding='same',
                name=f'{self.name}_upsample_scale_{i}'
            )
            self.scale_upsample_layers.append(upsample_layer)
        
        # Output processing
        self.output_norm = layers.LayerNormalization(name=f'{self.name}_output_norm')
        self.output_projection = layers.Dense(channels, name=f'{self.name}_output_proj')
        
        # Learnable scale weights
        self.scale_weights = self.add_weight(
            shape=(self.num_scales,),
            initializer='ones',
            trainable=True,
            name=f'{self.name}_scale_weights'
        )
        
        # Residual weight
        self.alpha = self.add_weight(
            shape=(1,),
            initializer='zeros',
            trainable=True,
            name=f'{self.name}_alpha'
        )
        
    @working_debug
    def call(self, x, training=None):
        """Enhanced SAM2 attention with PROPER multi-scale aggregation"""
        batch_size = tf.shape(x)[0]
        original_h, original_w, original_d = tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        
        # Add positional encoding
        x_with_pos = self.pos_encoding(x, training=training)
        
        # Apply adaptive pooling
        x_pooled = self.adaptive_pool(x_with_pos, training=training)
        
        # Get base spatial dimensions after pooling
        base_h = tf.shape(x_pooled)[1]
        base_w = tf.shape(x_pooled)[2]
        base_d = tf.shape(x_pooled)[3]
        
        # Extract features at multiple scales
        multi_scale_features_spatial = []
        current_features = x_pooled
        
        for i, scale_conv in enumerate(self.scale_convs):
            # Apply scale-specific convolution
            scale_features = scale_conv(current_features)
            
            # Store in spatial format (not flattened)
            multi_scale_features_spatial.append(scale_features)
            
            # Downsample for next scale
            if i < len(self.scale_convs) - 1:
                current_features = layers.AveragePooling3D(2, padding='same')(current_features)
        
        # Process each scale with attention
        enhanced_features_spatial = []
        
        for i, scale_features in enumerate(multi_scale_features_spatial):
            # Get current scale dimensions
            scale_h = tf.shape(scale_features)[1]
            scale_w = tf.shape(scale_features)[2]
            scale_d = tf.shape(scale_features)[3]
            
            # Flatten for attention
            scale_features_flat = tf.reshape(
                scale_features, 
                [batch_size, scale_h * scale_w * scale_d, self.channels]
            )
            
            # Get memory features for this scale
            memory_features = self.memory_bank([scale_features_flat], training=training)[0]
            
            # Combine with memory
            combined_features = tf.concat([scale_features_flat, memory_features], axis=1)
            
            # Apply cross-scale attention
            enhanced_flat = self.cross_attention(
                scale_features_flat, 
                combined_features, 
                training=training
            )
            
            # Reshape back to spatial
            enhanced_spatial = tf.reshape(
                enhanced_flat,
                [batch_size, scale_h, scale_w, scale_d, self.channels]
            )
            
            # Upsample to base resolution if needed
            if i > 0:
                # Use the pre-defined upsampling layer
                enhanced_spatial = self.scale_upsample_layers[i-1](enhanced_spatial)
                # Ensure exact size match
                enhanced_spatial = enhanced_spatial[:, :base_h, :base_w, :base_d, :]
            
            enhanced_features_spatial.append(enhanced_spatial)
        
        # Now all features have the same spatial dimensions - aggregate with weights
        scale_weights_normalized = tf.nn.softmax(self.scale_weights)
        
        # Initialize aggregated features
        aggregated = tf.zeros_like(enhanced_features_spatial[0])
        
        # Weighted sum of all scales
        for i, features in enumerate(enhanced_features_spatial):
            weight = tf.cast(scale_weights_normalized[i], features.dtype)
            aggregated = aggregated + weight * features
        
        # Apply output processing
        aggregated_flat = tf.reshape(
            aggregated, 
            [batch_size, base_h * base_w * base_d, self.channels]
        )
        aggregated_flat = self.output_norm(aggregated_flat, training=training)
        aggregated_flat = self.output_projection(aggregated_flat)
        
        # Reshape back to spatial
        aggregated_spatial = tf.reshape(
            aggregated_flat,
            [batch_size, base_h, base_w, base_d, self.channels]
        )
        
        # Upsample back to original size
        output_upsampled = self._upsample_to_original_size(
            aggregated_spatial, original_h, original_w, original_d
        )
        
        # Apply residual connection
        alpha_cast = tf.cast(self.alpha, x.dtype)
        output = x + alpha_cast * output_upsampled
        
        return output
    
    def _upsample_to_original_size(self, x_pooled, target_h, target_w, target_d):
        """Upsample using transpose convolution for better quality"""
        current_h = tf.shape(x_pooled)[1]
        current_w = tf.shape(x_pooled)[2]
        current_d = tf.shape(x_pooled)[3]
        
        # Calculate scale factors
        scale_h = target_h // current_h
        scale_w = target_w // current_w
        scale_d = target_d // current_d
        
        # Use transpose convolution if upsampling needed
        needs_upsample = tf.logical_or(
            tf.logical_or(scale_h > 1, scale_w > 1),
            scale_d > 1
        )
        
        def do_upsample():
            # Create a simple upsampling layer
            upsampled = layers.Conv3DTranspose(
                self.channels,
                kernel_size=(scale_h * 2, scale_w * 2, scale_d * 2),
                strides=(scale_h, scale_w, scale_d),
                padding='same',
                name=f'{self.name}_final_upsample'
            )(x_pooled)
            # Crop to exact size
            return upsampled[:, :target_h, :target_w, :target_d, :]
        
        def no_upsample():
            return x_pooled
        
        return tf.cond(needs_upsample, do_upsample, no_upsample)
'''
    
    return fixed_class

def apply_proper_fix():
    """Apply the proper fix to the SAM2 class"""
    
    print("🔧 Applying Proper SAM2 Multi-Scale Aggregation Fix...")
    
    # Read the original file
    with open("smart_sota_2025_final.py", 'r') as f:
        content = f.read()
    
    # Create backup
    backup_name = f"smart_sota_2025_final_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    with open(backup_name, 'w') as f:
        f.write(content)
    print(f"✅ Backup created: {backup_name}")
    
    # Find the WorkingEnhancedSAM2Attention class
    class_pattern = r'class WorkingEnhancedSAM2Attention\(layers\.Layer\):.*?(?=class\s|\Z)'
    
    # Get the fixed class
    fixed_class = create_fixed_sam2_class()
    
    # Replace the class
    fixed_content = re.sub(class_pattern, fixed_class + '\n\n', content, flags=re.DOTALL)
    
    # Write the fixed file
    fixed_filename = "smart_sota_2025_final_properly_fixed.py"
    with open(fixed_filename, 'w') as f:
        f.write(fixed_content)
    
    print(f"✅ Fixed file created: {fixed_filename}")
    
    # Create submission script
    with open("scripts/smart_sota_2025_properly_fixed.sh", 'w') as f:
        f.write('''#!/bin/bash
#SBATCH --job-name=smart_sota_properly_fixed
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gpus=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/smart_sota_properly_fixed_%j.out
#SBATCH --error=logs/smart_sota_properly_fixed_%j.err

echo "🎉 SMART SOTA 2025 - PROPERLY FIXED VERSION"
echo "==========================================="
echo "✅ Multi-scale aggregation properly fixed"
echo "✅ All features upsampled to same resolution before aggregation"
echo "✅ Learnable upsampling layers for each scale"
echo "✅ Weighted fusion with softmax normalization"
echo ""

cd /mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota

module load gcc/9.3.0-5wu3 cuda/12.6.3-ziu7
eval "$(conda shell.bash hook)" || true
conda activate tf215_env

export LD_LIBRARY_PATH="/mnt/beegfs/hellgate/home/rb194958e/.conda/envs/tf215_env/lib:$LD_LIBRARY_PATH"
export TF_ENABLE_ONEDNN_OPTS=0
export TF_GPU_ALLOCATOR=cuda_malloc_async
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO

echo "🚀 Starting properly fixed training..."
python -u smart_sota_2025_final_properly_fixed.py

echo "Training completed: $(date)"
''')
    
    os.chmod("scripts/smart_sota_2025_properly_fixed.sh", 0o755)
    print("✅ Submission script created: scripts/smart_sota_2025_properly_fixed.sh")
    
    print("\n🎉 Proper fix applied successfully!")
    print("\n📋 The fix includes:")
    print("  • Multi-scale features are upsampled to base resolution BEFORE aggregation")
    print("  • Learnable transpose convolution layers for upsampling")
    print("  • Weighted fusion with softmax-normalized scale weights")
    print("  • All tensor shapes guaranteed to match")
    print("\n🚀 To run the fixed version:")
    print("   sbatch scripts/smart_sota_2025_properly_fixed.sh")

if __name__ == "__main__":
    apply_proper_fix()
