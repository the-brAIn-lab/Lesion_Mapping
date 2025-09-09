#!/usr/bin/env python3
"""
SMART SOTA 2025: Stroke Lesion Segmentation - CROPPED DATASET VERSION
Modified for: /mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training/Cropped_655_for_training
COMPLETE VERSION with all working components from successful model
"""

import os
import sys
from pathlib import Path
import logging

# Set up directory structure FIRST
os.makedirs("logs", exist_ok=True)
os.makedirs("callbacks", exist_ok=True)

# Environment configuration - must be before TF import
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# Disable XLA compilation which can cause CUDNN tensor descriptor issues
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'

# ========================
# LOGGING CONFIGURATION
# ========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/smart_sota_cropped.log'),
        logging.FileHandler(f'logs/training_cropped_{os.getpid()}.debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('SmartSOTA_Cropped')
logger.setLevel(logging.DEBUG)

# ========================
# IMPORTS WITH ERROR HANDLING
# ========================
try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model, regularizers
    import numpy as np
    import nibabel as nib
    from scipy.ndimage import zoom, binary_dilation, rotate
    from sklearn.model_selection import train_test_split, KFold
    from skimage import measure
    import json, time, math, random, traceback, warnings
    
    logger.info("✅ All imports successful")
except ImportError as e:
    logger.critical(f"❌ Import failed: {str(e)}")
    sys.exit(1)

# ========================
# RUNTIME CONFIGURATION
# ========================
tf.config.run_functions_eagerly(False)
logger.info(f"TensorFlow eager execution: {tf.executing_eagerly()}")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="nibabel")

logger.info(f"Environment verified:\n"
            f"- Python {sys.version}\n"
            f"- TF {tf.__version__}\n"
            f"- NumPy {np.__version__}\n"
            f"- GPU devices: {len(tf.config.list_physical_devices('GPU'))}")

# Updated configuration section for the cropped training script
# Replace the CroppedTrainingConfig class with this:

class CroppedTrainingConfig:
    # ======================= DATA CONFIG =======================
    # UPDATED for the new cropped combined dataset
    DATA_DIR = Path("/home/rbielski/Atlas_2/Training/Cropped_128_Combined
")
    
    # Fixed input shape since we're cropping to exactly 128x128x128
    INPUT_SHAPE = (128, 128, 128, 1)
    VALIDATION_SPLIT = 0.15
    SMALL_LESION_THRESHOLD = 100
    
    # ===================== TRAINING CONFIG =====================
    # Can use larger batch size since 128^3 is much smaller than 192x224x176
    BATCH_SIZE = 6  # Increased from 4 due to smaller input size
    INITIAL_EPOCH = 0
    TOTAL_EPOCHS = 200
    INITIAL_LR = 1e-4
    MIN_LR = 1e-6
    WARMUP_EPOCHS = 10
    MAX_GRAD_NORM = 1.0
    
    # ====================== MODEL CONFIG =======================
    BASE_FILTERS = 12  # Slightly increased since we have more memory available
    DROPOUT_RATE = 0.5  # Reduced since smaller input size
    L2_REG = 1e-3      # Reduced regularization
    MAMBA_DEPTH = 2
    SAM_HEADS = 4
    
    # ==================== AUGMENTATION CONFIG ==================
    AUGMENTATION_INTENSITY = 0.4
    SYNTHETIC_LESION_PROB = 0.3
    ROTATION_RANGE = 15
    
    # ==================== LOSS CONFIG ==========================
    USE_BOUNDARY_LOSS = True
    DICE_LOSS_WEIGHT = 0.5
    BOUNDARY_LOSS_WEIGHT = 0.5
    DEEP_SUPERVISION_WEIGHTS = [0.1, 0.2, 0.3]
    
    # ==================== DIRECTORIES & PATHS ==================
    MODEL_DIR = Path("models/cropped128_production")
    CALLBACKS_DIR = Path("callbacks/cropped128_production")

    def __init__(self):
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.CALLBACKS_DIR.mkdir(parents=True, exist_ok=True)
        
        config_dict = {k: v for k, v in self.__dict__.items() 
                      if not k.startswith('__') and not callable(v)}
        with open(self.MODEL_DIR / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        logger.info(f"🎯 Configuration for 128x128x128 cropped dataset:")
        logger.info(f"   Input shape: {self.INPUT_SHAPE}")
        logger.info(f"   Batch size: {self.BATCH_SIZE}")
        logger.info(f"   Base filters: {self.BASE_FILTERS}")
        logger.info(f"   Data directory: {self.DATA_DIR}")
    
    @property
    def model_path(self):
        return self.MODEL_DIR / f"smart_sota_cropped128_{self.timestamp}.keras"
    
    @property
    def checkpoint_path(self):
        return self.CALLBACKS_DIR / "best_model_cropped128.keras"

# Updated data loading function for the combined directory
def load_cropped_combined_dataset(config):
    """Load the combined cropped dataset (images and masks in same directory)"""
    logger.info("📚 Loading combined cropped dataset...")
    log_memory_usage("cropped_combined_dataset_load_start")
    
    if not config.DATA_DIR.exists():
        raise FileNotFoundError(f"Combined cropped data directory not found: {config.DATA_DIR}")
    
    # Look for image and mask files in the same directory
    image_files = list(config.DATA_DIR.glob("*_T1w_cropped128.nii.gz"))
    mask_files = list(config.DATA_DIR.glob("*_mask_cropped128.nii.gz"))
    
    logger.info(f"Found {len(image_files)} image files")
    logger.info(f"Found {len(mask_files)} mask files")
    
    if not image_files or not mask_files:
        raise FileNotFoundError("Could not find cropped image or mask files")
    
    # Match pairs based on base filename
    pairs = []
    lesion_counts = []
    
    for img_file in image_files:
        # Extract base identifier (remove _T1w_cropped128 suffix)
        img_base = img_file.stem.replace('_T1w_cropped128', '')
        
        # Find matching mask
        mask_file = config.DATA_DIR / f"{img_base}_mask_cropped128.nii.gz"
        
        if mask_file.exists():
            try:
                # Quick check for lesion presence
                mask_obj = nib.load(str(mask_file))
                mask_data = mask_obj.get_fdata()
                has_lesion = np.any(mask_data > 0)
                
                pairs.append((img_file, mask_file))
                lesion_counts.append(1 if has_lesion else 0)
                
                del mask_obj, mask_data
                gc.collect()
                
            except Exception as e:
                logger.warning(f"Skipping {mask_file}: {e}")
                continue
        else:
            logger.warning(f"No matching mask found for {img_file.name}")
    
    logger.info(f"📊 Created {len(pairs)} image-mask pairs")
    logger.info(f"🧠 Class balance: {np.mean(lesion_counts)*100:.2f}% contain lesions")
    log_memory_usage("cropped_combined_dataset_load_end")
    
    return pairs, np.array(lesion_counts)


# ============================================================================
# CUSTOM LAYERS (SAME AS WORKING VERSION)
# ============================================================================

class ResidualConvBlock(layers.Layer):
    """Simplified residual block to avoid CUDNN gradient issues"""
    def __init__(self, filters, kernel_reg=None, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_reg = kernel_reg
        
    def build(self, input_shape):
        self.conv1 = layers.Conv3D(self.filters, 3, padding='same', kernel_regularizer=self.kernel_reg)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv3D(self.filters, 3, padding='same', kernel_regularizer=self.kernel_reg)
        self.bn2 = layers.BatchNormalization()
        self.dropout = layers.SpatialDropout3D(0.1)
        
        # Always create projection layer
        self.residual_conv = layers.Conv3D(self.filters, 1, padding='same')
        self.residual_bn = layers.BatchNormalization()
        
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        # Simplified call without complex operations
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)  # Use ReLU instead of GELU
        x = self.dropout(x, training=training)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        # Residual connection
        residual = self.residual_conv(inputs)
        residual = self.residual_bn(residual, training=training)
        
        return tf.nn.relu(x + residual)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_reg": tf.keras.regularizers.serialize(self.kernel_reg) 
                        if self.kernel_reg else None
        })
        return config

class VisionMambaBlock(layers.Layer):
    """Efficient Vision Mamba block with fixed 5D tensor handling"""
    def __init__(self, filters, kernel_size=3, expansion=2, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.expansion = expansion
        
    def build(self, input_shape):
        self.in_conv = layers.Conv3D(
            self.filters * self.expansion, 
            1, 
            use_bias=False,
            padding='same',
            data_format='channels_last'
        )
        self.spatial_conv = layers.Conv3D(
            self.filters * self.expansion,
            self.kernel_size,
            padding='same',
            use_bias=False,
            data_format='channels_last'
        )
        self.out_conv = layers.Conv3D(
            self.filters, 1, 
            padding='same',
            data_format='channels_last'
        )
        self.norm = layers.LayerNormalization()
        self.dropout = layers.SpatialDropout3D(0.1)
        
    def call(self, inputs, training=None):
        # Simplified activation functions to avoid CUDNN gradient issues
        x = self.in_conv(inputs)
        x = tf.nn.relu(x)  # Use ReLU instead of GELU
        
        x = self.spatial_conv(x)
        x = tf.cast(tf.nn.relu(x), inputs.dtype)  # Use ReLU instead of GELU
        x = self.dropout(x, training=training)
        
        x = self.out_conv(x)
        x = self.norm(x)
        
        # Residual connection
        return x + inputs
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "expansion": self.expansion
        })
        return config

class SAM2Attention(layers.Layer):
    """Enhanced SAM2 Attention with Hierarchical Memory Banks"""
    def __init__(self, filters, heads, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.heads = heads
        self.depth = filters // heads
        
        if filters % heads != 0:
            raise ValueError(f"Filters ({filters}) must be divisible by number of heads ({heads})")
        
    def build(self, input_shape):
        self.query = layers.Conv3D(self.filters, 1, data_format='channels_last')
        self.key = layers.Conv3D(self.filters, 1, data_format='channels_last')
        self.value = layers.Conv3D(self.filters, 1, data_format='channels_last')
        self.out_conv = layers.Conv3D(input_shape[-1], 1, data_format='channels_last')
        self.memory_bank = self.add_weight(
            name='memory_bank',
            shape=(1, 1, 1, 1, self.filters),
            initializer='zeros',
            trainable=True
        )
        self.dropout = layers.SpatialDropout3D(0.1)
        
    def call(self, inputs, training=None):
        # FIXED: Ensure all tensors maintain 5D shape throughout
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        depth = tf.shape(inputs)[3]
        
        # Multi-head projection
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)
        
        # Add memory bank
        k = k + self.memory_bank
        v = v + self.memory_bank
        
        # FIXED: Safer head splitting with explicit shape handling
        q = self._split_heads_safe(q, batch_size, height, width, depth)
        k = self._split_heads_safe(k, batch_size, height, width, depth)
        v = self._split_heads_safe(v, batch_size, height, width, depth)
        
        # Scaled dot-product attention
        dk = tf.cast(self.depth, q.dtype)
        attn_logits = tf.matmul(q, k, transpose_b=True)
        attn_logits = attn_logits / tf.math.sqrt(dk)
        attn_weights = tf.nn.softmax(attn_logits, axis=-1)
        
        # Attention output
        attn_output = tf.matmul(attn_weights, v)
        attn_output = self._combine_heads_safe(attn_output, batch_size, height, width, depth)
        
        # Project back to original channels
        output = self.out_conv(attn_output)
        output = self.dropout(output, training=training)
        
        # Residual connection
        return output + inputs
    
    def _split_heads_safe(self, x, batch_size, height, width, depth):
        """FIXED: Safe head splitting with explicit shape management"""
        # Reshape: [batch, h, w, d, filters] -> [batch, h, w, d, heads, depth]
        x = tf.reshape(x, [batch_size, height, width, depth, self.heads, self.depth])
        # Transpose: [batch, heads, h, w, d, depth]
        return tf.transpose(x, perm=[0, 4, 1, 2, 3, 5])
    
    def _combine_heads_safe(self, x, batch_size, height, width, depth):
        """FIXED: Safe head combining with explicit shape management"""
        # Transpose: [batch, h, w, d, heads, depth]
        x = tf.transpose(x, perm=[0, 2, 3, 4, 1, 5])
        # Reshape: [batch, h, w, d, filters]
        return tf.reshape(x, [batch_size, height, width, depth, self.filters])
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "heads": self.heads
        })
        return config

# ============================================================================
# MEMORY MONITORING AND DATA LOADING (ADAPTED FOR CROPPED DATASET)
# ============================================================================

import psutil
import gc
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

def log_memory_usage(stage):
    process = psutil.Process(os.getpid())
    gb_used = process.memory_info().rss / 1024**3
    gpu_mem = []
    
    try:
        for i in range(4):
            alloc = tf.config.experimental.get_memory_info(f'GPU:{i}')
            gpu_mem.append(f"GPU{i}: {alloc['current']/1e9:.2f}GB")
    except:
        gpu_mem = ["GPU mem tracking failed"]
    
    try:
        disk_usage = psutil.disk_usage('/')
        disk_free_gb = disk_usage.free / 1024**3
        disk_info = f"Disk: {disk_free_gb:.1f}GB free"
    except:
        disk_info = "Disk: unavailable"
    
    logger.info(f"Memory at {stage}: CPU={gb_used:.2f}GB | {' | '.join(gpu_mem)} | {disk_info}")

def inspect_cropped_data_shapes(config):
    """Inspect cropped dataset and auto-detect optimal input shape"""
    logger.info("🔍 INSPECTING CROPPED DATASET FOR OPTIMAL CONFIGURATION")
    
    if not config.DATA_DIR.exists():
        logger.error(f"❌ Cropped data directory not found: {config.DATA_DIR}")
        return None
    
    # Check various possible directory structures
    possible_structures = [
        # Structure 1: Images/ and Masks/ subdirectories  
        ("Images", "Masks"),
        ("images", "masks"),
        ("Images", "Labels"),
        ("images", "labels"),
        # Structure 2: Direct files
        ("", "")  # Will use glob patterns
    ]
    
    image_files = []
    mask_files = []
    found_structure = None
    
    for img_subdir, mask_subdir in possible_structures:
        if img_subdir and mask_subdir:
            img_dir = config.DATA_DIR / img_subdir
            mask_dir = config.DATA_DIR / mask_subdir
            if img_dir.exists() and mask_dir.exists():
                image_files = list(img_dir.glob("*.nii.gz"))
                mask_files = list(mask_dir.glob("*.nii.gz"))
                if image_files and mask_files:
                    found_structure = (img_subdir, mask_subdir)
                    logger.info(f"✅ Found structure: {img_subdir}/ ({len(image_files)} files), {mask_subdir}/ ({len(mask_files)} files)")
                    break
        else:
            # Try direct file patterns
            patterns = [
                ("*T1w*.nii.gz", "*mask*.nii.gz"),
                ("*t1*.nii.gz", "*mask*.nii.gz"),
                ("*image*.nii.gz", "*label*.nii.gz"),
                ("*brain*.nii.gz", "*lesion*.nii.gz"),
            ]
            for img_pattern, mask_pattern in patterns:
                image_files = list(config.DATA_DIR.glob(img_pattern))
                mask_files = list(config.DATA_DIR.glob(mask_pattern))
                if image_files and mask_files:
                    found_structure = (img_pattern, mask_pattern)
                    logger.info(f"✅ Found files: {img_pattern} ({len(image_files)}), {mask_pattern} ({len(mask_files)})")
                    break
            if found_structure:
                break
    
    if not image_files or not mask_files:
        logger.warning("⚠️ Could not auto-detect files. Manual inspection:")
        all_files = list(config.DATA_DIR.rglob("*.nii.gz"))
        logger.info(f"All .nii.gz files found: {len(all_files)}")
        for f in all_files[:10]:  # Show first 10
            logger.info(f"  {f.relative_to(config.DATA_DIR)}")
        return None
    
    # Inspect actual data shapes
    optimal_shape = None
    for i in range(min(5, len(image_files))):
        try:
            img_obj = nib.load(str(image_files[i]))
            mask_obj = nib.load(str(mask_files[i] if i < len(mask_files) else mask_files[0]))
            
            img_shape = img_obj.shape
            mask_shape = mask_obj.shape
            
            logger.info(f"Sample {i+1}:")
            logger.info(f"  Image: {image_files[i].name} - Shape: {img_shape}")
            logger.info(f"  Mask: {mask_files[i].name if i < len(mask_files) else mask_files[0].name} - Shape: {mask_shape}")
            
            if i == 0:
                # Calculate optimal shape for first sample
                # Round up to nearest 16 for GPU efficiency
                optimal_shape = tuple((s + 15) // 16 * 16 for s in img_shape)
                logger.info(f"💡 Optimal INPUT_SHAPE: {optimal_shape + (1,)}")
                
                # Auto-update config
                config.INPUT_SHAPE = optimal_shape + (1,)
                logger.info(f"🔄 Updated config.INPUT_SHAPE to: {config.INPUT_SHAPE}")
                
        except Exception as e:
            logger.error(f"Error inspecting sample {i}: {e}")
    
    # Store structure info for data loading
    config._data_structure = found_structure
    
    return optimal_shape

def load_cropped_dataset(config):
    """Load cropped dataset with auto-detected structure"""
    logger.info("📚 Loading cropped dataset...")
    log_memory_usage("cropped_dataset_load_start")
    
    if not hasattr(config, '_data_structure') or config._data_structure is None:
        inspect_cropped_data_shapes(config)
    
    structure = getattr(config, '_data_structure', None)
    if structure is None:
        raise FileNotFoundError("Could not determine dataset structure")
    
    # Load files based on detected structure
    if len(structure) == 2 and structure[0] and structure[1] and not '*' in structure[0]:
        # Subdirectory structure
        img_dir = config.DATA_DIR / structure[0]
        mask_dir = config.DATA_DIR / structure[1]
        images = list(img_dir.glob("*.nii.gz"))
        masks = list(mask_dir.glob("*.nii.gz"))
    else:
        # Pattern structure
        images = list(config.DATA_DIR.glob(structure[0]))
        masks = list(config.DATA_DIR.glob(structure[1]))
    
    logger.info(f"Found {len(images)} images and {len(masks)} masks")
    
    # Create pairs and analyze lesion presence
    pairs = []
    lesion_counts = []
    
    # For cropped data, try to match by filename similarity
    for img_file in images:
        base_name = img_file.stem.replace('.nii', '')
        
        # Remove common suffixes to find base ID
        for suffix in ['_T1w', '_t1', '_image', '_brain']:
            base_name = base_name.replace(suffix, '')
        
        # Find matching mask
        matching_masks = []
        for mask_file in masks:
            mask_base = mask_file.stem.replace('.nii', '')
            for suffix in ['_mask', '_label', '_lesion', '_seg']:
                mask_base = mask_base.replace(suffix, '')
            
            if base_name in mask_base or mask_base in base_name:
                matching_masks.append(mask_file)
        
        if matching_masks:
            mask_file = matching_masks[0]  # Take first match
            try:
                # Quick check for lesion presence
                mask_obj = nib.load(str(mask_file))
                mask_data = mask_obj.get_fdata()
                has_lesion = np.any(mask_data > 0)
                
                pairs.append((img_file, mask_file))
                lesion_counts.append(1 if has_lesion else 0)
                
                del mask_obj, mask_data
                gc.collect()
                
            except Exception as e:
                logger.warning(f"Skipping {mask_file}: {e}")
                continue
    
    logger.info(f"📊 Created {len(pairs)} image-mask pairs")
    logger.info(f"🧠 Class balance: {np.mean(lesion_counts)*100:.2f}% contain lesions")
    log_memory_usage("cropped_dataset_load_end")
    
    return pairs, np.array(lesion_counts)

def create_batch_compatible_splits(pairs, lesion_presence, batch_size, test_size=0.1):
    """Same as original - create batch-compatible splits"""
    total_samples = len(pairs)
    test_samples = math.floor(total_samples * test_size)
    train_samples = total_samples - test_samples
    
    test_samples = (test_samples // batch_size) * batch_size
    train_samples = total_samples - test_samples
    
    if test_samples < batch_size:
        test_samples = batch_size
        train_samples = total_samples - test_samples
    if train_samples < batch_size:
        train_samples = batch_size
        test_samples = total_samples - train_samples
    
    logger.info(f"🧮 Dataset split: Train={train_samples} ({train_samples/total_samples*100:.1f}%), "
                f"Validation={test_samples} ({test_samples/total_samples*100:.1f}%)")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(pairs, lesion_presence):
        if len(test_idx) >= test_samples:
            test_idx = test_idx[:test_samples]
            break
    
    train_pairs = [pairs[i] for i in train_idx]
    test_pairs = [pairs[i] for i in test_idx]
    
    train_lesions = np.mean([lesion_presence[i] for i in train_idx])
    test_lesions = np.mean([lesion_presence[i] for i in test_idx])
    
    logger.info(f"⚖️ Lesion representation: Train={train_lesions*100:.1f}%, "
                f"Validation={test_lesions*100:.1f}%")
    
    return train_pairs, test_pairs

# ============================================================================
# DATA GENERATOR (SAME AS WORKING VERSION)
# ============================================================================

class MemoryEfficientDataGenerator(tf.keras.utils.Sequence):
    """Memory-optimized data generator with advanced augmentation"""
    
    def __init__(self, pairs, config, is_training=True):
        self.pair_paths = [(str(img), str(mask)) for img, mask in pairs]
        self.batch_size = config.BATCH_SIZE
        self.target_shape = config.INPUT_SHAPE[:-1]
        self.config = config
        self.is_training = is_training
        self.current_epoch = 0
        self.indexes = np.arange(len(self.pair_paths))
        
        self.executor = ThreadPoolExecutor(max_workers=min(4, config.BATCH_SIZE))
        
        self._cache_enabled = psutil.virtual_memory().available > 50 * 1024**3
        self._volume_cache = {} if self._cache_enabled else None
        
        if self.is_training:
            np.random.shuffle(self.indexes)
            
        logger.info(f"🔧 Data generator initialized: {len(self.pair_paths)} samples, "
                   f"cache={'enabled' if self._cache_enabled else 'disabled'}")
    
    def __len__(self):
        return len(self.pair_paths) // self.batch_size
    
    def __getitem__(self, index):
        """Memory-efficient batch loading with optional caching"""
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_paths = [self.pair_paths[i] for i in batch_indexes]
        
        # FIXED: Ensure exact shape consistency
        X = np.zeros((len(batch_paths), *self.target_shape, 1), dtype=np.float32)
        y = np.zeros((len(batch_paths), *self.target_shape, 1), dtype=np.float32)
        
        if self.is_training and len(batch_paths) > 1:
            futures = [
                self.executor.submit(self._load_sample_pair, img_path, mask_path, i)
                for i, (img_path, mask_path) in enumerate(batch_paths)
            ]
            
            for future in futures:
                i, img, mask = future.result()
                X[i, ..., 0] = img
                y[i, ..., 0] = mask
        else:
            for i, (img_path, mask_path) in enumerate(batch_paths):
                _, img, mask = self._load_sample_pair(img_path, mask_path, i)
                X[i, ..., 0] = img
                y[i, ..., 0] = mask
        
        gc.collect()
        
        return X, y
    
    def _load_sample_pair(self, img_path, mask_path, index):
        try:
            img = self._load_volume(img_path)
            mask = self._load_mask(mask_path)
            
            # FIXED: Ensure exact target shape
            if img.shape != self.target_shape:
                img = self._ensure_exact_shape(img, self.target_shape)
            if mask.shape != self.target_shape:
                mask = self._ensure_exact_shape(mask, self.target_shape, is_mask=True)
            
            if self.is_training:
                img, mask = self.augment(img, mask)
            
            return index, img, mask
            
        except Exception as e:
            logger.warning(f"Error loading {img_path}: {e}")
            return index, np.zeros(self.target_shape, dtype=np.float32), np.zeros(self.target_shape, dtype=np.float32)
    
    def _ensure_exact_shape(self, volume, target_shape, is_mask=False):
        """FIXED: Ensure exact target shape with proper handling"""
        if volume.shape == target_shape:
            return volume
        
        # Create output array with exact target shape
        output = np.zeros(target_shape, dtype=volume.dtype)
        
        # Calculate how much we can copy
        copy_shape = tuple(min(v, t) for v, t in zip(volume.shape, target_shape))
        
        # Create slices for copying
        vol_slices = tuple(slice(0, s) for s in copy_shape)
        out_slices = tuple(slice(0, s) for s in copy_shape)
        
        # Copy data
        output[out_slices] = volume[vol_slices]
        
        return output.astype(np.float32)
    
    @lru_cache(maxsize=32)
    def _load_volume(self, path):
        if self._cache_enabled and path in self._volume_cache:
            return self._volume_cache[path].copy()
        
        try:
            img_obj = nib.load(path)
            img = img_obj.get_fdata().astype(np.float32)
            
            img = self.resize_volume(img, order=1)
            img = self.normalize(img)
            
            if self._cache_enabled and img.nbytes < 100 * 1024**2:
                self._volume_cache[path] = img.copy()
            
            return img
            
        except Exception as e:
            logger.warning(f"Error loading volume {path}: {e}")
            return np.zeros(self.target_shape, dtype=np.float32)
    
    def _load_mask(self, path):
        try:
            mask_obj = nib.load(path)
            mask = mask_obj.get_fdata().astype(np.float32)
            
            mask = self.resize_volume(mask, order=0)
            mask = (mask > 0.5).astype(np.float32)
            
            return mask
            
        except Exception as e:
            logger.warning(f"Error loading mask {path}: {e}")
            return np.zeros(self.target_shape, dtype=np.float32)
    
    def resize_volume(self, vol, order=1):
        if vol.shape == self.target_shape:
            return vol
            
        factors = [t/s for t, s in zip(self.target_shape, vol.shape)]
        
        if vol.nbytes > 500 * 1024**2:
            return self._chunked_resize(vol, factors, order)
        else:
            return zoom(vol, factors, order=order)
    
    def _chunked_resize(self, vol, factors, order):
        try:
            return zoom(vol, factors, order=order)
        except MemoryError:
            logger.warning("Memory error during resize, using fallback")
            return np.zeros(self.target_shape, dtype=np.float32)
    
    def normalize(self, img):
        if np.max(img) == 0:
            return np.zeros_like(img)
        
        non_zero = img[img > 0]
        if len(non_zero) == 0:
            return np.zeros_like(img)
        
        p1, p99 = np.percentile(non_zero, [1, 99])
        img = np.clip(img, p1, p99)
        
        mean_val = np.mean(non_zero)
        std_val = np.std(non_zero)
        
        if std_val > 0:
            img = (img - mean_val) / std_val
            img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
        else:
            img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
        
        return img.astype(np.float32)
    
    def enhance_lesions(self, mask):
        try:
            if np.sum(mask) == 0:
                return mask
                
            labeled = measure.label(mask > 0.5)
            regions = measure.regionprops(labeled)
            
            enhanced_mask = mask.copy()
            
            for region in regions:
                if region.area < self.config.SMALL_LESION_THRESHOLD:
                    lesion_mask = (labeled == region.label).astype(bool)
                    kernel = np.ones((3, 3, 3), dtype=bool)
                    expanded = binary_dilation(lesion_mask, structure=kernel, iterations=1)
                    enhanced_mask = np.logical_or(enhanced_mask, expanded).astype(np.float32)
            
            return enhanced_mask
            
        except Exception as e:
            logger.warning(f"Lesion enhancement failed: {e}")
            return mask
    
    def add_synthetic_lesion(self, img, mask):
        try:
            brain_mask = img > 0.1
            if np.sum(brain_mask) < 1000:
                return img, mask
            
            brain_coords = np.where(brain_mask)
            if len(brain_coords[0]) == 0:
                return img, mask
            
            idx = np.random.randint(0, len(brain_coords[0]))
            center = [brain_coords[i][idx] for i in range(3)]
            
            size = np.random.randint(2, min(8, self.config.SMALL_LESION_THRESHOLD // 10))
            intensity_factor = np.random.uniform(0.7, 1.3)
            
            lesion_shape = [
                size + np.random.randint(-1, 2),
                size + np.random.randint(-1, 2), 
                max(1, size // 2 + np.random.randint(-1, 2))
            ]
            
            for i in range(3):
                start = max(0, center[i] - lesion_shape[i] // 2)
                end = min(img.shape[i], center[i] + lesion_shape[i] // 2)
                
                if start < end:
                    coords = np.mgrid[start:end, start:end, start:end]
                    distances = np.sqrt(
                        ((coords[0] - center[0]) / lesion_shape[0]) ** 2 +
                        ((coords[1] - center[1]) / lesion_shape[1]) ** 2 +
                        ((coords[2] - center[2]) / lesion_shape[2]) ** 2
                    )
                    
                    lesion_region = distances <= 1.0
                    
                    mask[start:end, start:end, start:end][lesion_region] = 1
                    
                    original_intensity = img[start:end, start:end, start:end][lesion_region]
                    img[start:end, start:end, start:end][lesion_region] = (
                        original_intensity * intensity_factor
                    )
            
            img = np.clip(img, 0, 1)
            
        except Exception as e:
            logger.warning(f"Synthetic lesion creation failed: {e}")
        
        return img, mask
    
    def random_rotate(self, img, mask):
        try:
            angle = np.random.uniform(-self.config.ROTATION_RANGE, self.config.ROTATION_RANGE)
            axes_pairs = [(0, 1), (0, 2), (1, 2)]
            axes = random.choice(axes_pairs)
            
            img_rot = rotate(img, angle, axes=axes, reshape=False, order=1, mode='reflect', prefilter=False)
            mask_rot = rotate(mask, angle, axes=axes, reshape=False, order=0, mode='reflect', prefilter=False)
            
            return img_rot, mask_rot
            
        except Exception as e:
            logger.warning(f"Rotation failed: {e}")
            return img, mask
    
    def augment(self, img, mask):
        original_lesion_volume = np.sum(mask)
        
        if np.random.rand() > 0.3:
            if np.random.rand() > 0.5:
                img, mask = self.random_rotate(img, mask)
            
            if np.random.rand() > 0.4:
                axis = np.random.choice([0, 1, 2])
                img = np.flip(img, axis=axis)
                mask = np.flip(mask, axis=axis)
        
        if np.random.rand() > 0.2:
            if np.random.rand() > 0.3:
                brightness_factor = np.random.uniform(0.8, 1.2)
                contrast_factor = np.random.uniform(0.9, 1.1)
                img = img * contrast_factor + (brightness_factor - 1) * 0.5
                img = np.clip(img, 0, 1)
            
            if np.random.rand() > 0.6:
                noise_std = np.random.uniform(0.01, 0.05)
                noise = np.random.normal(0, noise_std, img.shape)
                img = np.clip(img + noise, 0, 1)
            
            if np.random.rand() > 0.7:
                gamma = np.random.uniform(0.8, 1.2)
                img = np.power(img, gamma)
        
        if original_lesion_volume > 0:
            if original_lesion_volume < self.config.SMALL_LESION_THRESHOLD and np.random.rand() > 0.4:
                mask = self.enhance_lesions(mask)
        
        if original_lesion_volume < self.config.SMALL_LESION_THRESHOLD // 2 and np.random.rand() < self.config.SYNTHETIC_LESION_PROB:
            img, mask = self.add_synthetic_lesion(img, mask)
        
        final_lesion_volume = np.sum(mask)
        if final_lesion_volume > original_lesion_volume * 3:
            logger.warning("Excessive lesion augmentation detected")
        
        return img.astype(np.float32), mask.astype(np.float32)
    
    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indexes)
        
        if self._cache_enabled and self.current_epoch % 10 == 0:
            self._volume_cache.clear()
            gc.collect()
            logger.info("🧹 Cache cleared for memory optimization")
        
        self.current_epoch += 1
        
        if self.current_epoch % 5 == 0:
            log_memory_usage(f"epoch_{self.current_epoch}_end")
    
    def __del__(self):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

# ============================================================================
# MEMORY MONITORING CALLBACK
# ============================================================================

class MemoryMonitoringCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_frequency=1):
        super().__init__()
        self.log_frequency = log_frequency
    
    def on_epoch_begin(self, epoch, logs=None):
        if epoch % self.log_frequency == 0:
            log_memory_usage(f"epoch_{epoch}_start")
    
    def on_batch_begin(self, batch, logs=None):
        if batch % 10 == 0:
            log_memory_usage(f"batch_{batch}")
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.log_frequency == 0:
            log_memory_usage(f"epoch_{epoch}_end")
            gc.collect()
            tf.keras.backend.clear_session()

def create_optimized_generators(config):
    """Create memory-optimized data generators for cropped dataset"""
    log_memory_usage("generator_creation_start")
    
    # FIRST: Inspect the cropped data to understand structure and dimensions
    inspect_cropped_data_shapes(config)
    
    pairs, lesion_presence = load_cropped_dataset(config)
    
    train_pairs, val_pairs = create_batch_compatible_splits(
        pairs, lesion_presence, config.BATCH_SIZE, config.VALIDATION_SPLIT
    )
    
    train_gen = MemoryEfficientDataGenerator(train_pairs, config, is_training=True)
    val_gen = MemoryEfficientDataGenerator(val_pairs, config, is_training=False)
    
    log_memory_usage("generator_creation_end")
    
    logger.info(f"✅ Created optimized generators for cropped data: "
               f"Train={len(train_gen)} batches, Val={len(val_gen)} batches")
    
    return train_gen, val_gen

# ============================================================================
# MODEL ARCHITECTURE (SAME AS WORKING VERSION)
# ============================================================================

def build_smart_sota(config):
    """U-Net architecture with Vision Mamba and Enhanced SAM2 (single output)"""
    inputs = layers.Input(shape=config.INPUT_SHAPE)
    kernel_reg = regularizers.l2(config.L2_REG) if config.L2_REG > 0 else None
    
    # ENCODER PATH
    x1 = ResidualConvBlock(config.BASE_FILTERS, kernel_reg)(inputs)
    p1 = layers.MaxPooling3D(2)(x1)
    
    x2 = ResidualConvBlock(config.BASE_FILTERS * 2, kernel_reg)(p1)
    p2 = layers.MaxPooling3D(2)(x2)
    
    x3 = ResidualConvBlock(config.BASE_FILTERS * 4, kernel_reg)(p2)
    p3 = layers.MaxPooling3D(2)(x3)
    
    x4 = ResidualConvBlock(config.BASE_FILTERS * 8, kernel_reg)(p3)
    p4 = layers.MaxPooling3D(2)(x4)
    
    # BOTTLENECK
    bottleneck = ResidualConvBlock(config.BASE_FILTERS * 16, kernel_reg)(p4)
    for _ in range(config.MAMBA_DEPTH):
        bottleneck = VisionMambaBlock(config.BASE_FILTERS * 16)(bottleneck)
    
    # DECODER PATH
    u4 = layers.Conv3DTranspose(config.BASE_FILTERS * 8, 2, strides=2, padding='same')(bottleneck)
    u4 = layers.concatenate([u4, x4])
    u4 = ResidualConvBlock(config.BASE_FILTERS * 8, kernel_reg)(u4)
    u4 = SAM2Attention(config.BASE_FILTERS * 8, config.SAM_HEADS)(u4)
    
    u3 = layers.Conv3DTranspose(config.BASE_FILTERS * 4, 2, strides=2, padding='same')(u4)
    u3 = layers.concatenate([u3, x3])
    u3 = ResidualConvBlock(config.BASE_FILTERS * 4, kernel_reg)(u3)
    u3 = SAM2Attention(config.BASE_FILTERS * 4, config.SAM_HEADS)(u3)
    
    u2 = layers.Conv3DTranspose(config.BASE_FILTERS * 2, 2, strides=2, padding='same')(u3)
    u2 = layers.concatenate([u2, x2])
    u2 = ResidualConvBlock(config.BASE_FILTERS * 2, kernel_reg)(u2)
    
    u1 = layers.Conv3DTranspose(config.BASE_FILTERS, 2, strides=2, padding='same')(u2)
    u1 = layers.concatenate([u1, x1])
    u1 = ResidualConvBlock(config.BASE_FILTERS, kernel_reg)(u1)
    
    # OUTPUT
    x = layers.SpatialDropout3D(config.DROPOUT_RATE)(u1)
    output = layers.Conv3D(1, 1, activation='sigmoid', name='output')(x)
    
    model = Model(inputs, output, name="SmartSOTA_Cropped_2025")
    logger.info(f"✅ Built model with {model.count_params()/1e6:.2f}M parameters")
    return model

# ============================================================================
# LOSS FUNCTIONS & METRICS (SAME AS WORKING VERSION)
# ============================================================================

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    intersection = tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    return (2. * intersection + smooth) / denominator

def dice_loss(y_true, y_pred, smooth=1e-6):
    return 1 - dice_coefficient(y_true, y_pred, smooth)

def boundary_weighted_loss(y_true, y_pred, alpha=0.7, beta=0.3, sigma=1.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # FIXED: Ensure 5D tensors throughout
    if len(tf.shape(y_true)) == 4:
        y_true = tf.expand_dims(y_true, axis=1)
    if len(tf.shape(y_pred)) == 4:
        y_pred = tf.expand_dims(y_pred, axis=1)
    
    kernel_arr = np.zeros((3, 3, 3), dtype=np.float32)
    kernel_arr[1, 1, 1] = 8
    kernel_arr[0, 1, 1] = kernel_arr[2, 1, 1] = -1
    kernel_arr[1, 0, 1] = kernel_arr[1, 2, 1] = -1
    kernel_arr[1, 1, 0] = kernel_arr[1, 1, 2] = -1

    kernel = tf.constant(kernel_arr, dtype=tf.float32)
    kernel = tf.expand_dims(tf.expand_dims(kernel, -1), -1)

    edges_true = tf.nn.conv3d(y_true, kernel, strides=[1,1,1,1,1], padding='SAME')
    edges_pred = tf.nn.conv3d(y_pred, kernel, strides=[1,1,1,1,1], padding='SAME')

    term1 = tf.exp(-tf.square(edges_true) / (sigma**2))
    term2 = tf.exp(-tf.square(edges_pred) / (sigma**2))
    weight_map = 1.0 + alpha * term1 + beta * term2

    epsilon = 1e-7
    y_pred_clipped = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    bce = - (y_true * tf.math.log(y_pred_clipped) +
             (1 - y_true) * tf.math.log(1 - y_pred_clipped))

    weighted_bce = weight_map * bce
    weighted_bce = tf.reduce_mean(weighted_bce)

    return weighted_bce + dice_loss(y_true, y_pred)

def dynamic_loss(y_true, y_pred, config):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    dice = dice_loss(y_true, y_pred)
    
    if config.USE_BOUNDARY_LOSS:
        try:
            boundary = boundary_weighted_loss(y_true, y_pred)
            dice_weight = tf.cast(config.DICE_LOSS_WEIGHT, tf.float32)
            boundary_weight = tf.cast(config.BOUNDARY_LOSS_WEIGHT, tf.float32)
            return dice_weight * dice + boundary_weight * boundary
        except:
            return dice
    return dice

# ============================================================================
# CALLBACKS & LEARNING RATE SCHEDULE (SAME AS WORKING VERSION)
# ============================================================================

class WarmupCosineDecay(tf.keras.callbacks.Callback):
    def __init__(self, warmup_epochs, total_epochs, initial_lr, min_lr, verbose=0):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.verbose = verbose
        
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            decay_epochs = self.total_epochs - self.warmup_epochs
            epoch_in_decay = epoch - self.warmup_epochs
            cosine_decay = 0.5 * (1 + math.cos(math.pi * epoch_in_decay / decay_epochs))
            lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay
            
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        
        if self.verbose:
            logger.info(f"🔥 Epoch {epoch+1}: LR = {lr:.2e}")

# ============================================================================
# MAIN TRAINING SCRIPT (ADAPTED FOR CROPPED DATASET)
# ============================================================================

def main():
    """Memory-optimized training pipeline for cropped dataset"""
    
    def detailed_crash_handler(exc_type, exc_value, exc_traceback):
        crash_info = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'exception_type': str(exc_type.__name__),
            'exception_message': str(exc_value),
            'traceback': traceback.format_exception(exc_type, exc_value, exc_traceback),
            'memory_info': None,
            'gpu_info': None
        }
        
        try:
            process = psutil.Process(os.getpid())
            crash_info['memory_info'] = {
                'cpu_memory_gb': process.memory_info().rss / 1024**3,
                'cpu_memory_percent': process.memory_percent(),
                'available_memory_gb': psutil.virtual_memory().available / 1024**3
            }
            
            gpu_info = []
            try:
                for i in range(4):
                    alloc = tf.config.experimental.get_memory_info(f'GPU:{i}')
                    gpu_info.append({
                        'gpu_id': i,
                        'current_mb': alloc['current'] / 1e6,
                        'peak_mb': alloc['peak'] / 1e6
                    })
                crash_info['gpu_info'] = gpu_info
            except:
                crash_info['gpu_info'] = "GPU info unavailable"
                
        except Exception as e:
            crash_info['memory_capture_error'] = str(e)
        
        with open("detailed_crash_report_cropped.json", "w") as f:
            json.dump(crash_info, f, indent=2, default=str)
        
        with open("crash_report_cropped.txt", "w") as f:
            f.write(f"CROPPED DATASET CRASH REPORT - {crash_info['timestamp']}\n")
            f.write("="*50 + "\n\n")
            f.write(f"Exception: {crash_info['exception_type']}\n")
            f.write(f"Message: {crash_info['exception_message']}\n\n")
            f.write("Memory State:\n")
            if crash_info['memory_info']:
                f.write(f"  CPU Memory Used: {crash_info['memory_info']['cpu_memory_gb']:.2f}GB\n")
                f.write(f"  CPU Memory %: {crash_info['memory_info']['cpu_memory_percent']:.1f}%\n")
                f.write(f"  Available Memory: {crash_info['memory_info']['available_memory_gb']:.2f}GB\n")
            f.write("\nFull Traceback:\n")
            f.write(''.join(crash_info['traceback']))
        
        logger.critical(f"💥 CROPPED DATASET CRASH: {crash_info['exception_type']} - {crash_info['exception_message']}")
        logger.critical("📄 Detailed crash reports saved to detailed_crash_report_cropped.json and crash_report_cropped.txt")
        sys.exit(1)
    
    sys.excepthook = detailed_crash_handler
    
    # INITIALIZATION
    logger.info("🔍 CROPPED DATASET TRAINING: Main function started")
    logger.info(f"🔍 Current working directory: {os.getcwd()}")
    
    config = CroppedTrainingConfig()
    
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.CALLBACKS_DIR, exist_ok=True)
    
    log_memory_usage("cropped_initialization_start")
    
    # GPU SETUP
    try:
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"✅ GPU memory growth enabled for {len(gpus)} devices")
    except Exception as e:
        logger.warning(f"⚠️ GPU setup issue: {e}")

    # STRATEGY SETUP
    try:
        # Use single GPU strategy for stability
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        logger.info(f"✅ Using single GPU strategy for cropped dataset training")
        
        log_memory_usage("strategy_setup_complete")
        
    except Exception as e:
        logger.error(f"❌ Strategy setup failed: {str(e)}")
        sys.exit(1)

    # DATA LOADING
    try:
        logger.info("🔄 Loading and preparing cropped dataset...")
        
        train_gen, val_gen = create_optimized_generators(config)
        
        logger.info("🧪 Testing cropped data generators...")
        test_batch_x, test_batch_y = train_gen[0]
        logger.info(f"✅ Generator test successful: "
                   f"X shape: {test_batch_x.shape}, Y shape: {test_batch_y.shape}")
        logger.info(f"📊 Input shape confirmed: {config.INPUT_SHAPE}")
        
        del test_batch_x, test_batch_y
        gc.collect()
        
        log_memory_usage("cropped_data_loading_complete")
        
    except Exception as e:
        logger.error(f"❌ Cropped data loading failed: {str(e)}", exc_info=True)
        sys.exit(1)

    # MODEL BUILDING
    with strategy.scope():
        try:
            logger.info("🏗️ Building model for cropped dataset...")
            model = build_smart_sota(config)
            
            initial_lr = config.INITIAL_LR / strategy.num_replicas_in_sync
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=initial_lr,
                epsilon=1e-7,
                clipnorm=config.MAX_GRAD_NORM
            )
            
            def compiled_loss(y_true, y_pred):
                if isinstance(y_pred, list):
                    return dice_loss(y_true, y_pred[0])
                else:
                    return dice_loss(y_true, y_pred)
            
            model.compile(
                optimizer=optimizer,
                loss=compiled_loss,
                metrics=[dice_coefficient]
            )

            logger.info(f"✅ Cropped model compiled: {model.count_params()/1e6:.2f}M parameters")
            log_memory_usage("cropped_model_build_complete")
            
        except Exception as e:
            logger.error(f"❌ Cropped model building failed: {str(e)}", exc_info=True)
            sys.exit(1)

    # CALLBACK SETUP
    try:
        callbacks = [
            WarmupCosineDecay(
                warmup_epochs=config.WARMUP_EPOCHS,
                total_epochs=config.TOTAL_EPOCHS,
                initial_lr=initial_lr,
                min_lr=config.MIN_LR,
                verbose=1
            ),
            
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(config.checkpoint_path),
                monitor='val_dice_coefficient',
                mode='max',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            tf.keras.callbacks.EarlyStopping(
                monitor='val_dice_coefficient',
                mode='max',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_dice_coefficient',
                mode='max',
                factor=0.5,
                patience=10,
                min_lr=config.MIN_LR,
                verbose=1
            ),
            
            MemoryMonitoringCallback(log_frequency=1),
            
            tf.keras.callbacks.CSVLogger(
                str(config.MODEL_DIR / 'training_history_cropped.csv'),
                append=True
            ),
            
            tf.keras.callbacks.TensorBoard(
                log_dir=str(config.MODEL_DIR / 'tensorboard_cropped'),
                histogram_freq=0,
                write_graph=False,
                update_freq='epoch'
            )
        ]
        
        logger.info(f"✅ Configured {len(callbacks)} callbacks for cropped training")
        
    except Exception as e:
        logger.error(f"❌ Callback setup failed: {str(e)}")
        sys.exit(1)

    # TRAINING EXECUTION
    try:
        logger.info("🚀 Starting cropped dataset training...")
        log_memory_usage("cropped_training_start")
        
        logger.info("🔍 Running pre-training validation...")
        val_loss = model.evaluate(val_gen, verbose=1, return_dict=True)
        logger.info(f"Pre-training validation: {val_loss}")
        
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=config.TOTAL_EPOCHS,
            initial_epoch=config.INITIAL_EPOCH,
            callbacks=callbacks,
            verbose=1,
            workers=1,
            use_multiprocessing=False,
            max_queue_size=2
        )
        
        log_memory_usage("cropped_training_complete")
        logger.info("✅ Cropped dataset training completed successfully!")
        
        final_model_path = config.MODEL_DIR / f"final_cropped_model_{config.timestamp}.keras"
        model.save(str(final_model_path))
        logger.info(f"💾 Final cropped model saved: {final_model_path}")
        
        history_path = config.MODEL_DIR / f"history_cropped_{config.timestamp}.json"
        with open(history_path, 'w') as f:
            history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
            json.dump(history_dict, f, indent=2)
        logger.info(f"📊 Cropped training history saved: {history_path}")
        
    except Exception as e:
        logger.error(f"❌ Cropped training failed: {str(e)}", exc_info=True)
        log_memory_usage("cropped_training_failed")
        
        try:
            emergency_save_path = config.MODEL_DIR / f"emergency_cropped_save_{config.timestamp}.keras"
            model.save(str(emergency_save_path))
            logger.info(f"💾 Emergency cropped model save: {emergency_save_path}")
        except:
            logger.error("Failed to save emergency cropped model")
        
        sys.exit(1)

    # POST-TRAINING EVALUATION
    try:
        logger.info("📊 Running final cropped evaluation...")
        
        best_model = tf.keras.models.load_model(
            str(config.checkpoint_path),
            custom_objects={
                'dice_coefficient': dice_coefficient,
                'compiled_loss': lambda y_true, y_pred: dice_loss(y_true, y_pred)
            }
        )
        
        final_metrics = best_model.evaluate(val_gen, verbose=1, return_dict=True)
        logger.info(f"🎯 Final cropped validation metrics: {final_metrics}")
        
        eval_results = {
            'timestamp': config.timestamp,
            'dataset_type': 'cropped',
            'input_shape': config.INPUT_SHAPE,
            'final_metrics': {k: float(v) for k, v in final_metrics.items()},
            'config': {k: v for k, v in config.__dict__.items() 
                      if not k.startswith('__') and not callable(v)},
            'training_summary': {
                'total_epochs': len(history.history['loss']),
                'best_val_dice': float(max(history.history.get('val_dice_coefficient', [0]))),
                'final_training_loss': float(history.history['loss'][-1]),
                'final_validation_loss': float(history.history['val_loss'][-1])
            }
        }
        
        eval_path = config.MODEL_DIR / f"evaluation_cropped_{config.timestamp}.json"
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f, indent=2, default=str)
        
        logger.info(f"📋 Cropped evaluation results saved: {eval_path}")
        
    except Exception as e:
        logger.error(f"❌ Post-training cropped evaluation failed: {str(e)}")

    # CLEANUP
    try:
        if hasattr(train_gen, '__del__'):
            train_gen.__del__()
        if hasattr(val_gen, '__del__'):
            val_gen.__del__()
        
        tf.keras.backend.clear_session()
        gc.collect()
        
        log_memory_usage("cropped_cleanup_complete")
        logger.info("🧹 Cropped training cleanup completed")
        
    except Exception as e:
        logger.warning(f"⚠️ Cropped cleanup issues: {str(e)}")

    logger.info("🎉 Cropped dataset training pipeline completed successfully!")

if __name__ == "__main__":
    main()
