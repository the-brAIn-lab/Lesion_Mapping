"""
T1-weighted Stroke Lesion Segmentation (Dynamic Input Version)

This training script builds upon prior production and cropped variants but adds
support for arbitrary volumetric input sizes. It automatically determines
the largest spatial dimensions present in a dataset and pads smaller volumes
so that all inputs share the same shape. The script retains detailed
logging, memory monitoring, data augmentation and custom layers from the
previous versions while incorporating recommendations from the latest model
evaluation:

* Dice/boundary loss weights adjusted to emphasise boundary precision
* Over-segmentation mitigation via adjustable decision threshold
* Slightly stronger augmentation (rotations/flips/gamma) when overfitting
  is suspected
* Tunable L2 regularisation and dropout rates
* Longer warm-up and lower minimum learning rate
* Works out-of-the-box with T1-weighted MRI volumes (e.g., MNI-normalised ARC data)
  while remaining configurable through DynamicTrainingConfig.
"""

import os
import sys
import logging
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

# ---- Environment (set BEFORE importing TensorFlow) ----
import os

# Keep: quiet logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Optional: better GPU allocator (helps reduce fragmentation on long runs)
# Works with TF 2.10+ built for CUDA 11/12.
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# Don't set for normal training:
# - CUDA_LAUNCH_BLOCKING=1  # debug-only; forces sync and can make training very slow
# - TF_XLA_FLAGS / XLA_FLAGS  # generally unnecessary on TF 2.20; can cause confusion
# - TF_ENABLE_ONEDNN_OPTS=0  # controls CPU-only kernels; leave default unless you need bit-for-bit CPU numerics


gpus = tf.config.list_physical_devices("GPU")
print("Visible GPUs:", gpus)
for gpu in gpus:
    try: tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e: print(f"Could not set memory growth on {gpu}: {e}")

# ✅ Only mirror if multi-GPU
strategy = tf.distribute.MirroredStrategy() if len(gpus) > 1 else tf.distribute.get_strategy()
print("Strategy:", type(strategy).__name__)







# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
# Use two file handlers and one stream handler.  One file captures all
# high‑level events (INFO and above) and the other captures per‑process
# debugging output.  A console stream is kept for quick feedback.

from pathlib import Path

LOG_DIR_ENV = os.environ.get("SMARTSOTA_LOG_DIR")
LOG_DIR = Path(LOG_DIR_ENV).expanduser() if LOG_DIR_ENV else (Path(__file__).resolve().parent / "logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'smart_sota_dynamic.log'),
        logging.FileHandler(LOG_DIR / f'training_dynamic_{os.getpid()}.debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('SmartSOTA_Dynamic')
logger.setLevel(logging.DEBUG)

# ---------------------------------------------------------------------------
# Imports with graceful degradation
# ---------------------------------------------------------------------------
try:
    import tensorflow as tf
    from tensorflow.keras import layers
    import numpy as np
    import nibabel as nib
    from scipy.ndimage import zoom, binary_dilation, rotate
    from sklearn.model_selection import KFold
    from skimage import measure
    import json
    import time
    import math
    import random
    import gc
    import psutil
    from functools import lru_cache
    from concurrent.futures import ThreadPoolExecutor
    logger.info("✅ All imports successful")
except ImportError as e:
    logger.critical(f"❌ Import failed: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------
# Disable eager execution for performance and to avoid certain CUDNN issues.
tf.config.run_functions_eagerly(False)
logger.info(f"TensorFlow eager execution: {tf.executing_eagerly()}")

warnings_to_ignore = [UserWarning, DeprecationWarning, FutureWarning]
for w in warnings_to_ignore:
    tf.autograph.set_verbosity(0)
import warnings
for w in warnings_to_ignore:
    warnings.filterwarnings("ignore", category=w)
warnings.filterwarnings("ignore", module="nibabel")

logger.info(
    f"Environment verified:\n"
    f"- Python {sys.version}\n"
    f"- TensorFlow {tf.__version__}\n"
    f"- NumPy {np.__version__}\n"
    f"- GPU devices: {len(tf.config.list_physical_devices('GPU'))}"
)

# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
from dataclasses import dataclass, field, asdict
from typing import Optional

def _env_path(name: str) -> Optional[Path]:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None

def _default_dir(env_var: str, fallback: Path) -> Path:
    value = os.environ.get(env_var)
    return Path(value).expanduser() if value else fallback

@dataclass
class DynamicTrainingConfig:
    DATA_DIR: Optional[Path] = field(default_factory=lambda: _env_path("SMARTSOTA_DATA_DIR"))
    IMAGES_DIR: Optional[Path] = None
    MASKS_DIR: Optional[Path] = None
    IMAGE_SUFFIXES: tuple[str, ...] = (
        "_T1w_MNI_norm", "_T1w_MNI", "_T1w_brain", "_T1w", "_T1",
    )
    MASK_SUFFIXES: tuple[str, ...] = (
        "_lesion_mask_MNI_clean", "_lesion_mask_MNI", "_lesion_mask",
        "_desc-lesion_mask", "_mask",
    )
    INPUT_SHAPE: Optional[tuple[int, int, int, int]] = None
    VALIDATION_SPLIT: float = 0.15
    SMALL_LESION_THRESHOLD: int = 100
    BATCH_SIZE: int = 2
    INITIAL_EPOCH: int = 0
    TOTAL_EPOCHS: int = 200
    INITIAL_LR: float = 1e-4
    MIN_LR: float = 5e-7
    WARMUP_EPOCHS: int = 15
    MAX_GRAD_NORM: float = 1.0
    BASE_FILTERS: int = 8
    DROPOUT_RATE: float = 0.55
    L2_REG: float = 1.5e-3
    MAMBA_DEPTH: int = 2
    SAM_HEADS: int = 4
    AUGMENTATION_INTENSITY: float = 0.5
    SYNTHETIC_LESION_PROB: float = 0.3
    ROTATION_RANGE: int = 20
    USE_BOUNDARY_LOSS: bool = True
    DICE_LOSS_WEIGHT: float = 0.4
    BOUNDARY_LOSS_WEIGHT: float = 0.6
    DEEP_SUPERVISION_WEIGHTS: tuple[float, ...] = (0.1, 0.2, 0.3)
    DICE_WEIGHT: float = 0.4
    BOUNDARY_WEIGHT: float = 0.6
    RESAMPLE_TO_TARGET: bool = True
    DECISION_THRESHOLD: float = 0.5
    MODEL_DIR: Path = field(default_factory=lambda: _default_dir("SMARTSOTA_MODEL_DIR", Path.cwd() / "models"))
    CALLBACKS_DIR: Path = field(default_factory=lambda: _default_dir("SMARTSOTA_CALLBACK_DIR", Path.cwd() / "callbacks"))
    timestamp: str = field(default_factory=lambda: time.strftime("%Y%m%d_%H%M%S"), init=False)

    def __post_init__(self):
        if self.DATA_DIR is None:
            raise ValueError("DATA_DIR must be supplied via argument or SMARTSOTA_DATA_DIR environment variable.")
        self.DATA_DIR = Path(self.DATA_DIR)
        self.IMAGES_DIR = Path(self.IMAGES_DIR) if self.IMAGES_DIR else self.DATA_DIR
        self.MASKS_DIR = Path(self.MASKS_DIR) if self.MASKS_DIR else self.DATA_DIR
        self.MODEL_DIR = Path(self.MODEL_DIR)
        self.CALLBACKS_DIR = Path(self.CALLBACKS_DIR)
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.CALLBACKS_DIR.mkdir(parents=True, exist_ok=True)
        self._write_config()

    def _write_config(self) -> None:
        payload = asdict(self)
        for key in ("DATA_DIR", "IMAGES_DIR", "MASKS_DIR", "MODEL_DIR", "CALLBACKS_DIR"):
            payload[key] = str(payload[key])
        if payload["INPUT_SHAPE"] is not None:
            payload["INPUT_SHAPE"] = list(payload["INPUT_SHAPE"])
        payload["IMAGE_SUFFIXES"] = list(payload["IMAGE_SUFFIXES"])
        payload["MASK_SUFFIXES"] = list(payload["MASK_SUFFIXES"])
        with open(self.MODEL_DIR / "config.json", "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    @property
    def model_path(self) -> Path:
        return self.MODEL_DIR / f"smart_sota_dynamic_{self.timestamp}.keras"

    @property
    def checkpoint_path(self) -> Path:
        return self.CALLBACKS_DIR / "best_model_dynamic.weights.h5"
    
    

    

# ---------------------------------------------------------------------------
# Custom layers (identical to previous implementations)
# ---------------------------------------------------------------------------
# put this once near the top (same place you imported for losses)
try:
    from keras.saving import register_keras_serializable
except Exception:
    from tensorflow.keras.utils import register_keras_serializable  # fallback


# --- ResidualConvBlock: force LN in float32 ---
@register_keras_serializable(package="custom")
class ResidualConvBlock(layers.Layer):
    def __init__(self, filters, kernel_reg=None, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_reg = kernel_reg

    def build(self, input_shape):
        self.conv1 = layers.Conv3D(self.filters, 3, padding='same',
                                   kernel_regularizer=self.kernel_reg)
        # LN in float32 for stability under mixed precision
        self.ln1 = layers.LayerNormalization(epsilon=1e-5, dtype="float32")

        self.conv2 = layers.Conv3D(self.filters, 3, padding='same',
                                   kernel_regularizer=self.kernel_reg)
        self.ln2 = layers.LayerNormalization(epsilon=1e-5, dtype="float32")

        self.dropout = layers.SpatialDropout3D(0.1)

        self.residual_conv = layers.Conv3D(self.filters, 1, padding='same')
        self.residual_ln = layers.LayerNormalization(epsilon=1e-5, dtype="float32")
        super().build(input_shape)

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.ln1(x)                         # fp32 here
        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)
        x = self.conv2(x)
        x = self.ln2(x)                         # fp32 here
        x = tf.cast(x, inputs.dtype)            # cast back to match inputs (fp16)

        residual = self.residual_conv(inputs)
        residual = self.residual_ln(residual)   # fp32 here
        residual = tf.cast(residual, inputs.dtype)

        return tf.nn.relu(x + residual)


    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_reg": tf.keras.regularizers.serialize(self.kernel_reg)
                           if self.kernel_reg else None
        })
        return config



@register_keras_serializable(package="custom")
class VisionMambaBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, expansion=2, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.expansion = expansion

    def build(self, input_shape):
        self.in_conv = layers.Conv3D(self.filters * self.expansion, 1, use_bias=False, padding='same')
        self.spatial_conv = layers.Conv3D(self.filters * self.expansion, self.kernel_size, padding='same', use_bias=False)
        self.out_conv = layers.Conv3D(self.filters, 1, padding='same')
        # LN in fp32, then cast back
        self.norm = layers.LayerNormalization(epsilon=1e-5, dtype="float32")
        self.dropout = layers.SpatialDropout3D(0.1)
        super().build(input_shape)

    def call(self, inputs, training=None):
        x = self.in_conv(inputs)
        x = tf.nn.relu(x)
        x = self.spatial_conv(x)
        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)
        x = self.out_conv(x)
        x = self.norm(x)                 # fp32
        x = tf.cast(x, inputs.dtype)     # back to fp16
        return x + inputs


    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "expansion": self.expansion
        })
        return config

@register_keras_serializable(package="custom")
class SAM2Attention(layers.Layer):
    def __init__(self, filters, heads, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.heads = heads
        self.depth = filters // heads
        if filters % heads != 0:
            raise ValueError("Filters must be divisible by heads")

    def build(self, input_shape):
        self.query = layers.Conv3D(self.filters, 1, data_format='channels_last')
        self.key   = layers.Conv3D(self.filters, 1, data_format='channels_last')
        self.value = layers.Conv3D(self.filters, 1, data_format='channels_last')
        self.out_conv = layers.Conv3D(input_shape[-1], 1, data_format='channels_last')
        self.memory_bank = self.add_weight(
            name='memory_bank',
            shape=(1, 1, 1, 1, self.filters),
            initializer='zeros',
            trainable=True
        )
        self.dropout = layers.SpatialDropout3D(0.1)
        super().build(input_shape)

    def call(self, inputs, training=None):
        b = tf.shape(inputs)[0]
        h = tf.shape(inputs)[1]
        w = tf.shape(inputs)[2]
        d = tf.shape(inputs)[3]

        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        # ensure memory_bank matches compute dtype (fp16 under mixed precision)
        mb = tf.cast(self.memory_bank, k.dtype)
        k = k + mb
        v = v + mb

        q = self._split_heads_safe(q, b, h, w, d)
        k = self._split_heads_safe(k, b, h, w, d)
        v = self._split_heads_safe(v, b, h, w, d)

        dk = tf.cast(self.depth, q.dtype)
        attn_logits = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(dk)
        attn_weights = tf.nn.softmax(attn_logits, axis=-1)
        attn_output = tf.matmul(attn_weights, v)
        attn_output = self._combine_heads_safe(attn_output, b, h, w, d)

        output = self.out_conv(attn_output)
        output = self.dropout(output, training=training)

        # cast back so residual add matches inputs dtype (fp16)
        output = tf.cast(output, inputs.dtype)
        return output + inputs

    def _split_heads_safe(self, x, b, h, w, d):
        x = tf.reshape(x, [b, h, w, d, self.heads, self.depth])
        return tf.transpose(x, perm=[0, 4, 1, 2, 3, 5])

    def _combine_heads_safe(self, x, b, h, w, d):
        x = tf.transpose(x, perm=[0, 2, 3, 4, 1, 5])
        return tf.reshape(x, [b, h, w, d, self.filters])


    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "heads": self.heads
        })
        return config

# ---------------------------------------------------------------------------
# Build the segmentation model (UNet-like with your custom blocks)
# ---------------------------------------------------------------------------
def build_dynamic_model(config: DynamicTrainingConfig) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=config.INPUT_SHAPE)  # (D,H,W,1)

    x = inputs
    skips = []
    filters = config.BASE_FILTERS

    # Encoder
    for _ in range(4):
        x = ResidualConvBlock(filters, kernel_reg=tf.keras.regularizers.l2(config.L2_REG))(x)
        x = VisionMambaBlock(filters)(x)
        skips.append(x)
        x = layers.MaxPool3D(pool_size=2)(x)
        filters *= 2

    # Bottleneck
    x = ResidualConvBlock(filters, kernel_reg=tf.keras.regularizers.l2(config.L2_REG))(x)
    x = SAM2Attention(filters, heads=config.SAM_HEADS)(x)

    # Decoder
    for d in reversed(range(4)):
        filters //= 2
        x = layers.UpSampling3D(size=2)(x)
        x = layers.Concatenate()([x, skips[d]])
        x = ResidualConvBlock(filters, kernel_reg=tf.keras.regularizers.l2(config.L2_REG))(x)
        x = VisionMambaBlock(filters)(x)

    # IMPORTANT: output logits (no activation). Dice in your loss applies sigmoid.
    # Build model – replace the head
    outputs = layers.Conv3D(1, kernel_size=1, activation="sigmoid", name="probs")(x)


    return tf.keras.Model(inputs=inputs, outputs=outputs, name="SmartSOTA_Dynamic")



# ---------------------------------------------------------------------------
# Utility functions for memory monitoring
# ---------------------------------------------------------------------------
def log_memory_usage(stage: str) -> None:
    process = psutil.Process(os.getpid())
    gb_used = process.memory_info().rss / 1024**3
    gpu_mem = []
    try:
        for i in range(4):
            alloc = tf.config.experimental.get_memory_info(f'GPU:{i}')
            gpu_mem.append(f"GPU{i}: {alloc['current']/1e9:.2f}GB")
    except Exception:
        gpu_mem = ["GPU mem tracking failed"]
    try:
        disk_usage = psutil.disk_usage('/')
        disk_free_gb = disk_usage.free / 1024**3
        disk_info = f"Disk: {disk_free_gb:.1f}GB free"
    except Exception:
        disk_info = "Disk: unavailable"
    logger.info(f"Memory at {stage}: CPU={gb_used:.2f}GB | {' | '.join(gpu_mem)} | {disk_info}")


class MemoryMonitoringCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_frequency=10):
        super().__init__()
        self.log_frequency = int(log_frequency)
        self._batch = 0

    def on_train_begin(self, logs=None):
        log_memory_usage("train_begin")

    def on_epoch_begin(self, epoch, logs=None):
        log_memory_usage(f"epoch_{epoch}_start")

    def on_train_batch_end(self, batch, logs=None):
        self._batch += 1
        if self._batch % self.log_frequency == 0:
            log_memory_usage(f"batch_{self._batch}")

    def on_epoch_end(self, epoch, logs=None):
        log_memory_usage(f"epoch_{epoch}_end")


# ---------------------------------------------------------------------------
# Losses & metrics (expects sigmoid outputs)
# ---------------------------------------------------------------------------
@register_keras_serializable(package="custom")
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7), tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    denom = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return (2.0 * intersection + smooth) / (denom + smooth)

@register_keras_serializable(package="custom")
def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)

def _sobel_3d(t):
    t = tf.cast(t, tf.float32)
    k = tf.constant([1., 2., 1.], dtype=tf.float32)
    d = tf.constant([-1., 0., 1.], dtype=tf.float32)

    def make_kernel(axis):
        if axis == "x":
            kx, ky, kz = d, k, k
        elif axis == "y":
            kx, ky, kz = k, d, k
        else:
            kx, ky, kz = k, k, d
        filt = tf.einsum("i,j,k->ijk", kz, ky, kx)[:, :, :, tf.newaxis, tf.newaxis] / 32.0
        return tf.cast(filt, tf.float32)

    fx, fy, fz = make_kernel("x"), make_kernel("y"), make_kernel("z")
    gx = tf.nn.conv3d(t, fx, strides=[1, 1, 1, 1, 1], padding="SAME")
    gy = tf.nn.conv3d(t, fy, strides=[1, 1, 1, 1, 1], padding="SAME")
    gz = tf.nn.conv3d(t, fz, strides=[1, 1, 1, 1, 1], padding="SAME")
    return gx, gy, gz

@register_keras_serializable(package="custom")
def boundary_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7), tf.float32)
    gxt, gyt, gzt = _sobel_3d(y_true)
    gxp, gyp, gzp = _sobel_3d(y_pred)
    grad_true = tf.sqrt(gxt**2 + gyt**2 + gzt**2 + 1e-7)
    grad_pred = tf.sqrt(gxp**2 + gyp**2 + gzp**2 + 1e-7)
    return tf.reduce_mean(tf.abs(grad_true - grad_pred))

@register_keras_serializable(package="custom")
class CombinedLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.4, beta=0.6, name="combined_loss"):
        super().__init__(name=name)
        self.alpha = float(alpha)
        self.beta = float(beta)

    def get_config(self):
        return {"alpha": self.alpha, "beta": self.beta}

    def call(self, y_true, y_pred):
        return self.alpha * dice_loss(y_true, y_pred) + self.beta * boundary_loss(y_true, y_pred)


# ---------------------------------------------------------------------------
# Dataset inspection and loading
# ---------------------------------------------------------------------------
from pathlib import Path
import numpy as np
import nibabel as nib
import gc



def detect_input_shape(data_dir: Path) -> tuple:
    """
    Determine the maximum spatial (D,H,W) across NIfTI volumes under `data_dir`,
    then round each dimension UP to the nearest multiple of 16.

    We consider any .nii.gz with at least 3 dims. If none are valid, an error is raised.
    Logs fall back to print() if a global `logger` isn't available.
    """
    import math
    import nibabel as nib

    log = globals().get("logger", None)
    def _info(msg: str):
        if log is not None:
            log.info(msg)
        else:
            print(msg)

    _info("🔍 Detecting input shape from dataset…")

    # Scan all NIfTI files under the root (Images/Masks are fine; we only read headers/shapes)
    image_files = list(data_dir.rglob("*.nii.gz"))
    max_shape = [0, 0, 0]
    invalid = []

    if not image_files:
        raise FileNotFoundError(f"No .nii.gz files found under {data_dir}")

    for f in image_files:
        try:
            img = nib.load(str(f))
            shp = img.shape
            # Need at least 3 spatial dims
            if len(shp) >= 3:
                for i in range(3):
                    max_shape[i] = max(max_shape[i], int(shp[i]))
            else:
                invalid.append(f"{f.name}: shape {shp} has fewer than 3 dims")
        except Exception as e:
            invalid.append(f"{f.name}: failed to load ({e})")

    if all(dim == 0 for dim in max_shape):
        details = ("Issues encountered:\n  - " + "\n  - ".join(invalid)) if invalid else "No details."
        raise RuntimeError(f"No valid 3-D NIfTI files found in {data_dir}. {details}")

    def _ceil16(x: int) -> int:
        return int(math.ceil(x / 16.0) * 16)

    rounded_shape = tuple(_ceil16(dim) for dim in max_shape)

    _info(
        f"📐 Detected max volume dimensions: {tuple(max_shape)} → "
        f"rounded up to: {rounded_shape}"
    )
    return rounded_shape


0
# --- Flexible loader: supports single-folder (preprocessed) or two-folder (raw) ---
import gc, re, json
import numpy as np
import nibabel as nib
from pathlib import Path

class DynamicDataGenerator(tf.keras.utils.Sequence):
    """Dynamic data generator for 3D medical volumes with optional augmentation.
    
    Implements the Keras Sequence interface for memory-efficient loading
    and preprocessing of 3D medical image volumes and their corresponding masks.
    """
    
    def __init__(self, pairs, config, is_training=False):
        """Initialize the generator with pairs of image/mask paths and config.
        
        Args:
            pairs: List of (image_path, mask_path) tuples.
            config: DynamicTrainingConfig object with parameters.
            is_training: If True, apply augmentation.
        """
        self.pairs = pairs
        self.config = config
        self.batch_size = config.BATCH_SIZE
        self.target_shape = tuple(config.INPUT_SHAPE[:-1])
        self.config = config
        self.is_training = is_training
        self.indexes = np.arange(len(self.pairs))
        self.resample_to_target = bool(config.RESAMPLE_TO_TARGET)
        
        # Shuffle at initialization
        if self.is_training:
            random.shuffle(self.pairs)
    
    def __len__(self):
        """Return the number of batches per epoch."""
        return math.ceil(len(self.pairs) / self.batch_size)
    
    def __getitem__(self, idx):
        """Get a batch of data."""
        batch_pairs = self.pairs[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.zeros((len(batch_pairs), *self.config.INPUT_SHAPE), dtype=np.float32)
        batch_y = np.zeros((len(batch_pairs), *self.config.INPUT_SHAPE), dtype=np.float32)
        
        for i, (img_path, msk_path) in enumerate(batch_pairs):
            # Load and preprocess image and mask
            img = _load_and_preprocess_image(str(img_path), self.target_shape)
            msk = _load_and_preprocess_mask(str(msk_path), self.target_shape)
            
            # Apply augmentations if in training mode
            if self.is_training and self.config.AUGMENTATION_INTENSITY > 0:
                img, msk = self._augment(img, msk)
            
            # Add channel dimension
            batch_x[i, ..., 0] = img
            batch_y[i, ..., 0] = msk
            
        return batch_x, batch_y
    
    def on_epoch_end(self):
        """Called at the end of each epoch."""
        if self.is_training:
            random.shuffle(self.pairs)
    
    def _augment(self, image, mask):
        """Apply augmentations to image and mask."""
        # Skip augmentation based on probability
        if random.random() > self.config.AUGMENTATION_INTENSITY:
            return image, mask
        
        # Random flips
        if random.random() > 0.5:
            image = np.flip(image, axis=0)
            mask = np.flip(mask, axis=0)
        if random.random() > 0.5:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)
            
        # Random rotation (limited to rotation_range degrees)
        if random.random() > 0.7:
            angle = random.uniform(-self.config.ROTATION_RANGE, self.config.ROTATION_RANGE)
            # Random rotation axis (0, 1, or 2)
            axis = random.randint(0, 2)
            axes = [(0, 1), (0, 2), (1, 2)][axis]
            image = rotate(image, angle, axes=axes, reshape=False, order=1, mode='constant')
            mask = rotate(mask, angle, axes=axes, reshape=False, order=0, mode='constant')
            # Ensure mask remains binary
            mask = (mask > 0.5).astype(np.float32)
        
        # Random gamma correction (image only)
        if random.random() > 0.8:
            gamma = random.uniform(0.7, 1.3)
            image_max = image.max()
            if image_max > 0:
                image = np.power(image / image_max, gamma) * image_max
        
        return image, mask

def load_generic_dataset(config: DynamicTrainingConfig):
    logger.info("📚 Loading dataset (flex loader for T1w volumes)…")
    log_memory_usage("dataset_load_start")

    images_dir = config.IMAGES_DIR
    masks_dir = config.MASKS_DIR
    single_folder_mode = images_dir == masks_dir

    image_suffixes = config.IMAGE_SUFFIXES
    mask_suffixes = config.MASK_SUFFIXES
    cleanup_suffixes = ("_img_prepped", "_mask_prepped", "_image", "_img")

    def strip_ext(name: str) -> str:
        if name.endswith(".nii.gz"):
            return name[:-7]
        if name.endswith(".nii"):
            return name[:-4]
        return name


    def strip_any_suffix(stem: str, suffixes: tuple[str, ...]) -> str:
        s = stem
        # remove the longest matching suffix first
        for suf in sorted(suffixes, key=len, reverse=True):
            if s.endswith(suf):
                s = s[: -len(suf)]
                break
        # common trailing cleanup tokens
        for cleanup in ("_img_prepped", "_mask_prepped", "_image", "_img", "_clean", "_brain"):
            if s.endswith(cleanup):
                s = s[: -len(cleanup)]
        # remove optional MNI tags if they linger
        for tag in ("_MNI_norm", "_MNI"):
            if s.endswith(tag):
                s = s[: -len(tag)]
        return s.rstrip("_")

    def normalise_key(name: str, suffixes: tuple[str, ...]) -> str:
        stem = strip_ext(name)
        stem = strip_any_suffix(stem, suffixes)
        # final fallback: reduce to sub- and optional ses- components
        import re
        m_sub = re.search(r"(sub-[A-Za-z0-9]+)", stem)
        m_ses = re.search(r"(ses-[A-Za-z0-9]+)", stem)
        parts = [m_sub.group(1) if m_sub else None, m_ses.group(1) if m_ses else None]
        fallback = "_".join([p for p in parts if p])
        return fallback or stem


    def is_image(name: str) -> bool:
        stem = strip_ext(name).lower()
        return (any(stem.endswith(sfx.lower()) for sfx in image_suffixes)
                or ("mask" not in stem and "lesion" not in stem and ("t1w" in stem or stem.endswith("t1"))))

    def is_mask(name: str) -> bool:
        stem = strip_ext(name).lower()
        return (any(stem.endswith(sfx.lower()) for sfx in mask_suffixes)
                or "mask" in stem or "lesion" in stem)


    # Replace these two blocks:

    if single_folder_mode:
        # OLD:
        # all_niis = sorted(images_dir.glob("*.nii.gz"))
        # images = [p for p in all_niis if is_image(p.name)]
        # masks  = [p for p in all_niis if is_mask(p.name)]

        # NEW (recursive + both .nii.gz and .nii):
        all_niis = sorted(list(images_dir.rglob("*.nii.gz")) + list(images_dir.rglob("*.nii")))
        images = [p for p in all_niis if is_image(p.name)]
        masks  = [p for p in all_niis if is_mask(p.name)]
        logger.info(f"📁 Single-folder mode: {len(images)} images, {len(masks)} masks in {images_dir}")

    else:
        # OLD:
        # images = sorted([p for p in images_dir.glob("*.nii.gz") if is_image(p.name)])
        # masks  = sorted([p for p in masks_dir.glob("*.nii.gz")  if is_mask(p.name)])

        # NEW (recursive + both .nii.gz and .nii):
        images = sorted([p for p in list(images_dir.rglob("*.nii.gz")) + list(images_dir.rglob("*.nii")) if is_image(p.name)])
        masks  = sorted([p for p in list(masks_dir.rglob("*.nii.gz")) + list(masks_dir.rglob("*.nii"))  if is_mask(p.name)])
        logger.info(f"📂 Two-folder mode: images={len(images)} ({images_dir}), masks={len(masks)} ({masks_dir})")


    logger.info(f"Found {len(images)} image files and {len(masks)} mask files")

    img_map = {normalise_key(p.name, image_suffixes): p for p in images}
    msk_map = {normalise_key(p.name, mask_suffixes): p for p in masks}
    keys = sorted(set(k for k in img_map.keys() if k) & set(k for k in msk_map.keys() if k))

    if not keys:
        logger.error("No image–mask pairs matched. Examples:")
        for k, v in list(img_map.items())[:5]:
            logger.error(f"  IMG key {k or '<empty>'} -> {v.name}")
        for k, v in list(msk_map.items())[:5]:
            logger.error(f"  MSK key {k or '<empty>'} -> {v.name}")
        raise RuntimeError("No pairs matched. Check filename patterns / suffix lists in DynamicTrainingConfig.")

    pairs, lesion_counts = [], []
    for k in keys:
        img_p = img_map[k]
        msk_p = msk_map[k]
        try:
            mask_obj = nib.load(str(msk_p))
            has_lesion = bool(np.any(mask_obj.get_fdata() > 0))
            lesion_counts.append(1 if has_lesion else 0)
            pairs.append((img_p, msk_p))
        except Exception as e:
            logger.warning(f"Skipping pair for {k}: {e}")
        finally:
            try:
                del mask_obj
            except Exception:
                pass
            gc.collect()

    logger.info(f"📊 Created {len(pairs)} image–mask pairs")
    if lesion_counts:
        logger.info(f"🧠 Lesion presence: {np.mean(lesion_counts)*100:.2f}%")
    log_memory_usage("dataset_load_end")
    return pairs, np.array(lesion_counts, dtype=np.int32)

def create_stratified_splits(pairs, lesion_presence, batch_size, test_size=0.1):
    """Generate train/validation splits compatible with batch size.

    We adapt the original function to ensure that each split contains a number
    of samples divisible by the batch size.  Stratification is performed
    based on lesion presence.
    """
    total_samples = len(pairs)
    test_samples = math.floor(total_samples * test_size)
    train_samples = total_samples - test_samples
    # Round down to nearest batch size
    test_samples = (test_samples // batch_size) * batch_size
    train_samples = total_samples - test_samples
    # Guarantee at least one batch in each split
    if test_samples < batch_size:
        test_samples = batch_size
        train_samples = total_samples - test_samples
    if train_samples < batch_size:
        train_samples = batch_size
        test_samples = total_samples - train_samples
    logger.info(
        f"🧮 Dataset split: Train={train_samples}"
        f" ({train_samples/total_samples*100:.1f}%), "
        f"Validation={test_samples} ({test_samples/total_samples*100:.1f}%)"
    )
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(pairs, lesion_presence):
        if len(test_idx) >= test_samples:
            test_idx = test_idx[:test_samples]
            break
    train_pairs = [pairs[i] for i in train_idx]
    test_pairs = [pairs[i] for i in test_idx]
    train_lesions = np.mean([lesion_presence[i] for i in train_idx])
    test_lesions = np.mean([lesion_presence[i] for i in test_idx])
    logger.info(
        f"⚖️ Lesion representation: Train={train_lesions*100:.1f}%, "
        f"Validation={test_lesions*100:.1f}%"
    )
    return train_pairs, test_pairs

def pad_and_center_crop(volume: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Symmetrically pad (if smaller) or center-crop (if larger) a 3D volume to target_shape.
    Works for both images (float) and masks (binary/float). No interpolation is used.
    """
    assert volume.ndim == 3, f"Expected 3D volume, got {volume.ndim}D"
    z, y, x = volume.shape
    tz, ty, tx = target_shape
    out = volume

    # Center-crop if needed
    if z > tz:
        start = (z - tz) // 2
        out = out[start:start+tz, :, :]
        z = tz
    if y > ty:
        start = (y - ty) // 2
        out = out[:, start:start+ty, :]
        y = ty
    if x > tx:
        start = (x - tx) // 2
        out = out[:, :, start:start+tx]
        x = tx

    # Symmetric pad if needed
    pad_z = max(0, tz - z)
    pad_y = max(0, ty - y)
    pad_x = max(0, tx - x)
    if pad_z or pad_y or pad_x:
        pz0, pz1 = pad_z // 2, pad_z - pad_z // 2
        py0, py1 = pad_y // 2, pad_y - pad_y // 2
        px0, px1 = pad_x // 2, pad_x - pad_x // 2
        out = np.pad(out, ((pz0, pz1), (py0, py1), (px0, px1)), mode="constant", constant_values=0)
    return out

# --- Center-slice helpers (shared crop/pad for image & mask) -----------------
def compute_center_slices(in_shape, out_shape):
    """
    Return input slices that pick the centered sub-volume when cropping, or the
    full axis when padding. Use these slices for BOTH image and mask.
    """
    inD, inH, inW = in_shape
    outD, outH, outW = out_shape
    slices = []
    for i_len, o_len in zip((inD, inH, inW), (outD, outH, outW)):
        if i_len >= o_len:
            start = (i_len - o_len) // 2
            end   = start + o_len
            slices.append(slice(start, end))
        else:
            # padding case: take the whole input on that axis
            slices.append(slice(0, i_len))
    return tuple(slices)  # (sd, sh, sw)

def apply_center_crop_or_pad(vol, in_slices, out_shape):
    """
    Apply the provided input slices, then center-pad into out_shape.
    Use the SAME in_slices for image and mask to guarantee identical transform.
    """
    sub = vol[in_slices[0], in_slices[1], in_slices[2]]
    out = np.zeros(out_shape, dtype=vol.dtype)
    # center place the 'sub' into out
    offs = tuple((o - s) // 2 for s, o in zip(sub.shape, out_shape))
    out[
        offs[0]:offs[0]+sub.shape[0],
        offs[1]:offs[1]+sub.shape[1],
        offs[2]:offs[2]+sub.shape[2],
    ] = sub
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Data generator (no augmentations). Only resampling (optional) + center crop/pad.
# ---------------------------------------------------------------------------
import gc
from functools import lru_cache

import nibabel as nib
import numpy as np
import psutil
from scipy.ndimage import zoom
import tensorflow as tf

@lru_cache(maxsize=128)
def _load_vol_canonical(path: str) -> np.ndarray:
    """Load NIfTI as RAS-canonical and return float32 array."""
    img = nib.load(path)
    img = nib.as_closest_canonical(img)  # standardize orientation
    return img.get_fdata().astype(np.float32)

def _load_image(path: str) -> np.ndarray:
    return _load_vol_canonical(path)

def _load_mask_bin(path: str) -> np.ndarray:
    return (_load_vol_canonical(path) > 0.5).astype(np.float32)

def _center_crop_or_pad_volume(volume: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Center-crop or pad a 3D volume to the target_shape.
    Works for both images (float) and masks (binary/float). No interpolation is used.
    """
    assert volume.ndim == 3, f"Expected 3D volume, got {volume.ndim}D"
    z, y, x = volume.shape
    tz, ty, tx = target_shape

    # Center-crop if larger
    if z > tz:
        start = (z - tz) // 2
        volume = volume[start:start+tz, :, :]
    if y > ty:
        start = (y - ty) // 2
        volume = volume[:, start:start+ty, :]
    if x > tx:
        start = (x - tx) // 2
        volume = volume[:, :, start:start+tx]

    # Pad if smaller
    pad_z = max(0, tz - z)
    pad_y = max(0, ty - y)
    pad_x = max(0, tx - x)
    if pad_z > 0 or pad_y > 0 or pad_x > 0:
        padding = ((pad_z//2, pad_z-pad_z//2), (pad_y//2, pad_y-pad_y//2), (pad_x//2, pad_x-pad_x//2))
        volume = np.pad(volume, padding, mode="constant", constant_values=0)

    return volume.astype(np.float32)

def _load_and_preprocess_image(path: str, target_shape: tuple) -> np.ndarray:
    """Load and preprocess a single image volume."""
    volume = _load_image(path)
    return _center_crop_or_pad_volume(volume, target_shape)

def _load_and_preprocess_mask(path: str, target_shape: tuple) -> np.ndarray:
    """Load and preprocess a single mask volume."""
    volume = _load_mask_bin(path)
    return _center_crop_or_pad_volume(volume, target_shape)

def _generate_batch(pairs, target_shape):
    """Generate a batch of image and mask pairs."""
    for img_path, msk_path in pairs:
        img = _load_and_preprocess_image(img_path, target_shape)
        msk = _load_and_preprocess_mask(msk_path, target_shape)
        yield img, msk


class ProgressPrinter(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        import tensorflow as tf, numpy as np
        logs = logs or {}
        # get current LR robustly
        lr = self.model.optimizer.learning_rate
        try:
            lr = tf.keras.backend.get_value(lr)
        except Exception:
            try: lr = float(lr)
            except Exception: lr = np.nan
        print(
            f"Epoch {epoch+1}: "
            f"dice={logs.get('dice_coefficient', float('nan')):.4f} "
            f"val_dice={logs.get('val_dice_coefficient', float('nan')):.4f} "
            f"loss={logs.get('loss', float('nan')):.4f} "
            f"val_loss={logs.get('val_loss', float('nan')):.4f} "
            f"lr={lr:.2e}",
            flush=True
        )


# ---------------------------------------------------------------------------
# Unified, refactored training entrypoint
# ---------------------------------------------------------------------------
def train_dynamic_model(config: Optional[DynamicTrainingConfig] = None, **overrides):
    """
    Train SmartSOTA Dynamic UNet.
    - Supports overrides via kwargs (or pass a prebuilt `config`)
    - Resumes from best or latest weights
    - Uses ReduceLROnPlateau (mode='max' on val_dice_coefficient)
    - Writes history CSV and JSON, plus latest/best checkpoints
    """
    # --- peel off fit() kwargs so they don't end up in the dataclass
    fit_keys = ("steps_per_epoch", "validation_steps", "shuffle", "class_weight")
    fit_kwargs = {k: overrides.pop(k) for k in list(overrides.keys()) if k in fit_keys}

    # resume-only kwargs (not part of the dataclass)
    load_weights_from   = overrides.pop("LOAD_WEIGHTS_FROM", None)   # str | None
    resume_from_latest  = overrides.pop("RESUME_FROM_LATEST", False) # bool

    if config is not None and overrides:
        raise ValueError("Provide either an existing config or keyword overrides, not both.")
    if config is None:
        config = DynamicTrainingConfig(**overrides)

    # ---- basic env checks ----
    if not tf.config.list_physical_devices("GPU"):
        logger.critical("🚫 No GPUs visible. Fix NVIDIA driver/CUDA before training.")
        raise SystemExit(1)

    # ensure global batch divisibility across replicas
    try:
        replicas = strategy.num_replicas_in_sync
    except Exception:
        replicas = 1
    if config.BATCH_SIZE % replicas != 0:
        raise ValueError(f"Global batch ({config.BATCH_SIZE}) must be divisible by replicas ({replicas}).")

    # ---- input shape: use override if provided; else detect from data ----
    if getattr(config, "INPUT_SHAPE", None) in (None, (), []):
        max_dims = detect_input_shape(config.DATA_DIR)
        config.INPUT_SHAPE = tuple(max_dims) + (1,)
    config._write_config()
    logger.info(f"🧭 INPUT_SHAPE set to: {config.INPUT_SHAPE}")

    # ---- dataset & splits ----
    pairs, lesion_presence = load_generic_dataset(config)
    train_pairs, val_pairs = create_stratified_splits(
        pairs, lesion_presence, batch_size=config.BATCH_SIZE, test_size=config.VALIDATION_SPLIT
    )
    train_gen = DynamicDataGenerator(train_pairs, config, is_training=True)
    val_gen   = DynamicDataGenerator(val_pairs,   config, is_training=False)

    # ---- build/compile under current strategy ----
    with strategy.scope():
        model = build_dynamic_model(config)
        model.summary(print_fn=logger.info)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.INITIAL_LR,
            global_clipnorm=config.MAX_GRAD_NORM
        )
        loss_obj = CombinedLoss(alpha=config.DICE_WEIGHT, beta=config.BOUNDARY_WEIGHT)

        # You can set steps_per_execution here if desired to reduce overhead:
        # model.compile(optimizer=optimizer, loss=loss_obj, metrics=[dice_coefficient], steps_per_execution=4)
        model.compile(optimizer=optimizer, loss=loss_obj, metrics=[dice_coefficient])

        # ---- optional resume BEFORE training ----
        if load_weights_from:
            model.load_weights(str(load_weights_from))
            logger.info(f"✅ Loaded weights from {load_weights_from}")
        elif resume_from_latest:
            latest = config.CALLBACKS_DIR / "latest.weights.h5"
            if latest.exists():
                model.load_weights(str(latest))
                logger.info(f"✅ Loaded latest weights from {latest}")
            else:
                logger.info("ℹ️ RESUME_FROM_LATEST requested but no latest.weights.h5 found.")

    # ---- callbacks ----
    # Reduce LR when val Dice plateaus (instead of a fixed cosine schedule)
    rlrop = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_dice_coefficient", mode="max",
        factor=0.5, patience=4, min_lr=config.MIN_LR, verbose=1
    )

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(config.checkpoint_path),
        monitor="val_dice_coefficient",
        mode="max",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )
    latest_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(config.CALLBACKS_DIR / "latest.weights.h5"),
        save_weights_only=True,
        save_freq="epoch",
        verbose=0,
    )
    csv_cb     = tf.keras.callbacks.CSVLogger(str(config.CALLBACKS_DIR / "history.csv"), append=True)
    memory_cb  = MemoryMonitoringCallback(log_frequency=10)
    progress_cb= ProgressPrinter()

    # Optional NVML GPU mem logger if you defined NvmlGpuMemLogger earlier
    try:
        nvml_cb = NvmlGpuMemLogger() if 'NvmlGpuMemLogger' in globals() and NvmlGpuMemLogger else None
    except Exception:
        nvml_cb = None

    callbacks = [checkpoint_cb, latest_cb, rlrop, csv_cb, memory_cb, progress_cb]
    if nvml_cb is not None:
        callbacks.append(nvml_cb)

    # ---- preflight save check ----
    status = preflight_model_saving(model, config, logger)
    if status is True:
        logger.info("🟢 Preflight: full-model saving is guaranteed to work.")
    elif status == "weights-only":
        logger.warning("🟡 Preflight: will rely on weights-only fallback at end.")
    else:
        logger.critical("🔴 Preflight FAILED: saving will error at end; fix before training.")
        raise RuntimeError("Preflight model saving failed")

    # ---- train ----
    logger.info("🚀 Starting training...")
    history = model.fit(
        train_gen,
        epochs=config.TOTAL_EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        initial_epoch=config.INITIAL_EPOCH,
        verbose=1,
        steps_per_epoch=len(train_gen),
        validation_steps=len(val_gen),
        **{k: v for k, v in fit_kwargs.items() if v is not None}
    )

    # ---- persist history JSON (optional) ----
    try:
        (config.CALLBACKS_DIR / "artifacts").mkdir(parents=True, exist_ok=True)
        with open(config.CALLBACKS_DIR / "artifacts" / "history_epoch.json", "w") as f:
            json.dump(getattr(history, "history", {}), f)
    except Exception as e:
        logger.warning(f"Could not save history JSON: {e}")

    # ---- save final weights and full model ----
    final_weights = config.model_path.with_suffix(".final.weights.h5")
    try:
        model.save_weights(str(final_weights))
        logger.info(f"💾 Saved final weights to {final_weights}")
    except Exception:
        logger.exception("Saving final weights failed")

    try:
        model.save(str(config.model_path), include_optimizer=False)
        logger.info(f"💾 Saved full model to {config.model_path}")
    except Exception:
        logger.exception("Full-model save failed; falling back to weights-only duplicate")
        weights_only = config.model_path.with_suffix(".weights.h5")
        try:
            model.save_weights(str(weights_only))
            logger.info(f"💾 Saved weights only to {weights_only}")
        except Exception:
            logger.exception("Weights-only fallback also failed")

    logger.info("🏁 Training complete.")

    # free graph state in long sessions (best-effort)
    try:
        tf.keras.backend.clear_session()
        gc.collect()
    except Exception:
        pass

    return history




def preflight_model_saving(model, config, logger):
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.CALLBACKS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        probe = config.MODEL_DIR / "_preflight.keras"
        model.save(probe, include_optimizer=False)
        try:
            from keras.saving import load_model as _load_model
        except Exception:
            from tensorflow.keras.models import load_model as _load_model
        _load_model(probe, compile=False, custom_objects={
            "ResidualConvBlock": ResidualConvBlock,
            "VisionMambaBlock": VisionMambaBlock,
            "SAM2Attention": SAM2Attention,
            "CombinedLoss": CombinedLoss,
            "dice_coefficient": dice_coefficient,
            "dice_loss": dice_loss,
            "boundary_loss": boundary_loss,
        })
        probe.unlink(missing_ok=True)
        logger.info("Preflight full save/load succeeded.")
        return True
    except Exception:
        logger.warning("Preflight full save/load failed; trying weights-only fallback.", exc_info=True)

    try:
        weights_probe = config.MODEL_DIR / "_preflight.weights.h5"
        model.save_weights(weights_probe)
        model.load_weights(weights_probe)
        weights_probe.unlink(missing_ok=True)
        logger.info("Preflight weights-only save/load succeeded.")
        return "weights-only"
    except Exception:
        logger.exception("Preflight weights-only save/load failed.")
        return False


# If running as a script; in a notebook just call train_dynamic_model(DynamicTrainingConfig())
if __name__ == "__main__":
    cfg = DynamicTrainingConfig()
    _ = train_dynamic_model(cfg)