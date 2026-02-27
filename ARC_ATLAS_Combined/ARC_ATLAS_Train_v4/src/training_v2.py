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

# Precision policy:
# - Default is float32 for stability on custom 3D attention/loss stacks.
# - Override with SMARTSOTA_MIXED_PRECISION in {"auto","float32","mixed_float16","mixed_bfloat16"}.
_req_policy = os.environ.get("SMARTSOTA_MIXED_PRECISION", "float32").strip().lower()
if _req_policy in {"float32", "fp32", "off", "false", "0"}:
    _policy = "float32"
elif _req_policy in {"mixed_bfloat16", "bfloat16", "bf16"}:
    _policy = "mixed_bfloat16"
elif _req_policy in {"mixed_float16", "float16", "fp16"}:
    _policy = "mixed_float16" if gpus else "float32"
    if not gpus:
        print("No GPU detected; overriding float16 policy to float32.")
else:
    _policy = "mixed_float16" if gpus else "float32"
try:
    mixed_precision.set_global_policy(_policy)
except Exception as e:
    print(f"Could not set precision policy '{_policy}': {e}; falling back to float32.")
    mixed_precision.set_global_policy("float32")
print("Mixed precision policy:", mixed_precision.global_policy())

# ✅ Only mirror if multi-GPU
strategy = tf.distribute.MirroredStrategy() if len(gpus) > 1 else tf.distribute.get_strategy()
print("Strategy:", type(strategy).__name__)

SCRIPT_ROOT = Path(__file__).resolve().parent

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
    from scipy.ndimage import zoom, binary_dilation, rotate, binary_closing, binary_opening, label, generate_binary_structure, gaussian_filter
    from sklearn.model_selection import KFold
    from skimage import measure
    from skimage.filters import threshold_otsu
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
from dataclasses import dataclass, field, asdict, replace, fields
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
    BATCH_SIZE: int = 4
    INITIAL_EPOCH: int = 0
    TOTAL_EPOCHS: int = 250
    INITIAL_LR: float = 3e-5
    MIN_LR: float = 1e-6
    WARMUP_EPOCHS: int = 10
    COSINE_FIRST_CYCLE_EPOCHS: int = 30
    COSINE_T_MUL: float = 2.0
    COSINE_M_MUL: float = 1.0
    COSINE_MIN_LR_MULT: float = 0.1
    SWA_EPOCHS: int = 5
    SWA_LR_MULT: Optional[float] = None
    MAX_GRAD_NORM: float = 1.0
    BASE_FILTERS: int = 8
    DROPOUT_RATE: float = 0.30
    L2_REG: float = 3.0e-4
    MAMBA_DEPTH: int = 2
    SAM_HEADS: int = 2
    AUGMENTATION_INTENSITY: float = 0.4
    AUG_KSPACE_NOISE_STD: float = 0.01
    AUG_BIAS_FIELD_MAX: float = 0.1
    AUG_SLICE_JITTER: int = 2
    SYNTHETIC_LESION_PROB: float = 0.3
    ROTATION_RANGE: int = 20
    USE_BOUNDARY_LOSS: bool = True
    DICE_LOSS_WEIGHT: float = 0.6
    BOUNDARY_LOSS_WEIGHT: float = 0.4
    DEEP_SUPERVISION_WEIGHTS: tuple[float, ...] = (0.1, 0.2, 0.3)
    DICE_WEIGHT: float = 0.6
    BOUNDARY_WEIGHT: float = 0.4
    BOUNDARY_WARMUP_DICE: float = 0.6
    BOUNDARY_WARMUP_BOUNDARY: float = 0.4
    BOUNDARY_WARMUP_FRACTION: float = 0.33
    BOUNDARY_RAMP_EPOCHS: int = 15
    RESAMPLE_TO_TARGET: bool = True
    DECISION_THRESHOLD: float = 0.5
    GAUSSIAN_TILE_OVERLAP: float = 0.5
    GAUSSIAN_TILE_SIGMA: float = 0.125
    USE_TTA_FLIPS: bool = True
    USE_PER_CASE_OTSU: bool = True
    OTSU_MIN_PROB: float = 0.01
    OTSU_CLAMP: tuple[float, float] = (0.05, 0.25)
    MODEL_DIR: Path = field(default_factory=lambda: _default_dir("SMARTSOTA_MODEL_DIR", SCRIPT_ROOT / "models"))
    CALLBACKS_DIR: Path = field(default_factory=lambda: _default_dir("SMARTSOTA_CALLBACK_DIR", SCRIPT_ROOT / "callbacks"))
    timestamp: str = field(default_factory=lambda: time.strftime("%Y%m%d_%H%M%S"), init=False)
    SIZE_AWARE_ENABLED: bool = True
    SIZE_AWARE_MODE: str = "bucket"                     # "bucket" or "inverse"
    SIZE_BUCKET_EDGES: tuple[int, ...] = (2500, 13000, 31000, 55000)
    SIZE_BUCKET_PROBS: tuple[float, ...] = (0.35, 0.25, 0.20, 0.12, 0.08)
    INV_VOL_ALPHA: float = 0.7
    INV_VOL_EPS: float = 1e3
    PATCH_FG_PROB_BY_BIN: tuple[float, ...] = (0.95, 0.90, 0.80, 0.65, 0.55)
    PATCH_SIZE: tuple[int, int, int] | None = (112, 112, 112)
    PATCHES_PER_CASE: int = 1
    LOAD_FULL_IMAGE_FOR_PATCHING: bool = True
    FULL_RES_TARGET_SHAPE: tuple[int, int, int] | None = None
    PATCH_SAMPLING_STRATEGY: str = "random"            # "random" | "hemisphere"
    HEMISPHERE_AXIS: int = 2                           # RAS x-axis after canonicalization
    HEMISPHERE_BALANCED: bool = True
    MAX_PATCHES_PER_CASE_PER_EPOCH: int = 64
    DIFF_AWARE_ENABLED: bool = True
    DIFF_EMA_LAMBDA: float = 0.8
    DIFF_BETA: float = 1.5
    DIFF_MAX_EVAL_CASES: int = 32
    EPOCH_STEPS: int = 2000
    LOSS_MODE: str = "combined"                          # "combined" | "tversky" | "focal_tversky"
    TVERSKY_ALPHA: float = 0.7
    TVERSKY_BETA: float = 0.3
    FOCAL_TVERSKY_GAMMA: float = 1.5
    FOCAL_TVERSKY_WEIGHT: float = 0.2
    RNG_SEED: int = 1234

    def __post_init__(self):
        if self.DATA_DIR is None:
            raise ValueError("DATA_DIR must be supplied via argument or SMARTSOTA_DATA_DIR environment variable.")
        self.DATA_DIR = Path(self.DATA_DIR)
        self.IMAGES_DIR = Path(self.IMAGES_DIR) if self.IMAGES_DIR else self.DATA_DIR
        self.MASKS_DIR = Path(self.MASKS_DIR) if self.MASKS_DIR else self.DATA_DIR
        self.MODEL_DIR = Path(self.MODEL_DIR)
        self.CALLBACKS_DIR = Path(self.CALLBACKS_DIR)
        self.PATCH_SAMPLING_STRATEGY = str(self.PATCH_SAMPLING_STRATEGY).strip().lower()
        if self.PATCH_SAMPLING_STRATEGY not in {"random", "hemisphere"}:
            raise ValueError("PATCH_SAMPLING_STRATEGY must be 'random' or 'hemisphere'.")
        if int(self.HEMISPHERE_AXIS) not in (0, 1, 2):
            raise ValueError("HEMISPHERE_AXIS must be 0, 1, or 2.")
        self.HEMISPHERE_AXIS = int(self.HEMISPHERE_AXIS)
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.CALLBACKS_DIR.mkdir(parents=True, exist_ok=True)
        self._write_config()

    def _write_config(self) -> None:
        payload = asdict(self)
        for key in ("DATA_DIR", "IMAGES_DIR", "MASKS_DIR", "MODEL_DIR", "CALLBACKS_DIR"):
            payload[key] = str(payload[key])
        if payload["INPUT_SHAPE"] is not None:
            payload["INPUT_SHAPE"] = list(payload["INPUT_SHAPE"])
        tuple_fields = (
            "IMAGE_SUFFIXES",
            "MASK_SUFFIXES",
            "SIZE_BUCKET_EDGES",
            "SIZE_BUCKET_PROBS",
            "PATCH_FG_PROB_BY_BIN",
            "PATCH_SIZE",
            "FULL_RES_TARGET_SHAPE",
            "OTSU_CLAMP",
        )
        for key in tuple_fields:
            if payload.get(key) is not None:
                payload[key] = list(payload[key])
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
    def __init__(self, filters, kernel_reg=None, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_reg = kernel_reg
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.conv1 = layers.Conv3D(self.filters, 3, padding='same',
                                   kernel_regularizer=self.kernel_reg)
        # LN in float32 for stability under mixed precision
        self.ln1 = layers.LayerNormalization(epsilon=1e-5, dtype="float32")

        self.conv2 = layers.Conv3D(self.filters, 3, padding='same',
                                   kernel_regularizer=self.kernel_reg)
        self.ln2 = layers.LayerNormalization(epsilon=1e-5, dtype="float32")

        self.dropout = layers.SpatialDropout3D(self.dropout_rate)

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
                           if self.kernel_reg else None,
            "dropout_rate": self.dropout_rate
        })
        return config



@register_keras_serializable(package="custom")
class VisionMambaBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, expansion=2, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.expansion = expansion
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.in_conv = layers.Conv3D(self.filters * self.expansion, 1, use_bias=False, padding='same')
        self.spatial_conv = layers.Conv3D(self.filters * self.expansion, self.kernel_size, padding='same', use_bias=False)
        self.out_conv = layers.Conv3D(self.filters, 1, padding='same')
        # LN in fp32, then cast back
        self.norm = layers.LayerNormalization(epsilon=1e-5, dtype="float32")
        self.dropout = layers.SpatialDropout3D(self.dropout_rate)
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
            "expansion": self.expansion,
            "dropout_rate": self.dropout_rate
        })
        return config

@register_keras_serializable(package="custom")
class SAM2Attention(layers.Layer):
    def __init__(self, filters, heads, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.heads = heads
        self.dropout_rate = dropout_rate
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
        self.dropout = layers.SpatialDropout3D(self.dropout_rate)
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
            "heads": self.heads,
            "dropout_rate": self.dropout_rate
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
        x = ResidualConvBlock(filters, kernel_reg=tf.keras.regularizers.l2(config.L2_REG), dropout_rate=config.DROPOUT_RATE)(x)
        x = VisionMambaBlock(filters, dropout_rate=config.DROPOUT_RATE)(x)
        skips.append(x)
        x = layers.MaxPool3D(pool_size=2)(x)
        filters *= 2

    # Bottleneck
    x = ResidualConvBlock(filters, kernel_reg=tf.keras.regularizers.l2(config.L2_REG), dropout_rate=config.DROPOUT_RATE)(x)
    x = SAM2Attention(filters, heads=config.SAM_HEADS, dropout_rate=config.DROPOUT_RATE)(x)

    # Decoder
    for d in reversed(range(4)):
        filters //= 2
        x = layers.UpSampling3D(size=2)(x)
        x = layers.Concatenate()([x, skips[d]])
        x = ResidualConvBlock(filters, kernel_reg=tf.keras.regularizers.l2(config.L2_REG), dropout_rate=config.DROPOUT_RATE)(x)
        x = VisionMambaBlock(filters, dropout_rate=config.DROPOUT_RATE)(x)

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
    y_true = tf.where(tf.math.is_finite(y_true), y_true, tf.zeros_like(y_true))
    y_pred = tf.where(tf.math.is_finite(y_pred), y_pred, tf.zeros_like(y_pred))
    y_true = tf.cast(tf.clip_by_value(y_true, 0.0, 1.0), tf.float32)
    y_pred = tf.cast(tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7), tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    denom = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return (2.0 * intersection + smooth) / (denom + smooth)

@register_keras_serializable(package="custom")
def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)

@register_keras_serializable(package="custom")
def safe_binary_iou(y_true, y_pred, threshold=0.5, smooth=1e-6):
    """
    IoU metric that avoids confusion-matrix scatter indexing issues by
    thresholding tensors directly and masking non-finite predictions.
    """
    y_true = tf.cast(y_true > 0.5, tf.float32)
    y_pred = tf.where(tf.math.is_finite(y_pred), y_pred, tf.zeros_like(y_pred))
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

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
    y_true = tf.where(tf.math.is_finite(y_true), y_true, tf.zeros_like(y_true))
    y_pred = tf.where(tf.math.is_finite(y_pred), y_pred, tf.zeros_like(y_pred))
    y_true = tf.cast(tf.clip_by_value(y_true, 0.0, 1.0), tf.float32)
    y_pred = tf.cast(tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7), tf.float32)
    gxt, gyt, gzt = _sobel_3d(y_true)
    gxp, gyp, gzp = _sobel_3d(y_pred)
    grad_true = tf.sqrt(gxt**2 + gyt**2 + gzt**2 + 1e-7)
    grad_pred = tf.sqrt(gxp**2 + gyp**2 + gzp**2 + 1e-7)
    return tf.reduce_mean(tf.abs(grad_true - grad_pred))

@register_keras_serializable(package="custom")
class CombinedLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.6, beta=0.4, name="combined_loss"):
        super().__init__(name=name)
        self.alpha = float(alpha)
        self.beta = float(beta)

    def get_config(self):
        return {"alpha": self.alpha, "beta": self.beta}

    def call(self, y_true, y_pred):
        return self.alpha * dice_loss(y_true, y_pred) + self.beta * boundary_loss(y_true, y_pred)

@register_keras_serializable(package="custom")
def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, eps=1e-6):
    y_true = tf.where(tf.math.is_finite(y_true), y_true, tf.zeros_like(y_true))
    y_pred = tf.where(tf.math.is_finite(y_pred), y_pred, tf.zeros_like(y_pred))
    y_true = tf.cast(tf.clip_by_value(y_true, 0.0, 1.0), tf.float32)
    y_pred = tf.cast(tf.clip_by_value(y_pred, eps, 1.0 - eps), tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1.0 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1.0 - y_pred))
    score = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1.0 - score

@register_keras_serializable(package="custom")
def focal_tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, gamma=1.5, eps=1e-6):
    tv = tversky_loss(y_true, y_pred, alpha=alpha, beta=beta, eps=eps)
    tv = tf.clip_by_value(tv, 0.0, 1.0)
    return tf.pow(tv, gamma)

def make_tversky_loss(alpha, beta):
    @tf.function
    def _loss(y_true, y_pred):
        return tversky_loss(y_true, y_pred, alpha=alpha, beta=beta)
    return _loss

def make_focal_tversky_loss(alpha, beta, gamma):
    @tf.function
    def _loss(y_true, y_pred):
        return focal_tversky_loss(y_true, y_pred, alpha=alpha, beta=beta, gamma=gamma)
    return _loss

def dice_soft_np(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    y_true = y_true.astype(np.float32, copy=False)
    y_pred = y_pred.astype(np.float32, copy=False)
    inter = float(np.sum(y_true * y_pred, dtype=np.float64))
    denom = float(np.sum(y_true, dtype=np.float64) + np.sum(y_pred, dtype=np.float64))
    return float((2.0 * inter + eps) / (denom + eps))

@register_keras_serializable(package="custom")
class HybridLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        dice_weight=0.6,
        boundary_weight=0.4,
        focal_weight=0.0,
        tversky_alpha=0.7,
        tversky_beta=0.3,
        focal_gamma=1.5,
        name="hybrid_loss",
    ):
        super().__init__(name=name)
        self.dice_weight = float(dice_weight)
        self.boundary_weight = float(boundary_weight)
        self.focal_weight = float(focal_weight)
        self.tversky_alpha = float(tversky_alpha)
        self.tversky_beta = float(tversky_beta)
        self.focal_gamma = float(focal_gamma)

    def get_config(self):
        return {
            "dice_weight": self.dice_weight,
            "boundary_weight": self.boundary_weight,
            "focal_weight": self.focal_weight,
            "tversky_alpha": self.tversky_alpha,
            "tversky_beta": self.tversky_beta,
            "focal_gamma": self.focal_gamma,
        }

    def set_weights(self, dice_weight=None, boundary_weight=None, focal_weight=None):
        if dice_weight is not None:
            self.dice_weight = float(dice_weight)
        if boundary_weight is not None:
            self.boundary_weight = float(boundary_weight)
        if focal_weight is not None:
            self.focal_weight = float(focal_weight)

    def call(self, y_true, y_pred):
        loss_val = self.dice_weight * dice_loss(y_true, y_pred) + self.boundary_weight * boundary_loss(y_true, y_pred)
        if self.focal_weight > 0.0:
            loss_val += self.focal_weight * focal_tversky_loss(
                y_true, y_pred, alpha=self.tversky_alpha, beta=self.tversky_beta, gamma=self.focal_gamma
            )
        return loss_val


class LossRampScheduler(tf.keras.callbacks.Callback):
    """Ease in boundary-heavy loss, then ramp toward target weights."""
    def __init__(self, loss_obj: HybridLoss, cfg: DynamicTrainingConfig):
        super().__init__()
        self.loss_obj = loss_obj
        self.cfg = cfg
        self.start_dice = float(cfg.BOUNDARY_WARMUP_DICE)
        self.start_boundary = float(cfg.BOUNDARY_WARMUP_BOUNDARY)
        self.target_dice = float(cfg.DICE_WEIGHT)
        self.target_boundary = float(cfg.BOUNDARY_WEIGHT)
        self.warmup_epochs = max(1, int(round(cfg.TOTAL_EPOCHS * cfg.BOUNDARY_WARMUP_FRACTION)))
        self.ramp_epochs = max(1, int(round(cfg.BOUNDARY_RAMP_EPOCHS)))

    def on_epoch_begin(self, epoch, logs=None):
        if not isinstance(self.loss_obj, HybridLoss):
            return
        if epoch < self.warmup_epochs:
            dice_w, boundary_w = self.start_dice, self.start_boundary
        elif epoch < self.warmup_epochs + self.ramp_epochs:
            prog = (epoch - self.warmup_epochs) / max(1, self.ramp_epochs)
            dice_w = self.start_dice + (self.target_dice - self.start_dice) * prog
            boundary_w = self.start_boundary + (self.target_boundary - self.start_boundary) * prog
        else:
            dice_w, boundary_w = self.target_dice, self.target_boundary
        self.loss_obj.set_weights(dice_weight=dice_w, boundary_weight=boundary_w)
        logger.info(f"📉 Loss mix @epoch {epoch}: dice={dice_w:.3f}, boundary={boundary_w:.3f}, focal={self.loss_obj.focal_weight:.3f}")


class NonFiniteLossGuard(tf.keras.callbacks.Callback):
    """Stop as soon as loss becomes non-finite and emit a clear diagnostic line."""
    def _check(self, stage: str, batch: int, logs):
        logs = logs or {}
        loss = logs.get("loss")
        if loss is None:
            return
        if not np.isfinite(loss):
            logger.error(f"Non-finite loss detected at {stage} batch={batch}: {loss}. Stopping training.")
            self.model.stop_training = True

    def on_train_batch_end(self, batch, logs=None):
        self._check("train", int(batch), logs)

    def on_test_batch_end(self, batch, logs=None):
        self._check("val", int(batch), logs)


# ---------------------------------------------------------------------------
# Volume loading and preprocessing
# ---------------------------------------------------------------------------
_ACTIVE_CONFIG = None  # set inside train_dynamic_model so loaders can see config flags

@lru_cache(maxsize=128)
def _load_vol_canonical(path: str) -> np.ndarray:
    """Load NIfTI as RAS-canonical float32 array (cached)."""
    img = nib.load(path)
    img = nib.as_closest_canonical(img)
    return img.get_fdata().astype(np.float32)

def _should_resample() -> bool:
    cfg = globals().get("_ACTIVE_CONFIG")
    return bool(getattr(cfg, "RESAMPLE_TO_TARGET", True)) if cfg is not None else True

def _maybe_resample(volume: np.ndarray, target_shape: tuple[int, int, int] | None, order: int) -> np.ndarray:
    if target_shape is None or volume.shape == tuple(target_shape):
        return volume.astype(np.float32, copy=False)
    factors = [t / max(s, 1) for s, t in zip(volume.shape, target_shape)]
    try:
        return zoom(volume, factors, order=order).astype(np.float32, copy=False)
    except Exception as e:
        logger.warning(f"Resample failed for shape {volume.shape} -> {target_shape}: {e}; using crop/pad fallback.")
        return volume.astype(np.float32, copy=False)

def _center_crop_or_pad_volume(volume: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Center-crop or pad to target_shape without interpolation."""
    assert volume.ndim == 3, f"Expected 3D volume, got {volume.ndim}D"
    z, y, x = volume.shape
    tz, ty, tx = map(int, target_shape)

    if z > tz:
        start = (z - tz) // 2
        volume = volume[start:start+tz, :, :]
        z = tz
    if y > ty:
        start = (y - ty) // 2
        volume = volume[:, start:start+ty, :]
        y = ty
    if x > tx:
        start = (x - tx) // 2
        volume = volume[:, :, start:start+tx]
        x = tx

    pad_z = max(0, tz - z)
    pad_y = max(0, ty - y)
    pad_x = max(0, tx - x)
    if pad_z or pad_y or pad_x:
        padding = (
            (pad_z // 2, pad_z - pad_z // 2),
            (pad_y // 2, pad_y - pad_y // 2),
            (pad_x // 2, pad_x - pad_x // 2),
        )
        volume = np.pad(volume, padding, mode="constant", constant_values=0)
    return volume.astype(np.float32, copy=True)

def _load_and_preprocess_image(path: str, target_shape: tuple[int, int, int] | None) -> np.ndarray:
    target_shape = tuple(int(v) for v in target_shape) if target_shape is not None else None
    vol = _load_vol_canonical(path)
    vol = _maybe_resample(vol, target_shape if _should_resample() else None, order=1)
    if target_shape is not None and vol.shape != target_shape:
        vol = _center_crop_or_pad_volume(vol, target_shape)
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
    return vol.astype(np.float32, copy=True)

def _load_and_preprocess_mask(path: str, target_shape: tuple[int, int, int] | None) -> np.ndarray:
    target_shape = tuple(int(v) for v in target_shape) if target_shape is not None else None
    vol = (_load_vol_canonical(path) > 0.5).astype(np.float32)
    vol = _maybe_resample(vol, target_shape if _should_resample() else None, order=0)
    if target_shape is not None and vol.shape != target_shape:
        vol = _center_crop_or_pad_volume(vol, target_shape)
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
    return (vol > 0.5).astype(np.float32, copy=False)

def compute_lesion_sizes(pairs, load_mask_fn, target_shape=None):
    sizes = []
    for _, msk_p in pairs:
        y = load_mask_fn(str(msk_p), target_shape).astype(np.float32)
        sizes.append(int(np.sum(y > 0)))
    return np.asarray(sizes, dtype=np.int64)


def apply_augmentations(image: np.ndarray, mask: np.ndarray, cfg, rng) -> tuple[np.ndarray, np.ndarray]:
    """Shared augmentation pipeline for patch and full-volume loaders."""
    if cfg.AUGMENTATION_INTENSITY <= 0 or rng.random() > cfg.AUGMENTATION_INTENSITY:
        return image, mask

    # Flips
    for axis in (0, 1, 2):
        if rng.random() > 0.5:
            image = np.flip(image, axis=axis)
            mask = np.flip(mask, axis=axis)

    # Small rotation
    if rng.random() > 0.7:
        angle = float(rng.uniform(-cfg.ROTATION_RANGE, cfg.ROTATION_RANGE))
        axis = int(rng.integers(0, 3))
        axes = [(0, 1), (0, 2), (1, 2)][axis]
        image = rotate(image, angle, axes=axes, reshape=False, order=1, mode='constant')
        mask = rotate(mask, angle, axes=axes, reshape=False, order=0, mode='constant')

    # Slice jitter (shift along z)
    if getattr(cfg, "AUG_SLICE_JITTER", 0) > 0 and rng.random() > 0.5:
        shift = int(rng.integers(-cfg.AUG_SLICE_JITTER, cfg.AUG_SLICE_JITTER + 1))
        if shift != 0:
            image = np.roll(image, shift, axis=0)
            mask = np.roll(mask, shift, axis=0)
            if shift > 0:
                image[:shift, ...] = 0
                mask[:shift, ...] = 0
            else:
                image[shift:, ...] = 0
                mask[shift:, ...] = 0

    # Bias field (low-frequency multiplicative)
    if getattr(cfg, "AUG_BIAS_FIELD_MAX", 0) > 0 and rng.random() > 0.5:
        noise = rng.normal(0.0, 1.0, size=image.shape)
        sigma = max(image.shape) / 16.0
        field = gaussian_filter(noise, sigma=sigma)
        field = (field - field.min()) / (field.max() - field.min() + 1e-6) - 0.5
        scale = float(rng.uniform(-cfg.AUG_BIAS_FIELD_MAX, cfg.AUG_BIAS_FIELD_MAX))
        field = 1.0 + scale * field
        image = image * field

    # K-space noise (light)
    if getattr(cfg, "AUG_KSPACE_NOISE_STD", 0) > 0 and rng.random() > 0.7:
        F = np.fft.fftn(image)
        mag = np.mean(np.abs(F))
        noise_scale = float(cfg.AUG_KSPACE_NOISE_STD) * (mag if mag > 0 else 1.0)
        noise = rng.normal(0.0, noise_scale, size=F.shape) + 1j * rng.normal(0.0, noise_scale, size=F.shape)
        image = np.real(np.fft.ifftn(F + noise))

    # Gamma
    if rng.random() > 0.8:
        gamma = float(rng.uniform(0.7, 1.3))
        maxv = image.max()
        if maxv > 0:
            image = np.power(np.clip(image / maxv, 0, 1), gamma) * maxv

    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
    mask = (mask > 0.5).astype(np.float32, copy=False)
    return image, mask

class SizeAwareCaseSampler:
    def __init__(self, lesion_sizes: np.ndarray, cfg: DynamicTrainingConfig):
        self.cfg = cfg
        self.sizes = lesion_sizes.astype(np.int64)
        self.N = len(self.sizes)
        self.weights = self._init_weights()
        self.patch_quota = np.zeros(self.N, dtype=np.int64)

    def _init_weights(self):
        if not self.cfg.SIZE_AWARE_ENABLED or self.N == 0:
            w = np.ones(self.N, dtype=np.float64)
            return w / w.sum()
        if self.cfg.SIZE_AWARE_MODE == "inverse":
            w = 1.0 / np.power(self.sizes + self.cfg.INV_VOL_EPS, self.cfg.INV_VOL_ALPHA)
            w = np.clip(w, 1e-12, None)
            return w / w.sum()
        edges = np.asarray(self.cfg.SIZE_BUCKET_EDGES, dtype=np.int64)
        probs = np.asarray(self.cfg.SIZE_BUCKET_PROBS, dtype=np.float64)
        bins = np.digitize(self.sizes, edges, right=False)
        w = np.zeros(self.N, dtype=np.float64)
        for b in range(len(probs)):
            idx = np.where(bins == b)[0]
            if idx.size == 0:
                continue
            w[idx] = probs[b] / idx.size
        w = np.clip(w, 1e-12, None)
        return w / w.sum()

    def sample_indices(self, k: int) -> np.ndarray:
        if self.N == 0:
            return np.empty(0, dtype=np.int64)
        p = self.weights.copy()
        hit = self.patch_quota >= self.cfg.MAX_PATCHES_PER_CASE_PER_EPOCH
        if hit.any():
            if hit.all():
                self.patch_quota[:] = 0
                p = self.weights.copy()
            else:
                p[hit] = 0.0
                s = p.sum()
                if s <= 0:
                    self.patch_quota[:] = 0
                    p = self.weights.copy()
                else:
                    p /= s
        idx = np.random.choice(self.N, size=k, replace=True, p=p)
        self.patch_quota[idx] += 1
        return idx

    def start_epoch(self):
        self.patch_quota[:] = 0

    def diff_aware_update(self, val_case_dice: np.ndarray):
        if not self.cfg.DIFF_AWARE_ENABLED or self.N == 0:
            return
        dice = np.clip(val_case_dice, 0.0, 1.0)
        badness = np.power(1.0 - dice, self.cfg.DIFF_BETA)
        badness = np.clip(badness, 1e-6, None)
        w_new = (
            self.cfg.DIFF_EMA_LAMBDA * self.weights
            + (1.0 - self.cfg.DIFF_EMA_LAMBDA) * (badness / badness.sum())
        )
        self.weights = (w_new / w_new.sum()).astype(np.float64)

def bin_index(v, edges):
    import numpy as _np
    return int(_np.digitize([v], _np.asarray(edges, dtype=_np.int64), right=False)[0])

def sample_patch_center(
    mask,
    patch_size,
    p_fg,
    rng,
    hemisphere_axis: int | None = None,
    hemisphere_side: int | None = None,
):
    import numpy as _np

    Z, Y, X = mask.shape
    dz, dy, dx = patch_size
    dims = [Z, Y, X]
    patch_dims = [dz, dy, dx]

    fg = _np.argwhere(mask > 0)
    bg = _np.argwhere(mask == 0)

    use_hemi = hemisphere_axis is not None and hemisphere_side in (0, 1)
    if use_hemi:
        axis = int(hemisphere_axis)
        mid = dims[axis] // 2
        if hemisphere_side == 0:
            fg_side = fg[fg[:, axis] < mid] if fg.size > 0 else fg
            bg_side = bg[bg[:, axis] < mid] if bg.size > 0 else bg
            hemi_center = [Z // 2, Y // 2, X // 2]
            hemi_center[axis] = max(0, mid // 2)
        else:
            fg_side = fg[fg[:, axis] >= mid] if fg.size > 0 else fg
            bg_side = bg[bg[:, axis] >= mid] if bg.size > 0 else bg
            hemi_center = [Z // 2, Y // 2, X // 2]
            hemi_center[axis] = mid + max(0, (dims[axis] - mid) // 2)
        if fg_side.size > 0:
            fg = fg_side
        if bg_side.size > 0:
            bg = bg_side
    else:
        hemi_center = [Z // 2, Y // 2, X // 2]

    if rng.random() < p_fg and fg.size > 0:
        cz, cy, cx = fg[rng.integers(len(fg))]
    elif bg.size > 0:
        cz, cy, cx = bg[rng.integers(len(bg))]
    else:
        cz, cy, cx = hemi_center

    jitter = rng.integers(low=-4, high=5, size=3)
    centers = [int(cz + jitter[0]), int(cy + jitter[1]), int(cx + jitter[2])]
    starts = []
    for center, dim, p in zip(centers, dims, patch_dims):
        starts.append(max(0, min(center - p // 2, dim - p)))

    if use_hemi:
        axis = int(hemisphere_axis)
        dim = dims[axis]
        p = patch_dims[axis]
        mid = dim // 2
        if hemisphere_side == 0:
            hemi_lo, hemi_hi = 0, mid
        else:
            hemi_lo, hemi_hi = mid, dim

        # If patch fits in one hemisphere, clamp to that hemisphere.
        if p <= max(1, hemi_hi - hemi_lo):
            starts[axis] = int(_np.clip(starts[axis], hemi_lo, max(hemi_lo, hemi_hi - p)))

    z0, y0, x0 = starts
    z1, y1, x1 = z0 + dz, y0 + dy, x0 + dx
    return z0, z1, y0, y1, x0, x1

class SizeAwareSamplerCallback(tf.keras.callbacks.Callback):
    def __init__(self, sampler: SizeAwareCaseSampler | None):
        super().__init__()
        self.sampler = sampler

    def on_epoch_begin(self, epoch, logs=None):
        if self.sampler:
            self.sampler.start_epoch()

class DifficultyAwareCallback(tf.keras.callbacks.Callback):
    def __init__(self, sampler, cfg, train_pairs):
        super().__init__()
        self.sampler = sampler
        self.cfg = cfg
        self.train_pairs = train_pairs
        self.rng = np.random.default_rng(cfg.RNG_SEED + 1337)

    def on_epoch_end(self, epoch, logs=None):
        if not self.sampler or not self.cfg.DIFF_AWARE_ENABLED:
            return
        n_cases = len(self.train_pairs)
        if n_cases == 0:
            return
        dice = np.ones(n_cases, dtype=np.float32)
        eval_count = min(self.cfg.DIFF_MAX_EVAL_CASES, n_cases)
        eval_idx = self.rng.choice(n_cases, size=eval_count, replace=False)
        target_shape = self.cfg.INPUT_SHAPE[:-1]
        for idx in eval_idx:
            img_p, msk_p = self.train_pairs[idx]
            x = _load_and_preprocess_image(str(img_p), target_shape)
            y = _load_and_preprocess_mask(str(msk_p), target_shape)
            xb = x[np.newaxis, ..., np.newaxis]
            pred = self.model.predict(xb, verbose=0)[0, ..., 0]
            dice[idx] = dice_soft_np(y, pred)
        self.sampler.diff_aware_update(dice)


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
    
    def __init__(self, pairs, config, is_training=False, image_loader=None, mask_loader=None):
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
        self.image_loader = image_loader or globals().get("_load_and_preprocess_image")
        self.mask_loader = mask_loader or globals().get("_load_and_preprocess_mask")
        self.rng = np.random.default_rng(config.RNG_SEED + (1 if is_training else 0))
        if self.image_loader is None or self.mask_loader is None:
            raise RuntimeError("DynamicDataGenerator requires image_loader and mask_loader callables.")
        
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
            img = self.image_loader(str(img_path), self.target_shape)
            msk = self.mask_loader(str(msk_path), self.target_shape)
            
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
        return apply_augmentations(image, mask, self.config, self.rng)

def load_generic_dataset(config: DynamicTrainingConfig):
    logger.info("📚 Loading dataset (flex loader for T1w volumes)…")
    log_memory_usage("dataset_load_start")

    manifest_path = config.DATA_DIR / "manifest.csv"
    if manifest_path.exists():
        logger.info(f"📄 Using manifest-defined pairs from {manifest_path}")
        import csv

        def _resolve_manifest_path(raw_value: str) -> Path | None:
            raw = (raw_value or "").strip()
            if not raw:
                return None
            p = Path(raw)
            if p.is_absolute():
                return p

            # Support relative paths written from different working directories.
            candidates = [p, config.DATA_DIR / p]
            for base in manifest_path.parents:
                candidates.append(base / p)

            seen = set()
            for c in candidates:
                cs = str(c)
                if cs in seen:
                    continue
                seen.add(cs)
                if c.exists():
                    return c
            return None

        pairs, lesion_counts = [], []
        missing_rows = 0
        invalid_rows = 0
        slug_counts = {}
        with manifest_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_p = _resolve_manifest_path(row.get("t1", ""))
                msk_p = _resolve_manifest_path(row.get("mask", ""))
                if img_p is None or msk_p is None:
                    missing_rows += 1
                    continue
                try:
                    mask_obj = nib.load(str(msk_p))
                    has_lesion = bool(np.any(mask_obj.get_fdata() > 0))
                    lesion_counts.append(1 if has_lesion else 0)
                    pairs.append((img_p, msk_p))
                    slug = row.get("slug", "")
                    if slug:
                        slug_counts[slug] = slug_counts.get(slug, 0) + 1
                except Exception as e:
                    invalid_rows += 1
                    logger.warning(f"Skipping manifest row for {msk_p.name}: {e}")
        if pairs:
            if missing_rows:
                logger.warning(f"Manifest rows with unresolved files: {missing_rows}")
            if invalid_rows:
                logger.warning(f"Manifest rows with unreadable masks: {invalid_rows}")
            if slug_counts:
                logger.info(f"Manifest composition: {slug_counts}")
            logger.info(f"📊 Created {len(pairs)} image–mask pairs from manifest")
            if lesion_counts:
                logger.info(f"🧠 Lesion presence: {np.mean(lesion_counts)*100:.2f}%")
            log_memory_usage("dataset_load_end")
            return pairs, np.array(lesion_counts, dtype=np.int32)
        logger.warning(
            f"Manifest present but yielded 0 valid pairs (missing={missing_rows}, invalid={invalid_rows}); "
            "falling back to directory scan."
        )

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
        all_niis = sorted(list(images_dir.rglob("*.nii.gz")) + list(images_dir.rglob("*.nii")))
        images = [p for p in all_niis if is_image(p.name)]
        masks  = [p for p in all_niis if is_mask(p.name)]
        logger.info(f"📁 Single-folder mode: {len(images)} images, {len(masks)} masks in {images_dir}")
    else:
        images = sorted(list(images_dir.rglob("*.nii.gz")) + list(images_dir.rglob("*.nii")))
        masks  = sorted(list(masks_dir.rglob("*.nii.gz")) + list(masks_dir.rglob("*.nii")))
        logger.info(f"📂 Two-folder mode: images={len(images)} ({images_dir}), masks={len(masks)} ({masks_dir})")
    img_map, msk_map = {}, {}
    for p in images:
        key = normalise_key(p.name, image_suffixes)
        if key and key not in img_map:
            img_map[key] = p
    for p in masks:
        key = normalise_key(p.name, mask_suffixes)
        if key and key not in msk_map:
            msk_map[key] = p
    keys = sorted(set(img_map) & set(msk_map))
    if not keys:
        logger.error("No image–mask pairs matched. Examples:")
        for p in images + masks:
            logger.error(f" - {p.name}")
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
    logger.info(f"📊 Created {len(pairs)} image–mask pairs")
    if lesion_counts:
        logger.info(f"🧠 Lesion presence: {np.mean(lesion_counts)*100:.2f}%")
    log_memory_usage("dataset_load_end")
    return pairs, np.array(lesion_counts, dtype=np.int32)


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------
def train_dynamic_model(config: Optional[DynamicTrainingConfig] = None, **overrides):
    """
    Main entry point to train the dynamic model with specified configuration.
    """
    fit_keys = ("steps_per_epoch", "validation_steps", "shuffle", "class_weight")
    fit_kwargs = {k: overrides.pop(k) for k in list(overrides.keys()) if k in fit_keys}
    load_weights_from = overrides.pop("LOAD_WEIGHTS_FROM", None)
    resume_from_latest = overrides.pop("RESUME_FROM_LATEST", False)

    cfg_field_names = {f.name for f in fields(DynamicTrainingConfig)}
    config_kwargs = {k: overrides.pop(k) for k in list(overrides.keys()) if k in cfg_field_names}

    if overrides:
        raise TypeError(
            "train_dynamic_model() got unexpected keyword(s): "
            + ", ".join(sorted(overrides.keys()))
        )

    if config is None:
        config = DynamicTrainingConfig(**config_kwargs)
    elif config_kwargs:
        config = replace(config, **config_kwargs)
    weights_path = Path(load_weights_from).expanduser() if load_weights_from else None
    globals()["_ACTIVE_CONFIG"] = config

    config._write_config()

    logger.info(f"🔧 Config: {config.model_path.name}")
    log_memory_usage("start")
    if (
        int(getattr(config, "SMALL_LESION_THRESHOLD", 100)) != 100
        or float(getattr(config, "SYNTHETIC_LESION_PROB", 0.3)) != 0.3
    ):
        logger.warning(
            "SMALL_LESION_THRESHOLD and SYNTHETIC_LESION_PROB are currently metadata-only in training_v2 "
            "(no synthetic-lesion augmentation is applied)."
        )

    if getattr(config, "INPUT_SHAPE", None) in (None, (), []):
        max_dims = detect_input_shape(config.DATA_DIR)
        config.INPUT_SHAPE = tuple(max_dims) + (1,)
    patch_shape = tuple(config.PATCH_SIZE or config.INPUT_SHAPE[:-1])
    if tuple(config.INPUT_SHAPE[:-1]) != patch_shape:
        config.INPUT_SHAPE = patch_shape + (1,)
    config._write_config()

    # Case-loading shape (separate from model patch shape).
    # - LOAD_FULL_IMAGE_FOR_PATCHING=True + FULL_RES_TARGET_SHAPE=None: keep native full volume.
    # - LOAD_FULL_IMAGE_FOR_PATCHING=True + FULL_RES_TARGET_SHAPE=(...): resample/crop full volume to that shape.
    # - LOAD_FULL_IMAGE_FOR_PATCHING=False: legacy behavior (load directly to patch/model shape).
    case_target_shape = None
    if bool(getattr(config, "LOAD_FULL_IMAGE_FOR_PATCHING", True)):
        raw_full_shape = getattr(config, "FULL_RES_TARGET_SHAPE", None)
        if raw_full_shape is not None:
            case_target_shape = tuple(int(v) for v in raw_full_shape)
        logger.info(
            "Patch extraction source: full-volume mode "
            f"(target_shape={case_target_shape if case_target_shape is not None else 'native'})"
        )
    else:
        case_target_shape = tuple(config.INPUT_SHAPE[:-1])
        logger.info(f"Patch extraction source: legacy patch-shaped loading {case_target_shape}")

    # --- Detect and log available GPUs ---
    gpus = tf.config.list_physical_devices("GPU")
    logger.info(f"Visible GPUs: {gpus}")
    for gpu in gpus:
        try: tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e: logger.warning(f"Could not set memory growth on {gpu}: {e}")

    # --- Build and compile the model ---
    cosine_first_steps = max(1, int(config.EPOCH_STEPS * config.COSINE_FIRST_CYCLE_EPOCHS))
    cosine_alpha = max(config.MIN_LR, config.INITIAL_LR * config.COSINE_MIN_LR_MULT) / float(config.INITIAL_LR)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=config.INITIAL_LR,
        first_decay_steps=cosine_first_steps,
        t_mul=config.COSINE_T_MUL,
        m_mul=config.COSINE_M_MUL,
        alpha=cosine_alpha,
    )
    with strategy.scope():
        model = build_dynamic_model(config)
        if config.LOSS_MODE == "tversky":
            loss_obj = make_tversky_loss(config.TVERSKY_ALPHA, config.TVERSKY_BETA)
        elif config.LOSS_MODE == "focal_tversky":
            loss_obj = make_focal_tversky_loss(config.TVERSKY_ALPHA, config.TVERSKY_BETA, config.FOCAL_TVERSKY_GAMMA)
        else:
            loss_obj = HybridLoss(
                dice_weight=config.DICE_WEIGHT,
                boundary_weight=config.BOUNDARY_WEIGHT,
                focal_weight=config.FOCAL_TVERSKY_WEIGHT,
                tversky_alpha=config.TVERSKY_ALPHA,
                tversky_beta=config.TVERSKY_BETA,
                focal_gamma=config.FOCAL_TVERSKY_GAMMA,
            )
        adam_kwargs = {"learning_rate": lr_schedule, "epsilon": 1e-8}
        if float(getattr(config, "MAX_GRAD_NORM", 0.0) or 0.0) > 0.0:
            adam_kwargs["clipnorm"] = float(config.MAX_GRAD_NORM)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(**adam_kwargs),
            loss=loss_obj,
            metrics=[dice_coefficient, safe_binary_iou],
        )
    logger.info(f"Model built: {model.count_params():,} parameters")

    if weights_path:
        model.load_weights(str(weights_path))
        logger.info(f"Loaded weights from {weights_path}")
    elif resume_from_latest:
        latest = config.CALLBACKS_DIR / "latest.weights.h5"
        if latest.exists():
            model.load_weights(str(latest))
            logger.info(f"Loaded weights from {latest}")
        else:
            logger.info("Resume requested but no latest.weights.h5 found")

    # --- Prepare dataset generators ---
    pairs, lesion_presence = load_generic_dataset(config)
    mask_preprocess_fn = globals().get("_load_and_preprocess_mask")
    if mask_preprocess_fn is None:
        logger.warning("Mask preprocessor not found; rebuilding inline fallback for lesion sizing.")
        def mask_preprocess_fn(path: str, target_shape: tuple[int, int, int] | None):
            mask_data = (nib.load(path).get_fdata() > 0.5).astype(np.float32)
            if target_shape is None:
                return mask_data
            slices = []
            for cur, tgt in zip(mask_data.shape, target_shape):
                if cur >= tgt:
                    start = (cur - tgt) // 2
                    slices.append(slice(start, start + tgt))
                else:
                    slices.append(slice(0, cur))
            cropped = mask_data[slices[0], slices[1], slices[2]]
            output = np.zeros(target_shape, dtype=np.float32)
            offsets = tuple((t - c) // 2 for c, t in zip(cropped.shape, target_shape))
            z0, y0, x0 = offsets
            output[z0:z0+cropped.shape[0], y0:y0+cropped.shape[1], x0:x0+cropped.shape[2]] = cropped
            return output
    lesion_sizes_all = compute_lesion_sizes(pairs, mask_preprocess_fn, case_target_shape)
    pair_lookup = {(str(img_p), str(msk_p)): idx for idx, (img_p, msk_p) in enumerate(pairs)}
    split_fn = globals().get("create_stratified_splits")
    if split_fn is None:
        logger.warning("create_stratified_splits not found; using simple random split fallback.")
        def split_fn(pairs_seq, lesion_arr, batch_size, test_size):
            total = len(pairs_seq)
            if total <= batch_size:
                return list(pairs_seq), []
            rng = np.random.default_rng(42)
            indices = np.arange(total)
            rng.shuffle(indices)
            desired = max(batch_size, int(round(total * test_size)))
            val_count = max(1, min(total - batch_size, desired))
            val_idx = indices[:val_count]
            train_idx = indices[val_count:]
            return [pairs_seq[i] for i in train_idx], [pairs_seq[i] for i in val_idx]
    train_pairs, val_pairs = split_fn(
        pairs, lesion_presence, batch_size=config.BATCH_SIZE, test_size=config.VALIDATION_SPLIT
    )
    train_pairs = list(train_pairs)
    val_pairs = list(val_pairs)
    train_indices = np.asarray([pair_lookup[(str(img), str(msk))] for img, msk in train_pairs], dtype=np.int64)
    train_lesion_sizes = lesion_sizes_all[train_indices]
    case_sampler = SizeAwareCaseSampler(train_lesion_sizes, config) if len(train_pairs) else None
    patch_size = tuple(config.INPUT_SHAPE[:-1])
    batch_cases = max(config.BATCH_SIZE, 1)
    patch_sampling = str(getattr(config, "PATCH_SAMPLING_STRATEGY", "random")).strip().lower()
    hemisphere_mode = patch_sampling == "hemisphere"
    hemisphere_axis = int(getattr(config, "HEMISPHERE_AXIS", 2))
    patches_per_case = max(int(config.PATCHES_PER_CASE), 1)
    if hemisphere_mode and bool(getattr(config, "HEMISPHERE_BALANCED", True)):
        patches_per_case = max(patches_per_case, 2)
    batch_patches = batch_cases * patches_per_case
    rng = np.random.default_rng(config.RNG_SEED)

    def training_batch_generator():
        if not train_pairs:
            raise RuntimeError("No training pairs available for generator.")
        while True:
            idxs = case_sampler.sample_indices(batch_cases) if case_sampler else np.random.choice(len(train_pairs), size=batch_cases, replace=True)
            xs, ys = [], []
            for idx in idxs:
                img_p, msk_p = train_pairs[idx]
                x = _load_and_preprocess_image(str(img_p), case_target_shape)
                y = _load_and_preprocess_mask(str(msk_p), case_target_shape)
                lesion_size = train_lesion_sizes[idx] if len(train_lesion_sizes) > idx else int((y > 0).sum())
                if config.SIZE_AWARE_ENABLED and config.SIZE_AWARE_MODE == "bucket":
                    bin_id = bin_index(lesion_size, config.SIZE_BUCKET_EDGES)
                    bin_id = max(0, min(bin_id, len(config.PATCH_FG_PROB_BY_BIN) - 1))
                    p_fg = float(config.PATCH_FG_PROB_BY_BIN[bin_id])
                else:
                    p_fg = float(config.PATCH_FG_PROB_BY_BIN[0])
                mask_bin = (y > 0).astype(np.uint8)
                for patch_iter in range(patches_per_case):
                    hemisphere_side = None
                    if hemisphere_mode:
                        hemisphere_side = patch_iter % 2 if bool(getattr(config, "HEMISPHERE_BALANCED", True)) else int(rng.integers(0, 2))
                    z0, z1, y0, y1, x0, x1 = sample_patch_center(
                        mask_bin,
                        patch_size,
                        p_fg,
                        rng,
                        hemisphere_axis=hemisphere_axis if hemisphere_mode else None,
                        hemisphere_side=hemisphere_side,
                    )
                    patch_x = x[z0:z1, y0:y1, x0:x1]
                    patch_y = y[z0:z1, y0:y1, x0:x1]
                    if patch_x.shape != patch_size:
                        patch_x = _center_crop_or_pad_volume(patch_x, patch_size)
                    if patch_y.shape != patch_size:
                        patch_y = _center_crop_or_pad_volume(patch_y, patch_size)
                    if config.AUGMENTATION_INTENSITY > 0:
                        patch_x, patch_y = apply_augmentations(patch_x, patch_y, config, rng)
                    xs.append(patch_x[..., np.newaxis])
                    ys.append(patch_y[..., np.newaxis])
            xb = np.stack(xs, axis=0).astype(np.float32, copy=False)
            yb = np.stack(ys, axis=0).astype(np.float32, copy=False)
            if (not np.isfinite(xb).all()) or (not np.isfinite(yb).all()):
                bad_x = int(np.size(xb) - np.isfinite(xb).sum())
                bad_y = int(np.size(yb) - np.isfinite(yb).sum())
                logger.warning(
                    f"Non-finite batch values detected (x={bad_x}, y={bad_y}); replacing with zeros."
                )
                xb = np.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0)
                yb = np.nan_to_num(yb, nan=0.0, posinf=0.0, neginf=0.0)
            yb = (yb > 0.5).astype(np.float32, copy=False)
            yield xb, yb

    train_ds = tf.data.Dataset.from_generator(
        training_batch_generator,
        output_signature=(
            tf.TensorSpec(shape=(batch_patches, *patch_size, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_patches, *patch_size, 1), dtype=tf.float32),
        ),
    ).prefetch(tf.data.AUTOTUNE)
    val_gen = DynamicDataGenerator(
        val_pairs, config, is_training=False,
        image_loader=_load_and_preprocess_image, mask_loader=_load_and_preprocess_mask
    )

    # --- Configure callbacks ---
    try:
        from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, StochasticWeightAveraging
    except ImportError:
        from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
        StochasticWeightAveraging = None
    try:
        from smartsota import NVMLMemoryLogger as NvmlGpuMemLogger
    except ImportError:
        NvmlGpuMemLogger = None
        logger.warning("NVMLMemoryLogger not available; GPU telemetry callback disabled.")

    checkpoint_cb = ModelCheckpoint(
        filepath=str(config.checkpoint_path),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        verbose=1,
    )
    latest_cb = ModelCheckpoint(
        filepath=str(config.CALLBACKS_DIR / "latest.weights.h5"),
        save_weights_only=True,
        save_freq="epoch",
        verbose=0,
    )
    csv_cb = CSVLogger(LOG_DIR / "training_log.csv", append=True)
    memory_cb = MemoryMonitoringCallback(log_frequency=10)
    progress_cb = tf.keras.callbacks.ProgbarLogger()

    nvml_cb = None
    if gpus and NvmlGpuMemLogger is not None:
        try:
            nvml_cb = NvmlGpuMemLogger(gpus, interval=10)
            logger.info("NVML memory logging enabled")
        except Exception as e:
            logger.warning(f"Failed to initialize NVML logger: {e}")

    sampler_cb = SizeAwareSamplerCallback(case_sampler)
    diff_cb = DifficultyAwareCallback(case_sampler, config, train_pairs)
    loss_ramp_cb = LossRampScheduler(loss_obj, config) if isinstance(loss_obj, HybridLoss) else None
    swa_cb = None
    if StochasticWeightAveraging is not None:
        try:
            swa_lr_val = None
            if config.SWA_LR_MULT is not None:
                swa_lr_val = max(config.MIN_LR, float(config.INITIAL_LR) * float(config.SWA_LR_MULT))
            swa_cb = StochasticWeightAveraging(
                start_epoch=max(0, config.TOTAL_EPOCHS - config.SWA_EPOCHS),
                swa_lr=swa_lr_val,
            )
        except Exception as e:
            logger.warning(f"Unable to enable SWA: {e}")

    callbacks = [cb for cb in (sampler_cb, diff_cb, loss_ramp_cb) if cb is not None]
    callbacks.extend([checkpoint_cb, latest_cb, csv_cb, memory_cb, progress_cb])
    callbacks.append(NonFiniteLossGuard())
    callbacks.append(tf.keras.callbacks.TerminateOnNaN())
    if nvml_cb is not None:
        callbacks.append(nvml_cb)
    if swa_cb is not None:
        callbacks.append(swa_cb)

    # --- Train the model ---
    history = model.fit(
        train_ds,
        epochs=config.TOTAL_EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        initial_epoch=config.INITIAL_EPOCH,
        verbose=1,
        steps_per_epoch=config.EPOCH_STEPS,
        validation_steps=len(val_gen),
        **{k: v for k, v in fit_kwargs.items() if v is not None}
    )
    logger.info(f"Training complete: {history.history.keys()}")
    return history


# ---------------------------------------------------------------------------
# Inference / evaluation utilities (quick wins without retraining)
# ---------------------------------------------------------------------------
def _gaussian_patch_weights(shape: tuple[int, int, int], sigma: float = 0.125) -> np.ndarray:
    sigma = max(float(sigma), 1e-4)
    coords = [np.linspace(-1.0, 1.0, num=int(s), dtype=np.float32) for s in shape]
    zz, yy, xx = np.meshgrid(*coords, indexing="ij")
    dist2 = zz**2 + yy**2 + xx**2
    w = np.exp(-0.5 * dist2 / (sigma**2))
    return np.maximum(w.astype(np.float32), 1e-4)


def _pad_volume_to_shape(volume: np.ndarray, target_shape: tuple[int, int, int]):
    pads = []
    for cur, tgt in zip(volume.shape, target_shape):
        if cur >= tgt:
            pads.append((0, 0))
        else:
            diff = int(tgt - cur)
            pads.append((diff // 2, diff - diff // 2))
    padded = np.pad(volume, pads, mode="constant", constant_values=0)
    return padded, pads


def _crop_from_pad(volume: np.ndarray, pads) -> np.ndarray:
    slices = tuple(slice(p0, volume.shape[i] - p1) for i, (p0, p1) in enumerate(pads))
    return volume[slices]


def _sliding_window_positions(shape: tuple[int, int, int], patch: tuple[int, int, int], overlap: float):
    overlap = float(np.clip(overlap, 0.0, 0.9))
    stride = [max(1, int(p * (1.0 - overlap))) for p in patch]
    stops = []
    for dim, p, st in zip(shape, patch, stride):
        if dim <= p:
            stops.append([0])
        else:
            coords = list(range(0, dim - p, st))
            if coords[-1] != dim - p:
                coords.append(dim - p)
            stops.append(coords)
    for z in stops[0]:
        for y in stops[1]:
            for x in stops[2]:
                yield z, y, x


def gaussian_tta_predict(
    model: tf.keras.Model,
    volume: np.ndarray,
    patch_size: tuple[int, int, int],
    overlap: float = 0.5,
    sigma: float = 0.125,
    tta: bool = True,
):
    patch_size = tuple(int(v) for v in patch_size)
    padded, pads = _pad_volume_to_shape(volume, patch_size)
    weight_patch = _gaussian_patch_weights(patch_size, sigma=sigma)
    accum = np.zeros_like(padded, dtype=np.float32)
    flip_sets = [()]
    if tta:
        flip_sets = [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]

    for axes in flip_sets:
        vol_aug = np.flip(padded, axis=axes) if axes else padded
        blended = np.zeros_like(padded, dtype=np.float32)
        weight_accum = np.zeros_like(padded, dtype=np.float32)
        for z0, y0, x0 in _sliding_window_positions(vol_aug.shape, patch_size, overlap):
            z1, y1, x1 = z0 + patch_size[0], y0 + patch_size[1], x0 + patch_size[2]
            patch = vol_aug[z0:z1, y0:y1, x0:x1]
            if patch.shape != patch_size:
                patch = _center_crop_or_pad_volume(patch, patch_size)
            patch_pred = model.predict(patch[np.newaxis, ..., np.newaxis], verbose=0)[0, ..., 0].astype(np.float32)
            blended[z0:z1, y0:y1, x0:x1] += patch_pred * weight_patch
            weight_accum[z0:z1, y0:y1, x0:x1] += weight_patch
        blended = blended / np.maximum(weight_accum, 1e-6)
        if axes:
            blended = np.flip(blended, axis=axes)
        accum += blended
    blended_avg = accum / float(len(flip_sets))
    return _crop_from_pad(blended_avg, pads)


def compute_brain_mask(volume: np.ndarray) -> np.ndarray:
    vals = volume[np.isfinite(volume)]
    if vals.size == 0:
        return np.ones_like(volume, dtype=np.uint8)
    try:
        thresh = threshold_otsu(vals)
    except Exception:
        thresh = np.percentile(vals, 40)
    mask = volume >= thresh
    mask = binary_closing(mask, structure=generate_binary_structure(3, 1))
    return mask.astype(np.uint8)


def per_case_otsu_threshold(probs: np.ndarray, brain_mask: np.ndarray | None, clamp: tuple[float, float], min_prob: float) -> float | None:
    region = probs[brain_mask > 0] if brain_mask is not None else probs
    region = region[np.isfinite(region)]
    region = region[region > float(min_prob)]
    if region.size == 0:
        return None
    try:
        thr = float(threshold_otsu(region))
    except Exception as e:
        logger.warning(f"Otsu failed, falling back to mean: {e}")
        thr = float(np.mean(region))
    return float(np.clip(thr, float(clamp[0]), float(clamp[1])))


def hysteresis_mask(probs: np.ndarray, t_low: float, t_high: float) -> np.ndarray:
    if t_high <= t_low:
        t_high = t_low + 1e-3
    strong = probs >= t_high
    weak = (probs >= t_low) & ~strong
    lbl, n = label(weak, structure=generate_binary_structure(3, 1))
    if n == 0:
        return strong.astype(np.float32)
    strong_lbls = np.unique(lbl[strong])
    if strong_lbls.size == 0:
        return strong.astype(np.float32)
    keep = np.isin(lbl, strong_lbls)
    return (strong | keep).astype(np.float32)


def apply_postprocessing(
    probs: np.ndarray,
    threshold: float | None,
    min_size: int = 0,
    closing: int = 0,
    hysteresis: tuple[float, float] | None = None,
    brain_mask: np.ndarray | None = None,
    clamp: tuple[float, float] = (0.05, 0.25),
    min_prob: float = 0.01,
) -> np.ndarray:
    work = probs
    if brain_mask is not None:
        work = work * (brain_mask > 0)
    if hysteresis is not None:
        pred_mask = hysteresis_mask(work, hysteresis[0], hysteresis[1])
    else:
        thr = threshold
        if thr is None:
            thr = per_case_otsu_threshold(work, brain_mask, clamp=clamp, min_prob=min_prob)
        if thr is None:
            thr = 0.1
        pred_mask = (work >= float(thr)).astype(np.float32)
    if closing:
        pred_mask = binary_closing(pred_mask, structure=generate_binary_structure(3, 1)).astype(np.float32)
    if min_size and pred_mask.any():
        lbl, n = label(pred_mask, structure=generate_binary_structure(3, 1))
        if n > 0:
            counts = np.bincount(lbl.ravel())
            remove = counts < int(min_size)
            if remove.size:
                remove[0] = False
                pred_mask = np.where(remove[lbl], 0, 1).astype(np.float32)
    return pred_mask


def _update_metrics(store: dict, key: str, y_true: np.ndarray, y_pred: np.ndarray):
    if key not in store:
        store[key] = {"macro": [], "inter": 0.0, "pred": 0.0, "true": 0.0}
    store[key]["macro"].append(dice_soft_np(y_true, y_pred))
    store[key]["inter"] += float(np.sum(y_true * y_pred))
    store[key]["pred"] += float(np.sum(y_pred))
    store[key]["true"] += float(np.sum(y_true))


def summarize_metrics(store: dict) -> dict:
    summary = {}
    for key, vals in store.items():
        macro = float(np.mean(vals["macro"])) if vals["macro"] else 0.0
        micro = float((2.0 * vals["inter"] + 1e-6) / (vals["pred"] + vals["true"] + 1e-6))
        summary[key] = {"macro_dice": macro, "micro_dice": micro, "cases": len(vals["macro"])}
    if summary:
        best_key = max(summary.items(), key=lambda kv: kv[1]["macro_dice"])[0]
        summary["_best"] = {"key": best_key, **summary[best_key]}
    return summary


def run_threshold_sweeps(
    model: tf.keras.Model,
    pairs,
    cfg: DynamicTrainingConfig,
    thresholds=None,
    min_sizes=(0, 1500, 3000, 6000),
    closing_opts=(0, 1),
    hysteresis_pairs=((0.15, 0.45), (0.20, 0.50)),
    use_tta=None,
):
    thresholds = thresholds or [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25]
    stats = {}
    patch_size = tuple(cfg.PATCH_SIZE or cfg.INPUT_SHAPE[:-1])
    use_tta = cfg.USE_TTA_FLIPS if use_tta is None else bool(use_tta)
    for img_p, msk_p in pairs:
        x = _load_and_preprocess_image(str(img_p), cfg.INPUT_SHAPE[:-1])
        y_true = _load_and_preprocess_mask(str(msk_p), cfg.INPUT_SHAPE[:-1])
        brain_mask = compute_brain_mask(x)
        probs = gaussian_tta_predict(
            model,
            x,
            patch_size=patch_size,
            overlap=cfg.GAUSSIAN_TILE_OVERLAP,
            sigma=cfg.GAUSSIAN_TILE_SIGMA,
            tta=use_tta,
        )
        otsu_thr = per_case_otsu_threshold(probs, brain_mask, clamp=cfg.OTSU_CLAMP, min_prob=cfg.OTSU_MIN_PROB) if cfg.USE_PER_CASE_OTSU else None
        for t in thresholds:
            for min_sz in min_sizes:
                for closing in closing_opts:
                    key = f"thr_{t:.2f}_ms{int(min_sz)}_c{int(closing)}"
                    pred = apply_postprocessing(
                        probs,
                        threshold=float(t),
                        min_size=min_sz,
                        closing=closing,
                        brain_mask=brain_mask,
                        clamp=cfg.OTSU_CLAMP,
                        min_prob=cfg.OTSU_MIN_PROB,
                    )
                    _update_metrics(stats, key, y_true, pred)
        if otsu_thr is not None:
            for min_sz in min_sizes:
                for closing in closing_opts:
                    key = f"otsu_{otsu_thr:.3f}_ms{int(min_sz)}_c{int(closing)}"
                    agg_key = f"otsu_ms{int(min_sz)}_c{int(closing)}"  # aggregate across cases
                    pred = apply_postprocessing(
                        probs,
                        threshold=otsu_thr,
                        min_size=min_sz,
                        closing=closing,
                        brain_mask=brain_mask,
                        clamp=cfg.OTSU_CLAMP,
                        min_prob=cfg.OTSU_MIN_PROB,
                    )
                    _update_metrics(stats, key, y_true, pred)
                    _update_metrics(stats, agg_key, y_true, pred)
        for (t_low, t_high) in hysteresis_pairs:
            key = f"hyst_{t_low:.2f}_{t_high:.2f}"
            pred = apply_postprocessing(
                probs,
                threshold=None,
                min_size=min_sizes[0],
                closing=0,
                hysteresis=(t_low, t_high),
                brain_mask=brain_mask,
                clamp=cfg.OTSU_CLAMP,
                min_prob=cfg.OTSU_MIN_PROB,
            )
            _update_metrics(stats, key, y_true, pred)
    return summarize_metrics(stats)


def build_model_for_inference(cfg: DynamicTrainingConfig, weights_path: str | None = None) -> tf.keras.Model:
    with strategy.scope():
        model = build_dynamic_model(cfg)
    if weights_path:
        model.load_weights(str(weights_path))
    return model


def quick_eval_from_config(
    cfg: DynamicTrainingConfig,
    weights_path: str,
    thresholds=None,
    limit_cases: int | None = None,
):
    """Load pairs and run the threshold/morphology sweeps."""
    pairs, _ = load_generic_dataset(cfg)
    if limit_cases is not None:
        pairs = pairs[:limit_cases]
    model = build_model_for_inference(cfg, weights_path=weights_path)
    return run_threshold_sweeps(model, pairs, cfg, thresholds=thresholds)
