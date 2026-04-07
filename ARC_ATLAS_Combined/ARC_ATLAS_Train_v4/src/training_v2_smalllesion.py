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
import csv
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
    from scipy.ndimage import (
        zoom,
        binary_dilation,
        rotate,
        binary_closing,
        binary_opening,
        label,
        generate_binary_structure,
        gaussian_filter,
        distance_transform_edt,
    )
    try:
        from sklearn.model_selection import StratifiedShuffleSplit
        try:
            from sklearn.model_selection import StratifiedGroupKFold
        except Exception:
            StratifiedGroupKFold = None
    except Exception:
        StratifiedShuffleSplit = None
        StratifiedGroupKFold = None
    try:
        from skimage.filters import threshold_otsu
    except Exception:
        def threshold_otsu(values):
            values = np.asarray(values, dtype=np.float32)
            if values.size == 0:
                return 0.0
            return float(np.percentile(values, 60))
    import json
    import time
    import math
    import random
    import re
    import gc
    import psutil
    from types import SimpleNamespace
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
    BCE_WEIGHT: float = 0.0
    VOLUME_RATIO_WEIGHT: float = 0.0
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
    SOURCE_BALANCED_SAMPLING: bool = True
    OUTPUT_BIAS_INIT_PROB: float = 0.015
    USE_SYMMETRIC_FLIP_CHANNEL: bool = True
    CASE_SIZE_BINS: tuple[int, ...] = (100, 1000, 10000)
    CASE_SIZE_GROUP_PROBS: tuple[float, ...] = (0.45, 0.25, 0.15, 0.10)
    CASE_NONE_PROB: float = 0.05
    USE_SIZE_CURRICULUM: bool = True
    CURRICULUM_EPOCHS: int = 12
    CURRICULUM_START_CASE_GROUP_PROBS: tuple[float, ...] = (0.70, 0.20, 0.07, 0.03)
    CURRICULUM_START_PATCH_FG_PROB_BY_BIN: tuple[float, ...] = (0.995, 0.99, 0.92, 0.78)
    CURRICULUM_START_CASE_NONE_PROB: float = 0.02
    USE_ATLAS_FINE_TUNE: bool = True
    ATLAS_FINE_TUNE_START_EPOCH: int = 70
    ATLAS_FINE_TUNE_SOURCE_PREFIXES: tuple[str, ...] = ("ATLAS",)
    ATLAS_FINE_TUNE_SOURCE_MASS: float = 0.70
    ATLAS_FINE_TUNE_CASE_GROUP_PROBS: tuple[float, ...] = (0.60, 0.25, 0.10, 0.05)
    ATLAS_FINE_TUNE_PATCH_FG_PROB_BY_BIN: tuple[float, ...] = (0.995, 0.985, 0.94, 0.82)
    ATLAS_FINE_TUNE_CASE_NONE_PROB: float = 0.02
    USE_COMPONENT_AWARE_PATCH_SAMPLING: bool = False
    USE_TINY_COMPONENT_CENTERING: bool = True
    TINY_COMPONENT_CENTER_PROB: float = 0.95
    SMALL_COMPONENT_CENTER_PROB: float = 0.85
    TINY_COMPONENT_MAX_JITTER: int = 2
    SMALL_COMPONENT_MAX_JITTER: int = 4
    MSL_COMPONENT_THRESHOLDS: tuple[int, ...] = (100, 1000, 10000)
    USE_AUX_MSL_HEAD: bool = True
    USE_AUX_DBL_HEAD: bool = True
    AUX_MSL_WEIGHT: float = 0.15
    AUX_DBL_WEIGHT: float = 0.10
    AUX_MSL_CLASS_WEIGHTS: tuple[float, ...] = (0.02, 4.0, 2.5, 1.0, 0.6)
    AUX_DBL_CLASS_WEIGHTS: tuple[float, ...] = (0.02, 1.15, 1.0)
    USE_CENTER_HEATMAP_HEAD: bool = False
    USE_SIZE_HEAD: bool = False
    CENTER_HEATMAP_SIGMA: float = 4.0
    CENTER_POSITIVE_WEIGHT: float = 10.0
    AUX_CENTER_WEIGHT: float = 0.12
    AUX_SIZE_WEIGHT: float = 0.05
    SIZE_HEAD_CLASS_WEIGHTS: tuple[float, ...] = (0.02, 4.0, 2.5, 1.0, 0.6)
    CENTER_TOPK_VALUES: tuple[int, ...] = (1, 3, 5, 10)
    CENTER_MATCH_RADIUS: float = 6.0
    CENTER_NMS_RADIUS: int = 6
    CENTER_MIN_CONFIDENCE: float = 0.01
    CENTER_HEAD_BIAS_INIT_PROB: float = 0.01
    CENTER_LOSS_GAMMA: float = 2.0
    CENTER_LOSS_BETA: float = 4.0
    TOPK_VOXEL_FRACTION: float = 0.10
    TOPK_WEIGHT: float = 0.20
    LESION_INSERTION_PROB: float = 0.20
    LESION_INSERTION_MAX_COMPONENT_VOXELS: int = 1000
    LESION_INSERTION_MAX_BANK_COMPONENTS: int = 512
    EXTERNAL_VAL_DIR: Optional[Path] = field(default_factory=lambda: _env_path("SMARTSOTA_EXTERNAL_VAL_DIR"))
    EXTERNAL_VAL_IMAGES_DIR: Optional[Path] = None
    EXTERNAL_VAL_MASKS_DIR: Optional[Path] = None
    EXTERNAL_VAL_MANIFEST: Optional[Path] = field(default_factory=lambda: _env_path("SMARTSOTA_EXTERNAL_VAL_MANIFEST"))
    GROUPED_CV_FOLDS: int = 3
    USE_BRAINMASK_POSTPROC: bool = False
    USE_COMPONENT_SCORING_POSTPROC: bool = False
    COMPONENT_SCORE_TINY_MIN_MEAN: float = 0.22
    COMPONENT_SCORE_TINY_MIN_MAX: float = 0.40
    COMPONENT_SCORE_SMALL_MIN_MEAN: float = 0.16
    COMPONENT_SCORE_SMALL_MIN_MAX: float = 0.32
    COMPONENT_SCORE_MIN_SIZE: int = 24
    DIFF_AWARE_ENABLED: bool = True
    DIFF_EMA_LAMBDA: float = 0.8
    DIFF_BETA: float = 1.5
    DIFF_MAX_EVAL_CASES: int = 32
    EPOCH_STEPS: int = 2000
    FIT_VERBOSE: int = 2
    MEMORY_LOGS_ENABLED: bool = False
    MEMORY_LOG_BATCH_FREQUENCY: int = 0
    LOSS_MODE: str = "combined"                          # "combined" | "tversky" | "focal_tversky"
    TVERSKY_ALPHA: float = 0.7
    TVERSKY_BETA: float = 0.3
    FOCAL_TVERSKY_GAMMA: float = 1.5
    FOCAL_TVERSKY_WEIGHT: float = 0.2
    RNG_SEED: int = 1234
    WHOLE_BRAIN_VAL_ENABLED: bool = True
    WHOLE_BRAIN_VAL_EVERY_N_EPOCHS: int = 1
    WHOLE_BRAIN_VAL_MAX_CASES: Optional[int] = None
    WHOLE_BRAIN_VAL_TTA: bool = False
    DIAGNOSTICS_ENABLED: bool = True
    BATCH_LOG_EVERY_N_STEPS: int = 1
    VAL_DIAGNOSTICS_TOP_K: int = 5
    VAL_THRESHOLD_SWEEP: tuple[float, ...] = (0.30, 0.40, 0.50, 0.60, 0.70)
    DIAGNOSTICS_COMPARE_POSTPROC: bool = True

    def __post_init__(self):
        if self.DATA_DIR is None:
            raise ValueError("DATA_DIR must be supplied via argument or SMARTSOTA_DATA_DIR environment variable.")
        self.DATA_DIR = Path(self.DATA_DIR)
        self.IMAGES_DIR = Path(self.IMAGES_DIR) if self.IMAGES_DIR else self.DATA_DIR
        self.MASKS_DIR = Path(self.MASKS_DIR) if self.MASKS_DIR else self.DATA_DIR
        self.MODEL_DIR = Path(self.MODEL_DIR)
        self.CALLBACKS_DIR = Path(self.CALLBACKS_DIR)
        self.EXTERNAL_VAL_DIR = Path(self.EXTERNAL_VAL_DIR) if self.EXTERNAL_VAL_DIR else None
        self.EXTERNAL_VAL_IMAGES_DIR = Path(self.EXTERNAL_VAL_IMAGES_DIR) if self.EXTERNAL_VAL_IMAGES_DIR else None
        self.EXTERNAL_VAL_MASKS_DIR = Path(self.EXTERNAL_VAL_MASKS_DIR) if self.EXTERNAL_VAL_MASKS_DIR else None
        self.EXTERNAL_VAL_MANIFEST = Path(self.EXTERNAL_VAL_MANIFEST) if self.EXTERNAL_VAL_MANIFEST else None
        self.PATCH_SAMPLING_STRATEGY = str(self.PATCH_SAMPLING_STRATEGY).strip().lower()
        if self.PATCH_SAMPLING_STRATEGY not in {"random", "hemisphere"}:
            raise ValueError("PATCH_SAMPLING_STRATEGY must be 'random' or 'hemisphere'.")
        if int(self.HEMISPHERE_AXIS) not in (0, 1, 2):
            raise ValueError("HEMISPHERE_AXIS must be 0, 1, or 2.")
        self.HEMISPHERE_AXIS = int(self.HEMISPHERE_AXIS)
        self.WHOLE_BRAIN_VAL_EVERY_N_EPOCHS = max(1, int(self.WHOLE_BRAIN_VAL_EVERY_N_EPOCHS))
        self.BATCH_LOG_EVERY_N_STEPS = max(1, int(self.BATCH_LOG_EVERY_N_STEPS))
        self.VAL_DIAGNOSTICS_TOP_K = max(1, int(self.VAL_DIAGNOSTICS_TOP_K))
        self.OUTPUT_BIAS_INIT_PROB = float(np.clip(self.OUTPUT_BIAS_INIT_PROB, 1e-5, 1.0 - 1e-5))
        self.TOPK_VOXEL_FRACTION = float(np.clip(self.TOPK_VOXEL_FRACTION, 0.0, 1.0))
        self.GROUPED_CV_FOLDS = max(2, int(self.GROUPED_CV_FOLDS))
        center_topks = [max(1, int(v)) for v in (self.CENTER_TOPK_VALUES or ())]
        self.CENTER_TOPK_VALUES = tuple(sorted(set(center_topks))) if center_topks else (1, 3, 5, 10)
        self.CENTER_HEATMAP_SIGMA = max(0.5, float(self.CENTER_HEATMAP_SIGMA))
        self.CENTER_MATCH_RADIUS = max(0.0, float(self.CENTER_MATCH_RADIUS))
        self.CENTER_NMS_RADIUS = max(1, int(self.CENTER_NMS_RADIUS))
        self.CENTER_MIN_CONFIDENCE = float(np.clip(self.CENTER_MIN_CONFIDENCE, 0.0, 1.0))
        self.CENTER_HEAD_BIAS_INIT_PROB = float(np.clip(self.CENTER_HEAD_BIAS_INIT_PROB, 1e-5, 1.0 - 1e-5))
        self.CENTER_POSITIVE_WEIGHT = max(1.0, float(self.CENTER_POSITIVE_WEIGHT))
        if self.INPUT_SHAPE is not None:
            spatial = tuple(int(v) for v in self.INPUT_SHAPE[:-1])
            self.INPUT_SHAPE = spatial + (self.input_channels,)
        thresholds = [float(np.clip(t, 0.0, 1.0)) for t in (self.VAL_THRESHOLD_SWEEP or ())]
        if float(np.clip(self.DECISION_THRESHOLD, 0.0, 1.0)) not in thresholds:
            thresholds.append(float(np.clip(self.DECISION_THRESHOLD, 0.0, 1.0)))
        self.VAL_THRESHOLD_SWEEP = tuple(sorted(set(thresholds))) if thresholds else (float(self.DECISION_THRESHOLD),)
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.CALLBACKS_DIR.mkdir(parents=True, exist_ok=True)
        self._write_config()

    def _write_config(self) -> None:
        payload = asdict(self)
        for key in (
            "DATA_DIR",
            "IMAGES_DIR",
            "MASKS_DIR",
            "MODEL_DIR",
            "CALLBACKS_DIR",
            "EXTERNAL_VAL_DIR",
            "EXTERNAL_VAL_IMAGES_DIR",
            "EXTERNAL_VAL_MASKS_DIR",
            "EXTERNAL_VAL_MANIFEST",
        ):
            payload[key] = str(payload[key]) if payload.get(key) is not None else None
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
            "VAL_THRESHOLD_SWEEP",
            "CASE_SIZE_BINS",
            "CASE_SIZE_GROUP_PROBS",
            "CURRICULUM_START_CASE_GROUP_PROBS",
            "CURRICULUM_START_PATCH_FG_PROB_BY_BIN",
            "ATLAS_FINE_TUNE_SOURCE_PREFIXES",
            "ATLAS_FINE_TUNE_CASE_GROUP_PROBS",
            "ATLAS_FINE_TUNE_PATCH_FG_PROB_BY_BIN",
            "MSL_COMPONENT_THRESHOLDS",
            "AUX_MSL_CLASS_WEIGHTS",
            "AUX_DBL_CLASS_WEIGHTS",
            "SIZE_HEAD_CLASS_WEIGHTS",
            "CENTER_TOPK_VALUES",
        )
        for key in tuple_fields:
            if payload.get(key) is not None:
                payload[key] = list(payload[key])
        with open(self.MODEL_DIR / "config.json", "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    @property
    def input_channels(self) -> int:
        return 2 if bool(getattr(self, "USE_SYMMETRIC_FLIP_CHANNEL", False)) else 1

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
    prior = float(np.clip(getattr(config, "OUTPUT_BIAS_INIT_PROB", 0.015), 1e-5, 1.0 - 1e-5))
    prior_bias = float(np.log(prior / (1.0 - prior)))
    center_prior = float(np.clip(getattr(config, "CENTER_HEAD_BIAS_INIT_PROB", 0.01), 1e-5, 1.0 - 1e-5))
    center_prior_bias = float(np.log(center_prior / (1.0 - center_prior)))
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

    # Start with a low foreground prior so the model does not begin near 50% occupancy.
    probs = layers.Conv3D(
        1,
        kernel_size=1,
        activation="sigmoid",
        bias_initializer=tf.keras.initializers.Constant(prior_bias),
        name="probs",
    )(x)
    outputs: dict[str, tf.Tensor] = {"probs": probs}
    if bool(getattr(config, "USE_CENTER_HEATMAP_HEAD", False)):
        outputs["center_heatmap"] = layers.Conv3D(
            1,
            kernel_size=1,
            activation="sigmoid",
            bias_initializer=tf.keras.initializers.Constant(center_prior_bias),
            name="center_heatmap",
        )(x)
    if bool(getattr(config, "USE_SIZE_HEAD", False)):
        outputs["size_head"] = layers.Conv3D(
            5,
            kernel_size=1,
            activation="softmax",
            name="size_head",
        )(x)
    if bool(getattr(config, "USE_AUX_MSL_HEAD", True)):
        outputs["msl_head"] = layers.Conv3D(
            5,
            kernel_size=1,
            activation="softmax",
            name="msl_head",
        )(x)
    if bool(getattr(config, "USE_AUX_DBL_HEAD", True)):
        outputs["dbl_head"] = layers.Conv3D(
            3,
            kernel_size=1,
            activation="softmax",
            name="dbl_head",
        )(x)
    if len(outputs) == 1:
        return tf.keras.Model(inputs=inputs, outputs=probs, name="SmartSOTA_SmallLesion")
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="SmartSOTA_SmallLesion")


def _prediction_output(pred, model: tf.keras.Model | None = None, output_name: str = "probs"):
    if isinstance(pred, dict):
        out = pred.get(output_name)
        if out is None:
            out = next(iter(pred.values()))
        return out
    if isinstance(pred, (list, tuple)):
        if model is not None:
            names = list(getattr(model, "output_names", []) or [])
            if output_name in names:
                return pred[names.index(output_name)]
        return pred[0]
    return pred


def _binary_output_from_prediction(pred, model: tf.keras.Model | None = None):
    return _prediction_output(pred, model=model, output_name="probs")


def _make_input_channels(image: np.ndarray, cfg: DynamicTrainingConfig) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if bool(getattr(cfg, "USE_SYMMETRIC_FLIP_CHANNEL", False)):
        flipped = np.flip(image, axis=int(getattr(cfg, "HEMISPHERE_AXIS", 2))).astype(np.float32, copy=False)
        return np.stack([image, flipped], axis=-1).astype(np.float32, copy=False)
    return image[..., np.newaxis].astype(np.float32, copy=False)



# ---------------------------------------------------------------------------
# Utility functions for memory monitoring
# ---------------------------------------------------------------------------
def log_memory_usage(stage: str) -> None:
    cfg = globals().get("_ACTIVE_CONFIG")
    if cfg is not None and not bool(getattr(cfg, "MEMORY_LOGS_ENABLED", False)):
        return
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
        self.log_frequency = max(0, int(log_frequency))
        self._batch = 0

    def on_train_begin(self, logs=None):
        log_memory_usage("train_begin")

    def on_epoch_begin(self, epoch, logs=None):
        log_memory_usage(f"epoch_{epoch}_start")

    def on_train_batch_end(self, batch, logs=None):
        self._batch += 1
        if self.log_frequency > 0 and self._batch % self.log_frequency == 0:
            log_memory_usage(f"batch_{self._batch}")

    def on_epoch_end(self, epoch, logs=None):
        log_memory_usage(f"epoch_{epoch}_end")


def _path_case_id(path: str | Path) -> str:
    name = Path(path).name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(path).stem


def _path_source(path: str | Path) -> str:
    name = Path(path).name
    if "__" in name:
        return name.split("__", 1)[0]
    return name.split("_", 1)[0]


def _optimizer_lr_value(model: tf.keras.Model) -> float:
    opt = getattr(model, "optimizer", None)
    if opt is None:
        return float("nan")
    lr_obj = getattr(opt, "learning_rate", None)
    try:
        if callable(lr_obj):
            return float(tf.keras.backend.get_value(lr_obj(opt.iterations)))
        return float(tf.keras.backend.get_value(lr_obj))
    except Exception:
        try:
            return float(lr_obj)
        except Exception:
            return float("nan")


def _metric_from_logs(logs: dict, base_name: str) -> float:
    if logs is None:
        return float("nan")
    for key in (base_name, f"probs_{base_name}", f"val_{base_name}", f"val_probs_{base_name}"):
        if key in logs:
            try:
                return float(logs.get(key, np.nan))
            except Exception:
                return float("nan")
    return float("nan")


class BatchMetricsCSVLogger(tf.keras.callbacks.Callback):
    """Write per-train-batch metrics to CSV for fine-grained debugging."""

    def __init__(self, out_csv: Path, log_every_n_steps: int = 1):
        super().__init__()
        self.out_csv = Path(out_csv)
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self._fh = None
        self._writer = None
        self._global_step = 0
        self._epoch = 0

    def on_train_begin(self, logs=None):
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.out_csv, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(
            ["epoch", "batch", "global_step", "lr", "loss", "dice_coefficient", "safe_binary_iou"]
        )
        self._fh.flush()

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch = int(epoch)

    def on_train_batch_end(self, batch, logs=None):
        self._global_step += 1
        if self._writer is None or (self._global_step % self.log_every_n_steps) != 0:
            return
        logs = logs or {}
        self._writer.writerow(
            [
                self._epoch,
                int(batch),
                self._global_step,
                _optimizer_lr_value(self.model),
                float(logs.get("loss", np.nan)),
                _metric_from_logs(logs, "dice_coefficient"),
                _metric_from_logs(logs, "safe_binary_iou"),
            ]
        )
        self._fh.flush()

    def on_train_end(self, logs=None):
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._writer = None


class EpochMetricsJSONLLogger(tf.keras.callbacks.Callback):
    """Append one JSON record per epoch (after validation callbacks)."""

    def __init__(self, out_jsonl: Path):
        super().__init__()
        self.out_jsonl = Path(out_jsonl)
        self._t0 = None

    def on_train_begin(self, logs=None):
        self.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        self._t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        record = {
            "epoch": int(epoch),
            "elapsed_sec": float(time.time() - self._t0) if self._t0 is not None else None,
            "lr": _optimizer_lr_value(self.model),
            "metrics": {k: float(v) for k, v in logs.items() if isinstance(v, (int, float, np.floating))},
        }
        with open(self.out_jsonl, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")


def _lesion_size_bin(lesion_voxels: int) -> str:
    v = int(lesion_voxels)
    if v <= 0:
        return "none"
    if v < 2000:
        return "tiny"
    if v < 10000:
        return "small"
    if v < 50000:
        return "medium"
    return "large"


def _subject_group_key(path: str | Path) -> str:
    name = Path(path).name
    m = re.search(r"(sub-[A-Za-z0-9]+)", name)
    if m:
        return m.group(1)
    return _path_case_id(path).split("_", 1)[0]


def _smallest_positive_component(mask: np.ndarray) -> int:
    mask_bin = np.asarray(mask > 0.5, dtype=np.uint8)
    if not np.any(mask_bin):
        return 0
    lbl, n = label(mask_bin, structure=generate_binary_structure(3, 1))
    if n <= 0:
        return 0
    counts = np.bincount(lbl.ravel())
    pos = counts[1:]
    return int(pos.min()) if pos.size else 0


def _case_size_group(min_component_voxels: int, bins: tuple[int, ...]) -> str:
    v = int(min_component_voxels)
    edges = tuple(int(b) for b in bins)
    if v <= 0:
        return "none"
    if v < edges[0]:
        return f"1_{edges[0]-1}"
    if v < edges[1]:
        return f"{edges[0]}_{edges[1]-1}"
    if v < edges[2]:
        return f"{edges[1]}_{edges[2]-1}"
    return f"{edges[2]}_plus"


def _build_component_records(mask: np.ndarray) -> list[dict[str, object]]:
    mask_bin = np.asarray(mask > 0.5, dtype=np.uint8)
    if not np.any(mask_bin):
        return []
    lbl, n = label(mask_bin, structure=generate_binary_structure(3, 1))
    if n <= 0:
        return []
    counts = np.bincount(lbl.ravel())
    records: list[dict[str, object]] = []
    for comp_id in range(1, n + 1):
        size = int(counts[comp_id]) if comp_id < counts.size else 0
        if size <= 0:
            continue
        coords = np.argwhere(lbl == comp_id)
        mins = coords.min(axis=0).astype(np.int64)
        maxs = coords.max(axis=0).astype(np.int64)
        centroid = np.round(coords.mean(axis=0)).astype(np.int64)
        records.append(
            {
                "label": int(comp_id),
                "size": size,
                "coords": coords,
                "mins": mins,
                "maxs": maxs,
                "centroid": centroid,
            }
        )
    return records


def _component_size_group(size: int, bins: tuple[int, ...]) -> str:
    return _case_size_group(int(size), bins)


def _write_split_diagnostics(
    train_pairs,
    val_pairs,
    pair_lookup: dict[tuple[str, str], int],
    lesion_sizes_all: np.ndarray,
    smallest_component_sizes_all: np.ndarray | None,
    case_size_bins: tuple[int, ...],
    out_dir: Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for split_name, pairs in (("train", train_pairs), ("val", val_pairs)):
        for img_p, msk_p in pairs:
            idx = pair_lookup.get((str(img_p), str(msk_p)))
            lesion_voxels = int(lesion_sizes_all[idx]) if idx is not None else -1
            smallest_component = (
                int(smallest_component_sizes_all[idx])
                if idx is not None and smallest_component_sizes_all is not None
                else -1
            )
            rows.append(
                {
                    "split": split_name,
                    "source": _path_source(img_p),
                    "case_id": _path_case_id(img_p),
                    "lesion_voxels": lesion_voxels,
                    "lesion_bin": _lesion_size_bin(lesion_voxels),
                    "smallest_component_voxels": smallest_component,
                    "case_size_group": _case_size_group(smallest_component, case_size_bins),
                    "image": str(img_p),
                    "mask": str(msk_p),
                }
            )
    with open(out_dir / "split_cases.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "split",
                "source",
                "case_id",
                "lesion_voxels",
                "lesion_bin",
                "smallest_component_voxels",
                "case_size_group",
                "image",
                "mask",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {"total_cases": len(rows), "splits": {}}
    for split_name in ("train", "val"):
        split_rows = [r for r in rows if r["split"] == split_name]
        lesions = np.asarray([r["lesion_voxels"] for r in split_rows], dtype=np.int64)
        src_counts = {}
        bin_counts = {}
        case_group_counts = {}
        for r in split_rows:
            src_counts[r["source"]] = src_counts.get(r["source"], 0) + 1
            bin_counts[r["lesion_bin"]] = bin_counts.get(r["lesion_bin"], 0) + 1
            case_group_counts[r["case_size_group"]] = case_group_counts.get(r["case_size_group"], 0) + 1
        summary["splits"][split_name] = {
            "count": len(split_rows),
            "source_counts": src_counts,
            "lesion_bin_counts": bin_counts,
            "case_size_group_counts": case_group_counts,
            "lesion_presence_pct": float(np.mean(lesions > 0) * 100.0) if lesions.size else 0.0,
            "lesion_voxels": {
                "mean": float(np.mean(lesions)) if lesions.size else 0.0,
                "median": float(np.median(lesions)) if lesions.size else 0.0,
                "p90": float(np.percentile(lesions, 90)) if lesions.size else 0.0,
                "max": int(np.max(lesions)) if lesions.size else 0,
            },
        }
    with open(out_dir / "split_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("🧪 Wrote split diagnostics to %s", out_dir)


def _best_epoch_stat(values, mode: str = "max") -> dict[str, float | int] | None:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None
    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        return None
    finite_idx = np.where(finite_mask)[0]
    finite_vals = arr[finite_mask]
    if mode == "min":
        local_best = int(np.argmin(finite_vals))
    else:
        local_best = int(np.argmax(finite_vals))
    best_idx = int(finite_idx[local_best])
    return {"epoch": best_idx, "value": float(arr[best_idx])}


def _write_training_summary(history, config: DynamicTrainingConfig) -> None:
    """Write compact run-level diagnostics summary from epoch history + callbacks outputs."""
    callbacks_dir = Path(config.CALLBACKS_DIR)
    out_dir = callbacks_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    hist = getattr(history, "history", {}) or {}
    metric_names = sorted(hist.keys())
    epoch_count = max((len(v) for v in hist.values()), default=0)

    best_by_metric = {}
    for metric in metric_names:
        values = hist.get(metric, [])
        mode = "min" if ("loss" in metric and "dice" not in metric and "iou" not in metric) else "max"
        best_stat = _best_epoch_stat(values, mode=mode)
        if best_stat is not None:
            best_by_metric[metric] = best_stat

    final_metrics = {
        k: float(v[-1]) for k, v in hist.items() if isinstance(v, list) and len(v) > 0 and np.isfinite(v[-1])
    }

    def _final_metric(*names):
        for name in names:
            if name in final_metrics:
                return final_metrics[name]
        return None

    # Basic training health checks to quickly identify failure modes.
    warnings = []
    train_d = _final_metric("dice_coefficient", "probs_dice_coefficient")
    val_d = _final_metric("val_dice_coefficient", "val_probs_dice_coefficient")
    val_h = _final_metric("val_whole_dice_hard_raw", "val_whole_dice_hard_brainmask", "val_whole_dice_hard")
    val_h_post = final_metrics.get("val_whole_dice_hard")
    val_h_raw = final_metrics.get("val_whole_dice_hard_raw")
    if train_d is not None and val_d is not None and (train_d - val_d) > 0.15:
        warnings.append("Large train/val dice gap detected (>0.15): possible overfitting or split/domain mismatch.")
    if val_d is not None and val_d < 0.02:
        warnings.append("Validation dice stayed very low (<0.02): likely training collapse or severe class/domain mismatch.")
    if val_h is not None and val_d is not None and val_h < 0.01 and val_d > 0.05:
        warnings.append(
            "Hard whole-brain dice is much lower than soft dice: check threshold calibration and predicted volume bias."
        )
    if val_h_post is not None and val_h_raw is not None and (val_h_post - val_h_raw) < -0.02:
        warnings.append(
            "Postprocessed hard Dice is materially below raw hard Dice (<-0.02): brain-mask clipping or component filtering is likely suppressing true positives."
        )

    batch_csv = callbacks_dir / "batch_metrics.csv"
    batch_summary = {}
    if batch_csv.exists():
        losses, dices, ious = [], [], []
        with open(batch_csv, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    losses.append(float(row.get("loss", "nan")))
                    dices.append(float(row.get("dice_coefficient", "nan")))
                    ious.append(float(row.get("safe_binary_iou", "nan")))
                except Exception:
                    continue
        if losses:
            loss_arr = np.asarray(losses, dtype=np.float64)
            dice_arr = np.asarray(dices, dtype=np.float64)
            iou_arr = np.asarray(ious, dtype=np.float64)
            batch_summary = {
                "rows": int(len(loss_arr)),
                "loss": {
                    "mean": float(np.nanmean(loss_arr)),
                    "p95": float(np.nanpercentile(loss_arr, 95)),
                    "max": float(np.nanmax(loss_arr)),
                },
                "dice": {
                    "mean": float(np.nanmean(dice_arr)),
                    "p05": float(np.nanpercentile(dice_arr, 5)),
                    "min": float(np.nanmin(dice_arr)),
                },
                "iou": {
                    "mean": float(np.nanmean(iou_arr)),
                    "p05": float(np.nanpercentile(iou_arr, 5)),
                    "min": float(np.nanmin(iou_arr)),
                },
            }

    whole_summary_path = callbacks_dir / "whole_val_summary.jsonl"
    whole_epoch_rows = []
    if whole_summary_path.exists():
        with open(whole_summary_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    whole_epoch_rows.append(json.loads(line))
                except Exception:
                    continue

    source_best = {}
    for row in whole_epoch_rows:
        src_vals = row.get("source_soft_macro", {}) or {}
        epoch = int(row.get("epoch", -1))
        for src, val in src_vals.items():
            cur = source_best.get(src)
            fv = float(val)
            if cur is None or fv > cur["value"]:
                source_best[src] = {"epoch": epoch, "value": fv}

    summary_payload = {
        "run_dir": str(Path(config.CALLBACKS_DIR).parent),
        "callbacks_dir": str(callbacks_dir),
        "epochs_recorded": int(epoch_count),
        "metrics": metric_names,
        "best_by_metric": best_by_metric,
        "final_metrics": final_metrics,
        "warnings": warnings,
        "batch_summary": batch_summary,
        "source_best_soft_macro": source_best,
        "artifacts": {
            "training_log_csv": str(callbacks_dir / "training_log.csv"),
            "batch_metrics_csv": str(batch_csv),
            "epoch_metrics_jsonl": str(callbacks_dir / "epoch_metrics.jsonl"),
            "sampling_schedule_jsonl": str(callbacks_dir / "sampling_schedule.jsonl"),
            "whole_val_summary_jsonl": str(whole_summary_path),
            "split_summary_json": str(out_dir / "split_summary.json"),
            "split_cases_csv": str(out_dir / "split_cases.csv"),
        },
    }

    out_json = out_dir / "training_summary.json"
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, indent=2)

    lines = [
        "# Training Diagnostics Summary",
        "",
        f"- Epochs recorded: {summary_payload['epochs_recorded']}",
        f"- Run dir: `{summary_payload['run_dir']}`",
        "",
        "## Final Metrics",
    ]
    if final_metrics:
        for k in sorted(final_metrics.keys()):
            lines.append(f"- `{k}`: {final_metrics[k]:.6f}")
    else:
        lines.append("- No final metrics found.")

    lines.extend(["", "## Best Metrics (Epoch, Value)"])
    if best_by_metric:
        for k in sorted(best_by_metric.keys()):
            st = best_by_metric[k]
            lines.append(f"- `{k}`: epoch {st['epoch']} -> {st['value']:.6f}")
    else:
        lines.append("- No best-metric stats available.")

    lines.extend(["", "## Source Best Soft Dice"])
    if source_best:
        for src in sorted(source_best.keys()):
            st = source_best[src]
            lines.append(f"- `{src}`: epoch {st['epoch']} -> {st['value']:.6f}")
    else:
        lines.append("- No source-level whole-brain summaries found.")

    lines.extend(["", "## Warnings"])
    if warnings:
        for w in warnings:
            lines.append(f"- {w}")
    else:
        lines.append("- None")

    lines.extend(["", "## Artifacts"])
    for k, v in summary_payload["artifacts"].items():
        lines.append(f"- `{k}`: `{v}`")

    out_md = out_dir / "training_summary.md"
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    logger.info("🧪 Wrote training diagnostics summary to %s", out_json)


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
def foreground_ratio_loss(y_true, y_pred, eps=1e-6, delta=1.0):
    """Penalize patch-level foreground volume mismatch in log space."""
    y_true = tf.where(tf.math.is_finite(y_true), y_true, tf.zeros_like(y_true))
    y_pred = tf.where(tf.math.is_finite(y_pred), y_pred, tf.zeros_like(y_pred))
    y_true = tf.cast(tf.clip_by_value(y_true, 0.0, 1.0), tf.float32)
    y_pred = tf.cast(tf.clip_by_value(y_pred, eps, 1.0 - eps), tf.float32)
    axes = tuple(range(1, len(y_true.shape)))
    true_frac = tf.reduce_mean(y_true, axis=axes)
    pred_frac = tf.reduce_mean(y_pred, axis=axes)
    diff = tf.math.log(pred_frac + eps) - tf.math.log(true_frac + eps)
    abs_diff = tf.abs(diff)
    huber = tf.where(abs_diff <= delta, 0.5 * tf.square(diff), delta * (abs_diff - 0.5 * delta))
    return tf.reduce_mean(huber)

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
        bce_weight=0.0,
        topk_weight=0.0,
        topk_voxel_fraction=0.0,
        volume_ratio_weight=0.0,
        focal_weight=0.0,
        tversky_alpha=0.7,
        tversky_beta=0.3,
        focal_gamma=1.5,
        name="hybrid_loss",
    ):
        super().__init__(name=name)
        self.dice_weight = float(dice_weight)
        self.boundary_weight = float(boundary_weight)
        self.bce_weight = float(bce_weight)
        self.topk_weight = float(topk_weight)
        self.topk_voxel_fraction = float(topk_voxel_fraction)
        self.volume_ratio_weight = float(volume_ratio_weight)
        self.focal_weight = float(focal_weight)
        self.tversky_alpha = float(tversky_alpha)
        self.tversky_beta = float(tversky_beta)
        self.focal_gamma = float(focal_gamma)

    def get_config(self):
        return {
            "dice_weight": self.dice_weight,
            "boundary_weight": self.boundary_weight,
            "bce_weight": self.bce_weight,
            "topk_weight": self.topk_weight,
            "topk_voxel_fraction": self.topk_voxel_fraction,
            "volume_ratio_weight": self.volume_ratio_weight,
            "focal_weight": self.focal_weight,
            "tversky_alpha": self.tversky_alpha,
            "tversky_beta": self.tversky_beta,
            "focal_gamma": self.focal_gamma,
        }

    def set_weights(
        self,
        dice_weight=None,
        boundary_weight=None,
        bce_weight=None,
        topk_weight=None,
        volume_ratio_weight=None,
        focal_weight=None,
    ):
        if dice_weight is not None:
            self.dice_weight = float(dice_weight)
        if boundary_weight is not None:
            self.boundary_weight = float(boundary_weight)
        if bce_weight is not None:
            self.bce_weight = float(bce_weight)
        if topk_weight is not None:
            self.topk_weight = float(topk_weight)
        if volume_ratio_weight is not None:
            self.volume_ratio_weight = float(volume_ratio_weight)
        if focal_weight is not None:
            self.focal_weight = float(focal_weight)

    def call(self, y_true, y_pred):
        loss_val = self.dice_weight * dice_loss(y_true, y_pred) + self.boundary_weight * boundary_loss(y_true, y_pred)
        if self.bce_weight > 0.0:
            bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            loss_val += self.bce_weight * tf.reduce_mean(bce)
        if self.topk_weight > 0.0 and self.topk_voxel_fraction > 0.0:
            bce = tf.reshape(tf.keras.losses.binary_crossentropy(y_true, y_pred), [-1])
            k = tf.cast(tf.math.maximum(1.0, tf.cast(tf.size(bce), tf.float32) * self.topk_voxel_fraction), tf.int32)
            topk_vals = tf.math.top_k(bce, k=k, sorted=False).values
            loss_val += self.topk_weight * tf.reduce_mean(topk_vals)
        if self.volume_ratio_weight > 0.0:
            loss_val += self.volume_ratio_weight * foreground_ratio_loss(y_true, y_pred)
        if self.focal_weight > 0.0:
            loss_val += self.focal_weight * focal_tversky_loss(
                y_true, y_pred, alpha=self.tversky_alpha, beta=self.tversky_beta, gamma=self.focal_gamma
            )
        return loss_val


@register_keras_serializable(package="custom")
class WeightedSparseCategoricalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, class_weights, name="weighted_sparse_cce"):
        super().__init__(name=name)
        self.class_weights = [float(v) for v in class_weights]

    def get_config(self):
        return {"class_weights": self.class_weights}

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        if y_true.shape.rank == y_pred.shape.rank:
            y_true = tf.squeeze(y_true, axis=-1)
        weights = tf.constant(self.class_weights, dtype=tf.float32)
        sample_w = tf.gather(weights, tf.clip_by_value(y_true, 0, len(self.class_weights) - 1))
        per_voxel = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        return tf.reduce_mean(tf.cast(sample_w, per_voxel.dtype) * per_voxel)


@register_keras_serializable(package="custom")
class WeightedBinaryCrossentropy(tf.keras.losses.Loss):
    def __init__(self, positive_weight=1.0, negative_weight=1.0, name="weighted_bce"):
        super().__init__(name=name)
        self.positive_weight = float(positive_weight)
        self.negative_weight = float(negative_weight)

    def get_config(self):
        return {"positive_weight": self.positive_weight, "negative_weight": self.negative_weight}

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(tf.clip_by_value(y_pred, 1e-6, 1.0 - 1e-6), tf.float32)
        per_voxel = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        weights = tf.where(y_true > 0.5, self.positive_weight, self.negative_weight)
        if weights.shape.rank is not None and per_voxel.shape.rank is not None and weights.shape.rank == per_voxel.shape.rank + 1:
            weights = tf.squeeze(weights, axis=-1)
        else:
            while weights.shape.rank is not None and per_voxel.shape.rank is not None and weights.shape.rank > per_voxel.shape.rank:
                weights = tf.squeeze(weights, axis=-1)
        return tf.reduce_mean(tf.cast(weights, per_voxel.dtype) * per_voxel)


@register_keras_serializable(package="custom")
class GaussianHeatmapFocalLoss(tf.keras.losses.Loss):
    """CenterNet-style focal loss for sparse heatmap targets with Gaussian falloff."""

    def __init__(self, gamma=2.0, beta=4.0, name="gaussian_heatmap_focal_loss"):
        super().__init__(name=name)
        self.gamma = float(gamma)
        self.beta = float(beta)

    def get_config(self):
        return {"gamma": self.gamma, "beta": self.beta}

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(tf.clip_by_value(y_pred, 1e-6, 1.0 - 1e-6), tf.float32)

        pos_inds = tf.cast(tf.equal(y_true, 1.0), tf.float32)
        neg_inds = tf.cast(tf.less(y_true, 1.0), tf.float32)
        neg_weights = tf.pow(1.0 - y_true, self.beta)

        pos_loss = -tf.math.log(y_pred) * tf.pow(1.0 - y_pred, self.gamma) * pos_inds
        neg_loss = -tf.math.log(1.0 - y_pred) * tf.pow(y_pred, self.gamma) * neg_weights * neg_inds

        num_pos = tf.reduce_sum(pos_inds)
        pos_loss = tf.reduce_sum(pos_loss)
        neg_loss = tf.reduce_sum(neg_loss)
        return tf.where(num_pos > 0.0, (pos_loss + neg_loss) / num_pos, neg_loss)


def _size_class_index(size: int, thresholds: tuple[int, ...]) -> int:
    size = int(size)
    if size <= 0:
        return 0
    if size < int(thresholds[0]):
        return 1
    if size < int(thresholds[1]):
        return 2
    if size < int(thresholds[2]):
        return 3
    return 4


def _component_peak_coord(component_mask: np.ndarray) -> tuple[int, int, int] | None:
    component_mask = np.asarray(component_mask > 0, dtype=np.uint8)
    if not np.any(component_mask):
        return None
    dist = distance_transform_edt(component_mask)
    if np.any(dist > 0):
        return tuple(int(v) for v in np.unravel_index(int(np.argmax(dist)), dist.shape))
    coords = np.argwhere(component_mask > 0)
    if coords.size == 0:
        return None
    return tuple(int(v) for v in coords[len(coords) // 2])


def build_msl_labels(mask: np.ndarray, thresholds: tuple[int, ...]) -> np.ndarray:
    mask_bin = np.asarray(mask > 0.5, dtype=np.uint8)
    labels = np.zeros(mask_bin.shape, dtype=np.int32)
    if not np.any(mask_bin):
        return labels
    lbl, n = label(mask_bin, structure=generate_binary_structure(3, 1))
    counts = np.bincount(lbl.ravel())
    for comp_id in range(1, n + 1):
        size = int(counts[comp_id]) if comp_id < counts.size else 0
        if size <= 0:
            continue
        if size < int(thresholds[0]):
            cls = 1
        elif size < int(thresholds[1]):
            cls = 2
        elif size < int(thresholds[2]):
            cls = 3
        else:
            cls = 4
        labels[lbl == comp_id] = cls
    return labels


def build_dbl_labels(mask: np.ndarray, boundary_distance: float = np.sqrt(4.5)) -> np.ndarray:
    mask_bin = np.asarray(mask > 0.5, dtype=np.uint8)
    labels = np.zeros(mask_bin.shape, dtype=np.int32)
    if not np.any(mask_bin):
        return labels
    dist = distance_transform_edt(mask_bin > 0)
    labels[mask_bin > 0] = 1
    labels[(mask_bin > 0) & (dist > float(boundary_distance))] = 2
    return labels


def build_center_heatmap(mask: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    mask_bin = np.asarray(mask > 0.5, dtype=np.uint8)
    heatmap = np.zeros(mask_bin.shape, dtype=np.float32)
    if not np.any(mask_bin):
        return heatmap[..., np.newaxis]
    lbl, n = label(mask_bin, structure=generate_binary_structure(3, 1))
    for comp_id in range(1, n + 1):
        peak = _component_peak_coord(lbl == comp_id)
        if peak is not None:
            heatmap[peak] = 1.0
    if float(sigma) > 0.0 and np.any(heatmap > 0):
        heatmap = gaussian_filter(heatmap, sigma=float(sigma), mode="constant")
    mx = float(np.max(heatmap)) if heatmap.size else 0.0
    if mx > 0.0:
        heatmap /= mx
    return heatmap[..., np.newaxis].astype(np.float32, copy=False)


def build_size_head_labels(mask: np.ndarray, thresholds: tuple[int, ...]) -> np.ndarray:
    return build_msl_labels(mask, thresholds)


def extract_component_seed_records(mask: np.ndarray, case_size_bins=(100, 1000, 10000)) -> list[dict[str, object]]:
    mask_bin = np.asarray(mask > 0.5, dtype=np.uint8)
    if not np.any(mask_bin):
        return []
    lbl, n = label(mask_bin, structure=generate_binary_structure(3, 1))
    counts = np.bincount(lbl.ravel())
    thresholds = tuple(int(v) for v in case_size_bins)
    records = []
    for comp_id in range(1, n + 1):
        size = int(counts[comp_id]) if comp_id < counts.size else 0
        if size <= 0:
            continue
        peak = _component_peak_coord(lbl == comp_id)
        if peak is None:
            continue
        records.append(
            {
                "coord": tuple(int(v) for v in peak),
                "size": size,
                "group": _case_size_group(size, thresholds),
                "class_index": _size_class_index(size, thresholds),
            }
        )
    return records


def _format_training_targets(mask: np.ndarray, cfg: DynamicTrainingConfig):
    y_main = np.asarray(mask > 0.5, dtype=np.float32)[..., np.newaxis]
    targets: dict[str, np.ndarray] = {"probs": y_main}
    if bool(getattr(cfg, "USE_CENTER_HEATMAP_HEAD", False)):
        targets["center_heatmap"] = build_center_heatmap(mask, sigma=float(getattr(cfg, "CENTER_HEATMAP_SIGMA", 2.0)))
    if bool(getattr(cfg, "USE_SIZE_HEAD", False)):
        targets["size_head"] = build_size_head_labels(mask, tuple(getattr(cfg, "CASE_SIZE_BINS", (100, 1000, 10000))))
    if bool(getattr(cfg, "USE_AUX_MSL_HEAD", True)):
        targets["msl_head"] = build_msl_labels(mask, tuple(getattr(cfg, "MSL_COMPONENT_THRESHOLDS", (100, 1000, 10000))))
    if bool(getattr(cfg, "USE_AUX_DBL_HEAD", True)):
        targets["dbl_head"] = build_dbl_labels(mask)
    if len(targets) == 1:
        return y_main
    return targets


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
        logger.info(
            "📉 Loss mix @epoch %d: dice=%.3f, boundary=%.3f, bce=%.3f, topk=%.3f, volume=%.3f, focal=%.3f",
            epoch,
            dice_w,
            boundary_w,
            self.loss_obj.bce_weight,
            self.loss_obj.topk_weight,
            self.loss_obj.volume_ratio_weight,
            self.loss_obj.focal_weight,
        )


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


def compute_smallest_component_sizes(pairs, load_mask_fn, target_shape=None):
    sizes = []
    for _, msk_p in pairs:
        y = load_mask_fn(str(msk_p), target_shape).astype(np.float32)
        sizes.append(_smallest_positive_component(y))
    return np.asarray(sizes, dtype=np.int64)


def create_stratified_splits(
    pairs,
    lesion_presence,
    batch_size,
    lesion_sizes=None,
    smallest_component_sizes=None,
    case_size_bins=(100, 1000, 10000),
    test_size=0.1,
    random_state: int = 42,
):
    """Create deterministic train/val splits, preferring source+case-size stratification."""
    total = len(pairs)
    if total == 0:
        return [], []
    if total == 1:
        return list(pairs), []

    y = np.asarray(lesion_presence, dtype=np.int64).reshape(-1)
    if y.size != total:
        logger.warning(
            "Lesion labels length (%d) does not match pairs (%d); using non-stratified split.",
            y.size, total,
        )
        y = np.zeros(total, dtype=np.int64)

    batch_size = max(1, int(batch_size))
    ratio = float(np.clip(float(test_size), 0.01, 0.99))
    val_count = max(1, int(round(total * ratio)))
    val_count = min(val_count, total - 1)

    # Keep splits batch-aligned when there is enough data for that constraint.
    if batch_size > 1 and total >= (2 * batch_size):
        val_count = max(batch_size, int(round(val_count / batch_size)) * batch_size)
        val_count = min(val_count, total - batch_size)
        val_count = max(batch_size, val_count)
    train_count = total - val_count
    if train_count < 1:
        train_count, val_count = total - 1, 1

    def _can_stratify(labels: np.ndarray):
        unique, counts = np.unique(labels, return_counts=True)
        ok = (
            unique.size >= 2
            and np.all(counts >= 2)
            and val_count >= unique.size
            and train_count >= unique.size
        )
        return unique, counts, ok

    def _source_key(pair):
        img_p, _ = pair
        name = Path(str(img_p)).name
        if "__" in name:
            return name.split("__", 1)[0]
        return name.split("_", 1)[0]

    source_labels = np.asarray([_source_key(p) for p in pairs], dtype=object)
    if lesion_sizes is not None and len(lesion_sizes) == total:
        size_labels = np.asarray([_lesion_size_bin(int(v)) for v in lesion_sizes], dtype=object)
    else:
        size_labels = np.asarray(["unknown"] * total, dtype=object)
    if smallest_component_sizes is not None and len(smallest_component_sizes) == total:
        case_size_labels = np.asarray(
            [_case_size_group(int(v), tuple(case_size_bins)) for v in smallest_component_sizes],
            dtype=object,
        )
    else:
        case_size_labels = size_labels
    source_lesion_labels = np.asarray(
        [f"{src}|lesion={int(lbl)}" for src, lbl in zip(source_labels, y)],
        dtype=object,
    )
    source_size_labels = np.asarray(
        [f"{src}|size={size_lbl}" for src, size_lbl in zip(source_labels, size_labels)],
        dtype=object,
    )
    source_case_size_labels = np.asarray(
        [f"{src}|case={size_lbl}" for src, size_lbl in zip(source_labels, case_size_labels)],
        dtype=object,
    )

    split_mode = "random_fallback"
    strat_labels = None
    class_counts = None

    for mode_name, labels in (
        ("stratified_source+case_size", source_case_size_labels),
        ("stratified_source+size", source_size_labels),
        ("stratified_source+lesion", source_lesion_labels),
        ("stratified_source", source_labels),
        ("stratified_case_size", case_size_labels),
        ("stratified_size", size_labels),
        ("stratified_lesion", y),
    ):
        unique, counts, ok = _can_stratify(np.asarray(labels))
        if ok:
            split_mode = mode_name
            strat_labels = np.asarray(labels)
            class_counts = dict(zip(unique.tolist(), counts.tolist()))
            break

    if strat_labels is not None and StratifiedShuffleSplit is not None:
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=val_count, random_state=random_state
        )
        train_idx, val_idx = next(
            splitter.split(np.zeros(total, dtype=np.int8), strat_labels)
        )
    else:
        rng = np.random.default_rng(random_state)
        order = np.arange(total)
        rng.shuffle(order)
        val_idx = order[:val_count]
        train_idx = order[val_count:]
        unique, counts, _ = _can_stratify(y)
        class_counts = dict(zip(unique.tolist(), counts.tolist()))
        logger.warning(
            "Stratified split unavailable (sklearn=%s, class_counts=%s); using deterministic random split.",
            StratifiedShuffleSplit is not None,
            class_counts,
        )

    train_pairs = [pairs[i] for i in train_idx]
    val_pairs = [pairs[i] for i in val_idx]

    train_lesion = float(np.mean(y[train_idx])) if len(train_idx) else 0.0
    val_lesion = float(np.mean(y[val_idx])) if len(val_idx) else 0.0
    logger.info(
        "🧮 Dataset split (%s): Train=%d (%.1f%%), Validation=%d (%.1f%%)",
        split_mode,
        len(train_pairs),
        (len(train_pairs) / total) * 100.0,
        len(val_pairs),
        (len(val_pairs) / total) * 100.0,
    )
    if class_counts is not None:
        logger.info("🧩 Stratification groups: %s", class_counts)
    logger.info(
        "⚖️ Lesion prevalence: Train=%.2f%%, Validation=%.2f%%",
        train_lesion * 100.0,
        val_lesion * 100.0,
    )
    return train_pairs, val_pairs


def create_grouped_size_balanced_folds(
    pairs,
    smallest_component_sizes,
    n_splits: int = 3,
    case_size_bins=(100, 1000, 10000),
    random_state: int = 42,
):
    """Grouped CV folds stratified by source x tiny-lesion-aware case size."""
    total = len(pairs)
    if total == 0:
        return []
    n_splits = max(2, int(n_splits))
    source_labels = np.asarray([_path_source(img_p) for img_p, _ in pairs], dtype=object)
    size_groups = np.asarray(
        [_case_size_group(int(v), tuple(case_size_bins)) for v in np.asarray(smallest_component_sizes, dtype=np.int64)],
        dtype=object,
    )
    strat_labels = np.asarray([f"{src}|{grp}" for src, grp in zip(source_labels, size_groups)], dtype=object)
    groups = np.asarray([_subject_group_key(img_p) for img_p, _ in pairs], dtype=object)
    idx = np.arange(total, dtype=np.int64)

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    if StratifiedGroupKFold is not None:
        try:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            for train_idx, val_idx in splitter.split(idx, strat_labels, groups=groups):
                folds.append((np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)))
        except ValueError as e:
            logger.warning("StratifiedGroupKFold unavailable for this label distribution (%s); using greedy grouped fallback.", e)
            folds = []
    if not folds:
        # Fallback: greedy assignment by grouped label counts.
        rng = np.random.default_rng(random_state)
        group_to_indices: dict[str, list[int]] = {}
        for i, g in enumerate(groups.tolist()):
            group_to_indices.setdefault(g, []).append(int(i))
        group_items = list(group_to_indices.items())
        rng.shuffle(group_items)
        fold_groups: list[list[str]] = [[] for _ in range(n_splits)]
        fold_counts: list[dict[str, int]] = [dict() for _ in range(n_splits)]
        for group_name, group_idx in sorted(group_items, key=lambda kv: -len(kv[1])):
            local_counts: dict[str, int] = {}
            for i in group_idx:
                lbl = str(strat_labels[i])
                local_counts[lbl] = local_counts.get(lbl, 0) + 1
            best_fold = None
            best_score = None
            for f in range(n_splits):
                score = 0.0
                total_in_fold = sum(fold_counts[f].values())
                for lbl, cnt in local_counts.items():
                    score += (fold_counts[f].get(lbl, 0) + cnt) ** 2
                score += 0.1 * (total_in_fold + len(group_idx)) ** 2
                if best_score is None or score < best_score:
                    best_score = score
                    best_fold = f
            assert best_fold is not None
            fold_groups[best_fold].append(group_name)
            for lbl, cnt in local_counts.items():
                fold_counts[best_fold][lbl] = fold_counts[best_fold].get(lbl, 0) + cnt
        for f in range(n_splits):
            val_group_set = set(fold_groups[f])
            val_idx = np.asarray([i for i, g in enumerate(groups.tolist()) if g in val_group_set], dtype=np.int64)
            train_idx = np.asarray([i for i, g in enumerate(groups.tolist()) if g not in val_group_set], dtype=np.int64)
            folds.append((train_idx, val_idx))
    return folds


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
    def __init__(self, lesion_sizes: np.ndarray, cfg: DynamicTrainingConfig, source_labels: np.ndarray | None = None):
        self.cfg = cfg
        self.sizes = lesion_sizes.astype(np.int64)
        self.N = len(self.sizes)
        self.source_labels = None if source_labels is None else np.asarray(source_labels, dtype=object)
        self.source_mass_overrides: dict[str, float] | None = None
        self.base_case_weights = self._compute_base_case_weights()
        self.diff_multiplier = np.ones(self.N, dtype=np.float64) if self.N > 0 else np.empty(0, dtype=np.float64)
        self.weights = self._compose_weights()
        self.patch_quota = np.zeros(self.N, dtype=np.int64)

    def _apply_source_balance(self, weights: np.ndarray) -> np.ndarray:
        w = np.asarray(weights, dtype=np.float64).copy()
        if (
            not bool(getattr(self.cfg, "SOURCE_BALANCED_SAMPLING", True))
            or self.source_labels is None
            or len(self.source_labels) != self.N
            or self.N == 0
        ):
            w = np.clip(w, 1e-12, None)
            return w / w.sum()
        unique_sources = np.unique(self.source_labels)
        mass_by_source = {}
        if self.source_mass_overrides:
            remaining_sources = [src for src in unique_sources if src not in self.source_mass_overrides]
            override_mass = float(sum(max(0.0, float(v)) for v in self.source_mass_overrides.values()))
            override_mass = min(max(override_mass, 0.0), 1.0)
            remaining_mass = max(0.0, 1.0 - override_mass)
            default_mass = remaining_mass / max(len(remaining_sources), 1) if remaining_sources else 0.0
            for src in unique_sources:
                mass_by_source[src] = max(0.0, float(self.source_mass_overrides.get(src, default_mass)))
            total_mass = sum(mass_by_source.values())
            if total_mass > 0:
                for src in list(mass_by_source.keys()):
                    mass_by_source[src] = mass_by_source[src] / total_mass
        else:
            per_source_mass = 1.0 / max(len(unique_sources), 1)
            mass_by_source = {src: per_source_mass for src in unique_sources}
        out = np.zeros_like(w)
        for src in unique_sources:
            idx = np.where(self.source_labels == src)[0]
            if idx.size == 0:
                continue
            local = np.clip(w[idx], 1e-12, None)
            out[idx] = float(mass_by_source.get(src, 0.0)) * (local / local.sum())
        out = np.clip(out, 1e-12, None)
        return out / out.sum()

    def _compute_base_case_weights(self):
        if not self.cfg.SIZE_AWARE_ENABLED or self.N == 0:
            return np.ones(self.N, dtype=np.float64)
        if self.cfg.SIZE_AWARE_MODE == "inverse":
            safe_sizes = np.where(self.sizes > 0, self.sizes, 1)
            w = 1.0 / np.power(safe_sizes + min(float(self.cfg.INV_VOL_EPS), 100.0), self.cfg.INV_VOL_ALPHA)
            w[self.sizes <= 0] *= 0.05
            w = np.clip(w, 1e-12, None)
            return w
        edges = np.asarray(getattr(self.cfg, "CASE_SIZE_BINS", (100, 1000, 10000)), dtype=np.int64)
        group_weights = np.asarray(
            getattr(self.cfg, "CASE_SIZE_GROUP_PROBS", (0.45, 0.25, 0.15, 0.10)),
            dtype=np.float64,
        )
        if group_weights.size != (len(edges) + 1):
            fallback = np.asarray(getattr(self.cfg, "SIZE_BUCKET_PROBS", (0.45, 0.25, 0.15, 0.10, 0.05)), dtype=np.float64)
            if fallback.size >= (len(edges) + 1):
                group_weights = fallback[: len(edges) + 1]
            else:
                group_weights = np.asarray([1.0, 0.70, 0.35, 0.18], dtype=np.float64)
        bins = np.digitize(self.sizes, edges, right=False)
        w = np.zeros(self.N, dtype=np.float64)
        for b in range(len(group_weights)):
            idx = np.where(bins == b)[0]
            if idx.size == 0:
                continue
            w[idx] = group_weights[b] / idx.size
        none_idx = np.where(self.sizes <= 0)[0]
        if none_idx.size:
            none_weight = float(getattr(self.cfg, "CASE_NONE_PROB", 0.05))
            if none_weight <= 0.0:
                none_weight = 0.02
            w[none_idx] = none_weight / none_idx.size
        w = np.clip(w, 1e-12, None)
        return w

    def _compose_weights(self):
        if self.N == 0:
            return np.empty(0, dtype=np.float64)
        raw = np.clip(self.base_case_weights * self.diff_multiplier, 1e-12, None)
        return self._apply_source_balance(raw)

    def refresh_weights(self):
        self.base_case_weights = self._compute_base_case_weights()
        if self.diff_multiplier.shape != self.base_case_weights.shape:
            self.diff_multiplier = np.ones_like(self.base_case_weights, dtype=np.float64)
        self.weights = self._compose_weights()

    def set_source_mass_overrides(self, overrides: dict[str, float] | None):
        self.source_mass_overrides = None if not overrides else {
            str(k): float(v) for k, v in overrides.items() if float(v) > 0.0
        }
        self.weights = self._compose_weights()

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
        badness /= np.clip(np.mean(badness), 1e-6, None)
        self.diff_multiplier = (
            self.cfg.DIFF_EMA_LAMBDA * self.diff_multiplier
            + (1.0 - self.cfg.DIFF_EMA_LAMBDA) * badness
        )
        self.weights = self._compose_weights().astype(np.float64)

def bin_index(v, edges):
    import numpy as _np
    return int(_np.digitize([v], _np.asarray(edges, dtype=_np.int64), right=False)[0])

def sample_patch_center(
    mask,
    patch_size,
    p_fg,
    rng,
    component_records: list[dict[str, object]] | None = None,
    case_size_bins: tuple[int, ...] = (100, 1000, 10000),
    use_component_aware_sampling: bool = False,
    use_tiny_component_centering: bool = False,
    tiny_component_center_prob: float = 0.95,
    small_component_center_prob: float = 0.85,
    tiny_component_max_jitter: int = 2,
    small_component_max_jitter: int = 4,
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
    component_records = component_records or []

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
        chosen_component = None
        comp_candidates = component_records
        if component_records and (use_component_aware_sampling or use_tiny_component_centering):
            if use_hemi:
                axis = int(hemisphere_axis)
                mid = dims[axis] // 2
                if hemisphere_side == 0:
                    comp_candidates = [c for c in component_records if int(c["centroid"][axis]) < mid]
                else:
                    comp_candidates = [c for c in component_records if int(c["centroid"][axis]) >= mid]
                if not comp_candidates:
                    comp_candidates = component_records

            if use_tiny_component_centering:
                small_limit = int(case_size_bins[1]) if len(case_size_bins) > 1 else int(case_size_bins[0])
                preferred = [c for c in comp_candidates if int(c["size"]) < small_limit]
                if preferred:
                    comp_weights = np.asarray(
                        [
                            8.0 / np.sqrt(max(1.0, float(c["size"])))
                            if int(c["size"]) < int(case_size_bins[0])
                            else 3.0 / np.sqrt(max(1.0, float(c["size"])))
                            for c in preferred
                        ],
                        dtype=np.float64,
                    )
                    comp_weights /= np.clip(comp_weights.sum(), 1e-12, None)
                    chosen_component = preferred[int(rng.choice(len(preferred), p=comp_weights))]

            if chosen_component is None and use_component_aware_sampling and comp_candidates:
                comp_weights = []
                for comp in comp_candidates:
                    size = max(1, int(comp["size"]))
                    weight = 1.0 / np.sqrt(float(size))
                    if size < int(case_size_bins[0]):
                        weight *= 6.0
                    elif size < int(case_size_bins[1]):
                        weight *= 3.0
                    elif size < int(case_size_bins[2]):
                        weight *= 1.5
                    comp_weights.append(weight)
                comp_weights = _np.asarray(comp_weights, dtype=_np.float64)
                comp_weights /= _np.clip(comp_weights.sum(), 1e-12, None)
                chosen_component = comp_candidates[int(rng.choice(len(comp_candidates), p=comp_weights))]

        if chosen_component is None:
            fg_mins = fg.min(axis=0)
            fg_maxs = fg.max(axis=0)
            fg_center = _np.round((fg_mins + fg_maxs) / 2.0).astype(_np.int64)
            fg_span = _np.maximum(fg_maxs - fg_mins + 1, 1)
            bbox_jitter = _np.minimum(_np.maximum(fg_span // 6, 2), 16)
            if rng.random() < 0.7:
                cz, cy, cx = fg_center
                jitter = _np.asarray(
                    [
                        rng.integers(-int(bbox_jitter[0]), int(bbox_jitter[0]) + 1),
                        rng.integers(-int(bbox_jitter[1]), int(bbox_jitter[1]) + 1),
                        rng.integers(-int(bbox_jitter[2]), int(bbox_jitter[2]) + 1),
                    ],
                    dtype=_np.int64,
                )
            else:
                cz, cy, cx = fg[rng.integers(len(fg))]
                jitter = rng.integers(low=-4, high=5, size=3)
        else:
            fg_mins = chosen_component["mins"]
            fg_maxs = chosen_component["maxs"]
            fg_center = chosen_component["centroid"]
            fg_span = _np.maximum(fg_maxs - fg_mins + 1, 1)
            comp_size = int(chosen_component["size"])
            if comp_size < int(case_size_bins[0]):
                bbox_jitter = _np.full(3, max(0, int(tiny_component_max_jitter)), dtype=_np.int64)
                center_prob = float(tiny_component_center_prob)
            elif comp_size < int(case_size_bins[1]):
                bbox_jitter = _np.full(3, max(0, int(small_component_max_jitter)), dtype=_np.int64)
                center_prob = float(small_component_center_prob)
            else:
                bbox_jitter = _np.minimum(_np.maximum(fg_span // 6, 2), 16)
                center_prob = 0.80
            if rng.random() < center_prob:
                cz, cy, cx = fg_center
                jitter = _np.asarray(
                    [
                        rng.integers(-int(bbox_jitter[0]), int(bbox_jitter[0]) + 1),
                        rng.integers(-int(bbox_jitter[1]), int(bbox_jitter[1]) + 1),
                        rng.integers(-int(bbox_jitter[2]), int(bbox_jitter[2]) + 1),
                    ],
                    dtype=_np.int64,
                )
            else:
                coords = chosen_component["coords"]
                cz, cy, cx = coords[rng.integers(len(coords))]
                jitter = rng.integers(low=-3, high=4, size=3)
    elif bg.size > 0:
        cz, cy, cx = bg[rng.integers(len(bg))]
        jitter = rng.integers(low=-4, high=5, size=3)
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


class PrimaryOutputMetricAliasCallback(tf.keras.callbacks.Callback):
    """Expose primary-output metric names even when auxiliary heads are enabled."""

    @staticmethod
    def _alias(logs):
        if not logs:
            return
        if "dice_coefficient" not in logs and "probs_dice_coefficient" in logs:
            logs["dice_coefficient"] = logs["probs_dice_coefficient"]
        if "safe_binary_iou" not in logs and "probs_safe_binary_iou" in logs:
            logs["safe_binary_iou"] = logs["probs_safe_binary_iou"]

    def on_train_batch_end(self, batch, logs=None):
        self._alias(logs)

    def on_epoch_end(self, epoch, logs=None):
        self._alias(logs)


class SamplingPolicyController(tf.keras.callbacks.Callback):
    """Drive curriculum and late ATLAS-focused fine-tuning for the case sampler."""

    def __init__(self, sampler: SizeAwareCaseSampler | None, cfg: DynamicTrainingConfig, train_source_labels: np.ndarray | None):
        super().__init__()
        self.sampler = sampler
        self.cfg = cfg
        self.train_source_labels = np.asarray(train_source_labels, dtype=object) if train_source_labels is not None else np.asarray([], dtype=object)
        self.out_jsonl = Path(cfg.CALLBACKS_DIR) / "sampling_schedule.jsonl"
        self._last_signature = None
        self.base_case_group_probs = tuple(float(v) for v in getattr(cfg, "CASE_SIZE_GROUP_PROBS", (0.45, 0.25, 0.15, 0.10)))
        self.base_patch_fg_probs = tuple(float(v) for v in getattr(cfg, "PATCH_FG_PROB_BY_BIN", (0.98, 0.95, 0.85, 0.70)))
        self.base_case_none_prob = float(getattr(cfg, "CASE_NONE_PROB", 0.05))

    @staticmethod
    def _align_tuple(values, length: int, fallback):
        arr = np.asarray(values if values is not None else fallback, dtype=np.float64).reshape(-1)
        fb = np.asarray(fallback, dtype=np.float64).reshape(-1)
        if arr.size == length:
            return arr
        if arr.size == 0:
            arr = fb.copy()
        if arr.size > length:
            return arr[:length]
        fill = arr[-1] if arr.size > 0 else (fb[-1] if fb.size > 0 else 0.0)
        return np.pad(arr, (0, length - arr.size), constant_values=float(fill))

    def _interp(self, start, end, progress: float):
        start_arr = np.asarray(start, dtype=np.float64)
        end_arr = np.asarray(end, dtype=np.float64)
        return ((1.0 - progress) * start_arr) + (progress * end_arr)

    def _resolve_target_sources(self):
        prefixes = tuple(str(x) for x in getattr(self.cfg, "ATLAS_FINE_TUNE_SOURCE_PREFIXES", ("ATLAS",)))
        if self.train_source_labels.size == 0:
            return []
        unique_sources = sorted(set(self.train_source_labels.tolist()))
        matched = [
            src for src in unique_sources
            if any(src.startswith(pref) or pref in src for pref in prefixes)
        ]
        return matched

    def _compute_policy(self, epoch: int):
        target_case_probs = self._align_tuple(
            self.base_case_group_probs,
            len(getattr(self.cfg, "CASE_SIZE_BINS", (100, 1000, 10000))) + 1,
            self.base_case_group_probs,
        )
        target_patch_fg = self._align_tuple(
            self.base_patch_fg_probs,
            len(target_case_probs),
            self.base_patch_fg_probs,
        )
        none_prob = float(self.base_case_none_prob)
        phase = "steady"

        if bool(getattr(self.cfg, "USE_SIZE_CURRICULUM", False)) and int(getattr(self.cfg, "CURRICULUM_EPOCHS", 0)) > 0:
            start_case_probs = self._align_tuple(
                getattr(self.cfg, "CURRICULUM_START_CASE_GROUP_PROBS", target_case_probs),
                len(target_case_probs),
                target_case_probs,
            )
            start_patch_fg = self._align_tuple(
                getattr(self.cfg, "CURRICULUM_START_PATCH_FG_PROB_BY_BIN", target_patch_fg),
                len(target_patch_fg),
                target_patch_fg,
            )
            start_none_prob = float(getattr(self.cfg, "CURRICULUM_START_CASE_NONE_PROB", none_prob))
            denom = max(1, int(getattr(self.cfg, "CURRICULUM_EPOCHS", 1)) - 1)
            progress = min(1.0, float(epoch) / float(denom))
            case_probs = self._interp(start_case_probs, target_case_probs, progress)
            patch_fg = self._interp(start_patch_fg, target_patch_fg, progress)
            none_prob = float((1.0 - progress) * start_none_prob + progress * none_prob)
            phase = "curriculum" if progress < 1.0 else "steady"
        else:
            case_probs = target_case_probs
            patch_fg = target_patch_fg

        source_overrides = None
        if bool(getattr(self.cfg, "USE_ATLAS_FINE_TUNE", False)) and epoch >= int(getattr(self.cfg, "ATLAS_FINE_TUNE_START_EPOCH", 1)):
            target_sources = self._resolve_target_sources()
            if target_sources:
                atlas_mass = float(np.clip(getattr(self.cfg, "ATLAS_FINE_TUNE_SOURCE_MASS", 0.70), 0.0, 1.0))
                per_source_mass = atlas_mass / max(len(target_sources), 1)
                source_overrides = {src: per_source_mass for src in target_sources}
                case_probs = self._align_tuple(
                    getattr(self.cfg, "ATLAS_FINE_TUNE_CASE_GROUP_PROBS", case_probs),
                    len(case_probs),
                    case_probs,
                )
                patch_fg = self._align_tuple(
                    getattr(self.cfg, "ATLAS_FINE_TUNE_PATCH_FG_PROB_BY_BIN", patch_fg),
                    len(patch_fg),
                    patch_fg,
                )
                none_prob = float(getattr(self.cfg, "ATLAS_FINE_TUNE_CASE_NONE_PROB", none_prob))
                phase = "atlas_finetune"

        return {
            "phase": phase,
            "case_group_probs": tuple(float(v) for v in case_probs.tolist()),
            "patch_fg_probs": tuple(float(v) for v in patch_fg.tolist()),
            "case_none_prob": float(none_prob),
            "source_overrides": source_overrides,
        }

    def _weight_summary(self):
        if self.sampler is None or self.sampler.N == 0:
            return {}, {}
        source_mass = {}
        if self.sampler.source_labels is not None:
            for src in np.unique(self.sampler.source_labels):
                source_mass[str(src)] = float(np.sum(self.sampler.weights[self.sampler.source_labels == src]))
        size_mass = {}
        edges = np.asarray(getattr(self.cfg, "CASE_SIZE_BINS", (100, 1000, 10000)), dtype=np.int64)
        labels = ["1_99", "100_999", "1000_9999", "10000_plus"]
        bins = np.digitize(self.sampler.sizes, edges, right=False)
        for idx, label_name in enumerate(labels[: int(np.max(bins)) + 1 if bins.size else len(labels)]):
            size_mass[label_name] = float(np.sum(self.sampler.weights[bins == idx]))
        return source_mass, size_mass

    def on_train_begin(self, logs=None):
        self.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_begin(self, epoch, logs=None):
        if self.sampler is None:
            return
        policy = self._compute_policy(int(epoch))
        self.cfg.CASE_SIZE_GROUP_PROBS = tuple(policy["case_group_probs"])
        self.cfg.PATCH_FG_PROB_BY_BIN = tuple(policy["patch_fg_probs"])
        self.cfg.CASE_NONE_PROB = float(policy["case_none_prob"])
        self.sampler.set_source_mass_overrides(policy["source_overrides"])
        self.sampler.refresh_weights()
        source_mass, size_mass = self._weight_summary()
        record = {
            "epoch": int(epoch),
            "phase": str(policy["phase"]),
            "case_group_probs": list(self.cfg.CASE_SIZE_GROUP_PROBS),
            "patch_fg_probs": list(self.cfg.PATCH_FG_PROB_BY_BIN),
            "case_none_prob": float(self.cfg.CASE_NONE_PROB),
            "source_overrides": policy["source_overrides"] or {},
            "effective_source_mass": source_mass,
            "effective_case_group_mass": size_mass,
        }
        with open(self.out_jsonl, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        signature = (
            record["phase"],
            tuple(round(v, 6) for v in record["case_group_probs"]),
            tuple(round(v, 6) for v in record["patch_fg_probs"]),
            round(record["case_none_prob"], 6),
            tuple(sorted((record["source_overrides"] or {}).items())),
        )
        if signature != self._last_signature:
            logger.info(
                "Sampling policy @epoch %d: phase=%s case_group_probs=%s patch_fg_probs=%s case_none_prob=%.3f source_overrides=%s",
                int(epoch),
                record["phase"],
                tuple(round(v, 3) for v in record["case_group_probs"]),
                tuple(round(v, 3) for v in record["patch_fg_probs"]),
                float(record["case_none_prob"]),
                record["source_overrides"] or {},
            )
            self._last_signature = signature

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
            xb = _make_input_channels(x, self.cfg)[np.newaxis, ...]
            pred = _binary_output_from_prediction(self.model.predict(xb, verbose=0))[0, ..., 0]
            dice[idx] = dice_soft_np(y, pred)
        self.sampler.diff_aware_update(dice)


def build_small_lesion_bank(train_pairs, cfg: DynamicTrainingConfig, target_shape=None):
    """Build a train-only donor bank of tiny/small lesion crops for insertion augmentation."""
    bank = []
    max_voxels = int(getattr(cfg, "LESION_INSERTION_MAX_COMPONENT_VOXELS", 1000))
    max_items = int(getattr(cfg, "LESION_INSERTION_MAX_BANK_COMPONENTS", 512))
    margin = 4
    for img_p, msk_p in train_pairs:
        if len(bank) >= max_items:
            break
        img = _load_and_preprocess_image(str(img_p), target_shape)
        msk = _load_and_preprocess_mask(str(msk_p), target_shape)
        for comp in _build_component_records(msk):
            size = int(comp["size"])
            if size <= 0 or size > max_voxels:
                continue
            mins = np.maximum(np.asarray(comp["mins"], dtype=np.int64) - margin, 0)
            maxs = np.minimum(np.asarray(comp["maxs"], dtype=np.int64) + margin + 1, np.asarray(msk.shape))
            slc = tuple(slice(int(lo), int(hi)) for lo, hi in zip(mins, maxs))
            crop_img = img[slc].astype(np.float32, copy=True)
            crop_mask = (msk[slc] > 0.5).astype(np.uint8, copy=True)
            if not np.any(crop_mask):
                continue
            crop_context = binary_dilation(crop_mask, structure=generate_binary_structure(3, 1), iterations=1).astype(bool)
            donor_vals = crop_img[crop_context]
            donor_mean = float(np.mean(donor_vals)) if donor_vals.size else float(np.mean(crop_img))
            donor_std = float(np.std(donor_vals)) if donor_vals.size else float(np.std(crop_img))
            bank.append(
                {
                    "image": crop_img,
                    "mask": crop_mask,
                    "size": size,
                    "donor_mean": donor_mean,
                    "donor_std": max(donor_std, 1e-6),
                    "source": _path_source(img_p),
                    "case_id": _path_case_id(img_p),
                }
            )
            if len(bank) >= max_items:
                break
    logger.info("Built lesion insertion bank with %d donor components", len(bank))
    return bank


def apply_lesion_insertion(image: np.ndarray, mask: np.ndarray, lesion_bank, cfg: DynamicTrainingConfig, rng):
    if not lesion_bank or float(getattr(cfg, "LESION_INSERTION_PROB", 0.0)) <= 0.0:
        return image, mask
    if rng.random() > float(getattr(cfg, "LESION_INSERTION_PROB", 0.0)):
        return image, mask
    current_smallest = _smallest_positive_component(mask)
    if 0 < current_smallest < int(getattr(cfg, "CASE_SIZE_BINS", (100, 1000, 10000))[0]):
        return image, mask

    weights = np.asarray(
        [4.0 if int(item["size"]) < int(cfg.CASE_SIZE_BINS[0]) else 1.5 for item in lesion_bank],
        dtype=np.float64,
    )
    weights /= np.clip(weights.sum(), 1e-12, None)
    donor = lesion_bank[int(rng.choice(len(lesion_bank), p=weights))]
    donor_img = donor["image"]
    donor_mask = donor["mask"].astype(bool)
    donor_shape = donor_mask.shape
    if any(d > s for d, s in zip(donor_shape, image.shape)):
        return image, mask

    brain_mask = compute_brain_mask(image).astype(bool)
    valid_starts = [max(0, int(image.shape[i] - donor_shape[i])) for i in range(3)]
    if any(v < 0 for v in valid_starts):
        return image, mask

    donor_context = binary_dilation(donor_mask, structure=generate_binary_structure(3, 1), iterations=1).astype(bool)
    for _ in range(24):
        start = [int(rng.integers(0, v + 1)) if v > 0 else 0 for v in valid_starts]
        slc = tuple(slice(start[i], start[i] + donor_shape[i]) for i in range(3))
        existing = mask[slc] > 0.5
        if np.any(existing & donor_mask):
            continue
        local_brain = brain_mask[slc]
        if np.mean(local_brain[donor_mask]) < 0.95:
            continue
        dest_img = image[slc].copy()
        dest_vals = dest_img[donor_context & local_brain]
        if dest_vals.size == 0:
            continue
        dest_mean = float(np.mean(dest_vals))
        dest_std = float(np.std(dest_vals))
        donor_scaled = (donor_img - float(donor["donor_mean"])) / float(donor["donor_std"])
        donor_scaled = donor_scaled * max(dest_std, 1e-6) + dest_mean
        blend_mask = donor_context & local_brain
        alpha = 0.65
        dest_img[blend_mask] = alpha * donor_scaled[blend_mask] + (1.0 - alpha) * dest_img[blend_mask]
        image = image.copy()
        mask = mask.copy()
        image[slc] = dest_img
        mask[slc] = np.maximum(mask[slc], donor_mask.astype(mask.dtype))
        return image.astype(np.float32, copy=False), (mask > 0.5).astype(np.float32, copy=False)
    return image, mask


def _full_volume_target_shape(cfg: DynamicTrainingConfig) -> tuple[int, int, int] | None:
    """Return case-loading shape used for whole-volume inference/validation."""
    if bool(getattr(cfg, "LOAD_FULL_IMAGE_FOR_PATCHING", True)):
        raw_full_shape = getattr(cfg, "FULL_RES_TARGET_SHAPE", None)
        if raw_full_shape is None:
            return None
        return tuple(int(v) for v in raw_full_shape)
    return tuple(int(v) for v in cfg.INPUT_SHAPE[:-1])


def component_recall_summary(y_true: np.ndarray, pred_mask: np.ndarray, case_size_bins=(100, 1000, 10000)) -> dict[str, dict[str, float]]:
    lbl, n = label((y_true > 0.5).astype(np.uint8), structure=generate_binary_structure(3, 1))
    out: dict[str, dict[str, float]] = {}
    if n <= 0:
        return out
    counts = np.bincount(lbl.ravel())
    for comp_id in range(1, n + 1):
        size = int(counts[comp_id]) if comp_id < counts.size else 0
        grp = _case_size_group(size, tuple(case_size_bins))
        hit = float(np.any(pred_mask[lbl == comp_id] > 0))
        bucket = out.setdefault(grp, {"components": 0.0, "hits": 0.0})
        bucket["components"] += 1.0
        bucket["hits"] += hit
    for grp, vals in out.items():
        vals["recall"] = float(vals["hits"] / max(vals["components"], 1.0))
    return out


def extract_topk_centers(
    center_map: np.ndarray,
    topk: int,
    min_confidence: float | None = 0.10,
    nms_radius: int = 6,
) -> list[dict[str, object]]:
    work = np.asarray(center_map, dtype=np.float32).copy()
    work = np.nan_to_num(work, nan=0.0, posinf=0.0, neginf=0.0)
    candidates: list[dict[str, object]] = []
    topk = max(1, int(topk))
    nms_radius = max(1, int(nms_radius))
    min_conf = None if min_confidence is None else float(min_confidence)
    for _ in range(topk):
        flat_idx = int(np.argmax(work))
        score = float(work.flat[flat_idx])
        if min_conf is not None and score < min_conf:
            break
        coord = tuple(int(v) for v in np.unravel_index(flat_idx, work.shape))
        candidates.append({"coord": coord, "score": score})
        z, y, x = coord
        z0, z1 = max(0, z - nms_radius), min(work.shape[0], z + nms_radius + 1)
        y0, y1 = max(0, y - nms_radius), min(work.shape[1], y + nms_radius + 1)
        x0, x1 = max(0, x - nms_radius), min(work.shape[2], x + nms_radius + 1)
        work[z0:z1, y0:y1, x0:x1] = 0.0
    return candidates


def proposal_recall_summary(
    y_true: np.ndarray,
    center_map: np.ndarray,
    topk_values=(1, 3, 5, 10),
    case_size_bins=(100, 1000, 10000),
    match_radius: float = 6.0,
    min_confidence: float = 0.10,
    nms_radius: int = 6,
    size_map: np.ndarray | None = None,
) -> dict[str, object]:
    gt_records = extract_component_seed_records(y_true, case_size_bins=case_size_bins)
    if not gt_records:
        return {
            "candidates": [],
            "topk": {},
            "best_candidate_score": 0.0,
            "size_seed_acc": None,
            "size_seed_acc_by_group": {},
        }
    topk_values = tuple(sorted(set(max(1, int(v)) for v in topk_values)))
    candidates = extract_topk_centers(
        center_map,
        topk=max(topk_values),
        min_confidence=min_confidence,
        nms_radius=nms_radius,
    )
    out: dict[str, object] = {
        "candidates": candidates,
        "best_candidate_score": 0.0,
        "candidates_above_threshold": 0,
        "topk": {},
        "size_seed_acc": None,
        "size_seed_acc_by_group": {},
    }
    raw_candidates = extract_topk_centers(
        center_map,
        topk=max(topk_values),
        min_confidence=None,
        nms_radius=nms_radius,
    )
    out["best_candidate_score"] = float(raw_candidates[0]["score"]) if raw_candidates else 0.0
    out["candidates_above_threshold"] = int(len(candidates))
    for k in topk_values:
        cand_subset = raw_candidates[:k]
        hits = 0.0
        best_distances = []
        by_group: dict[str, dict[str, float]] = {}
        for rec in gt_records:
            grp = str(rec["group"])
            bucket = by_group.setdefault(grp, {"components": 0.0, "hits": 0.0})
            bucket["components"] += 1.0
            best_dist = float("inf")
            for cand in cand_subset:
                dist = float(np.linalg.norm(np.subtract(cand["coord"], rec["coord"])))
                if dist < best_dist:
                    best_dist = dist
            best_distances.append(best_dist)
            hit = float(best_dist <= float(match_radius))
            hits += hit
            bucket["hits"] += hit
        for grp, vals in by_group.items():
            vals["recall"] = float(vals["hits"] / max(vals["components"], 1.0))
        finite_best = [d for d in best_distances if np.isfinite(d)]
        out["topk"][k] = {
            "components": float(len(gt_records)),
            "hits": float(hits),
            "recall": float(hits / max(len(gt_records), 1)),
            "mean_best_dist": float(np.mean(finite_best)) if finite_best else float("inf"),
            "by_group": by_group,
        }
    if size_map is not None:
        size_map = np.asarray(size_map)
        size_hits = []
        size_by_group: dict[str, list[float]] = {}
        for rec in gt_records:
            z, y, x = rec["coord"]
            pred_cls = int(np.argmax(size_map[z, y, x]))
            hit = float(pred_cls == int(rec["class_index"]))
            size_hits.append(hit)
            size_by_group.setdefault(str(rec["group"]), []).append(hit)
        out["size_seed_acc"] = float(np.mean(size_hits)) if size_hits else None
        out["size_seed_acc_by_group"] = {
            grp: float(np.mean(vals)) for grp, vals in sorted(size_by_group.items()) if vals
        }
    return out


class WholeBrainValidationCallback(tf.keras.callbacks.Callback):
    """
    Compute validation Dice on full brain volumes by stitching patch predictions.

    This avoids center-crop validation bias and reports whole-volume metrics.
    """

    def __init__(self, val_pairs, cfg: DynamicTrainingConfig):
        super().__init__()
        self.val_pairs = list(val_pairs)
        self.cfg = cfg
        self.volume_target_shape = _full_volume_target_shape(cfg)
        self.patch_size = tuple(cfg.PATCH_SIZE or cfg.INPUT_SHAPE[:-1])
        self.overlap = float(cfg.GAUSSIAN_TILE_OVERLAP)
        self.sigma = float(cfg.GAUSSIAN_TILE_SIGMA)
        self.tta = bool(cfg.WHOLE_BRAIN_VAL_TTA)
        self.every_n = max(1, int(cfg.WHOLE_BRAIN_VAL_EVERY_N_EPOCHS))
        self.threshold = float(cfg.DECISION_THRESHOLD)
        self.max_cases = None if cfg.WHOLE_BRAIN_VAL_MAX_CASES in (None, 0) else int(cfg.WHOLE_BRAIN_VAL_MAX_CASES)
        self.threshold_sweep = tuple(float(t) for t in getattr(cfg, "VAL_THRESHOLD_SWEEP", (self.threshold,)))
        self.top_k = max(1, int(getattr(cfg, "VAL_DIAGNOSTICS_TOP_K", 5)))
        self.center_topk_values = tuple(int(v) for v in getattr(cfg, "CENTER_TOPK_VALUES", (1, 3, 5, 10)))
        self.center_match_radius = float(getattr(cfg, "CENTER_MATCH_RADIUS", 6.0))
        self.center_nms_radius = int(getattr(cfg, "CENTER_NMS_RADIUS", 6))
        self.center_min_confidence = float(getattr(cfg, "CENTER_MIN_CONFIDENCE", 0.10))
        self.use_center_head = bool(getattr(cfg, "USE_CENTER_HEATMAP_HEAD", False))
        self.use_size_head = bool(getattr(cfg, "USE_SIZE_HEAD", False))
        self.diagnostics_enabled = bool(getattr(cfg, "DIAGNOSTICS_ENABLED", True))
        self.out_dir = Path(cfg.CALLBACKS_DIR)
        self.summary_jsonl = self.out_dir / "whole_val_summary.jsonl"
        self._eps = 1e-6

    def _iter_pairs(self):
        if self.max_cases is None:
            return self.val_pairs
        return self.val_pairs[: max(1, self.max_cases)]

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        if (epoch + 1) % self.every_n != 0:
            return

        eval_pairs = self._iter_pairs()
        if not eval_pairs:
            logger.warning("Whole-brain validation skipped: no validation pairs available.")
            return

        case_soft = []
        case_hard = []
        case_hard_brainmask = []
        case_hard_raw = []
        source_soft: dict[str, list[float]] = {}
        source_hard: dict[str, list[float]] = {}
        source_hard_brainmask: dict[str, list[float]] = {}
        source_hard_raw: dict[str, list[float]] = {}
        lesion_recall_bins: dict[str, dict[str, float]] = {}
        threshold_case_scores: dict[float, list[float]] = {t: [] for t in self.threshold_sweep}
        threshold_case_scores_brainmask: dict[float, list[float]] = {t: [] for t in self.threshold_sweep}
        threshold_case_scores_raw: dict[float, list[float]] = {t: [] for t in self.threshold_sweep}
        case_rows: list[dict[str, object]] = []
        center_recall_macro: dict[int, list[float]] = {k: [] for k in self.center_topk_values}
        center_recall_by_group: dict[int, dict[str, dict[str, float]]] = {k: {} for k in self.center_topk_values}
        source_center_recall: dict[int, dict[str, list[float]]] = {k: {} for k in self.center_topk_values}
        size_seed_acc_all: list[float] = []
        source_size_seed_acc: dict[str, list[float]] = {}
        pred_soft_true_ratio = []
        pred_hard_true_ratio = []
        pred_hard_brainmask_true_ratio = []
        pred_hard_raw_true_ratio = []
        brainmask_hard_delta = []
        postproc_hard_delta = []
        inter_soft = pred_soft = true_sum = 0.0
        t0 = time.time()

        for i, (img_p, msk_p) in enumerate(eval_pairs, start=1):
            x = _load_and_preprocess_image(str(img_p), self.volume_target_shape)
            y = _load_and_preprocess_mask(str(msk_p), self.volume_target_shape).astype(np.float32)
            requested_outputs = ["probs"]
            if self.use_center_head:
                requested_outputs.append("center_heatmap")
            if self.use_size_head:
                requested_outputs.append("size_head")
            pred_maps = gaussian_tta_predict_outputs(
                self.model,
                x,
                patch_size=self.patch_size,
                overlap=self.overlap,
                sigma=self.sigma,
                tta=self.tta,
                output_names=tuple(requested_outputs),
            )
            probs = pred_maps["probs"]
            probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
            probs = np.clip(probs, 0.0, 1.0)
            brain_mask = compute_brain_mask(x)
            pred_hard_raw = (probs >= self.threshold).astype(np.float32, copy=False)
            pred_hard_brainmask = apply_postprocessing(
                probs,
                threshold=self.threshold,
                min_size=0,
                closing=0,
                use_component_scoring=False,
                brain_mask=brain_mask,
                clamp=self.cfg.OTSU_CLAMP,
                min_prob=self.cfg.OTSU_MIN_PROB,
            )
            pred_hard = apply_postprocessing(
                probs,
                threshold=self.threshold,
                min_size=0,
                closing=0,
                use_component_scoring=bool(getattr(self.cfg, "USE_COMPONENT_SCORING_POSTPROC", False)),
                brain_mask=brain_mask if bool(getattr(self.cfg, "USE_BRAINMASK_POSTPROC", False)) else None,
                clamp=self.cfg.OTSU_CLAMP,
                min_prob=self.cfg.OTSU_MIN_PROB,
            )
            comp_recall = component_recall_summary(y, pred_hard, case_size_bins=tuple(getattr(self.cfg, "CASE_SIZE_BINS", (100, 1000, 10000))))

            inter = float(np.sum(y * probs))
            pred = float(np.sum(probs))
            true = float(np.sum(y))
            inter_soft += inter
            pred_soft += pred
            true_sum += true

            case_soft_i = float((2.0 * inter + self._eps) / (pred + true + self._eps))
            case_soft.append(case_soft_i)
            inter_hard_raw = float(np.sum(y * pred_hard_raw))
            pred_hard_raw_sum = float(np.sum(pred_hard_raw))
            case_hard_raw_i = float((2.0 * inter_hard_raw + self._eps) / (pred_hard_raw_sum + true + self._eps))
            case_hard_raw.append(case_hard_raw_i)
            inter_hard_brainmask = float(np.sum(y * pred_hard_brainmask))
            pred_hard_brainmask_sum = float(np.sum(pred_hard_brainmask))
            case_hard_brainmask_i = float(
                (2.0 * inter_hard_brainmask + self._eps) / (pred_hard_brainmask_sum + true + self._eps)
            )
            case_hard_brainmask.append(case_hard_brainmask_i)
            inter_hard = float(np.sum(y * pred_hard))
            pred_hard_sum = float(np.sum(pred_hard))
            case_hard_i = float((2.0 * inter_hard + self._eps) / (pred_hard_sum + true + self._eps))
            case_hard.append(case_hard_i)
            source_ratio_denom = max(true, 1.0)
            pred_soft_true_ratio.append(float(pred / source_ratio_denom))
            pred_hard_raw_true_ratio.append(float(pred_hard_raw_sum / source_ratio_denom))
            pred_hard_brainmask_true_ratio.append(float(pred_hard_brainmask_sum / source_ratio_denom))
            pred_hard_true_ratio.append(float(pred_hard_sum / source_ratio_denom))
            brainmask_hard_delta.append(float(case_hard_brainmask_i - case_hard_raw_i))
            postproc_hard_delta.append(float(case_hard_i - case_hard_raw_i))

            src_name = _path_source(img_p)
            source_soft.setdefault(src_name, []).append(case_soft_i)
            source_hard.setdefault(src_name, []).append(case_hard_i)
            source_hard_brainmask.setdefault(src_name, []).append(case_hard_brainmask_i)
            source_hard_raw.setdefault(src_name, []).append(case_hard_raw_i)
            row = {
                "source": src_name,
                "case_id": _path_case_id(img_p),
                "soft_dice": case_soft_i,
                "hard_dice": case_hard_i,
                "hard_dice_brainmask": case_hard_brainmask_i,
                "hard_dice_raw": case_hard_raw_i,
                "true_voxels": int(true),
                "pred_soft_voxels": float(pred),
                "pred_hard_voxels": float(pred_hard_sum),
                "pred_hard_brainmask_voxels": float(pred_hard_brainmask_sum),
                "pred_hard_raw_voxels": float(pred_hard_raw_sum),
                "brainmask_hard_delta": float(case_hard_brainmask_i - case_hard_raw_i),
                "postproc_hard_delta": float(case_hard_i - case_hard_raw_i),
                "pred_soft_to_true_ratio": float(pred / source_ratio_denom),
                "pred_hard_to_true_ratio": float(pred_hard_sum / source_ratio_denom),
                "pred_hard_brainmask_to_true_ratio": float(pred_hard_brainmask_sum / source_ratio_denom),
                "pred_hard_raw_to_true_ratio": float(pred_hard_raw_sum / source_ratio_denom),
                "tiny_component_hits": float(comp_recall.get("1_99", {}).get("hits", 0.0)),
                "tiny_component_total": float(comp_recall.get("1_99", {}).get("components", 0.0)),
                "image": str(img_p),
                "mask": str(msk_p),
            }
            if self.use_center_head:
                center_map = np.asarray(pred_maps.get("center_heatmap"), dtype=np.float32)
                size_map = np.asarray(pred_maps.get("size_head"), dtype=np.float32) if self.use_size_head and "size_head" in pred_maps else None
                proposal_stats = proposal_recall_summary(
                    y_true=y,
                    center_map=center_map,
                    topk_values=self.center_topk_values,
                    case_size_bins=tuple(getattr(self.cfg, "CASE_SIZE_BINS", (100, 1000, 10000))),
                    match_radius=self.center_match_radius,
                    min_confidence=self.center_min_confidence,
                    nms_radius=self.center_nms_radius,
                    size_map=size_map,
                )
                row["center_candidates"] = int(len(proposal_stats.get("candidates", [])))
                row["center_candidates_above_threshold"] = int(proposal_stats.get("candidates_above_threshold", 0))
                row["center_best_score"] = float(proposal_stats.get("best_candidate_score", 0.0))
                for k in self.center_topk_values:
                    stats_k = proposal_stats.get("topk", {}).get(k, {})
                    rec_k = float(stats_k.get("recall", 0.0))
                    row[f"center_recall_at_{k}"] = rec_k
                    row[f"center_mean_best_dist_at_{k}"] = float(stats_k.get("mean_best_dist", float("inf")))
                    center_recall_macro[k].append(rec_k)
                    source_center_recall[k].setdefault(src_name, []).append(rec_k)
                    for grp, vals in (stats_k.get("by_group", {}) or {}).items():
                        bucket = center_recall_by_group[k].setdefault(grp, {"components": 0.0, "hits": 0.0})
                        bucket["components"] += float(vals.get("components", 0.0))
                        bucket["hits"] += float(vals.get("hits", 0.0))
                size_seed_acc = proposal_stats.get("size_seed_acc")
                row["size_seed_acc"] = float(size_seed_acc) if size_seed_acc is not None else float("nan")
                if size_seed_acc is not None:
                    size_seed_acc_all.append(float(size_seed_acc))
                    source_size_seed_acc.setdefault(src_name, []).append(float(size_seed_acc))
            for grp, vals in comp_recall.items():
                bucket = lesion_recall_bins.setdefault(grp, {"components": 0.0, "hits": 0.0})
                bucket["components"] += float(vals.get("components", 0.0))
                bucket["hits"] += float(vals.get("hits", 0.0))
            for thr in self.threshold_sweep:
                pred_thr_raw = (probs >= float(thr)).astype(np.float32, copy=False)
                inter_thr_raw = float(np.sum(y * pred_thr_raw))
                pred_thr_raw_sum = float(np.sum(pred_thr_raw))
                hard_thr_raw = float((2.0 * inter_thr_raw + self._eps) / (pred_thr_raw_sum + true + self._eps))
                threshold_case_scores_raw[thr].append(hard_thr_raw)
                row[f"hard_dice_raw_thr_{thr:.2f}"] = hard_thr_raw
                pred_thr_brainmask = apply_postprocessing(
                    probs,
                    threshold=thr,
                    min_size=0,
                    closing=0,
                    use_component_scoring=False,
                    brain_mask=brain_mask,
                    clamp=self.cfg.OTSU_CLAMP,
                    min_prob=self.cfg.OTSU_MIN_PROB,
                )
                inter_thr_brainmask = float(np.sum(y * pred_thr_brainmask))
                pred_thr_brainmask_sum = float(np.sum(pred_thr_brainmask))
                hard_thr_brainmask = float(
                    (2.0 * inter_thr_brainmask + self._eps) / (pred_thr_brainmask_sum + true + self._eps)
                )
                threshold_case_scores_brainmask[thr].append(hard_thr_brainmask)
                row[f"hard_dice_brainmask_thr_{thr:.2f}"] = hard_thr_brainmask
                pred_thr = apply_postprocessing(
                    probs,
                    threshold=thr,
                    min_size=0,
                    closing=0,
                    use_component_scoring=bool(getattr(self.cfg, "USE_COMPONENT_SCORING_POSTPROC", False)),
                    brain_mask=brain_mask if bool(getattr(self.cfg, "USE_BRAINMASK_POSTPROC", False)) else None,
                    clamp=self.cfg.OTSU_CLAMP,
                    min_prob=self.cfg.OTSU_MIN_PROB,
                )
                inter_thr = float(np.sum(y * pred_thr))
                pred_thr_sum = float(np.sum(pred_thr))
                hard_thr = float((2.0 * inter_thr + self._eps) / (pred_thr_sum + true + self._eps))
                threshold_case_scores[thr].append(hard_thr)
                row[f"hard_dice_thr_{thr:.2f}"] = hard_thr
            case_rows.append(row)

            if i % 8 == 0 or i == len(eval_pairs):
                logger.info(f"Whole-brain val progress: {i}/{len(eval_pairs)} cases")

        val_soft_macro = float(np.mean(case_soft)) if case_soft else 0.0
        val_soft_micro = float((2.0 * inter_soft + self._eps) / (pred_soft + true_sum + self._eps))
        val_hard_macro = float(np.mean(case_hard)) if case_hard else 0.0
        val_hard_macro_brainmask = float(np.mean(case_hard_brainmask)) if case_hard_brainmask else 0.0
        val_hard_macro_raw = float(np.mean(case_hard_raw)) if case_hard_raw else 0.0

        logs["val_dice_coefficient"] = val_soft_macro
        logs["val_whole_dice_micro"] = val_soft_micro
        logs["val_whole_dice_hard"] = val_hard_macro
        logs["val_whole_dice_hard_brainmask"] = val_hard_macro_brainmask
        logs["val_whole_dice_hard_raw"] = val_hard_macro_raw
        logs["val_whole_dice_hard_brainmask_delta"] = float(val_hard_macro_brainmask - val_hard_macro_raw)
        logs["val_whole_dice_hard_postproc_delta"] = float(val_hard_macro - val_hard_macro_raw)
        if self.use_center_head:
            for k, vals in sorted(center_recall_macro.items()):
                logs[f"val_center_recall_at_{int(k)}"] = float(np.mean(vals)) if vals else 0.0
            if size_seed_acc_all:
                logs["val_size_seed_acc"] = float(np.mean(size_seed_acc_all))
        hard_sweep_macro = {
            thr: float(np.mean(scores)) if scores else 0.0
            for thr, scores in threshold_case_scores.items()
        }
        hard_sweep_macro_brainmask = {
            thr: float(np.mean(scores)) if scores else 0.0
            for thr, scores in threshold_case_scores_brainmask.items()
        }
        hard_sweep_macro_raw = {
            thr: float(np.mean(scores)) if scores else 0.0
            for thr, scores in threshold_case_scores_raw.items()
        }
        for thr, val in hard_sweep_macro.items():
            key = f"val_whole_dice_hard_thr_{thr:.2f}".replace(".", "p")
            logs[key] = val
        for thr, val in hard_sweep_macro_raw.items():
            key = f"val_whole_dice_hard_raw_thr_{thr:.2f}".replace(".", "p")
            logs[key] = val
        for thr, val in hard_sweep_macro_brainmask.items():
            key = f"val_whole_dice_hard_brainmask_thr_{thr:.2f}".replace(".", "p")
            logs[key] = val
        if hard_sweep_macro:
            best_thr, best_thr_val = max(hard_sweep_macro.items(), key=lambda kv: kv[1])
            logs["val_whole_dice_hard_best_thr"] = float(best_thr)
            logs["val_whole_dice_hard_best_thr_score"] = float(best_thr_val)
        else:
            best_thr, best_thr_val = self.threshold, val_hard_macro
        if hard_sweep_macro_raw:
            best_thr_raw, best_thr_raw_val = max(hard_sweep_macro_raw.items(), key=lambda kv: kv[1])
            logs["val_whole_dice_hard_raw_best_thr"] = float(best_thr_raw)
            logs["val_whole_dice_hard_raw_best_thr_score"] = float(best_thr_raw_val)
        else:
            best_thr_raw, best_thr_raw_val = self.threshold, val_hard_macro_raw
        if hard_sweep_macro_brainmask:
            best_thr_brainmask, best_thr_brainmask_val = max(hard_sweep_macro_brainmask.items(), key=lambda kv: kv[1])
            logs["val_whole_dice_hard_brainmask_best_thr"] = float(best_thr_brainmask)
            logs["val_whole_dice_hard_brainmask_best_thr_score"] = float(best_thr_brainmask_val)
        else:
            best_thr_brainmask, best_thr_brainmask_val = self.threshold, val_hard_macro_brainmask

        dt = time.time() - t0
        logger.info(
            "Whole-brain val @epoch %d: soft_macro=%.5f soft_micro=%.5f hard_raw@thr%.2f=%.5f "
            "hard_brainmask@thr%.2f=%.5f delta_brain=%.5f hard_post@thr%.2f=%.5f delta_post=%.5f "
            "(cases=%d, %.1fs)"
            % (
                epoch,
                val_soft_macro,
                val_soft_micro,
                self.threshold,
                val_hard_macro_raw,
                self.threshold,
                val_hard_macro_brainmask,
                val_hard_macro_brainmask - val_hard_macro_raw,
                self.threshold,
                val_hard_macro,
                val_hard_macro - val_hard_macro_raw,
                len(eval_pairs),
                dt,
            )
        )
        if case_soft:
            soft_arr = np.asarray(case_soft, dtype=np.float32)
            hard_arr = np.asarray(case_hard, dtype=np.float32)
            hard_brainmask_arr = np.asarray(case_hard_brainmask, dtype=np.float32)
            hard_raw_arr = np.asarray(case_hard_raw, dtype=np.float32)
            logger.info(
                "Whole-brain val case stats: soft[min=%.5f p25=%.5f med=%.5f p75=%.5f max=%.5f] "
                "hard_raw[min=%.5f p25=%.5f med=%.5f p75=%.5f max=%.5f] "
                "hard_brainmask[min=%.5f p25=%.5f med=%.5f p75=%.5f max=%.5f] "
                "hard_post[min=%.5f p25=%.5f med=%.5f p75=%.5f max=%.5f]",
                float(np.min(soft_arr)),
                float(np.percentile(soft_arr, 25)),
                float(np.median(soft_arr)),
                float(np.percentile(soft_arr, 75)),
                float(np.max(soft_arr)),
                float(np.min(hard_raw_arr)),
                float(np.percentile(hard_raw_arr, 25)),
                float(np.median(hard_raw_arr)),
                float(np.percentile(hard_raw_arr, 75)),
                float(np.max(hard_raw_arr)),
                float(np.min(hard_brainmask_arr)),
                float(np.percentile(hard_brainmask_arr, 25)),
                float(np.median(hard_brainmask_arr)),
                float(np.percentile(hard_brainmask_arr, 75)),
                float(np.max(hard_brainmask_arr)),
                float(np.min(hard_arr)),
                float(np.percentile(hard_arr, 25)),
                float(np.median(hard_arr)),
                float(np.percentile(hard_arr, 75)),
                float(np.max(hard_arr)),
            )
            worst_soft = sorted(case_rows, key=lambda r: float(r["soft_dice"]))[: self.top_k]
            logger.info(
                "Whole-brain val worst soft-dice cases: %s",
                [
                    {
                        "case_id": r["case_id"],
                        "source": r["source"],
                        "soft_dice": round(float(r["soft_dice"]), 5),
                        "true_voxels": int(r["true_voxels"]),
                    }
                    for r in worst_soft
                ],
            )
        if source_soft:
            per_source_soft = {k: float(np.mean(v)) for k, v in sorted(source_soft.items())}
            per_source_hard = {k: float(np.mean(v)) for k, v in sorted(source_hard.items())}
            per_source_hard_brainmask = {k: float(np.mean(v)) for k, v in sorted(source_hard_brainmask.items())}
            per_source_hard_raw = {k: float(np.mean(v)) for k, v in sorted(source_hard_raw.items())}
            logger.info("Whole-brain val by source (soft_macro): %s", per_source_soft)
            logger.info("Whole-brain val by source (hard_raw@thr%.2f): %s", self.threshold, per_source_hard_raw)
            logger.info("Whole-brain val by source (hard_brainmask@thr%.2f): %s", self.threshold, per_source_hard_brainmask)
            logger.info("Whole-brain val by source (hard_macro@thr%.2f): %s", self.threshold, per_source_hard)
            if self.use_center_head:
                for k, src_vals in sorted(source_center_recall.items()):
                    logger.info(
                        "Whole-brain center proposal recall@%d by source: %s",
                        int(k),
                        {src: float(np.mean(vals)) for src, vals in sorted(src_vals.items()) if vals},
                    )
                if source_size_seed_acc:
                    logger.info(
                        "Whole-brain proposal size-seed accuracy by source: %s",
                        {src: float(np.mean(vals)) for src, vals in sorted(source_size_seed_acc.items()) if vals},
                    )
        if hard_sweep_macro:
            logger.info(
                "Whole-brain val threshold sweep (hard_macro): %s",
                {f"{k:.2f}": round(v, 5) for k, v in hard_sweep_macro.items()},
            )
            logger.info(
                "Whole-brain val threshold sweep (hard_brainmask_macro): %s",
                {f"{k:.2f}": round(v, 5) for k, v in hard_sweep_macro_brainmask.items()},
            )
            logger.info(
                "Whole-brain val threshold sweep (hard_raw_macro): %s",
                {f"{k:.2f}": round(v, 5) for k, v in hard_sweep_macro_raw.items()},
            )
            logger.info(
                "Whole-brain best thresholds: raw=%.2f(%.5f) brainmask=%.2f(%.5f) post=%.2f(%.5f)",
                float(best_thr_raw),
                float(best_thr_raw_val),
                float(best_thr_brainmask),
                float(best_thr_brainmask_val),
                float(best_thr),
                float(best_thr_val),
            )
        if pred_soft_true_ratio:
            logger.info(
                "Whole-brain voxel ratios vs truth: soft[med=%.3f p90=%.3f] raw_hard[med=%.3f p90=%.3f] "
                "brainmask_hard[med=%.3f p90=%.3f] post_hard[med=%.3f p90=%.3f]",
                float(np.median(pred_soft_true_ratio)),
                float(np.percentile(pred_soft_true_ratio, 90)),
                float(np.median(pred_hard_raw_true_ratio)),
                float(np.percentile(pred_hard_raw_true_ratio, 90)),
                float(np.median(pred_hard_brainmask_true_ratio)),
                float(np.percentile(pred_hard_brainmask_true_ratio, 90)),
                float(np.median(pred_hard_true_ratio)),
                float(np.percentile(pred_hard_true_ratio, 90)),
            )
            logger.info(
                "Whole-brain brain-mask hard-dice delta: mean=%.5f med=%.5f p75=%.5f",
                float(np.mean(brainmask_hard_delta)),
                float(np.median(brainmask_hard_delta)),
                float(np.percentile(brainmask_hard_delta, 75)),
            )
            logger.info(
                "Whole-brain postprocessing hard-dice delta: mean=%.5f med=%.5f p75=%.5f",
                float(np.mean(postproc_hard_delta)),
                float(np.median(postproc_hard_delta)),
                float(np.percentile(postproc_hard_delta, 75)),
            )

        if case_rows:
            bin_soft: dict[str, list[float]] = {}
            for r in case_rows:
                b = _lesion_size_bin(int(r["true_voxels"]))
                bin_soft.setdefault(b, []).append(float(r["soft_dice"]))
            logger.info(
                "Whole-brain val by lesion-size bin (soft_macro): %s",
                {k: float(np.mean(v)) for k, v in sorted(bin_soft.items())},
            )
        if lesion_recall_bins:
            lesionwise_recall = {
                grp: float(vals["hits"] / max(vals["components"], 1.0))
                for grp, vals in sorted(lesion_recall_bins.items())
            }
            logger.info("Whole-brain lesion-component recall proxy: %s", lesionwise_recall)
            tiny = lesion_recall_bins.get("1_99", {"components": 0.0, "hits": 0.0})
            if tiny["components"] > 0:
                missed = int(round(float(tiny["components"] - tiny["hits"])))
                logger.info("Whole-brain tiny-lesion misses: %d / %d", missed, int(round(float(tiny["components"]))))
        if self.use_center_head and center_recall_macro:
            logger.info(
                "Whole-brain center proposal recall: %s",
                {f"top{k}": round(float(np.mean(vals)), 5) for k, vals in sorted(center_recall_macro.items()) if vals},
            )
            center_best_scores = [float(r["center_best_score"]) for r in case_rows if "center_best_score" in r]
            center_candidate_counts = [float(r["center_candidates_above_threshold"]) for r in case_rows if "center_candidates_above_threshold" in r]
            if center_best_scores:
                logger.info(
                    "Whole-brain center score stats: max_mean=%.5f max_median=%.5f above_thr_mean=%.2f",
                    float(np.mean(center_best_scores)),
                    float(np.median(center_best_scores)),
                    float(np.mean(center_candidate_counts)) if center_candidate_counts else 0.0,
                )
            for k, grp_vals in sorted(center_recall_by_group.items()):
                logger.info(
                    "Whole-brain center proposal recall@%d by lesion group: %s",
                    int(k),
                    {
                        grp: round(float(vals["hits"] / max(vals["components"], 1.0)), 5)
                        for grp, vals in sorted(grp_vals.items())
                    },
                )
            if size_seed_acc_all:
                logger.info("Whole-brain proposal size-seed accuracy: %.5f", float(np.mean(size_seed_acc_all)))

        if self.diagnostics_enabled:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            csv_path = self.out_dir / f"whole_val_epoch_{int(epoch):04d}.csv"
            base_fields = [
                "source",
                "case_id",
                "soft_dice",
                "hard_dice",
                "hard_dice_brainmask",
                "hard_dice_raw",
                "true_voxels",
                "pred_soft_voxels",
                "pred_hard_voxels",
                "pred_hard_brainmask_voxels",
                "pred_hard_raw_voxels",
                "brainmask_hard_delta",
                "postproc_hard_delta",
                "pred_soft_to_true_ratio",
                "pred_hard_to_true_ratio",
                "pred_hard_brainmask_to_true_ratio",
                "pred_hard_raw_to_true_ratio",
                "image",
                "mask",
                "tiny_component_hits",
                "tiny_component_total",
                "center_candidates",
                "center_candidates_above_threshold",
                "center_best_score",
                "size_seed_acc",
            ]
            thr_fields = []
            for thr in self.threshold_sweep:
                thr_fields.append(f"hard_dice_thr_{thr:.2f}")
                thr_fields.append(f"hard_dice_brainmask_thr_{thr:.2f}")
                thr_fields.append(f"hard_dice_raw_thr_{thr:.2f}")
            center_fields = []
            for k in self.center_topk_values:
                center_fields.append(f"center_recall_at_{int(k)}")
                center_fields.append(f"center_mean_best_dist_at_{int(k)}")
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=base_fields + center_fields + thr_fields)
                writer.writeheader()
                for r in case_rows:
                    writer.writerow(r)
            summary = {
                "epoch": int(epoch),
                "elapsed_sec": float(dt),
                "n_cases": int(len(case_rows)),
                "val_soft_macro": float(val_soft_macro),
                "val_soft_micro": float(val_soft_micro),
                "val_hard_macro_brainmask": float(val_hard_macro_brainmask),
                "val_hard_macro_raw": float(val_hard_macro_raw),
                "val_hard_macro": float(val_hard_macro),
                "val_hard_macro_brainmask_delta": float(val_hard_macro_brainmask - val_hard_macro_raw),
                "val_hard_macro_postproc_delta": float(val_hard_macro - val_hard_macro_raw),
                "hard_sweep_macro": {f"{k:.2f}": float(v) for k, v in hard_sweep_macro.items()},
                "hard_sweep_macro_brainmask": {f"{k:.2f}": float(v) for k, v in hard_sweep_macro_brainmask.items()},
                "hard_sweep_macro_raw": {f"{k:.2f}": float(v) for k, v in hard_sweep_macro_raw.items()},
                "best_threshold_by_hard_macro": float(best_thr),
                "best_threshold_hard_macro": float(best_thr_val),
                "best_threshold_by_hard_macro_brainmask": float(best_thr_brainmask),
                "best_threshold_hard_macro_brainmask": float(best_thr_brainmask_val),
                "best_threshold_by_hard_macro_raw": float(best_thr_raw),
                "best_threshold_hard_macro_raw": float(best_thr_raw_val),
                "source_soft_macro": {k: float(np.mean(v)) for k, v in sorted(source_soft.items())},
                "source_hard_macro_brainmask": {k: float(np.mean(v)) for k, v in sorted(source_hard_brainmask.items())},
                "source_hard_macro_raw": {k: float(np.mean(v)) for k, v in sorted(source_hard_raw.items())},
                "source_hard_macro": {k: float(np.mean(v)) for k, v in sorted(source_hard.items())},
                "center_recall_macro": {
                    f"top{k}": float(np.mean(vals)) for k, vals in sorted(center_recall_macro.items()) if vals
                },
                "center_recall_by_group": {
                    f"top{k}": {
                        grp: {
                            "components": float(vals["components"]),
                            "hits": float(vals["hits"]),
                            "recall": float(vals["hits"] / max(vals["components"], 1.0)),
                        }
                        for grp, vals in sorted(grp_vals.items())
                    }
                    for k, grp_vals in sorted(center_recall_by_group.items())
                    if grp_vals
                },
                "source_center_recall_macro": {
                    f"top{k}": {src: float(np.mean(vals)) for src, vals in sorted(src_vals.items()) if vals}
                    for k, src_vals in sorted(source_center_recall.items())
                    if src_vals
                },
                "size_seed_acc": float(np.mean(size_seed_acc_all)) if size_seed_acc_all else None,
                "source_size_seed_acc": {
                    src: float(np.mean(vals)) for src, vals in sorted(source_size_seed_acc.items()) if vals
                },
                "pred_true_ratio_summary": {
                    "soft_median": float(np.median(pred_soft_true_ratio)) if pred_soft_true_ratio else 0.0,
                    "soft_p90": float(np.percentile(pred_soft_true_ratio, 90)) if pred_soft_true_ratio else 0.0,
                    "hard_raw_median": float(np.median(pred_hard_raw_true_ratio)) if pred_hard_raw_true_ratio else 0.0,
                    "hard_raw_p90": float(np.percentile(pred_hard_raw_true_ratio, 90)) if pred_hard_raw_true_ratio else 0.0,
                    "hard_brainmask_median": float(np.median(pred_hard_brainmask_true_ratio)) if pred_hard_brainmask_true_ratio else 0.0,
                    "hard_brainmask_p90": float(np.percentile(pred_hard_brainmask_true_ratio, 90)) if pred_hard_brainmask_true_ratio else 0.0,
                    "hard_post_median": float(np.median(pred_hard_true_ratio)) if pred_hard_true_ratio else 0.0,
                    "hard_post_p90": float(np.percentile(pred_hard_true_ratio, 90)) if pred_hard_true_ratio else 0.0,
                },
                "lesion_component_recall": {
                    grp: {
                        "components": float(vals["components"]),
                        "hits": float(vals["hits"]),
                        "recall": float(vals["hits"] / max(vals["components"], 1.0)),
                    }
                    for grp, vals in sorted(lesion_recall_bins.items())
                },
            }
            with open(self.summary_jsonl, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(summary) + "\n")


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
        batch_main = np.zeros((len(batch_pairs), *self.target_shape, 1), dtype=np.float32)
        batch_center = np.zeros((len(batch_pairs), *self.target_shape, 1), dtype=np.float32) if self.config.USE_CENTER_HEATMAP_HEAD else None
        batch_size = np.zeros((len(batch_pairs), *self.target_shape), dtype=np.int32) if self.config.USE_SIZE_HEAD else None
        batch_msl = np.zeros((len(batch_pairs), *self.target_shape), dtype=np.int32) if self.config.USE_AUX_MSL_HEAD else None
        batch_dbl = np.zeros((len(batch_pairs), *self.target_shape), dtype=np.int32) if self.config.USE_AUX_DBL_HEAD else None
        
        for i, (img_path, msk_path) in enumerate(batch_pairs):
            # Load and preprocess image and mask
            img = self.image_loader(str(img_path), self.target_shape)
            msk = self.mask_loader(str(msk_path), self.target_shape)
            
            # Apply augmentations if in training mode
            if self.is_training and self.config.AUGMENTATION_INTENSITY > 0:
                img, msk = self._augment(img, msk)
            
            batch_x[i] = _make_input_channels(img, self.config)
            batch_main[i, ..., 0] = (msk > 0.5).astype(np.float32, copy=False)
            if batch_center is not None:
                batch_center[i] = build_center_heatmap(msk, sigma=float(getattr(self.config, "CENTER_HEATMAP_SIGMA", 2.0)))
            if batch_size is not None:
                batch_size[i] = build_size_head_labels(msk, tuple(self.config.CASE_SIZE_BINS))
            if batch_msl is not None:
                batch_msl[i] = build_msl_labels(msk, tuple(self.config.MSL_COMPONENT_THRESHOLDS))
            if batch_dbl is not None:
                batch_dbl[i] = build_dbl_labels(msk)
        if batch_center is None and batch_size is None and batch_msl is None and batch_dbl is None:
            return batch_x, batch_main
        targets = {"probs": batch_main}
        if batch_center is not None:
            targets["center_heatmap"] = batch_center
        if batch_size is not None:
            targets["size_head"] = batch_size
        if batch_msl is not None:
            targets["msl_head"] = batch_msl
        if batch_dbl is not None:
            targets["dbl_head"] = batch_dbl
        return batch_x, targets
    
    def on_epoch_end(self):
        """Called at the end of each epoch."""
        if self.is_training:
            random.shuffle(self.pairs)
    
    def _augment(self, image, mask):
        return apply_augmentations(image, mask, self.config, self.rng)

def load_generic_dataset(config: DynamicTrainingConfig):
    logger.info("📚 Loading dataset (flex loader for T1w volumes)…")
    log_memory_usage("dataset_load_start")

    manifest_path = getattr(config, "MANIFEST_PATH", None)
    manifest_path = Path(manifest_path) if manifest_path is not None else (config.DATA_DIR / "manifest.csv")
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
    logger.info(
        "Small-lesion trainer enabled: flip_channel=%s center_head=%s size_head=%s aux_msl=%s aux_dbl=%s topk=%.2f "
        "lesion_insertion=%.2f component_patch_sampling=%s tiny_centering=%s brainmask_postproc=%s component_postproc=%s "
        "case_group_probs=%s case_none_prob=%.3f patch_fg_probs=%s curriculum=%s atlas_finetune=%s",
        bool(getattr(config, "USE_SYMMETRIC_FLIP_CHANNEL", False)),
        bool(getattr(config, "USE_CENTER_HEATMAP_HEAD", False)),
        bool(getattr(config, "USE_SIZE_HEAD", False)),
        bool(getattr(config, "USE_AUX_MSL_HEAD", False)),
        bool(getattr(config, "USE_AUX_DBL_HEAD", False)),
        float(getattr(config, "TOPK_VOXEL_FRACTION", 0.0)),
        float(getattr(config, "LESION_INSERTION_PROB", 0.0)),
        bool(getattr(config, "USE_COMPONENT_AWARE_PATCH_SAMPLING", False)),
        bool(getattr(config, "USE_TINY_COMPONENT_CENTERING", False)),
        bool(getattr(config, "USE_BRAINMASK_POSTPROC", False)),
        bool(getattr(config, "USE_COMPONENT_SCORING_POSTPROC", False)),
        tuple(getattr(config, "CASE_SIZE_GROUP_PROBS", (0.45, 0.25, 0.15, 0.10))),
        float(getattr(config, "CASE_NONE_PROB", 0.05)),
        tuple(getattr(config, "PATCH_FG_PROB_BY_BIN", ())),
        bool(getattr(config, "USE_SIZE_CURRICULUM", False)),
        bool(getattr(config, "USE_ATLAS_FINE_TUNE", False)),
    )

    if getattr(config, "INPUT_SHAPE", None) in (None, (), []):
        max_dims = detect_input_shape(config.DATA_DIR)
        config.INPUT_SHAPE = tuple(max_dims) + (config.input_channels,)
    patch_shape = tuple(config.PATCH_SIZE or config.INPUT_SHAPE[:-1])
    if tuple(config.INPUT_SHAPE[:-1]) != patch_shape:
        config.INPUT_SHAPE = patch_shape + (config.input_channels,)
    config._write_config()

    # Case-loading shape (separate from model patch shape).
    # - LOAD_FULL_IMAGE_FOR_PATCHING=True + FULL_RES_TARGET_SHAPE=None: keep native full volume.
    # - LOAD_FULL_IMAGE_FOR_PATCHING=True + FULL_RES_TARGET_SHAPE=(...): resample/crop full volume to that shape.
    # - LOAD_FULL_IMAGE_FOR_PATCHING=False: legacy behavior (load directly to patch/model shape).
    case_target_shape = _full_volume_target_shape(config)
    if bool(getattr(config, "LOAD_FULL_IMAGE_FOR_PATCHING", True)):
        logger.info(
            "Patch extraction source: full-volume mode "
            f"(target_shape={case_target_shape if case_target_shape is not None else 'native'})"
        )
    else:
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
                bce_weight=config.BCE_WEIGHT,
                topk_weight=config.TOPK_WEIGHT,
                topk_voxel_fraction=config.TOPK_VOXEL_FRACTION,
                volume_ratio_weight=config.VOLUME_RATIO_WEIGHT,
                focal_weight=config.FOCAL_TVERSKY_WEIGHT,
                tversky_alpha=config.TVERSKY_ALPHA,
                tversky_beta=config.TVERSKY_BETA,
                focal_gamma=config.FOCAL_TVERSKY_GAMMA,
            )
        adam_kwargs = {"learning_rate": lr_schedule, "epsilon": 1e-8}
        if float(getattr(config, "MAX_GRAD_NORM", 0.0) or 0.0) > 0.0:
            adam_kwargs["clipnorm"] = float(config.MAX_GRAD_NORM)
        compile_kwargs = {
            "optimizer": tf.keras.optimizers.Adam(**adam_kwargs),
        }
        use_multi_output = any(
            bool(getattr(config, name, False))
            for name in ("USE_CENTER_HEATMAP_HEAD", "USE_SIZE_HEAD", "USE_AUX_MSL_HEAD", "USE_AUX_DBL_HEAD")
        )
        if use_multi_output:
            loss_map = {"probs": loss_obj}
            loss_weights = {"probs": 1.0}
            if bool(getattr(config, "USE_CENTER_HEATMAP_HEAD", False)):
                loss_map["center_heatmap"] = GaussianHeatmapFocalLoss(
                    gamma=float(getattr(config, "CENTER_LOSS_GAMMA", 2.0)),
                    beta=float(getattr(config, "CENTER_LOSS_BETA", 4.0)),
                    name="center_heatmap_loss",
                )
                loss_weights["center_heatmap"] = float(getattr(config, "AUX_CENTER_WEIGHT", 0.12))
            if bool(getattr(config, "USE_SIZE_HEAD", False)):
                loss_map["size_head"] = WeightedSparseCategoricalCrossentropy(
                    getattr(config, "SIZE_HEAD_CLASS_WEIGHTS", (0.02, 4.0, 2.5, 1.0, 0.6)),
                    name="size_head_loss",
                )
                loss_weights["size_head"] = float(getattr(config, "AUX_SIZE_WEIGHT", 0.05))
            if bool(getattr(config, "USE_AUX_MSL_HEAD", True)):
                loss_map["msl_head"] = WeightedSparseCategoricalCrossentropy(
                    getattr(config, "AUX_MSL_CLASS_WEIGHTS", (0.02, 4.0, 2.5, 1.0, 0.6)),
                    name="msl_loss",
                )
                loss_weights["msl_head"] = float(getattr(config, "AUX_MSL_WEIGHT", 0.15))
            if bool(getattr(config, "USE_AUX_DBL_HEAD", True)):
                loss_map["dbl_head"] = WeightedSparseCategoricalCrossentropy(
                    getattr(config, "AUX_DBL_CLASS_WEIGHTS", (0.02, 1.15, 1.0)),
                    name="dbl_loss",
                )
                loss_weights["dbl_head"] = float(getattr(config, "AUX_DBL_WEIGHT", 0.10))
            compile_kwargs["loss"] = loss_map
            compile_kwargs["loss_weights"] = loss_weights
            compile_kwargs["metrics"] = {"probs": [dice_coefficient, safe_binary_iou]}
        else:
            compile_kwargs["loss"] = loss_obj
            compile_kwargs["metrics"] = [dice_coefficient, safe_binary_iou]
        model.compile(**compile_kwargs)
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
    def _make_dataset_namespace(data_dir, images_dir, masks_dir, manifest_path=None):
        return SimpleNamespace(
            DATA_DIR=Path(data_dir),
            IMAGES_DIR=Path(images_dir),
            MASKS_DIR=Path(masks_dir),
            IMAGE_SUFFIXES=config.IMAGE_SUFFIXES,
            MASK_SUFFIXES=config.MASK_SUFFIXES,
            MANIFEST_PATH=Path(manifest_path) if manifest_path else None,
        )

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
    smallest_component_sizes_all = compute_smallest_component_sizes(pairs, mask_preprocess_fn, case_target_shape)

    if config.EXTERNAL_VAL_DIR is not None or config.EXTERNAL_VAL_MANIFEST is not None:
        val_root = config.EXTERNAL_VAL_DIR if config.EXTERNAL_VAL_DIR is not None else config.DATA_DIR
        val_images = config.EXTERNAL_VAL_IMAGES_DIR if config.EXTERNAL_VAL_IMAGES_DIR is not None else val_root
        val_masks = config.EXTERNAL_VAL_MASKS_DIR if config.EXTERNAL_VAL_MASKS_DIR is not None else val_root
        ext_cfg = _make_dataset_namespace(
            data_dir=val_root,
            images_dir=val_images,
            masks_dir=val_masks,
            manifest_path=config.EXTERNAL_VAL_MANIFEST,
        )
        val_pairs, _ = load_generic_dataset(ext_cfg)
        val_pairs = list(val_pairs)
        train_pairs = list(pairs)
        val_lesion_sizes = compute_lesion_sizes(val_pairs, mask_preprocess_fn, case_target_shape)
        val_smallest_component_sizes = compute_smallest_component_sizes(val_pairs, mask_preprocess_fn, case_target_shape)
        logger.info(
            "Using explicit external validation set: train=%d val=%d",
            len(train_pairs),
            len(val_pairs),
        )
    else:
        train_pairs, val_pairs = create_stratified_splits(
            pairs,
            lesion_presence,
            lesion_sizes=lesion_sizes_all,
            smallest_component_sizes=smallest_component_sizes_all,
            case_size_bins=tuple(getattr(config, "CASE_SIZE_BINS", (100, 1000, 10000))),
            batch_size=config.BATCH_SIZE,
            test_size=config.VALIDATION_SPLIT,
        )
        train_pairs = list(train_pairs)
        val_pairs = list(val_pairs)
        pair_lookup = {(str(img_p), str(msk_p)): idx for idx, (img_p, msk_p) in enumerate(pairs)}
        train_indices = np.asarray([pair_lookup[(str(img), str(msk))] for img, msk in train_pairs], dtype=np.int64)
        val_indices = np.asarray([pair_lookup[(str(img), str(msk))] for img, msk in val_pairs], dtype=np.int64)
        val_lesion_sizes = lesion_sizes_all[val_indices] if len(val_indices) else np.asarray([], dtype=np.int64)
        val_smallest_component_sizes = smallest_component_sizes_all[val_indices] if len(val_indices) else np.asarray([], dtype=np.int64)
    train_lookup = {(str(img_p), str(msk_p)): i for i, (img_p, msk_p) in enumerate(pairs)}
    train_indices = np.asarray([train_lookup[(str(img), str(msk))] for img, msk in train_pairs], dtype=np.int64)
    train_source_counts = {}
    for img_p, _ in train_pairs:
        src = _path_source(img_p)
        train_source_counts[src] = train_source_counts.get(src, 0) + 1
    val_source_counts = {}
    for img_p, _ in val_pairs:
        src = _path_source(img_p)
        val_source_counts[src] = val_source_counts.get(src, 0) + 1
    logger.info("Train source composition: %s", train_source_counts)
    logger.info("Val source composition: %s", val_source_counts)
    train_lesion_sizes = lesion_sizes_all[train_indices]
    train_smallest_component_sizes = smallest_component_sizes_all[train_indices]
    logger.info(
        "Split lesion voxels: train_mean=%.1f train_median=%.1f | val_mean=%.1f val_median=%.1f",
        float(np.mean(train_lesion_sizes)) if len(train_lesion_sizes) else 0.0,
        float(np.median(train_lesion_sizes)) if len(train_lesion_sizes) else 0.0,
        float(np.mean(val_lesion_sizes)) if len(val_lesion_sizes) else 0.0,
        float(np.median(val_lesion_sizes)) if len(val_lesion_sizes) else 0.0,
    )
    logger.info(
        "Split smallest component voxels: train_mean=%.1f train_median=%.1f | val_mean=%.1f val_median=%.1f",
        float(np.mean(train_smallest_component_sizes)) if len(train_smallest_component_sizes) else 0.0,
        float(np.median(train_smallest_component_sizes)) if len(train_smallest_component_sizes) else 0.0,
        float(np.mean(val_smallest_component_sizes)) if len(val_smallest_component_sizes) else 0.0,
        float(np.median(val_smallest_component_sizes)) if len(val_smallest_component_sizes) else 0.0,
    )
    if bool(getattr(config, "DIAGNOSTICS_ENABLED", True)):
        diag_pairs = list(train_pairs) + list(val_pairs)
        diag_lesion_sizes = np.concatenate([train_lesion_sizes, val_lesion_sizes.astype(np.int64)], axis=0) if len(val_pairs) else train_lesion_sizes.copy()
        diag_smallest_sizes = (
            np.concatenate([train_smallest_component_sizes, val_smallest_component_sizes.astype(np.int64)], axis=0)
            if len(val_pairs)
            else train_smallest_component_sizes.copy()
        )
        diag_lookup = {(str(img_p), str(msk_p)): i for i, (img_p, msk_p) in enumerate(diag_pairs)}
        _write_split_diagnostics(
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            pair_lookup=diag_lookup,
            lesion_sizes_all=diag_lesion_sizes,
            smallest_component_sizes_all=diag_smallest_sizes,
            case_size_bins=tuple(getattr(config, "CASE_SIZE_BINS", (100, 1000, 10000))),
            out_dir=Path(config.CALLBACKS_DIR) / "diagnostics",
        )
        if len(train_pairs) >= config.GROUPED_CV_FOLDS:
            folds = create_grouped_size_balanced_folds(
                train_pairs,
                train_smallest_component_sizes,
                n_splits=config.GROUPED_CV_FOLDS,
                case_size_bins=tuple(getattr(config, "CASE_SIZE_BINS", (100, 1000, 10000))),
                random_state=config.RNG_SEED,
            )
            fold_rows = []
            for fold_id, (_, val_idx_fold) in enumerate(folds):
                for case_idx in val_idx_fold.tolist():
                    img_p, msk_p = train_pairs[int(case_idx)]
                    fold_rows.append(
                        {
                            "fold": int(fold_id),
                            "source": _path_source(img_p),
                            "case_id": _path_case_id(img_p),
                            "group": _subject_group_key(img_p),
                            "case_size_group": _case_size_group(
                                int(train_smallest_component_sizes[int(case_idx)]),
                                tuple(getattr(config, "CASE_SIZE_BINS", (100, 1000, 10000))),
                            ),
                            "image": str(img_p),
                            "mask": str(msk_p),
                        }
                    )
            cv_path = Path(config.CALLBACKS_DIR) / "diagnostics" / "grouped_cv_folds.csv"
            with open(cv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=["fold", "source", "case_id", "group", "case_size_group", "image", "mask"],
                )
                writer.writeheader()
                writer.writerows(fold_rows)
    train_source_labels = np.asarray([_path_source(img_p) for img_p, _ in train_pairs], dtype=object) if len(train_pairs) else None
    case_sampler = (
        SizeAwareCaseSampler(train_smallest_component_sizes, config, source_labels=train_source_labels)
        if len(train_pairs)
        else None
    )
    if case_sampler is not None and bool(getattr(config, "SOURCE_BALANCED_SAMPLING", True)):
        logger.info(
            "Source-balanced sampling enabled across %d sources: %s",
            len(np.unique(train_source_labels)),
            {src: int(np.sum(train_source_labels == src)) for src in np.unique(train_source_labels)},
        )
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
    lesion_bank = (
        build_small_lesion_bank(train_pairs, config, target_shape=case_target_shape)
        if float(getattr(config, "LESION_INSERTION_PROB", 0.0)) > 0.0
        else []
    )

    def training_batch_generator():
        if not train_pairs:
            raise RuntimeError("No training pairs available for generator.")
        while True:
            idxs = case_sampler.sample_indices(batch_cases) if case_sampler else np.random.choice(len(train_pairs), size=batch_cases, replace=True)
            xs = []
            y_main = []
            y_center = [] if config.USE_CENTER_HEATMAP_HEAD else None
            y_size = [] if config.USE_SIZE_HEAD else None
            y_msl = [] if config.USE_AUX_MSL_HEAD else None
            y_dbl = [] if config.USE_AUX_DBL_HEAD else None
            for idx in idxs:
                img_p, msk_p = train_pairs[idx]
                x = _load_and_preprocess_image(str(img_p), case_target_shape)
                y = _load_and_preprocess_mask(str(msk_p), case_target_shape)
                component_records = _build_component_records(y)
                lesion_size = train_smallest_component_sizes[idx] if len(train_smallest_component_sizes) > idx else _smallest_positive_component(y)
                if config.SIZE_AWARE_ENABLED and config.SIZE_AWARE_MODE == "bucket":
                    bin_id = bin_index(lesion_size, config.CASE_SIZE_BINS)
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
                        component_records=component_records,
                        case_size_bins=tuple(getattr(config, "CASE_SIZE_BINS", (100, 1000, 10000))),
                        use_component_aware_sampling=bool(getattr(config, "USE_COMPONENT_AWARE_PATCH_SAMPLING", False)),
                        use_tiny_component_centering=bool(getattr(config, "USE_TINY_COMPONENT_CENTERING", False)),
                        tiny_component_center_prob=float(getattr(config, "TINY_COMPONENT_CENTER_PROB", 0.95)),
                        small_component_center_prob=float(getattr(config, "SMALL_COMPONENT_CENTER_PROB", 0.85)),
                        tiny_component_max_jitter=int(getattr(config, "TINY_COMPONENT_MAX_JITTER", 2)),
                        small_component_max_jitter=int(getattr(config, "SMALL_COMPONENT_MAX_JITTER", 4)),
                        hemisphere_axis=hemisphere_axis if hemisphere_mode else None,
                        hemisphere_side=hemisphere_side,
                    )
                    patch_x = x[z0:z1, y0:y1, x0:x1]
                    patch_y = y[z0:z1, y0:y1, x0:x1]
                    if patch_x.shape != patch_size:
                        patch_x = _center_crop_or_pad_volume(patch_x, patch_size)
                    if patch_y.shape != patch_size:
                        patch_y = _center_crop_or_pad_volume(patch_y, patch_size)
                    if lesion_bank:
                        patch_x, patch_y = apply_lesion_insertion(patch_x, patch_y, lesion_bank, config, rng)
                    if config.AUGMENTATION_INTENSITY > 0:
                        patch_x, patch_y = apply_augmentations(patch_x, patch_y, config, rng)
                    xs.append(_make_input_channels(patch_x, config))
                    targets = _format_training_targets(patch_y, config)
                    if isinstance(targets, dict):
                        y_main.append(targets["probs"])
                        if y_center is not None:
                            y_center.append(targets["center_heatmap"])
                        if y_size is not None:
                            y_size.append(targets["size_head"])
                        if y_msl is not None:
                            y_msl.append(targets["msl_head"])
                        if y_dbl is not None:
                            y_dbl.append(targets["dbl_head"])
                    else:
                        y_main.append(targets)
            xb = np.stack(xs, axis=0).astype(np.float32, copy=False)
            yb_main = np.stack(y_main, axis=0).astype(np.float32, copy=False)
            if (not np.isfinite(xb).all()) or (not np.isfinite(yb_main).all()):
                bad_x = int(np.size(xb) - np.isfinite(xb).sum())
                bad_y = int(np.size(yb_main) - np.isfinite(yb_main).sum())
                logger.warning(
                    f"Non-finite batch values detected (x={bad_x}, y={bad_y}); replacing with zeros."
                )
                xb = np.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0)
                yb_main = np.nan_to_num(yb_main, nan=0.0, posinf=0.0, neginf=0.0)
            yb_main = (yb_main > 0.5).astype(np.float32, copy=False)
            if y_center is None and y_size is None and y_msl is None and y_dbl is None:
                yield xb, yb_main
            else:
                out_targets = {"probs": yb_main}
                if y_center is not None:
                    out_targets["center_heatmap"] = np.stack(y_center, axis=0).astype(np.float32, copy=False)
                if y_size is not None:
                    out_targets["size_head"] = np.stack(y_size, axis=0).astype(np.int32, copy=False)
                if y_msl is not None:
                    out_targets["msl_head"] = np.stack(y_msl, axis=0).astype(np.int32, copy=False)
                if y_dbl is not None:
                    out_targets["dbl_head"] = np.stack(y_dbl, axis=0).astype(np.int32, copy=False)
                yield xb, out_targets

    output_signature = (
        tf.TensorSpec(shape=(batch_patches, *patch_size, config.input_channels), dtype=tf.float32),
        tf.TensorSpec(shape=(batch_patches, *patch_size, 1), dtype=tf.float32),
    )
    if config.USE_CENTER_HEATMAP_HEAD or config.USE_SIZE_HEAD or config.USE_AUX_MSL_HEAD or config.USE_AUX_DBL_HEAD:
        target_signature = {"probs": tf.TensorSpec(shape=(batch_patches, *patch_size, 1), dtype=tf.float32)}
        if config.USE_CENTER_HEATMAP_HEAD:
            target_signature["center_heatmap"] = tf.TensorSpec(shape=(batch_patches, *patch_size, 1), dtype=tf.float32)
        if config.USE_SIZE_HEAD:
            target_signature["size_head"] = tf.TensorSpec(shape=(batch_patches, *patch_size), dtype=tf.int32)
        if config.USE_AUX_MSL_HEAD:
            target_signature["msl_head"] = tf.TensorSpec(shape=(batch_patches, *patch_size), dtype=tf.int32)
        if config.USE_AUX_DBL_HEAD:
            target_signature["dbl_head"] = tf.TensorSpec(shape=(batch_patches, *patch_size), dtype=tf.int32)
        output_signature = (
            tf.TensorSpec(shape=(batch_patches, *patch_size, config.input_channels), dtype=tf.float32),
            target_signature,
        )
    train_ds = tf.data.Dataset.from_generator(training_batch_generator, output_signature=output_signature).prefetch(tf.data.AUTOTUNE)
    use_whole_brain_val = bool(getattr(config, "WHOLE_BRAIN_VAL_ENABLED", True))
    val_gen = None
    if not use_whole_brain_val:
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

    monitor_metric = "val_whole_dice_hard_raw" if use_whole_brain_val else "val_loss"
    monitor_mode = "max" if use_whole_brain_val else "min"
    logger.info("Checkpoint monitor: %s (%s)", monitor_metric, monitor_mode)
    checkpoint_cb = ModelCheckpoint(
        filepath=str(config.checkpoint_path),
        monitor=monitor_metric,
        save_best_only=True,
        save_weights_only=True,
        mode=monitor_mode,
        verbose=1,
    )
    latest_cb = ModelCheckpoint(
        filepath=str(config.CALLBACKS_DIR / "latest.weights.h5"),
        save_weights_only=True,
        save_freq="epoch",
        verbose=0,
    )
    csv_cb = CSVLogger(Path(config.CALLBACKS_DIR) / "training_log.csv", append=False)
    memory_cb = None
    if bool(getattr(config, "MEMORY_LOGS_ENABLED", False)):
        memory_cb = MemoryMonitoringCallback(
            log_frequency=int(getattr(config, "MEMORY_LOG_BATCH_FREQUENCY", 0))
        )
    batch_metrics_cb = None
    epoch_jsonl_cb = None
    if bool(getattr(config, "DIAGNOSTICS_ENABLED", True)):
        batch_metrics_cb = BatchMetricsCSVLogger(
            out_csv=Path(config.CALLBACKS_DIR) / "batch_metrics.csv",
            log_every_n_steps=int(getattr(config, "BATCH_LOG_EVERY_N_STEPS", 1)),
        )
        epoch_jsonl_cb = EpochMetricsJSONLLogger(
            out_jsonl=Path(config.CALLBACKS_DIR) / "epoch_metrics.jsonl"
        )
        logger.info(
            "Diagnostics enabled: batch=%s epoch=%s whole-val-summary=%s",
            Path(config.CALLBACKS_DIR) / "batch_metrics.csv",
            Path(config.CALLBACKS_DIR) / "epoch_metrics.jsonl",
            Path(config.CALLBACKS_DIR) / "whole_val_summary.jsonl",
        )
    progress_cb = tf.keras.callbacks.ProgbarLogger()

    nvml_cb = None
    if gpus and NvmlGpuMemLogger is not None:
        try:
            nvml_cb = NvmlGpuMemLogger(gpus, interval=10)
            logger.info("NVML memory logging enabled")
        except Exception as e:
            logger.warning(f"Failed to initialize NVML logger: {e}")

    sampler_cb = SizeAwareSamplerCallback(case_sampler)
    sampling_policy_cb = SamplingPolicyController(case_sampler, config, train_source_labels)
    metric_alias_cb = PrimaryOutputMetricAliasCallback() if (
        config.USE_CENTER_HEATMAP_HEAD or config.USE_SIZE_HEAD or config.USE_AUX_MSL_HEAD or config.USE_AUX_DBL_HEAD
    ) else None
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

    whole_brain_val_cb = WholeBrainValidationCallback(val_pairs, config) if use_whole_brain_val else None
    callbacks = [
        cb for cb in (
            sampler_cb,
            sampling_policy_cb,
            metric_alias_cb,
            diff_cb,
            loss_ramp_cb,
            whole_brain_val_cb,
            epoch_jsonl_cb,
        ) if cb is not None
    ]
    fit_verbose = int(getattr(config, "FIT_VERBOSE", 2))
    if fit_verbose not in (0, 1, 2):
        logger.warning(f"Unsupported FIT_VERBOSE={fit_verbose}; using 2 (epoch-only).")
        fit_verbose = 2

    callbacks.extend([cb for cb in (checkpoint_cb, latest_cb, csv_cb, memory_cb, batch_metrics_cb) if cb is not None])
    if fit_verbose == 1:
        callbacks.append(progress_cb)
    callbacks.append(NonFiniteLossGuard())
    callbacks.append(tf.keras.callbacks.TerminateOnNaN())
    if nvml_cb is not None:
        callbacks.append(nvml_cb)
    if swa_cb is not None:
        callbacks.append(swa_cb)

    # --- Train the model ---
    if use_whole_brain_val and "validation_steps" in fit_kwargs:
        logger.warning("Ignoring validation_steps override: whole-brain validation callback is enabled.")
        fit_kwargs.pop("validation_steps", None)

    fit_args = dict(
        x=train_ds,
        epochs=config.TOTAL_EPOCHS,
        callbacks=callbacks,
        initial_epoch=config.INITIAL_EPOCH,
        verbose=fit_verbose,
        steps_per_epoch=config.EPOCH_STEPS,
    )
    if val_gen is not None:
        fit_args["validation_data"] = val_gen
        fit_args["validation_steps"] = len(val_gen)
    fit_args.update({k: v for k, v in fit_kwargs.items() if v is not None})

    history = model.fit(**fit_args)
    logger.info(f"Training complete: {history.history.keys()}")
    if bool(getattr(config, "DIAGNOSTICS_ENABLED", True)):
        try:
            _write_training_summary(history, config)
        except Exception as e:
            logger.warning(f"Could not write training diagnostics summary: {e}")
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


def gaussian_tta_predict_outputs(
    model: tf.keras.Model,
    volume: np.ndarray,
    patch_size: tuple[int, int, int],
    overlap: float = 0.5,
    sigma: float = 0.125,
    tta: bool = True,
    output_names: tuple[str, ...] = ("probs",),
):
    patch_size = tuple(int(v) for v in patch_size)
    padded, pads = _pad_volume_to_shape(volume, patch_size)
    weight_patch = _gaussian_patch_weights(patch_size, sigma=sigma)
    output_names = tuple(output_names or ("probs",))
    accumulators: dict[str, np.ndarray] = {}
    flip_sets = [()]
    if tta:
        flip_sets = [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]

    for axes in flip_sets:
        vol_aug = np.flip(padded, axis=axes) if axes else padded
        blended: dict[str, np.ndarray] = {}
        weight_accum: dict[str, np.ndarray] = {}
        for z0, y0, x0 in _sliding_window_positions(vol_aug.shape, patch_size, overlap):
            z1, y1, x1 = z0 + patch_size[0], y0 + patch_size[1], x0 + patch_size[2]
            patch = vol_aug[z0:z1, y0:y1, x0:x1]
            if patch.shape != patch_size:
                patch = _center_crop_or_pad_volume(patch, patch_size)
            in_channels = int(model.input_shape[-1]) if hasattr(model, "input_shape") else 1
            if in_channels == 2:
                cfg_ref = _ACTIVE_CONFIG if _ACTIVE_CONFIG is not None else SimpleNamespace(USE_SYMMETRIC_FLIP_CHANNEL=True, HEMISPHERE_AXIS=2)
                patch_input = _make_input_channels(patch, cfg_ref)[np.newaxis, ...]
            else:
                patch_input = patch[..., np.newaxis][np.newaxis, ...]
            raw_pred = model.predict(patch_input, verbose=0)
            for output_name in output_names:
                patch_pred = _prediction_output(raw_pred, model=model, output_name=output_name)[0]
                patch_pred = np.asarray(patch_pred, dtype=np.float32)
                if patch_pred.ndim == 3:
                    patch_pred = patch_pred[..., np.newaxis]
                if output_name not in blended:
                    blended[output_name] = np.zeros((*padded.shape, patch_pred.shape[-1]), dtype=np.float32)
                    weight_accum[output_name] = np.zeros((*padded.shape, patch_pred.shape[-1]), dtype=np.float32)
                    accumulators.setdefault(output_name, np.zeros((*padded.shape, patch_pred.shape[-1]), dtype=np.float32))
                blended[output_name][z0:z1, y0:y1, x0:x1] += patch_pred * weight_patch[..., np.newaxis]
                weight_accum[output_name][z0:z1, y0:y1, x0:x1] += weight_patch[..., np.newaxis]
        for output_name in output_names:
            blended_out = blended[output_name] / np.maximum(weight_accum[output_name], 1e-6)
            if axes:
                blended_out = np.flip(blended_out, axis=axes)
            accumulators[output_name] += blended_out
    outputs = {}
    for output_name in output_names:
        blended_avg = accumulators[output_name] / float(len(flip_sets))
        cropped = _crop_from_pad(blended_avg, pads)
        if cropped.ndim == 4 and cropped.shape[-1] == 1:
            cropped = cropped[..., 0]
        outputs[output_name] = cropped
    return outputs


def gaussian_tta_predict(
    model: tf.keras.Model,
    volume: np.ndarray,
    patch_size: tuple[int, int, int],
    overlap: float = 0.5,
    sigma: float = 0.125,
    tta: bool = True,
):
    return gaussian_tta_predict_outputs(
        model,
        volume,
        patch_size=patch_size,
        overlap=overlap,
        sigma=sigma,
        tta=tta,
        output_names=("probs",),
    )["probs"]


def gaussian_tta_predict_output(
    model: tf.keras.Model,
    volume: np.ndarray,
    patch_size: tuple[int, int, int],
    overlap: float = 0.5,
    sigma: float = 0.125,
    tta: bool = True,
    output_name: str = "probs",
):
    return gaussian_tta_predict_outputs(
        model,
        volume,
        patch_size=patch_size,
        overlap=overlap,
        sigma=sigma,
        tta=tta,
        output_names=(output_name,),
    )[output_name]


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


def score_connected_components(
    probs: np.ndarray,
    pred_mask: np.ndarray,
    cfg: DynamicTrainingConfig | None = None,
) -> np.ndarray:
    cfg = cfg or globals().get("_ACTIVE_CONFIG")
    if cfg is None or not bool(getattr(cfg, "USE_COMPONENT_SCORING_POSTPROC", False)):
        return pred_mask.astype(np.float32, copy=False)
    lbl, n = label(pred_mask > 0, structure=generate_binary_structure(3, 1))
    if n <= 0:
        return pred_mask.astype(np.float32, copy=False)
    counts = np.bincount(lbl.ravel())
    keep_mask = np.zeros_like(pred_mask, dtype=bool)
    bins = tuple(getattr(cfg, "CASE_SIZE_BINS", (100, 1000, 10000)))
    for comp_id in range(1, n + 1):
        size = int(counts[comp_id]) if comp_id < counts.size else 0
        if size <= 0:
            continue
        comp = lbl == comp_id
        comp_probs = probs[comp]
        mean_prob = float(np.mean(comp_probs)) if comp_probs.size else 0.0
        max_prob = float(np.max(comp_probs)) if comp_probs.size else 0.0
        if size < bins[0]:
            keep = mean_prob >= float(getattr(cfg, "COMPONENT_SCORE_TINY_MIN_MEAN", 0.22)) and max_prob >= float(getattr(cfg, "COMPONENT_SCORE_TINY_MIN_MAX", 0.40))
        elif size < bins[1]:
            keep = mean_prob >= float(getattr(cfg, "COMPONENT_SCORE_SMALL_MIN_MEAN", 0.16)) or max_prob >= float(getattr(cfg, "COMPONENT_SCORE_SMALL_MIN_MAX", 0.32))
        else:
            keep = size >= int(getattr(cfg, "COMPONENT_SCORE_MIN_SIZE", 24)) or mean_prob >= 0.10 or max_prob >= 0.25
        if keep:
            keep_mask |= comp
    return keep_mask.astype(np.float32)


def apply_postprocessing(
    probs: np.ndarray,
    threshold: float | None,
    min_size: int = 0,
    closing: int = 0,
    hysteresis: tuple[float, float] | None = None,
    use_component_scoring: bool = False,
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
    if use_component_scoring:
        pred_mask = score_connected_components(probs=work, pred_mask=pred_mask, cfg=globals().get("_ACTIVE_CONFIG")).astype(np.float32)
    if min_size and pred_mask.any() and not use_component_scoring:
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
    component_opts=(False, True),
    use_tta=None,
):
    thresholds = thresholds or [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25]
    stats = {}
    patch_size = tuple(cfg.PATCH_SIZE or cfg.INPUT_SHAPE[:-1])
    use_tta = cfg.USE_TTA_FLIPS if use_tta is None else bool(use_tta)
    volume_target_shape = _full_volume_target_shape(cfg)
    for img_p, msk_p in pairs:
        x = _load_and_preprocess_image(str(img_p), volume_target_shape)
        y_true = _load_and_preprocess_mask(str(msk_p), volume_target_shape)
        brain_mask = compute_brain_mask(x)
        active_brain_mask = brain_mask if bool(getattr(cfg, "USE_BRAINMASK_POSTPROC", False)) else None
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
                    for use_components in component_opts:
                        key = f"thr_{t:.2f}_ms{int(min_sz)}_c{int(closing)}_cc{int(use_components)}"
                        pred = apply_postprocessing(
                            probs,
                            threshold=float(t),
                            min_size=min_sz,
                            closing=closing,
                            use_component_scoring=bool(use_components),
                            brain_mask=active_brain_mask,
                            clamp=cfg.OTSU_CLAMP,
                            min_prob=cfg.OTSU_MIN_PROB,
                        )
                        _update_metrics(stats, key, y_true, pred)
        if otsu_thr is not None:
            for min_sz in min_sizes:
                for closing in closing_opts:
                    for use_components in component_opts:
                        key = f"otsu_{otsu_thr:.3f}_ms{int(min_sz)}_c{int(closing)}_cc{int(use_components)}"
                        agg_key = f"otsu_ms{int(min_sz)}_c{int(closing)}_cc{int(use_components)}"
                        pred = apply_postprocessing(
                            probs,
                            threshold=otsu_thr,
                            min_size=min_sz,
                            closing=closing,
                            use_component_scoring=bool(use_components),
                            brain_mask=active_brain_mask,
                            clamp=cfg.OTSU_CLAMP,
                            min_prob=cfg.OTSU_MIN_PROB,
                        )
                        _update_metrics(stats, key, y_true, pred)
                        _update_metrics(stats, agg_key, y_true, pred)
        for (t_low, t_high) in hysteresis_pairs:
            for use_components in component_opts:
                key = f"hyst_{t_low:.2f}_{t_high:.2f}_cc{int(use_components)}"
                pred = apply_postprocessing(
                    probs,
                    threshold=None,
                    min_size=min_sizes[0],
                    closing=0,
                    hysteresis=(t_low, t_high),
                    use_component_scoring=bool(use_components),
                    brain_mask=active_brain_mask,
                    clamp=cfg.OTSU_CLAMP,
                    min_prob=cfg.OTSU_MIN_PROB,
                )
                _update_metrics(stats, key, y_true, pred)
    summary = summarize_metrics(stats)
    best = summary.get("_best")
    if best:
        recipe = {"selection_key": best.get("key")}
        key = str(best.get("key", ""))
        if key.startswith("thr_"):
            parts = key.split("_")
            recipe.update(
                {
                    "mode": "threshold",
                    "threshold": float(parts[1]),
                    "min_size": int(parts[2].replace("ms", "")),
                    "closing": int(parts[3].replace("c", "")),
                    "use_component_scoring": bool(int(parts[4].replace("cc", ""))) if len(parts) > 4 else False,
                }
            )
        elif key.startswith("otsu"):
            parts = key.split("_")
            recipe.update(
                {
                    "mode": "otsu",
                    "min_size": int(parts[-3].replace("ms", "")),
                    "closing": int(parts[-2].replace("c", "")),
                    "use_component_scoring": bool(int(parts[-1].replace("cc", ""))) if parts[-1].startswith("cc") else False,
                }
            )
        elif key.startswith("hyst_"):
            parts = key.split("_")
            recipe.update(
                {
                    "mode": "hysteresis",
                    "t_low": float(parts[1]),
                    "t_high": float(parts[2]),
                    "use_component_scoring": bool(int(parts[3].replace("cc", ""))) if len(parts) > 3 else False,
                }
            )
        recipe["summary"] = {k: v for k, v in best.items() if k != "key"}
        out_path = Path(cfg.CALLBACKS_DIR) / "diagnostics" / "best_postprocessing_recipe.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(recipe, fh, indent=2)
    return summary


def build_model_for_inference(cfg: DynamicTrainingConfig, weights_path: str | None = None) -> tf.keras.Model:
    globals()["_ACTIVE_CONFIG"] = cfg
    with strategy.scope():
        train_model = build_dynamic_model(cfg)
    if weights_path:
        train_model.load_weights(str(weights_path))
    model = tf.keras.Model(train_model.input, train_model.get_layer("probs").output, name="SmartSOTA_SmallLesion_Inference")
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
