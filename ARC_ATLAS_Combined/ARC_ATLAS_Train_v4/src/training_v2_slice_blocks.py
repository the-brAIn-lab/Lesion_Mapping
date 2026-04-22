"""
2.5D full-slice block trainer for ARC/ATLAS T1 lesion segmentation.

This trainer replaces 3D spatial patches with overlapping full-slice slabs. For
each brain, the generator creates every adjacent ``BLOCK_DEPTH`` slice block along
``SLICE_AXIS`` and yields the whole brain as one Keras batch:

    x.shape == (num_slices, full_x, full_y, block_depth)
    y.shape == (num_slices, full_x, full_y, 1)

The network predicts the center slice of each slab. Inference stitches the
center-slice probabilities back into a 3D probability volume and also writes a
thresholded segmentation mask.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import nibabel as nib
import numpy as np
import tensorflow as tf
from scipy.ndimage import zoom


SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parent
LOG_DIR = SCRIPT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("SliceBlockTrainer")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(LOG_DIR / "slice_block_trainer.log")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)


def _env_path(name: str) -> Optional[Path]:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def _default_dir(env_var: str, fallback: Path) -> Path:
    value = os.environ.get(env_var)
    return Path(value).expanduser() if value else fallback


@dataclass(frozen=True)
class CaseRecord:
    source: str
    case_id: str
    image_path: Path
    mask_path: Path


@dataclass
class SliceBlockTrainingConfig:
    DATA_DIR: Optional[Path] = field(default_factory=lambda: _env_path("SMARTSOTA_DATA_DIR"))
    IMAGES_DIR: Optional[Path] = None
    MASKS_DIR: Optional[Path] = None
    MANIFEST_PATH: Optional[Path] = None

    TARGET_SHAPE: tuple[int, int, int] | None = (192, 224, 192)
    RESAMPLE_TO_TARGET: bool = False
    SLICE_AXIS: int = 2
    BLOCK_DEPTH: int = 3
    SLICE_STRIDE: int = 1

    MODEL_DIR: Path = field(default_factory=lambda: _default_dir("SMARTSOTA_MODEL_DIR", PROJECT_ROOT / "models_slice_blocks"))
    CALLBACKS_DIR: Path = field(default_factory=lambda: _default_dir("SMARTSOTA_CALLBACK_DIR", PROJECT_ROOT / "callbacks_slice_blocks"))
    INITIAL_WEIGHTS_PATH: Optional[Path] = None
    timestamp: str = field(default_factory=lambda: time.strftime("%Y%m%d_%H%M%S"), init=False)

    VALIDATION_SPLIT: float = 0.15
    RNG_SEED: int = 1234
    TOTAL_EPOCHS: int = 120
    INITIAL_EPOCH: int = 0
    STEPS_PER_EPOCH: Optional[int] = None
    VALIDATION_STEPS: Optional[int] = None
    BALANCED_CASE_SAMPLING: bool = False
    SOURCE_BALANCED_SAMPLING: bool = True
    SIZE_AWARE_SAMPLING: bool = False
    SIZE_BUCKET_EDGES: tuple[int, ...] = (100, 1000, 10000)
    SIZE_BUCKET_PROBS: tuple[float, ...] = (0.30, 0.30, 0.25, 0.15)

    BASE_FILTERS: int = 8
    UNET_DEPTH: int = 4
    DROPOUT_RATE: float = 0.10
    L2_REG: float = 1e-4
    INITIAL_LR: float = 2e-4
    MIN_LR: float = 1e-6
    WEIGHT_DECAY: float = 1e-5
    MAX_GRAD_NORM: float = 1.0
    MIXED_PRECISION: bool = False
    JIT_COMPILE: bool = False

    POSITIVE_WEIGHT: float = 35.0
    BCE_WEIGHT: float = 0.45
    DICE_WEIGHT: float = 0.45
    FOCAL_TVERSKY_WEIGHT: float = 0.10
    TVERSKY_ALPHA: float = 0.70
    TVERSKY_BETA: float = 0.30
    FOCAL_TVERSKY_GAMMA: float = 1.33
    LESION_SLICE_WEIGHT: float = 1.0
    EMPTY_SLICE_WEIGHT: float = 1.0
    DICE_ON_LESION_SLICES_ONLY: bool = False
    POSITIVE_TOPK_WEIGHT: float = 0.0
    POSITIVE_TOPK_FRACTION: float = 0.25
    SMALL_LESION_BOOST_REFERENCE: float = 0.0
    SMALL_LESION_BOOST_MAX: float = 1.0

    NORMALIZE_NONZERO: bool = True
    NORMALIZE_CLIP_PERCENTILES: tuple[float, float] = (0.5, 99.5)
    AUGMENT: bool = True
    AUG_FLIP_PROB: float = 0.5
    AUG_INTENSITY_SCALE: float = 0.10
    AUG_INTENSITY_SHIFT: float = 0.05
    AUG_NOISE_STD: float = 0.015

    DECISION_THRESHOLD: float = 0.35
    VAL_THRESHOLD_SWEEP: tuple[float, ...] = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)
    WHOLE_BRAIN_VAL_EVERY_N_EPOCHS: int = 1
    WHOLE_BRAIN_VAL_MAX_CASES: Optional[int] = None
    EARLY_STOPPING_PATIENCE: Optional[int] = None
    EARLY_STOPPING_MIN_DELTA: float = 1e-3
    RESTORE_BEST_WEIGHTS: bool = True
    TARGET_WHOLE_DICE: Optional[float] = None
    PREDICT_SLICE_BATCH_SIZE: Optional[int] = None
    SAVE_VAL_PREDICTIONS: bool = True
    NUM_VAL_PREDICTIONS: int = 3
    FIT_VERBOSE: int = 2

    IMAGE_SUFFIXES: tuple[str, ...] = (
        "_T1w_MNI_norm",
        "_T1w_MNI",
        "_T1w_brain",
        "_T1w",
        "_T1",
    )
    MASK_SUFFIXES: tuple[str, ...] = (
        "_lesion_mask_MNI_clean",
        "_lesion_mask_MNI",
        "_lesion_mask",
        "_desc-lesion_mask",
        "_mask",
    )

    def __post_init__(self) -> None:
        if self.DATA_DIR is None:
            raise ValueError("DATA_DIR must be supplied.")
        self.DATA_DIR = Path(self.DATA_DIR).expanduser()
        self.IMAGES_DIR = Path(self.IMAGES_DIR).expanduser() if self.IMAGES_DIR else self.DATA_DIR / "t1"
        self.MASKS_DIR = Path(self.MASKS_DIR).expanduser() if self.MASKS_DIR else self.DATA_DIR / "masks"
        self.MANIFEST_PATH = Path(self.MANIFEST_PATH).expanduser() if self.MANIFEST_PATH else self.DATA_DIR / "manifest.csv"
        self.MODEL_DIR = Path(self.MODEL_DIR).expanduser()
        self.CALLBACKS_DIR = Path(self.CALLBACKS_DIR).expanduser()
        self.INITIAL_WEIGHTS_PATH = Path(self.INITIAL_WEIGHTS_PATH).expanduser() if self.INITIAL_WEIGHTS_PATH else None
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.CALLBACKS_DIR.mkdir(parents=True, exist_ok=True)

        if self.TARGET_SHAPE is not None:
            self.TARGET_SHAPE = tuple(int(v) for v in self.TARGET_SHAPE)
            if len(self.TARGET_SHAPE) != 3:
                raise ValueError("TARGET_SHAPE must be a 3-tuple or None.")
        self.SLICE_AXIS = int(self.SLICE_AXIS)
        if self.SLICE_AXIS not in (0, 1, 2):
            raise ValueError("SLICE_AXIS must be 0, 1, or 2.")
        self.BLOCK_DEPTH = int(self.BLOCK_DEPTH)
        if self.BLOCK_DEPTH < 1 or self.BLOCK_DEPTH % 2 != 1:
            raise ValueError("BLOCK_DEPTH must be an odd positive integer.")
        self.SLICE_STRIDE = max(1, int(self.SLICE_STRIDE))
        self.UNET_DEPTH = max(2, int(self.UNET_DEPTH))
        self.BASE_FILTERS = max(2, int(self.BASE_FILTERS))
        self.WHOLE_BRAIN_VAL_EVERY_N_EPOCHS = max(1, int(self.WHOLE_BRAIN_VAL_EVERY_N_EPOCHS))
        self.NUM_VAL_PREDICTIONS = max(0, int(self.NUM_VAL_PREDICTIONS))
        if self.EARLY_STOPPING_PATIENCE is not None:
            self.EARLY_STOPPING_PATIENCE = max(0, int(self.EARLY_STOPPING_PATIENCE))
        self.EARLY_STOPPING_MIN_DELTA = max(0.0, float(self.EARLY_STOPPING_MIN_DELTA))
        if self.TARGET_WHOLE_DICE is not None:
            self.TARGET_WHOLE_DICE = float(np.clip(self.TARGET_WHOLE_DICE, 0.0, 1.0))
        self.LESION_SLICE_WEIGHT = max(0.0, float(self.LESION_SLICE_WEIGHT))
        self.EMPTY_SLICE_WEIGHT = max(0.0, float(self.EMPTY_SLICE_WEIGHT))
        self.POSITIVE_TOPK_WEIGHT = max(0.0, float(self.POSITIVE_TOPK_WEIGHT))
        self.POSITIVE_TOPK_FRACTION = float(np.clip(self.POSITIVE_TOPK_FRACTION, 0.0, 1.0))
        self.SMALL_LESION_BOOST_REFERENCE = max(0.0, float(self.SMALL_LESION_BOOST_REFERENCE))
        self.SMALL_LESION_BOOST_MAX = max(1.0, float(self.SMALL_LESION_BOOST_MAX))
        self.SIZE_BUCKET_EDGES = tuple(int(v) for v in self.SIZE_BUCKET_EDGES)
        self.SIZE_BUCKET_PROBS = tuple(float(v) for v in self.SIZE_BUCKET_PROBS)
        if len(self.SIZE_BUCKET_PROBS) != len(self.SIZE_BUCKET_EDGES) + 1:
            raise ValueError("SIZE_BUCKET_PROBS must have exactly len(SIZE_BUCKET_EDGES) + 1 values.")
        prob_sum = float(np.sum(self.SIZE_BUCKET_PROBS))
        if prob_sum <= 0:
            raise ValueError("SIZE_BUCKET_PROBS must sum to a positive value.")
        self.SIZE_BUCKET_PROBS = tuple(float(v) / prob_sum for v in self.SIZE_BUCKET_PROBS)
        self.VAL_THRESHOLD_SWEEP = tuple(
            sorted({float(np.clip(t, 0.0, 1.0)) for t in self.VAL_THRESHOLD_SWEEP + (self.DECISION_THRESHOLD,)})
        )
        self._write_config()

    @property
    def in_plane_shape(self) -> tuple[int, int]:
        if self.TARGET_SHAPE is None:
            raise ValueError("in_plane_shape is only defined when TARGET_SHAPE is fixed.")
        return tuple(self.TARGET_SHAPE[i] for i in range(3) if i != self.SLICE_AXIS)

    @property
    def input_shape(self) -> tuple[int, int, int]:
        h, w = self.in_plane_shape
        return h, w, self.BLOCK_DEPTH

    @property
    def checkpoint_path(self) -> Path:
        return self.CALLBACKS_DIR / "best_slice_block.weights.h5"

    @property
    def latest_path(self) -> Path:
        return self.CALLBACKS_DIR / "latest_slice_block.weights.h5"

    def _write_config(self) -> None:
        payload = asdict(self)
        for key in ("DATA_DIR", "IMAGES_DIR", "MASKS_DIR", "MANIFEST_PATH", "MODEL_DIR", "CALLBACKS_DIR", "INITIAL_WEIGHTS_PATH"):
            payload[key] = str(payload[key]) if payload.get(key) is not None else None
        for key in (
            "TARGET_SHAPE",
            "NORMALIZE_CLIP_PERCENTILES",
            "SIZE_BUCKET_EDGES",
            "SIZE_BUCKET_PROBS",
            "VAL_THRESHOLD_SWEEP",
            "IMAGE_SUFFIXES",
            "MASK_SUFFIXES",
        ):
            if payload.get(key) is not None:
                payload[key] = list(payload[key])
        with (self.MODEL_DIR / "config.json").open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)


def configure_runtime(cfg: SliceBlockTrainingConfig) -> None:
    tf.keras.utils.set_random_seed(int(cfg.RNG_SEED))
    random.seed(int(cfg.RNG_SEED))
    np.random.seed(int(cfg.RNG_SEED))
    if cfg.MIXED_PRECISION:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    else:
        tf.keras.mixed_precision.set_global_policy("float32")
    tf.config.optimizer.set_jit("autoclustering" if cfg.JIT_COMPILE else False)
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as exc:
            logger.warning("Could not set memory growth on %s: %s", gpu, exc)


def _resolve_manifest_path(raw_value: str, cfg: SliceBlockTrainingConfig) -> Optional[Path]:
    raw = (raw_value or "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    if p.is_absolute() and p.exists():
        return p
    candidates = [cfg.DATA_DIR / p, cfg.IMAGES_DIR / p, cfg.MASKS_DIR / p]
    if cfg.MANIFEST_PATH is not None:
        candidates.extend(parent / p for parent in cfg.MANIFEST_PATH.parents)
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return None


def _case_id_from_path(source: str, path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    return f"{source}__{name}" if source else name


def load_cases(cfg: SliceBlockTrainingConfig) -> list[CaseRecord]:
    cases: list[CaseRecord] = []
    if cfg.MANIFEST_PATH and cfg.MANIFEST_PATH.exists():
        logger.info("Using manifest: %s", cfg.MANIFEST_PATH)
        with cfg.MANIFEST_PATH.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                image = _resolve_manifest_path(row.get("t1", ""), cfg)
                mask = _resolve_manifest_path(row.get("mask", ""), cfg)
                if image is None or mask is None:
                    continue
                source = (row.get("slug") or "").strip()
                cases.append(CaseRecord(source, _case_id_from_path(source, image), image, mask))
        if cases:
            logger.info("Loaded %d cases from manifest", len(cases))
            return cases

    images = sorted(p for p in cfg.IMAGES_DIR.glob("*.nii*") if p.is_file())
    masks = sorted(p for p in cfg.MASKS_DIR.glob("*.nii*") if p.is_file())
    mask_by_stem = {p.name: p for p in masks}
    for image in images:
        image_name = image.name
        stem = image_name[:-7] if image_name.endswith(".nii.gz") else image.stem
        mask = None
        for img_suf in cfg.IMAGE_SUFFIXES:
            if stem.endswith(img_suf):
                base = stem[: -len(img_suf)]
                for mask_suf in cfg.MASK_SUFFIXES:
                    for ext in (".nii.gz", ".nii"):
                        candidate = mask_by_stem.get(base + mask_suf + ext)
                        if candidate is not None:
                            mask = candidate
                            break
                    if mask is not None:
                        break
            if mask is not None:
                break
        if mask is not None:
            cases.append(CaseRecord("", _case_id_from_path("", image), image, mask))
    logger.info("Loaded %d paired cases from folders", len(cases))
    return cases


def _load_vol_canonical(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.as_closest_canonical(nib.load(str(path)))
    data = img.get_fdata(dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data.astype(np.float32, copy=False), img


def _center_crop_or_pad(volume: np.ndarray, target_shape: tuple[int, int, int], pad_value: float = 0.0) -> np.ndarray:
    out = np.asarray(volume)
    slices = []
    pads = []
    for dim, target in zip(out.shape, target_shape):
        target = int(target)
        if dim > target:
            start = (dim - target) // 2
            slices.append(slice(start, start + target))
            pads.append((0, 0))
        else:
            slices.append(slice(0, dim))
            before = (target - dim) // 2
            pads.append((before, target - dim - before))
    out = out[tuple(slices)]
    if any(before or after for before, after in pads):
        out = np.pad(out, pads, mode="constant", constant_values=pad_value)
    return out.astype(np.float32, copy=False)


def _resample_to_shape(volume: np.ndarray, target_shape: tuple[int, int, int], order: int) -> np.ndarray:
    if tuple(volume.shape) == tuple(target_shape):
        return volume.astype(np.float32, copy=False)
    factors = [target / max(current, 1) for current, target in zip(volume.shape, target_shape)]
    return zoom(volume, factors, order=order).astype(np.float32, copy=False)


def _prepare_volume(
    volume: np.ndarray,
    cfg: SliceBlockTrainingConfig,
    is_mask: bool = False,
) -> np.ndarray:
    if cfg.TARGET_SHAPE is None:
        out = volume.astype(np.float32, copy=False)
    elif cfg.RESAMPLE_TO_TARGET:
        out = _resample_to_shape(volume, cfg.TARGET_SHAPE, order=0 if is_mask else 1)
        if tuple(out.shape) != tuple(cfg.TARGET_SHAPE):
            out = _center_crop_or_pad(out, cfg.TARGET_SHAPE)
    else:
        out = _center_crop_or_pad(volume, cfg.TARGET_SHAPE)
    if is_mask:
        return (out > 0.5).astype(np.float32, copy=False)
    return normalize_image(out, cfg)


def normalize_image(volume: np.ndarray, cfg: SliceBlockTrainingConfig) -> np.ndarray:
    vol = np.asarray(volume, dtype=np.float32)
    finite = np.isfinite(vol)
    if cfg.NORMALIZE_NONZERO:
        sample_mask = finite & (np.abs(vol) > 1e-6)
    else:
        sample_mask = finite
    if not np.any(sample_mask):
        return np.zeros_like(vol, dtype=np.float32)
    vals = vol[sample_mask]
    lo, hi = np.percentile(vals, cfg.NORMALIZE_CLIP_PERCENTILES)
    clipped = np.clip(vol, lo, hi)
    vals = clipped[sample_mask]
    mean = float(np.mean(vals))
    std = float(np.std(vals))
    if std < 1e-6:
        std = 1.0
    normed = (clipped - mean) / std
    normed[~finite] = 0.0
    if cfg.NORMALIZE_NONZERO:
        normed[~sample_mask] = 0.0
    return normed.astype(np.float32, copy=False)


def load_case_arrays(
    case: CaseRecord,
    cfg: SliceBlockTrainingConfig,
) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Image]:
    image, ref_img = _load_vol_canonical(case.image_path)
    mask, _ = _load_vol_canonical(case.mask_path)
    image = _prepare_volume(image, cfg, is_mask=False)
    mask = _prepare_volume(mask, cfg, is_mask=True)
    return image, mask, ref_img


def make_slice_blocks(
    image: np.ndarray,
    mask: np.ndarray | None,
    cfg: SliceBlockTrainingConfig,
) -> tuple[np.ndarray, np.ndarray | None]:
    moved = np.moveaxis(np.asarray(image, dtype=np.float32), cfg.SLICE_AXIS, -1)
    n_slices = moved.shape[-1]
    radius = cfg.BLOCK_DEPTH // 2
    padded = np.pad(moved, [(0, 0), (0, 0), (radius, radius)], mode="edge")
    centers = range(0, n_slices, cfg.SLICE_STRIDE)
    blocks = np.stack([padded[..., center : center + cfg.BLOCK_DEPTH] for center in centers], axis=0)
    y = None
    if mask is not None:
        moved_mask = np.moveaxis(np.asarray(mask, dtype=np.float32), cfg.SLICE_AXIS, -1)
        y = np.stack([moved_mask[..., center] for center in centers], axis=0)[..., np.newaxis]
        y = (y > 0.5).astype(np.float32, copy=False)
    return blocks.astype(np.float32, copy=False), y


def stitch_probability_slices(
    pred_slices: np.ndarray,
    volume_shape: tuple[int, int, int],
    cfg: SliceBlockTrainingConfig,
) -> np.ndarray:
    pred = np.asarray(pred_slices, dtype=np.float32)
    if pred.ndim == 4:
        pred = pred[..., 0]
    moved_shape = tuple(volume_shape[i] for i in range(3) if i != cfg.SLICE_AXIS) + (volume_shape[cfg.SLICE_AXIS],)
    if cfg.SLICE_STRIDE != 1:
        full = np.zeros(moved_shape, dtype=np.float32)
        counts = np.zeros(moved_shape[-1], dtype=np.float32)
        for idx, center in enumerate(range(0, moved_shape[-1], cfg.SLICE_STRIDE)):
            full[..., center] = pred[idx]
            counts[center] = 1.0
        missing = np.where(counts == 0)[0]
        for center in missing:
            nearest = int(round(center / cfg.SLICE_STRIDE) * cfg.SLICE_STRIDE)
            nearest = min(max(nearest, 0), moved_shape[-1] - 1)
            source_idx = nearest // cfg.SLICE_STRIDE
            full[..., center] = pred[source_idx]
        moved = full
    else:
        moved = np.stack([pred[i] for i in range(pred.shape[0])], axis=-1)
    return np.moveaxis(moved, -1, cfg.SLICE_AXIS).astype(np.float32, copy=False)


def soft_dice_np(y_true: np.ndarray, y_prob: np.ndarray, smooth: float = 1e-6) -> float:
    y = (np.asarray(y_true) > 0.5).astype(np.float32)
    p = np.asarray(y_prob, dtype=np.float32)
    return float((2.0 * np.sum(y * p) + smooth) / (np.sum(y) + np.sum(p) + smooth))


def hard_dice_np(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-6) -> float:
    y = (np.asarray(y_true) > 0.5).astype(np.float32)
    p = (np.asarray(y_pred) > 0.5).astype(np.float32)
    return float((2.0 * np.sum(y * p) + smooth) / (np.sum(y) + np.sum(p) + smooth))


def dice_coefficient(y_true, y_pred, smooth: float = 1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    denom = tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1)
    return tf.reduce_mean((2.0 * intersection + smooth) / (denom + smooth))


def hard_dice_metric(y_true, y_pred, smooth: float = 1e-6):
    return dice_coefficient(y_true, tf.cast(y_pred >= 0.5, tf.float32), smooth=smooth)


def foreground_fraction(y_true, y_pred):
    del y_true
    return tf.reduce_mean(tf.cast(y_pred >= 0.5, tf.float32))


class WeightedBceDiceTversky(tf.keras.losses.Loss):
    def __init__(
        self,
        positive_weight: float,
        bce_weight: float,
        dice_weight: float,
        focal_tversky_weight: float,
        tversky_alpha: float,
        tversky_beta: float,
        focal_tversky_gamma: float,
        lesion_slice_weight: float = 1.0,
        empty_slice_weight: float = 1.0,
        dice_on_lesion_slices_only: bool = False,
        positive_topk_weight: float = 0.0,
        positive_topk_fraction: float = 0.25,
        small_lesion_boost_reference: float = 0.0,
        small_lesion_boost_max: float = 1.0,
        name: str = "weighted_bce_dice_tversky",
    ):
        super().__init__(name=name)
        self.positive_weight = float(positive_weight)
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.focal_tversky_weight = float(focal_tversky_weight)
        self.tversky_alpha = float(tversky_alpha)
        self.tversky_beta = float(tversky_beta)
        self.focal_tversky_gamma = float(focal_tversky_gamma)
        self.lesion_slice_weight = float(lesion_slice_weight)
        self.empty_slice_weight = float(empty_slice_weight)
        self.dice_on_lesion_slices_only = bool(dice_on_lesion_slices_only)
        self.positive_topk_weight = float(positive_topk_weight)
        self.positive_topk_fraction = float(positive_topk_fraction)
        self.small_lesion_boost_reference = float(small_lesion_boost_reference)
        self.small_lesion_boost_max = float(small_lesion_boost_max)

    def get_config(self):
        return {
            "positive_weight": self.positive_weight,
            "bce_weight": self.bce_weight,
            "dice_weight": self.dice_weight,
            "focal_tversky_weight": self.focal_tversky_weight,
            "tversky_alpha": self.tversky_alpha,
            "tversky_beta": self.tversky_beta,
            "focal_tversky_gamma": self.focal_tversky_gamma,
            "lesion_slice_weight": self.lesion_slice_weight,
            "empty_slice_weight": self.empty_slice_weight,
            "dice_on_lesion_slices_only": self.dice_on_lesion_slices_only,
            "positive_topk_weight": self.positive_topk_weight,
            "positive_topk_fraction": self.positive_topk_fraction,
            "small_lesion_boost_reference": self.small_lesion_boost_reference,
            "small_lesion_boost_max": self.small_lesion_boost_max,
            "name": self.name,
        }

    @staticmethod
    def _weighted_mean(values, weights):
        values = tf.cast(values, tf.float32)
        weights = tf.cast(weights, tf.float32)
        return tf.math.divide_no_nan(tf.reduce_sum(values * weights), tf.reduce_sum(weights))

    def _positive_topk_loss(self, y_true, y_pred):
        if self.positive_topk_weight <= 0.0 or self.positive_topk_fraction <= 0.0:
            return tf.constant(0.0, dtype=tf.float32)
        positive = tf.cast(y_true > 0.5, tf.float32)
        focus_power = max(1.0, 1.0 / max(self.positive_topk_fraction, 1e-3))
        missed_positive = tf.pow(1.0 - y_pred, focus_power)
        return tf.math.divide_no_nan(tf.reduce_sum(positive * missed_positive), tf.reduce_sum(positive))

    def _case_lesion_boost(self, y_true):
        if self.small_lesion_boost_reference <= 0.0 or self.small_lesion_boost_max <= 1.0:
            return tf.constant(1.0, dtype=tf.float32)
        total_positive = tf.reduce_sum(tf.cast(y_true > 0.5, tf.float32))
        reference = tf.cast(self.small_lesion_boost_reference, tf.float32)
        boost = tf.sqrt(reference / tf.maximum(total_positive, 1.0))
        boost = tf.clip_by_value(boost, 1.0, tf.cast(self.small_lesion_boost_max, tf.float32))
        return tf.where(total_positive > 0.5, boost, tf.constant(1.0, dtype=tf.float32))

    def call(self, y_true, y_pred):
        eps = tf.keras.backend.epsilon()
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), eps, 1.0 - eps)
        slice_has_lesion = tf.reduce_sum(y_true, axis=[1, 2, 3]) > 0.5
        lesion_boost = self._case_lesion_boost(y_true)
        lesion_slice_weight = tf.cast(self.lesion_slice_weight, tf.float32) * lesion_boost
        slice_weights = tf.where(
            slice_has_lesion,
            tf.fill(tf.shape(slice_has_lesion), lesion_slice_weight),
            tf.fill(tf.shape(slice_has_lesion), tf.cast(self.empty_slice_weight, tf.float32)),
        )
        weights = 1.0 + y_true * (self.positive_weight - 1.0)
        bce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        bce_per_slice = tf.reduce_mean(weights * bce, axis=[1, 2, 3])
        bce = self._weighted_mean(bce_per_slice, slice_weights)

        y = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
        p = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
        tp = tf.reduce_sum(y * p, axis=1)
        fp = tf.reduce_sum((1.0 - y) * p, axis=1)
        fn = tf.reduce_sum(y * (1.0 - p), axis=1)
        y_sum = tf.reduce_sum(y, axis=1)
        p_sum = tf.reduce_sum(p, axis=1)
        dice = (2.0 * tp + eps) / (y_sum + p_sum + eps)
        dice_loss_per_slice = 1.0 - dice
        dice_weights = tf.where(
            slice_has_lesion,
            tf.fill(tf.shape(slice_has_lesion), lesion_slice_weight),
            (
                tf.zeros(tf.shape(slice_has_lesion), dtype=tf.float32)
                if self.dice_on_lesion_slices_only
                else tf.fill(tf.shape(slice_has_lesion), tf.cast(self.empty_slice_weight, tf.float32))
            ),
        )
        dice_loss = self._weighted_mean(dice_loss_per_slice, dice_weights)
        tversky = (tp + eps) / (tp + self.tversky_alpha * fp + self.tversky_beta * fn + eps)
        focal_tversky = self._weighted_mean(tf.pow(1.0 - tversky, self.focal_tversky_gamma), dice_weights)
        positive_topk = lesion_boost * self._positive_topk_loss(y_true, y_pred)
        return (
            self.bce_weight * bce
            + self.dice_weight * dice_loss
            + self.focal_tversky_weight * focal_tversky
            + self.positive_topk_weight * positive_topk
        )


def _conv_block(x, filters: int, cfg: SliceBlockTrainingConfig, name: str):
    reg = tf.keras.regularizers.l2(float(cfg.L2_REG)) if cfg.L2_REG else None
    for idx in range(2):
        x = tf.keras.layers.Conv2D(
            filters,
            3,
            padding="same",
            use_bias=False,
            kernel_regularizer=reg,
            name=f"{name}_conv{idx + 1}",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"{name}_bn{idx + 1}")(x)
        x = tf.keras.layers.Activation("swish", name=f"{name}_swish{idx + 1}")(x)
    if cfg.DROPOUT_RATE > 0:
        x = tf.keras.layers.SpatialDropout2D(float(cfg.DROPOUT_RATE), name=f"{name}_dropout")(x)
    return x


def build_slice_block_model(cfg: SliceBlockTrainingConfig) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=cfg.input_shape, name="slice_block")
    skips = []
    x = inputs
    filters = int(cfg.BASE_FILTERS)
    for depth in range(int(cfg.UNET_DEPTH)):
        x = _conv_block(x, filters, cfg, name=f"enc{depth + 1}")
        skips.append(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2, name=f"pool{depth + 1}")(x)
        filters *= 2
    x = _conv_block(x, filters, cfg, name="bottleneck")
    for depth, skip in reversed(list(enumerate(skips))):
        filters //= 2
        x = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear", name=f"up{depth + 1}")(x)
        x = tf.keras.layers.Concatenate(name=f"skip{depth + 1}")([x, skip])
        x = _conv_block(x, filters, cfg, name=f"dec{depth + 1}")
    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid", dtype="float32", name="probability")(x)
    return tf.keras.Model(inputs, outputs, name="SliceBlock2p5D_UNet")


def build_slice_block_inference_model(
    cfg: SliceBlockTrainingConfig,
    weights_path: str | Path | None = None,
) -> tf.keras.Model:
    model = build_slice_block_model(cfg)
    if weights_path is not None:
        model.load_weights(str(weights_path))
    probability = model.output
    segmentation = tf.keras.layers.Lambda(
        lambda p: tf.cast(p >= float(cfg.DECISION_THRESHOLD), tf.float32),
        name="segmentation",
    )(probability)
    return tf.keras.Model(model.input, {"probability": probability, "segmentation": segmentation}, name="SliceBlock2p5D_Inference")


def augment_case_batch(x: np.ndarray, y: np.ndarray, cfg: SliceBlockTrainingConfig) -> tuple[np.ndarray, np.ndarray]:
    if not cfg.AUGMENT:
        return x, y
    out_x = x
    out_y = y
    if random.random() < cfg.AUG_FLIP_PROB:
        out_x = np.flip(out_x, axis=1)
        out_y = np.flip(out_y, axis=1)
    if random.random() < cfg.AUG_FLIP_PROB:
        out_x = np.flip(out_x, axis=2)
        out_y = np.flip(out_y, axis=2)
    scale = 1.0 + random.uniform(-cfg.AUG_INTENSITY_SCALE, cfg.AUG_INTENSITY_SCALE)
    shift = random.uniform(-cfg.AUG_INTENSITY_SHIFT, cfg.AUG_INTENSITY_SHIFT)
    out_x = out_x * scale + shift
    if cfg.AUG_NOISE_STD > 0:
        out_x = out_x + np.random.normal(0.0, cfg.AUG_NOISE_STD, size=out_x.shape).astype(np.float32)
    return out_x.astype(np.float32, copy=False), out_y.astype(np.float32, copy=False)


def _dataset_signature(cfg: SliceBlockTrainingConfig):
    h, w, c = cfg.input_shape
    return (
        tf.TensorSpec(shape=(None, h, w, c), dtype=tf.float32),
        tf.TensorSpec(shape=(None, h, w, 1), dtype=tf.float32),
    )


def _case_sampling_weights(
    cases: Sequence[CaseRecord],
    lesion_sizes: Sequence[int] | None,
    cfg: SliceBlockTrainingConfig,
) -> np.ndarray:
    n_cases = len(cases)
    if n_cases == 0:
        return np.asarray([], dtype=np.float64)
    weights = np.ones(n_cases, dtype=np.float64)
    if cfg.SIZE_AWARE_SAMPLING and lesion_sizes is not None and len(lesion_sizes) == n_cases:
        sizes = np.asarray(lesion_sizes, dtype=np.int64)
        bins = np.digitize(sizes, np.asarray(cfg.SIZE_BUCKET_EDGES, dtype=np.int64), right=False)
        weights = np.zeros(n_cases, dtype=np.float64)
        probs = np.asarray(cfg.SIZE_BUCKET_PROBS, dtype=np.float64)
        for bucket_idx, bucket_prob in enumerate(probs):
            idx = np.where(bins == bucket_idx)[0]
            if idx.size:
                weights[idx] = float(bucket_prob) / float(idx.size)
        if not np.any(weights > 0):
            weights = np.ones(n_cases, dtype=np.float64)

    if cfg.SOURCE_BALANCED_SAMPLING:
        source_labels = np.asarray([case.source for case in cases], dtype=object)
        sources = np.unique(source_labels)
        balanced = np.zeros(n_cases, dtype=np.float64)
        per_source_mass = 1.0 / max(len(sources), 1)
        for source in sources:
            idx = np.where(source_labels == source)[0]
            if idx.size == 0:
                continue
            local = np.clip(weights[idx], 1e-12, None)
            balanced[idx] = per_source_mass * (local / local.sum())
        weights = balanced

    weights = np.clip(weights, 1e-12, None)
    return weights / weights.sum()


def write_sampling_diagnostics(
    cases: Sequence[CaseRecord],
    lesion_sizes: Sequence[int] | None,
    weights: np.ndarray,
    cfg: SliceBlockTrainingConfig,
) -> None:
    if not len(cases) or weights.size != len(cases):
        return
    diag_dir = cfg.CALLBACKS_DIR / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    with (diag_dir / "train_sampling_weights.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["source", "case_id", "lesion_voxels", "lesion_group", "sampling_weight", "image", "mask"],
        )
        writer.writeheader()
        for idx, case in enumerate(cases):
            size = int(lesion_sizes[idx]) if lesion_sizes is not None and len(lesion_sizes) > idx else 0
            writer.writerow(
                {
                    "source": case.source,
                    "case_id": case.case_id,
                    "lesion_voxels": size,
                    "lesion_group": lesion_size_group(size),
                    "sampling_weight": float(weights[idx]),
                    "image": str(case.image_path),
                    "mask": str(case.mask_path),
                }
            )


def case_batch_generator(
    cases: list[CaseRecord],
    cfg: SliceBlockTrainingConfig,
    training: bool,
    lesion_sizes: Sequence[int] | None = None,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    if not cases:
        raise ValueError("Cannot build dataset from an empty case list.")
    rng = random.Random(int(cfg.RNG_SEED) + (1 if training else 10_000))
    np_rng = np.random.default_rng(int(cfg.RNG_SEED) + (101 if training else 10_100))
    sampling_weights = _case_sampling_weights(cases, lesion_sizes, cfg) if training and cfg.BALANCED_CASE_SAMPLING else None
    while True:
        if training and sampling_weights is not None:
            order = [cases[int(np_rng.choice(len(cases), p=sampling_weights))] for _ in range(len(cases))]
        else:
            order = list(cases)
        if training and sampling_weights is None:
            rng.shuffle(order)
        for case in order:
            image, mask, _ = load_case_arrays(case, cfg)
            x, y = make_slice_blocks(image, mask, cfg)
            assert y is not None
            if training:
                x, y = augment_case_batch(x, y, cfg)
            yield x, y


def make_tf_dataset(
    cases: list[CaseRecord],
    cfg: SliceBlockTrainingConfig,
    training: bool,
    lesion_sizes: Sequence[int] | None = None,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_generator(
        lambda: case_batch_generator(cases, cfg, training=training, lesion_sizes=lesion_sizes),
        output_signature=_dataset_signature(cfg),
    )
    return ds.prefetch(1)


def lesion_size_group(size: int) -> str:
    if size <= 0:
        return "none"
    if size < 100:
        return "1_99"
    if size < 1000:
        return "100_999"
    if size < 10000:
        return "1000_9999"
    return "10000_plus"


def compute_lesion_sizes(cases: list[CaseRecord], cfg: SliceBlockTrainingConfig) -> list[int]:
    sizes = []
    for case in cases:
        mask, _ = _load_vol_canonical(case.mask_path)
        mask = _prepare_volume(mask, cfg, is_mask=True)
        sizes.append(int(np.sum(mask > 0.5)))
    return sizes


def split_cases(
    cases: list[CaseRecord],
    cfg: SliceBlockTrainingConfig,
) -> tuple[list[CaseRecord], list[CaseRecord], list[int]]:
    lesion_sizes = compute_lesion_sizes(cases, cfg)
    buckets: dict[tuple[str, str], list[int]] = {}
    for idx, (case, size) in enumerate(zip(cases, lesion_sizes)):
        buckets.setdefault((case.source, lesion_size_group(size)), []).append(idx)
    rng = random.Random(int(cfg.RNG_SEED))
    val_indices: set[int] = set()
    for indices in buckets.values():
        rng.shuffle(indices)
        if len(indices) <= 1:
            continue
        n_val = int(round(len(indices) * float(cfg.VALIDATION_SPLIT)))
        n_val = min(max(1, n_val), len(indices) - 1)
        val_indices.update(indices[:n_val])
    if not val_indices:
        all_indices = list(range(len(cases)))
        rng.shuffle(all_indices)
        n_val = max(1, int(round(len(all_indices) * float(cfg.VALIDATION_SPLIT))))
        val_indices.update(all_indices[:n_val])
    train_cases = [case for idx, case in enumerate(cases) if idx not in val_indices]
    val_cases = [case for idx, case in enumerate(cases) if idx in val_indices]
    logger.info("Split cases: train=%d val=%d", len(train_cases), len(val_cases))
    return train_cases, val_cases, lesion_sizes


def _counts(values: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return out


def write_split_diagnostics(
    train_cases: list[CaseRecord],
    val_cases: list[CaseRecord],
    all_cases: list[CaseRecord],
    lesion_sizes: list[int],
    cfg: SliceBlockTrainingConfig,
) -> None:
    size_by_id = {case.case_id: size for case, size in zip(all_cases, lesion_sizes)}
    diag_dir = cfg.CALLBACKS_DIR / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    with (diag_dir / "split_cases.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["split", "source", "case_id", "lesion_voxels", "lesion_group", "image", "mask"])
        writer.writeheader()
        for split_name, split_cases in (("train", train_cases), ("val", val_cases)):
            for case in split_cases:
                size = int(size_by_id.get(case.case_id, 0))
                writer.writerow(
                    {
                        "split": split_name,
                        "source": case.source,
                        "case_id": case.case_id,
                        "lesion_voxels": size,
                        "lesion_group": lesion_size_group(size),
                        "image": str(case.image_path),
                        "mask": str(case.mask_path),
                    }
                )
    summary = {}
    for name, split_cases in (("train", train_cases), ("val", val_cases)):
        sizes = [int(size_by_id.get(case.case_id, 0)) for case in split_cases]
        summary[name] = {
            "count": len(split_cases),
            "source_counts": _counts(case.source for case in split_cases),
            "lesion_group_counts": _counts(lesion_size_group(size) for size in sizes),
            "lesion_voxels": {
                "mean": float(np.mean(sizes)) if sizes else 0.0,
                "median": float(np.median(sizes)) if sizes else 0.0,
                "p90": float(np.percentile(sizes, 90)) if sizes else 0.0,
                "max": int(max(sizes)) if sizes else 0,
            },
        }
    with (diag_dir / "split_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


def predict_case_probability_map(
    model: tf.keras.Model,
    case: CaseRecord,
    cfg: SliceBlockTrainingConfig,
) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Image]:
    image, mask, ref_img = load_case_arrays(case, cfg)
    blocks, _ = make_slice_blocks(image, None, cfg)
    batch_size = int(cfg.PREDICT_SLICE_BATCH_SIZE or blocks.shape[0])
    pred = model.predict(blocks, batch_size=batch_size, verbose=0)
    if isinstance(pred, dict):
        pred = pred["probability"]
    prob = stitch_probability_slices(pred, mask.shape, cfg)
    return prob, mask, ref_img


def save_case_outputs(
    out_dir: Path,
    case: CaseRecord,
    prob: np.ndarray,
    threshold: float,
    reference_img: nib.Nifti1Image,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_id = case.case_id.replace("/", "_")
    affine = reference_img.affine
    header = reference_img.header.copy()
    prob_img = nib.Nifti1Image(prob.astype(np.float32), affine=affine, header=header)
    seg_img = nib.Nifti1Image((prob >= threshold).astype(np.uint8), affine=affine, header=header)
    nib.save(prob_img, str(out_dir / f"{safe_id}_probability.nii.gz"))
    nib.save(seg_img, str(out_dir / f"{safe_id}_seg_thr_{threshold:.2f}.nii.gz"))


def load_slice_block_model(
    cfg: SliceBlockTrainingConfig,
    weights_path: str | Path,
) -> tf.keras.Model:
    model = build_slice_block_model(cfg)
    model.load_weights(str(weights_path))
    return model


def predict_ensemble_probability_map(
    models: Sequence[tf.keras.Model],
    configs: Sequence[SliceBlockTrainingConfig],
    case: CaseRecord,
    weights: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Image]:
    if len(models) != len(configs):
        raise ValueError("models and configs must have the same length.")
    if not models:
        raise ValueError("At least one model is required for ensemble prediction.")
    if weights is None:
        weights_arr = np.ones(len(models), dtype=np.float32) / float(len(models))
    else:
        weights_arr = np.asarray(weights, dtype=np.float32)
        if weights_arr.shape[0] != len(models):
            raise ValueError("weights must have one value per model.")
        total = float(np.sum(weights_arr))
        if total <= 0:
            raise ValueError("Ensemble weights must sum to a positive value.")
        weights_arr = weights_arr / total

    probs = []
    reference_mask = None
    reference_img = None
    for model, cfg, weight in zip(models, configs, weights_arr):
        prob, mask, ref_img = predict_case_probability_map(model, case, cfg)
        probs.append(prob.astype(np.float32, copy=False) * float(weight))
        if reference_mask is None:
            reference_mask = mask
            reference_img = ref_img
    ensemble_prob = np.sum(np.stack(probs, axis=0), axis=0)
    assert reference_mask is not None and reference_img is not None
    return ensemble_prob.astype(np.float32, copy=False), reference_mask, reference_img


def _row_group_mean(rows: Sequence[dict], group_key: str, metric_key: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(group_key, "")), []).append(float(row[metric_key]))
    return {key: float(np.mean(values)) for key, values in sorted(grouped.items())}


def evaluate_slice_block_ensemble(
    configs: Sequence[SliceBlockTrainingConfig],
    weights_paths: Sequence[str | Path],
    cases: Sequence[CaseRecord] | None = None,
    out_dir: str | Path | None = None,
    thresholds: Sequence[float] | None = None,
    decision_threshold: float | None = None,
    max_cases: int | None = None,
    save_predictions: int = 3,
    ensemble_weights: Sequence[float] | None = None,
) -> dict:
    if len(configs) != len(weights_paths):
        raise ValueError("configs and weights_paths must have the same length.")
    if not configs:
        raise ValueError("At least one config is required.")
    cfg0 = configs[0]
    if cases is None:
        all_cases = load_cases(cfg0)
        _, val_cases, _ = split_cases(all_cases, cfg0)
        cases = val_cases
    cases = list(cases)
    if max_cases is not None:
        cases = cases[: int(max_cases)]
    if thresholds is None:
        thresholds = cfg0.VAL_THRESHOLD_SWEEP
    decision_threshold = float(cfg0.DECISION_THRESHOLD if decision_threshold is None else decision_threshold)
    thresholds = tuple(sorted({float(np.clip(t, 0.0, 1.0)) for t in tuple(thresholds) + (decision_threshold,)}))
    out_path = Path(out_dir) if out_dir is not None else cfg0.CALLBACKS_DIR / "ensemble"
    out_path.mkdir(parents=True, exist_ok=True)

    models = [load_slice_block_model(cfg, weights_path) for cfg, weights_path in zip(configs, weights_paths)]
    rows = []
    start = time.time()
    for idx, case in enumerate(cases, start=1):
        prob, mask, ref_img = predict_ensemble_probability_map(models, configs, case, weights=ensemble_weights)
        true_voxels = int(np.sum(mask > 0.5))
        sweep = {f"{thr:.2f}": hard_dice_np(mask, prob >= thr) for thr in thresholds}
        best_thr_key, best_thr_score = max(sweep.items(), key=lambda item: item[1])
        row = {
            "source": case.source,
            "case_id": case.case_id,
            "soft_dice": soft_dice_np(mask, prob),
            "hard_dice": hard_dice_np(mask, prob >= decision_threshold),
            "best_threshold": float(best_thr_key),
            "best_threshold_dice": float(best_thr_score),
            "true_voxels": true_voxels,
            "pred_soft_voxels": float(np.sum(prob)),
            "pred_hard_voxels": int(np.sum(prob >= decision_threshold)),
            "pred_max": float(np.max(prob)),
            "pred_mean": float(np.mean(prob)),
            "lesion_group": lesion_size_group(true_voxels),
            "image": str(case.image_path),
            "mask": str(case.mask_path),
        }
        row.update({f"dice_thr_{key}": value for key, value in sweep.items()})
        rows.append(row)
        if idx % 16 == 0 or idx == len(cases):
            logger.info("Slice-block ensemble val progress: %d/%d", idx, len(cases))
        if save_predictions > 0 and idx <= int(save_predictions):
            save_case_outputs(out_path / "predictions", case, prob, decision_threshold, ref_img)

    if rows:
        csv_path = out_path / "ensemble_val.csv"
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "elapsed_sec": float(time.time() - start),
        "n_cases": len(rows),
        "weights_paths": [str(p) for p in weights_paths],
        "axes": [int(cfg.SLICE_AXIS) for cfg in configs],
        "block_depths": [int(cfg.BLOCK_DEPTH) for cfg in configs],
        "decision_threshold": decision_threshold,
        "val_whole_dice_soft_macro": float(np.mean([row["soft_dice"] for row in rows])) if rows else 0.0,
        "val_whole_dice_hard": float(np.mean([row["hard_dice"] for row in rows])) if rows else 0.0,
        "val_whole_dice_hard_best_thr": float(np.median([row["best_threshold"] for row in rows])) if rows else 0.0,
        "val_whole_dice_hard_best_thr_score": float(np.mean([row["best_threshold_dice"] for row in rows])) if rows else 0.0,
        "source_best_threshold_dice": _row_group_mean(rows, "source", "best_threshold_dice") if rows else {},
        "lesion_group_best_threshold_dice": _row_group_mean(rows, "lesion_group", "best_threshold_dice") if rows else {},
    }
    with (out_path / "ensemble_val_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info(
        "Slice-block ensemble val: soft=%.5f hard@%.2f=%.5f best_thr_med=%.2f best_thr_score=%.5f",
        summary["val_whole_dice_soft_macro"],
        decision_threshold,
        summary["val_whole_dice_hard"],
        summary["val_whole_dice_hard_best_thr"],
        summary["val_whole_dice_hard_best_thr_score"],
    )
    return summary


class WholeBrainSliceBlockValidation(tf.keras.callbacks.Callback):
    def __init__(self, val_cases: list[CaseRecord], cfg: SliceBlockTrainingConfig):
        super().__init__()
        self.val_cases = list(val_cases)
        self.cfg = cfg
        self.best_score = -math.inf
        self.summary_path = cfg.CALLBACKS_DIR / "whole_val_summary.jsonl"
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if epoch % self.cfg.WHOLE_BRAIN_VAL_EVERY_N_EPOCHS != 0:
            return
        cases = self.val_cases
        if self.cfg.WHOLE_BRAIN_VAL_MAX_CASES is not None:
            cases = cases[: int(self.cfg.WHOLE_BRAIN_VAL_MAX_CASES)]
        rows = []
        start = time.time()
        for idx, case in enumerate(cases, start=1):
            prob, mask, ref_img = predict_case_probability_map(self.model, case, self.cfg)
            true_voxels = int(np.sum(mask > 0.5))
            sweep = {}
            for thr in self.cfg.VAL_THRESHOLD_SWEEP:
                sweep[f"{thr:.2f}"] = hard_dice_np(mask, prob >= thr)
            best_thr_key, best_thr_score = max(sweep.items(), key=lambda item: item[1])
            hard = hard_dice_np(mask, prob >= self.cfg.DECISION_THRESHOLD)
            row = {
                "epoch": epoch,
                "source": case.source,
                "case_id": case.case_id,
                "soft_dice": soft_dice_np(mask, prob),
                "hard_dice": hard,
                "best_threshold": float(best_thr_key),
                "best_threshold_dice": float(best_thr_score),
                "true_voxels": true_voxels,
                "pred_soft_voxels": float(np.sum(prob)),
                "pred_hard_voxels": int(np.sum(prob >= self.cfg.DECISION_THRESHOLD)),
                "pred_max": float(np.max(prob)),
                "pred_mean": float(np.mean(prob)),
                "lesion_group": lesion_size_group(true_voxels),
                "image": str(case.image_path),
                "mask": str(case.mask_path),
            }
            row.update({f"dice_thr_{key}": value for key, value in sweep.items()})
            rows.append(row)
            if idx % 16 == 0 or idx == len(cases):
                logger.info("Whole-brain slice-block val progress: %d/%d", idx, len(cases))

        if not rows:
            return
        csv_path = self.cfg.CALLBACKS_DIR / f"whole_val_epoch_{epoch:04d}.csv"
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        soft_macro = float(np.mean([row["soft_dice"] for row in rows]))
        hard_macro = float(np.mean([row["hard_dice"] for row in rows]))
        best_thr_score = float(np.mean([row["best_threshold_dice"] for row in rows]))
        best_thr = float(np.median([row["best_threshold"] for row in rows]))
        pred_max_p90 = float(np.percentile([row["pred_max"] for row in rows], 90))
        pred_vox_med = float(np.median([row["pred_hard_voxels"] for row in rows]))
        summary = {
            "epoch": int(epoch),
            "elapsed_sec": float(time.time() - start),
            "n_cases": len(rows),
            "val_whole_dice_soft_macro": soft_macro,
            "val_whole_dice_hard": hard_macro,
            "val_whole_dice_hard_best_thr": best_thr,
            "val_whole_dice_hard_best_thr_score": best_thr_score,
            "pred_max_p90": pred_max_p90,
            "pred_hard_voxels_median": pred_vox_med,
        }
        with self.summary_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(summary) + "\n")

        logs["val_whole_dice_soft_macro"] = soft_macro
        logs["val_whole_dice_hard"] = hard_macro
        logs["val_whole_dice_hard_best_thr"] = best_thr
        logs["val_whole_dice_hard_best_thr_score"] = best_thr_score
        logger.info(
            "Whole-brain slice-block val @epoch %d: soft=%.5f hard@%.2f=%.5f best_thr_med=%.2f best_thr_score=%.5f pred_max_p90=%.4f",
            epoch,
            soft_macro,
            self.cfg.DECISION_THRESHOLD,
            hard_macro,
            best_thr,
            best_thr_score,
            pred_max_p90,
        )

        if best_thr_score > self.best_score:
            self.best_score = best_thr_score
            if self.cfg.SAVE_VAL_PREDICTIONS and self.cfg.NUM_VAL_PREDICTIONS > 0:
                out_dir = self.cfg.CALLBACKS_DIR / "predictions" / f"best_epoch_{epoch:04d}"
                for case in cases[: self.cfg.NUM_VAL_PREDICTIONS]:
                    prob, _, ref_img = predict_case_probability_map(self.model, case, self.cfg)
                    save_case_outputs(out_dir, case, prob, self.cfg.DECISION_THRESHOLD, ref_img)


class StopAtWholeDice(tf.keras.callbacks.Callback):
    def __init__(self, target: float, metric: str = "val_whole_dice_hard_best_thr_score"):
        super().__init__()
        self.target = float(target)
        self.metric = metric

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        value = logs.get(self.metric)
        if value is None:
            return
        if float(value) >= self.target:
            logger.info(
                "Target whole-brain Dice reached at epoch %d: %s=%.5f >= %.5f",
                epoch,
                self.metric,
                float(value),
                self.target,
            )
            self.model.stop_training = True


def compile_model(model: tf.keras.Model, cfg: SliceBlockTrainingConfig) -> None:
    total_steps = max(1, int((cfg.STEPS_PER_EPOCH or 1) * max(1, cfg.TOTAL_EPOCHS - cfg.INITIAL_EPOCH)))
    lr = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=float(cfg.INITIAL_LR),
        decay_steps=total_steps,
        alpha=float(cfg.MIN_LR) / float(cfg.INITIAL_LR),
    )
    optimizer_kwargs = {"learning_rate": lr}
    if cfg.WEIGHT_DECAY > 0 and hasattr(tf.keras.optimizers, "AdamW"):
        optimizer = tf.keras.optimizers.AdamW(
            weight_decay=float(cfg.WEIGHT_DECAY),
            clipnorm=float(cfg.MAX_GRAD_NORM) if cfg.MAX_GRAD_NORM else None,
            **optimizer_kwargs,
        )
    else:
        if cfg.MAX_GRAD_NORM:
            optimizer_kwargs["clipnorm"] = float(cfg.MAX_GRAD_NORM)
        optimizer = tf.keras.optimizers.Adam(**optimizer_kwargs)
    loss = WeightedBceDiceTversky(
        positive_weight=cfg.POSITIVE_WEIGHT,
        bce_weight=cfg.BCE_WEIGHT,
        dice_weight=cfg.DICE_WEIGHT,
        focal_tversky_weight=cfg.FOCAL_TVERSKY_WEIGHT,
        tversky_alpha=cfg.TVERSKY_ALPHA,
        tversky_beta=cfg.TVERSKY_BETA,
        focal_tversky_gamma=cfg.FOCAL_TVERSKY_GAMMA,
        lesion_slice_weight=cfg.LESION_SLICE_WEIGHT,
        empty_slice_weight=cfg.EMPTY_SLICE_WEIGHT,
        dice_on_lesion_slices_only=cfg.DICE_ON_LESION_SLICES_ONLY,
        positive_topk_weight=cfg.POSITIVE_TOPK_WEIGHT,
        positive_topk_fraction=cfg.POSITIVE_TOPK_FRACTION,
        small_lesion_boost_reference=cfg.SMALL_LESION_BOOST_REFERENCE,
        small_lesion_boost_max=cfg.SMALL_LESION_BOOST_MAX,
    )
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[dice_coefficient, hard_dice_metric, foreground_fraction],
        jit_compile=bool(cfg.JIT_COMPILE),
    )


def write_training_summary(history, cfg: SliceBlockTrainingConfig) -> None:
    summary = {
        "epochs_recorded": len(getattr(history, "epoch", []) or []),
        "metrics": getattr(history, "history", {}),
        "config": str(cfg.MODEL_DIR / "config.json"),
        "callbacks_dir": str(cfg.CALLBACKS_DIR),
        "model_dir": str(cfg.MODEL_DIR),
    }
    diag_dir = cfg.CALLBACKS_DIR / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    with (diag_dir / "training_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


def train_slice_block_model(config: Optional[SliceBlockTrainingConfig] = None, **overrides):
    cfg = config or SliceBlockTrainingConfig(**overrides)
    if config is not None and overrides:
        for key, value in overrides.items():
            if not hasattr(cfg, key):
                raise TypeError(f"Unknown config override: {key}")
            setattr(cfg, key, value)
        cfg.__post_init__()

    configure_runtime(cfg)
    cases = load_cases(cfg)
    if not cases:
        raise ValueError(f"No cases found under {cfg.DATA_DIR}")
    train_cases, val_cases, lesion_sizes = split_cases(cases, cfg)
    write_split_diagnostics(train_cases, val_cases, cases, lesion_sizes, cfg)
    size_by_id = {case.case_id: int(size) for case, size in zip(cases, lesion_sizes)}
    train_lesion_sizes = [size_by_id.get(case.case_id, 0) for case in train_cases]
    val_lesion_sizes = [size_by_id.get(case.case_id, 0) for case in val_cases]
    if cfg.BALANCED_CASE_SAMPLING:
        sampling_weights = _case_sampling_weights(train_cases, train_lesion_sizes, cfg)
        write_sampling_diagnostics(train_cases, train_lesion_sizes, sampling_weights, cfg)
        logger.info(
            "Balanced case sampling enabled: source_balanced=%s size_aware=%s size_edges=%s size_probs=%s",
            cfg.SOURCE_BALANCED_SAMPLING,
            cfg.SIZE_AWARE_SAMPLING,
            cfg.SIZE_BUCKET_EDGES,
            tuple(round(v, 4) for v in cfg.SIZE_BUCKET_PROBS),
        )

    train_steps = int(cfg.STEPS_PER_EPOCH or len(train_cases))
    val_steps = int(cfg.VALIDATION_STEPS or max(1, min(len(val_cases), 16)))
    cfg.STEPS_PER_EPOCH = train_steps
    cfg.VALIDATION_STEPS = val_steps
    cfg._write_config()

    train_ds = make_tf_dataset(train_cases, cfg, training=True, lesion_sizes=train_lesion_sizes)
    val_ds = make_tf_dataset(val_cases, cfg, training=False, lesion_sizes=val_lesion_sizes)

    model = build_slice_block_model(cfg)
    if cfg.INITIAL_WEIGHTS_PATH is not None:
        if not cfg.INITIAL_WEIGHTS_PATH.exists():
            raise FileNotFoundError(f"Initial weights not found: {cfg.INITIAL_WEIGHTS_PATH}")
        logger.info("Loading initial weights from %s", cfg.INITIAL_WEIGHTS_PATH)
        model.load_weights(str(cfg.INITIAL_WEIGHTS_PATH))
        baseline_copy = cfg.CALLBACKS_DIR / "baseline_initial.weights.h5"
        if not baseline_copy.exists():
            shutil.copy2(cfg.INITIAL_WEIGHTS_PATH, baseline_copy)
            logger.info("Copied initial weights to %s", baseline_copy)
    compile_model(model, cfg)
    logger.info("Model built: %s params=%d input=%s", model.name, model.count_params(), cfg.input_shape)
    logger.info(
        "Slice-block design: one brain per step, block_depth=%d slice_axis=%d stride=%d train_steps=%d val_steps=%d",
        cfg.BLOCK_DEPTH,
        cfg.SLICE_AXIS,
        cfg.SLICE_STRIDE,
        train_steps,
        val_steps,
    )
    logger.info(
        "Loss focus: lesion_slice_weight=%.3f empty_slice_weight=%.3f small_lesion_boost=(ref=%.1f max=%.2f) dice_lesion_only=%s positive_topk_weight=%.3f topk_fraction=%.3f tversky=(alpha=%.3f beta=%.3f) threshold=%.3f",
        cfg.LESION_SLICE_WEIGHT,
        cfg.EMPTY_SLICE_WEIGHT,
        cfg.SMALL_LESION_BOOST_REFERENCE,
        cfg.SMALL_LESION_BOOST_MAX,
        cfg.DICE_ON_LESION_SLICES_ONLY,
        cfg.POSITIVE_TOPK_WEIGHT,
        cfg.POSITIVE_TOPK_FRACTION,
        cfg.TVERSKY_ALPHA,
        cfg.TVERSKY_BETA,
        cfg.DECISION_THRESHOLD,
    )

    callbacks = [
        WholeBrainSliceBlockValidation(val_cases, cfg),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(cfg.checkpoint_path),
            monitor="val_whole_dice_hard_best_thr_score",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(cfg.latest_path),
            save_best_only=False,
            save_weights_only=True,
            verbose=0,
        ),
        tf.keras.callbacks.CSVLogger(str(cfg.CALLBACKS_DIR / "training_log.csv"), append=cfg.INITIAL_EPOCH > 0),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    if cfg.EARLY_STOPPING_PATIENCE is not None:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_whole_dice_hard_best_thr_score",
                mode="max",
                min_delta=float(cfg.EARLY_STOPPING_MIN_DELTA),
                patience=int(cfg.EARLY_STOPPING_PATIENCE),
                restore_best_weights=bool(cfg.RESTORE_BEST_WEIGHTS),
                verbose=1,
            )
        )
    if cfg.TARGET_WHOLE_DICE is not None:
        callbacks.append(StopAtWholeDice(float(cfg.TARGET_WHOLE_DICE)))

    history = model.fit(
        train_ds,
        steps_per_epoch=train_steps,
        epochs=int(cfg.TOTAL_EPOCHS),
        initial_epoch=int(cfg.INITIAL_EPOCH),
        validation_data=val_ds,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=int(cfg.FIT_VERBOSE),
    )
    write_training_summary(history, cfg)
    return history
