#!/usr/bin/env python3
"""
Retrain ARC_ATLAS v3 model on reprocessed (March 2026) data.
Same architecture and hyperparams as the original Nov 2025 run.
New TRAIN_DIR: stroke_cleaned train_hires (522 subjects, new skull-strip preprocessing).
"""
from pathlib import Path
import importlib.util, os, sys, gc, time, traceback
import tensorflow as tf
from tensorflow.keras import mixed_precision

# ── Paths ─────────────────────────────────────────────────────────────────────
CUDA_ID     = "0"
TRAIN_DIR   = Path("/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/A_A_Combined_Data/Processed_HiresLowres_Split_Data/train_hires")
TRAIN_T1    = TRAIN_DIR / "t1"
TRAIN_MASKS = TRAIN_DIR / "masks"
RUN_ROOT    = Path("/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/ARC_ATLAS_Train_v3")
MODULE_PATH = Path("/home/rbielski/stroke_cleaned/stroke_segmentation_v1.2/stroke_seg_v1.2_train.py")

RUN_ID        = time.strftime("%Y%m%d_%H%M%S")
RUN_DIR       = RUN_ROOT / "runs" / RUN_ID
MODEL_DIR     = RUN_DIR / "models"
CALLBACKS_DIR = RUN_DIR / "callbacks"
LOG_DIR       = RUN_DIR / "logs"
for d in (MODEL_DIR, CALLBACKS_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Env ───────────────────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"]      = CUDA_ID
os.environ["TF_CPP_MIN_LOG_LEVEL"]      = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["SMARTSOTA_LOG_DIR"]         = str(LOG_DIR)

log_file = open(LOG_DIR / "train_stdout.log", "a", buffering=1)
class Tee:
    def __init__(self, *s): self.streams = s
    def write(self, d):
        for s in self.streams: s.write(d); s.flush()
        return len(d)
    def flush(self):
        for s in self.streams: s.flush()
sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)

tf.keras.backend.clear_session(); gc.collect()
mixed_precision.set_global_policy("mixed_float16")

for g in tf.config.list_physical_devices("GPU"):
    try: tf.config.experimental.set_memory_growth(g, True)
    except Exception as e: print("set_memory_growth:", e)

print(f"Run ID   : {RUN_ID}")
print(f"RUN_DIR  : {RUN_DIR}")
print(f"TRAIN_DIR: {TRAIN_DIR}")
print(f"TF       : {tf.__version__}")
print(f"GPUs     : {tf.config.list_physical_devices('GPU')}")

# ── Load module ───────────────────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location("arc_seg_train", MODULE_PATH)
seg  = importlib.util.module_from_spec(spec)
seg.tf = tf
spec.loader.exec_module(seg)

# Single-GPU strategy (same as successful Nov 2025 run)
seg.strategy = tf.distribute.get_strategy()
print(f"Strategy : {type(seg.strategy).__name__}")

n_train = len(list(TRAIN_T1.glob("*.nii.gz")))
n_masks = len(list(TRAIN_MASKS.glob("*.nii.gz")))
print(f"Images   : {n_train}  Masks: {n_masks}")

# ── Hyperparams (identical to Nov 2025 run) ───────────────────────────────────
try:
    history = seg.train_dynamic_model(
        DATA_DIR             = TRAIN_DIR,
        IMAGES_DIR           = TRAIN_T1,
        MASKS_DIR            = TRAIN_MASKS,
        MODEL_DIR            = MODEL_DIR,
        CALLBACKS_DIR        = CALLBACKS_DIR,
        TOTAL_EPOCHS         = 140,
        INITIAL_EPOCH        = 0,
        LOAD_WEIGHTS_FROM    = None,
        RESUME_FROM_LATEST   = False,
        INPUT_SHAPE          = (192, 224, 192, 1),
        BATCH_SIZE           = 1,
        BASE_FILTERS         = 8,
        SAM_HEADS            = 2,
        RESAMPLE_TO_TARGET   = True,
        AUGMENTATION_INTENSITY = 0.30,
        VALIDATION_SPLIT     = 0.15,
        INITIAL_LR           = 1e-4,
        MIN_LR               = 5e-7,
        WARMUP_EPOCHS        = 15,
    )
    print("Training complete. Keys:", list(getattr(history, "history", {}).keys()))
    print("Artifacts:", RUN_DIR)
except Exception:
    traceback.print_exc()
