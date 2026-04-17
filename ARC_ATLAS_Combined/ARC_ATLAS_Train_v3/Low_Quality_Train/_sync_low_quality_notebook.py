#!/usr/bin/env python3
from __future__ import annotations

import json
import textwrap
from pathlib import Path


NOTEBOOK = Path(
    "/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/"
    "ARC_ATLAS_Train_v3/Low_Quality_Train/ARC_ATLAS_Train_v3_Low_Quality.ipynb"
)

OLD_RUN_ROOT = (
    '/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/ARC_ATLAS_Train_v3'
)
NEW_RUN_ROOT = (
    '/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/ARC_ATLAS_Train_v3/Low_Quality_Train'
)

OLD_TRAIN_DIR = (
    '/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/'
    'A_A_Combined_Data/Processed_HiresLowres_Split_Data/train_hires'
)
NEW_TRAIN_DIR = (
    '/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/'
    'A_A_Combined_Data/Processed_LowQualityTrain_Split_Data/train_low_quality'
)


TRAINING_CELL_SOURCE = textwrap.dedent(
    f"""\
    # === ARC_ATLAS_Train_v3_Low_Quality — train on lowest-quality 522 non-held-out cases ===
    from pathlib import Path
    import importlib.util, os, sys, gc, time, traceback, shlex, subprocess

    # --------- Paths ----------
    PREFERRED_GPU = "0"   # set to None to auto-pick the GPU with the most free VRAM
    MIN_FREE_MB = 12000   # fail fast if the selected GPU has less than this free

    SPLIT_ROOT  = Path("{NEW_TRAIN_DIR}").parent
    TRAIN_DIR   = Path("{NEW_TRAIN_DIR}")
    TRAIN_T1    = TRAIN_DIR / "t1"
    TRAIN_MASKS = TRAIN_DIR / "masks"
    SPLIT_SUMMARY = SPLIT_ROOT / "split_summary.json"

    RUN_ROOT   = Path("{NEW_RUN_ROOT}")
    MODULE_PATH = Path("/home/rbielski/stroke_cleaned/stroke_segmentation_v1.2/stroke_seg_v1.2_train.py")

    def _query_gpus():
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, text=True)
        rows = []
        for line in out.strip().splitlines():
            idx, free_mb, total_mb = [x.strip() for x in line.split(",")]
            rows.append({{
                "index": idx,
                "free_mb": int(free_mb),
                "total_mb": int(total_mb),
            }})
        return rows

    def _query_gpu_processes():
        cmd = [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory,process_name",
            "--format=csv,noheader,nounits",
        ]
        try:
            out = subprocess.check_output(cmd, text=True)
        except Exception:
            return []
        rows = []
        for line in out.strip().splitlines():
            if not line.strip():
                continue
            pid, gpu_uuid, used_mb, process_name = [x.strip() for x in line.split(",", 3)]
            rows.append({{
                "pid": pid,
                "gpu_uuid": gpu_uuid,
                "used_mb": used_mb,
                "process_name": process_name,
            }})
        return rows

    def _select_gpu(preferred_gpu, min_free_mb):
        gpus = _query_gpus()
        if not gpus:
            raise RuntimeError("No GPUs reported by nvidia-smi.")

        selected = None
        if preferred_gpu is not None:
            selected = next((g for g in gpus if g["index"] == str(preferred_gpu)), None)
            if selected is None:
                raise RuntimeError(f"Preferred GPU {{preferred_gpu}} not found. Visible GPUs: {{gpus}}")
        else:
            selected = max(gpus, key=lambda g: g["free_mb"])

        if selected["free_mb"] < min_free_mb:
            proc_rows = _query_gpu_processes()
            proc_text = "\\n".join(
                f"  pid={{r['pid']}} used={{r['used_mb']}}MB name={{r['process_name']}}"
                for r in proc_rows
            ) or "  <none>"
            gpu_text = "\\n".join(
                f"  GPU {{g['index']}}: free={{g['free_mb']}}MB / total={{g['total_mb']}}MB"
                for g in gpus
            )
            raise RuntimeError(
                "Not enough free GPU memory to start training.\\n"
                f"Selected GPU {{selected['index']}} has only {{selected['free_mb']}}MB free; "
                f"require at least {{min_free_mb}}MB.\\n"
                "Current GPU memory:\\n"
                f"{{gpu_text}}\\n"
                "Active GPU compute processes:\\n"
                f"{{proc_text}}\\n"
                "Stop or restart the stale TensorFlow/Jupyter kernels, then restart this notebook kernel "
                "and rerun this cell."
            )

        return selected["index"]

    if "tensorflow" in sys.modules:
        raise RuntimeError(
            "TensorFlow is already imported in this notebook kernel. Restart the kernel before running "
            "this training cell so CUDA_VISIBLE_DEVICES and memory-growth settings apply cleanly."
        )

    CUDA_ID = _select_gpu(PREFERRED_GPU, MIN_FREE_MB)

    # --------- Env setup (must happen BEFORE TensorFlow import) ----------
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_ID)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    # Import TF only after device selection
    import tensorflow as tf
    from tensorflow.keras import mixed_precision

    # --------- New run folders ----------
    RUN_ID = time.strftime("%Y%m%d_%H%M%S")
    RUN_DIR = RUN_ROOT / "runs" / RUN_ID
    MODEL_DIR = RUN_DIR / "models"
    CALLBACKS_DIR = RUN_DIR / "callbacks"
    LOG_DIR = RUN_DIR / "logs"
    for d in (MODEL_DIR, CALLBACKS_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    os.environ["SMARTSOTA_LOG_DIR"] = str(LOG_DIR)

    tf.keras.backend.clear_session()
    gc.collect()
    mixed_precision.set_global_policy("mixed_float16")

    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()
            return len(data)
        def flush(self):
            for s in self.streams:
                s.flush()

    log_file = open(LOG_DIR / "train_stdout_stderr.log", "a", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    print("Run ID:", RUN_ID)
    print("Selected GPU:", CUDA_ID)
    print("TF:", tf.__version__)
    print("GPUs visible to TF:", tf.config.list_physical_devices("GPU"))
    print(f"Train images: {{len(list(TRAIN_T1.glob('*.nii.gz')))}}  masks: {{len(list(TRAIN_MASKS.glob('*.nii.gz')))}}")
    print("Split summary:", SPLIT_SUMMARY)

    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception as e:
            print("set_memory_growth failed:", e)

    spec = importlib.util.spec_from_file_location("arc_seg_train", MODULE_PATH)
    seg = importlib.util.module_from_spec(spec)
    seg.tf = tf
    spec.loader.exec_module(seg)

    seg.strategy = tf.distribute.get_strategy()
    print("Strategy:", type(seg.strategy).__name__)

    INPUT_SHAPE   = (192, 224, 192, 1)
    BATCH_SIZE    = 1
    BASE_FILTERS  = 8
    SAM_HEADS     = 2
    AUG_INTENSITY = 0.30
    VAL_SPLIT     = 0.15
    TOTAL_EPOCHS  = 140
    INITIAL_EPOCH = 0

    INITIAL_LR    = 1e-4
    MIN_LR        = 5e-7
    WARMUP_EPOCHS = 15

    try:
        history = seg.train_dynamic_model(
            DATA_DIR=TRAIN_DIR,
            IMAGES_DIR=TRAIN_T1,
            MASKS_DIR=TRAIN_MASKS,
            MODEL_DIR=MODEL_DIR,
            CALLBACKS_DIR=CALLBACKS_DIR,
            TOTAL_EPOCHS=TOTAL_EPOCHS,
            INITIAL_EPOCH=INITIAL_EPOCH,
            LOAD_WEIGHTS_FROM=None,
            RESUME_FROM_LATEST=False,
            INPUT_SHAPE=INPUT_SHAPE,
            BATCH_SIZE=BATCH_SIZE,
            BASE_FILTERS=BASE_FILTERS,
            SAM_HEADS=SAM_HEADS,
            RESAMPLE_TO_TARGET=True,
            AUGMENTATION_INTENSITY=AUG_INTENSITY,
            VALIDATION_SPLIT=VAL_SPLIT,
            INITIAL_LR=INITIAL_LR,
            MIN_LR=MIN_LR,
            WARMUP_EPOCHS=WARMUP_EPOCHS,
        )
        print("Training complete. Logged keys:", list(getattr(history, "history", {{}}).keys()))
        print("Run artifacts at:", RUN_DIR)
    except Exception:
        print("\\n================= UNCAUGHT EXCEPTION =================")
        traceback.print_exc()
        print("======================================================\\n")
        try:
            print("Last few GPU snapshots:")
            for _ in range(3):
                subprocess.run(shlex.split("nvidia-smi"), check=False)
                time.sleep(1)
        except Exception:
            pass
        raise
    finally:
        try:
            log_file.flush()
        except Exception:
            pass
    """
)


def replace_source(src: str, cell_index: int) -> str:
    src = src.replace(OLD_RUN_ROOT, NEW_RUN_ROOT)
    src = src.replace(f"{NEW_RUN_ROOT}/Low_Quality_Train", NEW_RUN_ROOT)
    src = src.replace(OLD_TRAIN_DIR, NEW_TRAIN_DIR)

    old_title = "# === ARC_ATLAS_Train_v3 — retrain on reprocessed (March 2026) data =========="
    new_title = "# === ARC_ATLAS_Train_v3_Low_Quality — train on lowest-quality 522 non-held-out cases ==="
    src = src.replace(old_title, new_title)

    if cell_index == 3:
        return TRAINING_CELL_SOURCE

    if "RUN_ROOT = Path(" in src:
        lines = []
        for line in src.splitlines():
            if line.startswith("RUN_ROOT = Path("):
                lines.append(f'RUN_ROOT = Path("{NEW_RUN_ROOT}")')
            else:
                lines.append(line)
        src = "\n".join(lines) + ("\n" if src.endswith("\n") else "")

    return src


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text())
    changed = False

    for i, cell in enumerate(nb.get("cells", [])):
        source = "".join(cell.get("source", []))
        new_source = replace_source(source, i)
        if new_source != source:
            cell["source"] = new_source.splitlines(keepends=True)
            changed = True
        if cell.get("cell_type") == "code" and (cell.get("outputs") or cell.get("execution_count") is not None):
            cell["outputs"] = []
            cell["execution_count"] = None
            changed = True

    if not changed:
        raise SystemExit("No notebook updates were applied.")

    NOTEBOOK.write_text(json.dumps(nb, indent=1))
    print(f"Updated {NOTEBOOK}")


if __name__ == "__main__":
    main()
