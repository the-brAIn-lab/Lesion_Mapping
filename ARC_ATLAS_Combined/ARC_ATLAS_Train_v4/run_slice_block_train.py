#!/usr/bin/env python
"""Launch the ARC/ATLAS 2.5D slice-block lesion trainer."""

from __future__ import annotations

import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import training_v2_slice_blocks as seg  # noqa: E402


def main() -> None:
    train_dir = PROJECT_ROOT / "data" / "splits" / "90_10_random" / "train"
    run_dir = PROJECT_ROOT / "runs" / f"{time.strftime('%Y%m%d_%H%M%S')}_slice_blocks"

    cfg = seg.SliceBlockTrainingConfig(
        DATA_DIR=train_dir,
        IMAGES_DIR=train_dir / "t1",
        MASKS_DIR=train_dir / "masks",
        MANIFEST_PATH=train_dir / "manifest.csv",
        MODEL_DIR=run_dir / "models",
        CALLBACKS_DIR=run_dir / "callbacks",
        TARGET_SHAPE=(192, 224, 192),
        RESAMPLE_TO_TARGET=False,
        SLICE_AXIS=2,
        BLOCK_DEPTH=3,
        SLICE_STRIDE=1,
        TOTAL_EPOCHS=120,
        INITIAL_LR=2e-4,
        BASE_FILTERS=8,
        UNET_DEPTH=4,
        DECISION_THRESHOLD=0.35,
        SAVE_VAL_PREDICTIONS=True,
        NUM_VAL_PREDICTIONS=3,
        FIT_VERBOSE=2,
    )

    print("2.5D slice-block training")
    print(f"Training data: {train_dir}")
    print(f"Run dir: {run_dir}")
    print(f"Input per brain: (num_slices, {cfg.input_shape[0]}, {cfg.input_shape[1]}, {cfg.input_shape[2]})")
    print(f"Checkpoint: {cfg.checkpoint_path}")
    print(f"Validation probability/segmentation outputs: {cfg.CALLBACKS_DIR / 'predictions'}")

    seg.train_slice_block_model(cfg)


if __name__ == "__main__":
    main()
