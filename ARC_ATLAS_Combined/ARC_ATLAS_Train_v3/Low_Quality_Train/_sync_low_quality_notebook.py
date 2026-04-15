#!/usr/bin/env python3
from __future__ import annotations

import json
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


def replace_source(src: str) -> str:
    src = src.replace(OLD_RUN_ROOT, NEW_RUN_ROOT)
    src = src.replace(OLD_TRAIN_DIR, NEW_TRAIN_DIR)

    old_title = "# === ARC_ATLAS_Train_v3 — retrain on reprocessed (March 2026) data =========="
    new_title = "# === ARC_ATLAS_Train_v3_Low_Quality — train on lowest-quality 522 non-held-out cases ==="
    src = src.replace(old_title, new_title)

    if "TRAIN_DIR   = Path" in src and "split_summary.json" not in src:
        marker = (
            f'TRAIN_DIR   = Path("{NEW_TRAIN_DIR}")\n'
            'TRAIN_T1    = TRAIN_DIR / "t1"      # <- use separated subfolder ONLY\n'
            'TRAIN_MASKS = TRAIN_DIR / "masks"   # <- use separated subfolder ONLY\n'
        )
        replacement = (
            f'SPLIT_ROOT  = Path("{NEW_TRAIN_DIR}").parent\n'
            f'TRAIN_DIR   = Path("{NEW_TRAIN_DIR}")\n'
            'TRAIN_T1    = TRAIN_DIR / "t1"      # <- use separated subfolder ONLY\n'
            'TRAIN_MASKS = TRAIN_DIR / "masks"   # <- use separated subfolder ONLY\n'
            'SPLIT_SUMMARY = SPLIT_ROOT / "split_summary.json"\n'
        )
        src = src.replace(marker, replacement)

    if 'print(f"Train images: {len(list(TRAIN_T1.glob(\'*.nii.gz\')))}  masks: {len(list(TRAIN_MASKS.glob(\'*.nii.gz\')))}")' in src:
        old_line = 'print(f"Train images: {len(list(TRAIN_T1.glob(\'*.nii.gz\')))}  masks: {len(list(TRAIN_MASKS.glob(\'*.nii.gz\')))}")'
        new_line = (
            'print(f"Train images: {len(list(TRAIN_T1.glob(\'*.nii.gz\')))}  masks: {len(list(TRAIN_MASKS.glob(\'*.nii.gz\')))}")\n'
            'print("Split summary:", SPLIT_SUMMARY)\n'
        )
        src = src.replace(old_line, new_line)

    return src


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text())
    changed = False

    for cell in nb.get("cells", []):
        source = "".join(cell.get("source", []))
        new_source = replace_source(source)
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
