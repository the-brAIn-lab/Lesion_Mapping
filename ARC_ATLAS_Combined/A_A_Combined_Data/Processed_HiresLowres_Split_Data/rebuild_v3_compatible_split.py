from __future__ import annotations

import csv
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path


BASE = Path(
    "/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/"
    "A_A_Combined_Data/Processed_HiresLowres_Split_Data"
)
ALL_OUT = BASE / "Processed_HiresLowres_All"
RUN_EVAL = Path(
    "/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/"
    "ARC_ATLAS_Train_v3/runs/20251110_101813/test_eval"
)

HIRES_EVAL_CSV = RUN_EVAL / "test_hires_metrics_with_manifest_20251111_130301.csv"
LORES_EVAL_CSV = RUN_EVAL / "test_lores_metrics_with_manifest_20251111_143340.csv"

SPLITS = ("train_hires", "test_hires", "test_lores")
SUBDIRS = ("t1", "masks")

CSV_FIELDS = [
    "split",
    "manifest_source",
    "dataset",
    "key",
    "subject",
    "t1_path",
    "mask_path",
    "source_t1_path",
    "source_mask_path",
    "res_bin",
    "fwhm_mm",
    "hi_freq_energy",
    "lap_var",
    "score",
    "mask_voxels",
    "mask_ml",
    "mask_voxels_clean",
    "mask_ml_clean",
    "brain_frac",
    "vox_total",
    "norm_img_nonzero",
    "brain_voxels",
    "voxel_mm3",
]


def infer_dataset(key: str) -> str:
    if key.startswith("sub-r"):
        return "ATLAS"
    if key.startswith("sub-M"):
        return "ARC"
    return "UNKNOWN"


def subject_from_key(key: str) -> str:
    if "_ses-" in key:
        return key.split("_ses-", 1)[0]
    return key


def key_from_t1_name(name: str) -> str:
    suffix = "_T1w_MNI_norm.nii.gz"
    if not name.endswith(suffix):
        raise ValueError(f"Unexpected T1 filename: {name}")
    return name[: -len(suffix)]


def key_from_mask_name(name: str) -> str:
    suffix = "_lesion_mask_MNI_clean.nii.gz"
    if not name.endswith(suffix):
        raise ValueError(f"Unexpected mask filename: {name}")
    return name[: -len(suffix)]


def load_processed_pairs() -> dict[str, dict[str, Path]]:
    t1_dir = ALL_OUT / "t1"
    mask_dir = ALL_OUT / "masks"
    t1_map = {key_from_t1_name(p.name): p for p in sorted(t1_dir.glob("*_T1w_MNI_norm.nii.gz"))}
    mask_map = {
        key_from_mask_name(p.name): p for p in sorted(mask_dir.glob("*_lesion_mask_MNI_clean.nii.gz"))
    }

    missing_masks = sorted(set(t1_map) - set(mask_map))
    if missing_masks:
        raise RuntimeError(f"Missing masks for {len(missing_masks)} keys, first few: {missing_masks[:10]}")

    pairs = {}
    for key, t1_path in t1_map.items():
        pairs[key] = {"t1": t1_path, "mask": mask_map[key]}
    return pairs


def load_eval_metadata(path: Path) -> dict[str, dict[str, str]]:
    meta: dict[str, dict[str, str]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row["key"]
            meta[key] = {
                "dataset": row.get("mf_dataset") or infer_dataset(key),
                "res_bin": row.get("mf_bin", ""),
                "fwhm_mm": row.get("mf_fwhm_mm", ""),
                "hi_freq_energy": row.get("mf_hi_freq_energy", ""),
                "lap_var": row.get("mf_lap_var", ""),
                "score": row.get("mf_score", ""),
                "mask_voxels": row.get("mf_mask_voxels", ""),
                "mask_ml": row.get("mf_mask_ml", ""),
                "mask_voxels_clean": row.get("mf_mask_voxels_clean", ""),
                "mask_ml_clean": row.get("mf_mask_ml_clean", ""),
                "brain_frac": row.get("mf_brain_frac", ""),
                "vox_total": row.get("mf_vox_total", ""),
                "norm_img_nonzero": row.get("mf_norm_img_nonzero", ""),
                "brain_voxels": row.get("mf_brain_voxels", ""),
                "voxel_mm3": row.get("mf_voxel_mm3", ""),
            }
    return meta


def backup_existing_csv(path: Path) -> None:
    if not path.exists():
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_suffix(path.suffix + f".bak_{stamp}")
    shutil.copy2(path, backup)


def sync_split_files(split: str, target_keys: set[str], processed_pairs: dict[str, dict[str, Path]]) -> None:
    split_dir = BASE / split
    for subdir in SUBDIRS:
        out_dir = split_dir / subdir
        out_dir.mkdir(parents=True, exist_ok=True)

    current_t1 = {key_from_t1_name(p.name): p for p in (split_dir / "t1").glob("*_T1w_MNI_norm.nii.gz")}
    current_masks = {
        key_from_mask_name(p.name): p for p in (split_dir / "masks").glob("*_lesion_mask_MNI_clean.nii.gz")
    }

    for key, path in sorted(current_t1.items()):
        if key not in target_keys:
            path.unlink()
    for key, path in sorted(current_masks.items()):
        if key not in target_keys:
            path.unlink()

    for key in sorted(target_keys):
        src_t1 = processed_pairs[key]["t1"]
        src_mask = processed_pairs[key]["mask"]
        dst_t1 = split_dir / "t1" / src_t1.name
        dst_mask = split_dir / "masks" / src_mask.name
        shutil.copy2(src_t1, dst_t1)
        shutil.copy2(src_mask, dst_mask)


def write_split_csv(
    split: str,
    target_keys: set[str],
    processed_pairs: dict[str, dict[str, Path]],
    eval_meta: dict[str, dict[str, str]],
) -> None:
    csv_path = BASE / f"{split}.csv"
    backup_existing_csv(csv_path)

    split_dir = BASE / split
    rows = []
    for key in sorted(target_keys):
        src_t1 = processed_pairs[key]["t1"]
        src_mask = processed_pairs[key]["mask"]
        dst_t1 = split_dir / "t1" / src_t1.name
        dst_mask = split_dir / "masks" / src_mask.name

        meta = eval_meta.get(key, {})
        row = {
            "split": split,
            "manifest_source": (
                "exact_v3_eval_keys" if split != "train_hires" else "processed_all_minus_exact_v3_tests"
            ),
            "dataset": meta.get("dataset") or infer_dataset(key),
            "key": key,
            "subject": subject_from_key(key),
            "t1_path": str(dst_t1),
            "mask_path": str(dst_mask),
            "source_t1_path": str(src_t1),
            "source_mask_path": str(src_mask),
            "res_bin": meta.get("res_bin", ""),
            "fwhm_mm": meta.get("fwhm_mm", ""),
            "hi_freq_energy": meta.get("hi_freq_energy", ""),
            "lap_var": meta.get("lap_var", ""),
            "score": meta.get("score", ""),
            "mask_voxels": meta.get("mask_voxels", ""),
            "mask_ml": meta.get("mask_ml", ""),
            "mask_voxels_clean": meta.get("mask_voxels_clean", ""),
            "mask_ml_clean": meta.get("mask_ml_clean", ""),
            "brain_frac": meta.get("brain_frac", ""),
            "vox_total": meta.get("vox_total", ""),
            "norm_img_nonzero": meta.get("norm_img_nonzero", ""),
            "brain_voxels": meta.get("brain_voxels", ""),
            "voxel_mm3": meta.get("voxel_mm3", ""),
        }
        rows.append(row)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def count_split(split: str) -> int:
    return len(list((BASE / split / "t1").glob("*_T1w_MNI_norm.nii.gz")))


def main() -> None:
    for path in [ALL_OUT / "t1", ALL_OUT / "masks", HIRES_EVAL_CSV, LORES_EVAL_CSV]:
        if not path.exists():
            raise FileNotFoundError(path)

    processed_pairs = load_processed_pairs()
    hires_meta = load_eval_metadata(HIRES_EVAL_CSV)
    lores_meta = load_eval_metadata(LORES_EVAL_CSV)
    eval_meta = {**hires_meta, **lores_meta}

    test_hires_keys = set(hires_meta)
    test_lores_keys = set(lores_meta)
    overlap = test_hires_keys & test_lores_keys
    if overlap:
        raise RuntimeError(f"test_hires/test_lores overlap detected: {sorted(list(overlap))[:10]}")

    processed_keys = set(processed_pairs)
    missing_hires = test_hires_keys - processed_keys
    missing_lores = test_lores_keys - processed_keys
    if missing_hires or missing_lores:
        raise RuntimeError(
            "Missing processed files for held-out keys: "
            f"hires={sorted(list(missing_hires))[:10]} lores={sorted(list(missing_lores))[:10]}"
        )

    train_hires_keys = processed_keys - test_hires_keys - test_lores_keys

    split_map = {
        "test_hires": test_hires_keys,
        "test_lores": test_lores_keys,
        "train_hires": train_hires_keys,
    }

    for split, keys in split_map.items():
        sync_split_files(split, keys, processed_pairs)
        write_split_csv(split, keys, processed_pairs, eval_meta)

    print("Rebuilt v3-compatible processed split from Processed_HiresLowres_All")
    print(f"Processed_HiresLowres_All: {len(processed_pairs)}")
    for split in ("test_hires", "test_lores", "train_hires"):
        keys = split_map[split]
        counts = Counter(infer_dataset(key) for key in keys)
        print(f"{split}: {count_split(split)} cases | dataset mix: {dict(counts)}")


if __name__ == "__main__":
    main()
