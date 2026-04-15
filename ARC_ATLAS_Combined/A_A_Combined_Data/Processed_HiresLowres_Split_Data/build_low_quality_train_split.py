#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from numpy.fft import fftn, fftshift
from scipy.ndimage import laplace


CURRENT_SPLIT_BASE = Path(
    "/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/"
    "A_A_Combined_Data/Processed_HiresLowres_Split_Data"
)
ALL_OUT = CURRENT_SPLIT_BASE / "Processed_HiresLowres_All"

OUT_BASE = Path(
    "/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/"
    "A_A_Combined_Data/Processed_LowQualityTrain_Split_Data"
)

FIXED_TEST_HIRES_CSV = CURRENT_SPLIT_BASE / "test_hires.csv"
CURRENT_TEST_LORES_CSV = CURRENT_SPLIT_BASE / "test_lores.csv"
CURRENT_TRAIN_HIRES_CSV = CURRENT_SPLIT_BASE / "train_hires.csv"

TRAIN_LOW_COUNT = 522
N_JOBS = min(4, os.cpu_count() or 1)
SUBDIRS = ("t1", "masks")

ALL_CASES_CSV = OUT_BASE / "all_cases_quality_manifest.csv"
SUMMARY_JSON = OUT_BASE / "split_summary.json"

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
    "current_v3_split",
    "fixed_test_hires",
    "quality_rank_all",
    "score",
    "fwhm_mm",
    "hi_freq_energy",
    "lap_var",
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
    t1_map = {
        key_from_t1_name(p.name): p
        for p in sorted((ALL_OUT / "t1").glob("*_T1w_MNI_norm.nii.gz"))
    }
    mask_map = {
        key_from_mask_name(p.name): p
        for p in sorted((ALL_OUT / "masks").glob("*_lesion_mask_MNI_clean.nii.gz"))
    }

    missing_masks = sorted(set(t1_map) - set(mask_map))
    if missing_masks:
        raise RuntimeError(f"Missing masks for {len(missing_masks)} keys, first few: {missing_masks[:10]}")

    return {
        key: {"t1": t1_path, "mask": mask_map[key]}
        for key, t1_path in t1_map.items()
    }


def load_current_split_map() -> dict[str, str]:
    split_map: dict[str, str] = {}
    for csv_path, split_name in [
        (CURRENT_TRAIN_HIRES_CSV, "train_hires"),
        (CURRENT_TEST_LORES_CSV, "test_lores"),
        (FIXED_TEST_HIRES_CSV, "test_hires"),
    ]:
        df = pd.read_csv(csv_path)
        for key in df["key"].tolist():
            split_map[key] = split_name
    return split_map


def zscore(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    valid = np.isfinite(a)
    out = np.full_like(a, np.nan, dtype=float)
    if not valid.any():
        return out
    mu = np.nanmean(a[valid])
    sd = np.nanstd(a[valid])
    out[valid] = (a[valid] - mu) / (sd + 1e-12)
    return out


def zscore_in_brain(vol: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = vol != 0
    if not mask.any():
        return vol.astype(np.float32), mask
    v = vol[mask].astype(np.float32)
    mu = v.mean()
    sd = v.std() or 1.0
    out = np.zeros_like(vol, dtype=np.float32)
    out[mask] = (vol[mask] - mu) / sd
    return out, mask


def estimate_fwhm_gaussian_acf(img: np.ndarray, mask: np.ndarray) -> float:
    cors: dict[int, list[float]] = {}
    for shift in (1, 2, 3):
        for axis in range(3):
            slicer_f = [slice(None)] * 3
            slicer_g = [slice(None)] * 3
            slicer_f[axis] = slice(0, -shift)
            slicer_g[axis] = slice(shift, None)
            f = img[tuple(slicer_f)]
            g = img[tuple(slicer_g)]
            m = mask[tuple(slicer_f)] & mask[tuple(slicer_g)]
            if m.sum() < 1000:
                continue
            fv = f[m] - f[m].mean()
            gv = g[m] - g[m].mean()
            r = (fv * gv).mean() / ((fv.std() * gv.std()) + 1e-8)
            cors.setdefault(shift, []).append(float(r))
    if not cors:
        return float("nan")

    s2 = np.array(sorted(cors), dtype=float) ** 2
    r = np.array([np.mean(cors[s]) for s in sorted(cors)], dtype=float)
    ln_r = np.log(np.clip(r, 1e-6, 0.999))
    a = np.vstack([np.ones_like(s2), s2]).T
    _, b = np.linalg.lstsq(a, ln_r, rcond=None)[0]
    if b >= 0:
        return float("nan")
    return float(2.355 * np.sqrt(-1.0 / (2.0 * b)))


def high_freq_energy_ratio(img: np.ndarray, mask: np.ndarray, cutoff: float = 0.25) -> float:
    vol = np.zeros_like(img, dtype=np.float32)
    vol[mask] = img[mask]
    spectrum = fftshift(fftn(vol))
    power = (np.abs(spectrum) ** 2).astype(np.float64)
    return float(power[_high_freq_mask(img.shape, cutoff)].sum() / (power.sum() + 1e-12))


@lru_cache(maxsize=8)
def _high_freq_mask(shape: tuple[int, int, int], cutoff: float) -> np.ndarray:
    nx, ny, nz = shape
    cx, cy, cz = (np.array(shape, dtype=np.float32) - 1.0) / 2.0
    x, y, z = np.meshgrid(
        np.arange(nx, dtype=np.float32) - cx,
        np.arange(ny, dtype=np.float32) - cy,
        np.arange(nz, dtype=np.float32) - cz,
        indexing="ij",
    )
    radius = np.sqrt(((x / (nx / 2)) ** 2 + (y / (ny / 2)) ** 2 + (z / (nz / 2)) ** 2) / 3.0)
    return radius >= cutoff


def laplacian_variance(img: np.ndarray, mask: np.ndarray) -> float:
    return float(laplace(img)[mask].var())


def compute_quality_row(key: str, t1_path: Path) -> dict[str, object]:
    vol = nib.load(str(t1_path)).get_fdata().astype(np.float32)
    zimg, mask = zscore_in_brain(vol)
    if mask.sum() < 5000:
        raise RuntimeError(f"Too few nonzero brain voxels for {key}")
    return {
        "dataset": infer_dataset(key),
        "key": key,
        "subject": subject_from_key(key),
        "source_t1_path": str(t1_path),
        "fwhm_mm": estimate_fwhm_gaussian_acf(zimg, mask),
        "hi_freq_energy": high_freq_energy_ratio(zimg, mask),
        "lap_var": laplacian_variance(zimg, mask),
    }


def _compute_quality_row_from_item(item: tuple[str, str]) -> dict[str, object]:
    key, t1_path = item
    return compute_quality_row(key, Path(t1_path))


def compute_quality_manifest(processed_pairs: dict[str, dict[str, Path]]) -> pd.DataFrame:
    items = [(key, str(paths["t1"])) for key, paths in sorted(processed_pairs.items())]
    with ProcessPoolExecutor(max_workers=N_JOBS) as ex:
        rows = list(ex.map(_compute_quality_row_from_item, items))

    df = pd.DataFrame(rows).sort_values("key").reset_index(drop=True)
    df["score"] = (
        zscore(df["hi_freq_energy"].to_numpy(float))
        + zscore(df["lap_var"].to_numpy(float))
        + zscore(-df["fwhm_mm"].to_numpy(float))
    )
    df = df.sort_values(["score", "key"], ascending=[True, True]).reset_index(drop=True)
    df["quality_rank_all"] = np.arange(1, len(df) + 1, dtype=int)
    return df


def mirror_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def sync_split_files(split: str, df_split: pd.DataFrame, processed_pairs: dict[str, dict[str, Path]]) -> None:
    split_dir = OUT_BASE / split
    for subdir in SUBDIRS:
        out_dir = split_dir / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        pair_key = "mask" if subdir == "masks" else "t1"
        expected_names = {processed_pairs[key][pair_key].name for key in df_split["key"].tolist()}
        for existing in out_dir.glob("*.nii.gz"):
            if existing.name not in expected_names:
                existing.unlink()

    for key in df_split["key"].tolist():
        src_t1 = processed_pairs[key]["t1"]
        src_mask = processed_pairs[key]["mask"]
        mirror_file(src_t1, OUT_BASE / split / "t1" / src_t1.name)
        mirror_file(src_mask, OUT_BASE / split / "masks" / src_mask.name)


def write_split_csv(split: str, df_split: pd.DataFrame, processed_pairs: dict[str, dict[str, Path]]) -> None:
    out_rows = []
    for row in df_split.to_dict(orient="records"):
        key = row["key"]
        src_t1 = processed_pairs[key]["t1"]
        src_mask = processed_pairs[key]["mask"]
        out_rows.append({
            "split": split,
            "manifest_source": row["manifest_source"],
            "dataset": row["dataset"],
            "key": key,
            "subject": row["subject"],
            "t1_path": str(OUT_BASE / split / "t1" / src_t1.name),
            "mask_path": str(OUT_BASE / split / "masks" / src_mask.name),
            "source_t1_path": str(src_t1),
            "source_mask_path": str(src_mask),
            "current_v3_split": row["current_v3_split"],
            "fixed_test_hires": bool(row["fixed_test_hires"]),
            "quality_rank_all": int(row["quality_rank_all"]),
            "score": float(row["score"]),
            "fwhm_mm": float(row["fwhm_mm"]),
            "hi_freq_energy": float(row["hi_freq_energy"]),
            "lap_var": float(row["lap_var"]),
        })

    pd.DataFrame(out_rows, columns=CSV_FIELDS).to_csv(OUT_BASE / f"{split}.csv", index=False)


def main() -> None:
    for path in [
        ALL_OUT / "t1",
        ALL_OUT / "masks",
        FIXED_TEST_HIRES_CSV,
        CURRENT_TEST_LORES_CSV,
        CURRENT_TRAIN_HIRES_CSV,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    OUT_BASE.mkdir(parents=True, exist_ok=True)

    processed_pairs = load_processed_pairs()
    current_split_map = load_current_split_map()
    fixed_test_hires_keys = set(pd.read_csv(FIXED_TEST_HIRES_CSV)["key"].tolist())

    if len(fixed_test_hires_keys) != 138:
        raise RuntimeError(f"Expected 138 fixed HiRes keys, found {len(fixed_test_hires_keys)}")

    quality_df = compute_quality_manifest(processed_pairs)
    quality_df["current_v3_split"] = quality_df["key"].map(current_split_map).fillna("unassigned")
    quality_df["fixed_test_hires"] = quality_df["key"].isin(fixed_test_hires_keys)

    if quality_df["fixed_test_hires"].sum() != len(fixed_test_hires_keys):
        raise RuntimeError("Some fixed test_hires keys are missing from the processed pool")

    candidate_df = quality_df.loc[~quality_df["fixed_test_hires"]].copy()
    if len(candidate_df) != len(quality_df) - len(fixed_test_hires_keys):
        raise RuntimeError("Candidate pool size mismatch after removing fixed test_hires")

    if len(candidate_df) < TRAIN_LOW_COUNT:
        raise RuntimeError(
            f"Candidate pool too small for TRAIN_LOW_COUNT={TRAIN_LOW_COUNT}: {len(candidate_df)}"
        )

    train_low_keys = set(candidate_df.iloc[:TRAIN_LOW_COUNT]["key"].tolist())
    midres_keys = set(candidate_df.iloc[TRAIN_LOW_COUNT:]["key"].tolist())

    quality_df["split"] = np.where(
        quality_df["fixed_test_hires"],
        "test_hires",
        np.where(quality_df["key"].isin(train_low_keys), "train_low_quality", "midres"),
    )
    quality_df["manifest_source"] = quality_df["split"].map({
        "test_hires": "fixed_current_test_hires",
        "train_low_quality": "lowest_quality_522_of_non_hires_pool",
        "midres": "remaining_non_hires_after_low_quality_train_selection",
    })

    split_dfs = {
        split: quality_df.loc[quality_df["split"] == split].copy().sort_values(["score", "key"])
        for split in ("train_low_quality", "midres", "test_hires")
    }

    expected_counts = {
        "train_low_quality": TRAIN_LOW_COUNT,
        "midres": len(candidate_df) - TRAIN_LOW_COUNT,
        "test_hires": len(fixed_test_hires_keys),
    }
    for split, expected in expected_counts.items():
        actual = len(split_dfs[split])
        if actual != expected:
            raise RuntimeError(f"{split} count mismatch: expected {expected}, got {actual}")

    for split, df_split in split_dfs.items():
        sync_split_files(split, df_split, processed_pairs)
        write_split_csv(split, df_split, processed_pairs)

    quality_df.to_csv(ALL_CASES_CSV, index=False)

    train_scores = split_dfs["train_low_quality"]["score"].to_numpy(float)
    mid_scores = split_dfs["midres"]["score"].to_numpy(float)

    summary = {
        "source_processed_pool_size": int(len(quality_df)),
        "fixed_test_hires_count": int(len(fixed_test_hires_keys)),
        "candidate_pool_size": int(len(candidate_df)),
        "train_low_quality_count": int(len(split_dfs["train_low_quality"])),
        "midres_count": int(len(split_dfs["midres"])),
        "n_jobs": int(N_JOBS),
        "train_low_quality_score_max": float(np.max(train_scores)),
        "midres_score_min": float(np.min(mid_scores)),
        "fixed_test_hires_keys_csv": str(FIXED_TEST_HIRES_CSV),
        "counts_by_split": {split: int(len(df_split)) for split, df_split in split_dfs.items()},
        "dataset_mix_by_split": {
            split: dict(Counter(df_split["dataset"].tolist()))
            for split, df_split in split_dfs.items()
        },
        "prior_v3_mix_by_split": {
            split: dict(Counter(df_split["current_v3_split"].tolist()))
            for split, df_split in split_dfs.items()
        },
        "paths": {
            "out_base": str(OUT_BASE),
            "all_cases_quality_manifest": str(ALL_CASES_CSV),
            "summary_json": str(SUMMARY_JSON),
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    print("Built low-quality training split")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
