"""
ATLAS R2 nnU-Net pipeline helpers.

This module replaces the old TensorFlow patch-training notebook path with the
workflow that is much closer to the ATLAS leaderboard systems:

- full-resolution 3D nnU-Net data layout
- five held-out folds
- fold-specific CV prediction to avoid leakage
- all-fold inference for final/test predictions
- adaptive connected-component postprocessing inspired by the MAPPING entry

The heavy training is still delegated to nnU-Net. These helpers make the local
data layout, command construction, postprocessing, and evaluation reproducible.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import nibabel as nib
import numpy as np


DEFAULT_DATASET_ID = 701
DEFAULT_DATASET_NAME = "ARC_ATLAS_TrainV4Native"
DEFAULT_MANIFEST = Path("data/splits/90_10_random/train/manifest.csv")
DEFAULT_NNUNET_ROOT = Path("data/splits/90_10_random/train/nnunet_view")
DEFAULT_RANDOM_SEED = 20260501


@dataclass(frozen=True)
class NnUNetLayout:
    project_root: Path
    nnunet_root: Path
    dataset_id: int = DEFAULT_DATASET_ID
    dataset_name: str = DEFAULT_DATASET_NAME

    @property
    def raw(self) -> Path:
        return self.nnunet_root / "nnUNet_raw"

    @property
    def preprocessed(self) -> Path:
        return self.nnunet_root / "nnUNet_preprocessed"

    @property
    def results(self) -> Path:
        return self.nnunet_root / "nnUNet_results"

    @property
    def dataset_folder_name(self) -> str:
        return f"Dataset{self.dataset_id:03d}_{self.dataset_name}"

    @property
    def raw_dataset_dir(self) -> Path:
        return self.raw / self.dataset_folder_name

    @property
    def preprocessed_dataset_dir(self) -> Path:
        return self.preprocessed / self.dataset_folder_name

    @property
    def mapping_csv(self) -> Path:
        return self.raw_dataset_dir / "case_mapping.csv"

    @property
    def folds_csv(self) -> Path:
        return self.raw_dataset_dir / "fold_index.csv"

    @property
    def splits_json(self) -> Path:
        return self.preprocessed_dataset_dir / "splits_final.json"

    def env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["nnUNet_raw"] = str(self.raw)
        env["nnUNet_preprocessed"] = str(self.preprocessed)
        env["nnUNet_results"] = str(self.results)
        return env


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_project_path(project_root: Path, path: Path | str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else project_root / p


def dataset_json(num_training: int, name: str) -> dict:
    return {
        "name": name,
        "description": "ARC/ATLAS v4 training split, exposed in nnU-Net v2 layout without resampling or target-shape resizing.",
        "reference": "Local ARC_ATLAS_Train_v4 data/splits/90_10_random/train",
        "licence": "See source dataset terms.",
        "release": "local",
        "channel_names": {"0": "T1"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": num_training,
        "file_ending": ".nii.gz",
    }


def read_manifest(manifest_csv: Path, source_filter: str | None = None, max_cases: int | None = None) -> list[dict[str, str]]:
    with manifest_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if source_filter:
        needle = source_filter.lower()
        rows = [
            r for r in rows
            if needle in r.get("slug", "").lower() or needle in r.get("key", "").lower()
        ]
    if max_cases:
        rows = rows[:max_cases]

    missing = [r for r in rows if not Path(r["t1"]).exists() or not Path(r["mask"]).exists()]
    if missing:
        sample = missing[0]
        raise FileNotFoundError(
            f"{len(missing)} manifest rows point at missing files. "
            f"First missing row: t1={sample.get('t1')} mask={sample.get('mask')}"
        )
    return rows


def sanitize_case_id(index: int, row: dict[str, str]) -> str:
    key = row.get("key", "")
    match = re.search(r"sub[-_][A-Za-z0-9]+", key)
    subject = match.group(0) if match else Path(key).name.split(".")[0]
    subject = re.sub(r"[^A-Za-z0-9]+", "_", subject).strip("_").lower()
    return f"case_{index:04d}_{subject}"


def lesion_group(voxels: int) -> str:
    if voxels < 100:
        return "000001_000099"
    if voxels < 1000:
        return "000100_000999"
    if voxels < 10000:
        return "001000_009999"
    return "010000_plus"


def _replace_or_link(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src)
    else:
        raise ValueError(f"Unsupported copy mode: {mode}")


def _save_binary_label(src: Path, dst: Path) -> tuple[int, tuple[int, ...], tuple[float, ...]]:
    img = nib.load(str(src))
    data = np.asanyarray(img.dataobj)
    label = (data > 0).astype(np.uint8)
    out = nib.Nifti1Image(label, img.affine, img.header)
    out.set_data_dtype(np.uint8)
    nib.save(out, str(dst))
    return int(label.sum()), tuple(int(x) for x in label.shape), tuple(float(x) for x in img.header.get_zooms()[:3])


def make_stratified_folds(case_rows: list[dict[str, str]], n_splits: int, seed: int) -> list[dict[str, list[str]]]:
    case_ids = np.array([r["case_id"] for r in case_rows])
    strata = np.array([r["lesion_group"] for r in case_rows])

    try:
        from sklearn.model_selection import StratifiedKFold

        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(case_ids, strata)
        folds: list[dict[str, list[str]]] = []
        for train_idx, val_idx in split_iter:
            folds.append({
                "train": sorted(case_ids[train_idx].tolist()),
                "val": sorted(case_ids[val_idx].tolist()),
            })
        return folds
    except Exception:
        rng = np.random.default_rng(seed)
        val_by_fold: list[list[str]] = [[] for _ in range(n_splits)]
        for group in sorted(set(strata.tolist())):
            group_cases = case_ids[strata == group].tolist()
            rng.shuffle(group_cases)
            for idx, case_id in enumerate(group_cases):
                val_by_fold[idx % n_splits].append(case_id)
        if len(case_ids) >= n_splits and any(not fold for fold in val_by_fold):
            shuffled = case_ids.tolist()
            rng.shuffle(shuffled)
            val_by_fold = [[] for _ in range(n_splits)]
            for idx, case_id in enumerate(shuffled):
                val_by_fold[idx % n_splits].append(case_id)

    folds: list[dict[str, list[str]]] = []
    all_cases = set(case_ids.tolist())
    for val_cases in val_by_fold:
        val = set(val_cases)
        folds.append({
            "train": sorted(all_cases - val),
            "val": sorted(val),
        })
    return folds


def prepare_nnunet_dataset(
    manifest_csv: Path,
    layout: NnUNetLayout,
    *,
    source_filter: str | None = None,
    max_cases: int | None = None,
    copy_mode: str = "symlink",
    overwrite: bool = False,
    n_splits: int = 5,
    seed: int = DEFAULT_RANDOM_SEED,
) -> dict:
    manifest_csv = resolve_project_path(layout.project_root, manifest_csv)
    rows = read_manifest(manifest_csv, source_filter=source_filter, max_cases=max_cases)
    if not rows:
        raise ValueError(f"No cases found in {manifest_csv}")

    raw_dir = layout.raw_dataset_dir
    if raw_dir.exists() and overwrite:
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    images_tr = raw_dir / "imagesTr"
    labels_tr = raw_dir / "labelsTr"
    images_ts = raw_dir / "imagesTs"
    for d in (images_tr, labels_tr, images_ts, layout.preprocessed_dataset_dir, layout.results):
        d.mkdir(parents=True, exist_ok=True)

    case_rows: list[dict[str, str]] = []
    for idx, row in enumerate(rows, start=1):
        case_id = sanitize_case_id(idx, row)
        image_src = Path(row["t1"]).resolve()
        label_src = Path(row["mask"]).resolve()
        image_dst = images_tr / f"{case_id}_0000.nii.gz"
        label_dst = labels_tr / f"{case_id}.nii.gz"

        if not overwrite and (image_dst.exists() or image_dst.is_symlink() or label_dst.exists()):
            raise FileExistsError(
                f"Converted files already exist for {case_id}. "
                "Pass overwrite=True or --overwrite to rebuild the nnU-Net dataset."
            )

        _replace_or_link(image_src, image_dst, copy_mode)
        voxels, shape, zooms = _save_binary_label(label_src, label_dst)
        case_rows.append({
            "case_id": case_id,
            "fold": "",
            "source_slug": row.get("slug", ""),
            "key": row.get("key", ""),
            "t1": str(image_src),
            "mask": str(label_src),
            "nnunet_image": str(image_dst),
            "nnunet_label": str(label_dst),
            "lesion_voxels": str(voxels),
            "lesion_group": lesion_group(voxels),
            "shape": "x".join(map(str, shape)),
            "spacing": "x".join(f"{z:.6g}" for z in zooms),
        })

    folds = make_stratified_folds(case_rows, n_splits=n_splits, seed=seed)
    fold_by_case: dict[str, int] = {}
    for fold_idx, fold in enumerate(folds):
        for case_id in fold["val"]:
            fold_by_case[case_id] = fold_idx
    for row in case_rows:
        row["fold"] = str(fold_by_case[row["case_id"]])

    with (raw_dir / "dataset.json").open("w") as f:
        json.dump(dataset_json(len(case_rows), layout.dataset_folder_name), f, indent=2)

    with layout.mapping_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(case_rows[0].keys()))
        writer.writeheader()
        writer.writerows(case_rows)

    with layout.folds_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "fold", "lesion_voxels", "lesion_group", "key"])
        writer.writeheader()
        for row in case_rows:
            writer.writerow({
                "case_id": row["case_id"],
                "fold": row["fold"],
                "lesion_voxels": row["lesion_voxels"],
                "lesion_group": row["lesion_group"],
                "key": row["key"],
            })

    with layout.splits_json.open("w") as f:
        json.dump(folds, f, indent=2)

    summary = {
        "source_manifest": str(manifest_csv),
        "dataset_dir": str(raw_dir),
        "preprocessed_dir": str(layout.preprocessed_dataset_dir),
        "results_dir": str(layout.results),
        "dataset_id": layout.dataset_id,
        "dataset_name": layout.dataset_folder_name,
        "num_cases": len(case_rows),
        "copy_mode": copy_mode,
        "folds": [
            {"fold": i, "train": len(fold["train"]), "val": len(fold["val"])}
            for i, fold in enumerate(folds)
        ],
        "lesion_groups": {
            group: sum(1 for row in case_rows if row["lesion_group"] == group)
            for group in sorted({row["lesion_group"] for row in case_rows})
        },
        "sources": {
            source: sum(1 for row in case_rows if row["source_slug"] == source)
            for source in sorted({row["source_slug"] for row in case_rows})
        },
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with (raw_dir / "conversion_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def load_case_mapping(mapping_csv: Path) -> list[dict[str, str]]:
    with mapping_csv.open(newline="") as f:
        return list(csv.DictReader(f))


def load_splits(splits_json: Path) -> list[dict[str, list[str]]]:
    with splits_json.open() as f:
        return json.load(f)


def preflight(layout: NnUNetLayout) -> dict:
    checks = {
        "python": sys.executable,
        "nnUNet_raw": str(layout.raw),
        "nnUNet_preprocessed": str(layout.preprocessed),
        "nnUNet_results": str(layout.results),
        "nnUNetv2_plan_and_preprocess": str(resolve_executable("nnUNetv2_plan_and_preprocess") or ""),
        "nnUNetv2_train": str(resolve_executable("nnUNetv2_train") or ""),
        "nnUNetv2_predict": str(resolve_executable("nnUNetv2_predict") or ""),
    }
    for module in ("torch", "nnunetv2", "nibabel", "scipy", "sklearn"):
        try:
            mod = __import__(module)
            checks[f"python:{module}"] = getattr(mod, "__version__", "installed")
        except Exception as exc:
            checks[f"python:{module}"] = f"missing: {exc.__class__.__name__}"
    return checks


def resolve_executable(name: str) -> Path | None:
    found = shutil.which(name)
    if found:
        return Path(found)
    sibling = Path(sys.executable).resolve().parent / name
    if sibling.exists():
        return sibling
    return None


def print_preflight(layout: NnUNetLayout) -> None:
    checks = preflight(layout)
    for key, value in checks.items():
        print(f"{key:32s} {value}")


def run_command(cmd: Sequence[str], layout: NnUNetLayout, *, dry_run: bool = False) -> subprocess.CompletedProcess | None:
    cmd = list(cmd)
    if cmd and str(cmd[0]).startswith("nnUNetv2_"):
        resolved = resolve_executable(str(cmd[0]))
        if resolved is None:
            raise FileNotFoundError(f"Cannot find {cmd[0]}. Install requirements_nnunet.txt in this Python environment.")
        cmd[0] = str(resolved)
    rendered = " ".join(str(c) for c in cmd)
    print(rendered)
    if dry_run:
        return None
    return subprocess.run([str(c) for c in cmd], check=True, env=layout.env())


def plan_and_preprocess(
    layout: NnUNetLayout,
    *,
    verify: bool = True,
    configurations: Sequence[str] = ("3d_fullres",),
    num_processes: Sequence[int] | None = None,
    gpu_memory_target: float | None = None,
    overwrite_plans_name: str | None = None,
    no_preprocess: bool = False,
    dry_run: bool = False,
) -> subprocess.CompletedProcess | None:
    cmd: list[str] = ["nnUNetv2_plan_and_preprocess", "-d", str(layout.dataset_id)]
    if verify:
        cmd.append("--verify_dataset_integrity")
    if no_preprocess:
        cmd.append("--no_pp")
    if configurations:
        cmd.extend(["-c", *configurations])
    if num_processes:
        cmd.extend(["-np", *[str(n) for n in num_processes]])
    if gpu_memory_target is not None:
        cmd.extend(["-gpu_memory_target", str(gpu_memory_target)])
    if overwrite_plans_name:
        cmd.extend(["-overwrite_plans_name", overwrite_plans_name])
    return run_command(cmd, layout, dry_run=dry_run)


def train_fold(
    layout: NnUNetLayout,
    fold: int | str,
    *,
    configuration: str = "3d_fullres",
    trainer: str | None = None,
    plans: str | None = None,
    save_npz: bool = True,
    dry_run: bool = False,
) -> subprocess.CompletedProcess | None:
    cmd: list[str] = ["nnUNetv2_train", str(layout.dataset_id), configuration, str(fold)]
    if trainer:
        cmd.extend(["-tr", trainer])
    if plans:
        cmd.extend(["-p", plans])
    if save_npz:
        cmd.append("--npz")

    plans_prefix = plans or "nnUNetPlans"
    expected_preprocessed = layout.preprocessed_dataset_dir / f"{plans_prefix}_{configuration}"
    if not dry_run and not expected_preprocessed.exists():
        raise FileNotFoundError(
            f"Missing preprocessed nnU-Net data: {expected_preprocessed}\n"
            "Run planning/preprocessing first with RUN_PLAN_AND_PREPROCESS=True in the notebook, "
            "or from the shell:\n"
            f"  {sys.executable} src/atlas_nnunet_pipeline.py plan "
            f"--project-root {layout.project_root} --gpu-memory-target 24 --plans-name {plans_prefix}"
        )
    return run_command(cmd, layout, dry_run=dry_run)


def train_all_folds(
    layout: NnUNetLayout,
    *,
    configuration: str = "3d_fullres",
    trainer: str | None = None,
    plans: str | None = None,
    dry_run: bool = False,
) -> None:
    for fold in range(5):
        train_fold(layout, fold, configuration=configuration, trainer=trainer, plans=plans, dry_run=dry_run)


def make_prediction_input_for_cases(
    layout: NnUNetLayout,
    case_ids: Iterable[str],
    output_dir: Path,
    *,
    overwrite: bool = True,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for old in output_dir.glob("*.nii.gz"):
            old.unlink()
    image_dir = layout.raw_dataset_dir / "imagesTr"
    for case_id in case_ids:
        src = image_dir / f"{case_id}_0000.nii.gz"
        dst = output_dir / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)
    return output_dir


def prepare_prediction_input_from_manifest(
    manifest_csv: Path,
    output_dir: Path,
    *,
    project_root: Path,
    source_filter: str | None = None,
    max_cases: int | None = None,
    copy_mode: str = "symlink",
    overwrite: bool = False,
) -> Path:
    manifest_csv = resolve_project_path(project_root, manifest_csv)
    output_dir = resolve_project_path(project_root, output_dir)
    rows = read_manifest(manifest_csv, source_filter=source_filter, max_cases=max_cases)
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping_rows: list[dict[str, str]] = []
    for idx, row in enumerate(rows, start=1):
        case_id = sanitize_case_id(idx, row)
        src = Path(row["t1"]).resolve()
        dst = output_dir / f"{case_id}_0000.nii.gz"
        if dst.exists() or dst.is_symlink():
            if not overwrite:
                raise FileExistsError(f"Prediction input already exists: {dst}")
            dst.unlink()
        _replace_or_link(src, dst, copy_mode)
        mapping_rows.append({
            "case_id": case_id,
            "source_slug": row.get("slug", ""),
            "key": row.get("key", ""),
            "t1": str(src),
            "mask": row.get("mask", ""),
            "nnunet_image": str(dst),
        })

    if mapping_rows:
        with (output_dir / "case_mapping.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(mapping_rows[0].keys()))
            writer.writeheader()
            writer.writerows(mapping_rows)
    return output_dir


def predict(
    layout: NnUNetLayout,
    input_dir: Path,
    output_dir: Path,
    *,
    folds: Sequence[int | str] = (0, 1, 2, 3, 4),
    configuration: str = "3d_fullres",
    trainer: str | None = None,
    plans: str | None = None,
    save_probabilities: bool = True,
    dry_run: bool = False,
) -> subprocess.CompletedProcess | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd: list[str] = [
        "nnUNetv2_predict",
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-d", str(layout.dataset_id),
        "-c", configuration,
        "-f", *[str(f) for f in folds],
    ]
    if trainer:
        cmd.extend(["-tr", trainer])
    if plans:
        cmd.extend(["-p", plans])
    if save_probabilities:
        cmd.append("--save_probabilities")
    return run_command(cmd, layout, dry_run=dry_run)


def predict_cv_folds(
    layout: NnUNetLayout,
    output_root: Path,
    *,
    configuration: str = "3d_fullres",
    trainer: str | None = None,
    plans: str | None = None,
    dry_run: bool = False,
) -> list[Path]:
    splits = load_splits(layout.splits_json)
    output_dirs: list[Path] = []
    for fold_idx, split in enumerate(splits):
        fold_input = output_root / f"fold_{fold_idx}_input"
        fold_output = output_root / f"fold_{fold_idx}_pred"
        if not dry_run:
            make_prediction_input_for_cases(layout, split["val"], fold_input)
        predict(
            layout,
            fold_input,
            fold_output,
            folds=(fold_idx,),
            configuration=configuration,
            trainer=trainer,
            plans=plans,
            dry_run=dry_run,
        )
        output_dirs.append(fold_output)
    return output_dirs


def foreground_probability(npz_path: Path) -> np.ndarray:
    archive = np.load(str(npz_path))
    key = "softmax" if "softmax" in archive else "probabilities" if "probabilities" in archive else archive.files[0]
    arr = archive[key]
    if arr.ndim == 4 and arr.shape[0] >= 2:
        return np.asarray(arr[1], dtype=np.float32)
    if arr.ndim == 4 and arr.shape[-1] >= 2:
        return np.asarray(arr[..., 1], dtype=np.float32)
    if arr.ndim == 3:
        return np.asarray(arr, dtype=np.float32)
    raise ValueError(f"Cannot find foreground probabilities in {npz_path}; key={key} shape={arr.shape}")


def adaptive_component_mask(
    foreground: np.ndarray,
    *,
    small_volume_cutoff: int = 3000,
    small_big_threshold: float = 0.70,
    small_grow_threshold: float = 0.50,
    large_big_threshold: float = 0.55,
    large_grow_threshold: float = 0.50,
    max_small_seed_components: int = 4,
) -> np.ndarray:
    try:
        from scipy import ndimage
    except Exception as exc:
        raise RuntimeError("Adaptive postprocessing requires scipy. Install requirements_nnunet.txt.") from exc

    fg = np.asarray(foreground, dtype=np.float32)
    num_fg = int((fg > 0.5).sum())
    small_case = num_fg < small_volume_cutoff
    seed_threshold = small_big_threshold if small_case else large_big_threshold
    grow_threshold = small_grow_threshold if small_case else large_grow_threshold

    seeds = fg > seed_threshold
    grow = fg > grow_threshold

    labeled_seeds, n_seeds = ndimage.label(seeds)
    if n_seeds == 0:
        if not np.isfinite(fg).any() or float(fg.max()) <= 0:
            return np.zeros(fg.shape, dtype=np.uint8)
        return (fg > max(float(fg.max()) - 0.1, 0.0)).astype(np.uint8)

    if small_case and n_seeds > max_small_seed_components:
        component_stats: list[tuple[float, int, int]] = []
        for component_idx in range(1, n_seeds + 1):
            component = labeled_seeds == component_idx
            size = int(component.sum())
            mean_prob = float(fg[component].mean()) if size else 0.0
            component_stats.append((mean_prob, size, component_idx))
        drop_idx = sorted(component_stats, key=lambda x: (x[0], x[1]))[0][2]
        seeds = seeds & (labeled_seeds != drop_idx)
        labeled_seeds, n_seeds = ndimage.label(seeds)

    labeled_grow, n_grow = ndimage.label(grow)
    out = np.zeros(fg.shape, dtype=bool)
    for component_idx in range(1, n_grow + 1):
        component = labeled_grow == component_idx
        intersects_seed = np.any(labeled_seeds[component] > 0)
        if intersects_seed or not small_case:
            out |= component
    if not out.any():
        out = seeds
    return out.astype(np.uint8)


def save_mask_like(mask: np.ndarray, reference_path: Path, output_path: Path) -> None:
    ref = nib.load(str(reference_path))
    out = nib.Nifti1Image(mask.astype(np.uint8), ref.affine, ref.header)
    out.set_data_dtype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(output_path))


def reference_for_case(case_id: str, pred_dir: Path, mapping: dict[str, dict[str, str]]) -> Path:
    pred_nii = pred_dir / f"{case_id}.nii.gz"
    if pred_nii.exists():
        return pred_nii
    if case_id in mapping:
        return Path(mapping[case_id]["t1"])
    raise FileNotFoundError(f"No reference NIfTI found for {case_id}")


def postprocess_probabilities(
    prediction_dir: Path,
    output_dir: Path,
    *,
    mapping_csv: Path | None = None,
) -> list[Path]:
    mapping_rows = load_case_mapping(mapping_csv) if mapping_csv and mapping_csv.exists() else []
    mapping = {row["case_id"]: row for row in mapping_rows}
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    for npz_path in sorted(prediction_dir.glob("*.npz")):
        case_id = npz_path.name.removesuffix(".npz")
        fg = foreground_probability(npz_path)
        mask = adaptive_component_mask(fg)
        ref = reference_for_case(case_id, prediction_dir, mapping)
        out_path = output_dir / f"{case_id}.nii.gz"
        save_mask_like(mask, ref, out_path)
        outputs.append(out_path)
    if not outputs:
        raise FileNotFoundError(f"No probability .npz files found in {prediction_dir}")
    return outputs


def ensemble_probability_dirs(
    prediction_dirs: Sequence[Path],
    output_dir: Path,
    *,
    mapping_csv: Path | None = None,
) -> list[Path]:
    if not prediction_dirs:
        raise ValueError("prediction_dirs is empty")
    by_case: dict[str, list[Path]] = {}
    for pred_dir in prediction_dirs:
        for npz_path in pred_dir.glob("*.npz"):
            by_case.setdefault(npz_path.name.removesuffix(".npz"), []).append(npz_path)

    expected = len(prediction_dirs)
    missing = {case: len(paths) for case, paths in by_case.items() if len(paths) != expected}
    if missing:
        sample = list(missing.items())[:5]
        raise ValueError(f"Some cases are missing variant predictions: {sample}")

    mapping_rows = load_case_mapping(mapping_csv) if mapping_csv and mapping_csv.exists() else []
    mapping = {row["case_id"]: row for row in mapping_rows}
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for case_id, paths in sorted(by_case.items()):
        probs = [foreground_probability(path) for path in sorted(paths)]
        fg = np.mean(np.stack(probs, axis=0), axis=0)
        mask = adaptive_component_mask(fg)
        ref = reference_for_case(case_id, paths[0].parent, mapping)
        out_path = output_dir / f"{case_id}.nii.gz"
        save_mask_like(mask, ref, out_path)
        outputs.append(out_path)
    return outputs


def dice_score(pred: np.ndarray, true: np.ndarray) -> float:
    pred_bool = pred > 0
    true_bool = true > 0
    denom = int(pred_bool.sum() + true_bool.sum())
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred_bool, true_bool).sum() / denom)


def evaluate_predictions(prediction_dirs: Sequence[Path], mapping_csv: Path, output_csv: Path) -> dict:
    mapping_rows = load_case_mapping(mapping_csv)
    rows_by_case = {row["case_id"]: row for row in mapping_rows}
    pred_by_case: dict[str, Path] = {}
    for pred_dir in prediction_dirs:
        for pred in pred_dir.glob("*.nii.gz"):
            case_id = pred.name.removesuffix(".nii.gz")
            if case_id in rows_by_case:
                pred_by_case[case_id] = pred

    eval_rows: list[dict[str, str]] = []
    for case_id, row in sorted(rows_by_case.items()):
        pred_path = pred_by_case.get(case_id)
        if pred_path is None:
            continue
        true = np.asanyarray(nib.load(row["nnunet_label"]).dataobj) > 0
        pred = np.asanyarray(nib.load(str(pred_path)).dataobj) > 0
        if pred.shape != true.shape:
            raise ValueError(f"Shape mismatch for {case_id}: pred={pred.shape} true={true.shape}")
        dsc = dice_score(pred, true)
        eval_rows.append({
            "case_id": case_id,
            "fold": row["fold"],
            "dice": f"{dsc:.8f}",
            "pred_voxels": str(int(pred.sum())),
            "true_voxels": str(int(true.sum())),
            "lesion_group": row["lesion_group"],
            "prediction": str(pred_path),
            "label": row["nnunet_label"],
        })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        fieldnames = ["case_id", "fold", "dice", "pred_voxels", "true_voxels", "lesion_group", "prediction", "label"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(eval_rows)

    dice_values = [float(row["dice"]) for row in eval_rows]
    summary = {
        "n": len(eval_rows),
        "dice_mean": float(np.mean(dice_values)) if dice_values else None,
        "dice_median": float(np.median(dice_values)) if dice_values else None,
        "dice_by_group": {},
        "output_csv": str(output_csv),
    }
    for group in sorted({row["lesion_group"] for row in eval_rows}):
        vals = [float(row["dice"]) for row in eval_rows if row["lesion_group"] == group]
        summary["dice_by_group"][group] = {
            "n": len(vals),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
        }
    with output_csv.with_suffix(".summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def print_recommended_commands(layout: NnUNetLayout) -> None:
    python_bin = sys.executable
    print("# Environment")
    print(f"export nnUNet_raw={layout.raw}")
    print(f"export nnUNet_preprocessed={layout.preprocessed}")
    print(f"export nnUNet_results={layout.results}")
    print()
    print("# Plan and preprocess")
    print(f"nnUNetv2_plan_and_preprocess -d {layout.dataset_id} --verify_dataset_integrity -c 3d_fullres")
    print()
    print("# Train five full-resolution 3D folds")
    for fold in range(5):
        print(f"nnUNetv2_train {layout.dataset_id} 3d_fullres {fold} --npz")
    print()
    print("# Leak-free CV inference")
    print(f"{python_bin} src/atlas_nnunet_pipeline.py predict-cv --nnunet-root {layout.nnunet_root}")
    print()
    print("# Evaluate CV predictions")
    print(f"{python_bin} src/atlas_nnunet_pipeline.py evaluate --nnunet-root {layout.nnunet_root}")


def build_layout(args: argparse.Namespace) -> NnUNetLayout:
    project_root = Path(args.project_root).resolve()
    nnunet_root = resolve_project_path(project_root, args.nnunet_root).resolve()
    return NnUNetLayout(
        project_root=project_root,
        nnunet_root=nnunet_root,
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
    )


def parse_folds(value: str) -> list[int | str]:
    if value == "all":
        return [0, 1, 2, 3, 4]
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", type=Path, default=default_project_root())
    parser.add_argument("--nnunet-root", type=Path, default=DEFAULT_NNUNET_ROOT)
    parser.add_argument("--dataset-id", type=int, default=DEFAULT_DATASET_ID)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ATLAS nnU-Net v2 pipeline helpers")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check")
    add_common_args(p)

    p = sub.add_parser("prepare")
    add_common_args(p)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--source-filter")
    p.add_argument("--max-cases", type=int)
    p.add_argument("--copy-mode", choices=["symlink", "copy"], default="symlink")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)

    p = sub.add_parser("commands")
    add_common_args(p)

    p = sub.add_parser("plan")
    add_common_args(p)
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--configuration", action="append", default=None)
    p.add_argument("--np", dest="num_processes", type=int, action="append")
    p.add_argument("--gpu-memory-target", type=float)
    p.add_argument("--plans-name")
    p.add_argument("--no-preprocess", action="store_true")
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("train")
    add_common_args(p)
    p.add_argument("--fold", default="all")
    p.add_argument("--configuration", default="3d_fullres")
    p.add_argument("--trainer")
    p.add_argument("--plans")
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("predict-cv")
    add_common_args(p)
    p.add_argument("--output-root", type=Path, default=Path("runs/nnunet_cv_predictions"))
    p.add_argument("--configuration", default="3d_fullres")
    p.add_argument("--trainer")
    p.add_argument("--plans")
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("prepare-predict-input")
    add_common_args(p)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--source-filter")
    p.add_argument("--max-cases", type=int)
    p.add_argument("--copy-mode", choices=["symlink", "copy"], default="symlink")
    p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("predict")
    add_common_args(p)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--folds", default="all")
    p.add_argument("--configuration", default="3d_fullres")
    p.add_argument("--trainer")
    p.add_argument("--plans")
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("postprocess")
    add_common_args(p)
    p.add_argument("--prediction-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)

    p = sub.add_parser("evaluate")
    add_common_args(p)
    p.add_argument("--prediction-dir", type=Path, action="append")
    p.add_argument("--output-csv", type=Path, default=Path("runs/nnunet_cv_predictions/evaluation.csv"))

    args = parser.parse_args(argv)
    layout = build_layout(args)

    if args.command == "check":
        print_preflight(layout)
        return 0
    if args.command == "prepare":
        summary = prepare_nnunet_dataset(
            args.manifest,
            layout,
            source_filter=args.source_filter,
            max_cases=args.max_cases,
            copy_mode=args.copy_mode,
            overwrite=args.overwrite,
            n_splits=args.folds,
            seed=args.seed,
        )
        print(json.dumps(summary, indent=2))
        return 0
    if args.command == "commands":
        print_recommended_commands(layout)
        return 0
    if args.command == "plan":
        plan_and_preprocess(
            layout,
            verify=not args.no_verify,
            configurations=tuple(args.configuration or ["3d_fullres"]),
            num_processes=args.num_processes,
            gpu_memory_target=args.gpu_memory_target,
            overwrite_plans_name=args.plans_name,
            no_preprocess=args.no_preprocess,
            dry_run=args.dry_run,
        )
        return 0
    if args.command == "train":
        folds = parse_folds(args.fold)
        for fold in folds:
            train_fold(
                layout,
                fold,
                configuration=args.configuration,
                trainer=args.trainer,
                plans=args.plans,
                dry_run=args.dry_run,
            )
        return 0
    if args.command == "predict-cv":
        output_root = resolve_project_path(layout.project_root, args.output_root)
        predict_cv_folds(
            layout,
            output_root,
            configuration=args.configuration,
            trainer=args.trainer,
            plans=args.plans,
            dry_run=args.dry_run,
        )
        return 0
    if args.command == "prepare-predict-input":
        out_dir = prepare_prediction_input_from_manifest(
            args.manifest,
            args.output_dir,
            project_root=layout.project_root,
            source_filter=args.source_filter,
            max_cases=args.max_cases,
            copy_mode=args.copy_mode,
            overwrite=args.overwrite,
        )
        print(out_dir)
        return 0
    if args.command == "predict":
        predict(
            layout,
            resolve_project_path(layout.project_root, args.input_dir),
            resolve_project_path(layout.project_root, args.output_dir),
            folds=parse_folds(args.folds),
            configuration=args.configuration,
            trainer=args.trainer,
            plans=args.plans,
            dry_run=args.dry_run,
        )
        return 0
    if args.command == "postprocess":
        postprocess_probabilities(
            resolve_project_path(layout.project_root, args.prediction_dir),
            resolve_project_path(layout.project_root, args.output_dir),
            mapping_csv=layout.mapping_csv,
        )
        return 0
    if args.command == "evaluate":
        pred_dirs = args.prediction_dir
        if not pred_dirs:
            default_root = resolve_project_path(layout.project_root, Path("runs/nnunet_cv_predictions"))
            pred_dirs = sorted(default_root.glob("fold_*_pred"))
        pred_dirs = [resolve_project_path(layout.project_root, p) for p in pred_dirs]
        summary = evaluate_predictions(
            pred_dirs,
            layout.mapping_csv,
            resolve_project_path(layout.project_root, args.output_csv),
        )
        print(json.dumps(summary, indent=2))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
