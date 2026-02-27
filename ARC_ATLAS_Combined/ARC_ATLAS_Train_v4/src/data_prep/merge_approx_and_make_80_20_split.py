#!/usr/bin/env python3
"""
Merge Approx_Numeracy cases into train_combined and build an 80/20 random split.

This script:
1) Copies Approx_Numeracy images/masks into:
     data/processed/train_combined/{t1,masks}
   using a dataset slug prefix to avoid collisions.
2) Rebuilds train_combined/manifest.csv.
3) Creates:
     data/splits/80_20_random/{train,test}/{t1,masks}
     data/splits/80_20_random/{train.csv,test.csv}
     data/splits/80_20_random/{train, test}/manifest.csv
     data/splits/80_20_random/meta/{all_cases.csv, split_summary.json}
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List


def _read_manifest(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def _write_manifest(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["slug", "key", "t1", "mask"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _iter_nii_gz(root: Path) -> Iterable[Path]:
    return sorted(p for p in root.glob("*.nii.gz") if p.is_file())


def _subject_from_key(key: str) -> str:
    m = re.match(r"^(sub-[^_]+)", key or "")
    return m.group(1) if m else (key.split("_")[0] if key else "")


def _copy_case(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if src.stat().st_size != dst.stat().st_size:
            raise RuntimeError(f"Existing file size mismatch:\n  src={src}\n  dst={dst}")
        return
    shutil.copy2(src, dst)


def _build_rows_from_combined_dirs(combined_t1: Path, combined_masks: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for t1 in _iter_nii_gz(combined_t1):
        expected_mask_name = t1.name.replace("_T1w_MNI_norm", "_lesion_mask_MNI_clean")
        mask = combined_masks / expected_mask_name
        if not mask.exists():
            continue
        if "__" in t1.name:
            slug, key = t1.name.split("__", 1)
        else:
            slug, key = "", t1.name
        rows.append(
            {
                "slug": slug,
                "key": key,
                "t1": str(t1.resolve()),
                "mask": str(mask.resolve()),
            }
        )
    rows.sort(key=lambda r: (r.get("slug", ""), r.get("key", "")))
    return rows


def merge_approx_into_combined(
    combined_t1: Path,
    combined_masks: Path,
    combined_manifest: Path,
    approx_images: Path,
    approx_masks: Path,
    approx_slug: str,
) -> List[Dict[str, str]]:
    combined_t1.mkdir(parents=True, exist_ok=True)
    combined_masks.mkdir(parents=True, exist_ok=True)

    approx_rows: List[Dict[str, str]] = []
    missing_masks = 0
    for img in _iter_nii_gz(approx_images):
        msk_name = img.name.replace("_T1w_MNI_norm", "_lesion_mask_MNI_clean")
        msk = approx_masks / msk_name
        if not msk.exists():
            missing_masks += 1
            continue

        out_img = combined_t1 / f"{approx_slug}__{img.name}"
        out_msk = combined_masks / f"{approx_slug}__{msk.name}"
        _copy_case(img, out_img)
        _copy_case(msk, out_msk)
        approx_rows.append({"slug": approx_slug, "key": img.name, "t1": str(out_img.resolve()), "mask": str(out_msk.resolve())})

    # Rebuild manifest from actual files to avoid losing existing cohorts.
    merged_rows = _build_rows_from_combined_dirs(combined_t1, combined_masks)
    _write_manifest(combined_manifest, merged_rows)

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "combined_manifest": str(combined_manifest.resolve()),
        "total_cases": len(merged_rows),
        "slug_counts": dict(Counter((r.get("slug") or "") for r in merged_rows)),
        "approx_added_or_refreshed": len(approx_rows),
        "approx_missing_masks": int(missing_masks),
    }
    with (combined_manifest.parent / "manifest_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return merged_rows


def create_80_20_split(
    rows: List[Dict[str, str]],
    split_root: Path,
    seed: int,
    test_fraction: float,
    overwrite: bool,
) -> Dict[str, object]:
    if overwrite and split_root.exists():
        shutil.rmtree(split_root)

    train_t1 = split_root / "train" / "t1"
    train_m = split_root / "train" / "masks"
    test_t1 = split_root / "test" / "t1"
    test_m = split_root / "test" / "masks"
    for d in (train_t1, train_m, test_t1, test_m, split_root / "meta"):
        d.mkdir(parents=True, exist_ok=True)

    valid_rows: List[Dict[str, str]] = []
    for r in rows:
        t1 = Path(r["t1"])
        m = Path(r["mask"])
        if t1.exists() and m.exists():
            valid_rows.append(r)

    if not valid_rows:
        raise RuntimeError("No valid rows with existing t1/mask paths found.")

    rng = random.Random(seed)
    shuffled = valid_rows[:]
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_test = max(1, int(round(n_total * test_fraction)))
    n_test = min(n_test, n_total - 1) if n_total > 1 else 1
    test_rows = shuffled[:n_test]
    train_rows = shuffled[n_test:]

    if not train_rows:
        raise RuntimeError("Train split is empty. Adjust split fraction.")

    def _materialize(rows_in: List[Dict[str, str]], out_t1: Path, out_m: Path, split_name: str):
        out_rows = []
        for r in rows_in:
            src_t1 = Path(r["t1"])
            src_m = Path(r["mask"])
            dst_t1 = out_t1 / src_t1.name
            dst_m = out_m / src_m.name
            _copy_case(src_t1, dst_t1)
            _copy_case(src_m, dst_m)
            key = r.get("key", src_t1.name)
            out_rows.append(
                {
                    "split": split_name,
                    "slug": r.get("slug", ""),
                    "key": key,
                    "subject": _subject_from_key(key),
                    "t1_path": str(dst_t1.resolve()),
                    "mask_path": str(dst_m.resolve()),
                }
            )
        return out_rows

    train_out = _materialize(train_rows, train_t1, train_m, "train")
    test_out = _materialize(test_rows, test_t1, test_m, "test")
    all_out = train_out + test_out

    def _write_split_csv(path: Path, rows_in: List[Dict[str, str]]) -> None:
        with path.open("w", newline="") as f:
            fields = ["split", "slug", "key", "subject", "t1_path", "mask_path"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows_in)

    _write_split_csv(split_root / "train.csv", train_out)
    _write_split_csv(split_root / "test.csv", test_out)
    _write_split_csv(split_root / "meta" / "all_cases.csv", all_out)

    # Training loader-friendly per-split manifests.
    def _write_split_manifest(path: Path, rows_in: List[Dict[str, str]]) -> None:
        with path.open("w", newline="") as f:
            fields = ["slug", "key", "t1", "mask"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows_in:
                w.writerow(
                    {
                        "slug": r["slug"],
                        "key": r["key"],
                        "t1": r["t1_path"],
                        "mask": r["mask_path"],
                    }
                )

    _write_split_manifest(split_root / "train" / "manifest.csv", train_out)
    _write_split_manifest(split_root / "test" / "manifest.csv", test_out)

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "test_fraction": test_fraction,
        "total_cases": n_total,
        "train_cases": len(train_out),
        "test_cases": len(test_out),
        "slug_counts_total": dict(Counter(r["slug"] for r in all_out)),
        "slug_counts_train": dict(Counter(r["slug"] for r in train_out)),
        "slug_counts_test": dict(Counter(r["slug"] for r in test_out)),
    }
    with (split_root / "meta" / "split_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="Merge Approx_Numeracy into combined data and create 80/20 split.")
    p.add_argument(
        "--combined-root",
        type=Path,
        default=project_root / "data" / "processed" / "train_combined",
        help="Root containing existing combined t1/masks/manifest.csv",
    )
    p.add_argument(
        "--approx-images",
        type=Path,
        default=Path("/home/rbielski/stroke_cleaned/Approx_Numeracy_Processed/Registered_Normalized_Images"),
        help="Approx_Numeracy normalized image folder",
    )
    p.add_argument(
        "--approx-masks",
        type=Path,
        default=Path("/home/rbielski/stroke_cleaned/Approx_Numeracy_Processed/Registered_Normalized_Masks"),
        help="Approx_Numeracy normalized mask folder",
    )
    p.add_argument(
        "--approx-slug",
        type=str,
        default="Approx-Numeracy-Processed",
        help="Slug prefix for copied Approx_Numeracy files in train_combined",
    )
    p.add_argument(
        "--split-root",
        type=Path,
        default=project_root / "data" / "splits" / "80_20_random",
        help="Destination split root for new random split",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    p.add_argument("--test-fraction", type=float, default=0.2, help="Fraction assigned to test split.")
    p.add_argument("--overwrite-split", action="store_true", help="Delete and rebuild split root if it exists.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    combined_t1 = args.combined_root / "t1"
    combined_m = args.combined_root / "masks"
    combined_manifest = args.combined_root / "manifest.csv"

    if not args.approx_images.exists():
        raise SystemExit(f"Approx image dir missing: {args.approx_images}")
    if not args.approx_masks.exists():
        raise SystemExit(f"Approx mask dir missing: {args.approx_masks}")
    if not combined_t1.exists() or not combined_m.exists():
        raise SystemExit(f"Combined root missing t1/masks: {args.combined_root}")

    merged_rows = merge_approx_into_combined(
        combined_t1=combined_t1,
        combined_masks=combined_m,
        combined_manifest=combined_manifest,
        approx_images=args.approx_images,
        approx_masks=args.approx_masks,
        approx_slug=args.approx_slug,
    )
    print(f"[merge] total combined rows: {len(merged_rows)}")
    print(f"[merge] manifest: {combined_manifest}")

    summary = create_80_20_split(
        rows=merged_rows,
        split_root=args.split_root,
        seed=args.seed,
        test_fraction=args.test_fraction,
        overwrite=args.overwrite_split,
    )
    print(f"[split] root: {args.split_root}")
    print(
        f"[split] train={summary['train_cases']} "
        f"test={summary['test_cases']} total={summary['total_cases']}"
    )


if __name__ == "__main__":
    main()
