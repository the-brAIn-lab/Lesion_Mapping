"""
Materialize the ARC+ATLAS hires/lores 50/25/25 split into the local project
folder with real files (no symlinks).

The default source is the combined split produced earlier at:
  ../../ARC/ds004884/derivatives/aggregates/t1w_with_masks/
      mni_1mm_ants_fixed/_standardized/_resolution_v2/_splits_50_25_25
which contains symlinks into both ARC and ATLAS standardized datasets.

The default destination is:
  ./data/splits/50_25_25

Usage
-----
python -m data_prep.materialize_split \
    --source ../../ARC/.../_splits_50_25_25 \
    --dest   ./data/splits/50_25_25 \
    --metadata ../../ARC/.../_resolution_v2/_resolution_manifest_v2.csv

If your raw data live elsewhere, override the paths with CLI flags.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _first_existing_path(candidates: list[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


DEFAULT_SOURCE = _first_existing_path(
    [
        PROJECT_ROOT.parent
        / "ARC"
        / "ds004884"
        / "derivatives"
        / "aggregates"
        / "t1w_with_masks"
        / "mni_1mm_ants_fixed"
        / "_standardized"
        / "_resolution_v2"
        / "_splits_50_25_25",
        Path("/home/rbielski/ARC/ds004884/derivatives/aggregates/t1w_with_masks/mni_1mm_ants_fixed/_standardized/_resolution_v2/_splits_50_25_25"),
    ]
)

DEFAULT_DEST = PROJECT_ROOT / "data" / "splits" / "50_25_25"

DEFAULT_METADATA = _first_existing_path(
    [
        PROJECT_ROOT.parent
        / "ARC"
        / "ds004884"
        / "derivatives"
        / "aggregates"
        / "t1w_with_masks"
        / "mni_1mm_ants_fixed"
        / "_standardized"
        / "_resolution_v2"
        / "_resolution_manifest_v2.csv",
        Path("/home/rbielski/ARC/ds004884/derivatives/aggregates/t1w_with_masks/mni_1mm_ants_fixed/_standardized/_resolution_v2/_resolution_manifest_v2.csv"),
    ]
)


class CopyStats:
    def __init__(self) -> None:
        self.files = 0
        self.bytes = 0
        self.symlinks_found: list[Path] = []

    def add(self, path: Path) -> None:
        self.files += 1
        try:
            self.bytes += path.stat().st_size
        except FileNotFoundError:
            pass


def copy_tree_following_symlinks(src: Path, dst: Path, stats: CopyStats) -> None:
    """Recursively copy src -> dst, following symlinks to materialize real files."""
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        out = dst / rel
        if path.is_dir():
            out.mkdir(parents=True, exist_ok=True)
            continue
        if path.is_symlink():
            stats.symlinks_found.append(path)
            target = path.resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, out)
            stats.add(out)
        elif path.is_file():
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out)
            stats.add(out)


def write_manifest(dest_root: Path, extra_meta: Iterable[Path]) -> None:
    manifest_path = dest_root / "manifest.txt"
    lines = ["Dataset materialized into: " + str(dest_root)]
    for meta in extra_meta:
        if meta and meta.exists():
            dest_meta = dest_root / "meta" / meta.name
            dest_meta.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(meta, dest_meta)
            lines.append(f"Copied metadata: {dest_meta.relative_to(dest_root)}")
    manifest_path.write_text("\n".join(lines))


def rewrite_resolution_manifest(dest_root: Path) -> None:
    """Rewrites resolution_manifest_v2.csv (if present) to use relative paths inside dest_root.

    Original manifest paths point to the source standardized tree. For portability we map by
    filename into the copied split (train_hires/test_hires/test_lores).
    """
    import csv

    meta_src = dest_root / "meta" / "_resolution_manifest_v2.csv"
    if not meta_src.exists():
        return

    # Map filename -> relative path inside dest
    name_to_rel = {}
    for f in dest_root.rglob("*.nii.gz"):
        name_to_rel[f.name] = f.relative_to(dest_root)

    rows = []
    with open(meta_src, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t1_name = Path(row["t1_path"]).name
            msk_name = Path(row["mask_path"]).name
            row["t1_path"] = str(dest_root / name_to_rel.get(t1_name, Path(t1_name)))
            row["mask_path"] = str(dest_root / name_to_rel.get(msk_name, Path(msk_name)))
            rows.append(row)

    out_path = dest_root / "meta" / "_resolution_manifest_v2_local.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    # also overwrite the original to avoid absolute paths lingering
    meta_src.write_text(out_path.read_text())
    print("Rewrote manifest with local paths ->", out_path)


def rewrite_split_csvs(dest_root: Path) -> None:
    """Update train/test CSVs so t1_path/mask_path point inside dest_root."""
    import csv

    for csv_path in dest_root.glob("*.csv"):
        rows = list(csv.DictReader(open(csv_path)))
        if not rows:
            continue
        for r in rows:
            t1_name = Path(r["t1_path"]).name
            msk_name = Path(r["mask_path"]).name
            # infer subfolder based on CSV stem
            split_dir = dest_root / csv_path.stem
            r["t1_path"] = str((split_dir / "t1" / t1_name))
            r["mask_path"] = str((split_dir / "masks" / msk_name))
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)
        print("Localized paths in", csv_path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Split root containing train_hires/test_hires/test_lores")
    p.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="Destination root for fully materialized data")
    p.add_argument("--metadata", type=Path, default=DEFAULT_METADATA, help="Path to resolution manifest CSV to copy alongside data")
    p.add_argument("--overwrite", action="store_true", help="Delete destination before copying")
    args = p.parse_args()

    if args.overwrite and args.dest.exists():
        shutil.rmtree(args.dest)

    if not args.source.exists():
        raise SystemExit(f"Source split not found: {args.source}")

    stats = CopyStats()
    copy_tree_following_symlinks(args.source, args.dest, stats)
    write_manifest(args.dest, [args.metadata])
    rewrite_resolution_manifest(args.dest)
    rewrite_split_csvs(args.dest)

    print(f"Copied {stats.files} files into {args.dest}")
    print(f"Total size ~ {stats.bytes/1e9:.2f} GB")
    if stats.symlinks_found:
        print(f"Materialized {len(stats.symlinks_found)} symlinks → real files")
    # Final sanity: ensure no symlinks in dest
    dangling = list(args.dest.rglob("*"))
    leftover_links = [p for p in dangling if p.is_symlink()]
    if leftover_links:
        raise SystemExit(f"Found symlinks in dest (expected none): {leftover_links[:3]} ...")
    print("✅ Dataset is fully materialized (no symlinks).")


if __name__ == "__main__":
    main()
