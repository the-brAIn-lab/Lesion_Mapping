"""Prep utilities to standardize T1 + lesion masks into MNI and normalized form.

All paths are project-relative. External dependency: ANTs binaries (`antsRegistration`,
`antsApplyTransforms`) must be on PATH or provided via env vars ANTS_REG / ANTS_APPLY.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os, shutil, subprocess, re, json, csv, hashlib
import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
from nibabel.processing import resample_from_to
from typing import Iterable
from collections import defaultdict


def _ants_bins():
    reg = Path(shutil.which("antsRegistration") or "")
    apply = Path(shutil.which("antsApplyTransforms") or "")
    reg_env = Path(os.environ.get("ANTS_REG", reg)) if os.environ.get("ANTS_REG") else reg
    app_env = Path(os.environ.get("ANTS_APPLY", apply)) if os.environ.get("ANTS_APPLY") else apply
    if not reg_env.exists() or not app_env.exists():
        raise FileNotFoundError(
            "ANTs binaries not found. Set ANTS_REG / ANTS_APPLY env vars or add to PATH."
        )
    return reg_env, app_env


def _run(cmd: list[str]):
    print(">>", " ".join(map(str, cmd)))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0:
        print(res.stdout)
        raise RuntimeError("Command failed")
    return res.stdout


def _key_from_name(name: str) -> str | None:
    sub = re.search(r"(sub-[^_]+)", name)
    ses = re.search(r"(ses-[^_]+)", name)
    parts = [m.group(1) for m in (sub, ses) if m]
    return "_".join(parts) if parts else None


def _slug_from_raw(raw_root: Path, name: str) -> str:
    stem = raw_root.name
    safe = re.sub(r"[^a-zA-Z0-9]+", "-", f"{name}-{stem}").strip("-")
    digest = hashlib.md5(str(raw_root.resolve()).encode()).hexdigest()[:8]
    return f"{safe}-{digest}"


def _tpl_path(resolution: int = 1) -> Path:
    """Return TemplateFlow MNI T1 path for given resolution, robust to return type."""
    from templateflow.api import get as tf_get
    tpl = tf_get("MNI152NLin2009cAsym", resolution=resolution, suffix="T1w", desc="brain", extension="nii.gz")
    if isinstance(tpl, (list, tuple)):
        return Path(tpl[0])
    return Path(tpl)

def _find_existing_output(out_root: Path, raw_root: Path) -> Path | None:
    """Look for an existing prep folder whose marker matches raw_root."""
    raw_root = raw_root.resolve()
    for m in out_root.glob("*/source.json"):
        try:
            info = json.loads(m.read_text())
            if Path(info.get("raw_root", "")).resolve() == raw_root:
                return m.parent
        except Exception:
            continue
    return None


def list_pairs(raw_root: Path, t1_glob: str, mask_glob: str):
    t1s = list(raw_root.glob(t1_glob))
    masks = list(raw_root.glob(mask_glob))
    return _match_pairs(t1s, masks)


def _norm_key_from_name(name: str) -> str:
    base = name
    if base.endswith(".nii.gz"):
        base = base[:-7]
    elif base.endswith(".nii"):
        base = base[:-4]
    drop = [
        "_T1w_MNI_norm", "_T1w_MNI", "_T1w_brain", "_T1w", "_T1",
        "_lesion_mask_MNI_clean", "_lesion_mask_MNI", "_lesion_mask",
        "_desc-lesion_mask", "_mask", "mask"
    ]
    for sfx in drop:
        if base.endswith(sfx):
            base = base[: -len(sfx)]
    return base


def _match_pairs(t1s: list[Path], masks: list[Path]):
    """Match T1 and mask by the most specific key available.
    Priority: exact normalized basename -> unique; else sub/ses key unique. Ambiguous cases are skipped.
    """
    mask_by_base = defaultdict(list)
    mask_by_subses = defaultdict(list)
    for m in masks:
        base = _norm_key_from_name(m.name)
        mask_by_base[base].append(m)
        key = _key_from_name(m.name) or base
        mask_by_subses[key].append(m)

    pairs = []
    for t1 in t1s:
        base = _norm_key_from_name(t1.name)
        key = _key_from_name(t1.name) or base
        chosen = None
        if mask_by_base.get(base):
            if len(mask_by_base[base]) == 1:
                chosen = mask_by_base[base][0]
            else:
                print(f"[warn] multiple masks share base {base}; skipping")
                continue
        elif mask_by_subses.get(key):
            if len(mask_by_subses[key]) == 1:
                chosen = mask_by_subses[key][0]
            else:
                print(f"[warn] ambiguous masks for {t1.name} (key {key}): {len(mask_by_subses[key])}; skipping")
                continue
        if not chosen:
            print(f"[warn] no mask for {t1.name} (key {key}); skipping")
            continue
        pairs.append((t1, chosen, base))
    return pairs


def resample_mask_to_t1(mask: Path, t1: Path, out_path: Path):
    mi = nib.load(str(mask))
    ti = nib.load(str(t1))
    if mi.shape[:3] != ti.shape[:3] or not np.allclose(mi.affine, ti.affine, atol=1e-4):
        rs = resample_from_to(mi, (ti.shape, ti.affine), order=0)
        data = (rs.get_fdata() > 0.5).astype(np.uint8)
    else:
        data = (mi.get_fdata() > 0.5).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, ti.affine, ti.header), str(out_path))


def ants_register(t1: Path, prefix: Path, tpl: Path, reg_bin: Path, use_2mm: bool = True):
    tpl_reg = tpl
    if use_2mm:
        try:
            from templateflow.api import get as tf_get
            tpl2 = tf_get("MNI152NLin2009cAsym", resolution=2, suffix="T1w", desc="brain", extension="nii.gz")
            tpl_reg = Path(tpl2[0]) if isinstance(tpl2, (list, tuple)) else Path(tpl2)
        except Exception:
            tpl_reg = tpl
    _run([
        str(reg_bin), '-d','3',
        '-r', f'[{tpl_reg},{t1},1]',
        '-m', f'Mattes[{tpl_reg},{t1},1,32,Regular,0.25]',
        '-t','Rigid[0.1]','-c','1000x500x250','-s','3x2x1vox','-f','4x2x1',
        '-m', f'Mattes[{tpl_reg},{t1},1,32,Regular,0.25]',
        '-t','Affine[0.1]','-c','1000x500x250','-s','3x2x1vox','-f','4x2x1',
        '-m', f'CC[{tpl_reg},{t1},1,4]',
        '-t','SyN[0.1,3,0]','-c','60x40x20','-s','2x1x0vox','-f','4x2x1',
        '-o', f'[{prefix},{prefix}warped.nii.gz,{prefix}invwarped.nii.gz]'
    ])


def ants_apply(img_in: Path, ref: Path, xfm_prefix: Path, out_path: Path, apply_bin: Path, nn: bool=False):
    args = [str(apply_bin), '-d','3', '-i', str(img_in), '-r', str(ref), '-o', str(out_path)]
    if nn:
        args += ['-n','NearestNeighbor']
    args += ['-t', str(xfm_prefix)+'1Warp.nii.gz', '-t', str(xfm_prefix)+'0GenericAffine.mat']
    _run(args)


def normalize_t1(vol: np.ndarray) -> np.ndarray:
    nz = vol[vol>0]
    if nz.size == 0:
        return np.zeros_like(vol, np.float32)
    p1,p99 = np.percentile(nz,[1,99])
    vol = np.clip(vol, p1, p99)
    mu, sd = nz.mean(), nz.std()
    vol = (vol - mu)/(sd+1e-8)
    mn, mx = vol.min(), vol.max()
    return ((vol - mn)/(mx - mn + 1e-8)).astype(np.float32)


def largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, nlab = ndi.label(mask)
    if nlab <= 1:
        return mask.astype(np.uint8)
    sizes = np.bincount(labeled.ravel())
    keep = sizes[1:].argmax() + 1
    return (labeled == keep).astype(np.uint8)


def _bbox(mask: np.ndarray):
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return mins, maxs


def _overlap_score(mask: np.ndarray, brain: np.ndarray) -> float:
    inter = np.logical_and(mask > 0, brain > 0).sum()
    return inter / max((mask > 0).sum(), 1)


def _try_flips(mask: np.ndarray, brain: np.ndarray):
    base_score = _overlap_score(mask, brain)
    best = (mask, base_score, "none")
    flips = [
        (np.flip(mask, axis=0), "flip_x"),
        (np.flip(mask, axis=1), "flip_y"),
        (np.flip(mask, axis=2), "flip_z"),
    ]
    for flipped, tag in flips:
        score = _overlap_score(flipped, brain)
        if score > best[1]:
            best = (flipped, score, tag)
    return best


@dataclass
class DatasetConfig:
    name: str
    raw_root: Path | None = None
    images_dir: Path | None = None
    masks_dir: Path | None = None
    t1_glob: str = "**/*_T1w.nii.gz"
    mask_glob: str = "**/*mask*.nii.gz"
    overwrite: bool = False
    already_mni: bool = False


def run_prep(datasets: Iterable[DatasetConfig], out_root: Path, force_overwrite: bool = False):
    tpl = None
    reg_bin = apply_bin = None
    out_root.mkdir(parents=True, exist_ok=True)
    qc_rows = []
    outputs = []

    for ds in datasets:
        raw = Path(ds.raw_root).expanduser() if ds.raw_root else None
        img_root = Path(ds.images_dir).expanduser() if ds.images_dir else raw
        msk_root = Path(ds.masks_dir).expanduser() if ds.masks_dir else raw
        needs_ants = not ds.already_mni

        print(f"[{ds.name}] image root: {img_root} | mask root: {msk_root}")
        print(f"[{ds.name}] globs: t1={ds.t1_glob} masks={ds.mask_glob}")

        if not img_root or not img_root.exists():
            print(f"[skip] {ds.name}: images root missing {img_root}")
            continue
        if not msk_root or not msk_root.exists():
            print(f"[skip] {ds.name}: masks root missing {msk_root}")
            continue

        t1s = list(img_root.glob(ds.t1_glob))
        mks = list(msk_root.glob(ds.mask_glob))
        pairs = _match_pairs(t1s, mks)
        print(f"[{ds.name}] images: {len(t1s)} masks: {len(mks)} pairs found: {len(pairs)}")
        slug_base = raw or img_root
        slug = _slug_from_raw(slug_base, ds.name)
        overwrite = force_overwrite or ds.overwrite
        out_ds_existing = None if overwrite else _find_existing_output(out_root, slug_base)
        out_ds = out_ds_existing or (out_root / slug)
        marker = out_ds / "source.json"
        if overwrite and out_ds.exists():
            shutil.rmtree(out_ds, ignore_errors=True)
            print(f"[{ds.name}] overwrite=True -> cleared {out_ds}")
        elif marker.exists() and not overwrite:
            print(f"[{ds.name}] already processed -> {out_ds}, skipping (overwrite=True to redo).")
            outputs.append(out_ds)
            continue
        if not pairs:
            print(f"[warn] {ds.name}: no matched pairs; skipping dataset.")
            continue
        out_nat = out_ds / 'native_resampled_masks'
        out_mni = out_ds / 'mni_1mm_ants_fixed'
        out_norm = out_mni / 't1_norm'
        out_clean = out_mni / 'masks_clean'
        out_xfm = out_ds / 'xfm'
        for d in (out_nat, out_mni, out_norm, out_clean, out_xfm):
            d.mkdir(parents=True, exist_ok=True)

        for t1, mask, key in pairs:
            mask_t1 = out_nat / f"{key}_lesion_mask_T1w_native.nii.gz"
            prefix = out_xfm / f"{key}_t1_to_mni_"
            t1_mni = out_mni / f"{key}_T1w_MNI.nii.gz"
            mask_mni = out_mni / f"{key}_lesion_mask_MNI.nii.gz"
            t1_norm = out_norm / f"{key}_T1w_MNI_norm.nii.gz"
            mask_clean = out_clean / f"{key}_lesion_mask_MNI_clean.nii.gz"

            if ds.already_mni:
                if not mask_t1.exists():
                    resample_mask_to_t1(mask, t1, mask_t1)
                if not t1_mni.exists():
                    shutil.copy2(t1, t1_mni)
                if not mask_mni.exists():
                    shutil.copy2(mask_t1, mask_mni)
            else:
                if tpl is None:
                    tpl = _tpl_path(resolution=1)
                if reg_bin is None or apply_bin is None:
                    reg_bin, apply_bin = _ants_bins()
                if not mask_t1.exists():
                    resample_mask_to_t1(mask, t1, mask_t1)
                if not (prefix.with_name(prefix.name+'0GenericAffine.mat')).exists():
                    ants_register(t1, prefix, tpl, reg_bin, use_2mm=True)
                if not t1_mni.exists():
                    ants_apply(t1, tpl, prefix, t1_mni, apply_bin, nn=False)
                if not mask_mni.exists():
                    ants_apply(mask_t1, tpl, prefix, mask_mni, apply_bin, nn=True)
                    mi = nib.load(str(mask_mni)); data=(mi.get_fdata()>0.5).astype(np.uint8)
                    nib.save(nib.Nifti1Image(data, mi.affine, mi.header), str(mask_mni))
            if not t1_norm.exists():
                vol = nib.load(str(t1_mni)).get_fdata().astype(np.float32)
                norm = normalize_t1(vol)
                nib.save(nib.Nifti1Image(norm, nib.load(str(t1_mni)).affine, nib.load(str(t1_mni)).header), str(t1_norm))
            if not mask_clean.exists():
                data = (nib.load(str(mask_mni)).get_fdata()>0.5).astype(np.uint8)
                data = largest_component(data)
                # alignment sanity check: ensure mask overlaps brain; try flips if not
                brain = (nib.load(str(t1_mni)).get_fdata()>0).astype(np.uint8)
                best_mask, best_score, tag = _try_flips(data, brain)
                if best_score < 0.1:
                    print(f"[warn] low overlap for {mask_mni.name} (score {best_score:.3f}); keeping original")
                elif tag != "none":
                    print(f"[info] flipped {tag} for {mask_mni.name} (overlap {best_score:.3f})")
                    data = best_mask
                nib.save(nib.Nifti1Image(data, nib.load(str(mask_mni)).affine, nib.load(str(mask_mni)).header), str(mask_clean))

            ti = nib.load(str(t1_mni)); mi = nib.load(str(mask_clean))
            qc_rows.append(dict(
                dataset=ds.name,
                slug=slug,
                key=key,
                t1_mni=str(t1_mni),
                mask_mni=str(mask_clean),
                t1_shape=str(ti.shape[:3]),
                t1_zooms=str(tuple(round(z,3) for z in ti.header.get_zooms()[:3])),
                mask_nonzero=int(np.count_nonzero(mi.get_fdata()>0)),
            ))
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"raw_root": str(slug_base), "slug": slug}, indent=2))
        outputs.append(out_ds)

    if qc_rows:
        qc_csv = out_root / 'prep_qc.csv'
        with open(qc_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=qc_rows[0].keys())
            writer.writeheader(); writer.writerows(qc_rows)
        print('QC written', qc_csv)
    else:
        print('No QC rows written (no datasets processed).')
    return outputs


def combine_standardized(dataset_roots: Iterable[Path], dest: Path):
    dest_t1 = dest / "t1"
    dest_mk = dest / "masks"
    dest_t1.mkdir(parents=True, exist_ok=True)
    dest_mk.mkdir(parents=True, exist_ok=True)
    manifest = []
    for ds_root in dataset_roots:
        slug = ds_root.name
        t1_dir = ds_root / "mni_1mm_ants_fixed" / "t1_norm"
        msk_dir = ds_root / "mni_1mm_ants_fixed" / "masks_clean"
        if not t1_dir.exists() or not msk_dir.exists():
            print(f"[combine] missing t1_norm or masks_clean in {ds_root}, skipping")
            continue
        for t1 in sorted(t1_dir.glob("*.nii.gz")):
            base = t1.name
            mask = msk_dir / base.replace("_T1w_MNI_norm", "_lesion_mask_MNI_clean")
            if not mask.exists():
                continue
            out_t1 = dest_t1 / f"{slug}__{base}"
            out_mk = dest_mk / f"{slug}__{mask.name}"
            shutil.copy2(t1, out_t1)
            shutil.copy2(mask, out_mk)
            manifest.append({"slug": slug, "key": base, "t1": str(out_t1), "mask": str(out_mk)})
    if manifest:
        mf = dest / "manifest.csv"
        with open(mf, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
            writer.writeheader(); writer.writerows(manifest)
        print("Combined manifest ->", mf)
    else:
        print("No combined manifest written (no pairs).")


__all__ = [
    'DatasetConfig',
    'run_prep',
    'combine_standardized',
]
