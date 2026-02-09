#!/usr/bin/env python
# coding: utf-8

# # Helper functions - Define Directories

# In[1]:


# --- SETUP: paths + robust pairing + I/O helpers (run once) ---
from pathlib import Path
import re, os, json, math
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter, zoom, rotate, shift, binary_closing, generate_binary_structure, label

# Paths rewritten to be project-relative
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# SOURCE hires test set (images + masks split into subfolders)
SRC_ROOT     = PROJECT_ROOT / "data" / "splits" / "50_25_25" / "test_hires"
SRC_T1_DIR   = SRC_ROOT / "t1"
SRC_MASK_DIR = SRC_ROOT / "masks"
if not (SRC_T1_DIR.exists() and SRC_MASK_DIR.exists()):
    raise FileNotFoundError(f"Expected t1/ and masks/ under {SRC_ROOT}")

# Where to place all degraded datasets inside the project
OUT_ROOT  = PROJECT_ROOT / "data" / "downsampled"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# --------- pairing (mirrors your training loader’s spirit) ----------
def is_mask_name(name: str) -> bool:
    n = name.lower()
    return ("mask" in n) or ("lesion" in n)

def strip_ext(name: str) -> str:
    return name[:-7] if name.endswith(".nii.gz") else os.path.splitext(name)[0]

def norm_key(name: str) -> str:
    # remove known suffixes to match img<->mask
    stem = strip_ext(name)
    # drop common endings
    for sfx in ["_T1w_MNI_norm","_T1w_MNI","_T1w_brain","_T1w","_T1","_image","_img","_img_prepped"]:
        if stem.endswith(sfx): stem = stem[: -len(sfx)]
    for sfx in ["_lesion_mask_MNI_clean","_lesion_mask_MNI","_lesion_mask","_desc-lesion_mask","_mask","_mask_prepped"]:
        if stem.endswith(sfx): stem = stem[: -len(sfx)]
    return stem.rstrip("_")

# build maps
imgs, msks = {}, {}
for p in sorted(SRC_T1_DIR.rglob("*.nii.gz")):
    imgs[norm_key(p.name)] = p
for p in sorted(SRC_MASK_DIR.rglob("*.nii.gz")):
    msks[norm_key(p.name)] = p
keys = sorted(set(imgs) & set(msks))
pairs = [(imgs[k], msks[k]) for k in keys]
print(f"Found pairs: {len(pairs)}  (imgs={len(imgs)}, msks={len(msks)})")

# --------- helpers ----------
def load_nii(p: Path) -> nib.Nifti1Image:
    return nib.load(str(p))

def data_f32(img: nib.Nifti1Image) -> np.ndarray:
    return np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)

def save_like(ref_img: nib.Nifti1Image, array: np.ndarray, out_path: Path, dtype=None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = (array.astype(dtype) if dtype is not None else array.astype(np.float32))
    nii = nib.Nifti1Image(arr, ref_img.affine, ref_img.header.copy())
    nib.save(nii, str(out_path))

def save_and_update_spacing(ref_img: nib.Nifti1Image, array: np.ndarray, out_path: Path, new_spacing_xyz):
    """Use ref affine but update pixdim to reflect a new voxel size (mm)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hdr = ref_img.header.copy()
    hdr["pixdim"][1:4] = np.array(new_spacing_xyz, dtype=np.float32)
    nii = nib.Nifti1Image(array.astype(np.float32), ref_img.affine, hdr)
    nib.save(nii, str(out_path))

def resample_factor(arr, factors, order):
    """Resample by 1/factors using scipy.zoom (anti-alias BEFORE calling)."""
    # zoom expects output/input; if factor=2 (coarsen), zoom=1/2
    zf = tuple(1.0/f for f in factors)
    return zoom(arr, zf, order=order, prefilter=True)

def ensure_uint8_mask(b):
    return (b > 0).astype(np.uint8)

def percentile_scale(vol, pmin=1, pmax=99):
    lo, hi = np.percentile(vol, [pmin, pmax])
    hi = max(hi, lo + 1e-6)
    x = np.clip((vol - lo) / (hi - lo), 0, 1)
    return x

def _is_top_level(p: Path) -> bool:
    try:
        parent = p.parent.resolve()
    except Exception:
        return False
    return parent == SRC_T1_DIR.resolve() or parent == SRC_MASK_DIR.resolve()

def discover_pairs(img_dir: Path, mask_dir: Path):
    """Recursively scan the provided t1/ and masks/ dirs and keep one pair per norm_key."""
    img_cands, msk_cands = {}, {}
    for p in img_dir.rglob("*.nii.gz"):
        key = norm_key(p.name)
        if not key:
            continue
        keep = img_cands.get(key)
        if keep is None or (_is_top_level(p) and not _is_top_level(keep)):
            img_cands[key] = p
    for p in mask_dir.rglob("*.nii.gz"):
        key = norm_key(p.name)
        if not key:
            continue
        keep = msk_cands.get(key)
        if keep is None or (_is_top_level(p) and not _is_top_level(keep)):
            msk_cands[key] = p
    keys = sorted(set(img_cands) & set(msk_cands))
    return [(img_cands[k], msk_cands[k]) for k in keys]


# # 1) Crude downsampling (no anti-alias) — “worst-case decimation”
# 
# Why / real-world: Stress-tests robustness to botched resampling pipelines that skip anti-aliasing, producing jaggies and aliasing. Rare, but if your preproc ever fails, this is what it looks like.

# ## Crude downsampling → crude upsampling (2×2×5)
# 
# **What this does**
# - **Decimate** the volume by fixed factors (e.g., 2× in X/Y and 5× in Z) via simple voxel picking (no anti-aliasing).
# - **Nearest-repeat upsample** back to the original matrix size (repeat along each axis), then **center pad/crop** to match the exact shape.
# - **Save on the original 1 mm grid** (affine/header unchanged), so spacing/pixdim remain the same while *effective* resolution is degraded.
# 
# **Why do it this way?**
# - Keeps geometry identical to your evaluation set (no resampling surprises downstream).
# - Emulates severe partial-volume and aliasing artifacts that arise from coarse acquisitions without regridding the header.
# 
# **What it mimics**
# - Low through-plane resolution and thick slices (factor 5 in Z).
# - In-plane coarsening (factor 2 in X/Y) without any anti-alias prefilter—i.e., a “worst-case” crude reconstruction.
# 
# **Caveats**
# - No anti-aliasing before decimation → intentionally introduces aliasing/ringing.
# - Nearest repetition back to full size creates blocky edges; not a physically accurate recon, just a stress test.
# - Because header spacing is unchanged, the images *look* like 1 mm voxels but have degraded information content.
# 
# **Useful knobs**
# - `factors = (fx, fy, fz)`: coarsening per axis (e.g., `(2,2,5)`).
# - Swap in anti-aliasing (e.g., `gaussian_filter`) before decimation if you want a “less harsh” variant.
# - Replace nearest-repeat with linear/b-spline upsampling for different artifact profiles (still keep final matrix the same).
# 

# 

# In[2]:


# === Crude low-res simulation (dedup-aware): decimate -> nearest upsample back to original shape ===
from pathlib import Path
import numpy as np
import nibabel as nib
import shutil

# ------- config -------
FACTORS   = (2, 2, 5)  # (x,y,z) crude coarsening
OVERWRITE = True      # set True to re-generate outputs even if they exist
# Use your existing SRC_DIR and OUT_ROOT from the setup cell
OUT_DIR   = OUT_ROOT / "test_hires_crude_down"
OUT_IMG_DIR = OUT_DIR / "t1"
OUT_MSK_DIR = OUT_DIR / "masks"
for d in (OUT_DIR, OUT_IMG_DIR, OUT_MSK_DIR):
    d.mkdir(parents=True, exist_ok=True)

def _is_top_level(p: Path) -> bool:
    try:
        parent = p.parent.resolve()
    except Exception:
        return False
    return parent == SRC_T1_DIR.resolve() or parent == SRC_MASK_DIR.resolve()

def discover_pairs(img_dir: Path, mask_dir: Path):
    """Recursively scan the supplied image/mask dirs (flat + nested) and dedup per key."""
    img_cands, msk_cands = {}, {}
    for p in img_dir.rglob("*.nii.gz"):
        key = norm_key(p.name)
        if not key:
            continue
        keep = img_cands.get(key)
        if keep is None or (_is_top_level(p) and not _is_top_level(keep)):
            img_cands[key] = p
    for p in mask_dir.rglob("*.nii.gz"):
        key = norm_key(p.name)
        if not key:
            continue
        keep = msk_cands.get(key)
        if keep is None or (_is_top_level(p) and not _is_top_level(keep)):
            msk_cands[key] = p
    keys = sorted(set(img_cands) & set(msk_cands))
    return [(img_cands[k], msk_cands[k]) for k in keys]

def decimate_crude(arr: np.ndarray, fx, fy, fz) -> np.ndarray:
    # integer sub-sampling (aliasing intentionally preserved)
    return arr[::fx, ::fy, ::fz]

def nearest_repeat(arr: np.ndarray, fx, fy, fz) -> np.ndarray:
    # zero-order hold upsampling back to large matrix (blocky)
    out = np.repeat(arr, fx, axis=0)
    out = np.repeat(out, fy, axis=1)
    out = np.repeat(out, fz, axis=2)
    return out

def pad_or_crop_to(arr: np.ndarray, target_shape):
    """Pad with edge values or crop centrally to hit exact target shape."""
    out = arr
    for ax in range(3):
        cur = out.shape[ax]
        tgt = target_shape[ax]
        if cur == tgt:
            continue
        if cur > tgt:
            # centered crop
            start = (cur - tgt)//2
            sl = [slice(None), slice(None), slice(None)]
            sl[ax] = slice(start, start+tgt)
            out = out[tuple(sl)]
        else:
            # pad by repeating edge values to maintain blockiness
            pad_before = (tgt - cur)//2
            pad_after  = tgt - cur - pad_before
            pads = [(0,0)]*out.ndim
            pads[ax] = (pad_before, pad_after)
            out = np.pad(out, pads, mode="edge")
    return out

# ---- run ----
pairs = discover_pairs(SRC_T1_DIR, SRC_MASK_DIR)
print(f"Discovered unique pairs: {len(pairs)} (deduped across flat + subfolders)")

fx, fy, fz = FACTORS
wrote = 0
for i, (img_p, msk_p) in enumerate(pairs, 1):
    img_ref = load_nii(img_p)
    msk_ref = load_nii(msk_p)
    x = data_f32(img_ref)
    y = data_f32(msk_ref)

    # crude downsample IMAGE only
    xd = decimate_crude(x, fx, fy, fz)

    # crude upsample (nearest repeat) back to original matrix size
    xu = nearest_repeat(xd, fx, fy, fz)

    # pad/crop to match exactly
    target_shape = x.shape
    xu = pad_or_crop_to(xu, target_shape)

    # output names (match your naming scheme)
    base = strip_ext(img_p.name).replace("_T1w_MNI_norm","").replace("_T1w","")
    out_img = OUT_IMG_DIR / f"{base}_T1w_MNI_norm.nii.gz"
    out_msk = OUT_MSK_DIR / f"{base}_lesion_mask_MNI_clean.nii.gz"

    if not OVERWRITE and out_img.exists() and out_msk.exists():
        if i % 25 == 0 or i == len(pairs):
            print(f"[{i}/{len(pairs)}] (skip, exists) {out_img.name}")
        continue

    # save using the original affine/header so volumes are back on 1mm grid
    save_like(img_ref, xu.astype(np.float32), out_img, dtype=np.float32)
    out_msk.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(msk_p), str(out_msk))
    wrote += 1

    if i % 10 == 0 or i == len(pairs):
        print(f"[{i}/{len(pairs)}] wrote {out_img.name} & {out_msk.name}")

print(f"Done → {OUT_DIR} | wrote {wrote} case(s)")


# In[ ]:


import numpy as np, nibabel as nib, matplotlib.pyplot as plt, ipywidgets as W
from functools import lru_cache
from IPython.display import display, clear_output
from pathlib import Path

ORIG_T1_DIR   = Path("data/splits/50_25_25/test_hires/t1")
ORIG_MASK_DIR = Path("data/splits/50_25_25/test_hires/masks")
DS_ROOT       = Path("data/downsampled/test_hires_crude_down")
DS_T1_DIR     = DS_ROOT / "t1" if (DS_ROOT / "t1").exists() else DS_ROOT
DS_MASK_DIR   = DS_ROOT / "masks" if (DS_ROOT / "masks").exists() else DS_ROOT

def _collect_pairs(t1_dir: Path, mask_dir: Path) -> dict[str, dict[str, Path]]:
    imgs, msks = {}, {}
    for p in t1_dir.rglob("*.nii.gz"): imgs[norm_key(p.name)] = p
    for p in mask_dir.rglob("*.nii.gz"): msks[norm_key(p.name)] = p
    return {k: {"img": imgs[k], "msk": msks[k]} for k in sorted(set(imgs) & set(msks))}

orig_pairs = _collect_pairs(ORIG_T1_DIR, ORIG_MASK_DIR)
ds_pairs   = _collect_pairs(DS_T1_DIR, DS_MASK_DIR)
common_keys = sorted(set(orig_pairs) & set(ds_pairs))
if not common_keys:
    raise RuntimeError("No overlapping cases between original and downsampled datasets.")

@lru_cache(maxsize=64)
def _load_vol(path: str):
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    if data.ndim == 4 and data.shape[-1] == 1: data = data[..., 0]
    return data

def _normalize(vol):
    nz = vol[vol > 0]
    if nz.size == 0: return vol * 0
    p1, p99 = np.percentile(nz, [1, 99])
    return np.clip((vol - p1) / max(p99 - p1, 1e-5), 0, 1)

key_dd = W.Dropdown(options=common_keys, description="Case:", layout=W.Layout(width="50%"))
slice_sl = W.IntSlider(description="Slice:", min=0, max=1, value=0, continuous_update=False, layout=W.Layout(width="50%"))
out = W.Output()

def _update_slider(*_):
    vol = _load_vol(str(orig_pairs[key_dd.value]["img"]))
    slice_sl.max = max(0, vol.shape[2] - 1)

def _render(*_):
    with out:
        clear_output(wait=True)
        key = key_dd.value
        img_orig = _load_vol(str(orig_pairs[key]["img"]))
        msk_orig = (_load_vol(str(orig_pairs[key]["msk"])) > 0.5)
        img_ds   = _load_vol(str(ds_pairs[key]["img"]))
        msk_ds   = (_load_vol(str(ds_pairs[key]["msk"])) > 0.5)
        z = int(slice_sl.value)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, img, msk, title in zip(
            axes,
            (_normalize(img_orig[..., z]), _normalize(img_ds[..., z])),
            (msk_orig[..., z], msk_ds[..., z]),
            ("Original", "Crude downsample")
        ):
            ax.imshow(img.T, cmap="gray", origin="lower")
            ax.contour(msk.T, levels=[0.5], colors="r", linewidths=0.8)
            ax.set_title(f"{title}\n{key} | slice {z}")
            ax.axis("off")
        plt.tight_layout(); plt.show()

_update_slider()
_render()
key_dd.observe(_update_slider, names="value")
key_dd.observe(_render, names="value")
slice_sl.observe(_render, names="value")
display(W.VBox([W.HBox([key_dd, slice_sl]), out]))
