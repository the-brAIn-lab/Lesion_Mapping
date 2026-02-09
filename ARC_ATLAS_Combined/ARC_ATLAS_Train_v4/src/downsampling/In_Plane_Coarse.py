#!/usr/bin/env python
# coding: utf-8

# # Helper functions - set paths

# In[5]:


# --- SETUP: shared paths + helpers (run once) ---
from pathlib import Path
import numpy as np
import nibabel as nib
from functools import lru_cache

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
SRC_ROOT      = PROJECT_ROOT / "data" / "splits" / "50_25_25" / "test_hires"
SRC_T1_DIR    = SRC_ROOT / "t1"
SRC_MASK_DIR  = SRC_ROOT / "masks"
if not (SRC_T1_DIR.exists() and SRC_MASK_DIR.exists()):
    raise FileNotFoundError(f"Expected t1/ and masks/ under {SRC_ROOT}")

OUT_ROOT      = PROJECT_ROOT / "data" / "downsampled"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
INPLANE_OUT     = OUT_ROOT / "test_hires_inplane_2x2x1mm_backTo1mm"
INPLANE_T1_DIR  = INPLANE_OUT / "t1"
INPLANE_MASK_DIR= INPLANE_OUT / "masks"
for d in (INPLANE_OUT, INPLANE_T1_DIR, INPLANE_MASK_DIR):
    d.mkdir(parents=True, exist_ok=True)

def is_mask_name(name: str) -> bool:
    n = name.lower()
    return ("mask" in n) or ("lesion" in n)

def strip_ext(name: str) -> str:
    return name[:-7] if name.endswith(".nii.gz") else Path(name).stem

def norm_key(name: str) -> str:
    stem = strip_ext(name)
    for suf in ["_T1w_MNI_norm","_T1w_MNI","_T1w_brain","_T1w","_T1","_image","_img","_img_prepped"]:
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    for suf in ["_lesion_mask_MNI_clean","_lesion_mask_MNI","_lesion_mask","_desc-lesion_mask","_mask","_mask_prepped"]:
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    return stem.rstrip("_")

def _is_top_level(p: Path) -> bool:
    try:
        parent = p.parent.resolve()
    except Exception:
        return False
    return parent in {SRC_T1_DIR.resolve(), SRC_MASK_DIR.resolve()}

def _pair_maps(img_dir: Path, mask_dir: Path):
    imgs, msks = {}, {}
    for p in img_dir.rglob("*.nii.gz"):
        if is_mask_name(p.name):
            continue
        key = norm_key(p.name)
        keep = imgs.get(key)
        if key and (keep is None or (_is_top_level(p) and not _is_top_level(keep))):
            imgs[key] = p
    for p in mask_dir.rglob("*.nii.gz"):
        if not is_mask_name(p.name):
            continue
        key = norm_key(p.name)
        keep = msks.get(key)
        if key and (keep is None or (_is_top_level(p) and not _is_top_level(keep))):
            msks[key] = p
    keys = sorted(set(imgs) & set(msks))
    return {k: {"img": imgs[k], "msk": msks[k]} for k in keys}

def discover_pairs(img_dir: Path, mask_dir: Path):
    return [(v["img"], v["msk"]) for v in _pair_maps(img_dir, mask_dir).values()]

# basic I/O helpers reused below
@lru_cache(maxsize=128)
def load_nii(path: Path | str) -> nib.Nifti1Image:
    return nib.load(str(path))

def data_f32(img: nib.Nifti1Image) -> np.ndarray:
    return np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)

def save_like(ref_img: nib.Nifti1Image, array: np.ndarray, out_path: Path, dtype=None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = array.astype(dtype or np.float32)
    nib.save(nib.Nifti1Image(arr, ref_img.affine, ref_img.header.copy()), str(out_path))

def pad_or_crop_to(arr: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    out = arr
    for axis, tgt in enumerate(target_shape):
        cur = out.shape[axis]
        if cur == tgt:
            continue
        if cur > tgt:
            start = (cur - tgt) // 2
            sl = [slice(None)] * out.ndim
            sl[axis] = slice(start, start + tgt)
            out = out[tuple(sl)]
        else:
            pad_before = (tgt - cur) // 2
            pad_after = tgt - cur - pad_before
            pads = [(0, 0)] * out.ndim
            pads[axis] = (pad_before, pad_after)
            out = np.pad(out, pads, mode="edge")
    return out.astype(arr.dtype, copy=False)

# Preview pairing status up front
img_candidates  = [p for p in SRC_T1_DIR.rglob("*.nii.gz")]
mask_candidates = [p for p in SRC_MASK_DIR.rglob("*.nii.gz")]
_pair_preview   = _pair_maps(SRC_T1_DIR, SRC_MASK_DIR)
print(
    f"Found pairs: {len(_pair_preview)}  (imgs={len(img_candidates)}, masks={len(mask_candidates)})"
)
# (Viewer moved to Cell 7; helpers above are now shared by all cells.)


# # 3) In-plane coarsening (XY downsample; Z intact)
# 
# Why / real-world: Protocols that keep thin slices but coarsen in-plane to reduce time. Tests sensitivity to small in-plane structures.

# What the code does
# 
# Applies a Gaussian blur in X/Y: gaussian_filter(x, sigma=(sigma_xy, sigma_xy, 0)).
# 
# Decimates in X/Y by an integer factor (e.g., 2): xr = xb[::2, ::2, :].
# 
# Updates only X/Y spacing; Z spacing unchanged.
# 
# Mask is downsampled with nearest in X/Y by the same stride.
# 
# What this mimics
# 
# Protocols that keep thin slices but reduce the in-plane matrix (e.g., 256→128) to cut scan time or extend coverage.
# 
# Small cortical or juxtacortical lesions get “blockier” and less distinct in-plane.
# 
# Why it’s useful
# 
# Tests sensitivity to in-plane resolution loss separately from slice thickness effects.
# 
# Caveats
# 
# Real recon often uses vendor-specific filters; our Gaussian + decimate is a principled, reproducible stand-in.

# In[6]:


# === In-plane coarsening: blur XY → decimate XY → nearest-repeat back to original matrix ===
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter
import shutil

# ------- config -------
OUT_DIR    = INPLANE_OUT
OUT_IMG_DIR = INPLANE_T1_DIR
OUT_MSK_DIR = INPLANE_MASK_DIR
factor_xy  = 2           # coarsen only in-plane
sigma_xy   = 1.0         # pre-blur in X/Y (voxels) before decimation
OVERWRITE  = False

for d in (OUT_DIR, OUT_IMG_DIR, OUT_MSK_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Reuse: discover_pairs(...), load_nii, data_f32, pad_or_crop_to, etc.

pairs = discover_pairs(SRC_T1_DIR, SRC_MASK_DIR)
print(f"Discovered unique pairs: {len(pairs)} (deduped across flat + subfolders)")

wrote = 0
for i, (img_p, msk_p) in enumerate(pairs, 1):
    img_ref = load_nii(img_p)
    x = data_f32(img_ref)

    # 1) Anti-alias in-plane only
    xb = gaussian_filter(x, sigma=(float(sigma_xy), float(sigma_xy), 0.0), mode="nearest")

    # 2) Decimate X/Y (mask stays native grid)
    xd = xb[::factor_xy, ::factor_xy, :]

    # 3) Crude upsample back to original matrix with nearest (repeat) in X/Y
    xu = np.repeat(np.repeat(xd, factor_xy, axis=0), factor_xy, axis=1)

    # 4) Pad/crop to match exactly (handles odd sizes / non-divisible dims)
    target_shape = x.shape
    xu = pad_or_crop_to(xu, target_shape)

    # 5) Save degraded IMAGE; copy mask unchanged
    base = strip_ext(img_p.name).replace("_T1w_MNI_norm","").replace("_T1w","")
    out_img = OUT_IMG_DIR / f"{base}_T1w_MNI_norm.nii.gz"
    out_msk = OUT_MSK_DIR / f"{base}_lesion_mask_MNI_clean.nii.gz"

    if not OVERWRITE and out_img.exists() and out_msk.exists():
        if i % 25 == 0 or i == len(pairs):
            print(f"[{i}/{len(pairs)}] (skip, exists) {out_img.name}")
        continue

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

orig_pairs = _pair_maps(SRC_T1_DIR, SRC_MASK_DIR)
ds_pairs   = _pair_maps(INPLANE_T1_DIR, INPLANE_MASK_DIR)
common_keys = sorted(set(orig_pairs) & set(ds_pairs))
if not common_keys:
    raise RuntimeError("No overlapping cases between original and in-plane datasets.")

@lru_cache(maxsize=64)
def _load_vol(path: str):
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    return data[..., 0] if data.ndim == 4 and data.shape[-1] == 1 else data

def _normalize(vol):
    nz = vol[vol > 0]
    if nz.size == 0:
        return vol * 0
    p1, p99 = np.percentile(nz, [1, 99])
    return np.clip((vol - p1) / max(p99 - p1, 1e-5), 0, 1)

key_dd   = W.Dropdown(options=common_keys, description="Case:", layout=W.Layout(width="45%"))
slice_sl = W.IntSlider(description="Slice:", min=0, max=1, value=0, continuous_update=False, layout=W.Layout(width="45%"))
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
        titles = ["Original", "In-plane coarse (2×2×1mm)"]
        for ax, img, msk, title in zip(axes,
                                       (_normalize(img_orig[..., z]), _normalize(img_ds[..., z])),
                                       (msk_orig[..., z], msk_ds[..., z]),
                                       titles):
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
