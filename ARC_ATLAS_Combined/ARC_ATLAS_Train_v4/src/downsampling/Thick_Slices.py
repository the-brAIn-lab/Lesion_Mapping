#!/usr/bin/env python
# coding: utf-8

# # Helper functions - set paths

# In[8]:


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
THICK_OUT     = OUT_ROOT / "test_hires_thick_slices"
THICK_T1_DIR  = THICK_OUT / "t1"
THICK_MASK_DIR= THICK_OUT / "masks"
for d in (THICK_OUT, THICK_T1_DIR, THICK_MASK_DIR):
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

# Preview pairing status up front
img_candidates  = [p for p in SRC_T1_DIR.rglob("*.nii.gz")]
mask_candidates = [p for p in SRC_MASK_DIR.rglob("*.nii.gz")]
_pair_preview   = _pair_maps(SRC_T1_DIR, SRC_MASK_DIR)
print(
    f"Found pairs: {len(_pair_preview)}  (imgs={len(img_candidates)}, masks={len(mask_candidates)})"
)
# (Viewer moved to Cell 7; helpers above are now shared by all cells.)


# # 2) Thick slices (Z-only downsample with anti-alias)
# 
# Why / real-world: Very common: 1×1×5 mm or similar to shorten scans. Causes strong partial-volume in Z.

# What the code does
# 
# Applies a Gaussian blur along Z only: gaussian_filter(x, sigma=(0,0,sigma_z)).
# 
# Then decimates in Z by an integer factor (e.g., 5): xr = xb[:, :, ::5].
# 
# Updates only the Z voxel spacing: new_dz = old_dz * 5.
# 
# Mask is decimated slice-wise using nearest (simple stride). No smoothing is applied to masks to keep labels crisp.
# 
# What this mimics
# 
# Common protocol trade-off: keep in-plane high (e.g., ~1×1 mm) but use thick slices (e.g., 5 mm) to shorten scan time.
# 
# Real images have through-plane blur and partial volume: structures smaller than the slice thickness smear into neighbors.
# 
# Why it’s useful
# 
# Many routine T1/T2/FLAIR stacks are anisotropic. Models trained on 1 mm isotropic can struggle here; this tests that gap.
# 
# Caveats
# 
# True slice thickness combines slice profile + gaps; here we simulate the net blur/gross thickness (a good approximation).

# In[9]:


# === Thick-slice simulation: blur Z → decimate Z → nearest repeat back to original depth ===
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter
import shutil

# ------- config -------
factor_z    = 5         # coarsen only through-plane
sigma_z_vox = 2.0       # Gaussian blur along Z (in voxels) before decimation
OVERWRITE   = True
OUT_DIR     = THICK_OUT
OUT_IMG_DIR = THICK_T1_DIR
OUT_MSK_DIR = THICK_MASK_DIR
for d in (OUT_DIR, OUT_IMG_DIR, OUT_MSK_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- run ----
pairs = discover_pairs(SRC_T1_DIR, SRC_MASK_DIR)
print(f"Discovered unique pairs: {len(pairs)} (deduped across flat + subfolders)")

wrote = 0
for i, (img_p, msk_p) in enumerate(pairs, 1):
    img_ref = load_nii(str(img_p))
    msk_ref = load_nii(str(msk_p))
    x = data_f32(img_ref)
    y = data_f32(msk_ref)

    # 1) blur only along Z to mimic slice thickness PSF
    xb = gaussian_filter(x, sigma=(0.0, 0.0, float(sigma_z_vox)), mode="nearest")

    # 2) decimate Z ONLY for the image (keep XY)
    xd = xb[:, :, ::factor_z]

    # 3) upsample image back to original depth (nearest repeat)
    xu = np.repeat(xd, factor_z, axis=2)

    # 4) pad/crop image to match exactly
    target_shape = x.shape
    xu = pad_or_crop_to(xu, target_shape)

    # output names (match your naming scheme)
    base = strip_ext(img_p.name).replace("_T1w_MNI_norm", "").replace("_T1w", "")
    out_img = OUT_IMG_DIR / f"{base}_T1w_MNI_norm.nii.gz"
    out_msk = OUT_MSK_DIR / f"{base}_lesion_mask_MNI_clean.nii.gz"

    if not OVERWRITE and out_img.exists() and out_msk.exists():
        if i % 25 == 0 or i == len(pairs):
            print(f"[{i}/{len(pairs)}] (skip, exists) {out_img.name}")
        continue

    save_like(img_ref, xu.astype(np.float32), out_img, dtype=np.float32)
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
ds_pairs   = _pair_maps(THICK_T1_DIR, THICK_MASK_DIR)
common_keys = sorted(set(orig_pairs) & set(ds_pairs))
if not common_keys:
    raise RuntimeError("No overlapping cases between original and thick-slice datasets.")

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
        titles = ["Original", "Thick-slice (1×1×5mm→1mm)"]
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
