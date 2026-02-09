#!/usr/bin/env python
# coding: utf-8

# # Helper Functions - path set

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
SNR_OUT       = OUT_ROOT / "test_hires_snr_kspace_v1"
SNR_T1_DIR    = SNR_OUT / "t1"
SNR_MASK_DIR  = SNR_OUT / "masks"
for d in (SNR_OUT, SNR_T1_DIR, SNR_MASK_DIR):
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


# # 4) Reduced SNR (Rician noise)
# 
# Why / real-world: Fewer averages / higher acceleration → noisier magnitude images.

# What the code does
# 
# Normalizes the image to ~[0,1] via robust percentiles.
# 
# Estimates foreground mean μ (proxy for “signal”).
# 
# Applies signal attenuation α·x and adds Gaussian noise N(0,σ) with
# σ ≈ (α·μ)/TARGET_SNR to hit the requested SNR.
# 
# Clips to [0,1] and writes with original geometry (masks copied unchanged).
# 
# What this mimics
# 
# A stable, conservative SNR drop while preserving structure and intensity ordering.
# 
# Useful stand-in when you want predictable behavior across datasets.
# 
# Why it’s useful
# 
# Very robust (hard to break models), no magnitude or FFT artifacts, and keeps the intensity range aligned with training distributions.
# 
# Caveats
# 
# Not physically perfect (true MR magnitude noise is Rician and depends on coil configuration).
# 
# Use this when you want reliability over realism.

# In[6]:


# === Reduced SNR via k-space complex noise → magnitude (anchored) ===
from pathlib import Path
import numpy as np
import shutil

OUT_DIR     = SNR_OUT
OUT_IMG_DIR = SNR_T1_DIR
OUT_MSK_DIR = SNR_MASK_DIR
TARGET_SNR  = 20.0   # start moderately low (drop further to 15 for stronger noise)
SEED        = 123
OVERWRITE   = False

rng = np.random.default_rng(SEED)
pairs = discover_pairs(SRC_T1_DIR, SRC_MASK_DIR)
print(f"Discovered unique pairs: {len(pairs)} (deduped across flat + subfolders)")

def brain_like_mask(x: np.ndarray) -> np.ndarray:
    # foreground from intensities only (no lesion). Simple & robust.
    if (x > 0).sum() > 1000:
        m = x > 0
    else:
        thr = np.percentile(x, 60.0)
        m = x > thr
    # quick closing to fill small holes
    from scipy.ndimage import binary_closing, generate_binary_structure
    return binary_closing(m, structure=generate_binary_structure(3,1), iterations=1)

def pnorm01(x, p_lo=1, p_hi=99, eps=1e-6):
    lo, hi = np.percentile(x, [p_lo, p_hi])
    if hi - lo < eps: hi = lo + eps
    y = np.clip((x - lo) / (hi - lo), 0, 1)
    return y.astype(np.float32), float(lo), float(hi)

def affine_match(src, ref, mask):
    """Fit y ≈ a*src + b to ref on mask (LS), return a,b."""
    s = src[mask].ravel().astype(np.float64)
    r = ref[mask].ravel().astype(np.float64)
    if s.size < 50:
        return 1.0, 0.0
    A = np.vstack([s, np.ones_like(s)]).T
    a, b = np.linalg.lstsq(A, r, rcond=None)[0]
    if not np.isfinite(a): a = 1.0
    if not np.isfinite(b): b = 0.0
    return float(a), float(b)


def measure_snr(img, mask):
    vals = img[mask].astype(np.float32)
    mu = float(vals.mean())
    sd = float(vals.std(ddof=1) + 1e-6)
    return mu / sd, mu, sd

def add_kspace_noise_and_magnitude_calibrated(x_norm, brain_mask, target_snr, rng, iters=4):
    """
    Add white complex Gaussian noise in k-space, iFFT (unitary), magnitude image.
    Calibrate sigma_k so that measured SNR (mean/SD within brain_mask) ≈ target_snr.
    """
    # precompute FFT once (unitary keeps energy stable)
    X = np.fft.fftn(x_norm.astype(np.complex64), norm="ortho")

    # rough initial guess assuming sigma_img ≈ sigma_k with unitary FFT
    snr0, mu0, sd0 = measure_snr(x_norm, brain_mask)
    sigma_k_lo = 0.0
    sigma_k_hi = max(mu0 / max(target_snr, 1e-3) * 8.0, 1e-4)  # broad upper bound
    sigma_k = max(mu0 / max(target_snr, 1e-3), 1e-5)

    last_mag = None
    for _ in range(iters):
        nr = rng.normal(0.0, sigma_k, size=X.shape).astype(np.float32)
        ni = rng.normal(0.0, sigma_k, size=X.shape).astype(np.float32)
        X_noisy = X + (nr + 1j*ni).astype(np.complex64)

        img_complex = np.fft.ifftn(X_noisy, norm="ortho")  # unitary inverse
        mag = np.abs(img_complex).astype(np.float32)

        snr, mu, sd = measure_snr(mag, brain_mask)
        # binary search on sigma_k to hit target SNR
        if snr > target_snr:  # not noisy enough -> increase sigma
            sigma_k_lo = sigma_k
            sigma_k = 0.5 * (sigma_k + sigma_k_hi)
        else:                 # too noisy -> decrease sigma
            sigma_k_hi = sigma_k
            sigma_k = 0.5 * (sigma_k + sigma_k_lo)

        last_mag = mag

    # Optional gentle mean/scale match so contrasts stay comparable
    # (doesn't remove noise like percentile renorm would)
    _, mu_final, sd_final = measure_snr(last_mag, brain_mask)
    a = (mu0 / max(mu_final, 1e-6))
    mag_adj = np.clip(a * last_mag, 0.0, 1.0).astype(np.float32)
    return mag_adj


wrote = 0
for i, (img_p, msk_p) in enumerate(pairs, 1):
    img_ref = load_nii(img_p)
    msk_ref = load_nii(msk_p)
    x = data_f32(img_ref); y = data_f32(msk_ref)

    bl = brain_like_mask(x)
    x_norm, x_lo, x_hi = pnorm01(x, 1, 99)

    x_noisy = add_kspace_noise_and_magnitude_calibrated(
        x_norm, bl, TARGET_SNR, rng, iters=4
    )

    # light linear anchor only (preserve added noise)
    a, b = affine_match(x_noisy, x_norm, bl)
    x_adj = np.clip(a * x_noisy + b, 0, 1).astype(np.float32)

    base = strip_ext(img_p.name).replace("_T1w_MNI_norm","").replace("_T1w","")
    out_img = OUT_IMG_DIR / f"{base}_T1w_MNI_norm.nii.gz"
    out_msk = OUT_MSK_DIR / f"{base}_lesion_mask_MNI_clean.nii.gz"

    if not OVERWRITE and out_img.exists() and out_msk.exists():
        if i % 25 == 0 or i == len(pairs): print(f"[{i}/{len(pairs)}] (skip) {out_img.name}")
        continue

    save_like(img_ref, x_adj, out_img, dtype=np.float32)
    out_msk.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(msk_p), str(out_msk))
    wrote += 1

    if i % 10 == 0 or i == len(pairs):
        snr_final, mu_final, sd_final = measure_snr(x_adj, bl)
        corr = np.corrcoef(x_norm[bl].ravel(), x_adj[bl].ravel())[0,1] if bl.any() else np.nan
        print(
            f"[{i}/{len(pairs)}] wrote {out_img.name} | target_snr={TARGET_SNR:.1f} "
            f"| measured_snr={snr_final:.2f} | corr(brain)={corr:.3f}"
        )

print(f"Done → {OUT_DIR} | wrote {wrote} case(s)")


# In[ ]:


import numpy as np, nibabel as nib, matplotlib.pyplot as plt, ipywidgets as W
from functools import lru_cache
from IPython.display import display, clear_output

orig_pairs = _pair_maps(SRC_T1_DIR, SRC_MASK_DIR)
ds_pairs   = _pair_maps(SNR_T1_DIR, SNR_MASK_DIR)
common_keys = sorted(set(orig_pairs) & set(ds_pairs))
if not common_keys:
    raise RuntimeError("No overlapping cases between original and reduced-SNR datasets.")

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
        titles = ["Original", "Reduced SNR (k-space noise)"]
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
