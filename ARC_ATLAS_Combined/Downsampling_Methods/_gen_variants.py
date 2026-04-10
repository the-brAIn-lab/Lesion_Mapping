#!/usr/bin/env python3
"""
Generate *_variants.ipynb notebooks for all 5 downsampling methods.
v2: fixes IndexError, adds self-contained variant-selector widget, adds markdown labels.
"""
import json
from pathlib import Path

BASE    = Path("/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined")
DS_ROOT = BASE / "A_A_Combined_Data/Downsampled_Data"
HIRES   = BASE / "A_A_Combined_Data/Processed_HiresLowres_Split_Data/test_hires"
NB_DIR  = BASE / "Downsampling_Methods"

def mk_code(s): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s}
def mk_md(s):   return {"cell_type":"markdown","metadata":{},"source":s}
def nb_wrap(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
            "language_info": {"name":"python","pygments_lexer":"ipython3","version":"3.10.0"}
        },
        "nbformat": 4, "nbformat_minor": 5
    }

def load_nb(name):
    with open(NB_DIR / name) as f: return json.load(f)

def cell_src(cell): return "".join(cell.get("source", []))
def set_src(cell, s): cell["source"] = [s]


# ─────────────────────────────────────────────────────────────────────────────
# CANONICAL SETUP CELL — fully self-contained, no dependency on source notebooks
# ─────────────────────────────────────────────────────────────────────────────
def make_setup_cell(method_name, variant_paths_block, method_helpers_block=""):
    """
    Produces a single setup cell that defines ALL shared helpers needed by the
    variant downsampling cells: paths, load_nii, data_f32, discover_pairs,
    save_like, strip_ext, pad_or_crop_to, plus variant output paths.

    variant_paths_block: string of Python code that defines METHODNAME_V1_OUT...V5_OUT
    method_helpers_block: optional extra helpers specific to this method
    """
    code = f'''\
# === SETUP — run this cell first ===
# Defines shared paths, I/O helpers, and variant output directories.
# All variant downsampling cells below depend on this cell.

from pathlib import Path
from functools import lru_cache
import numpy as np
import nibabel as nib

SRC_ROOT     = Path("{HIRES}")
SRC_T1_DIR   = SRC_ROOT / "t1"
SRC_MASK_DIR = SRC_ROOT / "masks"
OUT_ROOT     = Path("{DS_ROOT}")

# ── I/O helpers ────────────────────────────────────────────────────────────────
def strip_ext(name: str) -> str:
    return name[:-7] if name.endswith(".nii.gz") else Path(name).stem

def _is_mask(name: str) -> bool:
    n = name.lower(); return "mask" in n or "lesion" in n

def _norm_key(name: str) -> str:
    stem = strip_ext(name)
    for s in ["_T1w_MNI_norm","_T1w_MNI","_T1w_brain","_T1w","_T1","_image","_img","_img_prepped"]:
        if stem.endswith(s): stem = stem[:-len(s)]; break
    for s in ["_lesion_mask_MNI_clean","_lesion_mask_MNI","_lesion_mask","_desc-lesion_mask","_mask","_mask_prepped"]:
        if stem.endswith(s): stem = stem[:-len(s)]; break
    return stem.rstrip("_")

def discover_pairs(img_dir: Path, mask_dir: Path):
    imgs, msks = {{}}, {{}}
    for p in img_dir.rglob("*.nii.gz"):
        if _is_mask(p.name): continue
        k = _norm_key(p.name)
        if k: imgs[k] = p
    for p in mask_dir.rglob("*.nii.gz"):
        if not _is_mask(p.name): continue
        k = _norm_key(p.name)
        if k: msks[k] = p
    return [(imgs[k], msks[k]) for k in sorted(set(imgs) & set(msks))]

@lru_cache(maxsize=256)
def load_nii(path) -> nib.Nifti1Image:
    return nib.load(str(path))

def data_f32(img: nib.Nifti1Image) -> np.ndarray:
    return np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)

def save_like(ref_img: nib.Nifti1Image, array: np.ndarray, out_path: Path, dtype=None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = array.astype(dtype or np.float32)
    nib.save(nib.Nifti1Image(arr, ref_img.affine, ref_img.header.copy()), str(out_path))

def pad_or_crop_to(arr, target_shape):
    out = arr
    for ax in range(3):
        cur = out.shape[ax]; tgt = target_shape[ax]
        if cur == tgt: continue
        if cur > tgt:
            s = (cur-tgt)//2; sl = [slice(None)]*out.ndim; sl[ax] = slice(s, s+tgt); out = out[tuple(sl)]
        else:
            pb = (tgt-cur)//2; pa = tgt-cur-pb; pads = [(0,0)]*out.ndim; pads[ax] = (pb, pa)
            out = np.pad(out, pads, mode="edge")
    return out

# ── Variant output paths ───────────────────────────────────────────────────────
{variant_paths_block}
# ── Method-specific helpers ────────────────────────────────────────────────────
{method_helpers_block}
# ── Verify source data ─────────────────────────────────────────────────────────
pairs_preview = discover_pairs(SRC_T1_DIR, SRC_MASK_DIR)
print(f"Source pairs found: {{len(pairs_preview)}}")
print(f"OUT_ROOT: {{OUT_ROOT}}")
'''
    return mk_code(code)


# ─────────────────────────────────────────────────────────────────────────────
# SELF-CONTAINED VARIANT-SELECTOR WIDGET
# ─────────────────────────────────────────────────────────────────────────────
def widget_cell(method_name, method_variants):
    """
    Fully self-contained widget: Original vs selected variant.
    method_variants: list of (label, t1_dir_str) for v1..v5
    """
    vlist = "[\n" + "".join(
        f'    ({repr(lbl)}, Path({repr(t1)})),\n'
        for lbl, t1 in method_variants
    ) + "]"

    code = f'''\
# === Self-contained interactive viewer: Original vs any {method_name} variant ===
# FULLY SELF-CONTAINED — run this cell independently, no other cells needed.
# Select subject, severity variant, and axial slice.

from pathlib import Path
from functools import lru_cache
import numpy as np, nibabel as nib
import matplotlib.pyplot as plt, ipywidgets as W
from IPython.display import display, clear_output

HIRES_T1   = Path("{HIRES}/t1")
HIRES_MASK = Path("{HIRES}/masks")
VARIANTS   = {vlist}

# ── helpers ──────────────────────────────────────────────────────────────────
def _is_mask(n): n=n.lower(); return "mask" in n or "lesion" in n

def _norm_key(name):
    stem = name[:-7] if name.endswith(".nii.gz") else Path(name).stem
    for s in ["_T1w_MNI_norm","_T1w_MNI","_T1w_brain","_T1w","_T1","_image","_img"]:
        if stem.endswith(s): stem=stem[:-len(s)]; break
    for s in ["_lesion_mask_MNI_clean","_lesion_mask_MNI","_lesion_mask","_desc-lesion_mask","_mask"]:
        if stem.endswith(s): stem=stem[:-len(s)]; break
    return stem.rstrip("_")

def _load_pairs():
    imgs, msks = {{}}, {{}}
    for p in HIRES_T1.rglob("*.nii.gz"):
        if _is_mask(p.name): continue
        k = _norm_key(p.name)
        if k: imgs[k] = p
    for p in HIRES_MASK.rglob("*.nii.gz"):
        if not _is_mask(p.name): continue
        k = _norm_key(p.name)
        if k: msks[k] = p
    return {{k: (imgs[k], msks[k]) for k in sorted(set(imgs) & set(msks))}}

@lru_cache(maxsize=256)
def _load(path):
    d = nib.load(str(path)).get_fdata(dtype=np.float32)
    return d[...,0] if d.ndim==4 and d.shape[-1]==1 else d

def _norm(vol):
    nz = vol[vol > 0]
    if not nz.size: return vol * 0
    lo, hi = np.percentile(nz, [1, 99])
    return np.clip((vol-lo)/max(hi-lo, 1e-5), 0, 1).astype(np.float32)

# ── build widgets ─────────────────────────────────────────────────────────────
pairs = _load_pairs()
keys  = sorted(pairs.keys())
if not keys: raise RuntimeError("No HiRes pairs found — check HIRES_T1 path")

var_dd  = W.Dropdown(options=[(lbl,i) for i,(lbl,_) in enumerate(VARIANTS)],
                     description="Variant:", layout=W.Layout(width="38%"))
case_dd = W.Dropdown(options=keys, description="Subject:",
                     layout=W.Layout(width="55%"))
sl_sl   = W.IntSlider(description="Slice:", min=0, max=1, value=0,
                      continuous_update=False, layout=W.Layout(width="55%"))
out     = W.Output()

def _update_max(*_):
    vol = _load(str(pairs[case_dd.value][0]))
    sl_sl.max = max(0, vol.shape[2]-1)
    sl_sl.value = min(sl_sl.value, sl_sl.max)

def _render(*_):
    with out:
        clear_output(wait=True)
        key = case_dd.value; z = int(sl_sl.value)
        vi  = int(var_dd.value); vlbl, vt1 = VARIANTS[vi]
        img_p, msk_p = pairs[key]
        base  = _norm_key(img_p.name)
        deg_p = vt1 / f"{{base}}_T1w_MNI_norm.nii.gz"

        orig = _norm(_load(str(img_p)))[:,:,z].T
        msk  = (_load(str(msk_p)) > 0.5)[:,:,z].T

        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        axes[0].imshow(orig, cmap="gray", origin="lower", interpolation="nearest")
        axes[0].contour(msk, levels=[0.5], colors="lime", linewidths=0.8)
        axes[0].set_title(f"Original  |  {{key}}  |  slice {{z}}", fontsize=10)
        axes[0].axis("off")

        if deg_p.exists():
            deg = _norm(_load(str(deg_p)))[:,:,z].T
            axes[1].imshow(deg, cmap="gray", origin="lower", interpolation="nearest")
            axes[1].contour(msk, levels=[0.5], colors="lime", linewidths=0.8)
            axes[1].set_title(f"{{vlbl.replace(chr(10), ' ')}}  |  {{key}}  |  slice {{z}}", fontsize=10)
        else:
            axes[1].text(0.5, 0.5,
                         f"Images not generated yet.\\nRun the {{vlbl.split()[0]}} downsampling cell first.",
                         ha="center", va="center", fontsize=10, color="#cc0000",
                         transform=axes[1].transAxes)
        axes[1].axis("off")
        plt.tight_layout(); plt.show()

_update_max(); _render()
case_dd.observe(_update_max, "value"); case_dd.observe(_render, "value")
var_dd.observe(_render, "value");     sl_sl.observe(_render, "value")
display(W.VBox([W.HBox([var_dd, case_dd]), sl_sl, out]))
'''
    return mk_code(code)


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON VIEWER (screenshot-style)
# ─────────────────────────────────────────────────────────────────────────────
def comparison_viewer_cell(method_title, versions_repr, save_png):
    code = f'''\
# === Severity comparison viewer: Original + {method_title} v1–v5 ===
# 2-row layout: axial slice at lesion centre (top) + zoomed lesion (bottom)
# IQM values annotated.  Self-contained — run after setup cell.

from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import binary_dilation, laplace

HIRES_MASK_DIR = Path("{HIRES}/masks")
SAVE_PATH      = Path("{save_png}")

VERSIONS = {versions_repr}

def _norm01(vol):
    fg = vol[vol > 0]
    if not fg.size: return vol * 0.0
    lo, hi = np.percentile(fg, [1, 99])
    return np.clip((vol-lo)/max(hi-lo,1e-5),0.,1.).astype(np.float32)

def _best_z(mask): return int(np.argmax(mask.sum(axis=(0,1))))

def _zoom_box(msk_sl, pad=20):
    rows=np.where(msk_sl.any(axis=1))[0]; cols=np.where(msk_sl.any(axis=0))[0]
    if not len(rows) or not len(cols): return slice(None), slice(None)
    return (slice(max(rows[0]-pad,0), min(rows[-1]+pad+1,msk_sl.shape[0])),
            slice(max(cols[0]-pad,0), min(cols[-1]+pad+1,msk_sl.shape[1])))

def _hfe(vol):
    f=np.fft.fftn(vol.astype(np.float64)); p=np.abs(f)**2; tot=p.sum()+1e-10
    hfm=np.zeros(vol.shape,bool)
    for ax in range(3):
        fr=np.abs(np.fft.fftfreq(vol.shape[ax])); idx=[np.newaxis]*3; idx[ax]=slice(None)
        hfm|=(fr[tuple(idx)]>0.25)
    return float(p[hfm].sum()/tot)

def _fwhm(vol):
    fg=(vol>np.percentile(vol[vol>0],10)) if (vol>0).sum()>200 else np.ones_like(vol,bool)
    vi=float(np.var(vol[fg])); vl=float(np.var(laplace(vol.astype(float))[fg]))
    return 0.0 if vl<1e-12 else float(2.355*np.sqrt(vi/vl)*0.5)

# Auto-select subject with median lesion size
all_msk=sorted(HIRES_MASK_DIR.glob("*_lesion_mask_MNI_clean.nii.gz"))
if not all_msk: raise FileNotFoundError(f"No masks in {{HIRES_MASK_DIR}}")
vol_sizes=[(nib.load(str(p)).get_fdata()>0).sum() for p in all_msk]
med_i=int(np.argsort(vol_sizes)[len(vol_sizes)//2])
msk_path=all_msk[med_i]
KEY    =msk_path.name.replace("_lesion_mask_MNI_clean.nii.gz","")
mask3d =nib.load(str(msk_path)).get_fdata()>0
z_ctr  =_best_z(mask3d)
msk_sl =mask3d[:,:,z_ctr].T
contour=binary_dilation(msk_sl)^msk_sl
rs,cs  =_zoom_box(msk_sl)
LV     =int(mask3d.sum())
print(f"Showing: {{KEY}}  |  lesion voxels: {{LV}}  |  axial z: {{z_ctr}}")

n=len(VERSIONS)
fig=plt.figure(figsize=(n*3.2,7.2))
gs =gridspec.GridSpec(2,n,figure=fig,wspace=0.04,hspace=0.26)
fig.suptitle(
    f"{method_title} — severity gradient\\n"
    f"Sample subject: {{KEY}}\\n"
    "Top: axial slice at lesion centre  |  Bottom: lesion zoom  |  IQM values annotated",
    fontsize=11,fontweight="bold",y=1.04)

for j,(label,t1_dir) in enumerate(VERSIONS):
    t1_path=Path(t1_dir)/f"{{KEY}}_T1w_MNI_norm.nii.gz"
    if not t1_path.exists():
        for r in [0,1]:
            ax=fig.add_subplot(gs[r,j]); ax.text(0.5,0.5,"not found",ha="center",va="center",fontsize=8,color="gray"); ax.axis("off")
        continue
    vol=nib.load(str(t1_path)).get_fdata(dtype=np.float32)
    vn=_norm01(vol); sl=vn[:,:,z_ctr].T
    hfe_val=_hfe(vn); fwhm_val=_fwhm(vn)

    ax0=fig.add_subplot(gs[0,j])
    ax0.imshow(sl,cmap="gray",origin="lower",interpolation="nearest")
    ov=np.zeros((*sl.shape,4)); ov[contour]=[0.,1.,0.3,0.9]
    ax0.imshow(ov,origin="lower",interpolation="nearest")
    ax0.set_title(label,fontsize=8.5,fontweight="bold",pad=4)
    ax0.text(0.03,0.03,f"FWHM={{fwhm_val:.1f}}  HFE={{hfe_val:.3f}}  LV={{LV}}",
             transform=ax0.transAxes,fontsize=5.5,color="white",va="bottom",
             bbox=dict(fc="black",alpha=0.55,pad=1.5,boxstyle="square,pad=0.2"))
    ax0.axis("off")

    ax1=fig.add_subplot(gs[1,j])
    ax1.imshow(sl[rs,cs],cmap="gray",origin="lower",interpolation="nearest")
    ov2=np.zeros((*sl[rs,cs].shape,4)); ov2[contour[rs,cs]]=[0.,1.,0.3,0.9]
    ax1.imshow(ov2,origin="lower",interpolation="nearest"); ax1.axis("off")

fig.text(0.5,-0.02,
    "Severity gradient from Original (HiRes) through progressively more aggressive degradation. "
    "Evenly spaced parameter increments — ordered severity scale suitable for: "
    "Dice ~ severity * degradation_type + (1 | case_id).\\n"
    "LV = lesion voxels  |  HFE = high-frequency energy ratio  |  FWHM = estimated smoothness FWHM (vox)",
    ha="center",fontsize=8,style="italic")
SAVE_PATH.parent.mkdir(parents=True,exist_ok=True)
plt.savefig(str(SAVE_PATH),dpi=150,bbox_inches="tight")
plt.show()
print(f"Saved → {{SAVE_PATH}}")
'''
    return mk_code(code)


# ─────────────────────────────────────────────────────────────────────────────
# PERILESIONAL CONTRAST CELL (VERSIONS = v1–v5 only, fixes IndexError)
# ─────────────────────────────────────────────────────────────────────────────
def perilesional_cell(method_title, versions_peri_repr, save_png, severity_labels):
    """
    versions_peri_repr: ONLY v1–v5 (no Original row) — fixes the IndexError.
    severity_labels: comma-separated quoted strings, 5 items.
    """
    code = f'''\
# === Population perilesional contrast: all severity variants of {method_title} ===
# For each subject: lesion mean − mean of ~5 mm perilesional ring.
# Panel 1: scatter (each variant vs Original)
# Panel 2: overlaid histograms by severity level
# Panel 3: mean ± SEM by ordered severity level (predictor for mixed model)
# Self-contained — run after generating variant images.

from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

HIRES_T1_DIR   = Path("{HIRES}/t1")
HIRES_MASK_DIR = Path("{HIRES}/masks")
SAVE_PATH      = Path("{save_png}")
DILATION_ITERS = 5

# v1–v5 degraded versions only (Original loaded separately as reference)
VERSIONS   = {versions_peri_repr}
SEV_LABELS = [{severity_labels}]
SEV_COLORS = ["#74c0fc","#4dabf7","#339af0","#1c7ed6","#1864ab"]

def _peri_contrast(vol, mask):
    dil=binary_dilation(mask,iterations=DILATION_ITERS); peri=dil&~mask
    return float(vol[mask].mean()) - (float(vol[peri].mean()) if peri.sum()>0 else 0.0)

# ── collect data ──────────────────────────────────────────────────────────────
mask_paths=sorted(HIRES_MASK_DIR.glob("*_lesion_mask_MNI_clean.nii.gz"))
hi_vals=[]; var_vals=[[] for _ in VERSIONS]

for msk_p in mask_paths:
    key=msk_p.name.replace("_lesion_mask_MNI_clean.nii.gz","")
    hi_p=HIRES_T1_DIR/f"{{key}}_T1w_MNI_norm.nii.gz"
    if not hi_p.exists(): continue
    mask=nib.load(str(msk_p)).get_fdata()>0
    if mask.sum()<10: continue
    hi=nib.load(str(hi_p)).get_fdata(dtype=np.float32)
    hi_vals.append(_peri_contrast(hi,mask))
    for vi,(_,t1_dir) in enumerate(VERSIONS):
        deg_p=Path(t1_dir)/f"{{key}}_T1w_MNI_norm.nii.gz"
        if deg_p.exists():
            deg=nib.load(str(deg_p)).get_fdata(dtype=np.float32)
            var_vals[vi].append(_peri_contrast(deg,mask))
        else:
            var_vals[vi].append(float("nan"))

hi_arr  =np.array(hi_vals)
var_arrs=[np.array(v) for v in var_vals]
print(f"Subjects with Original data: {{len(hi_arr)}}")
for vi,lbl in enumerate(SEV_LABELS):
    n_v=int(np.sum(~np.isnan(var_arrs[vi])))
    print(f"  {{lbl}}: {{n_v}} subjects  |  mean={{np.nanmean(var_arrs[vi]):+.3f}}")

# ── plot ──────────────────────────────────────────────────────────────────────
fig,axes=plt.subplots(1,3,figsize=(18,6),gridspec_kw={{"wspace":0.32}})
COLOR_HI="#adb5bd"

# Panel 1: scatter (each variant vs Original)
lim=max(abs(hi_arr).max(),0.01)
for vi in range(len(VERSIONS)):
    arr=var_arrs[vi]; valid=~np.isnan(arr)
    if valid.any(): lim=max(lim,abs(arr[valid]).max())
    axes[0].scatter(hi_arr[valid],arr[valid],label=SEV_LABELS[vi],
                    color=SEV_COLORS[vi],alpha=0.55,s=20,zorder=3)
lim*=1.18
axes[0].plot([-lim,lim],[-lim,lim],"k:",lw=0.8,label="no change",zorder=1)
axes[0].axhline(0,color="k",lw=1.2,ls="--",zorder=2)
axes[0].axvline(0,color="k",lw=1.2,ls="--",zorder=2)
axes[0].set_xlim(-lim,lim); axes[0].set_ylim(-lim,lim)
axes[0].set_xlabel("Original perilesional contrast",fontsize=11)
axes[0].set_ylabel("Variant perilesional contrast",fontsize=11)
axes[0].set_title(f"{method_title}\\nScatter: Original vs each variant  (n={{len(hi_arr)}})",
                  fontsize=11,fontweight="bold")
axes[0].legend(fontsize=8); axes[0].spines[["top","right"]].set_visible(False)

# Panel 2: overlaid histograms
all_v=np.concatenate([hi_arr]+[var_arrs[i][~np.isnan(var_arrs[i])] for i in range(len(VERSIONS))])
bins=np.linspace(np.nanmin(all_v)*1.15,np.nanmax(all_v)*1.15,40)
axes[1].hist(hi_arr,bins=bins,alpha=0.50,color=COLOR_HI,
             label=f"Original  (μ={{hi_arr.mean():+.3f}})",edgecolor="w",lw=0.3)
for vi in range(len(VERSIONS)):
    arr=var_arrs[vi][~np.isnan(var_arrs[vi])]
    axes[1].hist(arr,bins=bins,alpha=0.55,color=SEV_COLORS[vi],
                 label=f"{{SEV_LABELS[vi]}}  (μ={{arr.mean():+.3f}})",edgecolor="w",lw=0.3)
axes[1].axvline(0,color="k",lw=1.8,ls="--")
axes[1].set_xlabel("Lesion mean − perilesional mean (a.u.)",fontsize=11)
axes[1].set_ylabel("Subjects",fontsize=11)
axes[1].set_title("{method_title}\\nContrast distribution by severity",fontsize=11,fontweight="bold")
axes[1].legend(fontsize=8); axes[1].spines[["top","right"]].set_visible(False)

# Panel 3: mean ± SEM by severity (ordered predictor)
all_means=[hi_arr.mean()]+[np.nanmean(var_arrs[i]) for i in range(len(VERSIONS))]
all_sems =[hi_arr.std()/max(len(hi_arr)**0.5,1)]+[
    np.nanstd(var_arrs[i])/max(np.sum(~np.isnan(var_arrs[i]))**0.5,1)
    for i in range(len(VERSIONS))]
x_lbl=["Original"]+SEV_LABELS; c_bar=[COLOR_HI]+SEV_COLORS
for xi,(m,s,c) in enumerate(zip(all_means,all_sems,c_bar)):
    axes[2].bar(xi,m,color=c,width=0.72,zorder=3,alpha=0.88)
    axes[2].errorbar(xi,m,yerr=s,fmt="none",color="k",capsize=4,lw=1.5,zorder=4)
axes[2].set_xticks(range(len(x_lbl)))
axes[2].set_xticklabels(x_lbl,fontsize=9,rotation=30,ha="right")
axes[2].axhline(0,color="k",lw=1.2,ls="--")
axes[2].set_xlabel("Severity level",fontsize=11)
axes[2].set_ylabel("Mean perilesional contrast ± SEM (a.u.)",fontsize=11)
axes[2].set_title("{method_title}\\nMean contrast by ordered severity\\n(continuous predictor for mixed model)",
                  fontsize=11,fontweight="bold")
axes[2].spines[["top","right"]].set_visible(False)

fig.suptitle(
    f"Perilesional contrast attenuation: {method_title} severity variants\\n"
    "Perilesional ring ≈ 5 mm  |  Ordered severity: Dice ~ severity * degradation_type + (1|case_id)",
    fontsize=11,fontweight="bold",y=1.02)
SAVE_PATH.parent.mkdir(parents=True,exist_ok=True)
plt.savefig(str(SAVE_PATH),dpi=150,bbox_inches="tight")
plt.show()
print(f"Saved → {{SAVE_PATH}}")
'''
    return mk_code(code)


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-SUBJECT CROSS-TREATMENT PERILESIONAL VIEW
# ─────────────────────────────────────────────────────────────────────────────
def per_subject_cell(method_title, versions_all_repr):
    """
    Same layout as the IQM method comparison: rows = severity versions (v1-v5),
    columns = HiRes zoomed | Degraded zoomed | Diff + colorbar | spacer | Histogram.
    All image panels zoomed to the same lesion region. Contour lines only (no fill).
    green = lesion, orange = 5 mm perilesional ring.
    Subject-selectable widget, fully self-contained.
    versions_all_repr: 6 entries [Original, v1, v2, v3, v4, v5].
    """
    code = f'''\
# === Single-subject cross-treatment perilesional view: {method_title} ===
# Rows = severity versions (v1-v5). Columns = HiRes zoomed | Variant | Diff | Histogram.
# All panels zoomed to lesion region. Contour lines only: green=lesion, orange=5mm ring.
# Diff fixed scale +-0.5. Histogram = perilesional contrast (lesion voxels - ring mean).
# FULLY SELF-CONTAINED widget — select subject, run without any other cell.

from pathlib import Path
import numpy as np, nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import binary_dilation
import ipywidgets as W
from IPython.display import display, clear_output

HIRES_T1_DIR   = Path("{HIRES}/t1")
HIRES_MASK_DIR = Path("{HIRES}/masks")
DILATION_ITERS = 5
DIFF_SYM       = 0.5
ZOOM_PAD       = 30
COLOR_HI       = "#3a86ff"

VERSIONS_ALL = {versions_all_repr}
# v1-v5 are the rows; Original (index 0) is the HiRes reference column
DEG_VERSIONS = VERSIONS_ALL[1:]
ROW_COLORS   = ["#74c0fc","#4dabf7","#339af0","#1c7ed6","#1864ab"]

def _brain_bbox(sl, pad=14):
    rows = np.where(sl.max(axis=1) > 0)[0]
    cols = np.where(sl.max(axis=0) > 0)[0]
    if not len(rows) or not len(cols): return slice(None), slice(None)
    return (slice(max(rows[0]-pad,0), min(rows[-1]+pad+1, sl.shape[0])),
            slice(max(cols[0]-pad,0), min(cols[-1]+pad+1, sl.shape[1])))

def _lesion_zoom(mask_2d, pad=ZOOM_PAD):
    rows = np.where(mask_2d.max(axis=1) > 0)[0]
    cols = np.where(mask_2d.max(axis=0) > 0)[0]
    if not len(rows) or not len(cols): return slice(None), slice(None)
    return (slice(max(rows[0]-pad,0), min(rows[-1]+pad+1, mask_2d.shape[0])),
            slice(max(cols[0]-pad,0), min(cols[-1]+pad+1, mask_2d.shape[1])))

def _peri_contrast_voxels(vol, mask):
    peri = binary_dilation(mask, iterations=DILATION_ITERS) & ~mask
    peri_mean = float(vol[peri].mean()) if peri.sum() > 0 else 0.
    return vol[mask] - peri_mean

# ── subject list ──────────────────────────────────────────────────────────────
mask_paths_ps2 = sorted(HIRES_MASK_DIR.glob("*_lesion_mask_MNI_clean.nii.gz"))
keys_ps2 = [p.name.replace("_lesion_mask_MNI_clean.nii.gz","") for p in mask_paths_ps2]
if not keys_ps2: raise FileNotFoundError(f"No masks found in {{HIRES_MASK_DIR}}")

case_dd_ps2 = W.Dropdown(options=keys_ps2, description="Subject:",
                         layout=W.Layout(width="65%"))
out_ps2 = W.Output()

def _render_ps2(*_):
    with out_ps2:
        clear_output(wait=True)
        key   = case_dd_ps2.value
        hi_p  = HIRES_T1_DIR  / f"{{key}}_T1w_MNI_norm.nii.gz"
        msk_p = HIRES_MASK_DIR / f"{{key}}_lesion_mask_MNI_clean.nii.gz"
        if not hi_p.exists() or not msk_p.exists():
            print(f"Missing HiRes files for {{key}}"); return

        hi_vol = nib.load(str(hi_p)).get_fdata(dtype=np.float32)
        mask   = nib.load(str(msk_p)).get_fdata() > 0
        ring   = binary_dilation(mask, iterations=DILATION_ITERS) & ~mask
        best_z = int(mask.sum(axis=(0,1)).argmax())
        print(f"Subject: {{key}}  |  lesion voxels: {{int(mask.sum())}}  |  peak z: {{best_z}}")

        # compute shared crop/zoom coordinates from HiRes
        hi_sl   = hi_vol[:,:,best_z]
        mask_sl = mask[:,:,best_z]
        ring_sl = ring[:,:,best_z]
        rr,cc   = _brain_bbox(hi_sl, pad=14)
        hi_c    = hi_sl[rr,cc]; mask_c = mask_sl[rr,cc]
        vmin,vmax = hi_c.min(), hi_c.max()
        rz,cz   = _lesion_zoom(mask_c, pad=ZOOM_PAD)
        hi_z    = hi_c[rz,cz]
        mask_z  = mask_c[rz,cz]
        ring_z  = ring_sl[rr,cc][rz,cz]
        hi_pv   = _peri_contrast_voxels(hi_vol, mask)

        n_rows = len(DEG_VERSIONS)
        fig    = plt.figure(figsize=(17.5, n_rows * 3.4))
        outer  = gridspec.GridSpec(n_rows, 1, figure=fig, hspace=0.10)

        for ri,(label,t1_dir) in enumerate(DEG_VERSIONS):
            color = ROW_COLORS[ri]
            deg_p = Path(t1_dir) / f"{{key}}_T1w_MNI_norm.nii.gz"
            inner = gridspec.GridSpecFromSubplotSpec(
                1, 5, subplot_spec=outer[ri],
                width_ratios=[1.0, 1.0, 1.15, 0.12, 1.6], wspace=0.06)

            if not deg_p.exists():
                for ci in range(5):
                    ax = fig.add_subplot(inner[ci])
                    ax.text(0.5,0.5,"images not\\ngenerated yet",
                            ha="center",va="center",fontsize=8,color="#cc0000",
                            transform=ax.transAxes)
                    ax.axis("off")
                if ri==0:
                    for ci,ttl in enumerate(["HiRes (original)","Degraded",
                                             "Diff (scale +-0.5)","","Local contrast histogram"]):
                        if ttl:
                            fig.add_subplot(inner[ci]).set_title(ttl,fontsize=10,
                                fontweight="bold",pad=5)
                continue

            deg_vol = nib.load(str(deg_p)).get_fdata(dtype=np.float32)
            deg_sl  = deg_vol[:,:,best_z]
            diff_sl = deg_sl - hi_sl
            deg_z   = deg_sl[rr,cc][rz,cz]
            diff_z  = diff_sl[rr,cc][rz,cz]
            deg_pv  = _peri_contrast_voxels(deg_vol, mask)

            # Panel 0: HiRes zoomed + contours
            ax0 = fig.add_subplot(inner[0])
            ax0.imshow(hi_z.T, cmap="gray", vmin=vmin, vmax=vmax,
                       origin="lower", aspect="equal")
            if ring_z.sum()>0:
                ax0.contour(ring_z.T, levels=[0.5], colors=["#ff8c00"], linewidths=1.5)
            if mask_z.sum()>0:
                ax0.contour(mask_z.T, levels=[0.5], colors=["#00ff88"], linewidths=1.5)
            ax0.axis("off")
            if ri==0: ax0.set_title("HiRes (original)", fontsize=10,
                                     fontweight="bold", pad=5)
            ax0.text(-0.04, 0.5, label.replace("\\n"," "), transform=ax0.transAxes,
                     fontsize=9, fontweight="bold", color=color,
                     va="center", ha="right", rotation=90)

            # Panel 1: Degraded zoomed + contours
            ax1 = fig.add_subplot(inner[1])
            ax1.imshow(deg_z.T, cmap="gray", vmin=vmin, vmax=vmax,
                       origin="lower", aspect="equal")
            if ring_z.sum()>0:
                ax1.contour(ring_z.T, levels=[0.5], colors=["#ff8c00"], linewidths=1.5)
            if mask_z.sum()>0:
                ax1.contour(mask_z.T, levels=[0.5], colors=["#00ff88"], linewidths=1.5)
            ax1.axis("off")
            if ri==0: ax1.set_title("Degraded", fontsize=10,
                                     fontweight="bold", pad=5)

            # Panel 2: Diff zoomed + contours + per-row colorbar
            ax2 = fig.add_subplot(inner[2])
            im  = ax2.imshow(diff_z.T, cmap="RdBu_r",
                             vmin=-DIFF_SYM, vmax=DIFF_SYM,
                             origin="lower", aspect="equal")
            if ring_z.sum()>0:
                ax2.contour(ring_z.T, levels=[0.5], colors=["#ff8c00"], linewidths=2.0)
            if mask_z.sum()>0:
                ax2.contour(mask_z.T, levels=[0.5], colors=["#00ff88"], linewidths=2.0)
            ax2.axis("off")
            div = make_axes_locatable(ax2)
            cax = div.append_axes("right", size="7%", pad=0.06)
            cb  = fig.colorbar(im, cax=cax)
            cb.ax.tick_params(labelsize=6)
            cb.set_ticks([-DIFF_SYM, 0, DIFF_SYM])
            cb.set_ticklabels(["-0.5","0","+0.5"])
            if ri==0:
                ax2.set_title("Diff  (scale +-0.5)\\ngreen = lesion  |  orange = 5 mm ring",
                              fontsize=9, fontweight="bold", pad=5)

            # Panel 3: spacer
            fig.add_subplot(inner[3]).axis("off")

            # Panel 4: Perilesional contrast histogram (HiRes vs this variant)
            ax3 = fig.add_subplot(inner[4])
            all_v = np.concatenate([hi_pv, deg_pv])
            lo,hi_b = np.percentile(all_v,1), np.percentile(all_v,99)
            bins = np.linspace(lo*1.12, max(hi_b*1.12, 0.05), 38)
            ax3.hist(hi_pv,  bins=bins, alpha=0.65, color=COLOR_HI, density=True,
                     label=f"HiRes  mu={{hi_pv.mean():+.3f}}",
                     edgecolor="white", linewidth=0.3, zorder=3)
            ax3.hist(deg_pv, bins=bins, alpha=0.65, color=color,    density=True,
                     label=f"Degraded  mu={{deg_pv.mean():+.3f}}",
                     edgecolor="white", linewidth=0.3, zorder=3)
            ax3.axvline(0,             color="black",  lw=1.5, ls="--", zorder=5,
                        label="zero (same as surroundings)")
            ax3.axvline(hi_pv.mean(),  color=COLOR_HI, lw=1.8, ls="-",  zorder=4)
            ax3.axvline(deg_pv.mean(), color=color,    lw=1.8, ls="-",  zorder=4)
            ax3.axvspan(bins[0], 0, alpha=0.06, color=COLOR_HI, zorder=0)
            ax3.text(0.02,0.97,"<- darker", transform=ax3.transAxes,
                     ha="left",va="top",fontsize=7.5,color=COLOR_HI,fontweight="bold")
            ax3.text(0.98,0.97,"brighter ->", transform=ax3.transAxes,
                     ha="right",va="top",fontsize=7.5,color=color,fontweight="bold")
            ax3.set_xlabel("Voxel - perilesional mean (a.u.)", fontsize=8)
            ax3.set_ylabel("Density", fontsize=8)
            ax3.tick_params(labelsize=7)
            ax3.legend(fontsize=7.5, framealpha=0.88)
            ax3.spines["top"].set_visible(False)
            ax3.spines["right"].set_visible(False)
            if ri==0:
                ax3.set_title("Local contrast  =  voxel - perilesional mean",
                              fontsize=10, fontweight="bold", pad=5)

        fig.suptitle(
            f"{method_title} severity variants  -  subject: {{key}}  (z={{best_z}})\\n"
            "HiRes  |  Degraded  |  Difference (+-0.5)  |  Perilesional contrast histogram\\n"
            "Contours: green = lesion  |  orange = 5 mm perilesional ring",
            fontsize=11, fontweight="bold", y=1.01)
        plt.show()

_render_ps2()
case_dd_ps2.observe(_render_ps2, "value")
display(W.VBox([case_dd_ps2, out_ps2]))
'''
    return mk_code(code)


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 1: CRUDE
# ─────────────────────────────────────────────────────────────────────────────
def make_crude_variants():
    nb = load_nb("Crude.ipynb")
    cells = nb["cells"]

    variant_paths = """\
CRUDE_V1_OUT=OUT_ROOT/"Crude";      CRUDE_V1_T1=CRUDE_V1_OUT/"t1"; CRUDE_V1_MSK=CRUDE_V1_OUT/"masks"
CRUDE_V2_OUT=OUT_ROOT/"Crude_v2";   CRUDE_V2_T1=CRUDE_V2_OUT/"t1"; CRUDE_V2_MSK=CRUDE_V2_OUT/"masks"
CRUDE_V3_OUT=OUT_ROOT/"Crude_v3";   CRUDE_V3_T1=CRUDE_V3_OUT/"t1"; CRUDE_V3_MSK=CRUDE_V3_OUT/"masks"
CRUDE_V4_OUT=OUT_ROOT/"Crude_v4";   CRUDE_V4_T1=CRUDE_V4_OUT/"t1"; CRUDE_V4_MSK=CRUDE_V4_OUT/"masks"
CRUDE_V5_OUT=OUT_ROOT/"Crude_v5";   CRUDE_V5_T1=CRUDE_V5_OUT/"t1"; CRUDE_V5_MSK=CRUDE_V5_OUT/"masks"
for d in [CRUDE_V1_T1,CRUDE_V1_MSK,CRUDE_V2_T1,CRUDE_V2_MSK,CRUDE_V3_T1,CRUDE_V3_MSK,
          CRUDE_V4_T1,CRUDE_V4_MSK,CRUDE_V5_T1,CRUDE_V5_MSK]: d.mkdir(parents=True,exist_ok=True)"""

    crude_helpers = """\
def decimate_crude(arr,fx,fy,fz): return arr[::fx,::fy,::fz]
def nearest_repeat(arr,fx,fy,fz):
    out=np.repeat(arr,fx,axis=0); out=np.repeat(out,fy,axis=1); out=np.repeat(out,fz,axis=2); return out"""

    setup_cell = make_setup_cell("Crude", variant_paths, crude_helpers)

    set_src(cells[0],
        "# Crude Downsampling — Severity Variants (v2–v5)\n"
        "Run the **setup cell** first, then run each variant downsampling cell.\n\n"
        "| Version | FACTORS (fx,fy,fz) | Step from v1 |\n"
        "|---------|-------------------|---------------|\n"
        "| v1 (original) | (2, 2, 5) | — |\n"
        "| v2 | (3, 3, 6) | +1 each |\n"
        "| v3 | (4, 4, 7) | +1 each |\n"
        "| v4 | (5, 5, 8) | +1 each |\n"
        "| v5 | (6, 6, 9) | +1 each |"
    )

    VARIANTS = [("v2",(3,3,6)), ("v3",(4,4,7)), ("v4",(5,5,8)), ("v5",(6,6,9))]
    ds_cells = []
    for vname,(fx,fy,fz) in VARIANTS:
        vU = vname.upper()
        ds_cells.append(mk_md(
            f"## Generate `Crude_{vname}` — Severity {vname[1]}/5\n"
            f"**Parameters:** `FACTORS = ({fx}, {fy}, {fz})`  "
            f"(crude integer decimation, no anti-aliasing)\n\n"
            f"Output → `Downsampled_Data/Crude_{vname}/`"
        ))
        code = f"""\
# === Crude_{vname}: FACTORS = ({fx}, {fy}, {fz}) — severity {vname[1]}/5 ===
import shutil

FACTORS     = ({fx}, {fy}, {fz})
OVERWRITE   = False
OUT_DIR     = CRUDE_{vU}_OUT;  OUT_IMG_DIR = CRUDE_{vU}_T1;  OUT_MSK_DIR = CRUDE_{vU}_MSK

_fx, _fy, _fz = FACTORS
pairs = discover_pairs(SRC_T1_DIR, SRC_MASK_DIR)
print(f"[Crude_{vname}] pairs: {{len(pairs)}}  FACTORS={{FACTORS}}")

wrote = 0
for i, (img_p, msk_p) in enumerate(pairs, 1):
    img_ref=load_nii(img_p); x=data_f32(img_ref)
    xu=pad_or_crop_to(nearest_repeat(decimate_crude(x,_fx,_fy,_fz),_fx,_fy,_fz),x.shape)
    base=strip_ext(img_p.name).replace("_T1w_MNI_norm","").replace("_T1w","")
    out_img=OUT_IMG_DIR/f"{{base}}_T1w_MNI_norm.nii.gz"
    out_msk=OUT_MSK_DIR/f"{{base}}_lesion_mask_MNI_clean.nii.gz"
    if not OVERWRITE and out_img.exists() and out_msk.exists(): continue
    save_like(img_ref,xu.astype(np.float32),out_img,dtype=np.float32)
    out_msk.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(str(msk_p),str(out_msk))
    wrote+=1
    if i%10==0 or i==len(pairs): print(f"  [{{i}}/{{len(pairs)}}] {{out_img.name}}")
print(f"Done → {{OUT_DIR}} | wrote {{wrote}}")
"""
        ds_cells.append(mk_code(code))

    versions_repr = (
        '[\n'
        f'    ("Original",          "{HIRES}/t1"),\n'
        f'    ("v1\\n(2,2,5)",       "{DS_ROOT}/Crude/t1"),\n'
        f'    ("v2\\n(3,3,6)",       "{DS_ROOT}/Crude_v2/t1"),\n'
        f'    ("v3\\n(4,4,7)",       "{DS_ROOT}/Crude_v3/t1"),\n'
        f'    ("v4\\n(5,5,8)",       "{DS_ROOT}/Crude_v4/t1"),\n'
        f'    ("v5\\n(6,6,9)",       "{DS_ROOT}/Crude_v5/t1"),\n'
        ']'
    )
    versions_peri = (
        '[\n'
        f'    ("v1 (2,2,5)", "{DS_ROOT}/Crude/t1"),\n'
        f'    ("v2 (3,3,6)", "{DS_ROOT}/Crude_v2/t1"),\n'
        f'    ("v3 (4,4,7)", "{DS_ROOT}/Crude_v3/t1"),\n'
        f'    ("v4 (5,5,8)", "{DS_ROOT}/Crude_v4/t1"),\n'
        f'    ("v5 (6,6,9)", "{DS_ROOT}/Crude_v5/t1"),\n'
        ']'
    )
    sev_labels = '"v1 (2,2,5)","v2 (3,3,6)","v3 (4,4,7)","v4 (5,5,8)","v5 (6,6,9)"'
    wvars = [
        ("v1  (2,2,5)", f"{DS_ROOT}/Crude/t1"),
        ("v2  (3,3,6)", f"{DS_ROOT}/Crude_v2/t1"),
        ("v3  (4,4,7)", f"{DS_ROOT}/Crude_v3/t1"),
        ("v4  (5,5,8)", f"{DS_ROOT}/Crude_v4/t1"),
        ("v5  (6,6,9)", f"{DS_ROOT}/Crude_v5/t1"),
    ]

    new_cells = (
        [cells[0], setup_cell] +
        [mk_md("## Method Description"), cells[2], cells[3], cells[4]] +
        ds_cells +
        [mk_md("## Interactive Viewer\n"
               "**Fully self-contained** — run this cell without running any other cell.\n"
               "Select a subject, a severity variant, and a slice to compare Original vs degraded."),
         widget_cell("Crude", wvars)] +
        [mk_md("## Severity Gradient Visualization\n"
               "Run after generating at least some variant images.\n"
               "Produces a **2-row × 6-column** figure (Original + v1–v5): "
               "axial slice at lesion centre + lesion zoom. Saves PNG to `Downsampled_Data/`."),
         comparison_viewer_cell("Crude Downsampling", versions_repr,
                                f"{DS_ROOT}/Crude_variants_comparison.png")] +
        [mk_md("**Figure: Crude downsampling severity gradient.** "
               "Original through five levels of crude integer decimation. "
               "Parameter step: +1 xy, +1 z per version.")] +
        [mk_md("## Single-Subject Cross-Treatment View\n"
               "1 subject × 6 treatments (Original + v1–v5), selectable via dropdown.\n\n"
               "- **Row 1:** MRI axial slice with lesion fill (green) + perilesional ring (orange dashed, ~5 mm)\n"
               "- **Row 2:** Difference image (treatment − Original); red = brighter, blue = darker\n"
               "- **Row 3:** Local contrast histogram (lesion voxels − perilesional ring mean)\n\n"
               "Fully self-contained — run without any other cell."),
         per_subject_cell("Crude Downsampling", versions_repr)] +
        [mk_md("## Perilesional Contrast Analysis — All Severity Levels\n"
               "Run after generating all variant images.\n"
               "3-panel population plot: scatter vs Original, overlaid histograms, "
               "mean ± SEM by ordered severity.\n\n"
               "> Severity 1–5 is designed as an **evenly-spaced ordered predictor** for:\n"
               "> `Dice ~ severity * degradation_type + (1 | case_id)`"),
         perilesional_cell("Crude Downsampling", versions_peri,
                           f"{DS_ROOT}/Crude_variants_perilesional.png", sev_labels)]
    )

    out_path = NB_DIR / "Crude_variants.ipynb"
    with open(out_path, "w") as f: json.dump(nb_wrap(new_cells), f, indent=1)
    print(f"Wrote {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 2: THICK SLICES
# ─────────────────────────────────────────────────────────────────────────────
def make_thick_variants():
    nb = load_nb("Thick_Slices.ipynb")
    cells = nb["cells"]

    variant_paths = """\
THICK_V1_OUT=OUT_ROOT/"ThickSlices";    THICK_V1_T1=THICK_V1_OUT/"t1"; THICK_V1_MSK=THICK_V1_OUT/"masks"
THICK_V2_OUT=OUT_ROOT/"ThickSlices_v2"; THICK_V2_T1=THICK_V2_OUT/"t1"; THICK_V2_MSK=THICK_V2_OUT/"masks"
THICK_V3_OUT=OUT_ROOT/"ThickSlices_v3"; THICK_V3_T1=THICK_V3_OUT/"t1"; THICK_V3_MSK=THICK_V3_OUT/"masks"
THICK_V4_OUT=OUT_ROOT/"ThickSlices_v4"; THICK_V4_T1=THICK_V4_OUT/"t1"; THICK_V4_MSK=THICK_V4_OUT/"masks"
THICK_V5_OUT=OUT_ROOT/"ThickSlices_v5"; THICK_V5_T1=THICK_V5_OUT/"t1"; THICK_V5_MSK=THICK_V5_OUT/"masks"
for d in [THICK_V1_T1,THICK_V1_MSK,THICK_V2_T1,THICK_V2_MSK,THICK_V3_T1,THICK_V3_MSK,
          THICK_V4_T1,THICK_V4_MSK,THICK_V5_T1,THICK_V5_MSK]: d.mkdir(parents=True,exist_ok=True)"""

    setup_cell = make_setup_cell("ThickSlices", variant_paths)
    set_src(cells[0],
        "# Thick Slices — Severity Variants (v2–v5)\n\n"
        "| Version | factor_z | sigma_z_vox | Step |\n"
        "|---------|----------|-------------|------|\n"
        "| v1 (original) | 5 | 2.0 | — |\n"
        "| v2 | 6 | 2.5 | +1 / +0.5 |\n"
        "| v3 | 7 | 3.0 | +1 / +0.5 |\n"
        "| v4 | 8 | 3.5 | +1 / +0.5 |\n"
        "| v5 | 9 | 4.0 | +1 / +0.5 |"
    )

    TVARS = [("v2",6,2.5),("v3",7,3.0),("v4",8,3.5),("v5",9,4.0)]
    ds_cells = []
    for vname,fz,sz in TVARS:
        vU = vname.upper()
        ds_cells.append(mk_md(
            f"## Generate `ThickSlices_{vname}` — Severity {vname[1]}/5\n"
            f"**Parameters:** `factor_z={fz}`, `sigma_z_vox={sz}`\n\n"
            f"Output → `Downsampled_Data/ThickSlices_{vname}/`"
        ))
        code = f"""\
# === ThickSlices_{vname}: factor_z={fz}, sigma_z_vox={sz} — severity {vname[1]}/5 ===
import shutil
from scipy.ndimage import gaussian_filter

factor_z=({fz}); sigma_z_vox=({sz}); OVERWRITE=False
OUT_DIR=THICK_{vU}_OUT; OUT_IMG_DIR=THICK_{vU}_T1; OUT_MSK_DIR=THICK_{vU}_MSK
for d in (OUT_DIR,OUT_IMG_DIR,OUT_MSK_DIR): d.mkdir(parents=True,exist_ok=True)

pairs=discover_pairs(SRC_T1_DIR,SRC_MASK_DIR)
print(f"[ThickSlices_{vname}] pairs: {{len(pairs)}}  factor_z={fz}  sigma_z={sz}")

wrote=0
for i,(img_p,msk_p) in enumerate(pairs,1):
    img_ref=load_nii(str(img_p)); x=data_f32(img_ref)
    xb=gaussian_filter(x,sigma=(0.,0.,float(sigma_z_vox)),mode="nearest")
    xu=pad_or_crop_to(np.repeat(xb[:,:,::factor_z],factor_z,axis=2),x.shape)
    base=strip_ext(img_p.name).replace("_T1w_MNI_norm","").replace("_T1w","")
    out_img=OUT_IMG_DIR/f"{{base}}_T1w_MNI_norm.nii.gz"
    out_msk=OUT_MSK_DIR/f"{{base}}_lesion_mask_MNI_clean.nii.gz"
    if not OVERWRITE and out_img.exists() and out_msk.exists(): continue
    save_like(img_ref,xu.astype(np.float32),out_img,dtype=np.float32)
    shutil.copy2(str(msk_p),str(out_msk)); wrote+=1
    if i%10==0 or i==len(pairs): print(f"  [{{i}}/{{len(pairs)}}] {{out_img.name}}")
print(f"Done → {{OUT_DIR}} | wrote {{wrote}}")
"""
        ds_cells.append(mk_code(code))

    versions_repr = (
        '[\n'
        f'    ("Original",             "{HIRES}/t1"),\n'
        f'    ("v1\\nfz=5 σ=2.0",       "{DS_ROOT}/ThickSlices/t1"),\n'
        f'    ("v2\\nfz=6 σ=2.5",       "{DS_ROOT}/ThickSlices_v2/t1"),\n'
        f'    ("v3\\nfz=7 σ=3.0",       "{DS_ROOT}/ThickSlices_v3/t1"),\n'
        f'    ("v4\\nfz=8 σ=3.5",       "{DS_ROOT}/ThickSlices_v4/t1"),\n'
        f'    ("v5\\nfz=9 σ=4.0",       "{DS_ROOT}/ThickSlices_v5/t1"),\n'
        ']'
    )
    versions_peri = (
        '[\n'
        f'    ("v1 fz=5", "{DS_ROOT}/ThickSlices/t1"),\n'
        f'    ("v2 fz=6", "{DS_ROOT}/ThickSlices_v2/t1"),\n'
        f'    ("v3 fz=7", "{DS_ROOT}/ThickSlices_v3/t1"),\n'
        f'    ("v4 fz=8", "{DS_ROOT}/ThickSlices_v4/t1"),\n'
        f'    ("v5 fz=9", "{DS_ROOT}/ThickSlices_v5/t1"),\n'
        ']'
    )
    sev_labels = '"v1 fz=5","v2 fz=6","v3 fz=7","v4 fz=8","v5 fz=9"'
    wvars = [
        ("v1  fz=5 σ=2.0", f"{DS_ROOT}/ThickSlices/t1"),
        ("v2  fz=6 σ=2.5", f"{DS_ROOT}/ThickSlices_v2/t1"),
        ("v3  fz=7 σ=3.0", f"{DS_ROOT}/ThickSlices_v3/t1"),
        ("v4  fz=8 σ=3.5", f"{DS_ROOT}/ThickSlices_v4/t1"),
        ("v5  fz=9 σ=4.0", f"{DS_ROOT}/ThickSlices_v5/t1"),
    ]

    new_cells = (
        [cells[0], setup_cell] +
        [mk_md("## Method Description"), cells[2], cells[3]] +
        ds_cells +
        [mk_md("## Interactive Viewer\n"
               "**Fully self-contained** — run without any other cell.\n"
               "Select subject, variant, and slice."),
         widget_cell("Thick Slices", wvars)] +
        [mk_md("## Severity Gradient Visualization\n"
               "2-row × 6-column: axial slice at lesion centre + lesion zoom. "
               "Saves PNG to `Downsampled_Data/`."),
         comparison_viewer_cell("Thick Slices", versions_repr,
                                f"{DS_ROOT}/ThickSlices_variants_comparison.png")] +
        [mk_md("**Figure: Thick slices severity gradient.** "
               "Z-blur + Z-decimation from factor_z=5 (v1) to 9 (v5).")] +
        [mk_md("## Single-Subject Cross-Treatment View\n"
               "1 subject × 6 treatments (Original + v1–v5), selectable via dropdown.\n\n"
               "- **Row 1:** MRI + lesion (green) + perilesional ring (orange dashed, ~5 mm)\n"
               "- **Row 2:** Difference (treatment − Original)\n"
               "- **Row 3:** Local contrast histogram (lesion − peri mean)\n\n"
               "Fully self-contained — run without any other cell."),
         per_subject_cell("Thick Slices", versions_repr)] +
        [mk_md("## Perilesional Contrast Analysis — All Severity Levels\n"
               "3-panel population plot. Ordered severity 1–5 for mixed model analysis."),
         perilesional_cell("Thick Slices", versions_peri,
                           f"{DS_ROOT}/ThickSlices_variants_perilesional.png", sev_labels)]
    )

    out_path = NB_DIR / "Thick_Slices_variants.ipynb"
    with open(out_path, "w") as f: json.dump(nb_wrap(new_cells), f, indent=1)
    print(f"Wrote {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 3: IN-PLANE COARSE
# ─────────────────────────────────────────────────────────────────────────────
def make_inplane_variants():
    nb = load_nb("In_Plane_Coarse.ipynb")
    cells = nb["cells"]

    variant_paths = """\
INPLANE_V1_OUT=OUT_ROOT/"InPlaneCoarse";    INPLANE_V1_T1=INPLANE_V1_OUT/"t1"; INPLANE_V1_MSK=INPLANE_V1_OUT/"masks"
INPLANE_V2_OUT=OUT_ROOT/"InPlaneCoarse_v2"; INPLANE_V2_T1=INPLANE_V2_OUT/"t1"; INPLANE_V2_MSK=INPLANE_V2_OUT/"masks"
INPLANE_V3_OUT=OUT_ROOT/"InPlaneCoarse_v3"; INPLANE_V3_T1=INPLANE_V3_OUT/"t1"; INPLANE_V3_MSK=INPLANE_V3_OUT/"masks"
INPLANE_V4_OUT=OUT_ROOT/"InPlaneCoarse_v4"; INPLANE_V4_T1=INPLANE_V4_OUT/"t1"; INPLANE_V4_MSK=INPLANE_V4_OUT/"masks"
INPLANE_V5_OUT=OUT_ROOT/"InPlaneCoarse_v5"; INPLANE_V5_T1=INPLANE_V5_OUT/"t1"; INPLANE_V5_MSK=INPLANE_V5_OUT/"masks"
for d in [INPLANE_V1_T1,INPLANE_V1_MSK,INPLANE_V2_T1,INPLANE_V2_MSK,INPLANE_V3_T1,INPLANE_V3_MSK,
          INPLANE_V4_T1,INPLANE_V4_MSK,INPLANE_V5_T1,INPLANE_V5_MSK]: d.mkdir(parents=True,exist_ok=True)"""

    setup_cell = make_setup_cell("InPlaneCoarse", variant_paths)
    set_src(cells[0],
        "# In-Plane Coarse — Severity Variants (v2–v5)\n\n"
        "| Version | factor_xy | sigma_xy | Step |\n"
        "|---------|-----------|----------|------|\n"
        "| v1 (original) | 2 | 1.0 | — |\n"
        "| v2 | 3 | 1.5 | +1 / +0.5 |\n"
        "| v3 | 4 | 2.0 | +1 / +0.5 |\n"
        "| v4 | 5 | 2.5 | +1 / +0.5 |\n"
        "| v5 | 6 | 3.0 | +1 / +0.5 |"
    )

    IVARS = [("v2",3,1.5),("v3",4,2.0),("v4",5,2.5),("v5",6,3.0)]
    ds_cells = []
    for vname,fxy,sxy in IVARS:
        vU = vname.upper()
        ds_cells.append(mk_md(
            f"## Generate `InPlaneCoarse_{vname}` — Severity {vname[1]}/5\n"
            f"**Parameters:** `factor_xy={fxy}`, `sigma_xy={sxy}`\n\n"
            f"Output → `Downsampled_Data/InPlaneCoarse_{vname}/`"
        ))
        code = f"""\
# === InPlaneCoarse_{vname}: factor_xy={fxy}, sigma_xy={sxy} — severity {vname[1]}/5 ===
import shutil
from scipy.ndimage import gaussian_filter

factor_xy={fxy}; sigma_xy={sxy}; OVERWRITE=False
OUT_DIR=INPLANE_{vU}_OUT; OUT_IMG_DIR=INPLANE_{vU}_T1; OUT_MSK_DIR=INPLANE_{vU}_MSK
for d in (OUT_DIR,OUT_IMG_DIR,OUT_MSK_DIR): d.mkdir(parents=True,exist_ok=True)

pairs=discover_pairs(SRC_T1_DIR,SRC_MASK_DIR)
print(f"[InPlaneCoarse_{vname}] pairs: {{len(pairs)}}  factor_xy={fxy}  sigma_xy={sxy}")

wrote=0
for i,(img_p,msk_p) in enumerate(pairs,1):
    img_ref=load_nii(img_p); x=data_f32(img_ref)
    xb=gaussian_filter(x,sigma=(float(sigma_xy),float(sigma_xy),0.),mode="nearest")
    xd=xb[::factor_xy,::factor_xy,:]
    xu=pad_or_crop_to(np.repeat(np.repeat(xd,factor_xy,axis=0),factor_xy,axis=1),x.shape)
    base=strip_ext(img_p.name).replace("_T1w_MNI_norm","").replace("_T1w","")
    out_img=OUT_IMG_DIR/f"{{base}}_T1w_MNI_norm.nii.gz"
    out_msk=OUT_MSK_DIR/f"{{base}}_lesion_mask_MNI_clean.nii.gz"
    if not OVERWRITE and out_img.exists() and out_msk.exists(): continue
    save_like(img_ref,xu.astype(np.float32),out_img,dtype=np.float32)
    out_msk.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(str(msk_p),str(out_msk))
    wrote+=1
    if i%10==0 or i==len(pairs): print(f"  [{{i}}/{{len(pairs)}}] {{out_img.name}}")
print(f"Done → {{OUT_DIR}} | wrote {{wrote}}")
"""
        ds_cells.append(mk_code(code))

    versions_repr = (
        '[\n'
        f'    ("Original",               "{HIRES}/t1"),\n'
        f'    ("v1\\nfxy=2 σ=1.0",        "{DS_ROOT}/InPlaneCoarse/t1"),\n'
        f'    ("v2\\nfxy=3 σ=1.5",        "{DS_ROOT}/InPlaneCoarse_v2/t1"),\n'
        f'    ("v3\\nfxy=4 σ=2.0",        "{DS_ROOT}/InPlaneCoarse_v3/t1"),\n'
        f'    ("v4\\nfxy=5 σ=2.5",        "{DS_ROOT}/InPlaneCoarse_v4/t1"),\n'
        f'    ("v5\\nfxy=6 σ=3.0",        "{DS_ROOT}/InPlaneCoarse_v5/t1"),\n'
        ']'
    )
    versions_peri = (
        '[\n'
        f'    ("v1 fxy=2", "{DS_ROOT}/InPlaneCoarse/t1"),\n'
        f'    ("v2 fxy=3", "{DS_ROOT}/InPlaneCoarse_v2/t1"),\n'
        f'    ("v3 fxy=4", "{DS_ROOT}/InPlaneCoarse_v3/t1"),\n'
        f'    ("v4 fxy=5", "{DS_ROOT}/InPlaneCoarse_v4/t1"),\n'
        f'    ("v5 fxy=6", "{DS_ROOT}/InPlaneCoarse_v5/t1"),\n'
        ']'
    )
    sev_labels = '"v1 fxy=2","v2 fxy=3","v3 fxy=4","v4 fxy=5","v5 fxy=6"'
    wvars = [
        ("v1  fxy=2 σ=1.0", f"{DS_ROOT}/InPlaneCoarse/t1"),
        ("v2  fxy=3 σ=1.5", f"{DS_ROOT}/InPlaneCoarse_v2/t1"),
        ("v3  fxy=4 σ=2.0", f"{DS_ROOT}/InPlaneCoarse_v3/t1"),
        ("v4  fxy=5 σ=2.5", f"{DS_ROOT}/InPlaneCoarse_v4/t1"),
        ("v5  fxy=6 σ=3.0", f"{DS_ROOT}/InPlaneCoarse_v5/t1"),
    ]

    new_cells = (
        [cells[0], setup_cell] +
        [mk_md("## Method Description"), cells[2], cells[3]] +
        ds_cells +
        [mk_md("## Interactive Viewer\n**Fully self-contained** — run without any other cell."),
         widget_cell("In-Plane Coarse", wvars)] +
        [mk_md("## Severity Gradient Visualization\n"
               "2-row × 6-column figure. Saves PNG to `Downsampled_Data/`."),
         comparison_viewer_cell("In-Plane Coarse", versions_repr,
                                f"{DS_ROOT}/InPlaneCoarse_variants_comparison.png")] +
        [mk_md("**Figure: In-plane coarse severity gradient.** "
               "XY blur + decimation from factor_xy=2 (v1) to 6 (v5).")] +
        [mk_md("## Single-Subject Cross-Treatment View\n"
               "1 subject × 6 treatments (Original + v1–v5), selectable via dropdown.\n\n"
               "- **Row 1:** MRI + lesion (green) + perilesional ring (orange dashed, ~5 mm)\n"
               "- **Row 2:** Difference (treatment − Original)\n"
               "- **Row 3:** Local contrast histogram (lesion − peri mean)\n\n"
               "Fully self-contained — run without any other cell."),
         per_subject_cell("In-Plane Coarse", versions_repr)] +
        [mk_md("## Perilesional Contrast Analysis — All Severity Levels"),
         perilesional_cell("In-Plane Coarse", versions_peri,
                           f"{DS_ROOT}/InPlaneCoarse_variants_perilesional.png", sev_labels)]
    )

    out_path = NB_DIR / "In_Plane_Coarse_variants.ipynb"
    with open(out_path, "w") as f: json.dump(nb_wrap(new_cells), f, indent=1)
    print(f"Wrote {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 4: REDUCED SNR
# ─────────────────────────────────────────────────────────────────────────────
def make_snr_variants():
    nb = load_nb("Reduced_SNR.ipynb")
    cells = nb["cells"]

    variant_paths = """\
SNR_V1_OUT=OUT_ROOT/"ReducedSNR";    SNR_V1_T1=SNR_V1_OUT/"t1"; SNR_V1_MSK=SNR_V1_OUT/"masks"
SNR_V2_OUT=OUT_ROOT/"ReducedSNR_v2"; SNR_V2_T1=SNR_V2_OUT/"t1"; SNR_V2_MSK=SNR_V2_OUT/"masks"
SNR_V3_OUT=OUT_ROOT/"ReducedSNR_v3"; SNR_V3_T1=SNR_V3_OUT/"t1"; SNR_V3_MSK=SNR_V3_OUT/"masks"
SNR_V4_OUT=OUT_ROOT/"ReducedSNR_v4"; SNR_V4_T1=SNR_V4_OUT/"t1"; SNR_V4_MSK=SNR_V4_OUT/"masks"
SNR_V5_OUT=OUT_ROOT/"ReducedSNR_v5"; SNR_V5_T1=SNR_V5_OUT/"t1"; SNR_V5_MSK=SNR_V5_OUT/"masks"
for d in [SNR_V1_T1,SNR_V1_MSK,SNR_V2_T1,SNR_V2_MSK,SNR_V3_T1,SNR_V3_MSK,
          SNR_V4_T1,SNR_V4_MSK,SNR_V5_T1,SNR_V5_MSK]: d.mkdir(parents=True,exist_ok=True)"""

    snr_method_helpers = """\
def brain_like_mask(x):
    m = x > 0 if (x>0).sum()>1000 else x>np.percentile(x,60.)
    from scipy.ndimage import binary_closing, generate_binary_structure
    return binary_closing(m,structure=generate_binary_structure(3,1),iterations=1)

def pnorm01(x,p_lo=1,p_hi=99,eps=1e-6):
    lo,hi=np.percentile(x,[p_lo,p_hi])
    if hi-lo<eps: hi=lo+eps
    return np.clip((x-lo)/(hi-lo),0,1).astype(np.float32),float(lo),float(hi)

def affine_match(src,ref,mask):
    s=src[mask].ravel().astype(np.float64); r=ref[mask].ravel().astype(np.float64)
    if s.size<50: return 1.,0.
    A=np.vstack([s,np.ones_like(s)]).T; a,b=np.linalg.lstsq(A,r,rcond=None)[0]
    return float(a if np.isfinite(a) else 1.),float(b if np.isfinite(b) else 0.)

def measure_snr(img,mask):
    v=img[mask].astype(np.float32); mu=float(v.mean()); sd=float(v.std(ddof=1)+1e-6)
    return mu/sd,mu,sd

def add_kspace_noise_calibrated(x_norm,brain_mask,target_snr,rng,iters=4):
    X=np.fft.fftn(x_norm.astype(np.complex64),norm="ortho")
    _,mu0,_=measure_snr(x_norm,brain_mask)
    sk_lo=0.; sk_hi=max(mu0/max(target_snr,1e-3)*8.,1e-4); sk=max(mu0/max(target_snr,1e-3),1e-5)
    last_mag=None
    for _ in range(iters):
        nr=rng.normal(0.,sk,X.shape).astype(np.float32); ni=rng.normal(0.,sk,X.shape).astype(np.float32)
        mag=np.abs(np.fft.ifftn(X+(nr+1j*ni).astype(np.complex64),norm="ortho")).astype(np.float32)
        snr,_,_=measure_snr(mag,brain_mask)
        if snr>target_snr: sk_lo=sk; sk=.5*(sk+sk_hi)
        else: sk_hi=sk; sk=.5*(sk+sk_lo)
        last_mag=mag
    _,mu_f,_=measure_snr(last_mag,brain_mask)
    return np.clip(mu0/max(mu_f,1e-6)*last_mag,0.,1.).astype(np.float32)"""

    setup_cell = make_setup_cell("ReducedSNR", variant_paths, snr_method_helpers)
    set_src(cells[0],
        "# Reduced SNR — Severity Variants (v2–v5)\n\n"
        "| Version | TARGET_SNR | Step |\n"
        "|---------|------------|------|\n"
        "| v1 (original) | 20 | — |\n"
        "| v2 | 16 | −4 |\n"
        "| v3 | 12 | −4 |\n"
        "| v4 | 8  | −4 |\n"
        "| v5 | 4  | −4 |"
    )

    SVARS = [("v2",16.),("v3",12.),("v4",8.),("v5",4.)]
    ds_cells = []
    for vname,tsnr in SVARS:
        vU = vname.upper()
        ds_cells.append(mk_md(
            f"## Generate `ReducedSNR_{vname}` — Severity {vname[1]}/5\n"
            f"**Parameters:** `TARGET_SNR={tsnr}` (lower = noisier)\n\n"
            f"Output → `Downsampled_Data/ReducedSNR_{vname}/`"
        ))
        code = f"""\
# === ReducedSNR_{vname}: TARGET_SNR={tsnr} — severity {vname[1]}/5 ===
import shutil

TARGET_SNR={tsnr}; SEED=123; OVERWRITE=False
OUT_DIR=SNR_{vU}_OUT; OUT_IMG_DIR=SNR_{vU}_T1; OUT_MSK_DIR=SNR_{vU}_MSK
for d in (OUT_DIR,OUT_IMG_DIR,OUT_MSK_DIR): d.mkdir(parents=True,exist_ok=True)

rng=np.random.default_rng(SEED)
pairs=discover_pairs(SRC_T1_DIR,SRC_MASK_DIR)
print(f"[ReducedSNR_{vname}] pairs: {{len(pairs)}}  TARGET_SNR={tsnr}")

wrote=0
for i,(img_p,msk_p) in enumerate(pairs,1):
    img_ref=load_nii(img_p); x=data_f32(img_ref)
    bl=brain_like_mask(x); x_norm,_,_=pnorm01(x,1,99)
    x_noisy=add_kspace_noise_calibrated(x_norm,bl,TARGET_SNR,rng,iters=4)
    a,b=affine_match(x_noisy,x_norm,bl)
    x_adj=np.clip(a*x_noisy+b,0,1).astype(np.float32)
    base=strip_ext(img_p.name).replace("_T1w_MNI_norm","").replace("_T1w","")
    out_img=OUT_IMG_DIR/f"{{base}}_T1w_MNI_norm.nii.gz"
    out_msk=OUT_MSK_DIR/f"{{base}}_lesion_mask_MNI_clean.nii.gz"
    if not OVERWRITE and out_img.exists() and out_msk.exists(): continue
    save_like(img_ref,x_adj,out_img,dtype=np.float32)
    out_msk.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(str(msk_p),str(out_msk))
    wrote+=1
    if i%10==0 or i==len(pairs):
        snr_f,_,_=measure_snr(x_adj,bl)
        print(f"  [{{i}}/{{len(pairs)}}] {{out_img.name}}  measured_snr={{snr_f:.2f}}")
print(f"Done → {{OUT_DIR}} | wrote {{wrote}}")
"""
        ds_cells.append(mk_code(code))

    versions_repr = (
        '[\n'
        f'    ("Original",         "{HIRES}/t1"),\n'
        f'    ("v1\\nSNR=20",        "{DS_ROOT}/ReducedSNR/t1"),\n'
        f'    ("v2\\nSNR=16",        "{DS_ROOT}/ReducedSNR_v2/t1"),\n'
        f'    ("v3\\nSNR=12",        "{DS_ROOT}/ReducedSNR_v3/t1"),\n'
        f'    ("v4\\nSNR=8",         "{DS_ROOT}/ReducedSNR_v4/t1"),\n'
        f'    ("v5\\nSNR=4",         "{DS_ROOT}/ReducedSNR_v5/t1"),\n'
        ']'
    )
    versions_peri = (
        '[\n'
        f'    ("v1 SNR=20", "{DS_ROOT}/ReducedSNR/t1"),\n'
        f'    ("v2 SNR=16", "{DS_ROOT}/ReducedSNR_v2/t1"),\n'
        f'    ("v3 SNR=12", "{DS_ROOT}/ReducedSNR_v3/t1"),\n'
        f'    ("v4 SNR=8",  "{DS_ROOT}/ReducedSNR_v4/t1"),\n'
        f'    ("v5 SNR=4",  "{DS_ROOT}/ReducedSNR_v5/t1"),\n'
        ']'
    )
    sev_labels = '"v1 SNR=20","v2 SNR=16","v3 SNR=12","v4 SNR=8","v5 SNR=4"'
    wvars = [
        ("v1  SNR=20", f"{DS_ROOT}/ReducedSNR/t1"),
        ("v2  SNR=16", f"{DS_ROOT}/ReducedSNR_v2/t1"),
        ("v3  SNR=12", f"{DS_ROOT}/ReducedSNR_v3/t1"),
        ("v4  SNR=8",  f"{DS_ROOT}/ReducedSNR_v4/t1"),
        ("v5  SNR=4",  f"{DS_ROOT}/ReducedSNR_v5/t1"),
    ]

    new_cells = (
        [cells[0], setup_cell] +
        [mk_md("## Method Description"), cells[2], cells[3]] +
        ds_cells +
        [mk_md("## Interactive Viewer\n**Fully self-contained** — run without any other cell."),
         widget_cell("Reduced SNR", wvars)] +
        [mk_md("## Severity Gradient Visualization"),
         comparison_viewer_cell("Reduced SNR", versions_repr,
                                f"{DS_ROOT}/ReducedSNR_variants_comparison.png")] +
        [mk_md("**Figure: Reduced SNR severity gradient.** "
               "K-space complex noise from SNR=20 (v1) to SNR=4 (v5).")] +
        [mk_md("## Single-Subject Cross-Treatment View\n"
               "1 subject × 6 treatments (Original + v1–v5), selectable via dropdown.\n\n"
               "- **Row 1:** MRI + lesion (green) + perilesional ring (orange dashed, ~5 mm)\n"
               "- **Row 2:** Difference (treatment − Original)\n"
               "- **Row 3:** Local contrast histogram (lesion − peri mean)\n\n"
               "Fully self-contained — run without any other cell."),
         per_subject_cell("Reduced SNR", versions_repr)] +
        [mk_md("## Perilesional Contrast Analysis — All Severity Levels"),
         perilesional_cell("Reduced SNR", versions_peri,
                           f"{DS_ROOT}/ReducedSNR_variants_perilesional.png", sev_labels)]
    )

    out_path = NB_DIR / "Reduced_SNR_variants.ipynb"
    with open(out_path, "w") as f: json.dump(nb_wrap(new_cells), f, indent=1)
    print(f"Wrote {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 5: RIGID JITTER
# ─────────────────────────────────────────────────────────────────────────────
def make_jitter_variants():
    nb = load_nb("Rigid_Jitter.ipynb")
    cells = nb["cells"]

    variant_paths = """\
JITTER_V1_OUT=OUT_ROOT/"RigidJitter";    JITTER_V1_T1=JITTER_V1_OUT/"t1"; JITTER_V1_MSK=JITTER_V1_OUT/"masks"
JITTER_V2_OUT=OUT_ROOT/"RigidJitter_v2"; JITTER_V2_T1=JITTER_V2_OUT/"t1"; JITTER_V2_MSK=JITTER_V2_OUT/"masks"
JITTER_V3_OUT=OUT_ROOT/"RigidJitter_v3"; JITTER_V3_T1=JITTER_V3_OUT/"t1"; JITTER_V3_MSK=JITTER_V3_OUT/"masks"
JITTER_V4_OUT=OUT_ROOT/"RigidJitter_v4"; JITTER_V4_T1=JITTER_V4_OUT/"t1"; JITTER_V4_MSK=JITTER_V4_OUT/"masks"
JITTER_V5_OUT=OUT_ROOT/"RigidJitter_v5"; JITTER_V5_T1=JITTER_V5_OUT/"t1"; JITTER_V5_MSK=JITTER_V5_OUT/"masks"
for d in [JITTER_V1_T1,JITTER_V1_MSK,JITTER_V2_T1,JITTER_V2_MSK,JITTER_V3_T1,JITTER_V3_MSK,
          JITTER_V4_T1,JITTER_V4_MSK,JITTER_V5_T1,JITTER_V5_MSK]: d.mkdir(parents=True,exist_ok=True)"""

    setup_cell = make_setup_cell("RigidJitter", variant_paths)
    set_src(cells[0],
        "# Rigid Jitter — Severity Variants (v2–v5)\n\n"
        "| Version | deg_range | px_range | Step |\n"
        "|---------|-----------|----------|------|\n"
        "| v1 (original) | ±2° | ±2 px | — |\n"
        "| v2 | ±4° | ±4 px | +2 |\n"
        "| v3 | ±6° | ±6 px | +2 |\n"
        "| v4 | ±8° | ±8 px | +2 |\n"
        "| v5 | ±10° | ±10 px | +2 |"
    )

    JVARS = [("v2",4.,4.),("v3",6.,6.),("v4",8.,8.),("v5",10.,10.)]
    ds_cells = []
    for vname,deg,px in JVARS:
        vU = vname.upper()
        ds_cells.append(mk_md(
            f"## Generate `RigidJitter_{vname}` — Severity {vname[1]}/5\n"
            f"**Parameters:** `deg_range=±{deg}°`, `px_range=±{px}px`\n\n"
            f"Output → `Downsampled_Data/RigidJitter_{vname}/`"
        ))
        code = f"""\
# === RigidJitter_{vname}: deg_range=±{deg}, px_range=±{px} — severity {vname[1]}/5 ===
import shutil
from scipy.ndimage import rotate, shift as ndshift

deg_range={deg}; px_range={px}; SEED=7; OVERWRITE=False
OUT_DIR=JITTER_{vU}_OUT; OUT_IMG_DIR=JITTER_{vU}_T1; OUT_MSK_DIR=JITTER_{vU}_MSK
for d in (OUT_DIR,OUT_IMG_DIR,OUT_MSK_DIR): d.mkdir(parents=True,exist_ok=True)

pairs=discover_pairs(SRC_T1_DIR,SRC_MASK_DIR)
print(f"[RigidJitter_{vname}] pairs: {{len(pairs)}}  deg=±{deg}  px=±{px}")

rng=np.random.default_rng(SEED); wrote=0
for i,(img_p,msk_p) in enumerate(pairs,1):
    img_ref=load_nii(img_p); x=data_f32(img_ref)
    H,W,Z=x.shape; xm=np.empty_like(x,dtype=np.float32)
    angs=rng.uniform(-deg_range,deg_range,Z); dxs=rng.uniform(-px_range,px_range,Z); dys=rng.uniform(-px_range,px_range,Z)
    for k in range(Z):
        sl=rotate(x[:,:,k],angle=float(angs[k]),reshape=False,order=1,mode="nearest")
        xm[:,:,k]=ndshift(sl,shift=(float(dys[k]),float(dxs[k])),order=1,mode="nearest")
    base=strip_ext(img_p.name).replace("_T1w_MNI_norm","").replace("_T1w","")
    out_img=OUT_IMG_DIR/f"{{base}}_T1w_MNI_norm.nii.gz"
    out_msk=OUT_MSK_DIR/f"{{base}}_lesion_mask_MNI_clean.nii.gz"
    if not OVERWRITE and out_img.exists() and out_msk.exists(): continue
    save_like(img_ref,xm,out_img,dtype=np.float32)
    out_msk.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(str(msk_p),str(out_msk))
    wrote+=1
    if i%10==0 or i==len(pairs): print(f"  [{{i}}/{{len(pairs)}}] {{out_img.name}}")
print(f"Done → {{OUT_DIR}} | wrote {{wrote}}")
"""
        ds_cells.append(mk_code(code))

    versions_repr = (
        '[\n'
        f'    ("Original",              "{HIRES}/t1"),\n'
        f'    ("v1\\ndeg=±2 px=±2",      "{DS_ROOT}/RigidJitter/t1"),\n'
        f'    ("v2\\ndeg=±4 px=±4",      "{DS_ROOT}/RigidJitter_v2/t1"),\n'
        f'    ("v3\\ndeg=±6 px=±6",      "{DS_ROOT}/RigidJitter_v3/t1"),\n'
        f'    ("v4\\ndeg=±8 px=±8",      "{DS_ROOT}/RigidJitter_v4/t1"),\n'
        f'    ("v5\\ndeg=±10 px=±10",    "{DS_ROOT}/RigidJitter_v5/t1"),\n'
        ']'
    )
    versions_peri = (
        '[\n'
        f'    ("v1 deg=±2",  "{DS_ROOT}/RigidJitter/t1"),\n'
        f'    ("v2 deg=±4",  "{DS_ROOT}/RigidJitter_v2/t1"),\n'
        f'    ("v3 deg=±6",  "{DS_ROOT}/RigidJitter_v3/t1"),\n'
        f'    ("v4 deg=±8",  "{DS_ROOT}/RigidJitter_v4/t1"),\n'
        f'    ("v5 deg=±10", "{DS_ROOT}/RigidJitter_v5/t1"),\n'
        ']'
    )
    sev_labels = '"v1 deg=±2","v2 deg=±4","v3 deg=±6","v4 deg=±8","v5 deg=±10"'
    wvars = [
        ("v1  ±2°/±2px",  f"{DS_ROOT}/RigidJitter/t1"),
        ("v2  ±4°/±4px",  f"{DS_ROOT}/RigidJitter_v2/t1"),
        ("v3  ±6°/±6px",  f"{DS_ROOT}/RigidJitter_v3/t1"),
        ("v4  ±8°/±8px",  f"{DS_ROOT}/RigidJitter_v4/t1"),
        ("v5  ±10°/±10px",f"{DS_ROOT}/RigidJitter_v5/t1"),
    ]

    # Jitter has empty cell at index 4; downsampling was at index 5
    new_cells = (
        [cells[0], setup_cell] +
        [mk_md("## Method Description"), cells[2], cells[3]] +
        ds_cells +
        [mk_md("## Interactive Viewer\n**Fully self-contained** — run without any other cell."),
         widget_cell("Rigid Jitter", wvars)] +
        [mk_md("## Severity Gradient Visualization"),
         comparison_viewer_cell("Rigid Jitter", versions_repr,
                                f"{DS_ROOT}/RigidJitter_variants_comparison.png")] +
        [mk_md("**Figure: Rigid jitter severity gradient.** "
               "Per-slice rotation/shift from ±2°/±2px (v1) to ±10°/±10px (v5).")] +
        [mk_md("## Single-Subject Cross-Treatment View\n"
               "1 subject × 6 treatments (Original + v1–v5), selectable via dropdown.\n\n"
               "- **Row 1:** MRI + lesion (green) + perilesional ring (orange dashed, ~5 mm)\n"
               "- **Row 2:** Difference (treatment − Original)\n"
               "- **Row 3:** Local contrast histogram (lesion − peri mean)\n\n"
               "Fully self-contained — run without any other cell."),
         per_subject_cell("Rigid Jitter", versions_repr)] +
        [mk_md("## Perilesional Contrast Analysis — All Severity Levels"),
         perilesional_cell("Rigid Jitter", versions_peri,
                           f"{DS_ROOT}/RigidJitter_variants_perilesional.png", sev_labels)]
    )

    out_path = NB_DIR / "Rigid_Jitter_variants.ipynb"
    with open(out_path, "w") as f: json.dump(nb_wrap(new_cells), f, indent=1)
    print(f"Wrote {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
make_crude_variants()
make_thick_variants()
make_inplane_variants()
make_snr_variants()
make_jitter_variants()
print("\nAll variants notebooks regenerated.")
