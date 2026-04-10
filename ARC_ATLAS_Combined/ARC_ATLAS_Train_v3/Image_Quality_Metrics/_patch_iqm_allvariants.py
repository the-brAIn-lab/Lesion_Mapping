#!/usr/bin/env python3
"""Patch ARC_ATLAS_Image_Quality_Metrics_all_variants.ipynb:
Replace cells 7, 10, 12, 14 with severity-variant-aware versions.
"""
import json
from pathlib import Path

NB_PATH = Path("/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/ARC_ATLAS_Train_v3/Image_Quality_Metrics/ARC_ATLAS_Image_Quality_Metrics_all_variants.ipynb")
IQM_DIR = Path("/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/ARC_ATLAS_Train_v3/Image_Quality_Metrics")
DS_BASE = Path("/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/A_A_Combined_Data/Downsampled_Data")
HIRES   = Path("/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/A_A_Combined_Data/Processed_HiresLowres_Split_Data/test_hires")

def mk_code(s): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s}
def mk_md(s):   return {"cell_type":"markdown","metadata":{},"source":s}

nb = json.loads(NB_PATH.read_text())
cells = nb["cells"]

# ─────────────────────────────────────────────────────────────────────────────
# CELL 7: IQM violin — grouped by method, one colour-ramp per method for v1-v5
# ─────────────────────────────────────────────────────────────────────────────
NEW_CELL_7 = mk_code('''\
# === Image Quality Metrics — All Severity Variants: Visualisation ===
# Self-contained — run without executing other cells.
#
# Groups variants by method and shows how IQM changes across severity 1-5.
# Left:  3-panel violin matrix (one panel per IQM, columns = all 27 variants)
# Right: Line plot — mean IQM vs severity per method (dose-response view)

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import warnings

IQM_DIR = Path("/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/ARC_ATLAS_Train_v3/Image_Quality_Metrics")

csvs = sorted(IQM_DIR.glob("image_quality_metrics_all_variants_*.csv"))
assert csvs, "Run cell 4 first to generate the CSV."
df = pd.read_csv(csvs[-1])
print(f"Loading: {csvs[-1].name}  |  {len(df)} rows  |  variants: {sorted(df['variant'].unique())}")

# ── Config ──────────────────────────────────────────────────────────────────
METHODS = ["Crude","ThickSlices","InPlaneCoarse","ReducedSNR","RigidJitter"]
METHOD_COLORS = {
    "Crude":         "#e63946",
    "ThickSlices":   "#f4a261",
    "InPlaneCoarse": "#2a9d8f",
    "ReducedSNR":    "#457b9d",
    "RigidJitter":   "#9b5de5",
}
# Variant keys as they appear in the CSV  (v1 = original folder, no suffix)
def vkey(method, n):
    return f"{method}_v{n}"  # e.g. Crude_v1 ... Crude_v5

REF_ORDER = ["test_hires","test_lores"]
ALL_ORDER = REF_ORDER + [vkey(m,n) for m in METHODS for n in range(1,6)]
present   = [v for v in ALL_ORDER if v in df["variant"].values]

METRICS = [
    ("fwhm_mm",        "FWHM blur (mm)\\nhigher = blurrier",   "higher"),
    ("hi_freq_energy", "High-Freq Energy\\nhigher = sharper",   "lower"),
    ("lap_var",        "Laplacian Variance\\nhigher = sharper", "lower"),
]

# ── Figure A: violin matrix ──────────────────────────────────────────────────
n = len(present)
xtick_labels = []
xtick_colors = []
for v in present:
    if v in REF_ORDER:
        lbl   = "Original High Quality" if v=="test_hires" else "Original Low Quality subset"
        color = "#4c8cbf" if v=="test_hires" else "#adb5bd"
    else:
        for m in METHODS:
            if v.startswith(m):
                sev = v.replace(m+"_v","")
                lbl = f"{m}\\nv{sev}"
                color = METHOD_COLORS[m]
                break
    xtick_labels.append(lbl)
    xtick_colors.append(color)

fig, axes = plt.subplots(len(METRICS), 1, figsize=(max(n*0.9, 22), 11),
                         gridspec_kw={"hspace":0.55})
for ax,(col,ylabel,direction) in zip(axes, METRICS):
    pos = np.arange(1, n+1)
    data_seqs = []
    for v in present:
        vals = df[df["variant"]==v][col].dropna().values
        data_seqs.append(vals if len(vals) else np.array([np.nan]))
    parts = ax.violinplot(data_seqs, positions=pos, showmeans=True, showmedians=False, widths=0.75)
    for i,(pc,color) in enumerate(zip(parts["bodies"], xtick_colors)):
        pc.set_facecolor(color); pc.set_alpha(0.55)
    parts["cmeans"].set_linewidth(1.5); parts["cmeans"].set_color("black")
    # shade method groups
    xi = len(REF_ORDER)
    for m in METHODS:
        ax.axvspan(xi+0.5, xi+5.5, alpha=0.07, color=METHOD_COLORS[m], zorder=0)
        ax.text((xi+3), ax.get_ylim()[0] if ax.get_ylim()[0]!=0 else 0,
                m, ha="center", va="top", fontsize=7, color=METHOD_COLORS[m],
                fontweight="bold", clip_on=True)
        xi += 5
    ax.set_xticks(pos)
    ax.set_xticklabels(xtick_labels, fontsize=6.5, rotation=45, ha="right")
    for tick, color in zip(ax.get_xticklabels(), xtick_colors):
        tick.set_color(color)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_title(f"{col}  ({direction} = worse quality)", fontsize=9, pad=3)

fig.suptitle("Image Quality Metrics — All 5 Severity Variants per Method\\n"
             "Each group of 5 = same method, increasing degradation severity (v1→v5)",
             fontsize=12, fontweight="bold", y=1.01)
save_a = IQM_DIR / "iqm_allvariants_violins.png"
fig.savefig(save_a, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved → {save_a.name}")

# ── Figure B: dose-response line plot ───────────────────────────────────────
fig2, axes2 = plt.subplots(1, len(METRICS), figsize=(15, 4.5),
                           gridspec_kw={"wspace":0.35})
for ax,(col,ylabel,direction) in zip(axes2, METRICS):
    for m in METHODS:
        color = METHOD_COLORS[m]
        means, sems, sev_x = [], [], []
        for n_sev in range(1,6):
            v = vkey(m, n_sev)
            vals = df[df["variant"]==v][col].dropna().values
            if len(vals) > 0:
                means.append(vals.mean())
                sems.append(vals.std()/np.sqrt(len(vals)))
                sev_x.append(n_sev)
        if means:
            ax.errorbar(sev_x, means, yerr=[1.96*s for s in sems],
                        color=color, marker="o", markersize=5,
                        linewidth=1.8, capsize=3, label=m)
    # add HiRes reference line
    hi_vals = df[df["variant"]=="test_hires"][col].dropna().values
    if len(hi_vals):
        ax.axhline(hi_vals.mean(), color="#4c8cbf", lw=1.5, ls="--", label="Original High Quality mean")
    ax.set_xlabel("Severity (1=mild → 5=severe)", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(col, fontsize=9, fontweight="bold")
    ax.set_xticks([1,2,3,4,5])
    ax.legend(fontsize=7, framealpha=0.85)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(True, axis="y", alpha=0.2)

fig2.suptitle("IQM Dose-Response: Mean ± 95% CI by Severity Level\\n"
              "Original High Quality reference shown as dashed blue line",
              fontsize=11, fontweight="bold")
save_b = IQM_DIR / "iqm_allvariants_dose_response.png"
fig2.savefig(save_b, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved → {save_b.name}")
''')

# ─────────────────────────────────────────────────────────────────────────────
# CELL 10: Perilesional contrast — severity dose-response per method
# ─────────────────────────────────────────────────────────────────────────────
NEW_CELL_10 = mk_code('''\
# === Perilesional contrast — all 5 severity variants per method ===
# Self-contained — run without executing other cells.
#
# For each method: shows how lesion-vs-perilesional contrast shifts across v1-v5.
# Layout: 5 rows (one per method) × 3 columns (scatter | histogram | dose-response)
# Scatter and histogram compare HiRes vs each variant individually.
# Dose-response panel: mean contrast across v1-v5 for that method.

from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import binary_dilation

BASE_DIR       = Path("/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/A_A_Combined_Data")
HIRES_T1_DIR   = BASE_DIR / "Processed_HiresLowres_Split_Data/test_hires/t1"
HIRES_MASK_DIR = BASE_DIR / "Processed_HiresLowres_Split_Data/test_hires/masks"
DS_BASE        = BASE_DIR / "Downsampled_Data"
DILATION_ITERS = 5
COLOR_HI       = "#3a86ff"

METHODS = {
    "Crude":         {"dirs": ["Crude","Crude_v2","Crude_v3","Crude_v4","Crude_v5"],       "color":"#e63946"},
    "ThickSlices":   {"dirs": ["ThickSlices","ThickSlices_v2","ThickSlices_v3","ThickSlices_v4","ThickSlices_v5"], "color":"#f4a261"},
    "InPlaneCoarse": {"dirs": ["InPlaneCoarse","InPlaneCoarse_v2","InPlaneCoarse_v3","InPlaneCoarse_v4","InPlaneCoarse_v5"], "color":"#2a9d8f"},
    "ReducedSNR":    {"dirs": ["ReducedSNR","ReducedSNR_v2","ReducedSNR_v3","ReducedSNR_v4","ReducedSNR_v5"],     "color":"#457b9d"},
    "RigidJitter":   {"dirs": ["RigidJitter","RigidJitter_v2","RigidJitter_v3","RigidJitter_v4","RigidJitter_v5"],"color":"#9b5de5"},
}
SEV_ALPHAS = [0.85, 0.70, 0.55, 0.40, 0.30]  # v1 darkest → v5 lightest

def peri_contrast(vol, mask):
    peri = binary_dilation(mask, iterations=DILATION_ITERS) & ~mask
    return float(vol[mask].mean()) - (float(vol[peri].mean()) if peri.sum()>0 else 0.)

# ── Collect contrasts ────────────────────────────────────────────────────────
mask_paths = sorted(HIRES_MASK_DIR.glob("*_lesion_mask_MNI_clean.nii.gz"))
hi_vols_cache = {}

# pre-load HiRes contrasts once
hi_contrasts = {}
for msk_p in mask_paths:
    key = msk_p.name.replace("_lesion_mask_MNI_clean.nii.gz","")
    hi_p = HIRES_T1_DIR / f"{key}_T1w_MNI_norm.nii.gz"
    if not hi_p.exists(): continue
    mask = nib.load(str(msk_p)).get_fdata() > 0
    if mask.sum() < 10: continue
    hi_vol = nib.load(str(hi_p)).get_fdata(dtype=np.float32)
    hi_contrasts[key] = (peri_contrast(hi_vol, mask), mask)

# collect per variant
data = {}
for method, cfg in METHODS.items():
    data[method] = []
    for vdir in cfg["dirs"]:
        t1_dir = DS_BASE / vdir / "t1"
        row_hi, row_deg = [], []
        for key,(hi_c, mask) in hi_contrasts.items():
            deg_p = t1_dir / f"{key}_T1w_MNI_norm.nii.gz"
            if not deg_p.exists(): continue
            deg_vol = nib.load(str(deg_p)).get_fdata(dtype=np.float32)
            row_hi.append(hi_c)
            row_deg.append(peri_contrast(deg_vol, mask))
        data[method].append((np.array(row_hi), np.array(row_deg)))
    print(f"{method}: {[len(d[0]) for d in data[method]]} subjects per variant")

# ── Figure ────────────────────────────────────────────────────────────────────
n_methods = len(METHODS)
fig = plt.figure(figsize=(18, n_methods * 3.6))
outer = gridspec.GridSpec(n_methods, 1, figure=fig, hspace=0.14)

for ri, (method, cfg) in enumerate(METHODS.items()):
    color = cfg["color"]
    inner = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[ri],
                                             width_ratios=[1.2, 1.2, 1.0], wspace=0.38)
    # --- Panel 0: scatter (hi vs each variant, overlaid) ---
    ax0 = fig.add_subplot(inner[0])
    all_vals = np.concatenate([v for pair in data[method] for v in pair])
    lim = max(abs(all_vals).max(), 0.01) * 1.18
    ax0.plot([-lim,lim],[-lim,lim],"k:",lw=0.8,label="no change",zorder=1)
    ax0.axhline(0,color="k",lw=0.8,ls="--",zorder=1)
    ax0.axvline(0,color="k",lw=0.8,ls="--",zorder=1)
    for vi,(hi_arr,deg_arr) in enumerate(data[method]):
        if not len(hi_arr): continue
        ax0.scatter(hi_arr, deg_arr, s=8, color=color,
                    alpha=SEV_ALPHAS[vi], zorder=3,
                    label=f"v{vi+1} (μ={deg_arr.mean():+.3f})")
    ax0.set_xlim(-lim,lim); ax0.set_ylim(-lim,lim)
    ax0.set_xlabel("HiRes contrast", fontsize=8)
    ax0.set_ylabel("Variant contrast", fontsize=8)
    ax0.legend(fontsize=6.5, framealpha=0.85, ncol=2)
    ax0.tick_params(labelsize=7)
    ax0.spines[["top","right"]].set_visible(False)
    if ri == 0: ax0.set_title("Scatter: HiRes vs each variant", fontsize=9, fontweight="bold", pad=4)
    ax0.text(-0.12, 0.5, method, transform=ax0.transAxes, fontsize=10, fontweight="bold",
             color=color, va="center", ha="right", rotation=90)

    # --- Panel 1: overlaid histograms (HiRes + v1-v5) ---
    ax1 = fig.add_subplot(inner[1])
    # compute shared bins from HiRes
    hi_ref = data[method][0][0]
    all_v = np.concatenate([hi_ref] + [d[1] for d in data[method]])
    bins = np.linspace(np.nanpercentile(all_v,1)*1.12, np.nanpercentile(all_v,99)*1.12, 38)
    ax1.hist(hi_ref, bins=bins, alpha=0.50, color=COLOR_HI, density=True,
             label=f"HiRes μ={hi_ref.mean():+.3f}", edgecolor="w", lw=0.3)
    for vi,(_, deg_arr) in enumerate(data[method]):
        if not len(deg_arr): continue
        ax1.hist(deg_arr, bins=bins, alpha=SEV_ALPHAS[vi]*0.75, color=color, density=True,
                 label=f"v{vi+1} μ={deg_arr.mean():+.3f}", edgecolor="w", lw=0.3)
    ax1.axvline(0, color="k", lw=1.4, ls="--", zorder=5)
    ax1.axvline(hi_ref.mean(), color=COLOR_HI, lw=1.8, ls="-", zorder=4)
    ax1.set_xlabel("Lesion − perilesional mean", fontsize=8)
    ax1.set_ylabel("Density", fontsize=8)
    ax1.legend(fontsize=6, ncol=2, framealpha=0.85)
    ax1.tick_params(labelsize=7)
    ax1.spines[["top","right"]].set_visible(False)
    if ri == 0: ax1.set_title("Distributions: HiRes + v1–v5", fontsize=9, fontweight="bold", pad=4)

    # --- Panel 2: dose-response (severity x-axis, mean contrast y-axis) ---
    ax2 = fig.add_subplot(inner[2])
    hi_mean = data[method][0][0].mean()
    ax2.axhline(hi_mean, color=COLOR_HI, lw=1.5, ls="--", label=f"HiRes μ={hi_mean:+.3f}", zorder=2)
    sev_x, means, sems = [], [], []
    for vi,(_, deg_arr) in enumerate(data[method]):
        if not len(deg_arr): continue
        sev_x.append(vi+1)
        means.append(deg_arr.mean())
        sems.append(deg_arr.std()/np.sqrt(len(deg_arr)) * 1.96)
    ax2.errorbar(sev_x, means, yerr=sems, color=color, marker="o", markersize=6,
                 linewidth=2, capsize=4, label="Degraded mean ± 95%CI", zorder=3)
    ax2.set_xlabel("Severity (v1→v5)", fontsize=8)
    ax2.set_ylabel("Mean contrast", fontsize=8)
    ax2.set_xticks([1,2,3,4,5])
    ax2.legend(fontsize=7, framealpha=0.85)
    ax2.tick_params(labelsize=7)
    ax2.spines[["top","right"]].set_visible(False)
    if ri == 0: ax2.set_title("Dose-response curve", fontsize=9, fontweight="bold", pad=4)

fig.suptitle(
    "Perilesional Contrast — All 5 Severity Variants per Method  (n subjects per variant shown in legend)\\n"
    "Darker shading = milder degradation (v1), lighter = more severe (v5)",
    fontsize=11, fontweight="bold", y=1.01)
save_path = Path("/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/ARC_ATLAS_Train_v3/Image_Quality_Metrics/iqm_perilesional_allvariants.png")
fig.savefig(save_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved → {save_path.name}")
''')

# ─────────────────────────────────────────────────────────────────────────────
# CELL 14: Statistical test — Wilcoxon + dose-response across all 25 variants
# ─────────────────────────────────────────────────────────────────────────────
NEW_CELL_14 = mk_code('''\
# === Statistical test: perilesional contrast attenuation — all severity variants ===
# Self-contained — run without executing other cells.
#
# For each of the 25 degraded variants (5 methods × 5 severity levels):
#   Δ = HiRes_contrast − Degraded_contrast  (positive = hi has MORE contrast)
#   Wilcoxon signed-rank, H₁: Δ > 0 (one-sided, i.e. degradation attenuates)
#   Effect size: Cohen's d_z = mean(Δ) / SD(Δ)
#
# Figure 1: Forest plot (25 variants, colour-coded by method, grouped by severity)
# Figure 2: Dose-response — mean Δ (effect size) vs severity, one line per method

from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import binary_dilation
from scipy.stats import wilcoxon
import warnings

BASE_DIR       = Path("/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/A_A_Combined_Data")
HIRES_T1_DIR   = BASE_DIR / "Processed_HiresLowres_Split_Data/test_hires/t1"
HIRES_MASK_DIR = BASE_DIR / "Processed_HiresLowres_Split_Data/test_hires/masks"
DS_BASE        = BASE_DIR / "Downsampled_Data"
DILATION_ITERS = 5

METHODS = {
    "Crude":         {"dirs":["Crude","Crude_v2","Crude_v3","Crude_v4","Crude_v5"],         "color":"#e63946"},
    "ThickSlices":   {"dirs":["ThickSlices","ThickSlices_v2","ThickSlices_v3","ThickSlices_v4","ThickSlices_v5"],"color":"#f4a261"},
    "InPlaneCoarse": {"dirs":["InPlaneCoarse","InPlaneCoarse_v2","InPlaneCoarse_v3","InPlaneCoarse_v4","InPlaneCoarse_v5"],"color":"#2a9d8f"},
    "ReducedSNR":    {"dirs":["ReducedSNR","ReducedSNR_v2","ReducedSNR_v3","ReducedSNR_v4","ReducedSNR_v5"],   "color":"#457b9d"},
    "RigidJitter":   {"dirs":["RigidJitter","RigidJitter_v2","RigidJitter_v3","RigidJitter_v4","RigidJitter_v5"],"color":"#9b5de5"},
}

def peri_contrast(vol, mask):
    peri = binary_dilation(mask, iterations=DILATION_ITERS) & ~mask
    return float(vol[mask].mean()) - (float(vol[peri].mean()) if peri.sum()>0 else 0.)

# ── Collect deltas for all 25 variants ────────────────────────────────────────
mask_paths = sorted(HIRES_MASK_DIR.glob("*_lesion_mask_MNI_clean.nii.gz"))

# pre-load HiRes
hi_data = {}
for msk_p in mask_paths:
    key = msk_p.name.replace("_lesion_mask_MNI_clean.nii.gz","")
    hi_p = HIRES_T1_DIR / f"{key}_T1w_MNI_norm.nii.gz"
    if not hi_p.exists(): continue
    mask = nib.load(str(msk_p)).get_fdata() > 0
    if mask.sum() < 10: continue
    hi_vol = nib.load(str(hi_p)).get_fdata(dtype=np.float32)
    hi_data[key] = (peri_contrast(hi_vol, mask), mask)

results = {}   # key: (method, severity_int)
for method, cfg in METHODS.items():
    for vi, vdir in enumerate(cfg["dirs"]):
        sev = vi + 1
        t1_dir = DS_BASE / vdir / "t1"
        deltas = []
        for key, (hi_c, mask) in hi_data.items():
            deg_p = t1_dir / f"{key}_T1w_MNI_norm.nii.gz"
            if not deg_p.exists(): continue
            deg_vol = nib.load(str(deg_p)).get_fdata(dtype=np.float32)
            deltas.append(hi_c - peri_contrast(deg_vol, mask))
        if deltas:
            arr = np.array(deltas)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, p = wilcoxon(arr, alternative="greater") if len(arr)>=10 else (np.nan, 1.0)
            d_z = arr.mean() / (arr.std() + 1e-9)
            stars = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "ns"))
            results[(method,sev)] = {"delta":arr, "mean":arr.mean(), "sem":arr.std()/np.sqrt(len(arr)),
                                     "p":p, "d_z":d_z, "stars":stars, "n":len(arr)}
            print(f"{method:15s} v{sev}  n={len(arr):3d}  mean_Δ={arr.mean():+.4f}  d_z={d_z:+.3f}  {stars}")

# ── Figure 1: Forest plot (25 variants) ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 12))
y = 0
y_ticks, y_labels, y_colors = [], [], []
method_gap = 0.4

for method, cfg in METHODS.items():
    for vi in range(5):
        sev = vi + 1
        key = (method, sev)
        if key not in results: continue
        r = results[key]
        color = cfg["color"]
        alpha = 0.55 + vi * 0.09  # darker = more severe
        ci = r["sem"] * 1.96
        ax.errorbar(r["mean"], y, xerr=ci, fmt="o", color=color,
                    alpha=alpha, markersize=8, capsize=4, linewidth=1.8, zorder=4)
        ax.text(r["mean"] + ci + 0.002, y,
                f"  {r['stars']}  d={r['d_z']:.2f}  (n={r['n']})",
                va="center", ha="left", fontsize=7.5, color=color if r["stars"]!="ns" else "#888")
        y_ticks.append(y)
        y_labels.append(f"{method} v{sev}")
        y_colors.append(color)
        y -= 1
    y -= method_gap  # extra gap between methods

ax.axvline(0, color="black", lw=1.5, ls="--", zorder=2)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels, fontsize=8.5)
for lbl, color in zip(ax.get_yticklabels(), y_colors):
    lbl.set_color(color)
ax.set_xlabel("Mean Δ (HiRes − Degraded) ± 95% CI\\npositive = HiRes has MORE contrast (attenuation)", fontsize=9)
ax.set_title("Contrast Attenuation Forest Plot — All 25 Variants\\n"
             "Wilcoxon signed-rank (H₁: Δ > 0)", fontsize=10, fontweight="bold")
ax.invert_yaxis()
ax.spines[["top","right"]].set_visible(False)
ax.grid(True, axis="x", alpha=0.2)
fig.tight_layout()
save_a = Path("/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/ARC_ATLAS_Train_v3/Image_Quality_Metrics/iqm_attenuation_forest_allvariants.png")
fig.savefig(save_a, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved → {save_a.name}")

# ── Figure 2: Dose-response — mean Δ and d_z vs severity ──────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"wspace":0.38})

for method, cfg in METHODS.items():
    color = cfg["color"]
    sev_x, means, sems, dzs = [], [], [], []
    for sev in range(1,6):
        k = (method, sev)
        if k not in results: continue
        r = results[k]
        sev_x.append(sev); means.append(r["mean"]); sems.append(r["sem"]*1.96); dzs.append(r["d_z"])
    if sev_x:
        axes2[0].errorbar(sev_x, means, yerr=sems, color=color, marker="o",
                          markersize=5, linewidth=2, capsize=3, label=method)
        axes2[1].plot(sev_x, dzs, color=color, marker="o", markersize=5,
                      linewidth=2, label=method)

for ax, ylabel, title in zip(axes2,
    ["Mean Δ (HiRes−Degraded) ± 95% CI", "Cohen's d_z"],
    ["Contrast Attenuation Magnitude", "Effect Size (d_z)"]):
    ax.axhline(0, color="black", lw=1.2, ls="--", zorder=1)
    ax.set_xlabel("Severity level (1=mild → 5=severe)", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xticks([1,2,3,4,5])
    ax.legend(fontsize=8, framealpha=0.85)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(True, axis="y", alpha=0.2)

fig2.suptitle("Dose-Response: Contrast Attenuation vs Severity Level\\n"
              "Each line = one degradation method, across 5 severity levels",
              fontsize=11, fontweight="bold")
save_b = Path("/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/ARC_ATLAS_Train_v3/Image_Quality_Metrics/iqm_attenuation_dose_response.png")
fig2.savefig(save_b, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved → {save_b.name}")

# ── Summary table ─────────────────────────────────────────────────────────────
import pandas as pd
rows = []
for (method,sev), r in results.items():
    rows.append({"method":method,"severity":sev,"n":r["n"],
                 "mean_delta":round(r["mean"],4),"sem":round(r["sem"],4),
                 "cohens_dz":round(r["d_z"],3),"p_one_sided":r["p"],"sig":r["stars"]})
_stat_df = pd.DataFrame(rows).set_index(["method","severity"]).sort_index()
print("\\nFull results table:")
display(_stat_df)
''')

# ─────────────────────────────────────────────────────────────────────────────
# Also update markdown cell 6 (Next Steps) to reference the new CSV filename
# ─────────────────────────────────────────────────────────────────────────────
NEW_CELL_6 = mk_md(
    "## Next Steps\n\n"
    "Once `image_quality_metrics_all_variants_<timestamp>.csv` is generated (cell 4):\n\n"
    "### 1. Merge image-level metrics into evaluation CSVs\n\n"
    "The existing `run_eval_and_viz()` pipeline writes per-case metrics CSVs with manifest info. "
    "Merge the IQM CSV on `key` to add FWHM/HFE/LapVar as predictors for the Mixed Effects Model.\n\n"
    "### 2. Cell 7 — IQM violin + dose-response\n"
    "Violin matrix grouped by method (all 27 variants) + line plot of mean IQM vs severity 1–5.\n\n"
    "### 3. Cell 10 — Perilesional contrast per method × severity\n"
    "For each of the 5 methods: scatter, histogram, and dose-response curve across v1–v5.\n\n"
    "### 4. Cell 12 — Single-subject side-by-side comparison\n"
    "Shows HiRes vs all 5 degradation methods (v1 only) in the IQM-style row layout.\n\n"
    "### 5. Cell 14 — Statistical test: attenuation across all 25 variants\n"
    "Forest plot + dose-response effect size (Cohen's d_z) for all 5 methods × 5 severities.\n"
    "Supports the mixed model: `Dice ~ severity * degradation_type + (1 | case_id)`"
)

# ── Apply patches ─────────────────────────────────────────────────────────────
cells[6]  = NEW_CELL_6
cells[7]  = NEW_CELL_7
cells[10] = NEW_CELL_10
cells[14] = NEW_CELL_14

NB_PATH.write_text(json.dumps(nb, indent=1))
print(f"Patched {NB_PATH.name}")
print(f"  cell 6  → updated Next Steps markdown")
print(f"  cell 7  → IQM violin + dose-response (27 variants grouped by method)")
print(f"  cell 10 → perilesional contrast: scatter + histogram + dose-response per method")
print(f"  cell 14 → forest plot + dose-response for all 25 degraded variants")
