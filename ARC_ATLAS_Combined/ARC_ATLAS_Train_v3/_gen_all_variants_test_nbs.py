#!/usr/bin/env python3
"""
Generate 5 *_all_variants.ipynb test notebooks (one per downsampling method) and
an extended IQM notebook that covers all 5 severity variants per method.

Each _all_variants notebook:
  - Keeps the shared cells (imports+helpers, run_eval_and_viz, pretty_violins func)
  - Replaces the single-dataset run cell with a loop over all 5 variants (v1-v5)
  - Writes one CSV per variant to RUN_DIR/test_eval/ — format expected by MEM notebook
  - Adds a 'severity' (1-5) and 'method' column to every per-case row for the MEM

Output CSV naming:  test_{method_tag}_v{N}_metrics_with_manifest_<timestamp>.csv
MEM FILE_SPEC glob: test_{method_tag}_v{N}_metrics_with_manifest_*.csv
"""
import json
import shutil
from pathlib import Path

NB_DIR  = Path("/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/ARC_ATLAS_Train_v3")
IQM_DIR = NB_DIR / "Image_Quality_Metrics"
DS_BASE = Path("/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/A_A_Combined_Data/Downsampled_Data")
RUN_DIR = Path("/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/ARC_ATLAS_Train_v3/runs/20251110_101813")
MANIFEST = Path("/home/rbielski/ARC/ds004884/derivatives/aggregates/t1w_with_masks/mni_1mm_ants_fixed/_standardized/_combined_manifests_2bins_normal/combined_hires_lores_manifest.csv")
TRAIN_MOD = Path("/home/rbielski/stroke_cleaned/stroke_segmentation_v1.2/stroke_seg_v1.2_train.py")

def mk_code(s): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s}
def mk_md(s):   return {"cell_type":"markdown","metadata":{},"source":s}
def nb_wrap(cells, nb_template):
    return {
        "cells": cells,
        "metadata": nb_template["metadata"],
        "nbformat": nb_template["nbformat"],
        "nbformat_minor": nb_template.get("nbformat_minor", 5),
    }

def load_nb(name):
    with open(NB_DIR / name) as f: return json.load(f)

# ─────────────────────────────────────────────────────────────────────────────
# Method configs: source notebook, dir/tag/prefix for each variant v1-v5
# ─────────────────────────────────────────────────────────────────────────────
METHODS = [
    {
        "name":        "Crude",
        "label":       "Crude Downsampling",
        "source_nb":   "ARC_ATLAS_Test_v3_Crude.ipynb",    # 6 cells (has old cell 0)
        "out_nb":      "ARC_ATLAS_Test_v3_Crude_all_variants.ipynb",
        "cells_offset": 1,   # shared cells start at index 1 (skip old simple-eval cell)
        "dirs":     ["Crude",    "Crude_v2",    "Crude_v3",    "Crude_v4",    "Crude_v5"],
        "tags":     ["crude_v1", "crude_v2",    "crude_v3",    "crude_v4",    "crude_v5"],
        "prefixes": ["test_crude_v1","test_crude_v2","test_crude_v3","test_crude_v4","test_crude_v5"],
        "violin_save": "crude_all_variants_violins.png",
    },
    {
        "name":        "ThickSlices",
        "label":       "Thick Slices",
        "source_nb":   "ARC_ATLAS_Test_v3_ThickSlices.ipynb",  # 5 cells
        "out_nb":      "ARC_ATLAS_Test_v3_ThickSlices_all_variants.ipynb",
        "cells_offset": 0,
        "dirs":     ["ThickSlices",    "ThickSlices_v2",    "ThickSlices_v3",    "ThickSlices_v4",    "ThickSlices_v5"],
        "tags":     ["thickslices_v1", "thickslices_v2",    "thickslices_v3",    "thickslices_v4",    "thickslices_v5"],
        "prefixes": ["test_thickslices_v1","test_thickslices_v2","test_thickslices_v3","test_thickslices_v4","test_thickslices_v5"],
        "violin_save": "thickslices_all_variants_violins.png",
    },
    {
        "name":        "InPlaneCoarse",
        "label":       "In-Plane Coarsening",
        "source_nb":   "ARC_ATLAS_Test_v3_InPlaneCoarse.ipynb",
        "out_nb":      "ARC_ATLAS_Test_v3_InPlaneCoarse_all_variants.ipynb",
        "cells_offset": 0,
        "dirs":     ["InPlaneCoarse",    "InPlaneCoarse_v2",    "InPlaneCoarse_v3",    "InPlaneCoarse_v4",    "InPlaneCoarse_v5"],
        "tags":     ["inplane_v1",        "inplane_v2",          "inplane_v3",          "inplane_v4",          "inplane_v5"],
        "prefixes": ["test_inplane_v1",   "test_inplane_v2",     "test_inplane_v3",     "test_inplane_v4",     "test_inplane_v5"],
        "violin_save": "inplane_all_variants_violins.png",
    },
    {
        "name":        "ReducedSNR",
        "label":       "Reduced SNR",
        "source_nb":   "ARC_ATLAS_Test_v3_ReducedSNR.ipynb",
        "out_nb":      "ARC_ATLAS_Test_v3_ReducedSNR_all_variants.ipynb",
        "cells_offset": 0,
        "dirs":     ["ReducedSNR",    "ReducedSNR_v2",    "ReducedSNR_v3",    "ReducedSNR_v4",    "ReducedSNR_v5"],
        "tags":     ["reducedsnr_v1", "reducedsnr_v2",    "reducedsnr_v3",    "reducedsnr_v4",    "reducedsnr_v5"],
        "prefixes": ["test_reducedsnr_v1","test_reducedsnr_v2","test_reducedsnr_v3","test_reducedsnr_v4","test_reducedsnr_v5"],
        "violin_save": "reducedsnr_all_variants_violins.png",
    },
    {
        "name":        "RigidJitter",
        "label":       "Rigid Jitter",
        "source_nb":   "ARC_ATLAS_Test_v3_RigidJitter.ipynb",
        "out_nb":      "ARC_ATLAS_Test_v3_RigidJitter_all_variants.ipynb",
        "cells_offset": 0,
        "dirs":     ["RigidJitter",    "RigidJitter_v2",    "RigidJitter_v3",    "RigidJitter_v4",    "RigidJitter_v5"],
        "tags":     ["rigidjitter_v1", "rigidjitter_v2",    "rigidjitter_v3",    "rigidjitter_v4",    "rigidjitter_v5"],
        "prefixes": ["test_rigidjitter_v1","test_rigidjitter_v2","test_rigidjitter_v3","test_rigidjitter_v4","test_rigidjitter_v5"],
        "violin_save": "rigidjitter_all_variants_violins.png",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Build the multi-variant RUN cell (cell 2) — replaces the single-dataset cell
# ─────────────────────────────────────────────────────────────────────────────
def make_run_cell(method):
    name    = method["name"]
    label   = method["label"]
    dirs    = method["dirs"]
    tags    = method["tags"]
    prefixes= method["prefixes"]

    # Build VARIANTS list literal
    variants_lines = ["VARIANTS = [\n"]
    for i, (d, t, p) in enumerate(zip(dirs, tags, prefixes)):
        sev = i + 1
        variants_lines.append(
            f'    {{"tag": {repr(t)}, "prefix": {repr(p)}, '
            f'"dir": DS_BASE / {repr(d)}, "severity": {sev}}},\n'
        )
    variants_lines.append("]\n")
    variants_block = "".join(variants_lines)

    code = f"""\
# ===== All-variants evaluation: {label} (v1–v5) =====
# Loops over all 5 severity variants, runs full evaluation for each, and
# writes one CSV per variant to RUN_DIR/test_eval/.
# Each row gets 'severity' (1–5) and 'method' columns for the Mixed Effects Model.
# Skip variants whose images have not been generated yet.
#
# MEM FILE_SPEC glob for each variant:
{chr(10).join(f'#   {repr(p)}_metrics_with_manifest_*.csv  (severity {i+1})' for i, p in enumerate(prefixes))}

from pathlib import Path
import csv, json
import numpy as np
from typing import Dict, List, Any

# ---- helpers (NaN-safe) ----
def _dropna(a):
    a = np.asarray(a, float); return a[~np.isnan(a)]

def summarize_vector(x) -> Dict[str, float]:
    v = _dropna(np.array(x, float))
    if v.size == 0:
        return dict(n=0, mean=np.nan, median=np.nan, std=np.nan,
                    p25=np.nan, p75=np.nan, min=np.nan, max=np.nan, nan_count=len(x))
    return dict(n=int(v.size), mean=float(np.mean(v)), median=float(np.median(v)),
                std=float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
                p25=float(np.percentile(v, 25)), p75=float(np.percentile(v, 75)),
                min=float(np.min(v)), max=float(np.max(v)),
                nan_count=int(len(x) - v.size))

def summarize_dataset(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    hard_key = next((k for k in rows[0].keys() if k.startswith("hard_dice_at_")), "hard_dice_at_0.50")
    return {{
        "soft": summarize_vector([r["soft_dice"]  for r in rows]),
        "hard": summarize_vector([r[hard_key]     for r in rows]),
        "prec": summarize_vector([r["precision"]  for r in rows]),
        "rec":  summarize_vector([r["recall"]     for r in rows]),
        "hd":   summarize_vector([r["hd"]         for r in rows]),
        "hd95": summarize_vector([r["hd95"]       for r in rows]),
        "assd": summarize_vector([r["assd"]       for r in rows]),
    }}

def write_summary_csv(summary_table: List[Dict[str, Any]], out_path: Path):
    if not summary_table: return
    keys = sorted({{k for row in summary_table for k in row.keys()}})
    with open(out_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader(); wr.writerows(summary_table)
    print("Wrote summary CSV ->", out_path)

# ---- Paths ----
RUN_DIR   = Path({repr(str(RUN_DIR))})
TRAIN_MOD = Path({repr(str(TRAIN_MOD))})
MANIFEST  = Path({repr(str(MANIFEST))})
DS_BASE   = Path({repr(str(DS_BASE))})

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_XLA_FLAGS"]             = ""
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# ---- Variant configs ----
{variants_block}
# ---- Run evaluation for each variant ----
datasets: Dict[str, Dict[str, Any]] = {{}}

for v in VARIANTS:
    t1_dir  = v["dir"] / "t1"
    msk_dir = v["dir"] / "masks"
    if not t1_dir.exists():
        print(f"[SKIP] {{v['tag']}}: {{t1_dir}} not found — run the downsampling cells first")
        continue
    n_imgs = len(list(t1_dir.glob("*.nii.gz")))
    if n_imgs == 0:
        print(f"[SKIP] {{v['tag']}}: directory exists but contains no .nii.gz files")
        continue
    print(f"\\n[{{v['tag']}}] severity={{v['severity']}}  n={{n_imgs}} images")
    rows, summary, paths = run_eval_and_viz(
        RUN_DIR=RUN_DIR, TEST_DIR=v["dir"], TRAIN_MOD_PATH=TRAIN_MOD,
        T1_DIR=t1_dir, MSK_DIR=msk_dir, MANIFEST_CSV=MANIFEST,
        TH=0.50, save_prefix=v["prefix"], dataset_tag=v["tag"]
    )
    # annotate rows for MEM (severity + method columns)
    for row in rows:
        row["severity"]   = v["severity"]
        row["method"]     = {repr(name)}
        row["variant_id"] = v["tag"]

    datasets[v["tag"]] = {{
        "rows":         rows,
        "summary":      summarize_dataset(rows),
        "source_paths": [paths["per_case_csv"]],
        "severity":     v["severity"],
    }}

    flat = {{"dataset_tag": v["tag"], "severity": v["severity"], "method": {repr(name)}}}
    for metric, stats in datasets[v["tag"]]["summary"].items():
        for k, val in stats.items():
            flat[f"{{metric}}_{{k}}"] = val
    out_summary = Path(paths["per_case_csv"]).with_name(f"{{v['prefix']}}_summary_stats.csv")
    write_summary_csv([flat], out_summary)

print("\\nVariants evaluated:", list(datasets.keys()))
if datasets:
    first = next(iter(datasets.values()))
    print(f"Example row keys: {{list(first['rows'][0].keys())}}")
"""
    return mk_code(code)


# ─────────────────────────────────────────────────────────────────────────────
# Build the violin-call cell (cell 4)
# ─────────────────────────────────────────────────────────────────────────────
def make_violin_cell(method):
    save_name = method["violin_save"]
    label     = method["label"]
    code = f"""\
# ===== Severity gradient violin plots: {label} =====
from pathlib import Path

if not datasets:
    print("No datasets to plot — run Cell 2 first (and generate the variant images).")
else:
    pretty_violins_from_datasets(
        datasets,
        title={repr(f"{label} — Severity Variants v1–v5")},
        distance_units="mm",
        show_legend=True,
        save_path=Path({repr(str(RUN_DIR / "test_eval" / save_name))})
    )
"""
    return mk_code(code)


# ─────────────────────────────────────────────────────────────────────────────
# Generate all 5 test notebooks
# ─────────────────────────────────────────────────────────────────────────────
for method in METHODS:
    nb_src = load_nb(method["source_nb"])
    cells  = nb_src["cells"]
    off    = method["cells_offset"]

    # shared cells: imports+helpers, run_eval_and_viz, pretty_violins func
    imports_cell = cells[off + 0]   # imports + helpers
    runfunc_cell = cells[off + 1]   # run_eval_and_viz definition
    violins_cell = cells[off + 3]   # pretty_violins_from_datasets definition

    new_cells = [
        mk_md(
            f"# {method['label']} — All Severity Variants (v1–v5)\n\n"
            "Evaluates the trained v3 model on **all 5 severity variants** of this degradation method.\n"
            "One CSV per variant is written to `RUN_DIR/test_eval/` for the Mixed Effects Model.\n\n"
            "| Cell | Description |\n"
            "|------|-------------|\n"
            "| 1 | Imports, helpers, metric functions |\n"
            "| 2 | `run_eval_and_viz()` — full evaluation pipeline |\n"
            "| 3 | **Run all variants** — loops v1–v5, writes CSVs, builds `datasets` dict |\n"
            "| 4 | `pretty_violins_from_datasets()` — violin plot function |\n"
            "| 5 | Plot severity gradient violins |\n\n"
            "> Run cells in order. Cell 3 skips any variant whose images have not been generated yet."
        ),
        imports_cell,
        runfunc_cell,
        make_run_cell(method),
        violins_cell,
        make_violin_cell(method),
    ]

    out_path = NB_DIR / method["out_nb"]
    with open(out_path, "w") as f:
        json.dump(nb_wrap(new_cells, nb_src), f, indent=1)
    print(f"Wrote {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Generate extended IQM notebook
# ─────────────────────────────────────────────────────────────────────────────
iqm_src_path = IQM_DIR / "ARC_ATLAS_Image_Quality_Metrics.ipynb"
iqm_out_path = IQM_DIR / "ARC_ATLAS_Image_Quality_Metrics_all_variants.ipynb"

with open(iqm_src_path) as f:
    iqm_nb = json.load(f)

# Build new DATASETS list including all variants
split_base_str = str(Path("/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/A_A_Combined_Data/Processed_HiresLowres_Split_Data"))
ds_base_str    = str(DS_BASE)

new_datasets_block = '''\
from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd
from numpy.fft import fftn, fftshift
from scipy.ndimage import laplace
from joblib import Parallel, delayed
import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = Path("/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined")
SPLIT_BASE = BASE / "A_A_Combined_Data/Processed_HiresLowres_Split_Data"
DOWN_BASE  = BASE / "A_A_Combined_Data/Downsampled_Data"
OUT_DIR    = Path("/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/ARC_ATLAS_Train_v3/Image_Quality_Metrics")

# ── Dataset configs — all variants (v1-v5) per method ──────────────────────────
DATASETS = [
    # Reference sets
    {"variant": "test_hires",         "t1_dir": SPLIT_BASE / "test_hires/t1"},
    {"variant": "test_lores",         "t1_dir": SPLIT_BASE / "test_lores/t1"},
    # Crude variants
    {"variant": "Crude_v1",           "t1_dir": DOWN_BASE / "Crude/t1"},
    {"variant": "Crude_v2",           "t1_dir": DOWN_BASE / "Crude_v2/t1"},
    {"variant": "Crude_v3",           "t1_dir": DOWN_BASE / "Crude_v3/t1"},
    {"variant": "Crude_v4",           "t1_dir": DOWN_BASE / "Crude_v4/t1"},
    {"variant": "Crude_v5",           "t1_dir": DOWN_BASE / "Crude_v5/t1"},
    # ThickSlices variants
    {"variant": "ThickSlices_v1",     "t1_dir": DOWN_BASE / "ThickSlices/t1"},
    {"variant": "ThickSlices_v2",     "t1_dir": DOWN_BASE / "ThickSlices_v2/t1"},
    {"variant": "ThickSlices_v3",     "t1_dir": DOWN_BASE / "ThickSlices_v3/t1"},
    {"variant": "ThickSlices_v4",     "t1_dir": DOWN_BASE / "ThickSlices_v4/t1"},
    {"variant": "ThickSlices_v5",     "t1_dir": DOWN_BASE / "ThickSlices_v5/t1"},
    # InPlaneCoarse variants
    {"variant": "InPlaneCoarse_v1",   "t1_dir": DOWN_BASE / "InPlaneCoarse/t1"},
    {"variant": "InPlaneCoarse_v2",   "t1_dir": DOWN_BASE / "InPlaneCoarse_v2/t1"},
    {"variant": "InPlaneCoarse_v3",   "t1_dir": DOWN_BASE / "InPlaneCoarse_v3/t1"},
    {"variant": "InPlaneCoarse_v4",   "t1_dir": DOWN_BASE / "InPlaneCoarse_v4/t1"},
    {"variant": "InPlaneCoarse_v5",   "t1_dir": DOWN_BASE / "InPlaneCoarse_v5/t1"},
    # ReducedSNR variants
    {"variant": "ReducedSNR_v1",      "t1_dir": DOWN_BASE / "ReducedSNR/t1"},
    {"variant": "ReducedSNR_v2",      "t1_dir": DOWN_BASE / "ReducedSNR_v2/t1"},
    {"variant": "ReducedSNR_v3",      "t1_dir": DOWN_BASE / "ReducedSNR_v3/t1"},
    {"variant": "ReducedSNR_v4",      "t1_dir": DOWN_BASE / "ReducedSNR_v4/t1"},
    {"variant": "ReducedSNR_v5",      "t1_dir": DOWN_BASE / "ReducedSNR_v5/t1"},
    # RigidJitter variants
    {"variant": "RigidJitter_v1",     "t1_dir": DOWN_BASE / "RigidJitter/t1"},
    {"variant": "RigidJitter_v2",     "t1_dir": DOWN_BASE / "RigidJitter_v2/t1"},
    {"variant": "RigidJitter_v3",     "t1_dir": DOWN_BASE / "RigidJitter_v3/t1"},
    {"variant": "RigidJitter_v4",     "t1_dir": DOWN_BASE / "RigidJitter_v4/t1"},
    {"variant": "RigidJitter_v5",     "t1_dir": DOWN_BASE / "RigidJitter_v5/t1"},
]

N_JOBS = 12

# Verify all paths (skip missing variant dirs — they may not be generated yet)
print(f"{'Variant':<22} {'Status':<9} {'N T1s':>6}")
print("-" * 42)
for ds in DATASETS:
    p = ds["t1_dir"]
    status = "OK" if p.exists() else "MISSING"
    n = len(list(p.glob("*.nii.gz"))) if p.exists() else 0
    print(f"  {ds['variant']:<20}  [{status}]  {n:>4d}")
'''

# Replace the setup cell (cell 1) with the extended version
iqm_cells = list(iqm_nb["cells"])
iqm_cells[1] = mk_code(new_datasets_block)

# Insert a markdown header after cell 0 explaining the extended version
header_cell = mk_md(
    "# Image Quality Metrics — All Severity Variants\n\n"
    "Extended version of `ARC_ATLAS_Image_Quality_Metrics.ipynb` that covers\n"
    "**all 5 severity variants** (v1–v5) for each of the 5 downsampling methods\n"
    "(25 degraded datasets + 2 reference sets = 27 total).\n\n"
    "| Cells | Purpose |\n"
    "|-------|--------|\n"
    "| 1 | Paths + DATASETS config (27 variants) |\n"
    "| 2 | IQM function definitions (FWHM, HFE, LapVar) |\n"
    "| 3 | Run computation → `image_quality_metrics_all_variants_<ts>.csv` |\n"
    "| 4+ | Summary stats, visualisations, perilesional analysis |\n\n"
    "> Missing variant directories are silently skipped. Run the downsampling cells first."
)
iqm_cells.insert(0, header_cell)

with open(iqm_out_path, "w") as f:
    json.dump(nb_wrap(iqm_cells, iqm_nb), f, indent=1)
print(f"Wrote {iqm_out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Print MEM FILE_SPEC entries for all new variants (paste into MEM notebook)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("Paste these entries into ARC_ATLAS_Test_v3_Mixed_Effect_Model.ipynb")
print("VARIANT_GLOBS list (§3 cell), one entry per variant:")
print("="*72)
for method in METHODS:
    name = method["name"].lower().replace("inplanecoarse","inplane").replace("reducedsnr","reducedsnr").replace("rigidjitter","rigidjitter")
    for i,(tag,prefix) in enumerate(zip(method["tags"], method["prefixes"])):
        sev = i + 1
        print(f'    {{"variant": {repr(tag):<25}, "cohort": "hires", "role": "degraded", "severity": {sev}, "glob": {repr(prefix + "_metrics_with_manifest_*.csv")}}},')
    print()
