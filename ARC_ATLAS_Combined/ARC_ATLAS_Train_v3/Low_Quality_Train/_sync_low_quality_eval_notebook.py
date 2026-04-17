#!/usr/bin/env python3
from __future__ import annotations

import json
import textwrap
from pathlib import Path


NOTEBOOK = Path(
    "/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/"
    "ARC_ATLAS_Train_v3/Low_Quality_Train/ARC_ATLAS_Test_v3_HiRes_Low_Train.ipynb"
)

LOW_QUALITY_ROOT = Path(
    "/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/ARC_ATLAS_Train_v3/Low_Quality_Train"
)
LOW_QUALITY_RUNS = LOW_QUALITY_ROOT / "runs"
LOW_QUALITY_TEST_DIR = Path(
    "/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/"
    "A_A_Combined_Data/Processed_LowQualityTrain_Split_Data/test_hires"
)
LOW_QUALITY_MANIFEST = Path(
    "/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/"
    "A_A_Combined_Data/Processed_LowQualityTrain_Split_Data/test_hires.csv"
)
TRAIN_MOD = Path(
    "/home/rbielski/stroke_cleaned/stroke_segmentation_v1.2/stroke_seg_v1.2_train.py"
)


CELL0_PATH_BLOCK = textwrap.dedent(
    f"""\
    # --------- Low-quality run + held-out HiRes set ----------
    RUN_ROOT  = Path("{LOW_QUALITY_ROOT}")
    RUNS_DIR  = RUN_ROOT / "runs"
    RUN_NAME  = globals().get("RUN_NAME", "")  # set to a specific run id; leave blank for latest
    TEST_DIR  = Path("{LOW_QUALITY_TEST_DIR}")
    T1_DIR    = TEST_DIR / "t1"      # use these to avoid duplicate pairing
    MSK_DIR   = TEST_DIR / "masks"
    TRAIN_MOD = Path("{TRAIN_MOD}")

    def _find_low_quality_run_dir(runs_dir: Path, run_name: str = "") -> Path:
        existing = globals().get("RUN_DIR")
        if isinstance(existing, (str, Path)):
            existing = Path(existing)
            if (existing / "models").exists() and (existing / "callbacks").exists():
                return existing

        if run_name:
            run_dir = runs_dir / run_name
        else:
            candidates = sorted(p for p in runs_dir.iterdir() if p.is_dir())
            if not candidates:
                raise FileNotFoundError(f"No run directories found in {{runs_dir}}")
            run_dir = candidates[-1]

        if not run_dir.exists():
            raise FileNotFoundError(f"Run dir not found: {{run_dir}}")
        return run_dir

    RUN_DIR = _find_low_quality_run_dir(RUNS_DIR, RUN_NAME)
    print("Using run:", RUN_DIR)
    # -----------------------------------------------
    """
)


CELL1_CONFIG_BLOCK = textwrap.dedent(
    f"""\
    # CONFIG — low-quality-trained run on the same held-out HiRes test set:
    import pathlib

    RUN_ROOT = pathlib.Path("{LOW_QUALITY_ROOT}")
    RUNS_DIR = RUN_ROOT / "runs"
    RUN_NAME = globals().get("RUN_NAME", "")  # set to a specific run id; leave blank for latest
    TEST_DIR = pathlib.Path("{LOW_QUALITY_TEST_DIR}")

    def _find_low_quality_run_dir(runs_dir, run_name=""):
        existing = globals().get("RUN_DIR")
        if isinstance(existing, (str, pathlib.Path)):
            existing = pathlib.Path(existing)
            if (existing / "models").exists() and (existing / "callbacks").exists():
                return existing

        if run_name:
            run_dir = runs_dir / run_name
        else:
            candidates = sorted(p for p in runs_dir.iterdir() if p.is_dir())
            if not candidates:
                raise FileNotFoundError(f"No run directories found in {{runs_dir}}")
            run_dir = candidates[-1]

        if not run_dir.exists():
            raise FileNotFoundError(f"Run dir not found: {{run_dir}}")
        return run_dir

    RUN_DIR = str(_find_low_quality_run_dir(RUNS_DIR, RUN_NAME))
    TEST_DIR = str(TEST_DIR)
    print("Using run:", RUN_DIR)

    # Optional: if you already saved NIfTI preds, point here; else leave "" to compute on-the-fly.
    PREDS_DIR = ""   # e.g. "/.../test_preds_t050"
    """
)


CELL5_RUN_BLOCK = textwrap.dedent(
    f"""\
    # -------- RUN THIS BLOCK (edit RUN_NAME only if you want a specific run) --------
    LOW_QUALITY_ROOT = Path("{LOW_QUALITY_ROOT}")
    LOW_QUALITY_RUNS = LOW_QUALITY_ROOT / "runs"
    LOW_QUALITY_TEST_DIR = Path("{LOW_QUALITY_TEST_DIR}")
    LOW_QUALITY_MANIFEST = Path("{LOW_QUALITY_MANIFEST}")
    TRAIN_MOD = Path("{TRAIN_MOD}")
    RUN_NAME = globals().get("RUN_NAME", "")  # set to a specific run id; leave blank for latest

    def _find_low_quality_run_dir(run_name: str = "") -> Path:
        existing = globals().get("RUN_DIR")
        if isinstance(existing, (str, Path)):
            existing = Path(existing)
            if (existing / "models").exists() and (existing / "callbacks").exists():
                return existing

        if run_name:
            run_dir = LOW_QUALITY_RUNS / run_name
        else:
            candidates = sorted(p for p in LOW_QUALITY_RUNS.iterdir() if p.is_dir())
            if not candidates:
                raise FileNotFoundError(f"No run directories found in {{LOW_QUALITY_RUNS}}")
            run_dir = candidates[-1]

        if not run_dir.exists():
            raise FileNotFoundError(f"Run dir not found: {{run_dir}}")
        return run_dir

    RUN_DIR   = _find_low_quality_run_dir(RUN_NAME)
    TEST_DIR  = LOW_QUALITY_TEST_DIR
    T1_DIR    = TEST_DIR / "t1"
    MSK_DIR   = TEST_DIR / "masks"
    MANIFEST  = LOW_QUALITY_MANIFEST

    DATASET_TAG = "hires"
    SAVE_PREFIX = "test_hires_low_quality_train"

    print("Using run:", RUN_DIR)
    """
)


CELL6_SOURCE = textwrap.dedent(
    """\
    # === Pretty grouped violins (final tweaks) ===
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from pathlib import Path
    from typing import Dict, Any, List, Optional, Set

    PALETTE       = ["#4C78A8"]
    POINT_ALPHA   = 0.55
    POINT_SIZE    = 10
    VIOLIN_ALPHA  = 0.18
    EDGE_ALPHA    = 0.35
    ROBUST_PERC   = (5, 95)
    JITTER_WIDTH  = 0.08
    STAT_FONTSIZE = 7
    SEED          = 7
    _rng = np.random.default_rng(SEED)

    def _dropna(a):
        a = np.asarray(a, float)
        return a[~np.isnan(a)]

    def _robust_limits_linear(arr, p=(5, 95), min_span=1e-6):
        arr = _dropna(arr)
        if arr.size == 0:
            return (0.0, 1.0)
        lo, hi = np.percentile(arr, p[0]), np.percentile(arr, p[1])
        if hi - lo < min_span:
            hi = lo + min_span
        return float(lo), float(hi)

    def _beeswarm_x(center: float, n: int, spread: float) -> np.ndarray:
        if n == 0:
            return np.array([])
        return center + _rng.normal(0.0, spread, size=n)

    def _get_val(row: Dict[str, Any], key: str):
        if key == "soft":
            return row.get("soft", row.get("soft_dice"))
        if key == "hard":
            for k in row.keys():
                if k.startswith("hard_dice"):
                    return row[k]
            return row.get("hard")
        if key == "prec":
            return row.get("prec", row.get("precision"))
        if key == "rec":
            return row.get("rec", row.get("recall"))
        return row.get(key)

    def _fmt_stat(val: float, is_distance: bool) -> str:
        return f"{val:.2f}" if is_distance else f"{val:.3f}"

    def pretty_violins_from_datasets(
        datasets: Dict[str, Dict[str, Any]],
        title: str = "Higher Resolution Images Test Set | Low-Quality Trained Model",
        legend_label: str = "Individual Image Segmentation",
        distance_units: str = "mm",
        save_path=None,
        show_legend: bool = False,
    ):
        if not datasets:
            print("No datasets to plot.")
            return None

        tags = sorted(datasets.keys())

        def metric_arrays(std_key: str) -> List[np.ndarray]:
            return [_dropna([_get_val(r, std_key) for r in datasets[tag]["rows"]]) for tag in tags]

        soft, hard, prec, rec = (metric_arrays(k) for k in ["soft", "hard", "prec", "rec"])
        hd, hd95, assd = (metric_arrays(k) for k in ["hd", "hd95", "assd"])

        plt.style.use("default")
        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 1, height_ratios=[3, 2], hspace=0.40)
        ax_top = fig.add_subplot(gs[0])
        ax_bot = fig.add_subplot(gs[1])

        def draw_grouped(
            ax,
            data_seq: List[List[np.ndarray]],
            base_labels: List[str],
            ylabel=None,
            panel_title=None,
            is_distance: bool = False,
            annotate_groups: Optional[Set[int]] = None,
        ):
            groups = len(data_seq)
            nd = len(tags)
            base_x = np.arange(1, groups + 1, dtype=float)
            step = 0.9 / (nd + 1)

            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

            all_vals = []
            annotations = []
            for di, tag in enumerate(tags):
                color = PALETTE[di % len(PALETTE)]
                offset = (di - (nd - 1) / 2) * step
                for gi in range(groups):
                    vals = data_seq[gi][di]
                    all_vals.append(vals)
                    pos = base_x[gi] + offset
                    parts = ax.violinplot(
                        [vals if vals.size else np.array([np.nan])],
                        positions=[pos],
                        widths=step * 0.95,
                        showmeans=True,
                        showmedians=True,
                    )
                    for b in parts.get("bodies", []):
                        b.set_facecolor(color)
                        b.set_edgecolor((0, 0, 0, EDGE_ALPHA))
                        b.set_alpha(VIOLIN_ALPHA)
                    if "cmeans" in parts:
                        parts["cmeans"].set_linewidth(1.6)
                        parts["cmeans"].set_color((0, 0, 0, 0.85))
                    if "cmedians" in parts:
                        parts["cmedians"].set_linewidth(1.6)
                        parts["cmedians"].set_color((0, 0, 0, 0.85))
                    if vals.size:
                        xs = _beeswarm_x(pos, len(vals), JITTER_WIDTH)
                        ax.scatter(xs, vals, s=POINT_SIZE, alpha=POINT_ALPHA, color=color, edgecolor="none")
                        annotations.append(
                            {
                                "group_idx": gi,
                                "pos": pos,
                                "mean": float(np.mean(vals)),
                                "median": float(np.median(vals)),
                                "color": color,
                            }
                        )

            ax.set_xticks(base_x)
            ax.set_xticklabels(base_labels)
            if ylabel:
                ax.set_ylabel(ylabel)
            if panel_title:
                ax.set_title(panel_title, pad=18)

            flat = np.concatenate([v for v in all_vals if v.size]) if any(v.size for v in all_vals) else np.array([])
            if flat.size:
                lo, hi = _robust_limits_linear(flat, p=ROBUST_PERC)
                if not is_distance:
                    lo = max(0.0, lo)
                    hi = min(1.0, max(0.9, hi))
                ax.set_ylim(lo, hi)
            ax.grid(True, axis="y", alpha=0.25)

            if annotate_groups:
                for ann in annotations:
                    if ann["group_idx"] not in annotate_groups:
                        continue
                    stat_text = (
                        f"mean={_fmt_stat(ann['mean'], is_distance)}\\n"
                        f"med={_fmt_stat(ann['median'], is_distance)}"
                    )
                    ax.text(
                        ann["pos"],
                        0.98,
                        stat_text,
                        transform=ax.get_xaxis_transform(),
                        ha="center",
                        va="top",
                        fontsize=STAT_FONTSIZE,
                        color=ann["color"],
                        bbox=dict(boxstyle="round,pad=0.2", fc=(1, 1, 1, 0.85), ec=(0, 0, 0, 0.18)),
                    )

            if show_legend:
                ax.plot([], [], marker="o", ls="", color=PALETTE[0], label=legend_label)
                ax.legend(loc="upper left", bbox_to_anchor=(1.23, 1.03), frameon=False)

        draw_grouped(
            ax_top,
            data_seq=[soft, hard, prec, rec],
            base_labels=["Soft Dice", "Hard Dice", "Precision", "Recall"],
            ylabel="Value",
            panel_title="Per-case distributions (Dice / Precision / Recall)",
            is_distance=False,
            annotate_groups={0, 1},
        )

        draw_grouped(
            ax_bot,
            data_seq=[hd, hd95, assd],
            base_labels=["HD", "HD95", "ASSD"],
            ylabel=f"Distance ({distance_units})",
            panel_title="Surface distance metrics (per-case)",
            is_distance=True,
            annotate_groups=None,
        )

        fig.suptitle(title, y=0.99, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.98])

        if save_path is None:
            run_dir = globals().get("RUN_DIR")
            if isinstance(run_dir, (str, Path)):
                run_dir = Path(run_dir)
                save_path = run_dir / "test_eval" / "figs" / "pretty_violins_linear_final.png"
            else:
                any_row = next(iter(datasets[tags[0]]["rows"]), None)
                default_dir = Path(any_row["img_path"]).parent if any_row and any_row.get("img_path") else Path(".")
                save_path = default_dir / "pretty_violins_linear_final.png"

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        print("Saved figure ->", save_path)

        plt.show()
        return fig
    """
)


CELL7_SOURCE = textwrap.dedent(
    """\
    from pathlib import Path

    _run_dir = Path(RUN_DIR) if "RUN_DIR" in globals() else None
    _violin_save = (_run_dir / "test_eval" / "figs" / "pretty_violins_linear_final.png") if _run_dir else None

    pretty_violins_from_datasets(
        datasets,
        title="Higher Resolution Images Test Set | Low-Quality Trained Model",
        distance_units="mm",
        save_path=_violin_save,
    )
    """
)


CELL9_CONFIG_BLOCK = textwrap.dedent(
    f"""\
    # ---------- CONFIG ----------
    LOW_QUALITY_ROOT = Path("{LOW_QUALITY_ROOT}")
    LOW_QUALITY_RUNS = LOW_QUALITY_ROOT / "runs"
    RUN_NAME = globals().get("RUN_NAME", "")  # set to a specific run id; leave blank for latest

    def _find_low_quality_run_dir(run_name: str = "") -> Path:
        existing = globals().get("RUN_DIR")
        if isinstance(existing, (str, Path)):
            existing = Path(existing)
            if (existing / "models").exists() and (existing / "callbacks").exists():
                return existing

        if run_name:
            run_dir = LOW_QUALITY_RUNS / run_name
        else:
            candidates = sorted(p for p in LOW_QUALITY_RUNS.iterdir() if p.is_dir())
            if not candidates:
                raise FileNotFoundError(f"No run directories found in {{LOW_QUALITY_RUNS}}")
            run_dir = candidates[-1]

        if not run_dir.exists():
            raise FileNotFoundError(f"Run dir not found: {{run_dir}}")
        return run_dir

    RUN_DIR    = _find_low_quality_run_dir(RUN_NAME)
    TEST_EVAL  = RUN_DIR / "test_eval"
    # Use the plain per-case eval if you have it; otherwise this also supports the merged one:
    # Auto-find the latest plain eval CSV (test_metrics_*.csv, not _with_manifest_)
    _eval_csvs = sorted([p for p in TEST_EVAL.glob("test_metrics_*.csv")
                         if "_with_manifest_" not in p.name and "_summary_" not in p.name])
    assert _eval_csvs, f"No test_metrics_*.csv found in {{TEST_EVAL}}"
    EVAL_CSV   = _eval_csvs[-1]  # most recent
    print("Using run:", RUN_DIR)
    print("Using eval CSV:", EVAL_CSV.name)
    # If you only have the merged file, use that instead:

    MANIFEST   = Path("{LOW_QUALITY_MANIFEST}")
    # -------------------------------------------
    """
)


CELL10_BLOCK = textwrap.dedent(
    f"""\
    RUN_NAME = globals().get("RUN_NAME", "")  # set to a specific run id; leave blank for latest
    LOW_QUALITY_ROOT = Path("{LOW_QUALITY_ROOT}")
    LOW_QUALITY_RUNS = LOW_QUALITY_ROOT / "runs"

    def _find_low_quality_run_dir(run_name: str = "") -> Path:
        existing = globals().get("RUN_DIR")
        if isinstance(existing, (str, Path)):
            existing = Path(existing)
            if (existing / "models").exists() and (existing / "callbacks").exists():
                return existing

        if run_name:
            run_dir = LOW_QUALITY_RUNS / run_name
        else:
            candidates = sorted(p for p in LOW_QUALITY_RUNS.iterdir() if p.is_dir())
            if not candidates:
                raise FileNotFoundError(f"No run directories found in {{LOW_QUALITY_RUNS}}")
            run_dir = candidates[-1]

        if not run_dir.exists():
            raise FileNotFoundError(f"Run dir not found: {{run_dir}}")
        return run_dir

    RUN_DIR   = _find_low_quality_run_dir(RUN_NAME)
    # Auto-find the latest with_manifest CSV (written by cell 9)
    _manifest_csvs = sorted(
        (RUN_DIR / "test_eval").glob("test_metrics_with_manifest_FIXED_*.csv"),
        key=lambda p: p.stat().st_mtime)
    assert _manifest_csvs, "Run cell 9 first to generate the manifest-joined CSV"
    CSV_PATH  = _manifest_csvs[-1]
    print("Using run:", RUN_DIR)
    print("Using:", CSV_PATH.name)
    OUT_DIR   = RUN_DIR / "test_eval" / "figs"; OUT_DIR.mkdir(parents=True, exist_ok=True)
    """
)


CELL11_BLOCK = textwrap.dedent(
    f"""\
    # Point at your low-quality-trained run folder
    RUN_NAME = globals().get("RUN_NAME", "")  # set to a specific run id; leave blank for latest
    LOW_QUALITY_ROOT = Path("{LOW_QUALITY_ROOT}")
    LOW_QUALITY_RUNS = LOW_QUALITY_ROOT / "runs"

    def _find_low_quality_run_dir(run_name: str = "") -> Path:
        existing = globals().get("RUN_DIR")
        if isinstance(existing, (str, Path)):
            existing = Path(existing)
            if (existing / "models").exists() and (existing / "callbacks").exists():
                return existing

        if run_name:
            run_dir = LOW_QUALITY_RUNS / run_name
        else:
            candidates = sorted(p for p in LOW_QUALITY_RUNS.iterdir() if p.is_dir())
            if not candidates:
                raise FileNotFoundError(f"No run directories found in {{LOW_QUALITY_RUNS}}")
            run_dir = candidates[-1]

        if not run_dir.exists():
            raise FileNotFoundError(f"Run dir not found: {{run_dir}}")
        return run_dir

    RUN_DIR  = _find_low_quality_run_dir(RUN_NAME)
    EVAL_DIR = RUN_DIR / "test_eval"
    FIG_DIR  = EVAL_DIR / "figs"
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    print("Using run:", RUN_DIR)
    """
)


def lines(text: str) -> list[str]:
    if not text.endswith("\n"):
        text += "\n"
    return text.splitlines(keepends=True)


def replace_block(src: str, start_marker: str, end_marker: str, new_block: str) -> str:
    if start_marker not in src or end_marker not in src:
        return src
    start = src.index(start_marker)
    end = src.index(end_marker, start) + len(end_marker)
    return src[:start] + new_block + src[end:]


def patch_cell_0(src: str) -> str:
    src = src.replace(
        "# === Held-out evaluation for latest v3 run (macro/micro Dice, CSV+JSON) ===",
        "# === Held-out evaluation for low-quality-trained v3 run (macro/micro Dice, CSV+JSON) ===",
        1,
    )
    return replace_block(
        src,
        "# --------- Paths (v3 run + held-out set) ----------",
        "# -----------------------------------------------",
        CELL0_PATH_BLOCK,
    )


def patch_cell_1(src: str) -> str:
    src = src.replace(
        "# === Interactive MRI viewer for v3 held-out eval (uses saved preds *or* computes on-the-fly) ===",
        "# === Interactive MRI viewer for low-quality-trained v3 held-out eval (uses saved preds *or* computes on-the-fly) ===",
        1,
    )
    if "# CONFIG — low-quality-trained run on the same held-out HiRes test set:" in src:
        if "import pathlib" not in src.split("# -----------------------------------------------------------------------------------------------", 1)[0]:
            src = src.replace(
                "# CONFIG — low-quality-trained run on the same held-out HiRes test set:\n",
                "# CONFIG — low-quality-trained run on the same held-out HiRes test set:\nimport pathlib\n\n",
                1,
            )
        return src
    return replace_block(
        src,
        "# CONFIG — set these two:",
        'PREDS_DIR = ""   # e.g. "/.../test_preds_t050"',
        CELL1_CONFIG_BLOCK.rstrip(),
    )


def patch_cell_5(src: str) -> str:
    return replace_block(
        src,
        "# -------- RUN THIS BLOCK (edit paths/tag/prefix only) --------",
        'SAVE_PREFIX = "test_hires"     # <<< change per dataset',
        CELL5_RUN_BLOCK.rstrip(),
    )


def patch_cell_6(src: str) -> str:
    return CELL6_SOURCE


def patch_cell_7(_: str) -> str:
    return CELL7_SOURCE


def patch_cell_9(src: str) -> str:
    return replace_block(
        src,
        "# ---------- CONFIG ----------",
        "# -------------------------------------------",
        CELL9_CONFIG_BLOCK,
    )


def patch_cell_10(src: str) -> str:
    return replace_block(
        src,
        'RUN_DIR   = Path("/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/ARC_ATLAS_Train_v3/runs/20260410_174944")',
        'OUT_DIR   = RUN_DIR / "test_eval" / "figs"; OUT_DIR.mkdir(parents=True, exist_ok=True)',
        CELL10_BLOCK.rstrip(),
    )


def patch_cell_11(src: str) -> str:
    return replace_block(
        src,
        "# Point at your v3 run folder",
        'EVAL_DIR.mkdir(parents=True, exist_ok=True)',
        CELL11_BLOCK.rstrip(),
    )


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text())

    patches = {
        0: patch_cell_0,
        1: patch_cell_1,
        5: patch_cell_5,
        6: patch_cell_6,
        7: patch_cell_7,
        9: patch_cell_9,
        10: patch_cell_10,
        11: patch_cell_11,
    }

    for idx, patcher in patches.items():
        cell = nb["cells"][idx]
        src = "".join(cell.get("source", []))
        cell["source"] = lines(patcher(src))
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    NOTEBOOK.write_text(json.dumps(nb, indent=1))
    print(f"Updated {NOTEBOOK}")


if __name__ == "__main__":
    main()
