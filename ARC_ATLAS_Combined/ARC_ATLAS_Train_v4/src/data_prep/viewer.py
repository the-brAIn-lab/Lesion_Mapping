"""Interactive MRI+mask slice viewer for standardized datasets.

Usage in a notebook:
    from pathlib import Path
    from src.data_prep.viewer import show_viewer
    show_viewer(Path('data/processed/train_combined'))

Data layout expected:
    root/
      t1/    *_T1w_MNI_norm.nii.gz
      masks/ *_lesion_mask_MNI_clean.nii.gz
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import re
import csv
import numpy as np
import nibabel as nib
import ipywidgets as W
import matplotlib.pyplot as plt
from IPython.display import display, clear_output


def _pairs(root: Path):
    t1_dir, mk_dir = root / "t1", root / "masks"
    if not t1_dir.exists() or not mk_dir.exists():
        raise FileNotFoundError(f"Expected t1/ and masks/ under {root}")
    pairs = {}
    manifest = root / "manifest.csv"

    if manifest.exists():
        with manifest.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t1 = Path(row.get("t1", ""))
                mk = Path(row.get("mask", ""))
                if not t1.is_absolute():
                    t1 = root / t1
                if not mk.is_absolute():
                    mk = root / mk
                if not (t1.exists() and mk.exists()):
                    continue
                label = t1.name.replace("_T1w_MNI_norm", "")
                idx = 2
                lbl = label
                while lbl in pairs:
                    lbl = f"{label} ({idx})"; idx += 1
                pairs[lbl] = {"t1": t1, "mask": mk}
        if pairs:
            return pairs

    for t1 in sorted(t1_dir.glob("*.nii.gz")):
        base = t1.name.replace("_T1w_MNI_norm", "")
        mask = mk_dir / t1.name.replace("_T1w_MNI_norm", "_lesion_mask_MNI_clean")
        if not mask.exists():
            continue
        label = base
        idx = 2
        lbl = label
        while lbl in pairs:
            lbl = f"{label} ({idx})"; idx += 1
        pairs[lbl] = {"t1": t1, "mask": mask}
    if not pairs:
        raise RuntimeError(f"No T1/mask pairs found under {root}")
    return pairs

@lru_cache(maxsize=256)
def _img(path: str):
    # Canonicalize to avoid orientation flips in display
    return nib.as_closest_canonical(nib.load(path))

@lru_cache(maxsize=256)
def _vol(path: str):
    arr = _img(path).get_fdata()
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr.astype(np.float32)


def _normalize(img: np.ndarray) -> np.ndarray:
    nz = img[np.isfinite(img)]
    nz = nz[nz > 0]
    if nz.size == 0:
        return np.zeros_like(img, dtype=np.float32)
    p1, p99 = np.percentile(nz, [1, 99])
    img = np.clip(img, p1, p99)
    m, s = nz.mean(), nz.std()
    if s > 0:
        img = (img - m) / s
    mn, mx = img.min(), img.max()
    return (img - mn) / (mx - mn + 1e-8)


def _edges2d(mask2d):
    # Contouring the binary mask at 0.5 traces the true voxel boundary.
    return mask2d.astype(np.float32, copy=False)


def _zooms3(img: nib.Nifti1Image):
    z = img.header.get_zooms()[:3]
    return tuple(float(v) for v in z)


def _aff_equal(a, b, tol=1e-4):
    return np.allclose(a, b, atol=tol)


def show_viewer(root: Path):
    # Allow callers to pass None to use cwd/data/processed/train_combined
    if root is None:
        root = Path.cwd() / "data" / "processed" / "train_combined"
    root = Path(root)
    pairs = _pairs(root)
    keys = sorted(pairs.keys())

    dd_case   = W.Dropdown(options=keys, description="Case:", layout=W.Layout(width="100%"))
    sl_slice  = W.IntSlider(description="Axial slice:", min=0, max=1, value=0, continuous_update=False, layout=W.Layout(width="60%"))
    sl_alpha  = W.FloatSlider(description="Mask α:", min=0.0, max=1.0, step=0.05, value=0.55, layout=W.Layout(width="35%"))
    cb_edges  = W.Checkbox(description="Edges only", value=True)
    cb_invert = W.Checkbox(description="Invert image", value=False)
    status = W.HTML(f"<b>Viewer</b> — cases: {len(keys)} | source: {root}")
    controls = W.VBox([status, dd_case, W.HBox([sl_slice, sl_alpha]), W.HBox([cb_edges, cb_invert])])
    out = W.Output()

    def _update_slice_range(*_):
        key = dd_case.value
        t1 = _img(str(pairs[key]["t1"]))
        sl_slice.max = max(0, t1.shape[2] - 1)
        sl_slice.value = min(sl_slice.value, sl_slice.max)

    def _draw(*_):
        with out:
            clear_output(wait=True)
            key = dd_case.value
            t1_path = pairs[key]["t1"]
            mask_path = pairs[key]["mask"]

            t1_img   = _img(str(t1_path))
            mask_img = _img(str(mask_path))
            t1_vol   = _vol(str(t1_path))
            mask_vol = _vol(str(mask_path))

            same_shape  = t1_vol.shape[:3] == mask_vol.shape[:3]
            same_affine = _aff_equal(t1_img.affine, mask_img.affine)

            sl_slice.max = max(0, t1_vol.shape[2] - 1)
            idx = int(sl_slice.value)

            img2d = t1_vol[:, :, idx]
            img2d = _normalize(img2d)
            if cb_invert.value:
                img2d = 1.0 - img2d

            fig, axes = (plt.subplots(1, 2, figsize=(10, 5)) if not same_shape else (plt.subplots(1, 1, figsize=(5.6, 5.6))))
            if not same_shape:
                axes = np.atleast_1d(axes)

            if same_shape:
                mask2d = mask_vol[:, :, idx] > 0.5
                plt.imshow(img2d.T, cmap="gray", origin="lower")
                if cb_edges.value:
                    plt.contour(_edges2d(mask2d).T, levels=[0.5], linewidths=0.8, colors="r")
                else:
                    plt.imshow(
                        np.ma.masked_where(~mask2d.T, mask2d.T),
                        cmap="jet",
                        alpha=float(sl_alpha.value),
                        origin="lower",
                        interpolation="nearest",
                    )
                plt.axis("off"); plt.tight_layout(); plt.show(); plt.close()
            else:
                axes[0].imshow(img2d.T, cmap="gray", origin="lower")
                axes[0].set_title("Image slice"); axes[0].axis("off")
                mask_slice = mask_vol[:, :, min(idx, mask_vol.shape[2]-1)]
                axes[1].imshow(mask_slice.T, cmap="hot", origin="lower")
                axes[1].set_title("Mask slice (native)"); axes[1].axis("off")
                plt.tight_layout(); plt.show(); plt.close()

            status.value = (
                f"<b>{key}</b> | image: {t1_vol.shape[:3]} {tuple(round(z,3) for z in _zooms3(t1_img))} | "
                f"mask: {mask_vol.shape[:3]} {tuple(round(z,3) for z in _zooms3(mask_img))} | "
                f"affine match: {'✅' if same_affine else '⚠️'} | overlay: {'✅' if same_shape and same_affine else '❌'}"
            )
            print("Image:", t1_path)
            print("Mask :", mask_path)
            if not same_shape or not same_affine:
                print("⚠️ Shapes or affines differ; mask shown separately with no resampling.")

    dd_case.observe(_update_slice_range, names="value")
    dd_case.observe(_draw, names="value")
    sl_slice.observe(_draw, names="value")
    sl_alpha.observe(_draw, names="value")
    cb_edges.observe(_draw, names="value")
    cb_invert.observe(_draw, names="value")

    _update_slice_range(); _draw()
    display(controls, out)


__all__ = ["show_viewer"]
