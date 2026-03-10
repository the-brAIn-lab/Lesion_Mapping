"""Skull stripping using ANTsPyNet brain_extraction.

Strips skull from a T1w image in native space before registration.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def _skull_strip_external_python(
    python_exe: Path,
    t1: Path,
    out_path: Path,
    verbose: bool = False,
) -> None:
    code = textwrap.dedent(
        f"""
        from pathlib import Path
        import ants
        from antspynet.utilities import brain_extraction

        t1 = Path({str(t1)!r})
        out_path = Path({str(out_path)!r})
        out_path.parent.mkdir(parents=True, exist_ok=True)

        img = ants.image_read(str(t1))
        prob = brain_extraction(img, modality="t1", verbose={bool(verbose)!r})
        mask = ants.get_mask(prob, low_thresh=0.5)
        stripped = ants.mask_image(img, mask)
        ants.image_write(stripped, str(out_path))
        """
    )
    env = dict(os.environ)
    # Default to CPU for skull stripping to avoid GPU OOM contention with training.
    use_gpu = str(env.get("PREP_SKULL_STRIP_USE_GPU", "")).strip().lower() in {"1", "true", "yes", "on"}
    if not use_gpu:
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    # Optional hard isolation if explicitly requested by caller.
    if str(env.get("PREP_SKULL_STRIP_FORCE_NO_USERSITE", "")).strip().lower() in {"1", "true", "yes", "on"}:
        env["PYTHONNOUSERSITE"] = "1"
    else:
        env.pop("PYTHONNOUSERSITE", None)
    subprocess.run([str(python_exe), "-c", code], check=True, env=env)


def skull_strip_antspynet(t1: Path, out_path: Path, verbose: bool = False) -> Path:
    """Skull-strip a T1w image using ANTsPyNet brain_extraction.

    The output keeps the same voxel grid and affine as the input; non-brain
    voxels are set to zero.
    """
    t1 = Path(t1)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import ants
        from antspynet.utilities import brain_extraction
        img = ants.image_read(str(t1))
        prob = brain_extraction(img, modality="t1", verbose=verbose)
        mask = ants.get_mask(prob, low_thresh=0.5)
        stripped = ants.mask_image(img, mask)
        ants.image_write(stripped, str(out_path))
    except Exception as e:
        fallback_py_raw = (os.environ.get("PREP_SKULL_STRIP_PYTHON") or "").strip()
        fallback_py = Path(fallback_py_raw).expanduser() if fallback_py_raw else None
        if fallback_py and fallback_py.exists():
            try:
                _skull_strip_external_python(fallback_py, t1, out_path, verbose=verbose)
                print(f"  [skull_strip:fallback {fallback_py}] {t1.name} -> {out_path.name}")
                return out_path
            except Exception as fallback_err:
                raise RuntimeError(
                    "Local skull-strip import failed and fallback interpreter failed.\n"
                    f"Local error: {e}\n"
                    f"Fallback python: {fallback_py}\n"
                    f"Fallback error: {fallback_err}"
                ) from fallback_err
        raise RuntimeError(
            "Skull stripping failed in the active Python process.\n"
            "If this is a GPU OOM or TensorFlow runtime issue, set PREP_SKULL_STRIP_USE_GPU=0\n"
            "and PREP_SKULL_STRIP_PYTHON to a working interpreter.\n"
            "Example: /home/rbielski/miniconda3/envs/tf_310/bin/python\n"
            f"Original error: {e}"
        ) from e

    print(f"  [skull_strip] {t1.name} -> {out_path.name}")
    return out_path


__all__ = ["skull_strip_antspynet"]
