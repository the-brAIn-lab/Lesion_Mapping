"""Skull stripping using ANTsPyNet brain_extraction.

Strips the skull from a T1w image in native space. Should be run
BEFORE registration to MNI, since the registration target is a
brain-only MNI template (desc='brain'). Brain-to-brain registration
is substantially more accurate than skull+brain-to-brain.

Requires:
    pip install antspynet ants

Usage:
    from data_prep.skull_strip import skull_strip_antspynet
    skull_strip_antspynet(Path("sub-001_T1w.nii.gz"), Path("sub-001_T1w_brain.nii.gz"))
"""
from __future__ import annotations
from pathlib import Path


def skull_strip_antspynet(t1: Path, out_path: Path, verbose: bool = False) -> Path:
    """Skull-strip a T1w image using ANTsPyNet brain_extraction.

    The output has the same voxel grid and affine as the input — non-brain
    voxels are zeroed out. Safe to use as the registration source since the
    ANTs binary MNI template is also brain-only.

    Parameters
    ----------
    t1       : Path to the input T1w NIfTI (.nii.gz)
    out_path : Destination path for the skull-stripped T1w NIfTI
    verbose  : Show ANTsPyNet download / inference progress

    Returns
    -------
    out_path (for chaining)
    """
    try:
        import ants
        from antspynet.utilities import brain_extraction
    except ImportError as e:
        raise ImportError(
            "antspynet and ants are required for skull stripping.\n"
            "Install with:  pip install antspynet ants\n"
            f"Original error: {e}"
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = ants.image_read(str(t1))
    prob = brain_extraction(img, modality="t1", verbose=verbose)
    mask = ants.get_mask(prob, low_thresh=0.5)
    stripped = ants.mask_image(img, mask)
    ants.image_write(stripped, str(out_path))

    print(f"  [skull_strip] {t1.name} → {out_path.name}")
    return out_path


__all__ = ["skull_strip_antspynet"]
