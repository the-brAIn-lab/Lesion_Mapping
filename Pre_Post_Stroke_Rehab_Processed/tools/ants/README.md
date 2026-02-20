# Local ANTs Runtime

Place ANTs binaries here to keep prep self-contained inside `ARC_ATLAS_Train_v4`.

Expected paths:
- `tools/ants/bin/antsRegistration`
- `tools/ants/bin/antsApplyTransforms`

Notes:
- `src/data_prep/prep_utils.py` checks these local paths first.
- If these binaries are not present, it falls back to `ANTS_REG` / `ANTS_APPLY` or `PATH`.
- Binaries should be executable (`chmod +x`).

