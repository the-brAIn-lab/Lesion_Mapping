# Cross-Site Intensity Harmonization

Self-contained experiment derived from ARC_ATLAS_Train_v4.

## Data
- Uses local subset at `data/train` (9 total cases: 3 Approx + 3 ATLAS + 3 ARC).
- Validation target is `VAL_SPLIT=1/3` with `BATCH_SIZE=1` to keep 3 validation cases.

## Files
- `src/training_v2.py`: method-specific training variant.
- `train_experiment.ipynb`: run notebook for this method.
- `runs/`: outputs for this method.

## Method-Specific Changes
- Adds optional loader-level harmonization (`HARMONIZE_INTENSITY=True` in notebook override) before model input.
- The harmonizer uses robust clipping (`1st-99th percentile`), non-zero-voxel z-scoring, then bounded rescaling to a stable range.
- Why this helps: ARC/ATLAS/Approx can differ in scanner intensity distributions; harmonization reduces that domain gap.
- By normalizing intensity style, the model can focus more on anatomical/lesion patterns instead of site-specific brightness differences.

## How To Run
1. Open `train_experiment.ipynb` in this folder.
2. Run cell 1 to train and write outputs to `runs/<timestamp>`.
3. Run cell 2 for a quick zero-input sanity prediction.
