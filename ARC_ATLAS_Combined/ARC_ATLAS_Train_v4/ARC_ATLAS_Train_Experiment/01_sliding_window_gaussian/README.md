# Sliding-Window + Gaussian Blending

Self-contained experiment derived from ARC_ATLAS_Train_v4.

## Data
- Uses local subset at `data/train` (9 total cases: 3 Approx + 3 ATLAS + 3 ARC).
- Validation target is `VAL_SPLIT=1/3` with `BATCH_SIZE=1` to keep 3 validation cases.

## Files
- `src/training_v2.py`: method-specific training variant.
- `train_experiment.ipynb`: run notebook for this method.
- `runs/`: outputs for this method.

## Method-Specific Changes
- `GAUSSIAN_TILE_OVERLAP` is increased to `0.75` and `GAUSSIAN_TILE_SIGMA` to `0.18` in this variant.
- Higher overlap means each voxel is predicted from more neighboring patches, so seam artifacts at patch borders are reduced.
- A broader Gaussian blend downweights patch edges less aggressively, which usually improves continuity when stitching whole-brain outputs.
- This method mainly targets inference/reconstruction quality (especially around patch boundaries), not raw model capacity.

## How To Run
1. Open `train_experiment.ipynb` in this folder.
2. Run cell 1 to train and write outputs to `runs/<timestamp>`.
3. Run cell 2 for a quick zero-input sanity prediction.
