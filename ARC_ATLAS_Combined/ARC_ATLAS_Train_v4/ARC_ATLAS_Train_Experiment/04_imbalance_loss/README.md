# Imbalance-Aware Loss Mix

Self-contained experiment derived from ARC_ATLAS_Train_v4.

## Data
- Uses local subset at `data/train` (9 total cases: 3 Approx + 3 ATLAS + 3 ARC).
- Validation target is `VAL_SPLIT=1/3` with `BATCH_SIZE=1` to keep 3 validation cases.

## Files
- `src/training_v2.py`: method-specific training variant.
- `train_experiment.ipynb`: run notebook for this method.
- `runs/`: outputs for this method.

## Method-Specific Changes
- Loss weighting is shifted toward harder imbalance-aware behavior: `DICE_WEIGHT=0.35`, `BOUNDARY_WEIGHT=0.45`, `DICE_LOSS_WEIGHT=0.35`, `BOUNDARY_LOSS_WEIGHT=0.45`.
- Tversky settings are also biased (`TVERSKY_ALPHA=0.75`, `TVERSKY_BETA=0.25`) to penalize false positives more strongly.
- Why this helps: lesion segmentation is highly imbalanced; without weighting, models can drift toward background-heavy predictions.
- Emphasizing boundary + asymmetric overlap terms can improve lesion delineation and reduce trivial all-background solutions.

## How To Run
1. Open `train_experiment.ipynb` in this folder.
2. Run cell 1 to train and write outputs to `runs/<timestamp>`.
3. Run cell 2 for a quick zero-input sanity prediction.
