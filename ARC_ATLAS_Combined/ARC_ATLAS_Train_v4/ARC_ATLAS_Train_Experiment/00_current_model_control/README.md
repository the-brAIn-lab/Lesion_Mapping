# Current Model Control (Baseline)

Self-contained control experiment derived from the current `ARC_ATLAS_Train_v4` baseline.

## Data
- Uses local subset at `data/train` (9 total cases: 3 Approx + 3 ATLAS + 3 ARC).
- Validation target is `VAL_SPLIT=1/3` with `BATCH_SIZE=1` to keep 3 validation cases.

## Files
- `src/training_v2.py`: baseline training script (no method-specific experimental modifications).
- `train_experiment.ipynb`: run notebook for this control.
- `runs/`: outputs for this control.

## Method-Specific Changes
- None. This is the control arm.
- Purpose: provide a direct baseline for comparing experiments `01`-`06` under identical subset and notebook settings.

## How To Run
1. Open `train_experiment.ipynb` in this folder.
2. Run cell 1 to train and write outputs to `runs/<timestamp>`.
3. Run cell 2 for a quick zero-input sanity prediction.
