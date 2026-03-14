# Self-Config Heuristics

Self-contained experiment derived from ARC_ATLAS_Train_v4.

## Data
- Uses local subset at `data/train` (9 total cases: 3 Approx + 3 ATLAS + 3 ARC).
- Validation target is `VAL_SPLIT=1/3` with `BATCH_SIZE=1` to keep 3 validation cases.

## Files
- `src/training_v2.py`: method-specific training variant.
- `train_experiment.ipynb`: run notebook for this method.
- `runs/`: outputs for this method.

## Method-Specific Changes
- Adds `AUTO_SELF_CONFIG` logic that adapts schedule settings to dataset size.
- For very small sets (like this 9-case subset), it reduces oversized training loops (lower effective `EPOCH_STEPS`, capped `TOTAL_EPOCHS`) and disables overly noisy difficulty-aware updates.
- Why this helps: tiny datasets are easy to overfit and unstable with large-step schedules copied from full-scale training.
- The goal is to make optimization behavior proportional to available data so comparisons between methods are less noisy.

## How To Run
1. Open `train_experiment.ipynb` in this folder.
2. Run cell 1 to train and write outputs to `runs/<timestamp>`.
3. Run cell 2 for a quick zero-input sanity prediction.
