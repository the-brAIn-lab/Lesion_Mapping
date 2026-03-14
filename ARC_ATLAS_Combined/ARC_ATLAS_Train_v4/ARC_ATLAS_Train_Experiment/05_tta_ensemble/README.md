# TTA + SWA Ensemble Proxy

Self-contained experiment derived from ARC_ATLAS_Train_v4.

## Data
- Uses local subset at `data/train` (9 total cases: 3 Approx + 3 ATLAS + 3 ARC).
- Validation target is `VAL_SPLIT=1/3` with `BATCH_SIZE=1` to keep 3 validation cases.

## Files
- `src/training_v2.py`: method-specific training variant.
- `train_experiment.ipynb`: run notebook for this method.
- `runs/`: outputs for this method.

## Method-Specific Changes
- Enables test-time augmentation for whole-brain validation (`WHOLE_BRAIN_VAL_TTA=True`).
- Increases SWA usage (`SWA_EPOCHS=10`, `SWA_LR_MULT=0.5`) to average late-training weights.
- Why this helps: TTA and SWA both reduce prediction variance from any single forward pass or single checkpoint.
- This typically improves robustness/generalization at the cost of extra compute during evaluation.

## How To Run
1. Open `train_experiment.ipynb` in this folder.
2. Run cell 1 to train and write outputs to `runs/<timestamp>`.
3. Run cell 2 for a quick zero-input sanity prediction.
