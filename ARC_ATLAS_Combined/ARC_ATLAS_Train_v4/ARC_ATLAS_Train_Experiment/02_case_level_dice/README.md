# Case-Level Dice Emphasis

Self-contained experiment derived from ARC_ATLAS_Train_v4.

## Data
- Uses local subset at `data/train` (9 total cases: 3 Approx + 3 ATLAS + 3 ARC).
- Validation target is `VAL_SPLIT=1/3` with `BATCH_SIZE=1` to keep 3 validation cases.

## Files
- `src/training_v2.py`: method-specific training variant.
- `train_experiment.ipynb`: run notebook for this method.
- `runs/`: outputs for this method.

## Method-Specific Changes
- This variant adds case-level validation Dice percentiles (`p25`, `p50`, `p75`) on top of the existing macro/micro Dice.
- The core model and optimizer are unchanged, so metric changes reflect evaluation visibility, not architecture differences.
- Why this helps: mean Dice alone can hide unstable behavior; percentile metrics reveal if a few hard cases are failing while averages look acceptable.
- In mixed-cohort data, this is useful for catching subgroup collapse early (for example one dataset source underperforming badly).

## How To Run
1. Open `train_experiment.ipynb` in this folder.
2. Run cell 1 to train and write outputs to `runs/<timestamp>`.
3. Run cell 2 for a quick zero-input sanity prediction.
