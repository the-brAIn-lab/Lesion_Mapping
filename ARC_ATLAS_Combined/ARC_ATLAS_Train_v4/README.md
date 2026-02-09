# ARC_ATLAS_Train_v4 (self-contained)

Everything needed to prep + train the ARC/ATLAS v4 experiment lives here. Only the **raw** ARC and ATLAS datasets sit outside this folder (see `config/paths.yaml`). After running the prep step once, all standardized data and training artifacts stay inside this directory.

## Layout
- `src/`
  - `training_v2.py`, `training_v1.py`: local SmartSOTA trainers.
  - `data_prep/prep_utils.py`: raw→MNI prep helpers (ANTs + TemplateFlow) and combiner.
  - `data_prep/materialize_split.py`: legacy split copier (optional).
  - `downsampling/*.py`: degradation recipes using local data.
- `data/`
  - `prep_outputs/`: per-dataset standardized outputs (cached, slugged by source path).
  - `processed/train_combined/`: merged ARC+ATLAS set for training.
  - `processed/test_input/`: merged external test set after test prep.
  - `downsampled/`: targets for generated degraded sets.
- `notebooks/`
  - `ARC_ATLAS_TrainPrep_v4.ipynb`: fresh prep of ARC+ATLAS raw data into local standardized + combined train set.
  - `ARC_ATLAS_TestPrep_v4.ipynb`: prep any external dataset for testing.
  - `ARC_ATLAS_Train_v4.ipynb`: training (expects `data/processed/train_combined`).
  - `ARC_ATLAS_Test_v4.ipynb`: evaluate latest run; uses `data/processed/test_input` by default.
  - `prep/`: legacy prep notebooks (kept for reference).
- `requirements.txt`: runtime deps.

## Quick start
1. Run `ARC_ATLAS_TrainPrep_v4.ipynb` (defaults: ARC raw at `../ARC/ds004884/derivatives/aggregates/t1w_with_masks`; ATLAS raw at `../Atlas_2`, edit as needed). Outputs go to `data/prep_outputs/*` and are combined into `data/processed/train_combined`.
2. Run `ARC_ATLAS_Train_v4.ipynb` to train; it creates `runs/latest` and `runs/latest_best.weights.h5`.
3. For external data, run `ARC_ATLAS_TestPrep_v4.ipynb` to standardize into `data/processed/test_input`, then use `ARC_ATLAS_Test_v4.ipynb` to evaluate. Downsampling recipes live in `src/downsampling/`.

## Notes
- All logs, callbacks, and configs stay under `runs/`; `runs/latest` points to the most recent training run and `runs/latest_best.weights.h5` mirrors its best checkpoint.
