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
  - `templateflow/`: local TemplateFlow cache (default `TEMPLATEFLOW_HOME`).
  - `processed/train_atlas_only/`: default training set.
  - `processed/train_combined/`: optional merged ARC+ATLAS set for training.
  - `processed/test_input/`: merged external test set after test prep.
  - `downsampled/`: targets for generated degraded sets.
- `tools/`
  - `ants/bin/`: local ANTs runtime binaries (`antsRegistration`, `antsApplyTransforms`).
- `notebooks/`
  - `ARC_ATLAS_TrainPrep_v4.ipynb`: fresh prep of ARC+ATLAS raw data into local standardized + combined train set.
  - `ARC_ATLAS_TestPrep_v4.ipynb`: prep any external dataset for testing.
  - `ARC_ATLAS_Train_v4.ipynb`: training (default `data/processed/train_atlas_only`).
  - `ARC_ATLAS_Test_v4.ipynb`: evaluate latest run; uses `data/processed/test_input` by default.
  - `prep/`: legacy prep notebooks (kept for reference).
- `requirements.txt`: runtime deps.

## Self-contained Runtime Dependencies
- `prep_utils.py` now defaults to local runtime paths inside this project:
  - ANTs binaries: `tools/ants/bin/antsRegistration` and `tools/ants/bin/antsApplyTransforms`
  - TemplateFlow cache: `data/templateflow` (via `TEMPLATEFLOW_HOME`)
- You can still override paths with environment variables:
  - `ANTS_REG`, `ANTS_APPLY`, `TEMPLATEFLOW_HOME`
- If local ANTs binaries are missing, prep will fail with an explicit error showing the expected local path.

## Quick start
1. Run `ARC_ATLAS_TrainPrep_v4.ipynb` (current default is ATLAS-only). Outputs go to `data/prep_outputs/*` and are combined into `data/processed/train_atlas_only`.
2. Run `ARC_ATLAS_Train_v4.ipynb` to train from `data/processed/train_atlas_only`; it creates `runs/latest` and `runs/latest_best.weights.h5`.
3. For external data, run `ARC_ATLAS_TestPrep_v4.ipynb` to standardize into `data/processed/test_input`.
   Set `HAS_MASKS=True` for metric evaluation, or `HAS_MASKS=False` for image-only prediction prep.
4. Run `ARC_ATLAS_Test_v4.ipynb` to generate predicted masks for all test images.
   If masks are present, it also reports Dice and writes a per-case CSV in `runs/<run>/test_predictions/<timestamp>/`.
5. Downsampling recipes live in `src/downsampling/`.

## Notes
- All logs, callbacks, and configs stay under `runs/`; `runs/latest` points to the most recent training run and `runs/latest_best.weights.h5` mirrors its best checkpoint.
