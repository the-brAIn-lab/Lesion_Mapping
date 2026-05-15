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
- `requirements_nnunet.txt`: extra PyTorch/nnU-Net deps for the leaderboard-style ATLAS pipeline.

## Self-contained Runtime Dependencies
- `prep_utils.py` now defaults to local runtime paths inside this project:
  - ANTs binaries: `tools/ants/bin/antsRegistration` and `tools/ants/bin/antsApplyTransforms`
  - TemplateFlow cache: `data/templateflow` (via `TEMPLATEFLOW_HOME`)
- You can still override paths with environment variables:
  - `ANTS_REG`, `ANTS_APPLY`, `TEMPLATEFLOW_HOME`
- If local ANTs binaries are missing, prep will fail with an explicit error showing the expected local path.

## Quick start: native-split nnU-Net baseline
The primary v4 training path is now `ARC_ATLAS_Train_v4.ipynb`, refactored around the stronger ATLAS leaderboard recipe while keeping the same source split used by the recent v4 runs: `data/splits/90_10_random/train/manifest.csv`. The nnU-Net files are a format view under that split directory, not a different training dataset.

1. Install the nnU-Net runtime in the environment that will do training:
   ```bash
   conda activate tf_310
   pip install -r requirements_nnunet.txt
   ```
   The current notebook is configured for the `tf_310` kernel. If the default PyTorch wheel is wrong for the GPU/CUDA stack, install the matching `torch` wheel first and then install the rest of the file.
2. Run `ARC_ATLAS_Train_v4.ipynb`.
   It exposes `data/splits/90_10_random/train` as `data/splits/90_10_random/train/nnunet_view/nnUNet_raw/Dataset701_ARC_ATLAS_TrainV4Native`, writes `splits_final.json`, and prints the exact planning/training commands.
   The MRIs are symlinked from the same split and keep their native full dimensions/spacing; no `TARGET_SHAPE` resize is applied.
3. Run planning/preprocessing:
   ```bash
   export nnUNet_raw=/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/ARC_ATLAS_Train_v4/data/splits/90_10_random/train/nnunet_view/nnUNet_raw
   export nnUNet_preprocessed=/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/ARC_ATLAS_Train_v4/data/splits/90_10_random/train/nnunet_view/nnUNet_preprocessed
   export nnUNet_results=/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/ARC_ATLAS_Train_v4/data/splits/90_10_random/train/nnunet_view/nnUNet_results
   nnUNetv2_plan_and_preprocess -d 701 --verify_dataset_integrity -c 3d_fullres
   ```
   `3d_fullres` preserves full voxel resolution. If the whole volume does not fit GPU memory, nnU-Net's planner falls back to a full-resolution 3D patch size.
   On this machine, the default planning-only check proposed `128x160x112`; a 24 GB RTX 4090-targeted plan proposed `160x192x160`:
   ```bash
   nnUNetv2_plan_and_preprocess -d 701 --no_pp -c 3d_fullres -gpu_memory_target 24 -overwrite_plans_name nnUNetPlans_24GB
   ```
4. Train the five full-resolution folds:
   ```bash
   for f in 0 1 2 3 4; do nnUNetv2_train 701 3d_fullres "$f" -p nnUNetPlans_24GB --npz; done
   ```
5. Generate leak-free CV predictions and evaluate:
   ```bash
   /home/rbielski/miniconda3/envs/tf_310/bin/python src/atlas_nnunet_pipeline.py predict-cv
   /home/rbielski/miniconda3/envs/tf_310/bin/python src/atlas_nnunet_pipeline.py evaluate
   ```

## Legacy TensorFlow quick start
1. Run `ARC_ATLAS_TrainPrep_v4.ipynb` (current default is ATLAS-only). Outputs go to `data/prep_outputs/*` and are combined into `data/processed/train_atlas_only`.
2. Use `ARC_ATLAS_Train_v4_slices.ipynb` or `ARC_ATLAS_Train_v4_slices_resume.ipynb` for the older 2.5D slice-block experiments.
3. For external data, run `ARC_ATLAS_TestPrep_v4.ipynb` to standardize into `data/processed/test_input`.
   Set `HAS_MASKS=True` for metric evaluation, or `HAS_MASKS=False` for image-only prediction prep.
4. Run `ARC_ATLAS_Test_v4.ipynb` to generate predicted masks for all test images.
   If masks are present, it also reports Dice and writes a per-case CSV in `runs/<run>/test_predictions/<timestamp>/`.
5. Downsampling recipes live in `src/downsampling/`.

## Notes
- All logs, callbacks, and configs stay under `runs/`; `runs/latest` points to the most recent training run and `runs/latest_best.weights.h5` mirrors its best checkpoint.
