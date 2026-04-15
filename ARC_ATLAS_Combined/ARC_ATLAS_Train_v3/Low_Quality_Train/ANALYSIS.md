**Overview**
`ARC_ATLAS_Train_v3.ipynb` is a thin launcher around the real training code in `/home/rbielski/stroke_cleaned/stroke_segmentation_v1.2/stroke_seg_v1.2_train.py`. The copied notebook [`ARC_ATLAS_Train_v3_Low_Quality.ipynb`](./ARC_ATLAS_Train_v3_Low_Quality.ipynb) now points at the new low-quality split and writes runs under this `Low_Quality_Train` folder.

**Model**
The model is built in `build_dynamic_model()` inside `stroke_seg_v1.2_train.py`.

- 3D U-Net-like encoder/decoder.
- Encoder block per level: `ResidualConvBlock` then `VisionMambaBlock`.
- Bottleneck: `ResidualConvBlock` then `SAM2Attention`.
- Decoder mirrors the encoder with `UpSampling3D`, skip concatenation, `ResidualConvBlock`, `VisionMambaBlock`.
- Output head: `Conv3D(1, kernel_size=1, activation="sigmoid", name="probs")`.

Important training details from `train_dynamic_model()`:

- Loss: `CombinedLoss = 0.6 * dice_loss + 0.4 * boundary_loss`.
- Metric: `dice_coefficient`.
- Optimizer: Adam with `INITIAL_LR=1e-4`, `global_clipnorm=1.0`.
- LR schedule: `ReduceLROnPlateau` on `val_dice_coefficient`.
- Validation split is internal to the training set (`VALIDATION_SPLIT=0.15`).
- Input shape is hardcoded by the notebook to `(192, 224, 192, 1)`.
- Batch size is `1`.
- Augmentation in `DynamicDataGenerator`: flips, limited rotations, gamma adjustment.

One implementation detail worth knowing: despite the config flag `RESAMPLE_TO_TARGET=True`, the current generator path used here only center-crops/pads volumes; it does not resample voxel grids during training.

**Dependencies**
Primary runtime dependencies:

- TensorFlow / Keras mixed precision.
- `nibabel` for NIfTI I/O.
- `numpy`.
- `scipy.ndimage` for rotations, Laplacian, boundary loss support.
- `scikit-learn` for the KFold-based train/validation split helper.
- `psutil` for memory logging.

Data-prep dependencies for the processed ARC+ATLAS pool come from [`Process_All_v3.ipynb`](../../A_A_Combined_Data/Processed_HiresLowres_Split_Data/Process_All_v3.ipynb):

- ANTs binaries from `ARC_ATLAS_Train_v4/tools/ants/bin`.
- TemplateFlow MNI templates from `ARC_ATLAS_Train_v4/data/templateflow`.
- `prep_utils` from `ARC_ATLAS_Train_v4/src/data_prep`.

**Where The Images Came From**
The processed training/eval images in `Processed_HiresLowres_All` were created by registering and normalizing the raw ARC and ATLAS T1/mask pairs.

Raw source paths used by `Process_All_v3.ipynb`:

- ARC T1: `/home/rbielski/ARC/combined_t1_raw`
- ARC masks: `/home/rbielski/ARC/combined_t1_raw/combined_masks_raw`
- ATLAS T1: `/home/rbielski/Atlas_2/Training/Images`
- ATLAS masks: `/home/rbielski/Atlas_2/Training/Masks`

Each case is:

1. Registered to MNI with ANTs.
2. Transformed into template space.
3. Intensity-normalized.
4. Saved as `_T1w_MNI_norm.nii.gz` and `_lesion_mask_MNI_clean.nii.gz`.

That processed pool contains `858` paired cases.

**Existing v3 Split**
The current `Processed_HiresLowres_Split_Data` split is v3-compatible:

- `test_hires`: fixed `138`-case held-out set used by `ARC_ATLAS_Test_v3_HiRes.ipynb`
- `test_lores`: `198`
- `train_hires`: `522`

The fixed `test_hires` keys are preserved in the new split exactly.

**Original Quality Scoring Logic**
The split concept comes from `ARC_ATLAS_Combined_Prep.ipynb`. Quality is ranked with:

`score = z(hi_freq_energy) + z(lap_var) + z(-fwhm_mm)`

Higher score means sharper / higher-quality images.

**New Low-Quality Split**
Created by [`build_low_quality_train_split.py`](../../A_A_Combined_Data/Processed_HiresLowres_Split_Data/build_low_quality_train_split.py).

Output root:
`/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/A_A_Combined_Data/Processed_LowQualityTrain_Split_Data`

Rules:

1. Keep the same `138` fixed `test_hires` cases.
2. Take the remaining `720` non-held-out cases.
3. Rank them by ascending quality score.
4. Assign the lowest-quality `522` to `train_low_quality`.
5. Assign the remaining `198` to `midres`.

Resulting split summary:

- `train_low_quality`: `522`
- `midres`: `198`
- `test_hires`: `138`

Interesting consequence of the exact lowest-quality rule:

- `train_low_quality` ended up being all `ATLAS` cases.
- Composition relative to the prior v3 split:
  - `344` came from old `train_hires`
  - `178` came from old `test_lores`

Supporting artifacts written with the split:

- `Processed_LowQualityTrain_Split_Data/train_low_quality.csv`
- `Processed_LowQualityTrain_Split_Data/midres.csv`
- `Processed_LowQualityTrain_Split_Data/test_hires.csv`
- `Processed_LowQualityTrain_Split_Data/all_cases_quality_manifest.csv`
- `Processed_LowQualityTrain_Split_Data/split_summary.json`

**Training Notebook Changes**
[`ARC_ATLAS_Train_v3_Low_Quality.ipynb`](./ARC_ATLAS_Train_v3_Low_Quality.ipynb) now:

- trains from `Processed_LowQualityTrain_Split_Data/train_low_quality`
- writes runs to `ARC_ATLAS_Train_v3/Low_Quality_Train/runs`
- keeps using the same underlying model code and hyperparameters as v3
