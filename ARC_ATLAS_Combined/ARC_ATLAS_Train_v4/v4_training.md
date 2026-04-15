## V4 Training

Date saved: 2026-04-14

Primary notebook:
- `/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/ARC_ATLAS_Train_v4/ARC_ATLAS_Train_v4_smalllesion.ipynb`

Context:
- The successful older whole-brain run was from `v3`, but it used a different trainer:
  - `v3`: `/home/rbielski/stroke_cleaned/stroke_segmentation_v1.2/stroke_seg_v1.2_train.py`
  - `v4`: `/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/ARC_ATLAS_Train_v4/src/training_v2_smalllesion.py`
- The failed `v4` runs were not true whole-brain training. They used fixed large windows (`three_window` / `hemisphere`) and collapsed to near-constant background predictions.

Change made in the `v4_smalllesion` notebook:
- Switched training back to a full-brain regime while keeping the `v4` trainer and its extra machinery.

Current training-cell config in `ARC_ATLAS_Train_v4_smalllesion.ipynb`:
- `INPUT_SHAPE = (192, 224, 192, 1)`
- `PATCH_SIZE = (192, 224, 192)`
- `PATCHES_PER_CASE = 1`
- `EPOCH_STEPS = 258`
- `TOTAL_EPOCHS = 140`
- `BASE_FILTERS = 8`
- `BATCH_SIZE = 1`
- `VAL_SPLIT = 0.15`
- `DROPOUT_RATE = 0.55`
- `L2_REG = 1.5e-3`
- `AUG_INTENSITY = 0.30`
- `ROTATION_RANGE = 20`
- `SMALL_LESION_THRESHOLD = 100`
- `SYNTHETIC_LESION_PROB = 0.3`
- `INITIAL_LR = 1e-4`
- `MIN_LR = 5e-7`
- `WARMUP_EPOCHS = 15`
- `COSINE_FIRST_CYCLE_EPOCHS = 70`
- `DICE_WEIGHT = 0.40`
- `BOUNDARY_WEIGHT = 0.60`
- `BCE_WEIGHT = 0.20`
- `VOLUME_RATIO_WEIGHT = 0.05`
- `PATCH_FG_PROB_BY_BIN = (0.995, 0.98, 0.90, 0.75)`
- `OUTPUT_BIAS_INIT_PROB = 0.015`
- `PATCH_SAMPLING_STRATEGY = "random"`
- `RESAMPLE_TO_TARGET = True`

`v4` features intentionally kept:
- patch extraction path in the trainer
- source-balanced sampling
- size-aware sampling hooks
- difficulty-aware sampling
- hybrid loss terms
- cosine restart schedule
- whole-brain validation callbacks
- diagnostics and bookkeeping

Important note:
- The preview cell above the training cell still shows the old three-window geometry.
- Training itself was changed to full-brain patches. The preview cell was not updated.

Useful files to monitor after resuming:
- `/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/ARC_ATLAS_Train_v4/ARC_ATLAS_Train_v4_smalllesion.ipynb`
- `/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/ARC_ATLAS_Train_v4/src/training_v2_smalllesion.py`
- `callbacks/training_log.csv`
- `callbacks/batch_metrics.csv`
- `callbacks/whole_val_summary.jsonl`
- `callbacks/whole_val_epoch_XXXX.csv`

Next step when returning:
- Run the edited training cell in `ARC_ATLAS_Train_v4_smalllesion.ipynb`.
- Check whether early epochs now behave like the successful `v3` run:
  - training dice should rise instead of decaying toward zero
  - whole-brain hard predictions should stop being identically zero
