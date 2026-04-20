# Training Diagnostics Summary

- Epochs recorded: 140
- Run dir: `/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/ARC_ATLAS_Train_v4/runs/20260417_163116`

## Final Metrics
- `dice_coefficient`: 0.001122
- `loss`: 1.039385
- `safe_binary_iou`: 0.108527
- `val_dice_coefficient`: 0.001215
- `val_whole_dice_hard`: 0.000000
- `val_whole_dice_hard_best_thr`: 0.300000
- `val_whole_dice_hard_best_thr_score`: 0.000000
- `val_whole_dice_hard_brainmask`: 0.000000
- `val_whole_dice_hard_brainmask_best_thr`: 0.300000
- `val_whole_dice_hard_brainmask_best_thr_score`: 0.000000
- `val_whole_dice_hard_brainmask_delta`: 0.000000
- `val_whole_dice_hard_brainmask_thr_0p30`: 0.000000
- `val_whole_dice_hard_brainmask_thr_0p40`: 0.000000
- `val_whole_dice_hard_brainmask_thr_0p50`: 0.000000
- `val_whole_dice_hard_brainmask_thr_0p60`: 0.000000
- `val_whole_dice_hard_brainmask_thr_0p70`: 0.000000
- `val_whole_dice_hard_postproc_delta`: 0.000000
- `val_whole_dice_hard_raw`: 0.000000
- `val_whole_dice_hard_raw_best_thr`: 0.300000
- `val_whole_dice_hard_raw_best_thr_score`: 0.000000
- `val_whole_dice_hard_raw_thr_0p30`: 0.000000
- `val_whole_dice_hard_raw_thr_0p40`: 0.000000
- `val_whole_dice_hard_raw_thr_0p50`: 0.000000
- `val_whole_dice_hard_raw_thr_0p60`: 0.000000
- `val_whole_dice_hard_raw_thr_0p70`: 0.000000
- `val_whole_dice_hard_thr_0p30`: 0.000000
- `val_whole_dice_hard_thr_0p40`: 0.000000
- `val_whole_dice_hard_thr_0p50`: 0.000000
- `val_whole_dice_hard_thr_0p60`: 0.000000
- `val_whole_dice_hard_thr_0p70`: 0.000000
- `val_whole_dice_micro`: 0.001697

## Best Metrics (Epoch, Value)
- `dice_coefficient`: epoch 85 -> 0.004572
- `loss`: epoch 62 -> 0.996902
- `safe_binary_iou`: epoch 122 -> 0.282946
- `val_dice_coefficient`: epoch 78 -> 0.006119
- `val_whole_dice_hard`: epoch 0 -> 0.000000
- `val_whole_dice_hard_best_thr`: epoch 0 -> 0.300000
- `val_whole_dice_hard_best_thr_score`: epoch 0 -> 0.000000
- `val_whole_dice_hard_brainmask`: epoch 0 -> 0.000000
- `val_whole_dice_hard_brainmask_best_thr`: epoch 0 -> 0.300000
- `val_whole_dice_hard_brainmask_best_thr_score`: epoch 0 -> 0.000000
- `val_whole_dice_hard_brainmask_delta`: epoch 0 -> 0.000000
- `val_whole_dice_hard_brainmask_thr_0p30`: epoch 0 -> 0.000000
- `val_whole_dice_hard_brainmask_thr_0p40`: epoch 0 -> 0.000000
- `val_whole_dice_hard_brainmask_thr_0p50`: epoch 0 -> 0.000000
- `val_whole_dice_hard_brainmask_thr_0p60`: epoch 0 -> 0.000000
- `val_whole_dice_hard_brainmask_thr_0p70`: epoch 0 -> 0.000000
- `val_whole_dice_hard_postproc_delta`: epoch 0 -> 0.000000
- `val_whole_dice_hard_raw`: epoch 0 -> 0.000000
- `val_whole_dice_hard_raw_best_thr`: epoch 0 -> 0.300000
- `val_whole_dice_hard_raw_best_thr_score`: epoch 0 -> 0.000000
- `val_whole_dice_hard_raw_thr_0p30`: epoch 0 -> 0.000000
- `val_whole_dice_hard_raw_thr_0p40`: epoch 0 -> 0.000000
- `val_whole_dice_hard_raw_thr_0p50`: epoch 0 -> 0.000000
- `val_whole_dice_hard_raw_thr_0p60`: epoch 0 -> 0.000000
- `val_whole_dice_hard_raw_thr_0p70`: epoch 0 -> 0.000000
- `val_whole_dice_hard_thr_0p30`: epoch 0 -> 0.000000
- `val_whole_dice_hard_thr_0p40`: epoch 0 -> 0.000000
- `val_whole_dice_hard_thr_0p50`: epoch 0 -> 0.000000
- `val_whole_dice_hard_thr_0p60`: epoch 0 -> 0.000000
- `val_whole_dice_hard_thr_0p70`: epoch 0 -> 0.000000
- `val_whole_dice_micro`: epoch 78 -> 0.010036

## Source Best Soft Dice
- `ARC-combined-t1-raw-ab0d1794`: epoch 78 -> 0.009546
- `ATLAS-Images-f0d7431e`: epoch 78 -> 0.004954
- `Approx-Numeracy-Processed`: epoch 78 -> 0.006259

## Warnings
- Validation dice stayed very low (<0.02): likely training collapse or severe class/domain mismatch.

## Artifacts
- `training_log_csv`: `/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/ARC_ATLAS_Train_v4/runs/20260417_163116/callbacks/training_log.csv`
- `batch_metrics_csv`: `/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/ARC_ATLAS_Train_v4/runs/20260417_163116/callbacks/batch_metrics.csv`
- `epoch_metrics_jsonl`: `/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/ARC_ATLAS_Train_v4/runs/20260417_163116/callbacks/epoch_metrics.jsonl`
- `sampling_schedule_jsonl`: `/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/ARC_ATLAS_Train_v4/runs/20260417_163116/callbacks/sampling_schedule.jsonl`
- `whole_val_summary_jsonl`: `/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/ARC_ATLAS_Train_v4/runs/20260417_163116/callbacks/whole_val_summary.jsonl`
- `split_summary_json`: `/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/ARC_ATLAS_Train_v4/runs/20260417_163116/callbacks/diagnostics/split_summary.json`
- `split_cases_csv`: `/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/ARC_ATLAS_Train_v4/runs/20260417_163116/callbacks/diagnostics/split_cases.csv`
