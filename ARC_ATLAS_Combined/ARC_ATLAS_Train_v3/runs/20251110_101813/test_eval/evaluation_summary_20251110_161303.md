# Held-out Evaluation Summary

**Run:** `20251110_101813`
**Per-case macro Dice (soft):** 0.5888
**Per-case macro Dice (hard @0.50):** 0.5895

## Correlations
- r(soft Dice, log10 lesion size): **0.641**
- r(soft Dice, hi_freq_energy): **-0.195**  | partial (| size): **-0.075**
- r(soft Dice, lap_var): **0.304**       | partial (| size): **0.018**
- r(soft Dice, score): **0.019**        | partial (| size): **-0.071**

## Mean Dice by lesion-size bin
- **[0,2.5k)**: n=28, mean soft=0.2797, mean hard=0.2809
- **[2.5k,10k)**: n=20, mean soft=0.5299, mean hard=0.5310
- **[10k,30k)**: n=32, mean soft=0.6071, mean hard=0.6076
- **[30k,60k)**: n=34, mean soft=0.7103, mean hard=0.7107
- **[60k,+inf)**: n=24, mean soft=0.8020, mean hard=0.8022

## Cohort summary (by key prefix)
- **s** | n=138 | mean soft=0.5888

## Figures
- ![scatter_soft_vs_logvox.png](figs/scatter_soft_vs_logvox.png)
- ![scatter_soft_vs_hfe.png](figs/scatter_soft_vs_hfe.png)
- ![scatter_soft_vs_lapvar.png](figs/scatter_soft_vs_lapvar.png)
- ![scatter_soft_vs_score.png](figs/scatter_soft_vs_score.png)

## Sources
- Joined per-case CSV: `/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/ARC_ATLAS_Train_v3/runs/20251110_101813/test_eval/test_metrics_with_manifest_FIXED_20251110_160551.csv`
- Figures dir (if generated): `/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/ARC_ATLAS_Train_v3/runs/20251110_101813/test_eval/figs`