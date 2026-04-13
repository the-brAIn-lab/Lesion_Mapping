# Held-out Evaluation Summary

**Run:** `20260410_174944`
**Per-case macro Dice (soft):** 0.5619
**Per-case macro Dice (hard @0.50):** 0.5623

## Correlations
- r(soft Dice, log10 lesion size): **0.553**
- r(soft Dice, hi_freq_energy): **-0.140**  | partial (| size): **-0.027**
- r(soft Dice, lap_var): **0.317**       | partial (| size): **0.089**
- r(soft Dice, score): **0.091**        | partial (| size): **0.032**

## Mean Dice by lesion-size bin
- **[0,2.5k)**: n=27, mean soft=0.3257, mean hard=0.3262
- **[2.5k,10k)**: n=20, mean soft=0.4991, mean hard=0.4996
- **[10k,30k)**: n=32, mean soft=0.5456, mean hard=0.5461
- **[30k,60k)**: n=35, mean soft=0.6481, mean hard=0.6483
- **[60k,+inf)**: n=24, mean soft=0.7761, mean hard=0.7762

## Cohort summary (by key prefix)
- **s** | n=138 | mean soft=0.5619

## Figures
- ![scatter_soft_vs_logvox.png](figs/scatter_soft_vs_logvox.png)
- ![scatter_soft_vs_hfe.png](figs/scatter_soft_vs_hfe.png)
- ![scatter_soft_vs_lapvar.png](figs/scatter_soft_vs_lapvar.png)
- ![scatter_soft_vs_score.png](figs/scatter_soft_vs_score.png)

## Sources
- Joined per-case CSV: `/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/ARC_ATLAS_Train_v3/runs/20260410_174944/test_eval/test_metrics_with_manifest_FIXED_20260413_112415.csv`
- Figures dir (if generated): `/home/rbielski/stroke_cleaned/ARC_ATLAS_Combined/ARC_ATLAS_Train_v3/runs/20260410_174944/test_eval/figs`