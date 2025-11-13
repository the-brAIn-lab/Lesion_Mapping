# Image Quality Metrics Report

_Run: 20251112_120355_

## Datasets

- **High-Res (test_hires)**  
  Path: `/home/rbielski/ARC/ds004884/derivatives/aggregates/t1w_with_masks/mni_1mm_ants_fixed/_standardized/_combined_manifests_2bins_normal/_combined_splits_70_30_global/test_hires`
- **Low-Res (test_lores)**  
  Path: `/home/rbielski/ARC/ds004884/derivatives/aggregates/t1w_with_masks/mni_1mm_ants_fixed/_standardized/_combined_manifests_2bins_normal/_combined_splits_70_30_global/test_lores`
- **Crude 2×2×5× (backTo1mm)**  
  Path: `/home/rbielski/ARC/ds004884/derivatives/aggregates/t1w_with_masks/mni_1mm_ants_fixed/_standardized/_combined_manifests_2bins_normal/_combined_splits_70_30_global/test_hires_crude_2x2x5x_backTo1mm`
- **In-Plane 2×2×1mm (backTo1mm)**  
  Path: `/home/rbielski/ARC/ds004884/derivatives/aggregates/t1w_with_masks/mni_1mm_ants_fixed/_standardized/_combined_manifests_2bins_normal/_combined_splits_70_30_global/test_hires_inplane_2x2x1mm_backTo1mm`
- **Motion (slice jitter)**  
  Path: `/home/rbielski/ARC/ds004884/derivatives/aggregates/t1w_with_masks/mni_1mm_ants_fixed/_standardized/_combined_manifests_2bins_normal/_combined_splits_70_30_global/test_hires_motion_slicejitter`
- **Reduced SNR (k-space v1)**  
  Path: `/home/rbielski/ARC/ds004884/derivatives/aggregates/t1w_with_masks/mni_1mm_ants_fixed/_standardized/_combined_manifests_2bins_normal/_combined_splits_70_30_global/test_hires_snr_kspace_v1`
- **Thick slice 1×1×5mm (backTo1mm)**  
  Path: `/home/rbielski/ARC/ds004884/derivatives/aggregates/t1w_with_masks/mni_1mm_ants_fixed/_standardized/_combined_manifests_2bins_normal/_combined_splits_70_30_global/test_hires_thickslice_1x1x5mm_backTo1mm`

---

## Percentile Summaries
```

=== Crude 2×2×5× (backTo1mm) — N=236 ===
  FWHM blur (mm, smaller=sharper): p10/50/90 = 7.661 / 8.640 / 9.751 | mean±sd = 8.641±0.798
  High-freq energy fraction (larger=sharper): p10/50/90 = 0.133 / 0.160 / 0.187 | mean±sd = 0.160±0.023
  Variance of Laplacian (larger=sharper): p10/50/90 = 0.000 / 1.841 / 2.446 | mean±sd = 1.269±1.097

=== High-Res (test_hires) — N=236 ===
  FWHM blur (mm, smaller=sharper): p10/50/90 = 6.763 / 7.834 / 8.894 | mean±sd = 7.790±0.822
  High-freq energy fraction (larger=sharper): p10/50/90 = 0.080 / 0.123 / 0.160 | mean±sd = 0.122±0.028
  Variance of Laplacian (larger=sharper): p10/50/90 = 0.000 / 1.416 / 2.563 | mean±sd = 1.181±1.080

=== In-Plane 2×2×1mm (backTo1mm) — N=236 ===
  FWHM blur (mm, smaller=sharper): p10/50/90 = 9.846 / 10.827 / 12.764 | mean±sd = 11.131±1.160
  High-freq energy fraction (larger=sharper): p10/50/90 = 0.056 / 0.063 / 0.073 | mean±sd = 0.064±0.007
  Variance of Laplacian (larger=sharper): p10/50/90 = 0.000 / 0.736 / 0.916 | mean±sd = 0.485±0.414

=== Low-Res (test_lores) — N=335 ===
  FWHM blur (mm, smaller=sharper): p10/50/90 = 7.531 / 8.675 / 10.264 | mean±sd = 8.841±1.109
  High-freq energy fraction (larger=sharper): p10/50/90 = 0.043 / 0.074 / 0.108 | mean±sd = 0.076±0.026
  Variance of Laplacian (larger=sharper): p10/50/90 = 0.000 / 0.781 / 1.679 | mean±sd = 0.734±0.702

=== Motion (slice jitter) — N=236 ===
  FWHM blur (mm, smaller=sharper): p10/50/90 = 7.537 / 8.538 / 9.889 | mean±sd = 8.582±0.931
  High-freq energy fraction (larger=sharper): p10/50/90 = 0.171 / 0.201 / 0.240 | mean±sd = 0.205±0.028
  Variance of Laplacian (larger=sharper): p10/50/90 = 0.000 / 2.637 / 3.451 | mean±sd = 1.800±1.552

=== Reduced SNR (k-space v1) — N=236 ===
  FWHM blur (mm, smaller=sharper): p10/50/90 = 14.160 / 15.539 / 17.664 | mean±sd = 15.821±1.456
  High-freq energy fraction (larger=sharper): p10/50/90 = 0.023 / 0.028 / 0.032 | mean±sd = 0.028±0.004
  Variance of Laplacian (larger=sharper): p10/50/90 = 0.000 / 0.406 / 0.549 | mean±sd = 0.282±0.243

=== Thick slice 1×1×5mm (backTo1mm) — N=236 ===
  FWHM blur (mm, smaller=sharper): p10/50/90 = 10.051 / 11.114 / 13.327 | mean±sd = 11.484±1.307
  High-freq energy fraction (larger=sharper): p10/50/90 = 0.061 / 0.071 / 0.082 | mean±sd = 0.071±0.008
  Variance of Laplacian (larger=sharper): p10/50/90 = 0.000 / 0.697 / 0.860 | mean±sd = 0.462±0.394
```

## Comparisons vs High-Res

### FWHM blur (mm, smaller=sharper)
- **Reduced SNR (k-space v1)**: mean=15.8208 (CI 15.5802..16.0670) vs base 7.7899; Δ%=+103.1%; Mann–Whitney U p=9.12e-47; Cliff’s δ=1.000
- **Thick slice 1×1×5mm (backTo1mm)**: mean=11.4844 (CI 11.2625..11.6996) vs base 7.7899; Δ%=+47.4%; Mann–Whitney U p=1.41e-46; Cliff’s δ=0.998
- **In-Plane 2×2×1mm (backTo1mm)**: mean=11.1314 (CI 10.9434..11.3291) vs base 7.7899; Δ%=+42.9%; Mann–Whitney U p=1.44e-46; Cliff’s δ=0.998
- **Low-Res (test_lores)**: mean=8.8407 (CI 8.6929..9.0011) vs base 7.7899; Δ%=+13.5%; Mann–Whitney U p=7.58e-18; Cliff’s δ=0.552
- **Crude 2×2×5× (backTo1mm)**: mean=8.6410 (CI 8.5093..8.7695) vs base 7.7899; Δ%=+10.9%; Mann–Whitney U p=2.65e-14; Cliff’s δ=0.530
- **Motion (slice jitter)**: mean=8.5815 (CI 8.4253..8.7372) vs base 7.7899; Δ%=+10.2%; Mann–Whitney U p=5.01e-11; Cliff’s δ=0.458

### High-freq energy fraction (larger=sharper)
- **Motion (slice jitter)**: mean=0.2055 (CI 0.2010..0.2102) vs base 0.1220; Δ%=+68.5%; Mann–Whitney U p=2.48e-44; Cliff’s δ=0.973
- **Crude 2×2×5× (backTo1mm)**: mean=0.1600 (CI 0.1562..0.1638) vs base 0.1220; Δ%=+31.2%; Mann–Whitney U p=1.63e-24; Cliff’s δ=0.712
- **Low-Res (test_lores)**: mean=0.0759 (CI 0.0722..0.0794) vs base 0.1220; Δ%=-37.8%; Mann–Whitney U p=9.87e-33; Cliff’s δ=-0.764
- **Thick slice 1×1×5mm (backTo1mm)**: mean=0.0714 (CI 0.0701..0.0727) vs base 0.1220; Δ%=-41.5%; Mann–Whitney U p=9.36e-38; Cliff’s δ=-0.894
- **In-Plane 2×2×1mm (backTo1mm)**: mean=0.0640 (CI 0.0630..0.0651) vs base 0.1220; Δ%=-47.5%; Mann–Whitney U p=2.37e-42; Cliff’s δ=-0.950
- **Reduced SNR (k-space v1)**: mean=0.0277 (CI 0.0272..0.0284) vs base 0.1220; Δ%=-77.2%; Mann–Whitney U p=9.12e-47; Cliff’s δ=-1.000

### Variance of Laplacian (larger=sharper)
- **Motion (slice jitter)**: mean=1.7998 (CI 1.5885..2.0047) vs base 1.1808; Δ%=+52.4%; Mann–Whitney U p=4.24e-09; Cliff’s δ=0.301
- **Crude 2×2×5× (backTo1mm)**: mean=1.2690 (CI 1.1221..1.4150) vs base 1.1808; Δ%=+7.5%; Mann–Whitney U p=0.261; Cliff’s δ=0.058
- **Low-Res (test_lores)**: mean=0.7343 (CI 0.6570..0.8118) vs base 1.1808; Δ%=-37.8%; Mann–Whitney U p=3.47e-07; Cliff’s δ=-0.241
- **In-Plane 2×2×1mm (backTo1mm)**: mean=0.4848 (CI 0.4291..0.5405) vs base 1.1808; Δ%=-58.9%; Mann–Whitney U p=5.28e-11; Cliff’s δ=-0.336
- **Thick slice 1×1×5mm (backTo1mm)**: mean=0.4616 (CI 0.4086..0.5152) vs base 1.1808; Δ%=-60.9%; Mann–Whitney U p=4.19e-11; Cliff’s δ=-0.338
- **Reduced SNR (k-space v1)**: mean=0.2819 (CI 0.2486..0.3136) vs base 1.1808; Δ%=-76.1%; Mann–Whitney U p=2.57e-11; Cliff’s δ=-0.342


## Comparisons vs Low-Res

### FWHM blur (mm, smaller=sharper)
- **Reduced SNR (k-space v1)**: mean=15.8208 (CI 15.5802..16.0670) vs base 8.8407; Δ%=+79.0%; Mann–Whitney U p=7.8e-55; Cliff’s δ=1.000
- **Thick slice 1×1×5mm (backTo1mm)**: mean=11.4844 (CI 11.2625..11.6996) vs base 8.8407; Δ%=+29.9%; Mann–Whitney U p=1.49e-43; Cliff’s δ=0.887
- **In-Plane 2×2×1mm (backTo1mm)**: mean=11.1314 (CI 10.9434..11.3291) vs base 8.8407; Δ%=+25.9%; Mann–Whitney U p=5.24e-41; Cliff’s δ=0.860
- **Crude 2×2×5× (backTo1mm)**: mean=8.6410 (CI 8.5093..8.7695) vs base 8.8407; Δ%=-2.3%; Mann–Whitney U p=0.258; Cliff’s δ=-0.073
- **Motion (slice jitter)**: mean=8.5815 (CI 8.4253..8.7372) vs base 8.8407; Δ%=-2.9%; Mann–Whitney U p=0.0696; Cliff’s δ=-0.116
- **High-Res (test_hires)**: mean=7.7899 (CI 7.6575..7.9259) vs base 8.8407; Δ%=-11.9%; Mann–Whitney U p=7.58e-18; Cliff’s δ=-0.552

### High-freq energy fraction (larger=sharper)
- **Motion (slice jitter)**: mean=0.2055 (CI 0.2010..0.2102) vs base 0.0759; Δ%=+170.7%; Mann–Whitney U p=8.53e-55; Cliff’s δ=1.000
- **Crude 2×2×5× (backTo1mm)**: mean=0.1600 (CI 0.1562..0.1638) vs base 0.0759; Δ%=+110.8%; Mann–Whitney U p=4.85e-53; Cliff’s δ=0.983
- **High-Res (test_hires)**: mean=0.1220 (CI 0.1172..0.1265) vs base 0.0759; Δ%=+60.7%; Mann–Whitney U p=9.87e-33; Cliff’s δ=0.764
- **Thick slice 1×1×5mm (backTo1mm)**: mean=0.0714 (CI 0.0701..0.0727) vs base 0.0759; Δ%=-5.9%; Mann–Whitney U p=0.29; Cliff’s δ=-0.068
- **In-Plane 2×2×1mm (backTo1mm)**: mean=0.0640 (CI 0.0630..0.0651) vs base 0.0759; Δ%=-15.6%; Mann–Whitney U p=3.47e-05; Cliff’s δ=-0.265
- **Reduced SNR (k-space v1)**: mean=0.0277 (CI 0.0272..0.0284) vs base 0.0759; Δ%=-63.4%; Mann–Whitney U p=7.88e-54; Cliff’s δ=-0.990

### Variance of Laplacian (larger=sharper)
- **Motion (slice jitter)**: mean=1.7998 (CI 1.5885..2.0047) vs base 0.7343; Δ%=+145.1%; Mann–Whitney U p=8.61e-13; Cliff’s δ=0.339
- **Crude 2×2×5× (backTo1mm)**: mean=1.2690 (CI 1.1221..1.4150) vs base 0.7343; Δ%=+72.8%; Mann–Whitney U p=2.42e-10; Cliff’s δ=0.300
- **High-Res (test_hires)**: mean=1.1808 (CI 1.0372..1.3179) vs base 0.7343; Δ%=+60.8%; Mann–Whitney U p=3.47e-07; Cliff’s δ=0.241
- **In-Plane 2×2×1mm (backTo1mm)**: mean=0.4848 (CI 0.4291..0.5405) vs base 0.7343; Δ%=-34.0%; Mann–Whitney U p=4.58e-06; Cliff’s δ=-0.217
- **Thick slice 1×1×5mm (backTo1mm)**: mean=0.4616 (CI 0.4086..0.5152) vs base 0.7343; Δ%=-37.1%; Mann–Whitney U p=5.4e-07; Cliff’s δ=-0.237
- **Reduced SNR (k-space v1)**: mean=0.2819 (CI 0.2486..0.3136) vs base 0.7343; Δ%=-61.6%; Mann–Whitney U p=3.89e-13; Cliff’s δ=-0.344
