#!/usr/bin/env python3
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_validation_samples(epoch):
    val_dir = Path("/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training/Masks")
    pred_file = Path(f"resumed_pred_val_epoch_{epoch+1}.png")  # Corrected filename
    
    if not pred_file.exists():
        print(f"❌ Prediction visual missing for epoch {epoch+1}")
        return

    # Find matching mask sample
    sample_files = list(val_dir.glob("*"))
    if not sample_files:
        print("❌ No validation samples found!")
        return
        
    mask_path = sample_files[0]  # Use first sample
    mask = nib.load(mask_path).get_fdata()
    
    # Analyze lesion characteristics
    lesion_mask = (mask > 0.5).astype(int)
    lesion_voxels = np.sum(lesion_mask)
    
    print(f"📊 Epoch {epoch+1} Validation Sample Analysis:")
    print(f"  Sample: {mask_path.name}")
    print(f"  Total voxels: {mask.size}")
    print(f"  Lesion voxels: {lesion_voxels} ({lesion_voxels/mask.size:.4%})")
    
    # Visual comparison
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Middle slice of ground truth
    slice_idx = mask.shape[2] // 2
    ax[0].imshow(mask[:, :, slice_idx], cmap='jet', vmin=0, vmax=1)
    ax[0].set_title(f'Epoch {epoch+1} Ground Truth')
    
    # Prediction visualization
    pred_img = plt.imread(pred_file)
    ax[1].imshow(pred_img)
    ax[1].set_title(f'Epoch {epoch+1} Prediction')
    
    plt.savefig(f"validation_comparison_epoch_{epoch+1}.png", bbox_inches='tight')
    print(f"✅ Saved comparison: validation_comparison_epoch_{epoch+1}.png")

# Analyze recent epochs
for epoch in [27, 28, 29, 30, 31, 32, 33]:
    analyze_validation_samples(epoch)
