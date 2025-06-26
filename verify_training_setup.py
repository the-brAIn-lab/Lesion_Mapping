#!/usr/bin/env python3
"""
VERIFICATION script - run this BEFORE the 12-hour training
Confirms everything is correct: data loading, model architecture, memory usage
"""

import os
import sys
import numpy as np
import nibabel as nib
import tensorflow as tf
from pathlib import Path
import glob
from sklearn.model_selection import train_test_split

# Import model architecture
sys.path.append('/mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota')
from working_sota_model import build_sota_model

def verify_gpu_memory():
    """Check GPU memory availability"""
    print("🔧 GPU MEMORY VERIFICATION")
    print("=" * 50)
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"✅ GPU configured: {gpus[0]}")
    else:
        print("❌ No GPU found!")
        return False
    
    # Test memory allocation
    try:
        test_tensor = tf.random.normal((1, 192, 224, 176, 1))
        print(f"✅ Test tensor created: {test_tensor.shape}")
        del test_tensor
        return True
    except Exception as e:
        print(f"❌ GPU memory test failed: {e}")
        return False

def verify_data_loading():
    """Verify we can load all 655 image/mask pairs correctly"""
    print("\n📂 DATA LOADING VERIFICATION")
    print("=" * 50)
    
    data_dir = "/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training"
    images_dir = os.path.join(data_dir, "Images")
    masks_dir = os.path.join(data_dir, "Masks")
    
    # Get all image files
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.nii.gz')])
    print(f"Found {len(image_files)} image files")
    
    # Check image/mask pairing
    all_pairs = []
    missing_masks = []
    
    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        
        # Convert image filename to mask filename
        mask_file = img_file.replace('_T1w.nii.gz', '_label-L_desc-T1lesion_mask.nii.gz')
        mask_path = os.path.join(masks_dir, mask_file)
        
        if os.path.exists(mask_path):
            all_pairs.append((img_path, mask_path))
        else:
            missing_masks.append(img_file)
    
    print(f"✅ Valid image/mask pairs: {len(all_pairs)}")
    
    if missing_masks:
        print(f"❌ Missing masks for {len(missing_masks)} images:")
        for missing in missing_masks[:5]:  # Show first 5
            print(f"   {missing}")
        return False, []
    
    if len(all_pairs) != 655:
        print(f"⚠️  Expected 655 pairs, got {len(all_pairs)}")
    
    return True, all_pairs

def verify_train_val_split(all_pairs):
    """Verify train/validation split"""
    print("\n🔄 TRAIN/VALIDATION SPLIT VERIFICATION")
    print("=" * 50)
    
    # Same split as training will use
    train_pairs, val_pairs = train_test_split(
        all_pairs, 
        test_size=0.2, 
        random_state=42,
        shuffle=True
    )
    
    print(f"Training samples: {len(train_pairs)} ({len(train_pairs)/len(all_pairs)*100:.1f}%)")
    print(f"Validation samples: {len(val_pairs)} ({len(val_pairs)/len(all_pairs)*100:.1f}%)")
    
    # Check for overlap (should be none)
    train_files = {os.path.basename(p[0]) for p in train_pairs}
    val_files = {os.path.basename(p[0]) for p in val_pairs}
    overlap = train_files.intersection(val_files)
    
    if overlap:
        print(f"❌ Overlap detected: {len(overlap)} files in both sets!")
        return False
    else:
        print("✅ No overlap between train and validation sets")
    
    return True

def test_data_loading_sample(all_pairs):
    """Test loading a few samples to verify preprocessing works"""
    print("\n🧪 SAMPLE DATA LOADING TEST")
    print("=" * 50)
    
    # Test loading 3 random samples
    test_indices = [0, len(all_pairs)//2, len(all_pairs)-1]
    
    for i, idx in enumerate(test_indices):
        img_path, mask_path = all_pairs[idx]
        case_name = os.path.basename(img_path).split('_space')[0]
        
        try:
            # Load image and mask
            img_data = nib.load(img_path).get_fdata(dtype=np.float32)
            mask_data = nib.load(mask_path).get_fdata(dtype=np.float32)
            
            # Check properties
            lesion_voxels = np.sum(mask_data)
            
            print(f"  Sample {i+1}: {case_name}")
            print(f"    Image shape: {img_data.shape}")
            print(f"    Mask shape: {mask_data.shape}")
            print(f"    Lesion voxels: {lesion_voxels:,.0f}")
            print(f"    Image range: [{img_data.min():.2f}, {img_data.max():.2f}]")
            
            if img_data.shape != mask_data.shape:
                print(f"    ❌ Shape mismatch!")
                return False
                
        except Exception as e:
            print(f"    ❌ Failed to load: {e}")
            return False
    
    print("✅ Sample data loading successful")
    return True

def verify_model_architecture():
    """Verify model architecture matches working model"""
    print("\n🏗️  MODEL ARCHITECTURE VERIFICATION")
    print("=" * 50)
    
    try:
        # Build model with same parameters as working model
        model = build_sota_model(input_shape=(192, 224, 176, 1), base_filters=8)
        
        param_count = model.count_params()
        print(f"Model parameters: {param_count:,}")
        
        # Check if it matches working model
        expected_params = 5695045  # Your working model
        if abs(param_count - expected_params) < 100000:  # Allow small variation
            print(f"✅ Parameter count matches working model (~{expected_params:,})")
        else:
            print(f"❌ Parameter mismatch! Expected ~{expected_params:,}, got {param_count:,}")
            return False
        
        # Test model compilation
        model.compile(optimizer='adam', loss='binary_crossentropy')
        print("✅ Model compilation successful")
        
        # Test forward pass
        test_input = tf.random.normal((1, 192, 224, 176, 1))
        test_output = model(test_input, training=False)
        print(f"✅ Test forward pass: {test_input.shape} → {test_output.shape}")
        
        del model, test_input, test_output
        tf.keras.backend.clear_session()
        
        return True
        
    except Exception as e:
        print(f"❌ Model verification failed: {e}")
        return False

def estimate_training_time(num_samples):
    """Estimate training time"""
    print("\n⏱️  TRAINING TIME ESTIMATION")
    print("=" * 50)
    
    batch_size = 2
    batches_per_epoch = num_samples // batch_size
    epochs = 50
    
    # Rough estimates based on your hardware
    seconds_per_batch = 3  # Conservative estimate for RTX 4500
    seconds_per_epoch = batches_per_epoch * seconds_per_batch
    hours_per_epoch = seconds_per_epoch / 3600
    total_hours = hours_per_epoch * epochs
    
    print(f"Batch size: {batch_size}")
    print(f"Batches per epoch: {batches_per_epoch}")
    print(f"Estimated time per epoch: {hours_per_epoch:.1f} hours")
    print(f"Estimated total training time: {total_hours:.1f} hours")
    
    if total_hours > 20:
        print("⚠️  Training may take longer than 24-hour SLURM limit!")
    
    return total_hours

def main():
    """Run all verification checks"""
    print("🔍 TRAINING SETUP VERIFICATION")
    print("=" * 70)
    print("Verifying EVERYTHING before 12-hour training...")
    print("=" * 70)
    
    checks_passed = 0
    total_checks = 6
    
    # 1. GPU Memory
    if verify_gpu_memory():
        checks_passed += 1
    
    # 2. Data Loading
    data_ok, all_pairs = verify_data_loading()
    if data_ok:
        checks_passed += 1
    else:
        print("❌ Cannot proceed without valid data!")
        return False
    
    # 3. Train/Val Split
    if verify_train_val_split(all_pairs):
        checks_passed += 1
    
    # 4. Sample Data Loading
    if test_data_loading_sample(all_pairs):
        checks_passed += 1
    
    # 5. Model Architecture
    if verify_model_architecture():
        checks_passed += 1
    
    # 6. Time Estimation
    train_samples = int(len(all_pairs) * 0.8)
    total_hours = estimate_training_time(train_samples)
    if total_hours < 24:
        checks_passed += 1
    
    # Final verdict
    print(f"\n{'='*70}")
    print(f"VERIFICATION RESULTS: {checks_passed}/{total_checks} PASSED")
    print(f"{'='*70}")
    
    if checks_passed == total_checks:
        print("🎉 ALL CHECKS PASSED!")
        print("✅ Ready to start 12-hour training")
        print(f"✅ Will train on {len(all_pairs)} total samples")
        print(f"✅ Train: {train_samples} samples, Val: {len(all_pairs) - train_samples} samples")
        print(f"✅ Model architecture verified ({checks_passed} checks)")
        return True
    else:
        print("❌ VERIFICATION FAILED!")
        print("🛑 DO NOT START TRAINING YET")
        print("Fix the issues above first")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
