#!/usr/bin/env python3
"""
DEBUG SCRIPT - Find exact memory failure point
"""

import os
import sys
import traceback
import psutil
import gc
import nibabel as nib
import numpy as np
sys.path.append('.')

def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        'rss_gb': memory_info.rss / (1024**3),
        'vms_gb': memory_info.vms / (1024**3),
        'percent': process.memory_percent()
    }

def print_memory(step):
    """Print memory usage at each step"""
    mem = get_memory_usage()
    print(f"MEMORY at {step}: RSS={mem['rss_gb']:.2f}GB, VMS={mem['vms_gb']:.2f}GB, %={mem['percent']:.1f}%")

print("🔍 MEMORY DEBUG: Finding exact failure point...")
print_memory("START")

try:
    print("\n1. Testing imports...")
    from correct_full_training import (
        load_full_655_dataset, 
        Config, 
        configure_hardware, 
        setup_mixed_precision
    )
    from sklearn.model_selection import train_test_split
    print_memory("IMPORTS")
    
    print("\n2. Testing dataset loading...")
    all_pairs = load_full_655_dataset()
    print(f"✅ Dataset loaded: {len(all_pairs)} pairs")
    print_memory("DATASET_LOADED")
    
    print("\n3. Testing train/validation split...")
    train_pairs, val_pairs = train_test_split(
        all_pairs,
        test_size=Config.VALIDATION_SPLIT,
        random_state=42,
        shuffle=True
    )
    print(f"✅ Split: {len(train_pairs)} train, {len(val_pairs)} val")
    print_memory("SPLIT_DONE")
    
    print("\n4. Testing hardware setup...")
    setup_mixed_precision()
    strategy = configure_hardware()
    print("✅ Hardware configured")
    print_memory("HARDWARE_SETUP")
    
    print("\n5. Testing model building...")
    from working_sota_model import build_sota_model
    model = build_sota_model(input_shape=Config.INPUT_SHAPE, base_filters=Config.BASE_FILTERS)
    param_count = model.count_params()
    print(f"✅ Model built: {param_count:,} parameters")
    print_memory("MODEL_BUILT")
    
    print("\n6. Testing SINGLE file loading (no batch)...")
    # Test loading just ONE file to see memory impact
    img_path, mask_path = all_pairs[0]
    case_name = os.path.basename(img_path).split('_space')[0]
    print(f"Loading case: {case_name}")
    print_memory("BEFORE_SINGLE_LOAD")
    
    print("  6a. Loading image...")
    img_data = nib.load(img_path).get_fdata(dtype=np.float32)
    print(f"  Image shape: {img_data.shape}, Size: {img_data.nbytes / (1024**3):.3f} GB")
    print_memory("IMAGE_LOADED")
    
    print("  6b. Loading mask...")
    mask_data = nib.load(mask_path).get_fdata(dtype=np.float32)
    print(f"  Mask shape: {mask_data.shape}, Size: {mask_data.nbytes / (1024**3):.3f} GB")
    print_memory("MASK_LOADED")
    
    print("  6c. Testing resize...")
    from scipy.ndimage import zoom
    if img_data.shape != Config.INPUT_SHAPE[:-1]:
        factors = [t / s for t, s in zip(Config.INPUT_SHAPE[:-1], img_data.shape)]
        print(f"  Resize factors: {factors}")
        print_memory("BEFORE_RESIZE")
        
        img_resized = zoom(img_data, factors, order=1)
        print(f"  Resized image: {img_resized.shape}, Size: {img_resized.nbytes / (1024**3):.3f} GB")
        print_memory("IMAGE_RESIZED")
        
        mask_resized = zoom(mask_data, factors, order=1)
        print(f"  Resized mask: {mask_resized.shape}, Size: {mask_resized.nbytes / (1024**3):.3f} GB")
        print_memory("MASK_RESIZED")
        
        # Clean up
        del img_resized, mask_resized
        gc.collect()
        print_memory("AFTER_CLEANUP")
    
    # Clean up single file test
    del img_data, mask_data
    gc.collect()
    print_memory("SINGLE_FILE_CLEANUP")
    
    print("\n7. Testing THREE file loading (verification simulation)...")
    test_indices = [0, len(all_pairs)//2, len(all_pairs)-1]
    
    for i, idx in enumerate(test_indices):
        print(f"  Loading sample {i+1}/3 (index {idx})...")
        img_path, mask_path = all_pairs[idx]
        case_name = os.path.basename(img_path).split('_space')[0]
        
        print_memory(f"BEFORE_SAMPLE_{i+1}")
        
        # Load files
        img_data = nib.load(img_path).get_fdata(dtype=np.float32)
        mask_data = nib.load(mask_path).get_fdata(dtype=np.float32)
        lesion_size = np.sum(mask_data)
        
        print(f"    {case_name}: {img_data.shape}, {lesion_size:,.0f} lesion voxels")
        print_memory(f"SAMPLE_{i+1}_LOADED")
        
        # Immediate cleanup
        del img_data, mask_data
        gc.collect()
        print_memory(f"SAMPLE_{i+1}_CLEANED")
    
    print("\n8. Testing data generator creation...")
    from correct_full_training import CorrectAtlasDataGenerator
    print_memory("BEFORE_GENERATOR")
    
    # Create generator with minimal samples
    test_generator = CorrectAtlasDataGenerator(
        train_pairs[:2],  # Only 2 samples
        1,  # Batch size 1
        Config.INPUT_SHAPE[:-1], 
        shuffle=False
    )
    print("✅ Generator created")
    print_memory("GENERATOR_CREATED")
    
    print("\n9. Testing batch loading...")
    print_memory("BEFORE_BATCH")
    X_batch, y_batch = next(iter(test_generator))
    print(f"✅ Batch loaded: X={X_batch.shape}, y={y_batch.shape}")
    print_memory("BATCH_LOADED")
    
    print("\n🎉 ALL TESTS PASSED!")
    print("Memory usage stayed within limits")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print_memory("ERROR_POINT")
    print("\nFull traceback:")
    traceback.print_exc()
    
    print(f"\nSystem memory info:")
    mem = psutil.virtual_memory()
    print(f"Total: {mem.total / (1024**3):.1f} GB")
    print(f"Available: {mem.available / (1024**3):.1f} GB") 
    print(f"Used: {mem.used / (1024**3):.1f} GB")
    print(f"Percent: {mem.percent:.1f}%")
