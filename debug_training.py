#!/usr/bin/env python3
"""
DEBUG version of training script to see the actual error
"""

import os
import sys
import traceback
sys.path.append('.')

from correct_full_training import (
    load_full_655_dataset, 
    Config, 
    configure_hardware, 
    setup_mixed_precision
)
from sklearn.model_selection import train_test_split

print("🔍 DEBUG: Testing each component...")

try:
    print("1. Testing dataset loading...")
    all_pairs = load_full_655_dataset()
    print(f"✅ Dataset loaded: {len(all_pairs)} pairs")
    
    print("2. Testing train/validation split...")
    train_pairs, val_pairs = train_test_split(
        all_pairs,
        test_size=Config.VALIDATION_SPLIT,
        random_state=42,
        shuffle=True
    )
    print(f"✅ Split successful: {len(train_pairs)} train, {len(val_pairs)} val")
    
    print("3. Testing hardware configuration...")
    setup_mixed_precision()
    strategy = configure_hardware()
    print("✅ Hardware configured")
    
    print("4. Testing model import...")
    from working_sota_model import build_sota_model
    print("✅ Model import successful")
    
    print("5. Testing model building...")
    model = build_sota_model(input_shape=Config.INPUT_SHAPE, base_filters=Config.BASE_FILTERS)
    param_count = model.count_params()
    print(f"✅ Model built: {param_count:,} parameters")
    
    print("6. Testing data generator import...")
    from correct_full_training import CorrectAtlasDataGenerator
    print("✅ Data generator import successful")
    
    print("7. Testing data generator creation...")
    train_generator = CorrectAtlasDataGenerator(
        train_pairs[:4],  # Test with just 4 samples
        Config.BATCH_SIZE, 
        Config.INPUT_SHAPE[:-1], 
        shuffle=True
    )
    print("✅ Train generator created")
    
    print("8. Testing data loading...")
    X_train, y_train = next(iter(train_generator))
    print(f"✅ Data loading successful: X={X_train.shape}, y={y_train.shape}")
    
    print("\n🎉 ALL COMPONENTS WORKING!")
    print("The issue must be in the verification function or model compilation.")
    
except Exception as e:
    print(f"\n❌ ERROR at step: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
