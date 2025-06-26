#!/usr/bin/env python3
"""
Test script for advanced SOTA components
Verify everything works before full training
"""

import os
import sys
import numpy as np
import tensorflow as tf
from datetime import datetime

# Add current directory to path
sys.path.append('.')

def test_environment():
    """Test environment setup"""
    print("🔍 TESTING ENVIRONMENT")
    print("="*50)
    
    # TensorFlow
    print(f"✅ TensorFlow: {tf.__version__}")
    
    # GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU: {gpus[0]}")
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print("✅ Memory growth enabled")
        except Exception as e:
            print(f"⚠️ Memory growth warning: {e}")
    else:
        print("❌ No GPU detected!")
        return False
    
    # Mixed precision
    try:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("✅ Mixed precision enabled")
    except Exception as e:
        print(f"❌ Mixed precision failed: {e}")
        return False
    
    return True

def test_advanced_model():
    """Test advanced model creation"""
    print("\n🏗️ TESTING ADVANCED MODEL")
    print("="*50)
    
    try:
        from advanced_sota_training import build_advanced_sota_model
        
        # Test model creation
        print("Building model...")
        model = build_advanced_sota_model(
            input_shape=(64, 64, 64, 1),  # Smaller for testing
            base_filters=16,  # Smaller for testing
            use_swin=True,
            deep_supervision=True
        )
        
        param_count = model.count_params()
        print(f"✅ Model created: {param_count:,} parameters")
        
        # Test forward pass
        dummy_input = np.random.random((1, 64, 64, 64, 1)).astype(np.float32)
        outputs = model(dummy_input, training=False)
        
        if isinstance(outputs, list):
            print(f"✅ Deep supervision: {len(outputs)} outputs")
            for i, output in enumerate(outputs):
                print(f"   Output {i}: {output.shape}")
        else:
            print(f"✅ Single output: {outputs.shape}")
        
        # Clean up
        del model
        tf.keras.backend.clear_session()
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_loss_functions():
    """Test advanced loss functions"""
    print("\n🎯 TESTING LOSS FUNCTIONS")
    print("="*50)
    
    try:
        from advanced_sota_training import topology_aware_loss, deep_supervision_loss
        
        # Create dummy data
        y_true = tf.random.uniform((2, 32, 32, 32, 1), 0, 1)
        y_pred = tf.random.uniform((2, 32, 32, 32, 1), 0, 1)
        
        # Test topology-aware loss
        topo_loss = topology_aware_loss(y_true, y_pred)
        print(f"✅ Topology-aware loss: {topo_loss.numpy():.4f}")
        
        # Test deep supervision loss
        y_pred_list = [y_pred, y_pred * 0.8, y_pred * 0.6]  # Simulate different scales
        ds_loss = deep_supervision_loss(y_true, y_pred_list)
        print(f"✅ Deep supervision loss: {ds_loss.numpy():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Loss function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_generator():
    """Test advanced data generator"""
    print("\n📊 TESTING DATA GENERATOR")
    print("="*50)
    
    try:
        from advanced_sota_training import AdvancedDataGenerator
        
        # Create dummy file pairs
        dummy_pairs = [
            ("/fake/path1.nii.gz", "/fake/mask1.nii.gz"),
            ("/fake/path2.nii.gz", "/fake/mask2.nii.gz"),
        ]
        
        # Test generator creation (won't load actual files)
        generator = AdvancedDataGenerator(
            dummy_pairs,
            batch_size=1,
            target_shape=(64, 64, 64),
            shuffle=True,
            augment=True,
            mixup_alpha=0.2
        )
        
        print(f"✅ Generator created: {len(generator)} batches")
        print(f"✅ Augmentation enabled: {generator.augment}")
        print(f"✅ Mixup alpha: {generator.mixup_alpha}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data generator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_augmentation():
    """Test augmentation functions"""
    print("\n🎲 TESTING AUGMENTATION")
    print("="*50)
    
    try:
        from advanced_sota_training import elastic_deformation_3d
        
        # Create dummy volume
        img = np.random.random((32, 32, 32)).astype(np.float32)
        mask = (np.random.random((32, 32, 32)) > 0.8).astype(np.float32)
        
        print(f"Original shapes: img {img.shape}, mask {mask.shape}")
        
        # Test elastic deformation
        img_def, mask_def = elastic_deformation_3d(img, mask, alpha=5, sigma=1)
        
        print(f"✅ Elastic deformation: {img_def.shape}, {mask_def.shape}")
        print(f"✅ Mask preservation: {np.sum(mask_def):.1f} vs {np.sum(mask):.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Augmentation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_scheduling():
    """Test learning rate scheduling"""
    print("\n📈 TESTING LR SCHEDULING")
    print("="*50)
    
    try:
        from advanced_sota_training import cosine_schedule_with_warmup
        
        # Test schedule for first 20 epochs
        epochs = list(range(20))
        lrs = [cosine_schedule_with_warmup(e, warmup_epochs=5, total_epochs=100) for e in epochs]
        
        print("✅ Learning rate schedule:")
        for i, lr in enumerate(lrs[:10]):
            print(f"   Epoch {i:2d}: {lr:.6f}")
        
        # Check warmup
        assert lrs[0] < lrs[4], "Warmup should increase LR"
        print("✅ Warmup working correctly")
        
        # Check cosine decay
        assert lrs[10] > lrs[19], "Should decay after warmup"
        print("✅ Cosine decay working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Scheduling test failed: {e}")
        return False

def test_dataset_access():
    """Test access to the real dataset"""
    print("\n📁 TESTING DATASET ACCESS")
    print("="*50)
    
    try:
        data_dir = "/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training"
        
        if not os.path.exists(data_dir):
            print(f"❌ Dataset directory not found: {data_dir}")
            return False
        
        images_dir = os.path.join(data_dir, "Images")
        masks_dir = os.path.join(data_dir, "Masks")
        
        if not os.path.exists(images_dir):
            print(f"❌ Images directory not found: {images_dir}")
            return False
            
        if not os.path.exists(masks_dir):
            print(f"❌ Masks directory not found: {masks_dir}")
            return False
        
        # Count files
        image_files = [f for f in os.listdir(images_dir) if f.endswith('.nii.gz')]
        mask_files = [f for f in os.listdir(masks_dir) if f.endswith('.nii.gz')]
        
        print(f"✅ Found {len(image_files)} images")
        print(f"✅ Found {len(mask_files)} masks")
        
        if len(image_files) >= 650:
            print("✅ Full dataset available")
        else:
            print(f"⚠️ Dataset smaller than expected: {len(image_files)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset access failed: {e}")
        return False

def run_full_test():
    """Run all tests"""
    print("🧪 ADVANCED SOTA COMPONENT TESTING")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    tests = [
        ("Environment", test_environment),
        ("Advanced Model", test_advanced_model),
        ("Loss Functions", test_loss_functions),
        ("Data Generator", test_data_generator),
        ("Augmentation", test_augmentation),
        ("LR Scheduling", test_scheduling),
        ("Dataset Access", test_dataset_access),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("🏁 TEST SUMMARY")
    print("="*70)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:8} {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Ready for advanced SOTA training!")
        print("\nNext steps:")
        print("  1. Submit job: sbatch scripts/advanced_sota_training.sh")
        print("  2. Monitor: tail -f logs/advanced_sota_*.out")
        print("  3. Target: >75% validation Dice coefficient")
        return True
    else:
        print(f"\n❌ {len(tests) - passed} TESTS FAILED!")
        print("🔧 Fix the issues above before training")
        return False

if __name__ == "__main__":
    success = run_full_test()
    exit(0 if success else 1)
