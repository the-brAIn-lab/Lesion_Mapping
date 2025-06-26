#!/usr/bin/env python3
"""
Minimal memory test - uses CPU fallback if GPU OOM
"""

import os
import sys
import gc
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.ndimage import zoom

# Custom imports
sys.path.append('/mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota')
from models.losses import dice_loss, combined_loss, focal_loss

def force_cpu_mode():
    """Force TensorFlow to use CPU only"""
    print("⚠️  Forcing CPU mode...")
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.config.set_visible_devices([], 'GPU')

def setup_minimal_gpu():
    """Minimal GPU setup with CPU fallback"""
    try:
        # Try GPU first
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Test GPU allocation with a tiny tensor
            with tf.device('/GPU:0'):
                test_tensor = tf.constant([[1.0]])
                _ = tf.matmul(test_tensor, test_tensor)
            
            print("✅ GPU mode enabled")
            return 'GPU'
        else:
            print("⚠️  No GPU detected, using CPU")
            return 'CPU'
            
    except Exception as e:
        print(f"⚠️  GPU setup failed: {e}")
        print("Falling back to CPU mode...")
        force_cpu_mode()
        return 'CPU'

def test_minimal():
    """Minimal test with maximum memory efficiency"""
    print("=" * 60)
    print("MINIMAL MEMORY TEST")
    print("=" * 60)
    
    # Setup device
    device_type = setup_minimal_gpu()
    
    # Paths
    model_path = "/mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota/callbacks/sota_20250616_190015/best_model.h5"
    test_case = "sub-r048s014_ses-1"
    test_dir = "/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Testing_Split"
    image_path = f"{test_dir}/Images/{test_case}_space-MNI152NLin2009aSym_T1w.nii.gz"
    mask_path = f"{test_dir}/Masks/{test_case}_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz"
    
    try:
        # 1. Load model
        print(f"\n1. Loading model on {device_type}...")
        custom_objects = {
            'dice_loss': dice_loss,
            'combined_loss': combined_loss,
            'focal_loss': focal_loss
        }
        
        model = load_model(model_path, custom_objects=custom_objects, compile=False)
        print(f"   ✅ Model loaded: {model.count_params():,} parameters")
        
        # 2. Load image (smaller chunks)
        print(f"\n2. Loading image...")
        nii_img = nib.load(image_path)
        image_data = nii_img.get_fdata().astype(np.float32)
        original_shape = image_data.shape
        print(f"   Original: {original_shape}")
        
        # 3. Preprocess in smaller target size first (test with smaller volume)
        print(f"\n3. Preprocessing (using smaller target size)...")
        target_shape = (96, 112, 88)  # Half size for testing
        
        # Normalize
        image_data = (image_data - np.mean(image_data)) / (np.std(image_data) + 1e-8)
        
        # Resize
        zoom_factors = [target_shape[i] / original_shape[i] for i in range(3)]
        resized_image = zoom(image_data, zoom_factors, order=1, mode='constant', cval=0)
        
        # Add dimensions
        image_batch = resized_image[np.newaxis, ..., np.newaxis]
        print(f"   Preprocessed: {image_batch.shape}")
        
        # Clean up
        del image_data, resized_image
        gc.collect()
        
        # 4. Test prediction with small input first
        print(f"\n4. Testing prediction...")
        
        if device_type == 'GPU':
            device_str = '/GPU:0'
        else:
            device_str = '/CPU:0'
        
        # Clear session before prediction
        tf.keras.backend.clear_session()
        gc.collect()
        
        print(f"   Using device: {device_str}")
        
        with tf.device(device_str):
            # Create model with smaller input shape
            # Resize input to match smaller target
            if image_batch.shape[1:4] != (192, 224, 176):
                print(f"   Model expects: (192, 224, 176), got: {image_batch.shape[1:4]}")
                print(f"   Resizing to model input shape...")
                
                # Extract volume
                vol = image_batch[0, ..., 0]  # (96, 112, 88)
                
                # Resize to model input
                model_zoom = [192/96, 224/112, 176/88]
                vol_resized = zoom(vol, model_zoom, order=1, mode='constant', cval=0)
                
                # Recreate batch
                image_batch = vol_resized[np.newaxis, ..., np.newaxis]
                print(f"   Final input shape: {image_batch.shape}")
                
                del vol, vol_resized
                gc.collect()
            
            # Run prediction
            print(f"   Running prediction...")
            prediction = model.predict(image_batch, batch_size=1, verbose=0)
            
        print(f"   ✅ Prediction successful!")
        print(f"   Output shape: {prediction.shape}")
        print(f"   Output range: [{prediction.min():.4f}, {prediction.max():.4f}]")
        
        # 5. Basic postprocessing
        print(f"\n5. Postprocessing...")
        pred_volume = prediction[0, ..., 0]
        binary_pred = (pred_volume > 0.5).astype(np.uint8)
        
        print(f"   Predicted lesion voxels: {np.sum(binary_pred)}")
        
        # 6. Calculate rough metrics
        if os.path.exists(mask_path):
            print(f"\n6. Quick metrics...")
            true_mask = nib.load(mask_path).get_fdata().astype(np.uint8)
            
            # Resize true mask to match prediction
            true_zoom = [pred_volume.shape[i] / true_mask.shape[i] for i in range(3)]
            true_resized = zoom(true_mask, true_zoom, order=0, mode='constant', cval=0)
            true_binary = (true_resized > 0.5).astype(np.uint8)
            
            # Calculate Dice
            intersection = np.sum(binary_pred * true_binary)
            union = np.sum(binary_pred) + np.sum(true_binary)
            dice = (2.0 * intersection) / union if union > 0 else 0.0
            
            print(f"   True lesion voxels: {np.sum(true_binary)}")
            print(f"   Rough Dice score: {dice:.4f}")
        
        # 7. Cleanup and success
        del model, image_batch, prediction
        tf.keras.backend.clear_session()
        gc.collect()
        
        print(f"\n✅ SUCCESS! Minimal test completed on {device_type}")
        print(f"The model can run - now we can optimize for full resolution")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        
        # Try CPU fallback if GPU failed
        if device_type == 'GPU':
            print("\n🔄 Trying CPU fallback...")
            force_cpu_mode()
            # Recursively call with CPU
            tf.keras.backend.clear_session()
            gc.collect()
            return test_minimal()
        
        return False

def check_gpu_status():
    """Check current GPU status"""
    print("GPU Status Check:")
    print("-" * 30)
    
    # Check if CUDA is available
    print(f"CUDA available: {tf.test.is_built_with_cuda()}")
    print(f"GPU available: {tf.test.is_gpu_available()}")
    
    # List devices
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Physical GPUs: {len(gpus)}")
    
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
    
    # Check memory
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                used, total = line.split(', ')
                print(f"  GPU {i} Memory: {used}MB / {total}MB used")
    except:
        print("  Could not get GPU memory info")

if __name__ == "__main__":
    print("Starting minimal memory test...\n")
    
    # Check status first
    check_gpu_status()
    
    # Run test
    success = test_minimal()
    
    if success:
        print("\n🎉 Great! The model works. Now we can:")
        print("   1. Run with full resolution")
        print("   2. Process all test cases")
        print("   3. Generate complete results")
    else:
        print("\n❌ Still having issues. Let's debug further.")
