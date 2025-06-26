import sys
print("🔍 Verifying installations...\n")

packages = {
    'tensorflow': 'tf',
    'torch': 'torch',
    'monai': 'monai',
    'nibabel': 'nib',
    'SimpleITK': 'sitk',
    'numpy': 'np',
    'scipy': 'scipy',
    'skimage': 'skimage',
    'sklearn': 'sklearn',
    'matplotlib': 'plt',
    'einops': 'einops',
    'timm': 'timm'
}

failed = []
for package, import_name in packages.items():
    try:
        if package == 'tensorflow':
            import tensorflow as tf
            print(f"✅ TensorFlow: {tf.__version__}")
            print(f"   GPUs detected: {len(tf.config.list_physical_devices('GPU'))}")
        elif package == 'torch':
            import torch
            print(f"✅ PyTorch: {torch.__version__}")
            print(f"   CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"   GPU count: {torch.cuda.device_count()}")
        elif package == 'monai':
            import monai
            print(f"✅ MONAI: {monai.__version__}")
        elif package == 'nibabel':
            import nibabel as nib
            print(f"✅ Nibabel: {nib.__version__}")
        elif package == 'SimpleITK':
            import SimpleITK as sitk
            print(f"✅ SimpleITK: {sitk.Version.VersionString()}")
        elif package == 'skimage':
            import skimage
            print(f"✅ Scikit-image: {skimage.__version__}")
        elif package == 'sklearn':
            import sklearn
            print(f"✅ Scikit-learn: {sklearn.__version__}")
        else:
            exec(f"import {import_name}")
            print(f"✅ {package}: installed")
    except ImportError as e:
        failed.append(package)
        print(f"❌ {package}: FAILED - {e}")

if failed:
    print(f"\n⚠️  Failed packages: {', '.join(failed)}")
    print("Please try installing them individually")
    sys.exit(1)
else:
    print("\n✅ All packages installed successfully!")
    
# Test GPU memory
print("\n🔍 Testing GPU memory allocation...")
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("✅ TensorFlow GPU memory growth enabled")
except Exception as e:
    print(f"⚠️  GPU memory configuration warning: {e}")
