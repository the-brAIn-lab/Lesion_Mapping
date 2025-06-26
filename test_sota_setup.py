import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPUs: {tf.config.list_physical_devices('GPU')}")

# Test data loading
from corrected_sota_training import load_image_ids, Config
ids = load_image_ids(Config.DATA_DIR)
print(f"Found {len(ids)} images")

# Test model building
from corrected_sota_training import build_sota_model
model = build_sota_model((128, 128, 128, 1), 32)
print(f"Model parameters: {model.count_params():,}")
