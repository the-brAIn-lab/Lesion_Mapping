
#!/usr/bin/env python3
"""
Refactored training script for stroke lesion segmentation using ATLAS dataset.
Supports single-GPU training with reduced memory usage.
"""

import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import logging
import tensorflow as tf
import numpy as np
import nibabel as nib
from pathlib import Path
from datetime import datetime
from scipy.ndimage import zoom
from working_sota_model import build_sota_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/sota_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    DATA_DIR = "/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training_Split"
    INPUT_SHAPE = (192, 224, 176, 1)
    BATCH_SIZE = 16
    EPOCHS = 50
    BASE_FILTERS = 32
    VALIDATION_SPLIT = 0.2
    LEARNING_RATE = 1e-4
    GRADIENT_ACCUM_STEPS = 2
    CALLBACKS_DIR = lambda timestamp: Path(f'callbacks/sota_{timestamp}')
    MODEL_SAVE_PATH = lambda timestamp: f'models/sota_final_{timestamp}.h5'

def configure_hardware():
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Detected {len(gpus)} GPU(s), using 1")
            strategy = tf.distribute.get_strategy()
            global_batch_size = 1
        else:
            logger.warning("No GPUs detected, using CPU")
            strategy = tf.distribute.get_strategy()
            global_batch_size = 1
        return strategy, global_batch_size
    except Exception as e:
        logger.error(f"Hardware configuration failed: {e}, falling back to CPU")
        return tf.distribute.get_strategy(), 1

def setup_mixed_precision():
    pass  # Disabled

def resize_volume(volume, target_shape):
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=1)

def combined_loss(y_true, y_pred, smooth=1e-6, focal_gamma=3.0, focal_alpha=0.25):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice_loss = 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    alpha_t = tf.where(tf.equal(y_true, 1), focal_alpha, 1 - focal_alpha)
    focal_loss = -alpha_t * tf.pow(1 - p_t, focal_gamma) * tf.math.log(p_t)
    focal_loss = tf.reduce_mean(focal_loss)
    return 0.7 * dice_loss + 0.3 * focal_loss

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred, tf.float32))
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def binary_dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Binary Dice with threshold 0.5"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

class AtlasDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_dir, image_ids, batch_size, target_shape, shuffle=True):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "Images")
        self.masks_dir = os.path.join(data_dir, "Masks")
        self.image_ids = image_ids
        self.batch_size = batch_size
        self.target_shape = target_shape
        self.shuffle = shuffle
        self.indexes = np.arange(len(image_ids))
        logger.info(f"Initialized generator with {len(image_ids)} samples")
        if shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return len(self.image_ids) // self.batch_size

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X = np.zeros((self.batch_size, *self.target_shape, 1), dtype=np.float32)
        y = np.zeros((self.batch_size, *self.target_shape, 1), dtype=np.float32)
        for i, idx in enumerate(batch_indexes):
            try:
                img_id = self.image_ids[idx]
                img_path = os.path.join(self.images_dir, f"{img_id}_space-MNI152NLin2009aSym_T1w.nii.gz")
                mask_path = os.path.join(self.masks_dir, f"{img_id}_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz")
                img_data = nib.load(img_path).get_fdata(dtype=np.float32)
                mask_data = nib.load(mask_path).get_fdata(dtype=np.float32)
                logger.info(f"Loaded {img_id}: img_shape={img_data.shape}, mask_shape={mask_data.shape}")
                if img_data.shape != self.target_shape:
                    logger.info(f"Resizing {img_id} from {img_data.shape} to {self.target_shape}")
                    img_data = resize_volume(img_data, self.target_shape)
                    mask_data = resize_volume(mask_data, self.target_shape)
                    mask_data = (mask_data > 0.5).astype(np.float32)
                logger.info(f"Post-resize {img_id}: img_shape={img_data.shape}, mask_shape={mask_data.shape}")
                p1, p99 = np.percentile(img_data[img_data > 0], [1, 99])
                img_data = np.clip(img_data, p1, p99)
                img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min() + 1e-8)
                X[i] = img_data[..., np.newaxis]
                y[i] = mask_data[..., np.newaxis]
            except Exception as e:
                logger.error(f"Error loading {img_id}: {e}")
                X[i] = np.zeros((*self.target_shape, 1), dtype=np.float32)
                y[i] = np.zeros((*self.target_shape, 1), dtype=np.float32)
        logger.info(f"Batch shape: X={X.shape}, y={y.shape}")
        if len(X.shape) != 5 or len(y.shape) != 5:
            raise ValueError(f"Invalid batch shape: X={X.shape}, y={y.shape}, expected 5D")
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

def load_image_ids(data_dir):
    images_dir = os.path.join(data_dir, "Images")
    masks_dir = os.path.join(data_dir, "Masks")
    image_ids = [
        fname.replace("_space-MNI152NLin2009aSym_T1w.nii.gz", "") for fname in os.listdir(images_dir)
        if fname.endswith("_space-MNI152NLin2009aSym_T1w.nii.gz") and
        os.path.exists(os.path.join(masks_dir, f"{fname.replace('_space-MNI152NLin2009aSym_T1w.nii.gz', '')}_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz"))
    ]
    logger.info(f"Found {len(image_ids)} valid image/mask pairs")
    return image_ids

def create_callbacks(callbacks_dir):
    callbacks_dir.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            str(callbacks_dir / 'best_model.h5'),
            monitor='val_dice_coefficient',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_dice_coefficient',
            mode='max',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(str(callbacks_dir / 'training_log.csv')),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(callbacks_dir / 'tensorboard'),
            histogram_freq=0,
            write_graph=False,
            update_freq='epoch'
        )
    ]

def train_model():
    logger.info("Starting Stroke Lesion Segmentation Training")
    setup_mixed_precision()
    strategy, global_batch_size = configure_hardware()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    image_ids = load_image_ids(Config.DATA_DIR)
    np.random.shuffle(image_ids)
    train_size = int((1 - Config.VALIDATION_SPLIT) * len(image_ids))
    train_ids = image_ids[:train_size]
    val_ids = image_ids[train_size:]
    train_generator = AtlasDataGenerator(
        Config.DATA_DIR, train_ids, global_batch_size, Config.INPUT_SHAPE[:-1], shuffle=True
    )
    val_generator = AtlasDataGenerator(
        Config.DATA_DIR, val_ids, global_batch_size, Config.INPUT_SHAPE[:-1], shuffle=False
    )
    logger.info(f"Training samples: {len(train_ids)}")
    logger.info(f"Validation samples: {len(val_ids)}")
    with strategy.scope():
        model = build_sota_model(input_shape=Config.INPUT_SHAPE, base_filters=Config.BASE_FILTERS)
        logger.info(f"Model input shape: {model.input_shape}, output shape: {model.output_shape}")
        optimizer = tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss=combined_loss,
            metrics=[dice_coefficient, binary_dice_coefficient, 'accuracy']
        )
    try:
        X, y = next(iter(train_generator))
        logger.info(f"Validating model with batch: X={X.shape}, y={y.shape}")
        pred = model.predict(X)
        logger.info(f"Model prediction shape: {pred.shape}")
        with tf.GradientTape() as tape:
            pred = model(X, training=True)
            loss = combined_loss(y, pred)
        grads = tape.gradient(loss, model.trainable_variables)
        logger.info(f"Gradient shapes: {[g.shape if g is not None else None for g in grads]}")
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        raise
    try:
        model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=Config.EPOCHS,
            callbacks=create_callbacks(Config.CALLBACKS_DIR(timestamp)),
            verbose=1
        )
        logger.info("Training completed successfully")
        model.save(Config.MODEL_SAVE_PATH(timestamp))
        logger.info(f"Model saved to: {Config.MODEL_SAVE_PATH(timestamp)}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    try:
        train_model()
        logger.info("Training pipeline executed successfully")
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise SystemExit(1)

