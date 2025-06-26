#!/usr/bin/env python3
"""
Fixed ATLAS 2.0 Data Loader for 3D Volume Resizing
"""

import tensorflow as tf
import nibabel as nib
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, List, Generator
from scipy.ndimage import zoom

logger = logging.getLogger(__name__)

class RealAtlasDataLoader:
    def __init__(self, data_dir: str, target_shape: Tuple[int, int, int] = (192, 224, 176)):
        self.data_dir = Path(data_dir)
        self.target_shape = target_shape
        self.image_dir = self.data_dir / 'Images'
        self.mask_dir = self.data_dir / 'Masks'
        self.file_pairs = self._get_file_pairs()
        logger.info(f"Found {len(self.file_pairs)} image-mask pairs")
        
    def _get_file_pairs(self) -> List[Tuple[Path, Path]]:
        pairs = []
        image_files = list(self.image_dir.glob('*_T1w.nii.gz'))
        
        for img_file in image_files:
            base_name = img_file.name.replace('_T1w.nii.gz', '')
            mask_file = self.mask_dir / f"{base_name}_label-L_desc-T1lesion_mask.nii.gz"
            
            if mask_file.exists():
                pairs.append((img_file, mask_file))
        
        return pairs
    
    def _load_nifti_volume(self, file_path: Path) -> np.ndarray:
        try:
            nii = nib.load(str(file_path))
            volume = nii.get_fdata()
            
            if volume.ndim == 4:
                volume = volume[:, :, :, 0]
            elif volume.ndim != 3:
                return np.zeros(self.target_shape, dtype=np.float32)
            
            volume = np.nan_to_num(volume, nan=0.0)
            volume = self._resize_volume_3d(volume, self.target_shape)
            
            return volume.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return np.zeros(self.target_shape, dtype=np.float32)
    
    def _resize_volume_3d(self, volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        current_shape = volume.shape
        zoom_factors = [target_shape[i] / current_shape[i] for i in range(3)]
        resized_volume = zoom(volume, zoom_factors, order=1)
        
        # Ensure exact target shape
        if resized_volume.shape != target_shape:
            resized_volume = self._crop_or_pad_to_shape(resized_volume, target_shape)
        
        return resized_volume
    
    def _crop_or_pad_to_shape(self, volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        current_shape = volume.shape
        result = volume.copy()
        
        for i in range(3):
            current_size = current_shape[i]
            target_size = target_shape[i]
            
            if current_size > target_size:
                start = (current_size - target_size) // 2
                end = start + target_size
                
                if i == 0:
                    result = result[start:end, :, :]
                elif i == 1:
                    result = result[:, start:end, :]
                else:
                    result = result[:, :, start:end]
                    
            elif current_size < target_size:
                pad_before = (target_size - current_size) // 2
                pad_after = target_size - current_size - pad_before
                
                pad_width = [(0, 0), (0, 0), (0, 0)]
                pad_width[i] = (pad_before, pad_after)
                
                result = np.pad(result, pad_width, mode='constant', constant_values=0)
        
        return result
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        p1, p99 = np.percentile(image, [1, 99])
        image = np.clip(image, p1, p99)
        
        if image.max() > image.min():
            image = (image - image.min()) / (image.max() - image.min())
        
        return image
    
    def _preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        return (mask > 0).astype(np.float32)
    
    def data_generator(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        for img_path, mask_path in self.file_pairs:
            try:
                image = self._load_nifti_volume(img_path)
                mask = self._load_nifti_volume(mask_path)
                
                image = self._preprocess_image(image)
                mask = self._preprocess_mask(mask)
                
                image = image[..., np.newaxis]
                mask = mask[..., np.newaxis]
                
                yield image, mask
                
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                continue
    
    def create_dataset(self, batch_size: int = 2, shuffle: bool = True, validation_split: float = 0.2) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        dataset = tf.data.Dataset.from_generator(
            self.data_generator,
            output_signature=(
                tf.TensorSpec(shape=(*self.target_shape, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(*self.target_shape, 1), dtype=tf.float32)
            )
        )
        
        dataset_size = len(self.file_pairs)
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(100, dataset_size))
        
        val_size = int(dataset_size * validation_split)
        train_size = dataset_size - val_size
        
        train_dataset = dataset.take(train_size)
        val_dataset = dataset.skip(train_size)
        
        train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        logger.info(f"Training samples: {train_size}, Validation samples: {val_size}")
        
        return train_dataset, val_dataset

def create_real_data_loader(config: dict) -> RealAtlasDataLoader:
    data_dir = config['data']['data_dir']
    target_shape = tuple(config['model']['input_shape'][:-1])
    return RealAtlasDataLoader(data_dir, target_shape)
