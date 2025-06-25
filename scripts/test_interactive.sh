#!/bin/bash
#SBATCH --job-name=test_setup
#SBATCH --output=logs/test_setup_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --partition=interactive

cd /mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota

module purge
module load gcc/9.3.0-5wu3
module load cuda/12.6.3-ziu7

eval "$(conda shell.bash hook)"
conda activate stroke_sota

export CUDA_VISIBLE_DEVICES=0

# Test GPU and model
python test_gpu.py

# Test data loading
python -c "
from data.data_loader import StrokeDataGenerator
import numpy as np

print('\nTesting data loader...')
gen = StrokeDataGenerator(
    data_dir='/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training_Split',
    batch_size=1,
    target_shape=(192, 224, 176),
    augment=False,
    cache_size=1
)

print(f'Found {len(gen)} batches')
if len(gen) > 0:
    X, y = gen[0]
    print(f'Loaded batch shape: {X.shape}')
    print(f'Lesion voxels: {np.sum(y)} ({100*np.sum(y)/y.size:.3f}%)')
"
