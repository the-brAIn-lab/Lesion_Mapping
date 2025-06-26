#!/bin/bash
#SBATCH --job-name=test_sota
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
source activate tf215_env
python test_sota_model.py | tee logs/test_sota_20250618_fixed6.log
