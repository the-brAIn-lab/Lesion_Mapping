#!/bin/bash
#SBATCH --job-name=sota_working
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:a4500:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/sota_working_%j.out
#SBATCH --error=logs/sota_working_%j.err

cd /mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota

module load gcc/9.3.0-5wu3 cuda/12.6.3-ziu7
eval "$(conda shell.bash hook)" || true
conda activate tf215_env
export LD_LIBRARY_PATH="/mnt/beegfs/hellgate/home/rb194958e/.conda/envs/tf215_env/lib:$LD_LIBRARY_PATH"
export TF_ENABLE_ONEDNN_OPTS=0
export NIBABEL_NIFTI1_QFAC_CHECK=0

echo "Starting SOTA Training: $(date)"
python -u working_sota_training.py || {
    echo "Training failed, check logs/sota_working_$SLURM_JOB_ID.err"
    exit 1
}
echo "Completed: $(date)"
