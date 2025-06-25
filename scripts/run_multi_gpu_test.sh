#!/bin/bash
#SBATCH --job-name=multi_gpu_test       # Job name
#SBATCH --partition=interactive         # Partition (or 'batch' for longer jobs)
#SBATCH --nodes=1                       # Request 1 node
#SBATCH --gpus=2                        # Request 2 GPUs (adjust based on node config)
#SBATCH --time=00:10:00                 # Time limit (HH:MM:SS) - 10 minutes
#SBATCH --output=multi_gpu_test_%j.out  # Standard output and error log
#SBATCH --error=multi_gpu_test_%j.err   # Standard error log
#SBATCH --mail-type=END,FAIL            # Email notifications
#SBATCH --mail-user=rb194958e@your.email.com # Your email address (use your actual email)

# --- CORRECTED CONDA ACTIVATION ---
# Replace /path/to/your/miniconda3 with the actual path to your conda installation (e.g., /home/rb194958e/miniconda3)
CONDA_BASE=$(conda info --base) # This tries to get the path dynamically, often works
if [ -z "$CONDA_BASE" ]; then
    # Fallback if conda info --base doesn't work (e.g., if conda is not in PATH yet)
    # You might need to hardcode this like: CONDA_BASE="/home/rb194958e/miniconda3"
    # or CONDA_BASE="/mnt/beegfs/hellgate/home/rb194958e/miniconda3"
    # If you're unsure, run 'conda info --base' in your login shell and use that output.
    echo "Warning: Could not dynamically determine CONDA_BASE. Assuming /mnt/beegfs/hellgate/home/rb194958e/miniconda3. Please update if incorrect."
    CONDA_BASE="/mnt/beegfs/hellgate/home/rb194958e/miniconda3" # Adjust if your actual conda installation path is different
fi

# Source the conda.sh script directly to initialize conda for the current shell
source $CONDA_BASE/etc/profile.d/conda.sh

# Activate your specific Conda environment
conda activate tf215_env
# --- END CORRECTED CONDA ACTIVATION ---


# Navigate to your project directory using the correct absolute path
cd /mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota

# Run your Python script (it's in the current directory now)
echo "Starting multi_gpu_test.py on $(hostname) with \$CUDA_VISIBLE_DEVICES"
python multi_gpu_test.py

echo "Script finished."
