#!/bin/bash
#SBATCH --job-name=conda_install_stroke_sota
#SBATCH --output=conda_install_stroke_sota_%j.log
#SBATCH --error=conda_install_stroke_sota_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4          # Request some CPU cores for Conda's solver
#SBATCH --mem=16G                  # Request more memory, e.g., 16GB
#SBATCH --time=0:30:00             # Give it 30 minutes
#SBATCH --partition=normal         # Use a regular compute partition

echo "--- $(date): Starting conda environment setup ---"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"

# Ensure base conda is initialized (if not already in your .bashrc/.profile)
eval "$(conda shell.bash hook)"

# Execute the conda install command
conda install -n stroke_sota python=3.11 \
    cudnn=8.9 \
    cuda-toolkit=12.2 \
    numpy scipy scikit-image matplotlib h5py nibabel \
    -c nvidia \
    -c conda-forge \
    --yes # Non-interactive installation

INSTALL_STATUS=$?

if [ $INSTALL_STATUS -eq 0 ]; then
    echo "--- $(date): Conda installation completed successfully ---"
    # Now install tensorflow with pip, as it often works better after conda sets up libs
    conda activate stroke_sota
    pip install tensorflow==2.15.1
    PIP_STATUS=$?
    if [ $PIP_STATUS -eq 0 ]; then
        echo "--- $(date): TensorFlow pip installation completed successfully ---"
        echo "Environment 'stroke_sota' should now be fully set up for TensorFlow GPU."
        exit 0
    else
        echo "--- $(date): TensorFlow pip installation FAILED with error code $PIP_STATUS ---"
        exit $PIP_STATUS
    fi
else
    echo "--- $(date): Conda installation FAILED with error code $INSTALL_STATUS ---"
    exit $INSTALL_STATUS
fi
