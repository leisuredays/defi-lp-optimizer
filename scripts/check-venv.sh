#!/bin/bash

# Check if conda environment exists for backend
CONDA_ENV_NAME="uniswap-v3-backend"

echo "Checking Python conda environment setup..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo ""
    echo "========================================="
    echo "WARNING: conda not found in PATH!"
    echo "========================================="
    echo ""
    echo "This project uses conda for Python dependency management."
    echo "Please install conda first:"
    echo ""
    echo "  Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    echo "  Anaconda: https://www.anaconda.com/download"
    echo ""
    echo "========================================="
    echo ""
    exit 0
fi

# Initialize conda for this shell session
eval "$(conda shell.bash hook 2>/dev/null)" || source /home/zekiya/miniconda3/etc/profile.d/conda.sh 2>/dev/null

# Check if the conda environment exists
if ! conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    echo ""
    echo "========================================="
    echo "WARNING: Conda environment '${CONDA_ENV_NAME}' not found!"
    echo "========================================="
    echo ""
    echo "This project requires a conda environment for backend dependencies."
    echo ""
    echo "To set up the conda environment, run:"
    echo ""
    echo "  cd backend"
    echo "  ./setup_env.sh"
    echo ""
    echo "Or manually:"
    echo "  cd backend"
    echo "  conda env create -f environment.yml"
    echo "  conda activate ${CONDA_ENV_NAME}"
    echo ""
    echo "========================================="
    echo ""
else
    echo "✓ Conda environment '${CONDA_ENV_NAME}' found"
fi
