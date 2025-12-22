#!/bin/bash

echo "========================================="
echo "Setting up Conda Environment"
echo "========================================="
echo ""

CONDA_ENV_NAME="uniswap-v3-backend"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found in PATH!"
    echo ""
    echo "Please install conda first:"
    echo "  Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    echo "  Anaconda: https://www.anaconda.com/download"
    echo ""
    echo "After installation, run:"
    echo "  conda init bash"
    echo "  source ~/.bashrc"
    exit 1
fi

# Initialize conda for this shell session
eval "$(conda shell.bash hook 2>/dev/null)" || source /home/zekiya/miniconda3/etc/profile.d/conda.sh 2>/dev/null

# Check if environment already exists
if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    echo "Conda environment '${CONDA_ENV_NAME}' already exists."
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping environment creation."
        echo ""
        echo "To activate the existing environment, run:"
        echo "  conda activate ${CONDA_ENV_NAME}"
        exit 0
    fi
    echo "Removing existing environment..."
    conda env remove -n ${CONDA_ENV_NAME} -y
fi

# Create conda environment from environment.yml
if [ -f "environment.yml" ]; then
    echo "Creating conda environment from environment.yml..."
    conda env create -f environment.yml

    if [ $? -eq 0 ]; then
        echo ""
        echo "========================================="
        echo "Setup Complete!"
        echo "========================================="
        echo ""
        echo "Conda environment '${CONDA_ENV_NAME}' created successfully."
        echo ""
        echo "To activate the environment:"
        echo "  conda activate ${CONDA_ENV_NAME}"
        echo ""
        echo "To deactivate:"
        echo "  conda deactivate"
        echo ""
        echo "To update the environment in the future:"
        echo "  conda env update -f environment.yml --prune"
        echo ""
        echo "Note: pip.conf enforces that pip can only install packages"
        echo "      inside conda/virtual environments."
        echo ""
    else
        echo ""
        echo "Error: Failed to create conda environment."
        echo "Please check environment.yml and try again."
        exit 1
    fi
else
    echo ""
    echo "Error: environment.yml not found."
    echo "Cannot create conda environment without environment.yml file."
    exit 1
fi
