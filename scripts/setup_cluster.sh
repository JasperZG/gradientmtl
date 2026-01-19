#!/bin/bash
# =============================================================================
# Setup script for gradient conflict project on HPC cluster
# =============================================================================
#
# Usage:
#   bash scripts/setup_cluster.sh
#
# This script:
#   1. Creates a conda environment
#   2. Installs required packages
#   3. Verifies the installation
#
# =============================================================================

set -e

ENV_NAME="${1:-gradient}"

echo "=========================================="
echo "Setting up Gradient Conflict Project"
echo "=========================================="

# Load modules if available
module load python 2>/dev/null || true
module load cuda 2>/dev/null || true

# Check for conda
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install miniconda first."
    exit 1
fi

# Create conda environment
echo ""
echo "Creating conda environment: $ENV_NAME"

conda create -n $ENV_NAME python=3.10 -y
conda activate $ENV_NAME

# Install PyTorch with CUDA support
echo ""
echo "Installing PyTorch..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install PyTorch Geometric
echo ""
echo "Installing PyTorch Geometric..."
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Install other dependencies
echo ""
echo "Installing other dependencies..."
pip install rdkit
pip install numpy pandas scikit-learn scipy matplotlib seaborn tqdm

# Verify installation
echo ""
echo "Verifying installation..."

python << 'EOF'
import sys
print(f"Python: {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

import torch_geometric
print(f"PyTorch Geometric: {torch_geometric.__version__}")

from rdkit import Chem
print("RDKit: OK")

print("\nAll dependencies installed successfully!")
EOF

echo ""
echo "=========================================="
echo "Setup complete!"
echo ""
echo "To activate the environment:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To run experiments:"
echo "  sbatch --array=0-5 scripts/slurm_experiment3.sh"
echo "  sbatch --array=0-2 scripts/slurm_experiment4.sh"
echo "=========================================="
