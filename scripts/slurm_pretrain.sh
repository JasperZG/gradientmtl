#!/bin/bash
# =============================================================================
# SLURM script for pre-training GNN model and generating gradient conflict matrix
# =============================================================================
#
# This MUST run before experiments 3 and 4, which depend on the gradient matrix.
#
# Outputs:
#   - outputs/gradients/gnn_conflict_matrices.npz
#   - outputs/checkpoints/best_tox21_gnn_model.pt
#
# =============================================================================

#SBATCH --job-name=pretrain_gnn
#SBATCH --output=logs/pretrain_%j.out
#SBATCH --error=logs/pretrain_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu

# Exit on error
set -e

echo "============================================================"
echo "Pre-training GNN Model for Gradient Conflict Matrix"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Load modules (adjust for your cluster)
module load cuda/11.8 2>/dev/null || true
module load python/3.10 2>/dev/null || true

# Activate conda environment if available
if command -v conda &> /dev/null; then
    conda activate gradient 2>/dev/null || true
fi

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Create output directories
mkdir -p outputs/gradients
mkdir -p outputs/checkpoints
mkdir -p outputs/raw_data

# Run pre-training
echo "Starting pre-training..."
echo ""

python train_tox21_gnn.py \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-3 \
    --encoder_type gcn \
    --min_tasks 10

echo ""
echo "============================================================"
echo "Pre-training complete!"
echo "============================================================"
echo "End time: $(date)"

# Verify outputs exist
if [ -f "outputs/gradients/gnn_conflict_matrices.npz" ]; then
    echo "SUCCESS: Gradient matrix generated"
    ls -la outputs/gradients/
else
    echo "ERROR: Gradient matrix not found!"
    exit 1
fi

if [ -f "outputs/checkpoints/best_gnn_model.pt" ]; then
    echo "SUCCESS: Model checkpoint saved"
    ls -la outputs/checkpoints/
else
    echo "WARNING: Model checkpoint not found (experiments may still work)"
fi

echo ""
echo "Ready to run experiments 3 and 4."
