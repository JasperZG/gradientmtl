#!/bin/bash
# =============================================================================
# SLURM script for Experiment 2: SAR Validation
# =============================================================================
#
# Validates gradient conflicts against literature-documented relationships
#
# Prerequisites:
#   - Gradient matrix from pretrain job (gnn_conflict_matrices.npz)
#
# =============================================================================

#SBATCH --job-name=exp2_sar
#SBATCH --output=logs/exp2_sar_%j.out
#SBATCH --error=logs/exp2_sar_%j.err
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu

set -e

echo "============================================================"
echo "Experiment 2: SAR Validation"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

# Load modules
module load cuda/11.8 2>/dev/null || true
module load python/3.10 2>/dev/null || true

# Activate environment
if command -v conda &> /dev/null; then
    conda activate gradient 2>/dev/null || true
fi

cd $SLURM_SUBMIT_DIR

# Create output directory
mkdir -p outputs/sar_validation

# Check prerequisite
if [ ! -f "outputs/gradients/gnn_conflict_matrices.npz" ]; then
    echo "ERROR: Gradient matrix not found. Run pretrain first."
    exit 1
fi

# Run SAR validation
python scripts/experiment2_sar_validation.py \
    --gradient-matrix outputs/gradients/gnn_conflict_matrices.npz \
    --output-dir outputs/sar_validation

echo ""
echo "============================================================"
echo "SAR Validation Complete!"
echo "============================================================"
echo "End time: $(date)"
