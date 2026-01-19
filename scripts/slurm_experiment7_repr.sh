#!/bin/bash
# =============================================================================
# SLURM script for Experiment 7: Representation Generalization
# =============================================================================

#SBATCH --job-name=exp7_repr
#SBATCH --output=logs/exp7_repr_%j.out
#SBATCH --error=logs/exp7_repr_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu

set -e

echo "============================================================"
echo "Experiment 7: Representation Generalization"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

module load cuda/11.8 2>/dev/null || true
module load python/3.10 2>/dev/null || true

if command -v conda &> /dev/null; then
    conda activate gradient 2>/dev/null || true
fi

cd $SLURM_SUBMIT_DIR
mkdir -p outputs/representation

if [ ! -f "outputs/gradients/gnn_conflict_matrices.npz" ]; then
    echo "ERROR: GNN gradient matrix not found. Run pretrain first."
    exit 1
fi

python scripts/experiment7_representation.py \
    --gnn-matrix outputs/gradients/gnn_conflict_matrices.npz \
    --output-dir outputs/representation \
    --seed 42 \
    --epochs 100

echo ""
echo "Representation comparison complete! End time: $(date)"
