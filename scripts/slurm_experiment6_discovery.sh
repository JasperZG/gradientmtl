#!/bin/bash
# =============================================================================
# SLURM script for Experiment 6: Novel Trade-off Discovery
# =============================================================================

#SBATCH --job-name=exp6_discovery
#SBATCH --output=logs/exp6_discovery_%j.out
#SBATCH --error=logs/exp6_discovery_%j.err
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu

set -e

echo "============================================================"
echo "Experiment 6: Novel Trade-off Discovery"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

module load cuda/11.8 2>/dev/null || true
module load python/3.10 2>/dev/null || true

if command -v conda &> /dev/null; then
    conda activate gradient 2>/dev/null || true
fi

cd $SLURM_SUBMIT_DIR
mkdir -p outputs/novel_discovery

if [ ! -f "outputs/gradients/gnn_conflict_matrices.npz" ]; then
    echo "ERROR: Gradient matrix not found. Run pretrain first."
    exit 1
fi

python scripts/experiment6_novel_discovery.py \
    --gradient-matrix outputs/gradients/gnn_conflict_matrices.npz \
    --output-dir outputs/novel_discovery \
    --conflict-threshold -0.3

echo ""
echo "Discovery complete! End time: $(date)"
