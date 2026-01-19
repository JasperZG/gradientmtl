#!/bin/bash
# =============================================================================
# SLURM script for Single-Task Baselines
# =============================================================================

#SBATCH --job-name=baselines
#SBATCH --output=logs/baselines_%j.out
#SBATCH --error=logs/baselines_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu

set -e

echo "============================================================"
echo "Single-Task Baselines"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

module load cuda/11.8 2>/dev/null || true
module load python/3.10 2>/dev/null || true

if command -v conda &> /dev/null; then
    conda activate gradient 2>/dev/null || true
fi

cd $SLURM_SUBMIT_DIR
mkdir -p outputs/baselines

python scripts/single_task_baselines.py \
    --output-dir outputs/baselines \
    --seed 42 \
    --epochs 100

echo ""
echo "Baselines complete! End time: $(date)"
