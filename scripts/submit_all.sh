#!/bin/bash
# =============================================================================
# Submit all gradient conflict experiments to SLURM
# =============================================================================
#
# Usage:
#   bash scripts/submit_all.sh [SEED]
#
# This submits:
#   - Experiment 3: Transfer learning (6 pairs as array job)
#   - Experiment 4: Task selection (3 modes as array job)
#
# =============================================================================

SEED=${1:-42}

echo "=========================================="
echo "Submitting Gradient Conflict Experiments"
echo "=========================================="
echo "Seed: $SEED"
echo ""

# Create logs directory
mkdir -p logs

# Check that we're in the project directory
if [ ! -f "scripts/experiment3_transfer_learning.py" ]; then
    echo "ERROR: Please run this script from the project root directory"
    exit 1
fi

# Export seed for SLURM scripts
export SEED=$SEED

# Submit Experiment 3: Transfer Learning (6 pairs)
echo "Submitting Experiment 3: Transfer Learning Validation"
echo "  6 transfer pairs as array job..."
JOB_ID_EXP3=$(sbatch --array=0-5 --parsable scripts/slurm_experiment3.sh)
echo "  Job ID: $JOB_ID_EXP3"

# Submit Experiment 4: Task Selection (3 modes)
echo ""
echo "Submitting Experiment 4: Task Selection"
echo "  3 modes (baseline, grouped, single) as array job..."
JOB_ID_EXP4=$(sbatch --array=0-2 --parsable scripts/slurm_experiment4.sh)
echo "  Job ID: $JOB_ID_EXP4"

echo ""
echo "=========================================="
echo "All experiments submitted!"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "Check logs:"
echo "  tail -f logs/exp3_${JOB_ID_EXP3}_*.out"
echo "  tail -f logs/exp4_${JOB_ID_EXP4}_*.out"
echo "=========================================="
