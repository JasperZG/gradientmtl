#!/bin/bash
# =============================================================================
# Submit ALL gradient conflict experiments to CSHL SLURM cluster
# =============================================================================
#
# Usage:
#   bash scripts/submit_all_cshl.sh [SEED]
#
# This submits:
#   - Experiment 3: Full transfer learning matrix (792 jobs)
#   - Experiment 4: Task selection algorithms (20 jobs)
#   - Experiment 5: PCGrad validation (15 jobs)
#
# Total: 827 GPU jobs
# Estimated time with CSHL parallelization: ~9 hours
#
# =============================================================================

set -e

SEED=${1:-42}

echo "============================================================"
echo "Submitting Full Gradient Conflict Experiment Suite to CSHL"
echo "============================================================"
echo "Seed: $SEED"
echo "Date: $(date)"
echo ""

# Create directories
mkdir -p logs
mkdir -p outputs/transfer_learning
mkdir -p outputs/task_selection
mkdir -p outputs/pcgrad

# Check that we're in the project directory
if [ ! -f "scripts/experiment3_transfer_learning.py" ]; then
    echo "ERROR: Please run this script from the project root directory"
    exit 1
fi

# Export seed for SLURM scripts
export SEED=$SEED

# =============================================================================
# Summary of experiments
# =============================================================================

echo "Experiment Summary:"
echo "-------------------"
echo ""
echo "Experiment 3: Transfer Learning Validation"
echo "  - 792 jobs (132 pairs × 3 data regimes × 2 conditions)"
echo "  - Tests if G matrix predicts transfer success"
echo "  - ~30 min/job, ~396 GPU-hours total"
echo ""
echo "Experiment 4: Task Selection Algorithms"
echo "  - 20 jobs (4 methods × 5 budgets)"
echo "  - Tests greedy vs clustering vs diversity vs random"
echo "  - ~2 hours/job, ~40 GPU-hours total"
echo ""
echo "Experiment 5: PCGrad Validation"
echo "  - 15 jobs (5 high-conflict + 5 synergistic + 5 random pairs)"
echo "  - Tests if PCGrad helps conflicting pairs"
echo "  - ~1 hour/job, ~15 GPU-hours total"
echo ""
echo "Total: 827 jobs, ~451 GPU-hours"
echo ""

# Prompt for confirmation
read -p "Submit all experiments? (y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "Submitting experiments..."
echo ""

# =============================================================================
# Submit Experiment 3: Full Transfer Learning Matrix
# =============================================================================

echo "=========================================="
echo "Experiment 3: Transfer Learning (792 jobs)"
echo "=========================================="

JOB_ID_EXP3=$(sbatch --parsable scripts/slurm_experiment3_full.sh)
echo "Submitted job array: $JOB_ID_EXP3"
echo "Array range: 0-791"

# =============================================================================
# Submit Experiment 4: Task Selection
# =============================================================================

echo ""
echo "=========================================="
echo "Experiment 4: Task Selection (20 jobs)"
echo "=========================================="

JOB_ID_EXP4=$(sbatch --parsable scripts/slurm_experiment4.sh)
echo "Submitted job array: $JOB_ID_EXP4"
echo "Array range: 0-19"

# =============================================================================
# Submit Experiment 5: PCGrad Validation
# =============================================================================

echo ""
echo "=========================================="
echo "Experiment 5: PCGrad Validation (15 jobs)"
echo "=========================================="

JOB_ID_EXP5=$(sbatch --parsable scripts/slurm_experiment5_pcgrad.sh)
echo "Submitted job array: $JOB_ID_EXP5"
echo "Array range: 0-14"

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "============================================================"
echo "All experiments submitted!"
echo "============================================================"
echo ""
echo "Job IDs:"
echo "  Experiment 3 (Transfer): $JOB_ID_EXP3"
echo "  Experiment 4 (Selection): $JOB_ID_EXP4"
echo "  Experiment 5 (PCGrad):    $JOB_ID_EXP5"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  squeue -u \$USER | wc -l  # Count running jobs"
echo ""
echo "Check logs:"
echo "  tail -f logs/exp3_${JOB_ID_EXP3}_*.out"
echo "  tail -f logs/exp4_${JOB_ID_EXP4}_*.out"
echo "  tail -f logs/exp5_${JOB_ID_EXP5}_*.out"
echo ""
echo "Cancel all jobs:"
echo "  scancel $JOB_ID_EXP3 $JOB_ID_EXP4 $JOB_ID_EXP5"
echo ""
echo "After completion, aggregate results:"
echo "  python scripts/experiment3_transfer_learning.py --aggregate"
echo "  python scripts/experiment4_task_selection.py --aggregate"
echo "  python scripts/experiment5_pcgrad.py --aggregate"
echo ""
echo "============================================================"

# Save job IDs for later reference
echo "SEED=$SEED" > logs/submitted_jobs.txt
echo "EXP3_JOB_ID=$JOB_ID_EXP3" >> logs/submitted_jobs.txt
echo "EXP4_JOB_ID=$JOB_ID_EXP4" >> logs/submitted_jobs.txt
echo "EXP5_JOB_ID=$JOB_ID_EXP5" >> logs/submitted_jobs.txt
echo "SUBMIT_TIME=$(date)" >> logs/submitted_jobs.txt

echo "Job IDs saved to logs/submitted_jobs.txt"
