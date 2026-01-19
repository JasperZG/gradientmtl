#!/bin/bash
# =============================================================================
# Submit parallel experiments to CSHL SLURM cluster
# =============================================================================
#
# Prerequisites (run locally first):
#   python scripts/run_local.py
#   - Pre-trains GNN model and generates gradient matrix
#   - Runs experiments 2, 6, 7, and baselines
#
# Then upload gradient matrix:
#   scp outputs/gradients/gnn_conflict_matrices.npz $USER@hpc:gradient/outputs/gradients/
#
# This script submits:
#   - Experiment 3: Transfer learning (792 jobs)
#   - Experiment 4: Task selection (20 jobs)
#   - Experiment 5: PCGrad validation (15 jobs)
#
# Total: 827 GPU jobs
#
# =============================================================================

set -e

SEED=${1:-42}

echo "============================================================"
echo "Submitting Parallel Experiments to CSHL"
echo "============================================================"
echo "Seed: $SEED"
echo "Date: $(date)"
echo ""

# Create directories
mkdir -p logs
mkdir -p outputs/gradients
mkdir -p outputs/transfer_learning
mkdir -p outputs/task_selection
mkdir -p outputs/pcgrad

# Check that we're in the project directory
if [ ! -f "scripts/experiment3_transfer_learning.py" ]; then
    echo "ERROR: Please run this script from the project root directory"
    exit 1
fi

# Check that gradient matrix exists
if [ ! -f "outputs/gradients/gnn_conflict_matrices.npz" ]; then
    echo "ERROR: Gradient matrix not found!"
    echo ""
    echo "Please run local experiments first:"
    echo "  python scripts/run_local.py"
    echo ""
    echo "Then upload the gradient matrix:"
    echo "  scp outputs/gradients/gnn_conflict_matrices.npz \$USER@hpc:gradient/outputs/gradients/"
    exit 1
fi

echo "[OK] Gradient matrix found"
echo ""

# Export seed for SLURM scripts
export SEED=$SEED

# =============================================================================
# Summary of experiments
# =============================================================================

echo "HPC Experiment Summary:"
echo "-----------------------"
echo ""
echo "Experiment 3: Transfer Learning Validation"
echo "  - 792 jobs (12 targets × 11 data regimes × 6 pretrain conditions)"
echo "  - Tests if G matrix predicts transfer success"
echo "  - ~30 min/job"
echo ""
echo "Experiment 4: Task Selection Algorithms"
echo "  - 20 jobs (4 methods × 5 budgets)"
echo "  - Tests greedy vs clustering vs diversity vs random"
echo "  - ~30 min/job"
echo ""
echo "Experiment 5: PCGrad Validation"
echo "  - 15 jobs (5 high-conflict + 5 synergistic + 5 random pairs)"
echo "  - Tests if PCGrad helps conflicting pairs"
echo "  - ~1 hour/job"
echo ""
echo "Total: 827 jobs"
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
# Submit Experiment 3: Transfer Learning
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
echo "  Experiment 5 (PCGrad):   $JOB_ID_EXP5"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  squeue -u \$USER | wc -l  # Count running jobs"
echo ""
echo "Check experiment logs:"
echo "  tail -f logs/exp3_${JOB_ID_EXP3}_*.out"
echo "  tail -f logs/exp4_${JOB_ID_EXP4}_*.out"
echo "  tail -f logs/exp5_${JOB_ID_EXP5}_*.out"
echo ""
echo "Cancel all jobs:"
echo "  scancel $JOB_ID_EXP3 $JOB_ID_EXP4 $JOB_ID_EXP5"
echo ""
echo "After completion, aggregate results:"
echo "  python scripts/experiment3_transfer_learning.py --aggregate --output-dir outputs/transfer_learning"
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
