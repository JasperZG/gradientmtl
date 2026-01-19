#!/bin/bash
# =============================================================================
# Submit ALL gradient conflict experiments to CSHL SLURM cluster
# =============================================================================
#
# Usage:
#   bash scripts/submit_all_cshl.sh [SEED]
#
# Workflow:
#   1. Pre-train GNN model to generate gradient conflict matrix (1 job, ~2-4 hours)
#   2. After pretrain completes:
#      - Experiment 2: SAR validation (1 job)
#      - Experiment 3: Transfer learning (792 jobs)
#      - Experiment 4: Task selection (20 jobs)
#      - Experiment 6: Novel discovery (1 job)
#   3. Independent (no prerequisite):
#      - Experiment 5: PCGrad validation (15 jobs)
#      - Experiment 7: Representation generalization (2 jobs)
#      - Single-task baselines (12 jobs)
#
# Total: 844 GPU jobs
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
mkdir -p outputs/gradients
mkdir -p outputs/checkpoints
mkdir -p outputs/transfer_learning
mkdir -p outputs/task_selection
mkdir -p outputs/pcgrad
mkdir -p outputs/raw_data
mkdir -p outputs/sar_validation
mkdir -p outputs/novel_discovery
mkdir -p outputs/representation
mkdir -p outputs/baselines

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
echo "Step 0: Pre-train GNN Model (PREREQUISITE)"
echo "  - 1 job"
echo "  - Generates gradient conflict matrix (gnn_conflict_matrices.npz)"
echo "  - Required for experiments 2, 3, 4, and 6"
echo "  - ~2-4 hours"
echo ""
echo "Experiment 2: SAR Validation (depends on pretrain)"
echo "  - 1 job"
echo "  - Validates gradient conflicts against literature relationships"
echo "  - Expected: Pearson r > 0.6, p < 0.001"
echo "  - ~30 min"
echo ""
echo "Experiment 3: Transfer Learning Validation (depends on pretrain)"
echo "  - 792 jobs (12 targets × 11 data regimes × 6 pretrain conditions)"
echo "  - Tests if G matrix predicts transfer success"
echo "  - ~30 min/job"
echo ""
echo "Experiment 4: Task Selection Algorithms (depends on pretrain)"
echo "  - 20 jobs (4 methods × 5 budgets)"
echo "  - Tests greedy vs clustering vs diversity vs random"
echo "  - ~30 min/job"
echo ""
echo "Experiment 5: PCGrad Validation (NO prerequisite - runs immediately)"
echo "  - 15 jobs (5 high-conflict + 5 synergistic + 5 random pairs)"
echo "  - Tests if PCGrad helps conflicting pairs"
echo "  - ~1 hour/job"
echo ""
echo "Experiment 6: Novel Discovery (depends on pretrain)"
echo "  - 1 job"
echo "  - Discovers novel trade-offs not in literature"
echo "  - Expected: 3-5 novel relationships"
echo "  - ~30 min"
echo ""
echo "Experiment 7: Representation Generalization (NO prerequisite)"
echo "  - 2 jobs (ECFP vs GNN comparison)"
echo "  - Tests if gradient patterns are representation-invariant"
echo "  - Expected: Pearson r > 0.8 between G_ECFP and G_GNN"
echo "  - ~2 hours/job"
echo ""
echo "Single-Task Baselines (NO prerequisite - runs immediately)"
echo "  - 12 jobs (one per Tox21 task)"
echo "  - Establishes upper bound without negative transfer"
echo "  - ~1 hour/job"
echo ""
echo "Total: 844 jobs"
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
# Step 0: Pre-train GNN Model (generates gradient matrix)
# =============================================================================

echo "=========================================="
echo "Step 0: Pre-training GNN Model"
echo "=========================================="

JOB_ID_PRETRAIN=$(sbatch --parsable scripts/slurm_pretrain.sh)
echo "Submitted pretrain job: $JOB_ID_PRETRAIN"
echo "This must complete before experiments 3 and 4 can start."

# =============================================================================
# Submit Experiment 5: PCGrad Validation (NO dependency - starts immediately)
# =============================================================================

echo ""
echo "=========================================="
echo "Experiment 5: PCGrad Validation (15 jobs)"
echo "=========================================="
echo "(No prerequisite - starts immediately)"

JOB_ID_EXP5=$(sbatch --parsable scripts/slurm_experiment5_pcgrad.sh)
echo "Submitted job array: $JOB_ID_EXP5"
echo "Array range: 0-14"

# =============================================================================
# Submit Experiment 7: Representation Generalization (NO dependency)
# =============================================================================

echo ""
echo "=========================================="
echo "Experiment 7: Representation (2 jobs)"
echo "=========================================="
echo "(No prerequisite - starts immediately)"

JOB_ID_EXP7=$(sbatch --parsable scripts/slurm_experiment7_repr.sh)
echo "Submitted job array: $JOB_ID_EXP7"
echo "Array range: 0-1"

# =============================================================================
# Submit Single-Task Baselines (NO dependency - starts immediately)
# =============================================================================

echo ""
echo "=========================================="
echo "Single-Task Baselines (12 jobs)"
echo "=========================================="
echo "(No prerequisite - starts immediately)"

JOB_ID_BASELINES=$(sbatch --parsable scripts/slurm_baselines.sh)
echo "Submitted job: $JOB_ID_BASELINES"

# =============================================================================
# Submit Experiment 2: SAR Validation (depends on pretrain)
# =============================================================================

echo ""
echo "=========================================="
echo "Experiment 2: SAR Validation (1 job)"
echo "=========================================="
echo "(Waiting for pretrain job $JOB_ID_PRETRAIN to complete)"

JOB_ID_EXP2=$(sbatch --parsable --dependency=afterok:$JOB_ID_PRETRAIN scripts/slurm_experiment2_sar.sh)
echo "Submitted job: $JOB_ID_EXP2"
echo "Dependency: afterok:$JOB_ID_PRETRAIN"

# =============================================================================
# Submit Experiment 3: Transfer Learning (depends on pretrain)
# =============================================================================

echo ""
echo "=========================================="
echo "Experiment 3: Transfer Learning (792 jobs)"
echo "=========================================="
echo "(Waiting for pretrain job $JOB_ID_PRETRAIN to complete)"

JOB_ID_EXP3=$(sbatch --parsable --dependency=afterok:$JOB_ID_PRETRAIN scripts/slurm_experiment3_full.sh)
echo "Submitted job array: $JOB_ID_EXP3"
echo "Array range: 0-791"
echo "Dependency: afterok:$JOB_ID_PRETRAIN"

# =============================================================================
# Submit Experiment 4: Task Selection (depends on pretrain)
# =============================================================================

echo ""
echo "=========================================="
echo "Experiment 4: Task Selection (20 jobs)"
echo "=========================================="
echo "(Waiting for pretrain job $JOB_ID_PRETRAIN to complete)"

JOB_ID_EXP4=$(sbatch --parsable --dependency=afterok:$JOB_ID_PRETRAIN scripts/slurm_experiment4.sh)
echo "Submitted job array: $JOB_ID_EXP4"
echo "Array range: 0-19"
echo "Dependency: afterok:$JOB_ID_PRETRAIN"

# =============================================================================
# Submit Experiment 6: Novel Discovery (depends on pretrain)
# =============================================================================

echo ""
echo "=========================================="
echo "Experiment 6: Novel Discovery (1 job)"
echo "=========================================="
echo "(Waiting for pretrain job $JOB_ID_PRETRAIN to complete)"

JOB_ID_EXP6=$(sbatch --parsable --dependency=afterok:$JOB_ID_PRETRAIN scripts/slurm_experiment6_discovery.sh)
echo "Submitted job: $JOB_ID_EXP6"
echo "Dependency: afterok:$JOB_ID_PRETRAIN"

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "============================================================"
echo "All experiments submitted!"
echo "============================================================"
echo ""
echo "Job IDs:"
echo "  Pre-train:               $JOB_ID_PRETRAIN (runs first)"
echo "  Experiment 2 (SAR):      $JOB_ID_EXP2 (after pretrain)"
echo "  Experiment 3 (Transfer): $JOB_ID_EXP3 (after pretrain)"
echo "  Experiment 4 (Selection): $JOB_ID_EXP4 (after pretrain)"
echo "  Experiment 5 (PCGrad):   $JOB_ID_EXP5 (runs immediately)"
echo "  Experiment 6 (Novel):    $JOB_ID_EXP6 (after pretrain)"
echo "  Experiment 7 (Repr):     $JOB_ID_EXP7 (runs immediately)"
echo "  Baselines:               $JOB_ID_BASELINES (runs immediately)"
echo ""
echo "Execution order:"
echo "  1. Pretrain + Exp5 + Exp7 + Baselines start immediately"
echo "  2. After pretrain completes: Exp2, Exp3, Exp4, Exp6 start"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  squeue -u \$USER | wc -l  # Count running jobs"
echo ""
echo "Check pretrain status:"
echo "  tail -f logs/pretrain_${JOB_ID_PRETRAIN}.out"
echo ""
echo "Check experiment logs:"
echo "  tail -f logs/exp2_${JOB_ID_EXP2}.out"
echo "  tail -f logs/exp3_${JOB_ID_EXP3}_*.out"
echo "  tail -f logs/exp4_${JOB_ID_EXP4}_*.out"
echo "  tail -f logs/exp5_${JOB_ID_EXP5}_*.out"
echo "  tail -f logs/exp6_${JOB_ID_EXP6}.out"
echo "  tail -f logs/exp7_${JOB_ID_EXP7}_*.out"
echo "  tail -f logs/baselines_${JOB_ID_BASELINES}.out"
echo ""
echo "Cancel all jobs:"
echo "  scancel $JOB_ID_PRETRAIN $JOB_ID_EXP2 $JOB_ID_EXP3 $JOB_ID_EXP4 $JOB_ID_EXP5 $JOB_ID_EXP6 $JOB_ID_EXP7 $JOB_ID_BASELINES"
echo ""
echo "After completion, aggregate results:"
echo "  python scripts/experiment2_sar_validation.py --aggregate"
echo "  python scripts/experiment3_transfer_learning.py --aggregate --output-dir outputs/transfer_learning"
echo "  python scripts/experiment4_task_selection.py --aggregate"
echo "  python scripts/experiment5_pcgrad.py --aggregate"
echo "  python scripts/experiment6_novel_discovery.py --aggregate"
echo "  python scripts/experiment7_representation.py --aggregate"
echo ""
echo "============================================================"

# Save job IDs for later reference
echo "SEED=$SEED" > logs/submitted_jobs.txt
echo "PRETRAIN_JOB_ID=$JOB_ID_PRETRAIN" >> logs/submitted_jobs.txt
echo "EXP2_JOB_ID=$JOB_ID_EXP2" >> logs/submitted_jobs.txt
echo "EXP3_JOB_ID=$JOB_ID_EXP3" >> logs/submitted_jobs.txt
echo "EXP4_JOB_ID=$JOB_ID_EXP4" >> logs/submitted_jobs.txt
echo "EXP5_JOB_ID=$JOB_ID_EXP5" >> logs/submitted_jobs.txt
echo "EXP6_JOB_ID=$JOB_ID_EXP6" >> logs/submitted_jobs.txt
echo "EXP7_JOB_ID=$JOB_ID_EXP7" >> logs/submitted_jobs.txt
echo "BASELINES_JOB_ID=$JOB_ID_BASELINES" >> logs/submitted_jobs.txt
echo "SUBMIT_TIME=$(date)" >> logs/submitted_jobs.txt

echo "Job IDs saved to logs/submitted_jobs.txt"
