#!/bin/bash
#SBATCH --job-name=exp4_selection
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/exp4_%A_%a.out
#SBATCH --error=logs/exp4_%A_%a.err
#SBATCH --array=0-19

# =============================================================================
# Experiment 4: Task Selection Algorithms
# =============================================================================
#
# SLURM array job for task selection experiments:
#   - 4 selection methods × 5 budget levels = 20 jobs
#   - Methods: greedy, clustering, max_diversity, random
#   - Budgets: 3, 4, 5, 6, 7 tasks
#
# Job mapping: job_index = method_idx * 5 + budget_idx
#   - 0-4: greedy with budgets 3-7
#   - 5-9: clustering with budgets 3-7
#   - 10-14: max_diversity with budgets 3-7
#   - 15-19: random with budgets 3-7
#
# Each job:
#   1. Loads pre-computed gradient conflict matrix
#   2. Selects tasks using the specified algorithm
#   3. Trains MTL model on selected tasks
#   4. Evaluates on held-out tasks
#
# Expected runtime: ~1-2 hours per job
#
# =============================================================================

echo "============================================"
echo "Experiment 4: Task Selection Validation"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""

# Load modules (adjust for CSHL cluster)
module load cuda/11.8 2>/dev/null || true
module load python/3.10 2>/dev/null || true
module load anaconda 2>/dev/null || true

# Activate conda environment
source activate gradient 2>/dev/null || conda activate gradient

# Navigate to project directory
cd $SLURM_SUBMIT_DIR
cd ..

# Create output directories
mkdir -p outputs/task_selection
mkdir -p logs

# Get seed from environment or use default
SEED=${SEED:-42}

echo "Configuration:"
echo "  Array index: $SLURM_ARRAY_TASK_ID"
echo "  Seed: $SEED"
echo "  Python: $(which python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Run experiment
python scripts/experiment4_task_selection.py \
    --job-index $SLURM_ARRAY_TASK_ID \
    --seed $SEED \
    --output-dir outputs/task_selection

echo ""
echo "End time: $(date)"
echo "============================================"
