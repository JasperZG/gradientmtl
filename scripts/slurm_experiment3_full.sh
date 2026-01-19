#!/bin/bash
#SBATCH --job-name=exp3_transfer
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/exp3_%A_%a.out
#SBATCH --error=logs/exp3_%A_%a.err
#SBATCH --array=0-791

# =============================================================================
# Experiment 3: Full Transfer Learning Matrix
# =============================================================================
#
# SLURM array job for 792 transfer learning experiments:
#   - 132 task pairs (12 tasks × 11 targets)
#   - 3 data regimes (n=50, 100, 200)
#   - 2 conditions (transfer vs scratch)
#
# Job mapping: job_index = pair_idx * 6 + regime_idx * 2 + condition_idx
#
# Expected runtime: ~30 min per job
# Total: 792 jobs × 0.5 hours = 396 GPU-hours
# With parallelization: ~9 hours wall time
#
# =============================================================================

echo "============================================"
echo "Experiment 3: Transfer Learning Validation"
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
mkdir -p outputs/transfer_learning
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
python scripts/experiment3_transfer_learning.py \
    --job-index $SLURM_ARRAY_TASK_ID \
    --seed $SEED \
    --output-dir outputs/transfer_learning

echo ""
echo "End time: $(date)"
echo "============================================"
