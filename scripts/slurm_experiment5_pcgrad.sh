#!/bin/bash
#SBATCH --job-name=exp5_pcgrad
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/exp5_%A_%a.out
#SBATCH --error=logs/exp5_%A_%a.err
#SBATCH --array=0-14

# =============================================================================
# Experiment 5: PCGrad Validation
# =============================================================================
#
# Validates that gradient conflicts detected by our G matrix are real
# by showing PCGrad helps conflicting pairs but not synergistic ones.
#
# SLURM array job for 15 task pairs:
#   - Jobs 0-4: High-conflict pairs (PCGrad should HELP)
#   - Jobs 5-9: Synergistic pairs (PCGrad should NOT help)
#   - Jobs 10-14: Random pairs (control)
#
# Each job:
#   1. Trains two-task model WITHOUT PCGrad (baseline)
#   2. Trains two-task model WITH PCGrad
#   3. Compares performance improvement
#
# Expected runtime: ~30-60 min per job (2 training runs)
# Total: 15 jobs × 1 hour = 15 GPU-hours
#
# =============================================================================

echo "============================================"
echo "Experiment 5: PCGrad Validation"
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
mkdir -p outputs/pcgrad
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
python scripts/experiment5_pcgrad.py \
    --job-index $SLURM_ARRAY_TASK_ID \
    --seed $SEED \
    --output-dir outputs/pcgrad

echo ""
echo "End time: $(date)"
echo "============================================"
