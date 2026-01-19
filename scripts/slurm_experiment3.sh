#!/bin/bash
#SBATCH --job-name=gradient_exp3
#SBATCH --output=logs/exp3_%A_%a.out
#SBATCH --error=logs/exp3_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --signal=B:TERM@120

# =============================================================================
# Experiment 3: Transfer Learning Validation
# =============================================================================
#
# Usage:
#   sbatch scripts/slurm_experiment3.sh                    # Run all pairs
#   sbatch --array=0-5 scripts/slurm_experiment3.sh        # Run as array job
#   sbatch --array=0-5%3 scripts/slurm_experiment3.sh      # Max 3 concurrent
#
# =============================================================================

set -e

# Print job info
echo "=========================================="
echo "Experiment 3: Transfer Learning Validation"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array Task: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="

# Create logs directory
mkdir -p logs

# Load modules
module purge 2>/dev/null || true
module load python/3.9 2>/dev/null || module load python 2>/dev/null || true
module load cuda/11.8 2>/dev/null || module load cuda 2>/dev/null || true

# Activate conda environment
if command -v conda &> /dev/null; then
    # Try different environment names
    for env_name in gradient gradientproject pytorch; do
        if conda activate $env_name 2>/dev/null; then
            echo "Activated conda environment: $env_name"
            break
        fi
    done
fi

# Set environment variables
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Navigate to project directory
cd "${SLURM_SUBMIT_DIR:-$(dirname $(dirname $(realpath $0)))}"

# Check CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Arguments for the experiment
EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-32}
LR=${LR:-0.001}
SEED=${SEED:-42}

# Determine pair index for array jobs
PAIR_ARG=""
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    PAIR_ARG="--pair_index $SLURM_ARRAY_TASK_ID"
    OUTPUT_DIR="outputs/experiment3/seed${SEED}_pair${SLURM_ARRAY_TASK_ID}"
else
    OUTPUT_DIR="outputs/experiment3/seed${SEED}_all"
fi

echo ""
echo "Running Experiment 3..."
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LR"
echo "  Seed: $SEED"
echo "  Output: $OUTPUT_DIR"
echo "  Pair Index: ${SLURM_ARRAY_TASK_ID:-all}"
echo ""

# Run experiment
python scripts/experiment3_transfer_learning.py \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --seed $SEED \
    --output_dir "$OUTPUT_DIR" \
    $PAIR_ARG

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job completed with exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "=========================================="

exit $EXIT_CODE
