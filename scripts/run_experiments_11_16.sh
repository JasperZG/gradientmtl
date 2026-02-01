#!/bin/bash
# Run experiments 11-16 sequentially
# Usage: bash scripts/run_experiments_11_16.sh [GPU_ID]

set -e

GPU=${1:-6}
export CUDA_VISIBLE_DEVICES=$GPU
echo "Using GPU $GPU"
echo "=========================================="

# Experiment 11: Pairwise Overlap Analysis (no GPU needed, uses existing data)
echo ""
echo ">>> Experiment 11: Pairwise Overlap Analysis"
python scripts/experiment11_pairwise_overlap.py

# Experiment 12: E Stability Under Overlap Reduction (no GPU needed)
echo ""
echo ">>> Experiment 12: E Stability"
python scripts/experiment12_e_stability.py --n-trials 20

# Experiment 13: Benchmark Dataset Overlap (no GPU needed, needs PyTDC)
echo ""
echo ">>> Experiment 13: Benchmark Overlap Measurement"
python scripts/experiment13_benchmark_overlap.py

# Experiment 14: Negative Transfer Prediction (no GPU needed, uses existing results)
echo ""
echo ">>> Experiment 14: Negative Transfer Prediction"
python scripts/experiment14_negative_transfer.py

# Experiment 15: Task2Vec Baseline (GPU training)
echo ""
echo ">>> Experiment 15: Task2Vec Baseline Comparison"
python scripts/experiment15_task2vec_baseline.py --n-trials 5

# Experiment 16: Synthetic Ground Truth Validation (GPU training)
echo ""
echo ">>> Experiment 16: Synthetic Ground Truth Validation"
python scripts/experiment16_synthetic_validation.py --n-trials 5

echo ""
echo "=========================================="
echo "All experiments complete!"
echo "Results in outputs/experiment{11..16}_*/"
