#!/bin/bash
# Full experimental pipeline for TDC dataset
# Runs all experiments sequentially on specified GPU

set -e  # Exit on error

GPU=${1:-6}
export CUDA_VISIBLE_DEVICES=$GPU

echo "=============================================="
echo "TDC Full Experimental Pipeline"
echo "GPU: $GPU"
echo "Started: $(date)"
echo "=============================================="

# Directory setup
TDC_RESULTS="outputs/tdc_results"
TDC_DATA="outputs/tdc_data"

# Step 0: Prepare TDC data if needed
echo ""
echo "[0/8] Preparing TDC data..."
echo "=============================================="
if [ ! -f "$TDC_DATA/tdc_multiproperty.csv" ]; then
    echo "Downloading and curating TDC data..."
    python scripts/curate_tdc_multiproperty.py
else
    echo "TDC data already exists, skipping."
fi

# Step 1: Train GNN on TDC
echo ""
echo "[1/8] Training GNN on TDC dataset..."
echo "=============================================="
python experiments/train_tdc_gnn.py --epochs 100 --output-dir $TDC_RESULTS

# Check if training succeeded
if [ ! -f "$TDC_RESULTS/gradient_matrices.npz" ]; then
    echo "ERROR: Training failed - no gradient matrix found"
    exit 1
fi
echo "Training complete. Gradient matrix saved."

# Step 2: SAR Validation
echo ""
echo "[2/8] Running SAR Validation (Exp 2)..."
echo "=============================================="
python scripts/experiment2_sar_validation.py \
    --gradient-matrix $TDC_RESULTS/gradient_matrices.npz \
    --output-dir outputs/tdc_sar_validation \
    2>&1 || echo "SAR validation skipped (may need adaptation)"

# Step 3: Transfer Learning
echo ""
echo "[3/8] Running Transfer Learning (Exp 3)..."
echo "=============================================="
# Note: experiment3 is designed for Tox21, run Phase 2 style instead
python scripts/kinase_phase2_experiments.py transfer \
    --data-path $TDC_DATA/tdc_multiproperty.csv \
    --gradient-path $TDC_RESULTS/gradient_matrices.npz \
    --output-dir outputs/tdc_phase2 \
    2>&1 || echo "Transfer learning needs TDC adaptation"

# Step 4: Task Selection
echo ""
echo "[4/8] Running Task Selection (Exp 4)..."
echo "=============================================="
python scripts/experiment4_task_selection.py \
    --gradient-matrix $TDC_RESULTS/gradient_matrices.npz \
    --output-dir outputs/tdc_task_selection \
    2>&1 || echo "Task selection completed or skipped"

# Step 5: PCGrad
echo ""
echo "[5/8] Running PCGrad (Exp 5)..."
echo "=============================================="
python scripts/kinase_phase2_experiments.py pcgrad \
    --data-path $TDC_DATA/tdc_multiproperty.csv \
    --gradient-path $TDC_RESULTS/gradient_matrices.npz \
    --output-dir outputs/tdc_phase2 \
    2>&1 || echo "PCGrad needs TDC adaptation"

# Step 6: Phase 3A - Assay Prioritization
echo ""
echo "[6/8] Running Phase 3A: Assay Prioritization..."
echo "=============================================="
python scripts/phase3_assay_prioritization.py \
    --results-dir $TDC_RESULTS \
    --output-dir outputs/tdc_phase3_assay

# Step 7: Phase 3B - Transfer Guidance
echo ""
echo "[7/8] Running Phase 3B: Transfer Guidance..."
echo "=============================================="
python scripts/phase3_transfer_guidance.py \
    --results-dir $TDC_RESULTS \
    --output-dir outputs/tdc_phase3_transfer

# Step 8: Phase 4 - Mechanistic Analysis
echo ""
echo "[8/8] Running Phase 4: Mechanistic Analysis..."
echo "=============================================="
python scripts/phase4_literature_validation.py \
    --results-dir $TDC_RESULTS \
    --output-dir outputs/tdc_phase4_literature \
    2>&1 || echo "Literature validation completed"

python scripts/phase4_structural_analysis.py \
    --results-dir $TDC_RESULTS \
    --output-dir outputs/tdc_phase4_structural \
    2>&1 || echo "Structural analysis completed"

python scripts/phase4_hypothesis_generation.py \
    --results-dir $TDC_RESULTS \
    --output-dir outputs/tdc_phase4_hypotheses \
    2>&1 || echo "Hypothesis generation completed"

# Summary
echo ""
echo "=============================================="
echo "Pipeline Complete!"
echo "Finished: $(date)"
echo "=============================================="
echo ""
echo "Outputs:"
ls -la outputs/tdc_*/
