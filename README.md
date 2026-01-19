# Gradient-Based Causal Discovery for Multi-Task Molecular Learning

Discovering mechanistic relationships between molecular properties through gradient conflict analysis in multi-task learning.

## Key Finding

**Gradient conflicts capture real biological relationships when tasks share compound overlap.**

| Dataset | Compound Overlap | G vs Empirical Correlation |
|---------|-----------------|---------------------------|
| Tox21 (12 toxicity endpoints) | 100% | r = 0.918 *** |
| ADME (diverse properties) | ~1% | r = 0.394 (n.s.) |

This validates that gradient-based analysis can discover property trade-offs, but requires datasets where the same compounds are measured across all tasks.

## Project Structure

```
gradientproject/
├── config.yaml                 # Hyperparameters
├── data/
│   ├── dataset.py              # Multi-task dataset with missing labels
│   ├── graph_dataset.py        # PyG graph dataset
│   ├── graph_preprocessing.py  # SMILES to molecular graphs
│   └── splitting.py            # Scaffold-based splits
├── models/
│   ├── gnn_encoder.py          # GCN encoder
│   ├── gnn_multitask.py        # Multi-task GNN model
│   └── heads.py                # Task-specific heads
├── training/
│   ├── gradient_logger.py      # Per-task gradient computation
│   ├── gnn_trainer.py          # Training loop
│   ├── losses.py               # Masked multi-task loss
│   └── pcgrad.py               # PCGrad optimizer
├── analysis/
│   ├── empirical_correlations.py  # Compute correlations from data
│   ├── literature_matrix.py       # Known relationships
│   ├── statistical_tests.py       # Permutation tests, bootstrap CI
│   └── visualization.py           # Heatmaps, plots
├── scripts/
│   ├── run_local.py               # Local experiment runner
│   ├── submit_all_cshl.sh         # HPC job submission
│   ├── experiment2_sar_validation.py
│   ├── experiment3_transfer_learning.py
│   ├── experiment4_task_selection.py
│   ├── experiment5_pcgrad.py
│   ├── experiment6_novel_discovery.py
│   ├── experiment7_representation.py
│   └── slurm_*.sh                 # SLURM job scripts
├── train_tox21_gnn.py          # Main GNN training script
├── train_adme_gnn.py           # ADME dataset training
└── outputs/
    └── gradients/
        └── gnn_conflict_matrices.npz  # Pre-computed gradient matrix
```

## Quick Start

### 1. Install Dependencies

```bash
pip install torch torch-geometric rdkit-pypi scipy pandas numpy matplotlib seaborn
```

### 2. Run Local Experiments

```bash
# Run all local experiments (pre-training + analysis)
python scripts/run_local.py

# Or run individual steps:
python scripts/run_local.py --pretrain      # Train GNN, generate gradient matrix
python scripts/run_local.py --exp 2         # SAR validation
python scripts/run_local.py --exp 7         # ECFP vs GNN comparison
python scripts/run_local.py --baselines     # Single-task baselines
```

### 3. Run HPC Experiments

```bash
# Upload gradient matrix to cluster
scp outputs/gradients/gnn_conflict_matrices.npz $USER@hpc:gradient/outputs/gradients/

# Submit all HPC jobs
bash scripts/submit_all_cshl.sh
```

## Experiments

| Exp | Name | Description | Location |
|-----|------|-------------|----------|
| 2 | SAR Validation | Validate gradient conflicts vs empirical correlations | Local |
| 3 | Transfer Learning | Test if G predicts transfer success | HPC (792 jobs) |
| 4 | Task Selection | Compare selection algorithms | HPC (20 jobs) |
| 5 | PCGrad | Validate PCGrad helps conflicting pairs | HPC (15 jobs) |
| 6 | Novel Discovery | Find unexpected trade-offs | Local |
| 7 | Representation | Compare ECFP vs GNN gradients | Local |

## Results Summary

### Tox21 Validation (Experiment 2)

- **Pearson r = 0.918** between gradient conflicts and empirical correlations
- p < 0.001 (highly significant)
- Validates that gradient analysis captures real biological relationships

### Representation Generalization (Experiment 7)

- **r = 0.853** correlation between ECFP and GNN gradient matrices
- Gradient patterns are consistent across molecular representations

### Key Insight: Compound Overlap Requirement

The gradient conflict approach works when:
- All tasks measured on the same compounds (e.g., Tox21 panel)
- Sufficient data overlap for gradient comparison

It fails when:
- Tasks from different compound libraries (e.g., combining MoleculeNet datasets)
- Low compound overlap between tasks

## Gradient Conflict Matrix

The core output is a K x K matrix G where:

```
G[i,j] = cosine_similarity(grad_i, grad_j)
```

- G[i,j] > 0: Tasks are synergistic (gradients align)
- G[i,j] < 0: Tasks conflict (gradients oppose)
- G[i,j] = 0: Tasks are independent

## HPC Job Configuration

| Experiment | Jobs | Time/Job | Total GPU-Hours |
|------------|------|----------|-----------------|
| Exp 3 (Transfer) | 792 | ~30 min | ~400 |
| Exp 4 (Selection) | 20 | ~30 min | ~10 |
| Exp 5 (PCGrad) | 15 | ~1 hour | ~15 |
| **Total** | **827** | | **~425** |

## Citation

If you use this code, please cite:

```bibtex
@article{gradient_causal_discovery,
  title={Gradient-Based Causal Discovery for Multi-Task Molecular Learning},
  author={...},
  year={2025}
}
```

## License

MIT License
