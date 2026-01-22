# Gradient-Based Causal Discovery for Multi-Task Molecular Learning

Discovering mechanistic relationships between molecular properties through gradient conflict analysis in multi-task learning.

## Key Findings

### 1. Method Validation

**Gradient conflicts capture real biological relationships when tasks share compound overlap.**

| Dataset | Tasks | Overlap | G vs Empirical | Status |
|---------|-------|---------|----------------|--------|
| Tox21 (toxicity panel) | 12 | 100% | r = 0.918*** | PASS |
| ToxCast (diverse assays) | 17 | 100% | r = 0.862*** | PASS |
| Diverse Properties | 6 | 5-40% | r = -0.34 (n.s.) | FAIL |
| ADME (merged datasets) | 4 | ~1% | r = 0.394 (n.s.) | FAIL |

### 2. Overlap Threshold

**~50% compound overlap required for reliable results.**

| Overlap | Correlation | Status |
|---------|-------------|--------|
| 100% | r = 0.927*** | PASS |
| 75% | r = 0.875*** | PASS |
| 50% | r = 0.814*** | PASS |
| 25% | r = 0.599*** | MARGINAL |
| 10% | r = 0.472*** | MARGINAL |

### 3. Assay Diversity

ToxCast validation demonstrates method generalizes across 7 assay families:
- **ATG**: Gene expression reporters (AP-1, AP-2 pathways)
- **BSK**: BioSeek immune panel (E-selectin, ICAM-1, HLA-DR)
- **NVS**: Nuclear receptors (hER, bER)
- **APR**: High-content imaging (cell cycle arrest)
- **ACEA**: Cell proliferation
- **OT**: Odyssey Thera
- **Tanguay**: Zebrafish developmental toxicity

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
│   ├── test_overlap_threshold.py  # Overlap threshold analysis
│   ├── prepare_toxcast_diverse.py # ToxCast data preparation
│   ├── curate_diverse_properties.py
│   └── slurm_*.sh                 # SLURM job scripts
├── train_tox21_gnn.py          # Tox21 GNN training (primary)
├── train_toxcast_gnn.py        # ToxCast validation (Strategy A)
├── train_diverse_gnn.py        # Diverse properties (Strategy B)
├── train_adme_gnn.py           # ADME negative control
└── outputs/
    └── gradients/
        └── gnn_conflict_matrices.npz  # Pre-computed gradient matrix
```

## Quick Start

### 1. Install Dependencies

```bash
pip install torch torch-geometric rdkit-pypi scipy pandas numpy matplotlib seaborn
```

### 2. Run Validation Experiments

```bash
# Primary validation (Tox21)
python train_tox21_gnn.py --epochs 50

# Strategy A: Dataset generalization (ToxCast)
python scripts/prepare_toxcast_diverse.py
python train_toxcast_gnn.py --epochs 30

# Overlap threshold analysis
python scripts/test_overlap_threshold.py
```

### 3. Run HPC Experiments

```bash
# Upload gradient matrix to cluster
scp outputs/gradients/gnn_conflict_matrices.npz $USER@hpc:gradient/outputs/gradients/

# Submit all HPC jobs
bash scripts/submit_all_cshl.sh
```

## Experiments

### Local Experiments (Completed)

| Exp | Name | Result |
|-----|------|--------|
| 2 | SAR Validation | r = 0.918*** (Tox21) |
| 7 | Representation | r = 0.853 (ECFP vs GNN) |
| - | ToxCast Validation | r = 0.862*** |
| - | Overlap Threshold | 50% threshold identified |

### HPC Experiments (Pending)

| Exp | Name | Description | Jobs |
|-----|------|-------------|------|
| 3 | Transfer Learning | Test if G predicts transfer success | 792 |
| 4 | Task Selection | Compare selection algorithms | 20 |
| 5 | PCGrad | Validate PCGrad helps conflicting pairs | 15 |

## Gradient Conflict Matrix

The core output is a K x K matrix G where:

```
G[i,j] = cosine_similarity(grad_i, grad_j)
```

- G[i,j] > 0: Tasks are synergistic (gradients align)
- G[i,j] < 0: Tasks conflict (gradients oppose)
- G[i,j] = 0: Tasks are independent

## Key Insights

### When It Works
- Panel assays with 100% compound overlap (Tox21, ToxCast)
- Overlap >= 50% for reliable correlations
- Generalizes across assay types within toxicology

### When It Fails
- Merged datasets from different compound libraries
- Overlap < 25% produces marginal results
- Cross-domain properties (binding + ADME + toxicity) lack overlap in public data

### Implications
- Gradient conflicts are a valid signal for task relationships
- Method requires intentional experimental design (same compounds, multiple endpoints)
- Ideal for pharma screening panels where compounds are tested across many assays

## Citation

```bibtex
@article{gradient_causal_discovery,
  title={Gradient-Based Causal Discovery for Multi-Task Molecular Learning},
  author={...},
  year={2025}
}
```

## License

MIT License
