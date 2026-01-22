# Gradient-Based Causal Discovery for Multi-Task Molecular Learning

Discovering mechanistic relationships between molecular properties through gradient conflict analysis in multi-task learning.

## Key Findings

### 1. Method Validation

**Gradient conflicts capture real biological relationships when tasks share compound overlap.**

| Dataset | Tasks | Overlap | G vs Empirical | Status |
|---------|-------|---------|----------------|--------|
| Tox21 (toxicity panel) | 12 | 100% | r = 0.918*** | PASS |
| ToxCast (diverse assays) | 17 | 100% | r = 0.862*** | PASS |
| **Tox21+ADME (cross-domain)** | 16 | 100%* | r = 0.606*** | **PASS** |
| Diverse Properties | 6 | 5-40% | r = -0.34 (n.s.) | FAIL |
| ADME (merged datasets) | 4 | ~1% | r = 0.394 (n.s.) | FAIL |

*100% overlap achieved by matching Tox21 compounds to measured ADME data

### 2. Overlap Threshold

**~50% compound overlap required for reliable results.**

| Overlap | Correlation | Status |
|---------|-------------|--------|
| 100% | r = 0.927*** | PASS |
| 75% | r = 0.875*** | PASS |
| 50% | r = 0.814*** | PASS |
| 25% | r = 0.599*** | MARGINAL |
| 10% | r = 0.472*** | MARGINAL |

### 3. Cross-Domain Validation (Tox21 + ADME)

**Key finding**: Gradient conflicts correlate with empirical property structure (r = 0.606***).

| Category | r (G vs Empirical) | Interpretation |
|----------|-------------------|----------------|
| **Overall** | **0.606***  | Method validated across domains |
| Within-Toxicity | 0.952*** | Near-perfect within domain |
| Within-ADME | 0.661** | Strong within domain |
| Cross-Domain | 0.226 (n.s.) | Both G and Emp ~0 (floor effect) |

**Same property validation:**
| Property | Empirical r | Gradient G |
|----------|------------|------------|
| Lipophilicity | 1.000 | 0.706 |
| Solubility | 0.990 | 0.245 |

### 4. Assay Diversity

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
├── data/                       # Data loading and preprocessing
│   ├── dataset.py              # Multi-task dataset with missing labels
│   ├── graph_dataset.py        # PyG graph dataset
│   ├── graph_preprocessing.py  # SMILES to molecular graphs
│   └── splitting.py            # Scaffold-based splits
├── models/                     # Neural network architectures
│   ├── gnn_encoder.py          # GCN encoder
│   ├── gnn_multitask.py        # Multi-task GNN model
│   └── heads.py                # Task-specific heads
├── training/                   # Training infrastructure
│   ├── gradient_logger.py      # Per-task gradient computation
│   ├── gnn_trainer.py          # Training loop
│   ├── losses.py               # Masked multi-task loss
│   └── pcgrad.py               # PCGrad optimizer
├── analysis/                   # Analysis utilities
│   ├── empirical_correlations.py  # Compute correlations from data
│   ├── statistical_tests.py       # Permutation tests, bootstrap CI
│   └── visualization.py           # Heatmaps, plots
├── scripts/                    # Experiment and data scripts
│   ├── augment_tox21_with_adme.py    # Create cross-domain dataset
│   ├── validate_tox21_adme_correlation.py  # Key validation script
│   ├── test_overlap_threshold.py     # Overlap threshold analysis
│   └── experiment*.py                # Individual experiment scripts
├── train_tox21_gnn.py          # Primary: Tox21 GNN training
├── train_toxcast_gnn.py        # ToxCast validation
├── train_tox21_adme_gnn.py     # Cross-domain validation
├── config.yaml                 # Hyperparameters
├── requirements.txt            # Python dependencies
├── RESULTS.md                  # Detailed experimental results
└── outputs/                    # Generated outputs (mostly gitignored)
    ├── sar_validation/         # SAR validation results
    ├── tox21_adme_results/     # Cross-domain results
    └── figures/                # Generated figures
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

# Strategy B: Cross-domain validation (Tox21 + ADME)
python scripts/augment_tox21_with_adme.py    # Creates matched dataset
python train_tox21_adme_gnn.py --epochs 30   # Trains and validates

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
| 2b | ToxCast Validation | r = 0.862*** |
| 7 | Representation | r = 0.853 (ECFP vs GNN) |
| 8 | Tox21 + Physicochemical | Cross-category orthogonal |
| **9** | **Tox21 + ADME (Cross-Domain)** | **r = 0.606*** |
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
- **Cross-domain validation works when compounds are matched** (Tox21+ADME: p=0.002)

### When It Fails
- Merged datasets from different compound libraries
- Overlap < 25% produces marginal results
- ~~Cross-domain properties (binding + ADME + toxicity) lack overlap in public data~~ **Solved with compound matching approach**

### Implications
- Gradient conflicts are a valid signal for task relationships
- Method requires intentional experimental design (same compounds, multiple endpoints)
- Ideal for pharma screening panels where compounds are tested across many assays
- **Dataset construction strategy**: Match compounds from one domain to existing measurements in another (rather than merging low-overlap datasets)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{gradient_causal_discovery,
  title={Gradient-Based Causal Discovery for Multi-Task Molecular Learning},
  year={2025},
  url={https://github.com/username/gradient-causal-discovery}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
