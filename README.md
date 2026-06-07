# Gradient Conflict Analysis for Multi-Task Molecular Learning

**Authors:** Jasper Zhang, Bryan Cheng

**Accepted to ACM-BCB 2026** (ACM Conference on Bioinformatics, Computational Biology, and Health Informatics).

Code to reproduce the gradient-conflict / sample-overlap analysis for multi-task
molecular property prediction. The repository is **code only** — no manuscript or
paper files are tracked here; all reported numbers are regenerated from the scripts
below.

## Overview

Gradient conflicts during multi-task learning reveal whether tasks share underlying
mechanisms: when tasks are synergistic their per-task gradients align, when they
conflict the gradients oppose. The core output is a `K × K` gradient conflict matrix

```
G[i, j] = cosine_similarity(∇L_i, ∇L_j)
```

- `G[i,j] > 0` — tasks are synergistic (gradients align)
- `G[i,j] < 0` — tasks conflict (gradients oppose)
- `G[i,j] ≈ 0` — tasks are independent

**Central finding:** this signal is only reliable when the tasks share sufficient
compound overlap (a sharp threshold around 40–50%); below it, gradient–task
correlations are indistinguishable from noise.

## Repository structure

```
gradientmtl/
├── data/            # Dataset download, preprocessing, splitting, TDC/graph integration
├── models/          # Encoders (fingerprint + GNN) and multi-task heads
├── training/        # Training loop, per-task gradient logger, losses, PCGrad
├── analysis/        # Statistical tests, empirical correlations, visualization
├── experiments/     # Per-dataset training entry points (train_*.py) + deep analyses
├── scripts/         # Data curation, overlap analysis, and HPC/SLURM submission
├── config.yaml      # Datasets, model architecture, training hyperparameters, seed
└── requirements.txt # Python dependencies
```

Generated artifacts (checkpoints, gradient matrices, figures) are written to
`outputs/` and are **not** tracked — they are produced by running the code.

## Installation

**Requirements:** Python 3.9+, PyTorch ≥ 2.0, a CUDA-capable GPU is recommended but
not required.

```bash
git clone https://github.com/JasperZG/gradientmtl.git
cd gradientmtl
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

PyTorch Geometric sparse extensions (`torch_scatter`, `torch_sparse`,
`torch_cluster`, `pyg_lib`) are optional and only needed for some GNN encoders.
Install them following the [PyG instructions](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
matched to your Torch/CUDA versions.

## Reproducibility

- **Determinism.** All entry points seed Python, NumPy, and PyTorch with the fixed
  seed `42` (see `config.yaml: splitting.random_seed`), and use the fixed
  80/10/10 train/val/test split in `data/splitting.py`. Re-running a script
  reproduces the same split and the same reported correlations.
- **Configuration.** Datasets, model architecture, and training hyperparameters are
  centralized in `config.yaml`; command-line flags (`--epochs`, `--lr`, ...) override
  them per run.
- **Hardware note.** Exact floating-point values can vary slightly across GPU/CPU and
  library versions; the reported correlations and significance levels are stable to
  that variation.

### 1. Download the data

MoleculeNet datasets are pulled from their public DeepChem mirrors (and hERG from
TDC). The download is driven by `config.yaml`:

```bash
python data/download.py            # downloads BACE, BBBP, ESOL, Lipophilicity, hERG
```

Some extended experiments curate their own datasets (TDC, ToxCast, ADME, kinase);
each has a dedicated `scripts/curate_*.py` / `scripts/download_*.py` invoked in the
steps below. Downloaded CSVs land under `data/` and are git-ignored.

### 2. Reproduce the primary result (Tox21, 100% overlap)

```bash
python experiments/train_tox21_gnn.py --epochs 50
```

Trains the shared-encoder multi-task GNN, logs per-task gradients, and writes the
gradient conflict matrix `G` plus its correlation with the empirical task-similarity
matrix to `outputs/`.

### 3. Reproduce the overlap threshold sweep

```bash
python scripts/test_overlap_threshold.py
```

Reproduces the correlation-vs-overlap curve (the 100% → 10% rows below).

### 4. Reproduce the cross-domain and extended results

```bash
# Tox21 + ADME (matched compounds → high overlap)
python scripts/augment_tox21_with_adme.py
python experiments/train_tox21_adme_gnn.py --epochs 30

# ToxCast
python scripts/prepare_toxcast_diverse.py
python experiments/train_toxcast_gnn.py --epochs 30

# Kinase selectivity
python scripts/curate_kinase_selectivity.py
python experiments/train_kinase_gnn.py --epochs 30

# Benchmark-overlap analyses (MoleculeNet / TDC) and deep analyses
python scripts/analyze_moleculenet_overlap.py
python scripts/tdc_overlap_analysis.py
python experiments/run_deep_analysis.py
```

### Running on a cluster

SLURM batch scripts for the larger sweeps (transfer learning, task selection, PCGrad)
live in `scripts/*.sh`; `scripts/run_local.py --check` validates prerequisites and
orchestrates the local-then-HPC workflow.

## Main results

| Dataset | Tasks | Overlap | G vs Empirical | Significant? |
|---------|-------|---------|----------------|--------------|
| Tox21 | 12 | 100% | r = 0.918 | Yes (p<0.001) |
| ToxCast | 17 | 100% | r = 0.862 | Yes (p<0.001) |
| Tox21+ADME | 16 | 100%* | r = 0.606 | Yes (p=0.002) |
| Diverse Properties | 6 | 5–40% | r = -0.34 | No |
| ADME (merged) | 4 | ~1% | r = 0.394 | No |

*100% overlap achieved by matching Tox21 compounds to measured ADME data.

### Overlap threshold

| Overlap | Correlation | Status |
|---------|-------------|--------|
| 100% | r = 0.927 | Reliable |
| 75% | r = 0.875 | Reliable |
| 50% | r = 0.814 | Reliable |
| 25% | r = 0.599 | Marginal |
| 10% | r = 0.472 | Marginal |

## Core components

| Component | Description |
|-----------|-------------|
| `training/gradient_logger.py` | Per-task gradient computation |
| `training/gnn_trainer.py` | Multi-task training loop |
| `models/gnn_multitask.py` | Shared encoder + per-task heads |
| `analysis/empirical_correlations.py` | Gradient ↔ empirical correlation |
| `scripts/test_overlap_threshold.py` | Overlap threshold sweep |

## License

MIT License — see [LICENSE](LICENSE) for details.
