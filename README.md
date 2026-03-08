# Gradient Conflict Analysis for Multi-Task Molecular Learning

## Overview

This repository contains code for analyzing gradient conflicts in multi-task learning to discover mechanistic relationships between molecular properties.

## Key Concept

Gradient conflicts during multi-task learning reveal whether tasks share underlying mechanisms. When tasks are synergistic, their gradients align; when tasks conflict, their gradients oppose.

**Critical finding**: This signal is only reliable when tasks share sufficient compound overlap (≥50%).

## Main Results

| Dataset | Tasks | Overlap | G vs Empirical | Significant? |
|---------|-------|---------|----------------|--------------|
| Tox21 | 12 | 100% | r = 0.918 | Yes (p<0.001) |
| ToxCast | 17 | 100% | r = 0.862 | Yes (p<0.001) |
| Tox21+ADME | 16 | 100%* | r = 0.606 | Yes (p=0.002) |
| Diverse Properties | 6 | 5-40% | r = -0.34 | No |
| ADME (merged) | 4 | ~1% | r = 0.394 | No |

*100% overlap achieved by matching Tox21 compounds to measured ADME data

## Overlap Threshold

| Overlap | Correlation | Status |
|---------|-------------|--------|
| 100% | r = 0.927 | Reliable |
| 75% | r = 0.875 | Reliable |
| 50% | r = 0.814 | Reliable |
| 25% | r = 0.599 | Marginal |
| 10% | r = 0.472 | Marginal |

## Installation

**Requirements:**
- Python 3.8+
- PyTorch 1.12+
- CUDA-capable GPU (recommended)

**Setup:**
```bash
git clone https://github.com/JasperZG/gradient.git
cd gradient
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Quick Start

**Primary validation (Tox21):**
```bash
python experiments/train_tox21_gnn.py --epochs 50
```

**Cross-domain validation (Tox21 + ADME):**
```bash
python scripts/augment_tox21_with_adme.py
python experiments/train_tox21_adme_gnn.py --epochs 30
```

**Overlap threshold analysis:**
```bash
python scripts/test_overlap_threshold.py
```

## Extended Experiments

```bash
# ToxCast validation
python scripts/prepare_toxcast_diverse.py
python experiments/train_toxcast_gnn.py --epochs 30

# Kinase selectivity
python scripts/curate_kinase_selectivity.py
python experiments/train_kinase_gnn.py --epochs 30

# Deep analysis experiments
python experiments/run_deep_analysis.py
```

## Project Structure

```
gradient/
├── data/           # Data loading and preprocessing
├── models/         # GNN architectures
├── training/       # Training loop, gradient logger, losses
├── analysis/       # Statistical analysis and visualization
├── experiments/    # Training scripts
├── scripts/        # Data curation and analysis scripts
├── config.yaml     # Hyperparameters
└── requirements.txt
```

## Core Components

| Component | Description |
|-----------|-------------|
| `training/gradient_logger.py` | Per-task gradient computation |
| `training/gnn_trainer.py` | Multi-task training loop |
| `models/gnn_multitask.py` | Shared encoder + task heads |
| `analysis/empirical_correlations.py` | Correlation computation |

## Gradient Conflict Matrix

The core output is a K × K matrix G where:

```
G[i,j] = cosine_similarity(∇L_i, ∇L_j)
```

- G[i,j] > 0: Tasks are synergistic (gradients align)
- G[i,j] < 0: Tasks conflict (gradients oppose)
- G[i,j] ≈ 0: Tasks are independent

## Citation

```bibtex
@article{gradient_mtl_2026,
  title={When Gradients Satisfice: The Sample Overlap Requirement for Multi-Task Learning},
  author={Zhang, Jasper and Cheng, Bryan},
  year={2026}
}
```

## License

MIT License — see LICENSE file for details.
