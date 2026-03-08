#!/usr/bin/env python3
"""
Train GNN on SIDER dataset (27 side effect categories, 1,427 drugs).

SIDER has 100% compound overlap (all drugs have labels for all side effects),
making it ideal for gradient conflict analysis validation.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
import json

from data.graph_preprocessing import MoleculeGraphPreprocessor, get_atom_feature_dim
from data.splitting import scaffold_split
from data.graph_dataset import MultiTaskGraphDataset
from models.gnn_multitask import GNNMultiTaskModel
from training.gnn_trainer import GNNMultiTaskTrainer
from analysis.visualization import plot_conflict_heatmap, print_conflict_summary
from analysis.empirical_correlations import compute_empirical_correlation


def load_sider():
    """Load SIDER dataset."""
    sider_path = Path('outputs/moleculenet_data/sider.csv')
    if not sider_path.exists():
        raise FileNotFoundError(
            f"SIDER not found at {sider_path}. "
            "Run: python -c \"import urllib.request; ...\" to download"
        )

    df = pd.read_csv(sider_path)

    # Find SMILES column
    smiles_col = 'smiles' if 'smiles' in df.columns else df.columns[0]
    task_cols = [c for c in df.columns if c != smiles_col and c != 'mol_id']

    return df, smiles_col, task_cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--encoder_type', type=str, default='gcn',
                       choices=['gcn', 'gat'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    print("\n" + "=" * 60)
    print("Loading SIDER Dataset")
    print("=" * 60)

    df, smiles_col, task_cols = load_sider()
    print(f"Compounds: {len(df)}")
    print(f"Tasks: {len(task_cols)}")
    print(f"Task names: {task_cols[:5]}... (showing first 5)")

    smiles_list = df[smiles_col].tolist()

    # All tasks are binary classification
    task_types = {task: 'classification' for task in task_cols}

    # Extract labels
    raw_labels = {}
    for task in task_cols:
        raw_labels[task] = df[task].values.astype(np.float32)

    # Check for missing data
    missing_pct = sum(np.isnan(raw_labels[t]).sum() for t in task_cols)
    total = len(df) * len(task_cols)
    print(f"Missing data: {missing_pct}/{total} ({100*missing_pct/total:.1f}%)")

    # Convert to graphs
    print("\n" + "=" * 60)
    print("Converting SMILES to Molecular Graphs")
    print("=" * 60)

    preprocessor = MoleculeGraphPreprocessor()
    valid_smiles, graphs, valid_indices = preprocessor.process_smiles_list(
        smiles_list, show_progress=True
    )

    # Filter labels to valid indices
    labels = {task: values[valid_indices] for task, values in raw_labels.items()}

    print(f"\nValid molecules: {len(graphs)}/{len(smiles_list)}")

    # Label statistics
    print("\nLabel statistics (first 10 tasks):")
    for task in task_cols[:10]:
        n_pos = np.nansum(labels[task])
        n_neg = np.sum(~np.isnan(labels[task])) - n_pos
        print(f"  {task[:30]:30s}: {n_pos:.0f} pos, {n_neg:.0f} neg")

    # Scaffold split
    print("\n" + "=" * 60)
    print("Scaffold Split")
    print("=" * 60)

    train_idx, val_idx, test_idx = scaffold_split(
        valid_smiles, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=seed
    )

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Create datasets
    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    train_labels = {task: arr[train_idx] for task, arr in labels.items()}
    val_labels = {task: arr[val_idx] for task, arr in labels.items()}

    train_dataset = MultiTaskGraphDataset(train_graphs, train_labels, task_types)
    val_dataset = MultiTaskGraphDataset(val_graphs, val_labels, task_types)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    print("\n" + "=" * 60)
    print(f"Creating GNN Model ({args.encoder_type.upper()})")
    print("=" * 60)

    atom_feature_dim = get_atom_feature_dim()
    task_names = list(task_types.keys())

    model = GNNMultiTaskModel(
        task_names=task_names,
        atom_feature_dim=atom_feature_dim,
        encoder_type=args.encoder_type,
        encoder_hidden_dims=[256, 256, 256],
        encoder_output_dim=256,
        head_hidden_dim=128,
        dropout=0.3,
    )

    print(model.summary())

    # Setup output directory
    output_dir = Path('outputs/sider_gnn')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train
    print("\n" + "=" * 60)
    print(f"Training ({args.epochs} epochs)")
    print("=" * 60)

    config = {
        'learning_rate': args.lr,
        'weight_decay': 0.01,
        'epochs': args.epochs,
        'early_stopping_patience': 25,
        'gradient_log_interval': 5,
        'gradient_clip_norm': 1.0,
    }

    trainer = GNNMultiTaskTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        task_types=task_types,
        config=config,
        device=device,
        output_dir=output_dir,
    )

    results = trainer.train()

    # Gradient conflict analysis
    print("\n" + "=" * 60)
    print("Gradient Conflict Analysis")
    print("=" * 60)

    G = results['conflict_matrix']

    # Save gradient matrix
    np.savez(
        output_dir / 'gradient_matrices.npz',
        conflict_matrix=G,
        task_names=task_names,
    )

    # Compute empirical correlation matrix
    print("\nComputing empirical correlation matrix...")
    E = compute_empirical_correlation(labels, task_names)
    np.save(output_dir / 'empirical_correlation.npy', E)

    # Compute gradient-empirical correlation
    upper_tri_idx = np.triu_indices(len(task_names), k=1)
    g_vals = G[upper_tri_idx]
    e_vals = E[upper_tri_idx]

    # Filter NaN
    mask = ~(np.isnan(g_vals) | np.isnan(e_vals))
    g_vals_clean = g_vals[mask]
    e_vals_clean = e_vals[mask]

    from scipy import stats
    r, p = stats.pearsonr(g_vals_clean, e_vals_clean)
    spearman_r, spearman_p = stats.spearmanr(g_vals_clean, e_vals_clean)

    print(f"\nGradient-Empirical Correlation:")
    print(f"  Pearson r: {r:.4f} (p = {p:.2e})")
    print(f"  Spearman r: {spearman_r:.4f} (p = {spearman_p:.2e})")
    print(f"  N pairs: {len(g_vals_clean)}")

    # Save validation results
    validation_results = {
        'pearson_r': float(r),
        'pearson_p': float(p),
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'n_pairs': int(len(g_vals_clean)),
        'n_tasks': len(task_names),
        'n_compounds': len(graphs),
        'significance': '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.')),
    }

    with open(output_dir / 'validation_results.json', 'w') as f:
        json.dump(validation_results, f, indent=2)

    # Visualize
    plot_conflict_heatmap(
        G, task_names,
        output_dir / 'sider_gradient_heatmap.png',
        title='SIDER Gradient Conflicts',
        cluster=True,
    )

    # Create correlation scatter plot
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(e_vals_clean, g_vals_clean, alpha=0.5, s=20)

    # Regression line
    slope, intercept = np.polyfit(e_vals_clean, g_vals_clean, 1)
    x_line = np.array([e_vals_clean.min(), e_vals_clean.max()])
    ax.plot(x_line, slope * x_line + intercept, 'r--', linewidth=2)

    ax.set_xlabel('Empirical Correlation')
    ax.set_ylabel('Gradient Conflict')
    ax.set_title(f'SIDER: Gradient vs Empirical (r = {r:.3f})')
    plt.tight_layout()
    plt.savefig(output_dir / 'sider_correlation_scatter.png', dpi=150)
    plt.close()

    # Summary
    print("\n" + "=" * 60)
    print("SIDER Training Complete")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    print(f"\nKey metrics:")
    print(f"  Tasks: {len(task_names)}")
    print(f"  Compounds: {len(graphs)}")
    print(f"  Gradient-Empirical r: {r:.4f}")

    if r > 0.8:
        print(f"\n>>> EXCELLENT: Very strong correlation (r > 0.8)")
    elif r > 0.6:
        print(f"\n>>> GOOD: Strong correlation (r > 0.6)")
    elif r > 0.4:
        print(f"\n>>> MODERATE: Moderate correlation (r > 0.4)")
    else:
        print(f"\n>>> WEAK: Weak correlation (r < 0.4)")


if __name__ == '__main__':
    main()
