#!/usr/bin/env python3
"""
ToxCast training with Graph Neural Network encoder.

Strategy A Validation: Test method generalization to a different panel assay.

This uses the diverse ToxCast subset (17 tasks from 7 assay families):
- ATG: Gene expression reporters
- BSK: BioSeek immune panel
- NVS: Nuclear receptor assays
- APR: High-content imaging
- ACEA: Cell proliferation
- OT: Odyssey Thera
- Tanguay: Zebrafish developmental

Success criteria: r > 0.6 correlation between G matrix and empirical correlations
(validates method is not Tox21-specific)
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from scipy import stats

from data.graph_preprocessing import MoleculeGraphPreprocessor, get_atom_feature_dim
from data.splitting import scaffold_split
from data.graph_dataset import MultiTaskGraphDataset
from models.gnn_multitask import GNNMultiTaskModel
from training.gnn_trainer import GNNMultiTaskTrainer
from analysis.visualization import plot_conflict_heatmap, print_conflict_summary
from analysis.empirical_correlations import compute_empirical_correlations


def load_toxcast_diverse():
    """Load diverse ToxCast subset."""
    data_path = Path('outputs/toxcast_data/toxcast_diverse.csv')

    if not data_path.exists():
        print("ToxCast diverse subset not found. Run prepare_toxcast_diverse.py first.")
        raise FileNotFoundError(data_path)

    df = pd.read_csv(data_path)
    print(f"Loaded ToxCast diverse: {len(df)} compounds")

    # Get task columns (all except smiles)
    task_cols = [c for c in df.columns if c != 'smiles']

    # Create task type dict (all classification for ToxCast)
    tasks = {task: 'classification' for task in task_cols}

    return df, tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--encoder_type', type=str, default='gcn',
                       choices=['gcn', 'gat'])
    parser.add_argument('--min_tasks', type=int, default=5,
                       help='Minimum number of tasks with labels per molecule')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    print("\n" + "=" * 60)
    print("Loading ToxCast Diverse Subset")
    print("=" * 60)

    df, tasks = load_toxcast_diverse()
    task_names = list(tasks.keys())

    smiles_list = df['smiles'].tolist()
    raw_labels = {}
    for task in task_names:
        raw_labels[task] = df[task].values.astype(np.float32)

    print(f"Total molecules: {len(smiles_list)}")
    print(f"Tasks: {len(task_names)}")

    # Check task coverage
    print("\nTask coverage:")
    for task in task_names[:5]:
        n = np.sum(~np.isnan(raw_labels[task]))
        print(f"  {task[:40]}: {n} ({100*n/len(smiles_list):.1f}%)")
    if len(task_names) > 5:
        print(f"  ... and {len(task_names) - 5} more tasks")

    # Convert SMILES to graphs
    print("\n" + "=" * 60)
    print("Converting SMILES to Molecular Graphs")
    print("=" * 60)

    preprocessor = MoleculeGraphPreprocessor()
    valid_smiles, graphs, valid_indices = preprocessor.process_smiles_list(
        smiles_list, show_progress=True
    )

    # Filter labels to valid indices
    labels = {task: values[valid_indices] for task, values in raw_labels.items()}

    # Filter to molecules with enough task labels
    n_labels_per_mol = np.zeros(len(valid_smiles))
    for task, values in labels.items():
        n_labels_per_mol += ~np.isnan(values)

    mask = n_labels_per_mol >= args.min_tasks
    keep_indices = np.where(mask)[0]

    print(f"\nFiltering to molecules with {args.min_tasks}+ task labels:")
    print(f"  Before: {len(valid_smiles)} molecules")
    print(f"  After: {len(keep_indices)} molecules")

    if len(keep_indices) < 500:
        print("Warning: Too few molecules. Reducing min_tasks requirement.")
        args.min_tasks = 3
        mask = n_labels_per_mol >= args.min_tasks
        keep_indices = np.where(mask)[0]
        print(f"  With min_tasks={args.min_tasks}: {len(keep_indices)} molecules")

    # Filter data
    graphs = [graphs[i] for i in keep_indices]
    valid_smiles = [valid_smiles[i] for i in keep_indices]
    labels = {task: arr[keep_indices] for task, arr in labels.items()}

    # Check label coverage after filtering
    print("\nLabel coverage after filtering:")
    for task in list(labels.keys())[:5]:
        n_valid = np.sum(~np.isnan(labels[task]))
        if n_valid > 0:
            n_pos = np.nansum(labels[task])
            print(f"  {task[:40]}: {n_valid} labels, {n_pos:.0f} positive")

    # Scaffold split
    print("\n" + "=" * 60)
    print("Scaffold Split")
    print("=" * 60)

    train_idx, val_idx, test_idx = scaffold_split(
        valid_smiles, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=seed
    )

    # Create datasets
    print("\n" + "=" * 60)
    print("Creating Data Loaders")
    print("=" * 60)

    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    test_graphs = [graphs[i] for i in test_idx]

    train_labels = {task: arr[train_idx] for task, arr in labels.items()}
    val_labels = {task: arr[val_idx] for task, arr in labels.items()}
    test_labels = {task: arr[test_idx] for task, arr in labels.items()}

    print(f"\nSplits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    train_dataset = MultiTaskGraphDataset(train_graphs, train_labels, tasks)
    val_dataset = MultiTaskGraphDataset(val_graphs, val_labels, tasks)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create GNN model
    print("\n" + "=" * 60)
    print(f"Creating GNN Model ({args.encoder_type.upper()})")
    print("=" * 60)

    atom_feature_dim = get_atom_feature_dim()

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

    # Train
    print("\n" + "=" * 60)
    print(f"Training GNN ({args.epochs} epochs)")
    print("=" * 60)

    config = {
        'learning_rate': args.lr,
        'weight_decay': 0.01,
        'epochs': args.epochs,
        'early_stopping_patience': 15,
        'gradient_log_interval': 5,
        'gradient_clip_norm': 1.0,
    }

    trainer = GNNMultiTaskTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        task_types=tasks,
        config=config,
        device=device,
        output_dir=Path('outputs/toxcast'),
    )

    results = trainer.train()

    # Gradient Conflict Analysis
    print("\n" + "=" * 60)
    print("Gradient Conflict Analysis")
    print("=" * 60)

    G = results['conflict_matrix']
    task_order = results['task_names']

    # Compute empirical correlations
    print("\nComputing empirical correlations...")

    # Save labels as CSV for empirical correlation computation
    labels_df = pd.DataFrame({task: labels[task] for task in task_order})
    labels_csv_path = Path('outputs/toxcast/labels.csv')
    labels_csv_path.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_csv(labels_csv_path, index=False)

    empirical_corr, _ = compute_empirical_correlations(
        str(labels_csv_path),
        task_order,
        min_samples=30
    )

    # Compare G vs empirical
    print("\n" + "=" * 60)
    print("VALIDATION: G vs Empirical Correlations")
    print("=" * 60)

    n_tasks = len(task_order)
    g_values = []
    emp_values = []

    for i in range(n_tasks):
        for j in range(i + 1, n_tasks):
            g_val = G[i, j]
            emp_val = empirical_corr[i, j]
            if not np.isnan(g_val) and not np.isnan(emp_val):
                g_values.append(g_val)
                emp_values.append(emp_val)

    if len(g_values) > 5:
        r, p = stats.pearsonr(g_values, emp_values)
        print(f"\nPearson correlation: r = {r:.3f}, p = {p:.4f}")

        if p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        elif p < 0.05:
            sig = "*"
        else:
            sig = "(n.s.)"

        print(f"Significance: {sig}")

        if r > 0.6:
            print("\n[PASS] Method generalizes to ToxCast (r > 0.6)")
        elif r > 0.4:
            print("\n[PARTIAL] Moderate generalization (0.4 < r < 0.6)")
        else:
            print("\n[FAIL] Weak generalization (r < 0.4)")

        # Save results
        results_summary = {
            'dataset': 'ToxCast_diverse',
            'n_compounds': len(graphs),
            'n_tasks': len(task_order),
            'pearson_r': float(r),
            'pearson_p': float(p),
            'n_pairs': len(g_values),
        }

        np.savez(
            Path('outputs/toxcast/validation_results.npz'),
            G=G,
            empirical=empirical_corr,
            task_names=task_order,
            **results_summary
        )
        print(f"\nSaved results to outputs/toxcast/")

    else:
        print(f"Warning: Only {len(g_values)} valid pairs for correlation")

    # Visualize
    output_dir = Path('outputs/toxcast/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_conflict_heatmap(
        G, task_order,
        title='ToxCast Gradient Conflict Matrix',
        output_path=output_dir / 'toxcast_gradient_conflicts.png'
    )

    print("\n" + "=" * 60)
    print("ToxCast Validation Complete")
    print("=" * 60)


if __name__ == '__main__':
    main()
