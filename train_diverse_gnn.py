#!/usr/bin/env python3
"""
Strategy B: Train on diverse property types.

Tests whether gradient conflicts work across DIFFERENT property categories:
- Physical Chemistry: Lipophilicity, ESOL, FreeSolv
- Physiology: BBBP (permeability)
- Biophysics: BACE (binding), HIV (activity)

Key test: Do gradient conflicts between DIFFERENT property types
correlate with empirical correlations?

Success criteria: r > 0.5 for cross-category pairs
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
from analysis.visualization import plot_conflict_heatmap
from analysis.empirical_correlations import compute_empirical_correlations


# Task definitions with property categories
DIVERSE_TASKS = {
    # Physical Chemistry (regression)
    'Lipophilicity': {'type': 'regression', 'category': 'Physical'},
    'ESOL': {'type': 'regression', 'category': 'Physical'},
    'FreeSolv': {'type': 'regression', 'category': 'Physical'},
    # Physiology (classification)
    'BBBP': {'type': 'classification', 'category': 'Physiology'},
    # Biophysics (classification)
    'BACE': {'type': 'classification', 'category': 'Biophysics'},
    'HIV': {'type': 'classification', 'category': 'Biophysics'},
}


def load_diverse_dataset():
    """Load the curated diverse properties dataset."""
    data_path = Path('outputs/diverse_properties/diverse_properties.csv')

    if not data_path.exists():
        print("Diverse properties dataset not found.")
        print("Run: python scripts/curate_diverse_properties.py")
        raise FileNotFoundError(data_path)

    df = pd.read_csv(data_path)
    print(f"Loaded: {len(df)} compounds")

    # Get available tasks
    available_tasks = {}
    for col in df.columns:
        if col != 'smiles' and col in DIVERSE_TASKS:
            n_valid = df[col].notna().sum()
            if n_valid >= 30:  # Minimum samples
                available_tasks[col] = DIVERSE_TASKS[col]['type']
                category = DIVERSE_TASKS[col]['category']
                print(f"  {col} ({category}): {n_valid} samples")

    return df, available_tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)  # Smaller for small dataset
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--min_tasks', type=int, default=2)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    print("\n" + "=" * 60)
    print("Loading Diverse Properties Dataset")
    print("=" * 60)

    df, task_types = load_diverse_dataset()
    task_names = list(task_types.keys())

    if len(task_names) < 3:
        print("Error: Need at least 3 tasks with sufficient data")
        return

    smiles_list = df['smiles'].tolist()
    raw_labels = {task: df[task].values.astype(np.float32) for task in task_names}

    # Convert SMILES to graphs
    print("\n" + "=" * 60)
    print("Converting SMILES to Molecular Graphs")
    print("=" * 60)

    preprocessor = MoleculeGraphPreprocessor()
    valid_smiles, graphs, valid_indices = preprocessor.process_smiles_list(
        smiles_list, show_progress=True
    )

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

    graphs = [graphs[i] for i in keep_indices]
    valid_smiles = [valid_smiles[i] for i in keep_indices]
    labels = {task: arr[keep_indices] for task, arr in labels.items()}

    if len(graphs) < 100:
        print("Warning: Very small dataset. Results may be unstable.")

    # Scaffold split
    print("\n" + "=" * 60)
    print("Scaffold Split")
    print("=" * 60)

    train_idx, val_idx, test_idx = scaffold_split(
        valid_smiles, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=seed
    )

    # Create datasets
    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]

    train_labels = {task: arr[train_idx] for task, arr in labels.items()}
    val_labels = {task: arr[val_idx] for task, arr in labels.items()}

    print(f"\nSplits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    train_dataset = MultiTaskGraphDataset(train_graphs, train_labels, task_types)
    val_dataset = MultiTaskGraphDataset(val_graphs, val_labels, task_types)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    print("\n" + "=" * 60)
    print("Creating GNN Model")
    print("=" * 60)

    atom_feature_dim = get_atom_feature_dim()

    model = GNNMultiTaskModel(
        task_names=task_names,
        atom_feature_dim=atom_feature_dim,
        encoder_type='gcn',
        encoder_hidden_dims=[128, 128],  # Smaller for small dataset
        encoder_output_dim=128,
        head_hidden_dim=64,
        dropout=0.3,
    )

    print(model.summary())

    # Train
    print("\n" + "=" * 60)
    print(f"Training GNN ({args.epochs} epochs)")
    print("=" * 60)

    output_dir = Path('outputs/diverse')
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        'learning_rate': args.lr,
        'weight_decay': 0.01,
        'epochs': args.epochs,
        'early_stopping_patience': 15,
        'gradient_log_interval': 3,  # More frequent for small dataset
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

    # Analysis
    print("\n" + "=" * 60)
    print("Strategy B Validation: Cross-Category Gradient Conflicts")
    print("=" * 60)

    G = results['conflict_matrix']
    task_order = results['task_names']

    # Compute empirical correlations
    labels_df = pd.DataFrame({task: labels[task] for task in task_order})
    labels_csv = output_dir / 'labels.csv'
    labels_df.to_csv(labels_csv, index=False)

    empirical_corr, _ = compute_empirical_correlations(
        str(labels_csv), task_order, min_samples=20
    )

    # Analyze by category
    print("\nCross-category analysis:")

    cross_category_g = []
    cross_category_emp = []
    same_category_g = []
    same_category_emp = []

    n_tasks = len(task_order)
    for i in range(n_tasks):
        for j in range(i + 1, n_tasks):
            g_val = G[i, j]
            emp_val = empirical_corr[i, j]

            if np.isnan(g_val) or np.isnan(emp_val):
                continue

            task_i = task_order[i]
            task_j = task_order[j]
            cat_i = DIVERSE_TASKS.get(task_i, {}).get('category', 'Unknown')
            cat_j = DIVERSE_TASKS.get(task_j, {}).get('category', 'Unknown')

            if cat_i != cat_j:
                cross_category_g.append(g_val)
                cross_category_emp.append(emp_val)
                print(f"  {task_i} x {task_j}: G={g_val:.3f}, Emp={emp_val:.3f} [{cat_i} x {cat_j}]")
            else:
                same_category_g.append(g_val)
                same_category_emp.append(emp_val)

    # Compute correlations
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    if len(cross_category_g) >= 3:
        r_cross, p_cross = stats.pearsonr(cross_category_g, cross_category_emp)
        print(f"\nCross-category pairs (n={len(cross_category_g)}):")
        print(f"  Pearson r = {r_cross:.3f}, p = {p_cross:.4f}")

        if r_cross > 0.5 and p_cross < 0.05:
            print("  [PASS] Gradient conflicts work across property types!")
        elif r_cross > 0.3:
            print("  [PARTIAL] Moderate cross-category correlation")
        else:
            print("  [FAIL] Weak cross-category correlation")
    else:
        print(f"\nInsufficient cross-category pairs: {len(cross_category_g)}")
        r_cross = np.nan

    if len(same_category_g) >= 3:
        r_same, p_same = stats.pearsonr(same_category_g, same_category_emp)
        print(f"\nSame-category pairs (n={len(same_category_g)}):")
        print(f"  Pearson r = {r_same:.3f}, p = {p_same:.4f}")
    else:
        r_same = np.nan

    # Overall correlation
    all_g = cross_category_g + same_category_g
    all_emp = cross_category_emp + same_category_emp

    if len(all_g) >= 5:
        r_all, p_all = stats.pearsonr(all_g, all_emp)
        print(f"\nOverall (n={len(all_g)}):")
        print(f"  Pearson r = {r_all:.3f}, p = {p_all:.4f}")

        # Save results
        np.savez(
            output_dir / 'validation_results.npz',
            G=G,
            empirical=empirical_corr,
            task_names=task_order,
            r_cross=r_cross,
            r_same=r_same,
            r_all=r_all,
        )

    # Visualize
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)
    plot_conflict_heatmap(
        G, task_order,
        title='Diverse Properties Gradient Conflicts',
        output_path=fig_dir / 'diverse_gradient_conflicts.png'
    )

    print("\n" + "=" * 60)
    print("Strategy B Complete")
    print("=" * 60)


if __name__ == '__main__':
    main()
