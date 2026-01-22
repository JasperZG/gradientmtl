#!/usr/bin/env python3
"""
Test overlap threshold: What % compound overlap is needed?

Uses Tox21 data with artificially reduced overlap to find the threshold
where gradient conflict correlation breaks down.

Test conditions:
- 100% overlap (baseline)
- 75% overlap
- 50% overlap
- 25% overlap
- 10% overlap
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.graph_preprocessing import MoleculeGraphPreprocessor, get_atom_feature_dim
from data.splitting import scaffold_split
from data.graph_dataset import MultiTaskGraphDataset
from models.gnn_multitask import GNNMultiTaskModel
from training.gnn_trainer import GNNMultiTaskTrainer
from analysis.empirical_correlations import compute_empirical_correlations


TOX21_TASKS = {
    'NR-AR': 'classification',
    'NR-AR-LBD': 'classification',
    'NR-AhR': 'classification',
    'NR-Aromatase': 'classification',
    'NR-ER': 'classification',
    'NR-ER-LBD': 'classification',
    'NR-PPAR-gamma': 'classification',
    'SR-ARE': 'classification',
    'SR-ATAD5': 'classification',
    'SR-HSE': 'classification',
    'SR-MMP': 'classification',
    'SR-p53': 'classification',
}


def create_reduced_overlap_labels(labels: dict, overlap_pct: float, seed: int = 42) -> dict:
    """
    Reduce overlap by randomly masking labels.

    For each compound-task pair, keep label with probability = overlap_pct.
    This simulates datasets with partial compound overlap.
    """
    np.random.seed(seed)

    reduced = {}
    n_compounds = len(next(iter(labels.values())))

    for task, values in labels.items():
        mask = np.random.random(n_compounds) < overlap_pct
        new_values = values.copy()
        new_values[~mask] = np.nan
        reduced[task] = new_values

    return reduced


def run_experiment(overlap_pct: float, graphs: list, labels: dict,
                   train_idx: list, val_idx: list, task_types: dict,
                   device: torch.device, epochs: int = 20) -> dict:
    """Run training with reduced overlap and return correlation."""

    print(f"\n{'='*60}")
    print(f"Testing {int(overlap_pct*100)}% overlap")
    print('='*60)

    # Create reduced labels
    reduced_labels = create_reduced_overlap_labels(labels, overlap_pct)

    # Check actual overlap
    task_names = list(task_types.keys())
    n_compounds = len(graphs)

    # Calculate pairwise overlap
    overlaps = []
    for i, task1 in enumerate(task_names):
        for task2 in task_names[i+1:]:
            mask1 = ~np.isnan(reduced_labels[task1])
            mask2 = ~np.isnan(reduced_labels[task2])
            overlap = (mask1 & mask2).sum()
            overlaps.append(overlap)

    avg_overlap = np.mean(overlaps)
    min_overlap = np.min(overlaps)
    print(f"Actual avg pairwise overlap: {avg_overlap:.0f} compounds")
    print(f"Actual min pairwise overlap: {min_overlap:.0f} compounds")

    # Create datasets
    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    train_labels = {task: arr[train_idx] for task, arr in reduced_labels.items()}
    val_labels = {task: arr[val_idx] for task, arr in reduced_labels.items()}

    train_dataset = MultiTaskGraphDataset(train_graphs, train_labels, task_types)
    val_dataset = MultiTaskGraphDataset(val_graphs, val_labels, task_types)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Create model
    atom_feature_dim = get_atom_feature_dim()
    model = GNNMultiTaskModel(
        task_names=task_names,
        atom_feature_dim=atom_feature_dim,
        encoder_type='gcn',
        encoder_hidden_dims=[256, 256],
        encoder_output_dim=256,
        head_hidden_dim=128,
        dropout=0.3,
    )

    # Train
    output_dir = Path(f'outputs/overlap_test/{int(overlap_pct*100)}pct')
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        'learning_rate': 1e-3,
        'weight_decay': 0.01,
        'epochs': epochs,
        'early_stopping_patience': 10,
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

    G = results['conflict_matrix']
    task_order = results['task_names']

    # Compute empirical correlations
    labels_df = pd.DataFrame({task: reduced_labels[task] for task in task_order})
    labels_csv = output_dir / 'labels.csv'
    labels_df.to_csv(labels_csv, index=False)

    empirical_corr, _ = compute_empirical_correlations(
        str(labels_csv), task_order, min_samples=10
    )

    # Compare G vs empirical
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

    if len(g_values) >= 5:
        r, p = stats.pearsonr(g_values, emp_values)
        print(f"\nResult: r = {r:.3f}, p = {p:.4f}, n = {len(g_values)} pairs")
    else:
        r, p = np.nan, np.nan
        print(f"\nResult: insufficient pairs ({len(g_values)})")

    return {
        'overlap_pct': overlap_pct,
        'r': r,
        'p': p,
        'n_pairs': len(g_values),
        'avg_overlap': avg_overlap,
        'min_overlap': min_overlap,
    }


def main():
    print("="*60)
    print("Overlap Threshold Test")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load Tox21 data
    print("\nLoading Tox21 data...")
    tox21_path = Path('outputs/raw_data/tox21.csv')
    df = pd.read_csv(tox21_path)

    smiles_list = df['smiles'].tolist()
    raw_labels = {}
    for task in TOX21_TASKS:
        if task in df.columns:
            raw_labels[task] = df[task].values.astype(np.float32)

    # Convert to graphs
    print("Converting to graphs...")
    preprocessor = MoleculeGraphPreprocessor()
    valid_smiles, graphs, valid_indices = preprocessor.process_smiles_list(
        smiles_list, show_progress=True
    )

    labels = {task: values[valid_indices] for task, values in raw_labels.items()}

    # Filter to molecules with enough labels
    n_labels_per_mol = np.zeros(len(valid_smiles))
    for task, values in labels.items():
        n_labels_per_mol += ~np.isnan(values)

    mask = n_labels_per_mol >= 10
    keep_indices = np.where(mask)[0]

    graphs = [graphs[i] for i in keep_indices]
    valid_smiles = [valid_smiles[i] for i in keep_indices]
    labels = {task: arr[keep_indices] for task, arr in labels.items()}

    print(f"Using {len(graphs)} molecules with 10+ task labels")

    # Scaffold split
    train_idx, val_idx, test_idx = scaffold_split(
        valid_smiles, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=seed
    )

    # Test different overlap levels
    overlap_levels = [1.0, 0.75, 0.50, 0.25, 0.10]
    results = []

    for overlap in overlap_levels:
        result = run_experiment(
            overlap_pct=overlap,
            graphs=graphs,
            labels=labels,
            train_idx=train_idx,
            val_idx=val_idx,
            task_types=TOX21_TASKS,
            device=device,
            epochs=20,
        )
        results.append(result)

    # Summary
    print("\n" + "="*60)
    print("OVERLAP THRESHOLD RESULTS")
    print("="*60)
    print(f"\n{'Overlap':>10} {'r':>8} {'p':>10} {'Pairs':>6} {'Avg Overlap':>12}")
    print("-"*50)

    for r in results:
        sig = "***" if r['p'] < 0.001 else "**" if r['p'] < 0.01 else "*" if r['p'] < 0.05 else ""
        print(f"{int(r['overlap_pct']*100):>10}% {r['r']:>8.3f} {r['p']:>8.4f}{sig:>2} {r['n_pairs']:>6} {r['avg_overlap']:>12.0f}")

    # Find threshold
    print("\n" + "="*60)
    print("THRESHOLD ANALYSIS")
    print("="*60)

    for r in results:
        if r['r'] >= 0.6 and r['p'] < 0.05:
            status = "PASS"
        elif r['r'] >= 0.4:
            status = "MARGINAL"
        else:
            status = "FAIL"
        print(f"{int(r['overlap_pct']*100)}% overlap: {status} (r = {r['r']:.3f})")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('outputs/overlap_test/threshold_results.csv', index=False)
    print(f"\nSaved to outputs/overlap_test/threshold_results.csv")


if __name__ == '__main__':
    main()
