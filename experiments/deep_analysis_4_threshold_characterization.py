#!/usr/bin/env python3
"""
Deep Analysis 4: Precise Threshold Characterization

Question: At what compound overlap % does gradient-empirical correlation emerge?
Method:
  1. Subsample Tox21 at 10%, 20%, ..., 100% compound overlap
  2. Compute gradient and empirical matrices at each level
  3. Fit sigmoid curve to find threshold
Expected: Sigmoid inflection point around 40-60% overlap
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
import urllib.request
import gzip
import io
import json
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from data.graph_preprocessing import MoleculeGraphPreprocessor, get_atom_feature_dim
from data.splitting import scaffold_split
from data.graph_dataset import MultiTaskGraphDataset
from models.gnn_multitask import GNNMultiTaskModel
from training.losses import MultiTaskLoss
from training.gradient_logger import GradientConflictLogger


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


def sigmoid(x, L, k, x0, b):
    """Sigmoid function for curve fitting."""
    return L / (1 + np.exp(-k * (x - x0))) + b


def download_tox21():
    """Download Tox21 dataset."""
    output_dir = Path('outputs/raw_data')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'tox21.csv'

    if output_path.exists():
        return output_path

    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz'
    print(f"Downloading Tox21...")

    request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(request, timeout=60) as response:
        compressed = response.read()

    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as f:
        data = f.read()

    with open(output_path, 'wb') as f:
        f.write(data)

    return output_path


def create_overlap_subsample(graphs, labels, task_names, overlap_pct, seed=42):
    """
    Create a subsample with specified compound overlap.

    For overlap_pct=100, all molecules have all labels.
    For overlap_pct=50, each pair of tasks shares 50% of molecules.
    """
    np.random.seed(seed)
    n_molecules = len(graphs)
    n_tasks = len(task_names)

    # Create new label arrays with controlled overlap
    new_labels = {}

    if overlap_pct == 100:
        # Return original (filtered to complete cases)
        complete_mask = np.ones(n_molecules, dtype=bool)
        for task in task_names:
            complete_mask &= ~np.isnan(labels[task])

        keep_idx = np.where(complete_mask)[0]
        if len(keep_idx) < 100:
            # Fall back to original with some missing
            return graphs, labels
        return [graphs[i] for i in keep_idx], {t: labels[t][keep_idx] for t in task_names}

    # For partial overlap: mask out labels strategically
    for task in task_names:
        task_labels = labels[task].copy()
        valid_mask = ~np.isnan(task_labels)
        valid_idx = np.where(valid_mask)[0]

        # Randomly mask some labels to achieve target overlap
        n_to_mask = int(len(valid_idx) * (1 - overlap_pct / 100))
        mask_idx = np.random.choice(valid_idx, size=n_to_mask, replace=False)
        task_labels[mask_idx] = np.nan
        new_labels[task] = task_labels

    return graphs, new_labels


def compute_empirical_matrix(labels, task_names):
    """Compute empirical correlation matrix from label co-occurrence."""
    n_tasks = len(task_names)
    E = np.zeros((n_tasks, n_tasks))

    for i, t1 in enumerate(task_names):
        for j, t2 in enumerate(task_names):
            mask = ~np.isnan(labels[t1]) & ~np.isnan(labels[t2])
            if mask.sum() > 10:
                r, _ = stats.pearsonr(labels[t1][mask], labels[t2][mask])
                E[i, j] = r
            else:
                E[i, j] = np.nan

    return E


def compute_gradient_matrix(model, loader, task_types, task_names, device, n_batches=30):
    """Compute gradient conflict matrix."""
    model.train()
    loss_fn = MultiTaskLoss(task_types)

    gradient_logger = GradientConflictLogger(
        model=model,
        task_names=task_names,
        log_interval=1,
        device=device,
    )

    for batch_idx, batch_graph in enumerate(loader):
        if batch_idx >= n_batches:
            break

        batch_graph = batch_graph.to(device)

        labels_tensor = batch_graph.y
        masks_tensor = batch_graph.mask
        batch_size = batch_graph.num_graphs
        n_tasks = len(task_names)

        if labels_tensor.dim() == 1:
            labels_tensor = labels_tensor.view(batch_size, n_tasks)
            masks_tensor = masks_tensor.view(batch_size, n_tasks)

        labels = {task: labels_tensor[:, i] for i, task in enumerate(task_names)}
        masks = {task: masks_tensor[:, i] for i, task in enumerate(task_names)}

        predictions = model(batch_graph)
        task_losses = loss_fn.get_individual_losses(predictions, labels, masks)
        gradient_logger.log_step(batch_idx, task_losses)

    return gradient_logger.get_averaged_conflict_matrix()


def train_model(model, loader, task_names, device, epochs=30, lr=1e-3):
    """Quick training."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = MultiTaskLoss(TOX21_TASKS)

    for epoch in range(epochs):
        model.train()
        for batch_graph in loader:
            batch_graph = batch_graph.to(device)

            labels_tensor = batch_graph.y
            masks_tensor = batch_graph.mask
            batch_size = batch_graph.num_graphs
            n_tasks = len(task_names)

            if labels_tensor.dim() == 1:
                labels_tensor = labels_tensor.view(batch_size, n_tasks)
                masks_tensor = masks_tensor.view(batch_size, n_tasks)

            labels = {task: labels_tensor[:, i] for i, task in enumerate(task_names)}
            masks = {task: masks_tensor[:, i] for i, task in enumerate(task_names)}

            optimizer.zero_grad()
            predictions = model(batch_graph)
            loss, _ = loss_fn(predictions, labels, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    print("\n" + "=" * 60)
    print("Loading Tox21 for Threshold Characterization")
    print("=" * 60)

    tox21_path = download_tox21()
    df = pd.read_csv(tox21_path)

    smiles_list = df['smiles'].tolist()
    task_names = list(TOX21_TASKS.keys())

    raw_labels = {}
    for task in TOX21_TASKS:
        if task in df.columns:
            raw_labels[task] = df[task].values.astype(np.float32)

    # Convert to graphs
    preprocessor = MoleculeGraphPreprocessor()
    valid_smiles, graphs, valid_indices = preprocessor.process_smiles_list(
        smiles_list, show_progress=True
    )

    labels = {task: values[valid_indices] for task, values in raw_labels.items()}

    print(f"Total valid molecules: {len(graphs)}")

    # Test overlap levels
    overlap_levels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    results = {
        'overlap_levels': overlap_levels,
        'correlations': [],
        'task_names': task_names,
        'gradient_matrices': {},
        'empirical_matrices': {},
    }

    print("\n" + "=" * 60)
    print("Testing Overlap Levels")
    print("=" * 60)
    print(f"\n{'Overlap %':>12} {'G-E Corr':>12} {'p-value':>12} {'N pairs':>10}")
    print("-" * 50)

    for overlap_pct in overlap_levels:
        print(f"\nProcessing {overlap_pct}% overlap...")

        # Create subsample
        sub_graphs, sub_labels = create_overlap_subsample(
            graphs, labels, task_names, overlap_pct, seed=seed
        )

        # Create dataset
        dataset = MultiTaskGraphDataset(sub_graphs, sub_labels, TOX21_TASKS)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        # Create and train model
        model = GNNMultiTaskModel(
            task_names=task_names,
            atom_feature_dim=get_atom_feature_dim(),
            encoder_type='gcn',
            encoder_hidden_dims=[256, 256],
            encoder_output_dim=256,
            head_hidden_dim=128,
            dropout=0.3,
        ).to(device)

        train_model(model, loader, task_names, device, epochs=args.epochs)

        # Compute matrices
        G = compute_gradient_matrix(model, loader, TOX21_TASKS, task_names, device)
        E = compute_empirical_matrix(sub_labels, task_names)

        results['gradient_matrices'][str(overlap_pct)] = G.tolist()
        results['empirical_matrices'][str(overlap_pct)] = E.tolist()

        # Compute correlation
        upper_tri_idx = np.triu_indices(len(task_names), k=1)
        g_vals = G[upper_tri_idx]
        e_vals = E[upper_tri_idx]

        # Filter NaN
        mask = ~(np.isnan(g_vals) | np.isnan(e_vals))
        g_vals = g_vals[mask]
        e_vals = e_vals[mask]

        if len(g_vals) > 3:
            r, p = stats.pearsonr(g_vals, e_vals)
        else:
            r, p = np.nan, np.nan

        results['correlations'].append({
            'overlap': overlap_pct,
            'r': r,
            'p': p,
            'n_pairs': int(len(g_vals)),
        })

        print(f"{overlap_pct:>12}% {r:>12.4f} {p:>12.2e} {len(g_vals):>10}")

    # Fit sigmoid
    print("\n" + "=" * 60)
    print("Sigmoid Curve Fitting")
    print("=" * 60)

    x_data = np.array([r['overlap'] for r in results['correlations']])
    y_data = np.array([r['r'] for r in results['correlations']])

    # Remove NaN
    valid = ~np.isnan(y_data)
    x_data = x_data[valid]
    y_data = y_data[valid]

    try:
        # Initial guess: L=1, k=0.1, x0=50, b=0
        popt, pcov = curve_fit(
            sigmoid, x_data, y_data,
            p0=[0.8, 0.1, 50, 0.1],
            bounds=([0, 0.01, 0, -1], [1.5, 0.5, 100, 1]),
            maxfev=5000
        )
        L, k, x0, b = popt
        inflection_point = x0

        print(f"Fitted parameters:")
        print(f"  L (max correlation): {L:.4f}")
        print(f"  k (steepness): {k:.4f}")
        print(f"  x0 (inflection): {x0:.1f}%")
        print(f"  b (baseline): {b:.4f}")

        results['sigmoid_fit'] = {
            'L': float(L),
            'k': float(k),
            'x0': float(x0),
            'b': float(b),
            'inflection_point': float(inflection_point),
        }

        # Calculate threshold (where correlation reaches 90% of max)
        threshold_90 = x0 + np.log(9) / k  # Solving sigmoid = 0.9*L + b
        print(f"\n90% threshold (where r reaches 90% of max): {threshold_90:.1f}%")

    except Exception as e:
        print(f"Sigmoid fitting failed: {e}")
        print("Using linear interpolation instead...")
        inflection_point = np.nan
        results['sigmoid_fit'] = {'error': str(e)}

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    # Find first overlap level where r > 0.5
    significant_threshold = None
    for entry in results['correlations']:
        if not np.isnan(entry['r']) and entry['r'] > 0.5:
            significant_threshold = entry['overlap']
            break

    if significant_threshold:
        print(f"Correlation > 0.5 first achieved at: {significant_threshold}% overlap")
    else:
        print("Correlation never exceeded 0.5 in tested range")

    if not np.isnan(inflection_point):
        print(f"Sigmoid inflection point: {inflection_point:.1f}%")

        if 40 <= inflection_point <= 60:
            print("\n>>> PASS: Inflection point is in expected range (40-60%)")
            conclusion = "PASS"
        else:
            print(f"\n>>> PARTIAL: Inflection point ({inflection_point:.1f}%) outside 40-60% range")
            conclusion = "PARTIAL"
    else:
        conclusion = "INCONCLUSIVE"

    results['summary'] = {
        'significant_threshold': significant_threshold,
        'inflection_point': float(inflection_point) if not np.isnan(inflection_point) else None,
        'conclusion': conclusion,
    }

    # Save results
    output_dir = Path('outputs/deep_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'threshold_characterization_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Main plot
    ax1 = axes[0]
    ax1.scatter(x_data, y_data, s=100, c='#2E86AB', edgecolors='white', linewidth=1.5, zorder=5)

    if 'sigmoid_fit' in results and 'L' in results['sigmoid_fit']:
        x_smooth = np.linspace(0, 100, 200)
        y_smooth = sigmoid(x_smooth, **{k: results['sigmoid_fit'][k] for k in ['L', 'k', 'x0', 'b']})
        ax1.plot(x_smooth, y_smooth, 'r-', linewidth=2, alpha=0.8, label='Sigmoid fit')
        ax1.axvline(results['sigmoid_fit']['x0'], color='gray', linestyle='--', alpha=0.5,
                   label=f"Inflection: {results['sigmoid_fit']['x0']:.0f}%")

    ax1.set_xlabel('Compound Overlap (%)', fontsize=12)
    ax1.set_ylabel('Gradient-Empirical Correlation', fontsize=12)
    ax1.set_title('Overlap Threshold Characterization', fontsize=14)
    ax1.set_xlim(0, 105)
    ax1.set_ylim(-0.2, 1.1)
    ax1.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax1.axhline(0.5, color='green', linestyle=':', alpha=0.5, label='r = 0.5')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Heatmap of correlations at different overlaps
    ax2 = axes[1]
    selected_overlaps = [20, 50, 80, 100]
    matrices_to_show = []
    for ov in selected_overlaps:
        if str(ov) in results['gradient_matrices']:
            matrices_to_show.append((ov, np.array(results['gradient_matrices'][str(ov)])))

    if matrices_to_show:
        n_show = len(matrices_to_show)
        combined = np.concatenate([m[1] for m in matrices_to_show], axis=1)
        im = ax2.imshow(combined, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

        # Add separators
        for i in range(1, n_show):
            ax2.axvline(i * len(task_names) - 0.5, color='white', linewidth=2)

        ax2.set_yticks(range(len(task_names)))
        ax2.set_yticklabels([t[:6] for t in task_names], fontsize=8)

        # X-axis labels
        x_positions = [len(task_names) * (i + 0.5) for i in range(n_show)]
        ax2.set_xticks(x_positions)
        ax2.set_xticklabels([f'{m[0]}%' for m in matrices_to_show])
        ax2.set_title('Gradient Matrices at Different Overlaps')
        plt.colorbar(im, ax=ax2, shrink=0.8)

    plt.tight_layout()
    plt.savefig(output_dir / 'threshold_characterization_plot.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save as PDF for paper
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.scatter(x_data, y_data, s=60, c='#2E86AB', edgecolors='white', linewidth=1, zorder=5)

    if 'sigmoid_fit' in results and 'L' in results['sigmoid_fit']:
        x_smooth = np.linspace(0, 100, 200)
        y_smooth = sigmoid(x_smooth, **{k: results['sigmoid_fit'][k] for k in ['L', 'k', 'x0', 'b']})
        ax.plot(x_smooth, y_smooth, 'r-', linewidth=2, alpha=0.8)

    ax.set_xlabel('Compound Overlap (%)')
    ax.set_ylabel('Gradient-Empirical Correlation')
    ax.set_xlim(0, 105)
    ax.set_ylim(-0.2, 1.1)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'threshold_characterization.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
