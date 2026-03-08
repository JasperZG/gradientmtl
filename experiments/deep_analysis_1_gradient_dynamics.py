#!/usr/bin/env python3
"""
Deep Analysis 1: Gradient Dynamics During Training

Question: When do gradient conflict patterns stabilize during training?
Method: Track gradient matrix at epochs 1, 5, 10, 20, 50, 100
Expected: Patterns emerge early (epoch ~10) and stabilize by epoch ~50
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


def compute_gradient_matrix(model, loader, task_types, task_names, device, n_batches=50):
    """Compute gradient conflict matrix for current model state."""
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

        # Extract labels and masks
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


def train_epoch(model, loader, optimizer, loss_fn, task_names, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

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

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load and preprocess data
    print("\n" + "=" * 60)
    print("Loading Tox21 for Gradient Dynamics Analysis")
    print("=" * 60)

    tox21_path = download_tox21()
    df = pd.read_csv(tox21_path)

    smiles_list = df['smiles'].tolist()
    raw_labels = {}
    for task in TOX21_TASKS:
        if task in df.columns:
            raw_labels[task] = df[task].values.astype(np.float32)

    preprocessor = MoleculeGraphPreprocessor()
    valid_smiles, graphs, valid_indices = preprocessor.process_smiles_list(
        smiles_list, show_progress=True
    )

    labels = {task: values[valid_indices] for task, values in raw_labels.items()}

    # Filter to molecules with 10+ task labels
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

    train_graphs = [graphs[i] for i in train_idx]
    train_labels = {task: arr[train_idx] for task, arr in labels.items()}

    train_dataset = MultiTaskGraphDataset(train_graphs, train_labels, TOX21_TASKS)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Create model
    atom_feature_dim = get_atom_feature_dim()
    task_names = list(TOX21_TASKS.keys())

    model = GNNMultiTaskModel(
        task_names=task_names,
        atom_feature_dim=atom_feature_dim,
        encoder_type='gcn',
        encoder_hidden_dims=[256, 256, 256],
        encoder_output_dim=256,
        head_hidden_dim=128,
        dropout=0.3,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loss_fn = MultiTaskLoss(TOX21_TASKS)

    # Checkpoints to analyze
    checkpoint_epochs = [1, 5, 10, 20, 50, 100]
    gradient_matrices = {}

    print("\n" + "=" * 60)
    print("Training with Gradient Tracking at Checkpoints")
    print("=" * 60)
    print(f"Checkpoints: {checkpoint_epochs}")

    for epoch in range(1, args.max_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, task_names, device)

        if epoch in checkpoint_epochs:
            print(f"\nEpoch {epoch}: Computing gradient matrix...")
            G = compute_gradient_matrix(model, train_loader, TOX21_TASKS, task_names, device)
            gradient_matrices[epoch] = G.copy()

            # Compute key statistics
            upper_tri = G[np.triu_indices(len(task_names), k=1)]
            print(f"  Mean conflict: {np.mean(upper_tri):.4f}")
            print(f"  Std conflict: {np.std(upper_tri):.4f}")
            print(f"  Max conflict: {np.max(upper_tri):.4f}")
            print(f"  Min conflict: {np.min(upper_tri):.4f}")
        else:
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: loss = {train_loss:.4f}")

    # Analyze stability
    print("\n" + "=" * 60)
    print("Stability Analysis: Correlation Between Epochs")
    print("=" * 60)

    results = {
        'checkpoints': checkpoint_epochs,
        'gradient_matrices': {str(e): gradient_matrices[e].tolist() for e in checkpoint_epochs if e in gradient_matrices},
        'task_names': task_names,
        'correlations': {},
        'stability_metrics': {},
    }

    # Compute pairwise correlations between epochs
    epochs_computed = sorted([e for e in checkpoint_epochs if e in gradient_matrices])

    print("\nPairwise correlations between epoch gradient matrices:")
    print(f"{'Epoch 1':>10} {'Epoch 2':>10} {'Pearson r':>12} {'p-value':>12}")
    print("-" * 50)

    for i, e1 in enumerate(epochs_computed):
        for e2 in epochs_computed[i+1:]:
            g1 = gradient_matrices[e1][np.triu_indices(len(task_names), k=1)]
            g2 = gradient_matrices[e2][np.triu_indices(len(task_names), k=1)]
            r, p = stats.pearsonr(g1, g2)
            results['correlations'][f'{e1}_vs_{e2}'] = {'r': r, 'p': p}
            print(f"{e1:>10} {e2:>10} {r:>12.4f} {p:>12.2e}")

    # Compute when patterns stabilize (correlation > 0.9 with final)
    final_epoch = max(epochs_computed)
    final_matrix = gradient_matrices[final_epoch]

    print(f"\nCorrelation with final (epoch {final_epoch}):")
    stabilization_epoch = None

    for e in epochs_computed[:-1]:
        g_e = gradient_matrices[e][np.triu_indices(len(task_names), k=1)]
        g_f = final_matrix[np.triu_indices(len(task_names), k=1)]
        r, _ = stats.pearsonr(g_e, g_f)
        print(f"  Epoch {e:3d}: r = {r:.4f}")
        if r > 0.9 and stabilization_epoch is None:
            stabilization_epoch = e

    results['stability_metrics']['stabilization_epoch'] = stabilization_epoch
    results['stability_metrics']['final_epoch'] = final_epoch

    if stabilization_epoch:
        print(f"\n>>> Patterns stabilize by epoch {stabilization_epoch} (r > 0.9 with final)")
    else:
        print(f"\n>>> Patterns may not have fully stabilized")

    # Save results
    output_dir = Path('outputs/deep_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'gradient_dynamics_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    np.savez(
        output_dir / 'gradient_dynamics_matrices.npz',
        **{f'epoch_{e}': gradient_matrices[e] for e in gradient_matrices},
        task_names=task_names,
        checkpoints=epochs_computed,
    )

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, epoch in enumerate(epochs_computed[:6]):
        if epoch not in gradient_matrices:
            continue
        ax = axes[idx]
        im = ax.imshow(gradient_matrices[epoch], cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title(f'Epoch {epoch}')
        ax.set_xticks(range(len(task_names)))
        ax.set_yticks(range(len(task_names)))
        ax.set_xticklabels([t[:6] for t in task_names], rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels([t[:6] for t in task_names], fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle('Gradient Conflict Matrices During Training', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'gradient_dynamics_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nResults saved to {output_dir}")
    print("Files: gradient_dynamics_results.json, gradient_dynamics_matrices.npz, gradient_dynamics_heatmaps.png")


if __name__ == '__main__':
    main()
