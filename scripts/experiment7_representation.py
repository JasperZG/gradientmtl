#!/usr/bin/env python3
"""
Experiment 7: Representation Generalization

Tests whether gradient conflicts are consistent across different molecular
representations (ECFP fingerprints vs GNN graphs).

Key hypothesis: If gradient conflicts reflect genuine mechanistic relationships,
they should be consistent across representation types.

Comparisons:
1. ECFP4 (2048-bit fingerprints) - baseline
2. GCN graphs - primary method
3. Correlation between G_ECFP and G_GNN

Expected outcome: Pearson r > 0.8 between representation types
"""

import os
import sys
import json
import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.preprocessing import MoleculePreprocessor
from data.dataset import MultiTaskMoleculeDataset as MultiTaskDataset
from models.encoder import SharedEncoder
from models.heads import TaskHead
from models.multitask import MultiTaskModel
from training.losses import MultiTaskLoss
from training.gradient_logger import GradientConflictLogger


# =============================================================================
# ECFP-based Multi-task Model
# =============================================================================

class ECFPMultiTaskModel(nn.Module):
    """Multi-task model using ECFP fingerprints."""

    def __init__(
        self,
        task_names: list,
        input_dim: int = 2048,
        encoder_hidden_dims: list = [1024, 512, 256],
        head_hidden_dim: int = 128,
        dropout: float = 0.2
    ):
        super().__init__()
        self.task_names = task_names

        # Shared encoder (MLP)
        layers = []
        prev_dim = input_dim
        for hidden_dim in encoder_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)
        self.encoder_output_dim = encoder_hidden_dims[-1]

        # Task-specific heads
        self.heads = nn.ModuleDict({
            task: TaskHead(
                input_dim=self.encoder_output_dim,
                hidden_dim=head_hidden_dim
            )
            for task in task_names
        })

    def forward(self, x):
        """Forward pass."""
        h = self.encoder(x)

        outputs = {}
        for task in self.task_names:
            outputs[task] = self.heads[task](h)

        return outputs

    def get_encoder_parameters(self):
        """Get encoder parameters for gradient logging."""
        return self.encoder.parameters()


def train_ecfp_model(
    fingerprints: np.ndarray,
    labels: dict,
    task_types: dict,
    config: dict,
    device: str = 'cuda'
) -> tuple:
    """
    Train ECFP-based multi-task model with gradient logging.

    Returns:
        model, gradient_logger
    """
    task_names = list(labels.keys())

    # Prepare data
    X = torch.FloatTensor(fingerprints)

    # Stack labels and create masks
    label_arrays = []
    mask_arrays = []
    for task in task_names:
        y = labels[task]
        label_arrays.append(y)
        mask_arrays.append(~np.isnan(y))

    Y = torch.FloatTensor(np.column_stack(label_arrays))
    Y = torch.nan_to_num(Y, nan=0.0)
    M = torch.FloatTensor(np.column_stack(mask_arrays))

    # Simple train/val split
    n = len(X)
    indices = np.random.permutation(n)
    train_idx = indices[:int(0.8*n)]
    val_idx = indices[int(0.8*n):]

    train_dataset = TensorDataset(X[train_idx], Y[train_idx], M[train_idx])
    val_dataset = TensorDataset(X[val_idx], Y[val_idx], M[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

    # Create model
    model = ECFPMultiTaskModel(
        task_names=task_names,
        input_dim=fingerprints.shape[1],
        encoder_hidden_dims=config.get('encoder_hidden_dims', [1024, 512, 256]),
        head_hidden_dim=config.get('head_hidden_dim', 128),
        dropout=config.get('dropout', 0.2)
    ).to(device)

    # Loss and optimizer
    loss_fn = MultiTaskLoss(task_types)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Gradient logger
    gradient_logger = GradientConflictLogger(
        model=model,
        task_names=task_names,
        log_interval=config.get('gradient_log_interval', 10),
        device=torch.device(device)
    )

    # Training loop
    print(f"\nTraining ECFP model for {config['epochs']} epochs...")
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config['epochs']):
        model.train()
        train_losses = []

        for batch_idx, (x, y, m) in enumerate(train_loader):
            x, y, m = x.to(device), y.to(device), m.to(device)
            global_step = epoch * len(train_loader) + batch_idx

            optimizer.zero_grad()
            outputs = model(x)

            # Convert to dict format
            labels_dict = {task: y[:, i] for i, task in enumerate(task_names)}
            masks_dict = {task: m[:, i] for i, task in enumerate(task_names)}

            # Get individual losses for gradient logging
            task_losses = loss_fn.get_individual_losses(outputs, labels_dict, masks_dict)

            # Log gradients
            gradient_logger.log_step(global_step, task_losses)

            # Compute total loss
            total_loss, _ = loss_fn(outputs, labels_dict, masks_dict)
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(total_loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y, m in val_loader:
                x, y, m = x.to(device), y.to(device), m.to(device)
                outputs = model(x)
                labels_dict = {task: y[:, i] for i, task in enumerate(task_names)}
                masks_dict = {task: m[:, i] for i, task in enumerate(task_names)}
                total_loss, _ = loss_fn(outputs, labels_dict, masks_dict)
                val_losses.append(total_loss.item())

        avg_val_loss = np.mean(val_losses)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: train_loss={np.mean(train_losses):.4f}, val_loss={avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.get('early_stopping_patience', 20):
                print(f"Early stopping at epoch {epoch}")
                break

    return model, gradient_logger


def compare_gradient_matrices(
    G1: np.ndarray,
    G2: np.ndarray,
    task_names: list,
    label1: str = 'ECFP',
    label2: str = 'GNN'
) -> dict:
    """
    Compare two gradient conflict matrices.

    Returns:
        Dict with comparison metrics
    """
    K = len(task_names)
    mask = ~np.eye(K, dtype=bool)

    g1_flat = G1[mask]
    g2_flat = G2[mask]

    # Correlation
    pearson_r, pearson_p = stats.pearsonr(g1_flat, g2_flat)
    spearman_r, spearman_p = stats.spearmanr(g1_flat, g2_flat)

    # MAE
    mae = np.mean(np.abs(g1_flat - g2_flat))

    # Sign agreement
    sign_agree = np.mean(np.sign(g1_flat) == np.sign(g2_flat))

    # Find largest discrepancies
    diffs = np.abs(G1 - G2)
    np.fill_diagonal(diffs, 0)
    flat_idx = np.argsort(diffs.flatten())[::-1]

    discrepancies = []
    for flat_i in flat_idx[:10]:
        i, j = flat_i // K, flat_i % K
        if i < j:
            discrepancies.append({
                'task1': task_names[i],
                'task2': task_names[j],
                f'{label1}_value': float(G1[i, j]),
                f'{label2}_value': float(G2[i, j]),
                'difference': float(diffs[i, j])
            })

    return {
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'mae': float(mae),
        'sign_agreement': float(sign_agree),
        'largest_discrepancies': discrepancies[:5]
    }


def plot_representation_comparison(
    G_ecfp: np.ndarray,
    G_gnn: np.ndarray,
    task_names: list,
    output_path: Path
):
    """Scatter plot comparing gradient matrices."""
    K = len(task_names)
    mask = ~np.eye(K, dtype=bool)

    g_ecfp = G_ecfp[mask]
    g_gnn = G_gnn[mask]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    ax1 = axes[0]
    ax1.scatter(g_ecfp, g_gnn, alpha=0.5, s=30)
    ax1.plot([-1, 1], [-1, 1], 'r--', label='y = x')

    # Add regression line
    slope, intercept, r, p, _ = stats.linregress(g_ecfp, g_gnn)
    x_line = np.array([-1, 1])
    ax1.plot(x_line, slope * x_line + intercept, 'b-',
             label=f'r = {r:.3f}')

    ax1.set_xlabel('ECFP Gradient Conflict', fontsize=12)
    ax1.set_ylabel('GNN Gradient Conflict', fontsize=12)
    ax1.set_title('Representation Comparison', fontsize=14)
    ax1.legend()
    ax1.set_xlim(-1.1, 1.1)
    ax1.set_ylim(-1.1, 1.1)

    # Difference heatmap
    ax2 = axes[1]
    diff = np.abs(G_ecfp - G_gnn)
    im = ax2.imshow(diff, cmap='Reds', vmin=0, vmax=0.5)
    plt.colorbar(im, ax=ax2, shrink=0.8, label='|Difference|')

    ax2.set_xticks(range(K))
    ax2.set_yticks(range(K))
    ax2.set_xticklabels(task_names, rotation=90, fontsize=8)
    ax2.set_yticklabels(task_names, fontsize=8)
    ax2.set_title('Absolute Difference |ECFP - GNN|', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {output_path}")


def run_representation_experiment(
    gnn_matrix_path: str,
    output_dir: str = 'outputs/representation',
    seed: int = 42,
    epochs: int = 100,
    verbose: bool = True
) -> dict:
    """
    Run representation generalization experiment.

    Args:
        gnn_matrix_path: Path to pre-computed GNN gradient matrix
        output_dir: Output directory
        seed: Random seed
        epochs: Training epochs for ECFP model
        verbose: Print detailed output

    Returns:
        Dict with comparison results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("EXPERIMENT 7: REPRESENTATION GENERALIZATION")
    print("=" * 70)

    np.random.seed(seed)
    torch.manual_seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load GNN gradient matrix
    print(f"\nLoading GNN gradient matrix from {gnn_matrix_path}...")
    gnn_data = np.load(gnn_matrix_path, allow_pickle=True)
    G_gnn = gnn_data['averaged']
    task_names = gnn_data['task_names'].tolist()
    print(f"Loaded {len(task_names)} tasks")

    # Load Tox21 data and compute ECFP fingerprints
    print("\nLoading Tox21 data and computing ECFP fingerprints...")

    tox21_path = project_root / 'outputs' / 'raw_data' / 'tox21.csv'
    if not tox21_path.exists():
        raise FileNotFoundError(f"Tox21 data not found at {tox21_path}. Run pre-training first.")
    df = pd.read_csv(tox21_path)

    # Compute fingerprints
    preprocessor = MoleculePreprocessor()
    smiles_list = df['smiles'].tolist()
    valid_smiles, fingerprints, valid_idx = preprocessor.process_smiles_list(
        smiles_list, show_progress=True
    )

    print(f"Valid molecules: {len(valid_smiles)}")

    # Get labels for valid molecules
    labels = {}
    task_types = {}
    for task in task_names:
        if task in df.columns:
            labels[task] = df[task].values[valid_idx].astype(np.float32)
            task_types[task] = 'classification'
        elif task.replace('Tox21_', '') in df.columns:
            col = task.replace('Tox21_', '')
            labels[task] = df[col].values[valid_idx].astype(np.float32)
            task_types[task] = 'classification'

    # Filter to tasks present in both
    common_tasks = [t for t in task_names if t in labels]
    print(f"Common tasks: {len(common_tasks)}")

    if len(common_tasks) < 5:
        print("ERROR: Not enough common tasks for comparison")
        return {'error': 'Insufficient common tasks'}

    # Train ECFP model
    config = {
        'batch_size': 32,
        'learning_rate': 1e-3,
        'weight_decay': 0.01,
        'epochs': epochs,
        'early_stopping_patience': 20,
        'gradient_log_interval': 10,
        'encoder_hidden_dims': [1024, 512, 256],
        'head_hidden_dim': 128,
        'dropout': 0.2,
    }

    filtered_labels = {t: labels[t] for t in common_tasks}
    filtered_types = {t: task_types[t] for t in common_tasks}

    model, gradient_logger = train_ecfp_model(
        fingerprints=fingerprints,
        labels=filtered_labels,
        task_types=filtered_types,
        config=config,
        device=device
    )

    G_ecfp = gradient_logger.get_averaged_conflict_matrix()

    # Save ECFP gradient matrix
    np.savez(
        output_dir / 'ecfp_conflict_matrices.npz',
        averaged=G_ecfp,
        task_names=np.array(common_tasks),
        history=gradient_logger.get_conflict_history()
    )
    print(f"Saved ECFP gradient matrix")

    # Extract common task subset from GNN matrix
    gnn_indices = [task_names.index(t) for t in common_tasks if t in task_names]
    G_gnn_subset = G_gnn[np.ix_(gnn_indices, gnn_indices)]

    # Compare matrices
    print("\nComparing gradient matrices...")
    comparison = compare_gradient_matrices(
        G_ecfp, G_gnn_subset, common_tasks,
        label1='ECFP', label2='GNN'
    )

    # Generate plots
    plot_representation_comparison(
        G_ecfp, G_gnn_subset, common_tasks,
        output_dir / 'representation_comparison.png'
    )

    # Results
    results = {
        'n_tasks': len(common_tasks),
        'task_names': common_tasks,
        'comparison': comparison,
        'ecfp_training_config': config,
    }

    # Print summary
    if verbose:
        print("\n" + "-" * 50)
        print("COMPARISON RESULTS")
        print("-" * 50)
        print(f"Pearson correlation: r = {comparison['pearson_r']:.4f} (p = {comparison['pearson_p']:.2e})")
        print(f"Spearman correlation: rho = {comparison['spearman_r']:.4f}")
        print(f"Mean absolute error: {comparison['mae']:.4f}")
        print(f"Sign agreement: {comparison['sign_agreement']:.1%}")

        print("\nLargest discrepancies:")
        for disc in comparison['largest_discrepancies'][:5]:
            print(f"  {disc['task1']} vs {disc['task2']}: "
                  f"ECFP={disc['ECFP_value']:.3f}, GNN={disc['GNN_value']:.3f}, "
                  f"diff={disc['difference']:.3f}")

    # Save results
    results_file = output_dir / 'representation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Check success criteria
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 70)

    if comparison['pearson_r'] > 0.8:
        print(f"[PASS] Pearson r > 0.8: PASS ({comparison['pearson_r']:.3f})")
        results['representation_consistent'] = True
    elif comparison['pearson_r'] > 0.6:
        print(f"[PARTIAL] Pearson r > 0.8: PARTIAL ({comparison['pearson_r']:.3f} > 0.6)")
        results['representation_consistent'] = 'partial'
    else:
        print(f"[FAIL] Pearson r > 0.8: FAIL ({comparison['pearson_r']:.3f})")
        results['representation_consistent'] = False

    return results


def main():
    parser = argparse.ArgumentParser(description='Representation Generalization Experiment')
    parser.add_argument('--gnn-matrix', type=str,
                       default='outputs/gradients/gnn_conflict_matrices.npz',
                       help='Path to GNN gradient matrix')
    parser.add_argument('--output-dir', type=str, default='outputs/representation',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--quiet', action='store_true')

    args = parser.parse_args()

    results = run_representation_experiment(
        gnn_matrix_path=args.gnn_matrix,
        output_dir=args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
