#!/usr/bin/env python3
"""
Experiment 16: Synthetic Ground Truth Validation

Creates synthetic multi-task regression with KNOWN task covariance,
then tests gradient method under controlled overlap.

Eliminates circularity concern: ground truth is the designed covariance,
not empirical correlation. Also tests domain generalization beyond molecules.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def generate_synthetic_data(n_samples=5000, n_features=100, n_tasks=8,
                            task_correlation_matrix=None, seed=42):
    """
    Generate synthetic multi-task data with known task relationships.

    Tasks share a common latent structure plus task-specific noise.
    The designed covariance matrix is the ground truth.
    """
    rng = np.random.RandomState(seed)

    if task_correlation_matrix is None:
        # Design a correlation matrix with clear structure:
        # - Tasks 0,1,2 are positively correlated (cluster A)
        # - Tasks 3,4,5 are positively correlated (cluster B)
        # - A vs B are negatively correlated
        # - Tasks 6,7 are independent
        C = np.eye(n_tasks)

        # Cluster A: tasks 0,1,2
        C[0, 1] = C[1, 0] = 0.7
        C[0, 2] = C[2, 0] = 0.5
        C[1, 2] = C[2, 1] = 0.6

        # Cluster B: tasks 3,4,5
        C[3, 4] = C[4, 3] = 0.8
        C[3, 5] = C[5, 3] = 0.4
        C[4, 5] = C[5, 4] = 0.5

        # A vs B: negative
        for i in range(3):
            for j in range(3, 6):
                C[i, j] = C[j, i] = -0.3

        # Tasks 6,7: independent
        task_correlation_matrix = C

    # Generate shared latent features
    X = rng.randn(n_samples, n_features)

    # Generate task-specific weight vectors with designed correlation
    # Use Cholesky decomposition to create correlated task weights
    L = np.linalg.cholesky(task_correlation_matrix)

    # Base weights for each latent dimension
    n_shared = 20  # shared latent dimensions
    W_base = rng.randn(n_shared, n_tasks)
    W_correlated = W_base @ L.T

    # Task labels = X[:, :n_shared] @ W_correlated + noise
    Y_clean = X[:, :n_shared] @ W_correlated
    noise = rng.randn(n_samples, n_tasks) * 0.5
    Y = Y_clean + noise

    # Verify empirical correlation matches design
    emp_corr = np.corrcoef(Y.T)

    return X, Y, task_correlation_matrix, emp_corr


def mask_overlap(Y, overlap_fraction, seed=42):
    """Mask labels to simulate reduced overlap."""
    rng = np.random.RandomState(seed)
    n_samples, n_tasks = Y.shape
    mask = np.ones_like(Y, dtype=bool)

    for t in range(n_tasks):
        n_hide = int(n_samples * (1 - overlap_fraction))
        hide_idx = rng.choice(n_samples, size=n_hide, replace=False)
        mask[hide_idx, t] = False

    Y_masked = Y.copy()
    Y_masked[~mask] = np.nan

    return Y_masked, mask


class MultiTaskMLP(nn.Module):
    """Simple MLP for multi-task learning."""
    def __init__(self, input_dim, hidden_dim, n_tasks):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(n_tasks)])
        self.n_tasks = n_tasks

    def forward(self, x):
        h = self.encoder(x)
        return [head(h).squeeze(-1) for head in self.heads]

    def get_encoder_params(self):
        return list(self.encoder.parameters())


def train_and_compute_gradients(X, Y, mask, n_tasks, device, epochs=50):
    """Train multi-task model and compute gradient correlation matrix."""
    n_samples, input_dim = X.shape

    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(np.nan_to_num(Y, nan=0.0), dtype=torch.float32)
    mask_tensor = torch.tensor(mask, dtype=torch.bool)

    dataset = TensorDataset(X_tensor, Y_tensor, mask_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

    model = MultiTaskMLP(input_dim, 128, n_tasks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    gradient_history = []

    for epoch in range(epochs):
        for xb, yb, mb in loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            optimizer.zero_grad()

            preds = model(xb)
            task_losses = {}

            for t in range(n_tasks):
                m = mb[:, t]
                if m.sum() == 0:
                    continue
                loss = F.mse_loss(preds[t][m], yb[:, t][m])
                task_losses[t] = loss

            # Compute gradient correlations every 5 epochs
            if epoch % 5 == 0 and len(task_losses) > 1:
                encoder_params = model.get_encoder_params()
                task_grads = {}
                for t, loss in task_losses.items():
                    grads = torch.autograd.grad(loss, encoder_params,
                                                retain_graph=True, allow_unused=True)
                    grad_vec = torch.cat([g.flatten() if g is not None
                                          else torch.zeros_like(p.flatten())
                                          for g, p in zip(grads, encoder_params)])
                    task_grads[t] = grad_vec

                G = np.eye(n_tasks)
                for ti in task_grads:
                    for tj in task_grads:
                        if ti >= tj:
                            continue
                        g_i = task_grads[ti]
                        g_j = task_grads[tj]
                        cos = torch.dot(g_i, g_j) / (torch.norm(g_i) * torch.norm(g_j) + 1e-8)
                        G[ti, tj] = cos.item()
                        G[tj, ti] = cos.item()

                gradient_history.append(G)

            # Backward
            total_loss = sum(task_losses.values()) / len(task_losses)
            total_loss.backward()
            optimizer.step()

    # Average gradient matrix
    if gradient_history:
        G_avg = np.nanmean(gradient_history, axis=0)
    else:
        G_avg = np.eye(n_tasks)

    return G_avg


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='outputs/experiment16_synthetic')
    parser.add_argument('--n-trials', type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Experiment 16: Synthetic Ground Truth Validation")
    print(f"Device: {device}")
    print("=" * 60)

    # Generate synthetic data
    print("\nGenerating synthetic multi-task data...")
    X, Y, C_true, C_emp = generate_synthetic_data(n_samples=5000, n_tasks=8)
    n_tasks = Y.shape[1]

    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features, {n_tasks} tasks")
    print(f"\nDesigned task correlation matrix:")
    np.set_printoptions(precision=3, suppress=True)
    print(C_true)

    r_design_emp, _ = stats.pearsonr(
        C_true[np.triu_indices(n_tasks, 1)],
        C_emp[np.triu_indices(n_tasks, 1)]
    )
    print(f"\nr(designed, empirical) = {r_design_emp:.3f}")

    # Test at different overlap levels
    overlap_levels = [1.0, 0.75, 0.50, 0.30, 0.20, 0.10]
    c_true_upper = C_true[np.triu_indices(n_tasks, 1)]

    results = []

    for overlap in overlap_levels:
        print(f"\n--- Overlap: {overlap:.0%} ---")

        trial_r_values = []

        for trial in range(args.n_trials):
            Y_masked, mask = mask_overlap(Y, overlap, seed=42 + trial)

            # Train and get gradient matrix
            G = train_and_compute_gradients(X, Y_masked, mask, n_tasks, device, epochs=50)
            g_upper = G[np.triu_indices(n_tasks, 1)]

            # Compare to TRUE correlation (no circularity!)
            valid = ~np.isnan(g_upper)
            if valid.sum() >= 5:
                r, _ = stats.pearsonr(g_upper[valid], c_true_upper[valid])
                trial_r_values.append(r)

        result = {
            'overlap': overlap,
            'r_G_Ctrue_mean': round(np.mean(trial_r_values), 4) if trial_r_values else None,
            'r_G_Ctrue_std': round(np.std(trial_r_values), 4) if trial_r_values else None,
            'n_trials': len(trial_r_values),
        }
        results.append(result)

        print(f"  r(G, C_true) = {result['r_G_Ctrue_mean']:.3f} ± {result['r_G_Ctrue_std']:.3f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Designed correlation matrix
    ax = axes[0]
    import matplotlib
    cmap = matplotlib.cm.RdYlGn
    im = ax.imshow(C_true, cmap=cmap, vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label='Correlation')
    ax.set_title('A) Designed Task Correlation (Ground Truth)')
    labels = [f'T{i}' for i in range(n_tasks)]
    ax.set_xticks(range(n_tasks))
    ax.set_yticks(range(n_tasks))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Annotate with values
    for i in range(n_tasks):
        for j in range(n_tasks):
            ax.text(j, i, f'{C_true[i,j]:.1f}', ha='center', va='center', fontsize=7)

    # Panel B: r(G, C_true) vs overlap
    ax = axes[1]
    overlaps = [r['overlap'] for r in results]
    means = [r['r_G_Ctrue_mean'] for r in results]
    stds = [r['r_G_Ctrue_std'] for r in results]
    ax.errorbar(overlaps, means, yerr=stds, fmt='b-o', linewidth=2, markersize=8, capsize=4)
    ax.set_xlabel('Overlap Level', fontsize=12)
    ax.set_ylabel('r(G, C_true)', fontsize=12)
    ax.set_title('B) Gradient Accuracy vs Overlap (No Circularity)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig(output_dir / 'synthetic_validation.png', dpi=150)
    plt.close()

    # Save
    summary = {
        'n_tasks': n_tasks,
        'n_samples': X.shape[0],
        'designed_correlation': C_true.tolist(),
        'results': results,
    }
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {output_dir}/")

    print("\n" + "=" * 60)
    print("SUMMARY: Gradient accuracy vs overlap (ground truth)")
    for r in results:
        print(f"  Overlap {r['overlap']:.0%}: r(G, C_true) = {r['r_G_Ctrue_mean']:.3f}")


if __name__ == '__main__':
    main()
