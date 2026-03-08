#!/usr/bin/env python3
"""
Cross-Domain Validation Experiment

Tests whether the ~30% overlap threshold generalizes beyond molecular property
prediction to other domains (vision, quantum chemistry).

Key insight: If the threshold is similar across domains, the finding becomes
much stronger - it's not specific to molecules but a general property of
gradient-based task analysis.

Domains to test:
1. QM9 - quantum chemistry (12 properties, 100% overlap by design)
2. NYUv2 - computer vision (depth, segmentation, normals)
3. CelebA - face attributes (40 binary attributes)
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
from typing import List, Tuple, Dict, Optional
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import warnings

warnings.filterwarnings('ignore')

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def sigmoid(x, L, k, x0, b):
    """Sigmoid function for fitting phase transition."""
    return L / (1 + np.exp(-k * (x - x0))) + b


def fit_sigmoid(overlaps: np.ndarray, correlations: np.ndarray) -> Dict:
    """Fit sigmoid to overlap vs correlation data."""
    try:
        # Initial guesses
        p0 = [0.8, 0.1, 30, 0.1]
        bounds = ([0, 0.01, 10, 0], [1, 0.5, 90, 0.5])

        popt, pcov = curve_fit(sigmoid, overlaps, correlations, p0=p0,
                               bounds=bounds, maxfev=10000)

        # Compute R^2
        y_pred = sigmoid(overlaps, *popt)
        ss_res = np.sum((correlations - y_pred) ** 2)
        ss_tot = np.sum((correlations - np.mean(correlations)) ** 2)
        r_squared = 1 - ss_res / ss_tot

        return {
            'L': popt[0],
            'k': popt[1],
            'x0': popt[2],  # Inflection point (threshold)
            'b': popt[3],
            'r_squared': r_squared
        }
    except Exception as e:
        print(f"Sigmoid fitting failed: {e}")
        return None


class QM9MultiTask:
    """
    QM9 quantum chemistry dataset with multiple targets.

    Properties (all computed for all 134k molecules):
    - mu: Dipole moment
    - alpha: Isotropic polarizability
    - homo: HOMO energy
    - lumo: LUMO energy
    - gap: HOMO-LUMO gap (= lumo - homo, perfect relationship)
    - r2: Electronic spatial extent
    - zpve: Zero point vibrational energy
    - u0: Internal energy at 0K
    - u298: Internal energy at 298K
    - h298: Enthalpy at 298K
    - g298: Free energy at 298K
    - cv: Heat capacity at 298K
    """

    TARGETS = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2',
               'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']

    def __init__(self, root: str = './data/qm9'):
        self.root = root
        self.data = None

    def load(self) -> bool:
        """Load QM9 data using PyTorch Geometric if available."""
        try:
            from torch_geometric.datasets import QM9
            print("Loading QM9 dataset...")
            dataset = QM9(root=self.root)
            print(f"  Loaded {len(dataset)} molecules")

            # Extract targets
            self.data = {
                'features': [],
                'targets': {t: [] for t in self.TARGETS}
            }

            for i, data in enumerate(dataset):
                if i >= 10000:  # Use subset for speed
                    break
                # Use node features sum as simple representation
                self.data['features'].append(data.x.sum(dim=0).numpy())
                for j, target in enumerate(self.TARGETS):
                    self.data['targets'][target].append(data.y[0, j].item())

            # Convert to arrays
            self.data['features'] = np.array(self.data['features'])
            for target in self.TARGETS:
                self.data['targets'][target] = np.array(self.data['targets'][target])

            print(f"  Using {len(self.data['features'])} molecules")
            return True

        except ImportError:
            print("PyTorch Geometric not installed.")
            print("Install with: pip install torch-geometric")
            return False

    def compute_empirical_correlations(self) -> np.ndarray:
        """Compute pairwise correlations between all targets."""
        n_targets = len(self.TARGETS)
        E = np.zeros((n_targets, n_targets))

        for i, t1 in enumerate(self.TARGETS):
            for j, t2 in enumerate(self.TARGETS):
                if i == j:
                    E[i, j] = 1.0
                else:
                    r, _ = pearsonr(self.data['targets'][t1],
                                   self.data['targets'][t2])
                    E[i, j] = r

        return E

    def degrade_overlap(self, alpha: float) -> Dict[str, np.ndarray]:
        """
        Artificially partition molecules to achieve target overlap.

        With alpha overlap, each task sees a random subset where
        alpha fraction is shared across all tasks.
        """
        n = len(self.data['features'])
        n_shared = int(alpha * n)
        n_disjoint = n - n_shared

        # Shared indices (same for all tasks)
        shared_idx = np.random.choice(n, size=n_shared, replace=False)
        remaining_idx = np.setdiff1d(np.arange(n), shared_idx)

        # Create task-specific indices
        task_indices = {}
        for i, target in enumerate(self.TARGETS):
            # Each task gets shared + unique disjoint portion
            n_task_disjoint = n_disjoint // len(self.TARGETS)
            start = i * n_task_disjoint
            end = start + n_task_disjoint
            if end > len(remaining_idx):
                end = len(remaining_idx)

            task_disjoint = remaining_idx[start:end] if start < len(remaining_idx) else np.array([])
            task_indices[target] = np.concatenate([shared_idx, task_disjoint])

        return task_indices


class SimpleEncoder(nn.Module):
    """Simple MLP encoder for QM9."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)


class MultiTaskHead(nn.Module):
    """Task-specific prediction heads."""

    def __init__(self, input_dim: int, n_tasks: int):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Linear(input_dim, 1) for _ in range(n_tasks)
        ])

    def forward(self, x, task_idx: int):
        return self.heads[task_idx](x)


def compute_gradient_similarity_qm9(qm9: QM9MultiTask,
                                     task_indices: Dict[str, np.ndarray],
                                     n_epochs: int = 10) -> np.ndarray:
    """
    Train MTL model and extract gradient similarity matrix.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare data
    X = torch.FloatTensor(qm9.data['features']).to(device)
    Y = {t: torch.FloatTensor(qm9.data['targets'][t]).to(device)
         for t in qm9.TARGETS}

    # Model
    input_dim = X.shape[1]
    encoder = SimpleEncoder(input_dim).to(device)
    heads = MultiTaskHead(128, len(qm9.TARGETS)).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(heads.parameters()),
        lr=1e-3
    )

    # Training and gradient collection
    gradient_matrices = []

    for epoch in range(n_epochs):
        encoder.train()
        heads.train()

        # Forward pass
        z = encoder(X)

        # Collect gradients for each task
        gradients = []
        for i, target in enumerate(qm9.TARGETS):
            idx = task_indices[target]
            if len(idx) == 0:
                gradients.append(None)
                continue

            pred = heads(z[idx], i).squeeze()
            loss = nn.MSELoss()(pred, Y[target][idx])

            # Get gradients w.r.t. encoder
            encoder.zero_grad()
            loss.backward(retain_graph=True)

            # Flatten encoder gradients
            grad = torch.cat([p.grad.flatten() for p in encoder.parameters()
                            if p.grad is not None])
            gradients.append(grad.detach().cpu().numpy())

        # Compute similarity matrix
        n_tasks = len(qm9.TARGETS)
        G = np.zeros((n_tasks, n_tasks))
        for i in range(n_tasks):
            for j in range(n_tasks):
                if gradients[i] is None or gradients[j] is None:
                    G[i, j] = 0
                else:
                    # Cosine similarity
                    dot = np.dot(gradients[i], gradients[j])
                    norm_i = np.linalg.norm(gradients[i])
                    norm_j = np.linalg.norm(gradients[j])
                    if norm_i > 0 and norm_j > 0:
                        G[i, j] = dot / (norm_i * norm_j)

        gradient_matrices.append(G)

        # Backward for optimization
        total_loss = sum(
            nn.MSELoss()(heads(z[task_indices[t]], i).squeeze(),
                        Y[t][task_indices[t]])
            for i, t in enumerate(qm9.TARGETS) if len(task_indices[t]) > 0
        )
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # Average over last half of training
    G_avg = np.mean(gradient_matrices[n_epochs//2:], axis=0)
    return G_avg


def run_overlap_degradation_experiment(qm9: QM9MultiTask,
                                        overlap_levels: List[float] = None
                                        ) -> pd.DataFrame:
    """
    Systematically degrade overlap and measure gradient-empirical correlation.
    """
    if overlap_levels is None:
        overlap_levels = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    # Ground truth empirical correlations (at 100% overlap)
    E = qm9.compute_empirical_correlations()

    results = []

    for alpha in overlap_levels:
        print(f"\nOverlap = {alpha*100:.0f}%")

        # Degrade overlap
        task_indices = qm9.degrade_overlap(alpha)

        # Compute gradient similarity
        G = compute_gradient_similarity_qm9(qm9, task_indices)

        # Correlation between G and E (upper triangle only)
        mask = np.triu(np.ones_like(G, dtype=bool), k=1)
        g_flat = G[mask]
        e_flat = E[mask]

        r, p = pearsonr(g_flat, e_flat)

        results.append({
            'overlap': alpha * 100,
            'correlation': r,
            'p_value': p,
            'significant': p < 0.01
        })

        print(f"  r(G, E) = {r:.3f}, p = {p:.2e}")

    return pd.DataFrame(results)


def main():
    """Run cross-domain validation experiment."""
    print("=" * 60)
    print("Cross-Domain Validation Experiment")
    print("=" * 60)

    # Try QM9 first
    print("\n--- QM9 (Quantum Chemistry) ---")
    qm9 = QM9MultiTask()

    if qm9.load():
        # Show empirical correlations
        E = qm9.compute_empirical_correlations()
        print("\nEmpirical correlation matrix (selected pairs):")
        print(f"  HOMO-LUMO: {E[2, 3]:.3f}")
        print(f"  HOMO-Gap: {E[2, 4]:.3f}")
        print(f"  LUMO-Gap: {E[3, 4]:.3f}")  # Should be ~1.0
        print(f"  u0-u298: {E[7, 8]:.3f}")  # Should be very high
        print(f"  u298-h298: {E[8, 9]:.3f}")  # Should be very high

        # Run degradation experiment
        print("\n" + "=" * 60)
        print("Overlap Degradation Experiment")
        print("=" * 60)

        results = run_overlap_degradation_experiment(qm9)

        # Fit sigmoid
        print("\n" + "=" * 60)
        print("Sigmoid Fit")
        print("=" * 60)

        fit = fit_sigmoid(results['overlap'].values, results['correlation'].values)
        if fit:
            print(f"\nFit parameters:")
            print(f"  L (max correlation): {fit['L']:.3f}")
            print(f"  k (steepness): {fit['k']:.4f}")
            print(f"  x0 (threshold): {fit['x0']:.1f}%")
            print(f"  b (baseline): {fit['b']:.3f}")
            print(f"  R^2: {fit['r_squared']:.3f}")

            print(f"\n*** Predicted threshold: {fit['x0']:.1f}% ***")
            print(f"*** Compare to molecular: 29.7% ***")

        # Save results
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
        os.makedirs(output_dir, exist_ok=True)

        results.to_csv(os.path.join(output_dir, 'qm9_overlap_degradation.csv'),
                      index=False)
        print(f"\nResults saved to outputs/qm9_overlap_degradation.csv")

        # Print table for paper
        print("\n" + "=" * 60)
        print("Table for Paper")
        print("=" * 60)
        print("\n| Overlap | r(G,E) | p-value | Significant |")
        print("|---------|--------|---------|-------------|")
        for _, row in results.iterrows():
            sig = "Yes" if row['significant'] else "No"
            print(f"| {row['overlap']:.0f}% | {row['correlation']:.3f} | "
                  f"{row['p_value']:.2e} | {sig} |")

    else:
        print("QM9 loading failed. Skipping.")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
To complete cross-domain validation:

1. QM9 (Quantum Chemistry):
   - 12 quantum properties
   - 100% overlap by design
   - Known relationships (HOMO-LUMO gap)
   - Run this script with PyTorch Geometric installed

2. NYUv2 (Computer Vision):
   - 3 tasks: depth, segmentation, surface normals
   - 1,449 images with all annotations
   - Use standard MTL vision code
   - Artificially partition images

3. CelebA (Face Attributes):
   - 40 binary attributes
   - ~200K images with all attributes
   - Similar protocol to NYUv2

Expected outcome:
- If all domains show threshold ~25-35%, strong evidence of universality
- If thresholds differ, characterize domain-specific values
""")


if __name__ == '__main__':
    main()
