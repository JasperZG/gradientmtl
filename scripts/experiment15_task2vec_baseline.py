#!/usr/bin/env python3
"""
Experiment 15: Task2Vec Baseline Comparison

Implements a simplified Task2Vec approach (Achille et al., 2019)
and compares degradation under reduced overlap against gradient method.

Task2Vec: Embed tasks using Fisher Information of a probe network,
compute cosine similarity between task embeddings.

Compares: gradient similarity vs Task2Vec vs transfer-based affinity
under controlled overlap on Tox21.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    print("RDKit required")
    exit(1)


def smiles_to_ecfp(smiles, radius=2, nbits=1024):
    """Convert SMILES to ECFP fingerprint."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    return np.array(fp, dtype=np.float32)


def load_tox21():
    """Load Tox21 with ECFP features."""
    path = Path('outputs/raw_data/tox21.csv')
    df = pd.read_csv(path)
    tasks = [c for c in df.columns if c.startswith('NR-') or c.startswith('SR-')]

    # Compute fingerprints
    fps = []
    valid_idx = []
    smiles_list = df['smiles'] if 'smiles' in df.columns else df.iloc[:, 0]
    for i, smi in enumerate(smiles_list):
        fp = smiles_to_ecfp(smi)
        if fp is not None:
            fps.append(fp)
            valid_idx.append(i)

    X = np.array(fps)
    df = df.iloc[valid_idx].reset_index(drop=True)

    return X, df, tasks


class ProbeNetwork(nn.Module):
    """Simple probe network for Task2Vec."""
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)


def compute_task2vec_embedding(X, y, mask, input_dim, device, n_epochs=20):
    """
    Compute Task2Vec embedding: diagonal of Fisher Information Matrix
    after training a probe network on the task.
    """
    valid = mask.astype(bool)
    X_valid = torch.tensor(X[valid], dtype=torch.float32)
    y_valid = torch.tensor(y[valid], dtype=torch.float32)

    if len(X_valid) < 20:
        return None

    dataset = TensorDataset(X_valid, y_valid)
    loader = DataLoader(dataset, batch_size=min(64, len(dataset)), shuffle=True)

    model = ProbeNetwork(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train probe
    model.train()
    for epoch in range(n_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = F.binary_cross_entropy_with_logits(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Compute Fisher Information (diagonal approximation)
    model.eval()
    fisher = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = F.binary_cross_entropy_with_logits(pred, yb)
        model.zero_grad()
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.data ** 2

    # Normalize
    n_batches = len(loader)
    embedding = torch.cat([f.flatten() / n_batches for f in fisher.values()])
    return embedding.cpu().numpy()


def compute_task2vec_similarity(embeddings, tasks):
    """Compute cosine similarity between task embeddings."""
    n = len(tasks)
    S = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):
            if embeddings[i] is not None and embeddings[j] is not None:
                e_i = embeddings[i]
                e_j = embeddings[j]
                norm_i = np.linalg.norm(e_i)
                norm_j = np.linalg.norm(e_j)
                if norm_i > 1e-8 and norm_j > 1e-8:
                    cos = np.dot(e_i, e_j) / (norm_i * norm_j)
                    S[i, j] = cos
                    S[j, i] = cos

    return S


def subsample_overlap(df, tasks, target_overlap, seed=42):
    """Reduce effective overlap by masking labels."""
    rng = np.random.RandomState(seed)
    df_sub = df.copy()

    for task in tasks:
        valid_idx = df_sub[task].notna()
        n_valid = valid_idx.sum()
        n_keep = int(n_valid * target_overlap)
        if n_keep < n_valid:
            valid_positions = df_sub.index[valid_idx].tolist()
            drop_positions = rng.choice(valid_positions, size=n_valid - n_keep, replace=False)
            df_sub.loc[drop_positions, task] = np.nan

    return df_sub


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='outputs/experiment15_task2vec')
    parser.add_argument('--n-trials', type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Experiment 15: Task2Vec Baseline Comparison")
    print(f"Device: {device}")
    print("=" * 60)

    # Load data
    print("\nLoading Tox21...")
    X, df, tasks = load_tox21()
    n_tasks = len(tasks)
    input_dim = X.shape[1]
    print(f"Tasks: {n_tasks}, Compounds: {len(X)}, Features: {input_dim}")

    # Compute empirical correlation at full overlap
    E_full = np.eye(n_tasks)
    for i in range(n_tasks):
        for j in range(i + 1, n_tasks):
            both = df[[tasks[i], tasks[j]]].dropna()
            if len(both) >= 10:
                r, _ = stats.pearsonr(both[tasks[i]], both[tasks[j]])
                E_full[i, j] = r
                E_full[j, i] = r

    e_upper = E_full[np.triu_indices(n_tasks, 1)]

    # Load gradient matrix
    grad_path = None
    for p in ['outputs/gradients/gnn_conflict_matrices.npz',
              'outputs/gradients/gradient_matrices.npz']:
        if Path(p).exists():
            grad_path = p
            break

    G = None
    if grad_path:
        gdata = np.load(grad_path, allow_pickle=True)
        G_key = 'average_matrix' if 'average_matrix' in gdata else 'averaged'
        G = gdata[G_key]

    # Test at different overlap levels
    overlap_levels = [1.0, 0.75, 0.50, 0.30, 0.10]
    results = []

    for overlap in overlap_levels:
        print(f"\n--- Overlap: {overlap:.0%} ---")

        t2v_correlations = []

        for trial in range(args.n_trials):
            # Subsample
            df_sub = subsample_overlap(df, tasks, overlap, seed=42 + trial)

            # Compute Task2Vec embeddings
            embeddings = []
            for task in tasks:
                y = df_sub[task].values.copy()
                mask = (~np.isnan(y)).astype(np.float32)
                y = np.nan_to_num(y, nan=0.0)
                emb = compute_task2vec_embedding(X, y, mask, input_dim, device)
                embeddings.append(emb)

            # Compute Task2Vec similarity
            T2V = compute_task2vec_similarity(embeddings, tasks)
            t2v_upper = T2V[np.triu_indices(n_tasks, 1)]

            # Correlation with E
            valid_mask = ~np.isnan(t2v_upper) & ~np.isnan(e_upper)
            if valid_mask.sum() >= 5:
                r_t2v, _ = stats.pearsonr(t2v_upper[valid_mask], e_upper[valid_mask])
                t2v_correlations.append(r_t2v)

        result = {
            'overlap': overlap,
            'task2vec_r_mean': round(np.mean(t2v_correlations), 4) if t2v_correlations else None,
            'task2vec_r_std': round(np.std(t2v_correlations), 4) if t2v_correlations else None,
            'n_trials': len(t2v_correlations),
        }

        # Add gradient correlation at this overlap (from experiment 12 if available)
        if G is not None:
            g_upper = G[np.triu_indices(n_tasks, 1)]
            valid_mask = ~np.isnan(g_upper) & ~np.isnan(e_upper)
            if valid_mask.sum() >= 5:
                r_g, _ = stats.pearsonr(g_upper[valid_mask], e_upper[valid_mask])
                result['gradient_r'] = round(r_g, 4)

        results.append(result)

        print(f"  Task2Vec r(T2V, E): {result['task2vec_r_mean']:.3f} ± {result['task2vec_r_std']:.3f}")
        if 'gradient_r' in result:
            print(f"  Gradient r(G, E): {result['gradient_r']:.3f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    overlaps = [r['overlap'] for r in results]
    t2v_means = [r['task2vec_r_mean'] for r in results if r['task2vec_r_mean'] is not None]
    t2v_stds = [r['task2vec_r_std'] for r in results if r['task2vec_r_std'] is not None]
    t2v_overlaps = [r['overlap'] for r in results if r['task2vec_r_mean'] is not None]

    ax.errorbar(t2v_overlaps, t2v_means, yerr=t2v_stds, fmt='g-^', linewidth=2,
                markersize=8, capsize=4, label='Task2Vec')

    if any('gradient_r' in r for r in results):
        g_vals = [r.get('gradient_r', None) for r in results]
        g_overlaps = [r['overlap'] for r in results if r.get('gradient_r') is not None]
        g_vals = [v for v in g_vals if v is not None]
        ax.plot(g_overlaps, g_vals, 'r-s', linewidth=2, markersize=8, label='Gradient (at full overlap)')

    ax.set_xlabel('Overlap Level', fontsize=12)
    ax.set_ylabel('Correlation with Empirical r(E)', fontsize=12)
    ax.set_title('Task2Vec vs Gradient Method Under Overlap Reduction', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig(output_dir / 'task2vec_comparison.png', dpi=150)
    plt.close()

    # Save
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump({'results': results}, f, indent=2)

    print(f"\nSaved to {output_dir}/")


if __name__ == '__main__':
    main()
