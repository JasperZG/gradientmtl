#!/usr/bin/env python3
"""
Experiment 12: Empirical Correlation Stability Under Overlap Reduction

Tests whether degradation in r(G,E) as overlap decreases is due to:
  (a) loss of gradient signal, or
  (b) instability in the empirical correlation E itself

Method: Start from Tox21 (100% overlap), subsample to various levels,
measure stability of E on the subsample vs E on the full dataset.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import json


def load_tox21():
    """Load Tox21 dataset."""
    path = Path('outputs/raw_data/tox21.csv')
    if not path.exists():
        # Try alternate location
        from tdc.benchmark_group import admet_group
        path = Path('data/tox21.csv')

    df = pd.read_csv(path)
    tasks = [c for c in df.columns if c.startswith('NR-') or c.startswith('SR-')]
    return df, tasks


def compute_empirical_correlation(df, tasks):
    """Compute full empirical correlation matrix."""
    n = len(tasks)
    E = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                E[i, j] = 1.0
                continue
            both = df[[tasks[i], tasks[j]]].dropna()
            if len(both) >= 10:
                E[i, j], _ = stats.pearsonr(both[tasks[i]], both[tasks[j]])
            else:
                E[i, j] = np.nan
    return E


def subsample_overlap(df, tasks, target_overlap, seed=42):
    """
    Reduce effective overlap by masking labels.
    For each compound, randomly mask task labels to achieve target overlap.
    """
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
    parser.add_argument('--output-dir', default='outputs/experiment12_e_stability')
    parser.add_argument('--n-trials', type=int, default=20)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Experiment 12: Empirical Correlation Stability")
    print("=" * 60)

    # Load Tox21
    print("\nLoading Tox21...")
    df, tasks = load_tox21()
    n_tasks = len(tasks)
    print(f"Tasks: {n_tasks}, Compounds: {len(df)}")

    # Full empirical correlation (ground truth)
    E_full = compute_empirical_correlation(df, tasks)
    e_full_upper = E_full[np.triu_indices(n_tasks, 1)]

    # Load gradient matrix
    grad_path = Path('outputs/gradients/gnn_conflict_matrices.npz')
    if not grad_path.exists():
        # Try alternate
        for p in ['outputs/gradients/gradient_matrices.npz',
                   'outputs/tox21_results/gradient_matrices.npz']:
            if Path(p).exists():
                grad_path = Path(p)
                break

    gdata = np.load(str(grad_path), allow_pickle=True)
    G_key = 'average_matrix' if 'average_matrix' in gdata else 'averaged'
    G = gdata[G_key]

    # Test overlap levels
    overlap_levels = [1.0, 0.75, 0.50, 0.30, 0.20, 0.10]

    results = []

    for overlap in overlap_levels:
        print(f"\n--- Overlap: {overlap:.0%} ---")

        e_stabilities = []
        g_e_correlations = []

        for trial in range(args.n_trials):
            # Subsample
            df_sub = subsample_overlap(df, tasks, overlap, seed=42 + trial)

            # Compute E on subsample
            E_sub = compute_empirical_correlation(df_sub, tasks)
            e_sub_upper = E_sub[np.triu_indices(n_tasks, 1)]

            # E stability: r(E_sub, E_full)
            valid_mask = ~np.isnan(e_sub_upper) & ~np.isnan(e_full_upper)
            if valid_mask.sum() >= 5:
                r_e_stability, _ = stats.pearsonr(e_sub_upper[valid_mask], e_full_upper[valid_mask])
                e_stabilities.append(r_e_stability)

            # G vs E_sub: r(G, E_sub)
            g_upper = G[np.triu_indices(n_tasks, 1)]
            valid_mask2 = ~np.isnan(e_sub_upper) & ~np.isnan(g_upper)
            if valid_mask2.sum() >= 5:
                r_g_e, _ = stats.pearsonr(g_upper[valid_mask2], e_sub_upper[valid_mask2])
                g_e_correlations.append(r_g_e)

        result = {
            'overlap': overlap,
            'e_stability_mean': round(np.mean(e_stabilities), 4) if e_stabilities else None,
            'e_stability_std': round(np.std(e_stabilities), 4) if e_stabilities else None,
            'r_G_E_mean': round(np.mean(g_e_correlations), 4) if g_e_correlations else None,
            'r_G_E_std': round(np.std(g_e_correlations), 4) if g_e_correlations else None,
            'n_trials': len(e_stabilities),
        }
        results.append(result)

        print(f"  E stability: r(E_sub, E_full) = {result['e_stability_mean']:.3f} ± {result['e_stability_std']:.3f}")
        print(f"  r(G, E_sub) = {result['r_G_E_mean']:.3f} ± {result['r_G_E_std']:.3f}")

    # --- Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    overlaps = [r['overlap'] for r in results]
    e_means = [r['e_stability_mean'] for r in results]
    e_stds = [r['e_stability_std'] for r in results]
    g_means = [r['r_G_E_mean'] for r in results]
    g_stds = [r['r_G_E_std'] for r in results]

    ax.errorbar(overlaps, e_means, yerr=e_stds, fmt='b-o', linewidth=2, markersize=8,
                capsize=4, label='E stability: r(E_subset, E_full)')
    ax.errorbar(overlaps, g_means, yerr=g_stds, fmt='r-s', linewidth=2, markersize=8,
                capsize=4, label='G accuracy: r(G, E_subset)')

    ax.set_xlabel('Overlap Level', fontsize=12)
    ax.set_ylabel('Correlation', fontsize=12)
    ax.set_title('Empirical Correlation Stability vs Gradient Accuracy', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.1, 1.1)
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig(output_dir / 'e_stability_analysis.png', dpi=150)
    plt.close()

    # Save
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump({'results': results}, f, indent=2)

    pd.DataFrame(results).to_csv(output_dir / 'e_stability_results.csv', index=False)

    print(f"\nSaved to {output_dir}/")

    # Conclusion
    print("\n" + "=" * 60)
    print("KEY COMPARISON:")
    for r in results:
        gap = (r['e_stability_mean'] or 0) - (r['r_G_E_mean'] or 0)
        print(f"  Overlap {r['overlap']:.0%}: E stability={r['e_stability_mean']:.3f}, "
              f"r(G,E)={r['r_G_E_mean']:.3f}, gap={gap:.3f}")


if __name__ == '__main__':
    main()
