#!/usr/bin/env python3
"""
Experiment 11: Pairwise Overlap Analysis

Investigates whether pairwise overlap (not dataset-average overlap)
determines gradient reliability. The kinase panel achieves r=0.67
despite ~8% average overlap — this experiment tests if within-family
pairs with higher overlap drive the correlation.

Key outputs:
1. Pairwise overlap vs r(G,E) scatter
2. r(G,E) binned by overlap level
3. Same-family vs cross-family overlap comparison
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import json


KINASE_FAMILIES = {
    'CDK': ['CDK1', 'CDK2', 'CDK6', 'CDK7', 'CDK9'],
    'JAK': ['JAK1', 'JAK2', 'JAK3', 'TYK2'],
    'SRC': ['SRC', 'FYN', 'LCK', 'YES1'],
    'Aurora': ['AURKA', 'AURKB', 'AURKC'],
    'EGFR': ['EGFR', 'HER2', 'HER3', 'HER4'],
    'ABL': ['ABL1'],
}


def get_family(kinase):
    for fam, members in KINASE_FAMILIES.items():
        if kinase in members:
            return fam
    return 'Other'


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='outputs/kinase_data/kinase_all_activity_matrix.csv')
    parser.add_argument('--gradient-path', default='outputs/kinase_all_results/gradient_matrices.npz')
    parser.add_argument('--output-dir', default='outputs/experiment11_pairwise_overlap')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Experiment 11: Pairwise Overlap Analysis")
    print("=" * 60)

    # Load data
    df = pd.read_csv(args.data_path)
    task_cols = [c for c in df.columns if c.endswith('_pIC50')]

    # Load gradient matrix
    gdata = np.load(args.gradient_path, allow_pickle=True)
    G = gdata['average_matrix']
    g_tasks = list(gdata['tasks'])

    # Match task ordering
    task_cols = [t for t in task_cols if t in g_tasks]
    n = len(task_cols)
    print(f"Tasks: {n}")

    # Compute pairwise overlap and empirical correlation
    results = []
    for i in range(n):
        for j in range(i + 1, n):
            t1, t2 = task_cols[i], task_cols[j]
            k1 = t1.replace('_pIC50', '')
            k2 = t2.replace('_pIC50', '')

            # Overlap
            both = df[[t1, t2]].dropna()
            overlap = len(both) / len(df)
            n_shared = len(both)

            # Empirical correlation
            if n_shared >= 10:
                emp_r, emp_p = stats.pearsonr(both[t1], both[t2])
            else:
                emp_r, emp_p = np.nan, np.nan

            # Gradient correlation
            gi = g_tasks.index(t1)
            gj = g_tasks.index(t2)
            grad_g = G[gi, gj]

            # Family info
            fam1 = get_family(k1)
            fam2 = get_family(k2)
            same_family = fam1 == fam2

            results.append({
                'kinase1': k1, 'kinase2': k2,
                'family1': fam1, 'family2': fam2,
                'same_family': same_family,
                'overlap': overlap,
                'n_shared': n_shared,
                'empirical_r': emp_r,
                'gradient_G': grad_g,
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'pairwise_overlap_analysis.csv', index=False)

    # --- Analysis 1: Overlap by family relationship ---
    print("\n--- Overlap by Family Relationship ---")
    same = results_df[results_df['same_family']]
    diff = results_df[~results_df['same_family']]
    print(f"Same family: n={len(same)}, mean overlap={same['overlap'].mean():.3f}, "
          f"range=[{same['overlap'].min():.3f}, {same['overlap'].max():.3f}]")
    print(f"Diff family: n={len(diff)}, mean overlap={diff['overlap'].mean():.3f}, "
          f"range=[{diff['overlap'].min():.3f}, {diff['overlap'].max():.3f}]")

    t_stat, p_val = stats.ttest_ind(same['overlap'], diff['overlap'])
    print(f"t-test: t={t_stat:.3f}, p={p_val:.6f}")

    # --- Analysis 2: r(G,E) binned by overlap ---
    print("\n--- r(G,E) Binned by Pairwise Overlap ---")
    valid = results_df.dropna(subset=['empirical_r', 'gradient_G'])
    bins = [(0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.35), (0.35, 1.0)]
    bin_results = []

    for lo, hi in bins:
        mask = (valid['overlap'] >= lo) & (valid['overlap'] < hi)
        subset = valid[mask]
        if len(subset) >= 5:
            r, p = stats.pearsonr(subset['gradient_G'], subset['empirical_r'])
            bin_results.append({
                'overlap_range': f'{lo:.0%}-{hi:.0%}',
                'n_pairs': len(subset),
                'r_G_E': round(r, 3),
                'p_value': round(p, 4),
                'mean_overlap': round(subset['overlap'].mean(), 3),
            })
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"  Overlap {lo:.0%}-{hi:.0%}: n={len(subset):3d}, r(G,E)={r:.3f}{sig}")
        else:
            bin_results.append({
                'overlap_range': f'{lo:.0%}-{hi:.0%}',
                'n_pairs': len(subset),
                'r_G_E': None,
                'p_value': None,
                'mean_overlap': round(subset['overlap'].mean(), 3) if len(subset) > 0 else None,
            })
            print(f"  Overlap {lo:.0%}-{hi:.0%}: n={len(subset):3d}, too few pairs")

    # --- Analysis 3: Overall r(G,E) ---
    print("\n--- Overall r(G,E) ---")
    r_all, p_all = stats.pearsonr(valid['gradient_G'], valid['empirical_r'])
    print(f"All pairs: r={r_all:.3f}, p={p_all:.2e}, n={len(valid)}")

    high = valid[valid['overlap'] >= 0.10]
    if len(high) >= 5:
        r_high, p_high = stats.pearsonr(high['gradient_G'], high['empirical_r'])
        print(f"Overlap >=10%: r={r_high:.3f}, p={p_high:.2e}, n={len(high)}")

    higher = valid[valid['overlap'] >= 0.20]
    if len(higher) >= 5:
        r_higher, p_higher = stats.pearsonr(higher['gradient_G'], higher['empirical_r'])
        print(f"Overlap >=20%: r={r_higher:.3f}, p={p_higher:.2e}, n={len(higher)}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A: G vs E colored by overlap
    ax = axes[0]
    sc = ax.scatter(valid['empirical_r'], valid['gradient_G'],
                    c=valid['overlap'], cmap='viridis', alpha=0.6, edgecolors='k', linewidths=0.3)
    plt.colorbar(sc, ax=ax, label='Pairwise Overlap')
    ax.set_xlabel('Empirical Correlation (E)')
    ax.set_ylabel('Gradient Similarity (G)')
    ax.set_title(f'A) G vs E (all pairs, r={r_all:.3f})')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.3)

    # Panel B: Same vs different family
    ax = axes[1]
    colors = ['#e41a1c' if sf else '#377eb8' for sf in valid['same_family']]
    ax.scatter(valid['empirical_r'], valid['gradient_G'], c=colors, alpha=0.6,
               edgecolors='k', linewidths=0.3)
    ax.set_xlabel('Empirical Correlation (E)')
    ax.set_ylabel('Gradient Similarity (G)')
    ax.set_title('B) Same family (red) vs Different (blue)')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.3)

    # Panel C: r(G,E) vs overlap bin
    ax = axes[2]
    valid_bins = [b for b in bin_results if b['r_G_E'] is not None]
    if valid_bins:
        x = [b['mean_overlap'] for b in valid_bins]
        y = [b['r_G_E'] for b in valid_bins]
        sizes = [b['n_pairs'] * 5 for b in valid_bins]
        ax.scatter(x, y, s=sizes, c='#4daf4a', edgecolors='k', linewidths=1, zorder=5)
        for b in valid_bins:
            ax.annotate(f"n={b['n_pairs']}", (b['mean_overlap'], b['r_G_E']),
                        textcoords='offset points', xytext=(5, 5), fontsize=8)
    ax.set_xlabel('Mean Pairwise Overlap')
    ax.set_ylabel('r(G, E) within bin')
    ax.set_title('C) Gradient reliability vs pairwise overlap')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'pairwise_overlap_analysis.png', dpi=150)
    plt.close()

    # --- Save summary ---
    summary = {
        'same_family_mean_overlap': round(same['overlap'].mean(), 4),
        'diff_family_mean_overlap': round(diff['overlap'].mean(), 4),
        'overlap_ttest_p': round(p_val, 6),
        'overall_r_G_E': round(r_all, 4),
        'overall_p': float(f'{p_all:.2e}'),
        'n_pairs': len(valid),
        'bin_results': bin_results,
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nSaved to {output_dir}/")


if __name__ == '__main__':
    main()
