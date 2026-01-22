#!/usr/bin/env python3
"""
Validate Tox21+ADME: Compute G vs Empirical Correlation.

This is the KEY validation for cross-domain gradient analysis:
- Compute empirical pairwise correlations between all properties
- Compare to gradient conflict matrix G
- Compute Pearson r(G, empirical)

Expected: r = 0.65-0.80 (similar to Tox21's r=0.918, ToxCast's r=0.862)

Usage:
    python scripts/validate_tox21_adme_correlation.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

project_root = Path(__file__).parent.parent


def compute_empirical_correlations(df: pd.DataFrame, tasks: list) -> np.ndarray:
    """
    Compute pairwise empirical correlations between properties.

    For each pair of tasks, compute Pearson correlation on compounds
    where both properties are measured.
    """
    n_tasks = len(tasks)
    E = np.eye(n_tasks)

    for i, task_i in enumerate(tasks):
        for j, task_j in enumerate(tasks):
            if i >= j:
                continue

            # Get compounds with both values
            mask = df[task_i].notna() & df[task_j].notna()
            n_overlap = mask.sum()

            if n_overlap < 10:
                E[i, j] = E[j, i] = np.nan
                continue

            x = df.loc[mask, task_i].values
            y = df.loc[mask, task_j].values

            # Pearson correlation
            r, p = stats.pearsonr(x, y)
            E[i, j] = E[j, i] = r

    return E


def main():
    print("=" * 60)
    print("TOX21+ADME: G vs EMPIRICAL CORRELATION VALIDATION")
    print("=" * 60)

    # Load data
    data_path = project_root / 'outputs' / 'tox21_adme_augmented' / 'tox21_adme_augmented.csv'
    results_path = project_root / 'outputs' / 'tox21_adme_results' / 'results.json'

    if not data_path.exists():
        print(f"Error: Data not found at {data_path}")
        return

    if not results_path.exists():
        print(f"Error: Results not found at {results_path}")
        return

    df = pd.read_csv(data_path)

    with open(results_path) as f:
        results = json.load(f)

    # Get tasks and gradient matrix
    tasks = results['tasks']
    G = np.array(results['gradient_matrix'])

    print(f"\nDataset: {len(df)} compounds")
    print(f"Tasks: {len(tasks)}")
    print(f"  Toxicity: {len(results['tox_tasks'])}")
    print(f"  ADME: {len(results['adme_tasks'])}")

    # Compute empirical correlations
    print("\nComputing empirical correlations...")
    E = compute_empirical_correlations(df, tasks)

    # Extract upper triangle (excluding diagonal)
    mask = np.triu(np.ones_like(G, dtype=bool), k=1)

    g_values = G[mask]
    e_values = E[mask]

    # Remove NaN pairs
    valid = ~np.isnan(g_values) & ~np.isnan(e_values)
    g_valid = g_values[valid]
    e_valid = e_values[valid]

    print(f"\nValid pairs: {len(g_valid)} / {len(g_values)}")

    # Compute G vs Empirical correlation
    r, p = stats.pearsonr(g_valid, e_valid)

    print("\n" + "=" * 60)
    print("MAIN RESULT: G vs EMPIRICAL CORRELATION")
    print("=" * 60)
    print(f"\nPearson r = {r:.3f}")
    print(f"p-value = {p:.2e}")
    print(f"N pairs = {len(g_valid)}")

    if p < 0.001:
        sig = "***"
    elif p < 0.01:
        sig = "**"
    elif p < 0.05:
        sig = "*"
    else:
        sig = "(n.s.)"

    print(f"\nResult: r = {r:.3f}{sig}")

    # Compare to previous validations
    print("\n" + "=" * 60)
    print("COMPARISON TO OTHER DATASETS")
    print("=" * 60)
    print(f"\n  Tox21 (12 tasks, 100% overlap): r = 0.918***")
    print(f"  ToxCast (17 tasks, ~80% overlap): r = 0.862***")
    print(f"  Tox21+ADME (16 tasks, 100% overlap): r = {r:.3f}{sig}")

    # Analyze by category
    print("\n" + "=" * 60)
    print("CORRELATION BY CATEGORY")
    print("=" * 60)

    tox_tasks = results['tox_tasks']
    adme_tasks = results['adme_tasks']

    # Within-Tox
    tox_pairs_g, tox_pairs_e = [], []
    for i, t_i in enumerate(tasks):
        for j, t_j in enumerate(tasks):
            if i >= j:
                continue
            if t_i in tox_tasks and t_j in tox_tasks:
                if not np.isnan(G[i, j]) and not np.isnan(E[i, j]):
                    tox_pairs_g.append(G[i, j])
                    tox_pairs_e.append(E[i, j])

    if len(tox_pairs_g) >= 3:
        r_tox, p_tox = stats.pearsonr(tox_pairs_g, tox_pairs_e)
        print(f"\nWithin-Toxicity: r = {r_tox:.3f} (p={p_tox:.3f}), N={len(tox_pairs_g)}")

    # Within-ADME
    adme_pairs_g, adme_pairs_e = [], []
    for i, t_i in enumerate(tasks):
        for j, t_j in enumerate(tasks):
            if i >= j:
                continue
            if t_i in adme_tasks and t_j in adme_tasks:
                if not np.isnan(G[i, j]) and not np.isnan(E[i, j]):
                    adme_pairs_g.append(G[i, j])
                    adme_pairs_e.append(E[i, j])

    if len(adme_pairs_g) >= 3:
        r_adme, p_adme = stats.pearsonr(adme_pairs_g, adme_pairs_e)
        print(f"Within-ADME: r = {r_adme:.3f} (p={p_adme:.3f}), N={len(adme_pairs_g)}")

    # Cross-domain
    cross_pairs_g, cross_pairs_e = [], []
    for i, t_i in enumerate(tasks):
        for j, t_j in enumerate(tasks):
            if i >= j:
                continue
            is_cross = (t_i in tox_tasks and t_j in adme_tasks) or \
                       (t_i in adme_tasks and t_j in tox_tasks)
            if is_cross:
                if not np.isnan(G[i, j]) and not np.isnan(E[i, j]):
                    cross_pairs_g.append(G[i, j])
                    cross_pairs_e.append(E[i, j])

    if len(cross_pairs_g) >= 3:
        r_cross, p_cross = stats.pearsonr(cross_pairs_g, cross_pairs_e)
        print(f"Cross-Domain: r = {r_cross:.3f} (p={p_cross:.3f}), N={len(cross_pairs_g)}")

    # Top empirical correlations vs G
    print("\n" + "=" * 60)
    print("TOP EMPIRICAL CORRELATIONS")
    print("=" * 60)

    pairs = []
    for i, t_i in enumerate(tasks):
        for j, t_j in enumerate(tasks):
            if i >= j:
                continue
            if not np.isnan(E[i, j]):
                pairs.append({
                    'task_i': t_i,
                    'task_j': t_j,
                    'empirical': E[i, j],
                    'gradient': G[i, j] if not np.isnan(G[i, j]) else 0
                })

    pairs.sort(key=lambda x: abs(x['empirical']), reverse=True)

    print("\nTop 10 by |empirical r|:")
    print(f"{'Task Pair':<50} {'Emp r':>8} {'G':>8}")
    print("-" * 70)
    for p in pairs[:10]:
        pair_name = f"{p['task_i']} vs {p['task_j']}"
        print(f"{pair_name:<50} {p['empirical']:>8.3f} {p['gradient']:>8.3f}")

    # Same property validation
    print("\n" + "=" * 60)
    print("SAME PROPERTY VALIDATION (KEY CHECK)")
    print("=" * 60)

    # Find same-property pairs (e.g., Lipophilicity from 2 sources)
    same_prop = [
        ('ADME_Lipophilicity', 'MN_Lipophilicity_MN', 'Lipophilicity'),
        ('ADME_Solubility', 'MN_ESOL', 'Solubility'),
    ]

    print("\nSame property measured by different sources:")
    print(f"{'Property':<15} {'Empirical r':>12} {'Gradient G':>12} {'Match':>8}")
    print("-" * 50)

    for t1, t2, name in same_prop:
        if t1 in tasks and t2 in tasks:
            i, j = tasks.index(t1), tasks.index(t2)
            emp_r = E[i, j]
            grad_g = G[i, j]
            match = "YES" if (emp_r > 0.5 and grad_g > 0.2) else "NO"
            print(f"{name:<15} {emp_r:>12.3f} {grad_g:>12.3f} {match:>8}")

    # Save results
    output = {
        'pearson_r': float(r),
        'p_value': float(p) if isinstance(p, (int, float)) else 0.0,
        'n_pairs': int(len(g_valid)),
        'significance': sig,
        'within_tox_r': float(r_tox) if len(tox_pairs_g) >= 3 else None,
        'within_tox_p': float(p_tox) if len(tox_pairs_g) >= 3 else None,
        'within_adme_r': float(r_adme) if len(adme_pairs_g) >= 3 else None,
        'within_adme_p': float(p_adme) if len(adme_pairs_g) >= 3 else None,
        'cross_domain_r': float(r_cross) if len(cross_pairs_g) >= 3 else None,
        'cross_domain_p': float(p_cross) if len(cross_pairs_g) >= 3 else None,
        'comparison': {
            'tox21': 0.918,
            'toxcast': 0.862,
            'tox21_adme': float(r)
        }
    }

    output_path = project_root / 'outputs' / 'tox21_adme_results' / 'validation_correlation.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Final assessment
    print("\n" + "=" * 60)
    print("VALIDATION ASSESSMENT")
    print("=" * 60)

    if r > 0.6 and p < 0.05:
        print("\n[PASS] Gradient conflicts correlate with empirical structure!")
        print(f"  r = {r:.3f} indicates gradient method captures property relationships")
    elif r > 0.4 and p < 0.05:
        print("\n[MARGINAL] Moderate correlation detected")
        print(f"  r = {r:.3f} - weaker than Tox21/ToxCast but still significant")
    else:
        print("\n[WEAK] Low correlation")
        print(f"  r = {r:.3f} - gradient patterns don't match empirical correlations well")

    return r, p


if __name__ == '__main__':
    main()
