#!/usr/bin/env python3
"""
Experiment 2: SAR (Structure-Activity Relationship) Validation

Validates that gradient conflicts correlate with documented mechanistic
relationships from medicinal chemistry literature.

Key analyses:
1. Correlation between gradient matrix G and literature matrix L
2. Permutation test for statistical significance
3. Bootstrap confidence intervals
4. Sign agreement analysis (do predicted trade-offs match literature?)
5. Cluster recovery score (do discovered clusters match known mechanisms?)

Expected outcome: Pearson r > 0.6, p < 0.001
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from analysis.literature_matrix import (
    get_literature_matrix,
    validate_gradient_matrix,
    get_literature_tradeoffs_list,
    get_literature_synergies_list,
    LITERATURE_RELATIONSHIPS,
)
from analysis.statistical_tests import (
    permutation_test,
    bootstrap_confidence_interval,
    compute_silhouette_score,
    comprehensive_statistical_report,
)


# Known mechanistic clusters for validation
KNOWN_CLUSTERS = {
    'CYP_enzymes': ['CYP2D6_Veith', 'CYP3A4_Veith', 'CYP2C9_Veith', 'CYP2C19_Veith', 'CYP1A2_Veith'],
    'Permeability': ['BBBP', 'BBB_Martins', 'Caco2_Wang', 'HIA_Hou'],
    'Clearance': ['Clearance_Hepatocyte_AZ', 'Clearance_Microsome_AZ', 'Half_Life_Obach'],
    'Nuclear_receptors': ['Tox21_NR-AR', 'Tox21_NR-AR-LBD', 'Tox21_NR-ER', 'Tox21_NR-ER-LBD'],
    'Stress_response': ['Tox21_SR-ARE', 'Tox21_SR-HSE', 'Tox21_SR-MMP', 'Tox21_SR-p53'],
    'Physicochemical': ['ESOL', 'Lipophilicity', 'FreeSolv', 'Solubility_AqSolDB'],
}


def load_gradient_matrix(path: Path) -> tuple:
    """Load gradient conflict matrix from .npz file."""
    data = np.load(path, allow_pickle=True)
    G = data['averaged']
    task_names = data['task_names'].tolist()
    return G, task_names


def plot_gradient_vs_literature(
    G: np.ndarray,
    L: np.ndarray,
    task_names: list,
    output_path: Path
):
    """
    Scatter plot comparing gradient conflicts to literature expectations.
    """
    K = len(task_names)
    mask = ~np.eye(K, dtype=bool)
    g_flat = G[mask]
    l_flat = L[mask]

    # Filter to documented pairs
    nonzero = l_flat != 0
    g_doc = g_flat[nonzero]
    l_doc = l_flat[nonzero]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: All pairs
    ax1 = axes[0]
    ax1.scatter(l_flat, g_flat, alpha=0.3, s=20)
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)

    # Add regression line
    if len(l_flat) > 2:
        slope, intercept, r, p, _ = stats.linregress(l_flat, g_flat)
        x_line = np.array([-1, 1])
        ax1.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2,
                 label=f'r = {r:.3f}, p = {p:.2e}')
        ax1.legend()

    ax1.set_xlabel('Literature Expected Correlation', fontsize=12)
    ax1.set_ylabel('Gradient Conflict', fontsize=12)
    ax1.set_title('All Task Pairs', fontsize=14)
    ax1.set_xlim(-1.1, 1.1)
    ax1.set_ylim(-1.1, 1.1)

    # Right: Only documented pairs
    ax2 = axes[1]
    colors = ['red' if l < 0 else 'green' for l in l_doc]
    ax2.scatter(l_doc, g_doc, c=colors, alpha=0.6, s=50)
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(0, color='gray', linestyle='--', alpha=0.5)

    if len(l_doc) > 2:
        slope, intercept, r, p, _ = stats.linregress(l_doc, g_doc)
        x_line = np.array([-1, 1])
        ax2.plot(x_line, slope * x_line + intercept, 'b-', linewidth=2,
                 label=f'r = {r:.3f}, p = {p:.2e}')
        ax2.legend()

    # Add identity line
    ax2.plot([-1, 1], [-1, 1], 'k:', alpha=0.5, label='y = x')

    ax2.set_xlabel('Literature Expected Correlation', fontsize=12)
    ax2.set_ylabel('Gradient Conflict', fontsize=12)
    ax2.set_title(f'Documented Pairs Only (n={len(g_doc)})', fontsize=14)
    ax2.set_xlim(-1.1, 1.1)
    ax2.set_ylim(-1.1, 1.1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved scatter plot to {output_path}")


def plot_permutation_distribution(
    perm_result: dict,
    output_path: Path
):
    """Plot permutation test null distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))

    null_dist = perm_result['null_distribution']
    obs = perm_result['observed_statistic']

    ax.hist(null_dist, bins=50, density=True, alpha=0.7, color='gray',
            label='Null distribution')
    ax.axvline(obs, color='red', linewidth=2, linestyle='--',
               label=f'Observed r = {obs:.3f}')

    # Add significance threshold
    threshold = np.percentile(np.abs(null_dist), 95)
    ax.axvline(threshold, color='orange', linewidth=1, linestyle=':',
               label=f'95% threshold = ±{threshold:.3f}')
    ax.axvline(-threshold, color='orange', linewidth=1, linestyle=':')

    ax.set_xlabel('Correlation Coefficient', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f"Permutation Test (p = {perm_result['p_value']:.4f})", fontsize=14)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved permutation plot to {output_path}")


def plot_specific_relationships(
    G: np.ndarray,
    task_names: list,
    output_path: Path
):
    """
    Bar plot comparing specific gradient values to literature expectations.
    """
    # Key relationships to highlight
    key_relationships = [
        ('ESOL', 'Lipophilicity', -0.8, 'Solubility vs Lipophilicity'),
        ('BBBP', 'Caco2_Wang', 0.6, 'BBB vs Caco-2 Permeability'),
        ('BBBP', 'Lipophilicity', 0.5, 'BBB vs Lipophilicity'),
        ('hERG', 'Lipophilicity', 0.4, 'hERG vs Lipophilicity'),
        ('CYP2D6_Veith', 'CYP3A4_Veith', 0.4, 'CYP2D6 vs CYP3A4'),
    ]

    # Find matching tasks
    relationships = []
    for t1, t2, expected, label in key_relationships:
        # Check for exact or partial matches
        i1 = j1 = None
        for i, t in enumerate(task_names):
            if t1 in t or t in t1:
                i1 = i
            if t2 in t or t in t2:
                j1 = i

        if i1 is not None and j1 is not None and i1 != j1:
            observed = G[i1, j1]
            relationships.append({
                'label': label,
                'expected': expected,
                'observed': observed,
                'tasks': f"{task_names[i1]} vs {task_names[j1]}"
            })

    if not relationships:
        print("No matching relationships found in gradient matrix")
        return

    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))

    labels = [r['label'] for r in relationships]
    expected = [r['expected'] for r in relationships]
    observed = [r['observed'] for r in relationships]

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, expected, width, label='Literature Expected',
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, observed, width, label='Gradient Observed',
                   color='coral', alpha=0.8)

    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.set_ylabel('Correlation', fontsize=12)
    ax.set_title('Key Mechanistic Relationships: Literature vs Gradient', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(-1, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved relationship comparison to {output_path}")


def run_sar_validation(
    gradient_matrix_path: str,
    output_dir: str = 'outputs/sar_validation',
    verbose: bool = True
) -> dict:
    """
    Run complete SAR validation experiment.

    Args:
        gradient_matrix_path: Path to gradient conflict matrix (.npz)
        output_dir: Directory for outputs
        verbose: Print detailed results

    Returns:
        Dict with all validation results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("EXPERIMENT 2: SAR VALIDATION")
    print("=" * 70)

    # Load gradient matrix
    print(f"\nLoading gradient matrix from {gradient_matrix_path}...")
    G, task_names = load_gradient_matrix(Path(gradient_matrix_path))
    print(f"Loaded {len(task_names)} tasks: {task_names[:5]}...")

    # Construct literature matrix
    print("\nConstructing literature relationship matrix...")
    L = get_literature_matrix(task_names)
    n_documented = np.sum(L[~np.eye(len(task_names), dtype=bool)] != 0)
    print(f"Found {n_documented} documented relationships for these tasks")

    # Basic validation
    print("\n" + "-" * 50)
    print("BASIC VALIDATION")
    print("-" * 50)
    basic_results = validate_gradient_matrix(G, task_names, verbose=verbose)

    # Comprehensive statistical analysis
    print("\n" + "-" * 50)
    print("STATISTICAL ANALYSIS")
    print("-" * 50)

    # Filter known clusters to those present in task_names
    filtered_clusters = {}
    for cluster_name, cluster_tasks in KNOWN_CLUSTERS.items():
        present = [t for t in cluster_tasks if any(t in tn or tn in t for tn in task_names)]
        if len(present) >= 2:
            filtered_clusters[cluster_name] = present

    stat_results = comprehensive_statistical_report(
        G, L, task_names,
        true_clusters=filtered_clusters if filtered_clusters else None,
        verbose=verbose
    )

    # Generate plots
    print("\n" + "-" * 50)
    print("GENERATING VISUALIZATIONS")
    print("-" * 50)

    plot_gradient_vs_literature(G, L, task_names,
                                output_dir / 'gradient_vs_literature_scatter.png')

    if 'permutation_test' in stat_results and 'error' not in stat_results['permutation_test']:
        plot_permutation_distribution(stat_results['permutation_test'],
                                      output_dir / 'permutation_test.png')

    plot_specific_relationships(G, task_names,
                                output_dir / 'key_relationships.png')

    # Compile results
    results = {
        'n_tasks': len(task_names),
        'task_names': task_names,
        'n_documented_pairs': n_documented,
        'basic_validation': basic_results,
        'statistical_analysis': {
            k: v for k, v in stat_results.items()
            if k not in ['permutation_test', 'bootstrap_ci']  # Exclude large arrays
        },
    }

    # Add key metrics
    if 'permutation_test' in stat_results and 'error' not in stat_results['permutation_test']:
        results['pearson_r'] = stat_results['permutation_test']['observed_statistic']
        results['permutation_p'] = stat_results['permutation_test']['p_value']

    if 'bootstrap_ci' in stat_results and 'error' not in stat_results['bootstrap_ci']:
        results['bootstrap_ci'] = [
            stat_results['bootstrap_ci']['ci_lower'],
            stat_results['bootstrap_ci']['ci_upper']
        ]

    if 'silhouette' in stat_results and 'error' not in stat_results['silhouette']:
        results['silhouette_score'] = stat_results['silhouette']['silhouette_score']
        results['n_clusters'] = stat_results['silhouette']['n_clusters']

    # Save results
    results_file = output_dir / 'sar_validation_results.json'
    with open(results_file, 'w') as f:
        # Convert numpy types for JSON
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, dict):
                # Convert both keys and values
                return {str(k) if isinstance(k, (np.integer, np.floating)) else k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        json.dump(convert(results), f, indent=2)

    print(f"\nResults saved to {results_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    if 'pearson_r' in results:
        print(f"Pearson correlation with literature: r = {results['pearson_r']:.4f}")
    if 'permutation_p' in results:
        print(f"Permutation test p-value: p = {results['permutation_p']:.4f}")

        if results['permutation_p'] < 0.001:
            print("*** HIGHLY SIGNIFICANT (p < 0.001) ***")
        elif results['permutation_p'] < 0.01:
            print("** SIGNIFICANT (p < 0.01) **")
        elif results['permutation_p'] < 0.05:
            print("* SIGNIFICANT (p < 0.05) *")
        else:
            print("Not significant at α = 0.05")

    if 'silhouette_score' in results:
        print(f"Clustering quality (silhouette): {results['silhouette_score']:.4f}")

    # Check success criteria from research plan
    print("\n" + "-" * 50)
    print("SUCCESS CRITERIA CHECK")
    print("-" * 50)

    success = True
    if 'pearson_r' in results:
        if results['pearson_r'] > 0.6:
            print(f"✓ Pearson r > 0.6: PASS ({results['pearson_r']:.3f})")
        else:
            print(f"✗ Pearson r > 0.6: FAIL ({results['pearson_r']:.3f})")
            success = False

    if 'permutation_p' in results:
        if results['permutation_p'] < 0.001:
            print(f"✓ p < 0.001: PASS ({results['permutation_p']:.4f})")
        else:
            print(f"✗ p < 0.001: FAIL ({results['permutation_p']:.4f})")
            success = False

    if 'silhouette_score' in results:
        if results['silhouette_score'] > 0.5:
            print(f"✓ Silhouette > 0.5: PASS ({results['silhouette_score']:.3f})")
        else:
            # Silhouette is a soft criterion - low score just means tasks are relatively independent
            print(f"~ Silhouette > 0.5: SOFT FAIL ({results['silhouette_score']:.3f}) - tasks may be independent")

    if 'basic_validation' in results:
        sign_agree = results['basic_validation'].get('sign_agreement', 0)
        if sign_agree > 0.8:
            print(f"✓ Sign agreement > 80%: PASS ({sign_agree:.1%})")
        else:
            print(f"✗ Sign agreement > 80%: FAIL ({sign_agree:.1%})")
            success = False

    results['all_criteria_passed'] = success

    return results


def main():
    parser = argparse.ArgumentParser(description='SAR Validation Experiment')
    parser.add_argument('--gradient-matrix', type=str,
                       default='outputs/gradients/gnn_conflict_matrices.npz',
                       help='Path to gradient conflict matrix')
    parser.add_argument('--output-dir', type=str, default='outputs/sar_validation',
                       help='Output directory')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')

    args = parser.parse_args()

    results = run_sar_validation(
        gradient_matrix_path=args.gradient_matrix,
        output_dir=args.output_dir,
        verbose=not args.quiet
    )

    # Exit with appropriate code
    sys.exit(0 if results.get('all_criteria_passed', False) else 1)


if __name__ == '__main__':
    main()
