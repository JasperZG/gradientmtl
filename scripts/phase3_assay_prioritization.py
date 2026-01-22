#!/usr/bin/env python3
"""
Phase 3A: Assay Prioritization Use Case

Demonstrates practical application: "Given budget for N assays, which kinases should we screen?"

Key outputs:
1. Coverage curves comparing selection strategies
2. Cost-benefit analysis at different budgets
3. Recommended kinase panels for different screening scenarios
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def load_gradient_matrix(results_dir: str) -> tuple:
    """Load gradient matrix and task names."""
    data = np.load(f"{results_dir}/gradient_matrices.npz", allow_pickle=True)
    G = data['average_matrix']
    tasks = list(data['tasks'])
    return G, tasks


def load_selection_results(phase2_dir: str) -> pd.DataFrame:
    """Load task selection results from Phase 2."""
    return pd.read_csv(f"{phase2_dir}/selection_results.csv")


def greedy_selection(G: np.ndarray, tasks: list, budget: int) -> tuple:
    """
    Greedy task selection maximizing gradient coverage.

    Returns: (selected_tasks, coverage_at_each_step)
    """
    n_tasks = len(tasks)
    selected_idx = []
    remaining_idx = list(range(n_tasks))
    coverages = []

    for _ in range(min(budget, n_tasks)):
        best_idx = None
        best_coverage = -np.inf

        for idx in remaining_idx:
            # Coverage = max gradient correlation to any selected task
            if not selected_idx:
                # First selection: pick task with highest sum of correlations
                coverage = np.sum(np.abs(G[idx, :]))
            else:
                # Subsequent: pick task maximizing coverage of unselected
                unselected = [i for i in remaining_idx if i != idx]
                if unselected:
                    coverage = np.max([G[idx, j] for j in unselected])
                else:
                    coverage = 0

            if coverage > best_coverage:
                best_coverage = coverage
                best_idx = idx

        selected_idx.append(best_idx)
        remaining_idx.remove(best_idx)

        # Compute total coverage: max correlation from any selected to any unselected
        if remaining_idx:
            total_cov = np.mean([
                np.max([G[s, u] for s in selected_idx])
                for u in remaining_idx
            ])
        else:
            total_cov = 1.0
        coverages.append(total_cov)

    selected_tasks = [tasks[i] for i in selected_idx]
    return selected_tasks, coverages


def random_selection_baseline(G: np.ndarray, tasks: list, budget: int, n_trials: int = 100) -> tuple:
    """Random selection baseline with confidence intervals."""
    n_tasks = len(tasks)
    all_coverages = []

    for _ in range(n_trials):
        perm = np.random.permutation(n_tasks)
        selected_idx = list(perm[:budget])
        remaining_idx = list(perm[budget:])

        if remaining_idx:
            coverage = np.mean([
                np.max([G[s, u] for s in selected_idx])
                for u in remaining_idx
            ])
        else:
            coverage = 1.0
        all_coverages.append(coverage)

    return np.mean(all_coverages), np.std(all_coverages)


def generate_coverage_plot(G: np.ndarray, tasks: list, output_dir: str):
    """Generate coverage curves comparing selection strategies."""
    max_budget = min(10, len(tasks) - 1)
    budgets = range(2, max_budget + 1)

    greedy_coverages = []
    random_means = []
    random_stds = []

    for b in budgets:
        _, covs = greedy_selection(G, tasks, b)
        greedy_coverages.append(covs[-1])

        r_mean, r_std = random_selection_baseline(G, tasks, b)
        random_means.append(r_mean)
        random_stds.append(r_std)

    plt.figure(figsize=(10, 6))
    plt.plot(budgets, greedy_coverages, 'b-o', linewidth=2, markersize=8, label='Gradient-informed (Greedy)')
    plt.errorbar(budgets, random_means, yerr=random_stds, fmt='r--s', linewidth=2, markersize=6,
                 capsize=4, label='Random (mean ± std)')

    plt.xlabel('Number of Kinases Selected (Budget)', fontsize=12)
    plt.ylabel('Coverage Score', fontsize=12)
    plt.title('Assay Prioritization: Gradient-Informed vs Random Selection', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(f"{output_dir}/assay_prioritization_coverage.png", dpi=150)
    plt.close()
    print(f"Saved coverage plot to {output_dir}/assay_prioritization_coverage.png")


def generate_recommendations(G: np.ndarray, tasks: list, output_dir: str):
    """Generate practical screening recommendations."""
    recommendations = {
        "title": "Kinase Screening Panel Recommendations",
        "description": "Gradient-informed kinase selection for different screening budgets",
        "panels": []
    }

    for budget in [3, 5, 7, 10]:
        if budget >= len(tasks):
            continue

        selected, coverages = greedy_selection(G, tasks, budget)
        r_mean, r_std = random_selection_baseline(G, tasks, budget)
        improvement = (coverages[-1] - r_mean) / r_mean * 100 if r_mean > 0 else 0

        panel = {
            "budget": budget,
            "selected_kinases": [t.replace('_pIC50', '') for t in selected],
            "coverage": round(coverages[-1], 3),
            "random_baseline": round(r_mean, 3),
            "improvement_pct": round(improvement, 1),
            "rationale": generate_rationale(selected, G, tasks)
        }
        recommendations["panels"].append(panel)

    # Save recommendations
    with open(f"{output_dir}/screening_recommendations.json", 'w') as f:
        json.dump(recommendations, f, indent=2)

    # Generate markdown report
    report = generate_markdown_report(recommendations)
    with open(f"{output_dir}/screening_recommendations.md", 'w') as f:
        f.write(report)

    print(f"Saved recommendations to {output_dir}/screening_recommendations.json")
    print(f"Saved report to {output_dir}/screening_recommendations.md")

    return recommendations


def generate_rationale(selected: list, G: np.ndarray, tasks: list) -> str:
    """Generate brief rationale for selection."""
    # Identify which families are represented
    families = {
        'CDK': ['CDK1', 'CDK2', 'CDK4', 'CDK5', 'CDK6', 'CDK7', 'CDK9'],
        'JAK': ['JAK1', 'JAK2', 'JAK3', 'TYK2'],
        'SRC': ['SRC', 'FYN', 'LCK', 'LYN', 'YES1'],
        'Aurora': ['AURKA', 'AURKB', 'AURKC'],
        'EGFR': ['EGFR', 'ERBB2', 'ERBB4']
    }

    selected_clean = [t.replace('_pIC50', '') for t in selected]
    represented = []

    for family, members in families.items():
        if any(m in selected_clean for m in members):
            represented.append(family)

    if len(represented) >= 3:
        return f"Diverse panel covering {', '.join(represented)} families"
    elif len(represented) == 2:
        return f"Focused panel on {' and '.join(represented)} families"
    else:
        return "Specialized panel for detailed selectivity profiling"


def generate_markdown_report(recommendations: dict) -> str:
    """Generate markdown report for recommendations."""
    lines = [
        "# Kinase Screening Panel Recommendations",
        "",
        "## Overview",
        "These recommendations are based on gradient-informed task selection,",
        "which identifies kinases that maximize information coverage while minimizing screening costs.",
        "",
        "## Recommended Panels by Budget",
        ""
    ]

    for panel in recommendations["panels"]:
        lines.extend([
            f"### Budget: {panel['budget']} Kinases",
            "",
            f"**Selected:** {', '.join(panel['selected_kinases'])}",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Coverage Score | {panel['coverage']} |",
            f"| Random Baseline | {panel['random_baseline']} |",
            f"| Improvement | +{panel['improvement_pct']}% |",
            "",
            f"**Rationale:** {panel['rationale']}",
            ""
        ])

    lines.extend([
        "## Methodology",
        "",
        "The gradient-informed selection algorithm:",
        "1. Computes pairwise gradient correlations during multi-task GNN training",
        "2. Iteratively selects kinases that maximize coverage of unselected kinases",
        "3. Coverage is measured as the average maximum gradient correlation",
        "",
        "## Limitations",
        "",
        "- Requires ≥50% compound overlap for reliable gradient estimates",
        "- Coverage scores are relative, not absolute predictive accuracy",
        "- Should be combined with domain expertise for final panel design",
        ""
    ])

    return '\n'.join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Phase 3A: Assay Prioritization')
    parser.add_argument('--results-dir', default='outputs/kinase_all_results',
                        help='Directory with gradient matrices')
    parser.add_argument('--output-dir', default='outputs/phase3_assay_prioritization',
                        help='Output directory')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Phase 3A: Assay Prioritization Analysis")
    print("=" * 50)

    # Load gradient matrix
    print("\nLoading gradient matrix...")
    G, tasks = load_gradient_matrix(args.results_dir)
    print(f"Loaded {len(tasks)} kinases")

    # Generate coverage plot
    print("\nGenerating coverage curves...")
    generate_coverage_plot(G, tasks, str(output_dir))

    # Generate recommendations
    print("\nGenerating screening recommendations...")
    recommendations = generate_recommendations(G, tasks, str(output_dir))

    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY: Recommended Kinase Panels")
    print("=" * 50)
    for panel in recommendations["panels"]:
        print(f"\nBudget {panel['budget']}: {', '.join(panel['selected_kinases'])}")
        print(f"  Coverage: {panel['coverage']} (+{panel['improvement_pct']}% vs random)")


if __name__ == '__main__':
    main()
