#!/usr/bin/env python3
"""
Experiment 4: Task Selection Based on Gradient Conflicts

Key contribution: Given a budget of B tasks to measure, which B should we choose
to maximize predictive coverage via transfer learning?

Algorithms compared:
1. Greedy selection (maximize coverage based on gradient correlations)
2. Random selection (baseline, 100 draws per budget)
3. Clustering-based selection (one task per cluster)
4. Max-diversity selection (maximize pairwise dissimilarity)

Success criteria:
- Greedy outperforms random by >20% for all budgets
- 5-6 tasks achieve >75% coverage
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import urllib.request
import gzip
import io
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.graph_preprocessing import MoleculeGraphPreprocessor, get_atom_feature_dim
from data.splitting import scaffold_split
from data.graph_dataset import MultiTaskGraphDataset
from models.gnn_multitask import GNNMultiTaskModel
from training.gnn_trainer import GNNMultiTaskTrainer


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

TASK_NAMES = list(TOX21_TASKS.keys())


def load_gradient_conflicts(path: Path) -> tuple[np.ndarray, list]:
    """Load gradient conflict matrix from previous GNN experiment."""
    data = np.load(path, allow_pickle=True)
    matrix = data['averaged']
    task_names = data['task_names'].tolist()
    return matrix, task_names


def greedy_task_selection(gradient_matrix: np.ndarray, task_names: list, budget: int) -> tuple[list, list]:
    """
    Greedy algorithm to select B tasks maximizing coverage.

    Coverage is defined as the sum of positive gradient correlations from
    selected tasks to unselected tasks.

    Args:
        gradient_matrix: N×N matrix of gradient correlations
        task_names: List of N task names
        budget: Number of tasks to select (e.g., 3, 4, 5, 6, 7)

    Returns:
        selected_tasks: List of selected task names
        coverage_history: Coverage score at each selection step
    """
    n_tasks = len(task_names)
    selected_indices = []
    remaining_indices = set(range(n_tasks))
    coverage_history = []

    for step in range(budget):
        best_task_idx = None
        best_marginal_coverage = -float('inf')

        for candidate_idx in remaining_indices:
            # Compute marginal coverage: sum of positive correlations to unselected tasks
            marginal_coverage = 0
            for other_idx in remaining_indices:
                if other_idx != candidate_idx:
                    # Coverage = how well this task can predict others via transfer
                    g = gradient_matrix[candidate_idx, other_idx]
                    marginal_coverage += max(0, g)

            if marginal_coverage > best_marginal_coverage:
                best_marginal_coverage = marginal_coverage
                best_task_idx = candidate_idx

        selected_indices.append(best_task_idx)
        remaining_indices.remove(best_task_idx)
        coverage_history.append(best_marginal_coverage)

    selected_tasks = [task_names[i] for i in selected_indices]
    return selected_tasks, coverage_history


def random_task_selection(task_names: list, budget: int, n_draws: int = 100, seed: int = 42) -> list:
    """
    Random baseline: sample random task subsets.

    Args:
        task_names: List of N task names
        budget: Number of tasks to select
        n_draws: Number of random draws for averaging
        seed: Random seed

    Returns:
        List of lists (each is a random selection)
    """
    rng = np.random.RandomState(seed)
    selections = []

    for _ in range(n_draws):
        selected = rng.choice(task_names, size=budget, replace=False).tolist()
        selections.append(selected)

    return selections


def clustering_task_selection(gradient_matrix: np.ndarray, task_names: list, budget: int) -> list:
    """
    Clustering-based selection: cluster tasks, select one representative per cluster.

    Args:
        gradient_matrix: N×N matrix of gradient correlations
        task_names: List of N task names
        budget: Number of tasks to select (= number of clusters)

    Returns:
        selected_tasks: List of selected task names
    """
    # Convert similarity to distance
    distance_matrix = 1 - gradient_matrix
    np.fill_diagonal(distance_matrix, 0)

    # Cluster
    clustering = AgglomerativeClustering(
        n_clusters=budget,
        metric='precomputed',
        linkage='average',
    )
    cluster_labels = clustering.fit_predict(distance_matrix)

    # Select the task with highest within-cluster centrality
    selected_tasks = []
    for cluster_id in range(budget):
        cluster_members = [i for i, l in enumerate(cluster_labels) if l == cluster_id]

        if len(cluster_members) == 1:
            selected_tasks.append(task_names[cluster_members[0]])
        else:
            # Select task with highest average correlation to other cluster members
            best_task = None
            best_centrality = -float('inf')

            for member in cluster_members:
                centrality = sum(
                    gradient_matrix[member, other]
                    for other in cluster_members if other != member
                )
                if centrality > best_centrality:
                    best_centrality = centrality
                    best_task = task_names[member]

            selected_tasks.append(best_task)

    return selected_tasks


def max_diversity_selection(gradient_matrix: np.ndarray, task_names: list, budget: int) -> list:
    """
    Max-diversity selection: select tasks to maximize pairwise dissimilarity.

    Greedy algorithm that maximizes sum of pairwise distances.

    Args:
        gradient_matrix: N×N matrix of gradient correlations
        task_names: List of N task names
        budget: Number of tasks to select

    Returns:
        selected_tasks: List of selected task names
    """
    n_tasks = len(task_names)
    distance_matrix = 1 - gradient_matrix
    np.fill_diagonal(distance_matrix, 0)

    # Start with the two most distant tasks
    max_dist = -1
    start_pair = (0, 1)
    for i in range(n_tasks):
        for j in range(i + 1, n_tasks):
            if distance_matrix[i, j] > max_dist:
                max_dist = distance_matrix[i, j]
                start_pair = (i, j)

    selected_indices = list(start_pair)
    remaining_indices = set(range(n_tasks)) - set(selected_indices)

    # Greedily add tasks that maximize minimum distance to selected
    while len(selected_indices) < budget:
        best_task_idx = None
        best_min_dist = -1

        for candidate in remaining_indices:
            min_dist = min(distance_matrix[candidate, s] for s in selected_indices)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_task_idx = candidate

        selected_indices.append(best_task_idx)
        remaining_indices.remove(best_task_idx)

    return [task_names[i] for i in selected_indices]


def compute_coverage(selected_tasks: list, gradient_matrix: np.ndarray, task_names: list) -> float:
    """
    Compute coverage score for a task selection.

    Coverage = average max correlation from selected to unselected tasks.
    """
    selected_indices = [task_names.index(t) for t in selected_tasks]
    unselected_indices = [i for i in range(len(task_names)) if i not in selected_indices]

    if len(unselected_indices) == 0:
        return 1.0

    total_coverage = 0
    for unselected in unselected_indices:
        # For this unselected task, find the best selected task to transfer from
        best_correlation = max(
            gradient_matrix[selected, unselected]
            for selected in selected_indices
        )
        total_coverage += max(0, best_correlation)

    return total_coverage / len(unselected_indices)


def plot_task_dendrogram(gradient_matrix: np.ndarray, task_names: list, output_path: Path):
    """Plot dendrogram of task clustering."""
    distance_matrix = 1 - gradient_matrix
    np.fill_diagonal(distance_matrix, 0)

    condensed = squareform(distance_matrix)
    Z = linkage(condensed, method='average')

    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(Z, labels=task_names, leaf_rotation=45, ax=ax)
    ax.set_title('Task Clustering Based on Gradient Similarity')
    ax.set_ylabel('Distance (1 - Similarity)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved dendrogram to {output_path}")


def plot_coverage_curves(results: pd.DataFrame, output_path: Path):
    """Plot coverage vs budget for all methods."""
    fig, ax = plt.subplots(figsize=(10, 6))

    budgets = sorted(results['budget'].unique())

    # Plot each method
    for method in ['greedy', 'clustering', 'diversity']:
        method_data = results[results['method'] == method]
        ax.plot(method_data['budget'], method_data['coverage'],
               marker='o', label=method.capitalize(), linewidth=2)

    # Random baseline with error bars
    random_data = results[results['method'] == 'random']
    random_mean = random_data.groupby('budget')['coverage'].mean()
    random_std = random_data.groupby('budget')['coverage'].std()
    ax.errorbar(random_mean.index, random_mean.values, yerr=random_std.values,
               marker='s', label='Random', linewidth=2, capsize=5)

    ax.set_xlabel('Budget (number of tasks)', fontsize=12)
    ax.set_ylabel('Coverage Score', fontsize=12)
    ax.set_title('Task Selection: Coverage vs Budget', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(budgets)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved coverage curves to {output_path}")


# Selection methods and budgets for array jobs
SELECTION_METHODS = ['greedy', 'clustering', 'diversity', 'random']
BUDGETS = [3, 4, 5, 6, 7]  # 5 different budget levels


def get_job_config(job_index: int) -> dict:
    """
    Map SLURM array index to experiment configuration.

    Total jobs: 4 methods × 5 budgets = 20

    Index mapping:
        job_index = method_idx * 5 + budget_idx
    """
    n_methods = len(SELECTION_METHODS)
    n_budgets = len(BUDGETS)

    if job_index >= n_methods * n_budgets:
        raise ValueError(f"Job index {job_index} out of range (max {n_methods * n_budgets - 1})")

    method_idx = job_index // n_budgets
    budget_idx = job_index % n_budgets

    return {
        'method': SELECTION_METHODS[method_idx],
        'budget': BUDGETS[budget_idx],
        'job_index': job_index
    }


def run_single_selection_experiment(
    method: str,
    budget: int,
    gradient_matrix: np.ndarray,
    task_names: list,
    n_random_draws: int = 100,
    seed: int = 42
) -> dict:
    """Run a single task selection experiment."""
    np.random.seed(seed)

    if method == 'greedy':
        selected_tasks, coverage_history = greedy_task_selection(gradient_matrix, task_names, budget)
        coverage = compute_coverage(selected_tasks, gradient_matrix, task_names)
        return {
            'method': method,
            'budget': budget,
            'selected_tasks': selected_tasks,
            'coverage': coverage,
            'coverage_history': coverage_history
        }
    elif method == 'clustering':
        selected_tasks = clustering_task_selection(gradient_matrix, task_names, budget)
        coverage = compute_coverage(selected_tasks, gradient_matrix, task_names)
        return {
            'method': method,
            'budget': budget,
            'selected_tasks': selected_tasks,
            'coverage': coverage
        }
    elif method == 'diversity':
        selected_tasks = max_diversity_selection(gradient_matrix, task_names, budget)
        coverage = compute_coverage(selected_tasks, gradient_matrix, task_names)
        return {
            'method': method,
            'budget': budget,
            'selected_tasks': selected_tasks,
            'coverage': coverage
        }
    elif method == 'random':
        # Run multiple random draws
        random_selections = random_task_selection(task_names, budget, n_random_draws, seed)
        random_coverages = [
            compute_coverage(sel, gradient_matrix, task_names)
            for sel in random_selections
        ]
        return {
            'method': method,
            'budget': budget,
            'mean_coverage': np.mean(random_coverages),
            'std_coverage': np.std(random_coverages),
            'all_coverages': random_coverages,
            'n_draws': n_random_draws
        }
    else:
        raise ValueError(f"Unknown method: {method}")


def aggregate_results(output_dir: Path):
    """Aggregate all task selection results from individual jobs."""
    print("\n" + "=" * 60)
    print("Aggregating Task Selection Results")
    print("=" * 60)

    results_by_method = {method: {} for method in SELECTION_METHODS}

    # Load all result files
    for f in output_dir.glob('selection_*.json'):
        with open(f) as fp:
            result = json.load(fp)
            method = result['method']
            budget = result['budget']

            if method in results_by_method:
                results_by_method[method][budget] = result

    # Print summary
    for method, budget_results in results_by_method.items():
        if not budget_results:
            continue

        print(f"\n{method.upper()}:")
        for budget in sorted(budget_results.keys()):
            result = budget_results[budget]
            if method == 'random':
                print(f"  Budget {budget}: coverage = {result['mean_coverage']:.4f} ± {result['std_coverage']:.4f}")
            else:
                print(f"  Budget {budget}: coverage = {result['coverage']:.4f}, tasks = {result['selected_tasks']}")

    # Compare greedy vs random
    print("\n" + "=" * 60)
    print("GREEDY VS RANDOM COMPARISON")
    print("=" * 60)

    for budget in BUDGETS:
        if budget in results_by_method['greedy'] and budget in results_by_method['random']:
            greedy_cov = results_by_method['greedy'][budget]['coverage']
            random_cov = results_by_method['random'][budget]['mean_coverage']
            improvement = (greedy_cov - random_cov) / random_cov * 100 if random_cov > 0 else 0

            print(f"Budget {budget}: Greedy={greedy_cov:.4f}, Random={random_cov:.4f}, Improvement=+{improvement:.1f}%")

    print(f"\nAggregation complete.")


def main():
    parser = argparse.ArgumentParser(description='Task Selection Experiment')
    parser.add_argument('--job-index', type=int, default=None,
                       help='SLURM array task ID (0-19). If not provided, runs all.')
    parser.add_argument('--gradient_matrix', type=str,
                       default='outputs/gradients/gnn_conflict_matrices.npz',
                       help='Path to gradient conflict matrix')
    parser.add_argument('--n_random_draws', type=int, default=100,
                       help='Number of random draws for baseline')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='outputs/experiment4')
    parser.add_argument('--aggregate', action='store_true',
                       help='Aggregate all results instead of running experiment')
    args = parser.parse_args()

    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate mode
    if args.aggregate:
        aggregate_results(output_dir)
        return

    # Load gradient conflict matrix
    print("\nLoading gradient conflict matrix...")
    gradient_path = Path(args.gradient_matrix)
    if not gradient_path.exists():
        print(f"ERROR: Gradient matrix not found at {gradient_path}")
        print("Please run train_tox21_gnn.py first to generate the matrix")
        return

    gradient_matrix, task_names = load_gradient_conflicts(gradient_path)
    print(f"Loaded matrix for {len(task_names)} tasks")

    # Determine what to run
    if args.job_index is not None:
        # Single job mode (for SLURM array)
        config = get_job_config(args.job_index)
        print(f"\nRunning single experiment: {config['method']} with budget {config['budget']}")

        result = run_single_selection_experiment(
            method=config['method'],
            budget=config['budget'],
            gradient_matrix=gradient_matrix,
            task_names=task_names,
            n_random_draws=args.n_random_draws,
            seed=args.seed
        )

        # Save result
        result_file = output_dir / f"selection_{config['method']}_budget{config['budget']}_seed{args.seed}.json"
        with open(result_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            result_json = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in result.items()}
            json.dump(result_json, f, indent=2)
        print(f"Saved result to {result_file}")
        return

    # Full run mode (all methods and budgets)
    print("\nRunning full experiment (all methods and budgets)...")

    # Plot dendrogram
    plot_task_dendrogram(gradient_matrix, task_names, output_dir / 'task_dendrogram.png')

    # Run selection algorithms for different budgets
    budgets = BUDGETS + [8]  # Include budget 8 in full mode
    all_results = []

    print("\n" + "=" * 60)
    print("TASK SELECTION EXPERIMENTS")
    print("=" * 60)

    for budget in budgets:
        print(f"\n--- Budget: {budget} tasks ---")

        # 1. Greedy selection
        greedy_tasks, _ = greedy_task_selection(gradient_matrix, task_names, budget)
        greedy_coverage = compute_coverage(greedy_tasks, gradient_matrix, task_names)
        print(f"Greedy:     {greedy_tasks}")
        print(f"            Coverage: {greedy_coverage:.4f}")
        all_results.append({
            'budget': budget,
            'method': 'greedy',
            'selected_tasks': greedy_tasks,
            'coverage': greedy_coverage,
        })

        # 2. Clustering-based selection
        cluster_tasks = clustering_task_selection(gradient_matrix, task_names, budget)
        cluster_coverage = compute_coverage(cluster_tasks, gradient_matrix, task_names)
        print(f"Clustering: {cluster_tasks}")
        print(f"            Coverage: {cluster_coverage:.4f}")
        all_results.append({
            'budget': budget,
            'method': 'clustering',
            'selected_tasks': cluster_tasks,
            'coverage': cluster_coverage,
        })

        # 3. Max-diversity selection
        diversity_tasks = max_diversity_selection(gradient_matrix, task_names, budget)
        diversity_coverage = compute_coverage(diversity_tasks, gradient_matrix, task_names)
        print(f"Diversity:  {diversity_tasks}")
        print(f"            Coverage: {diversity_coverage:.4f}")
        all_results.append({
            'budget': budget,
            'method': 'diversity',
            'selected_tasks': diversity_tasks,
            'coverage': diversity_coverage,
        })

        # 4. Random selection (multiple draws)
        random_selections = random_task_selection(task_names, budget, args.n_random_draws, args.seed)
        random_coverages = [
            compute_coverage(sel, gradient_matrix, task_names)
            for sel in random_selections
        ]
        random_mean = np.mean(random_coverages)
        random_std = np.std(random_coverages)
        print(f"Random:     Coverage: {random_mean:.4f} +/- {random_std:.4f}")

        for i, (sel, cov) in enumerate(zip(random_selections, random_coverages)):
            all_results.append({
                'budget': budget,
                'method': 'random',
                'draw': i,
                'selected_tasks': sel,
                'coverage': cov,
            })

    # Create summary DataFrame
    df_results = pd.DataFrame(all_results)

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY: COVERAGE BY METHOD AND BUDGET")
    print("=" * 60)

    summary_data = []
    for budget in budgets:
        budget_data = df_results[df_results['budget'] == budget]

        greedy_cov = budget_data[budget_data['method'] == 'greedy']['coverage'].values[0]
        cluster_cov = budget_data[budget_data['method'] == 'clustering']['coverage'].values[0]
        diversity_cov = budget_data[budget_data['method'] == 'diversity']['coverage'].values[0]
        random_mean = budget_data[budget_data['method'] == 'random']['coverage'].mean()
        random_std = budget_data[budget_data['method'] == 'random']['coverage'].std()

        improvement = (greedy_cov - random_mean) / random_mean * 100

        summary_data.append({
            'budget': budget,
            'greedy': greedy_cov,
            'clustering': cluster_cov,
            'diversity': diversity_cov,
            'random_mean': random_mean,
            'random_std': random_std,
            'improvement_vs_random': improvement,
        })

        print(f"Budget {budget}:")
        print(f"  Greedy:     {greedy_cov:.4f}")
        print(f"  Clustering: {cluster_cov:.4f}")
        print(f"  Diversity:  {diversity_cov:.4f}")
        print(f"  Random:     {random_mean:.4f} +/- {random_std:.4f}")
        print(f"  Greedy vs Random: +{improvement:.1f}%")

    df_summary = pd.DataFrame(summary_data)

    # Plot coverage curves
    plot_coverage_curves(df_results, output_dir / 'coverage_curves.png')

    # Success criteria check
    print("\n" + "=" * 60)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 60)

    criteria_met = True

    # Criterion 1: Greedy outperforms random by >20% for all budgets
    for _, row in df_summary.iterrows():
        if row['improvement_vs_random'] < 20:
            print(f"[X] Budget {row['budget']}: Improvement {row['improvement_vs_random']:.1f}% < 20%")
            criteria_met = False
        else:
            print(f"[+] Budget {row['budget']}: Improvement {row['improvement_vs_random']:.1f}% >= 20%")

    # Criterion 2: 5-6 tasks achieve >75% coverage
    cov_at_5 = df_summary[df_summary['budget'] == 5]['greedy'].values[0]
    cov_at_6 = df_summary[df_summary['budget'] == 6]['greedy'].values[0]

    if cov_at_5 >= 0.75 or cov_at_6 >= 0.75:
        print(f"[+] Coverage at budget 5-6: {cov_at_5:.2f}, {cov_at_6:.2f} (>= 0.75)")
    else:
        print(f"[X] Coverage at budget 5-6: {cov_at_5:.2f}, {cov_at_6:.2f} (< 0.75)")
        criteria_met = False

    if criteria_met:
        print("\n[+] SUCCESS: All criteria met!")
    else:
        print("\n[!] Some criteria not met - check results")

    # Save results
    df_results.to_csv(output_dir / 'task_selection_full_results.csv', index=False)
    df_summary.to_csv(output_dir / 'task_selection_summary.csv', index=False)

    # Save best selections for each budget
    best_selections = {}
    for budget in budgets:
        greedy_tasks, _ = greedy_task_selection(gradient_matrix, task_names, budget)
        best_selections[str(budget)] = greedy_tasks

    with open(output_dir / 'greedy_selections.json', 'w') as f:
        json.dump(best_selections, f, indent=2)

    print(f"\nResults saved to {output_dir}/")
    print("  - task_selection_full_results.csv (all results)")
    print("  - task_selection_summary.csv (summary statistics)")
    print("  - greedy_selections.json (best selections per budget)")
    print("  - coverage_curves.png (visualization)")
    print("  - task_dendrogram.png (clustering visualization)")


if __name__ == '__main__':
    main()
