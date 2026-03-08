"""
Task Grouping Experiment

Demonstrates that gradient-based task grouping outperforms random grouping.
This is a key "utility" experiment showing the practical value of gradient analysis.

Experiment Design:
1. Given N tasks, create K groups using different strategies:
   - Random grouping (baseline)
   - Gradient-based grouping (cluster by G similarity)
   - Oracle grouping (if available)
2. Train MTL models for each grouping
3. Compare average performance across groupings
4. Show: gradient-based grouping achieves higher average performance
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.cluster import AgglomerativeClustering
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Use the existing data loading infrastructure
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.tox21_loader import load_tox21_data
from src.models.gcn_encoder import GCNEncoder
from src.models.task_heads import TaskHead


@dataclass
class GroupingConfig:
    """Configuration for task grouping experiment."""
    dataset: str = 'tox21'
    n_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    hidden_dim: int = 256
    n_layers: int = 3
    dropout: float = 0.3
    n_seeds: int = 3
    n_random_trials: int = 10  # Number of random groupings to try
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class MultiTaskGroupModel(nn.Module):
    """MTL model for a group of tasks with shared encoder."""

    def __init__(self, encoder: nn.Module, task_heads: Dict[int, nn.Module]):
        super().__init__()
        self.encoder = encoder
        self.task_heads = nn.ModuleDict({str(k): v for k, v in task_heads.items()})

    def forward(self, x, edge_index, batch):
        z = self.encoder(x, edge_index, batch)
        outputs = {int(k): head(z) for k, head in self.task_heads.items()}
        return outputs


def train_task_group(
    train_loader: DataLoader,
    task_indices: List[int],
    config: GroupingConfig,
    input_dim: int,
    seed: int = 42
) -> Dict[int, float]:
    """Train an MTL model for a group of tasks and return per-task AUC."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create shared encoder
    encoder = GCNEncoder(
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        dropout=config.dropout
    ).to(config.device)

    # Create task heads
    task_heads = {}
    for task_idx in task_indices:
        task_heads[task_idx] = TaskHead(
            input_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim // 2,
            output_dim=1,
            task_type='classification'
        ).to(config.device)

    model = MultiTaskGroupModel(encoder, task_heads).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    # Training loop
    model.train()
    for epoch in range(config.n_epochs):
        for batch in train_loader:
            batch = batch.to(config.device)
            optimizer.zero_grad()

            outputs = model(batch.x, batch.edge_index, batch.batch)

            total_loss = 0
            for task_idx in task_indices:
                labels = batch.y[:, task_idx]
                mask = ~torch.isnan(labels)

                if mask.sum() == 0:
                    continue

                pred = outputs[task_idx].squeeze()
                loss = (criterion(pred, labels.float()) * mask).sum() / mask.sum()
                total_loss += loss

            if total_loss > 0:
                total_loss.backward()
                optimizer.step()

    # Evaluation
    model.eval()
    task_preds = {idx: [] for idx in task_indices}
    task_labels = {idx: [] for idx in task_indices}

    with torch.no_grad():
        for batch in train_loader:
            batch = batch.to(config.device)
            outputs = model(batch.x, batch.edge_index, batch.batch)

            for task_idx in task_indices:
                labels = batch.y[:, task_idx]
                mask = ~torch.isnan(labels)

                if mask.sum() == 0:
                    continue

                pred = torch.sigmoid(outputs[task_idx].squeeze())
                task_preds[task_idx].extend(pred[mask].cpu().numpy())
                task_labels[task_idx].extend(labels[mask].cpu().numpy())

    # Compute AUC for each task
    task_aucs = {}
    for task_idx in task_indices:
        if len(task_preds[task_idx]) > 10:
            try:
                task_aucs[task_idx] = roc_auc_score(task_labels[task_idx], task_preds[task_idx])
            except:
                task_aucs[task_idx] = 0.5
        else:
            task_aucs[task_idx] = 0.5

    return task_aucs


def create_gradient_based_groups(
    gradient_matrix: np.ndarray,
    n_groups: int,
    method: str = 'hierarchical'
) -> List[List[int]]:
    """
    Create task groups based on gradient similarity matrix.

    Tasks with similar gradients (positive G) should be grouped together.
    """
    n_tasks = gradient_matrix.shape[0]

    # Convert similarity to distance (1 - similarity)
    # Higher similarity = lower distance = more likely to be grouped
    distance_matrix = 1 - gradient_matrix
    np.fill_diagonal(distance_matrix, 0)

    if method == 'hierarchical':
        # Use hierarchical clustering
        # Convert to condensed distance matrix
        condensed = []
        for i in range(n_tasks):
            for j in range(i + 1, n_tasks):
                condensed.append(distance_matrix[i, j])
        condensed = np.array(condensed)

        # Perform hierarchical clustering
        Z = linkage(condensed, method='average')

        # Cut tree to get n_groups clusters
        labels = fcluster(Z, n_groups, criterion='maxclust')

        # Convert to list of groups
        groups = [[] for _ in range(n_groups)]
        for task_idx, group_idx in enumerate(labels):
            groups[group_idx - 1].append(task_idx)

    elif method == 'agglomerative':
        clustering = AgglomerativeClustering(
            n_clusters=n_groups,
            metric='precomputed',
            linkage='average'
        )
        labels = clustering.fit_predict(distance_matrix)

        groups = [[] for _ in range(n_groups)]
        for task_idx, group_idx in enumerate(labels):
            groups[group_idx].append(task_idx)

    # Remove empty groups
    groups = [g for g in groups if len(g) > 0]

    return groups


def create_random_groups(n_tasks: int, n_groups: int, seed: int = 42) -> List[List[int]]:
    """Create random task groups."""
    np.random.seed(seed)

    # Shuffle task indices
    indices = np.random.permutation(n_tasks)

    # Split into roughly equal groups
    groups = []
    group_size = n_tasks // n_groups
    remainder = n_tasks % n_groups

    start = 0
    for i in range(n_groups):
        # Add one extra to first 'remainder' groups
        size = group_size + (1 if i < remainder else 0)
        if size > 0:
            groups.append(indices[start:start + size].tolist())
            start += size

    return groups


def evaluate_grouping(
    groups: List[List[int]],
    train_loader: DataLoader,
    config: GroupingConfig,
    input_dim: int,
    task_names: List[str]
) -> Dict:
    """
    Evaluate a task grouping by training MTL models for each group.

    Returns:
        Dictionary with per-task AUCs and average performance.
    """
    all_task_aucs = {}

    for group_idx, group in enumerate(groups):
        if len(group) == 0:
            continue

        # Train multiple seeds and average
        group_aucs = {task_idx: [] for task_idx in group}

        for seed in range(config.n_seeds):
            task_aucs = train_task_group(
                train_loader, group, config, input_dim, seed
            )
            for task_idx, auc in task_aucs.items():
                group_aucs[task_idx].append(auc)

        # Average across seeds
        for task_idx in group:
            all_task_aucs[task_idx] = np.mean(group_aucs[task_idx])

    # Compute overall average
    avg_auc = np.mean(list(all_task_aucs.values()))

    return {
        'task_aucs': all_task_aucs,
        'average_auc': avg_auc,
        'groups': groups
    }


def run_task_grouping_experiment(
    config: GroupingConfig,
    gradient_matrix: np.ndarray,
    task_names: List[str],
    output_dir: str,
    n_groups: int = 3
) -> Dict:
    """
    Run the full task grouping comparison experiment.

    Compares:
    1. Gradient-based grouping
    2. Multiple random groupings

    Returns:
        Dictionary with comparison results.
    """

    print("=" * 60)
    print("Task Grouping Experiment")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader, input_dim, n_tasks = load_tox21_data(
        batch_size=config.batch_size
    )

    n_tasks = min(n_tasks, gradient_matrix.shape[0])
    print(f"Number of tasks: {n_tasks}")
    print(f"Number of groups: {n_groups}")

    # 1. Gradient-based grouping
    print("\n" + "-" * 40)
    print("Evaluating GRADIENT-BASED grouping...")
    print("-" * 40)

    gradient_groups = create_gradient_based_groups(gradient_matrix, n_groups)
    print("Groups formed:")
    for i, group in enumerate(gradient_groups):
        group_names = [task_names[idx] for idx in group]
        print(f"  Group {i+1}: {group_names}")

    gradient_results = evaluate_grouping(
        gradient_groups, train_loader, config, input_dim, task_names
    )
    print(f"\nGradient-based average AUC: {gradient_results['average_auc']:.4f}")

    # 2. Random groupings (multiple trials)
    print("\n" + "-" * 40)
    print(f"Evaluating {config.n_random_trials} RANDOM groupings...")
    print("-" * 40)

    random_results_list = []
    for trial in range(config.n_random_trials):
        print(f"\n  Random trial {trial + 1}/{config.n_random_trials}")
        random_groups = create_random_groups(n_tasks, n_groups, seed=trial * 100)

        random_results = evaluate_grouping(
            random_groups, train_loader, config, input_dim, task_names
        )
        random_results_list.append(random_results)
        print(f"    Average AUC: {random_results['average_auc']:.4f}")

    # Aggregate random results
    random_aucs = [r['average_auc'] for r in random_results_list]
    random_mean = np.mean(random_aucs)
    random_std = np.std(random_aucs)
    random_best = np.max(random_aucs)
    random_worst = np.min(random_aucs)

    # 3. Statistical comparison
    improvement = gradient_results['average_auc'] - random_mean
    improvement_pct = (improvement / random_mean) * 100 if random_mean > 0 else 0

    # One-sample t-test: is gradient result significantly better than random mean?
    t_stat, p_value = stats.ttest_1samp(random_aucs, gradient_results['average_auc'])
    # We want gradient > random, so use one-sided test
    p_value_onesided = p_value / 2 if t_stat < 0 else 1 - p_value / 2

    # How many random trials did gradient-based beat?
    n_beaten = sum(1 for r in random_aucs if gradient_results['average_auc'] > r)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS: Gradient-based vs Random Grouping")
    print("=" * 60)
    print(f"\nGradient-based AUC: {gradient_results['average_auc']:.4f}")
    print(f"Random mean AUC:    {random_mean:.4f} +/- {random_std:.4f}")
    print(f"Random best AUC:    {random_best:.4f}")
    print(f"Random worst AUC:   {random_worst:.4f}")
    print(f"\nImprovement: {improvement:+.4f} ({improvement_pct:+.2f}%)")
    print(f"Gradient-based beat {n_beaten}/{config.n_random_trials} random trials")
    print(f"Statistical significance (one-sided): p = {p_value_onesided:.4f}")

    if p_value_onesided < 0.05:
        print("\nGradient-based grouping is SIGNIFICANTLY better than random!")
    elif gradient_results['average_auc'] > random_mean:
        print("\nGradient-based grouping is better but not statistically significant.")
    else:
        print("\nRandom grouping performed better (unexpected).")

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        'n_tasks': n_tasks,
        'n_groups': n_groups,
        'gradient_based': {
            'average_auc': float(gradient_results['average_auc']),
            'task_aucs': {task_names[k]: float(v) for k, v in gradient_results['task_aucs'].items()},
            'groups': [[task_names[idx] for idx in g] for g in gradient_groups]
        },
        'random': {
            'mean_auc': float(random_mean),
            'std_auc': float(random_std),
            'best_auc': float(random_best),
            'worst_auc': float(random_worst),
            'all_aucs': [float(x) for x in random_aucs],
            'n_trials': config.n_random_trials
        },
        'comparison': {
            'improvement': float(improvement),
            'improvement_pct': float(improvement_pct),
            'n_random_beaten': n_beaten,
            'p_value': float(p_value_onesided),
            'significant': p_value_onesided < 0.05
        }
    }

    with open(os.path.join(output_dir, 'task_grouping_results.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}")

    return summary


def main():
    """Run the task grouping experiment."""

    # Load pre-computed gradient matrix
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gradient_path = os.path.join(base_path, 'outputs', 'tox21_gnn_gcn', 'gradient_matrices.npz')

    if not os.path.exists(gradient_path):
        print(f"Gradient matrix not found at {gradient_path}")
        print("Please run the main training first to generate gradient matrices.")
        return

    # Load gradient matrix
    data = np.load(gradient_path)
    G = data['averaged'] if 'averaged' in data else data[list(data.keys())[0]]

    # Load task names
    names_path = os.path.join(base_path, 'outputs', 'tox21_gnn_gcn', 'task_names.json')
    if os.path.exists(names_path):
        with open(names_path) as f:
            task_names = json.load(f)
    else:
        task_names = [f'Task_{i}' for i in range(G.shape[0])]

    # Run experiment
    config = GroupingConfig(
        n_epochs=30,  # Faster for grouping experiments
        n_seeds=3,
        n_random_trials=10,
        batch_size=32
    )

    output_dir = os.path.join(base_path, 'outputs', 'task_grouping_experiment')

    # Try different numbers of groups
    for n_groups in [2, 3, 4]:
        print(f"\n\n{'#' * 60}")
        print(f"EXPERIMENT: {n_groups} Groups")
        print(f"{'#' * 60}")

        results = run_task_grouping_experiment(
            config, G, task_names,
            os.path.join(output_dir, f'{n_groups}_groups'),
            n_groups=n_groups
        )

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
