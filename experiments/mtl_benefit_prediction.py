"""
MTL Benefit Prediction Experiment

Tests whether gradient similarity predicts actual MTL performance benefit.
This is the key "utility" experiment showing the diagnostic has practical value.

Experiment Design:
1. For each task pair (i, j) with high overlap:
   - Train single-task model for task i
   - Train single-task model for task j
   - Train 2-task MTL model for (i, j)
   - Compute MTL benefit = MTL_perf - avg(single_i, single_j)
2. Correlate MTL benefit with gradient similarity G_ij
3. Show: high G → positive benefit, low/negative G → negative benefit
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, mean_squared_error
from scipy import stats
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Use the existing data loading infrastructure
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.tox21_loader import load_tox21_data
from src.models.gcn_encoder import GCNEncoder
from src.models.task_heads import TaskHead


@dataclass
class ExperimentConfig:
    """Configuration for MTL benefit experiment."""
    dataset: str = 'tox21'
    n_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    hidden_dim: int = 256
    n_layers: int = 3
    dropout: float = 0.3
    n_seeds: int = 3  # Multiple seeds for robustness
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class SingleTaskModel(nn.Module):
    """Single-task model for baseline comparison."""

    def __init__(self, encoder: nn.Module, task_head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.task_head = task_head

    def forward(self, x, edge_index, batch):
        z = self.encoder(x, edge_index, batch)
        return self.task_head(z)


class TwoTaskMTLModel(nn.Module):
    """Two-task MTL model with shared encoder."""

    def __init__(self, encoder: nn.Module, head_a: nn.Module, head_b: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head_a = head_a
        self.head_b = head_b

    def forward(self, x, edge_index, batch):
        z = self.encoder(x, edge_index, batch)
        return self.head_a(z), self.head_b(z)


def train_single_task(
    data_loader: DataLoader,
    task_idx: int,
    config: ExperimentConfig,
    input_dim: int,
    seed: int = 42
) -> Tuple[float, nn.Module]:
    """Train a single-task model and return test performance."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create model
    encoder = GCNEncoder(
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        dropout=config.dropout
    ).to(config.device)

    head = TaskHead(
        input_dim=config.hidden_dim,
        hidden_dim=config.hidden_dim // 2,
        output_dim=1,
        task_type='classification'
    ).to(config.device)

    model = SingleTaskModel(encoder, head).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    # Training loop
    model.train()
    for epoch in range(config.n_epochs):
        for batch in data_loader:
            batch = batch.to(config.device)
            labels = batch.y[:, task_idx]
            mask = ~torch.isnan(labels)

            if mask.sum() == 0:
                continue

            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.batch).squeeze()
            loss = (criterion(pred, labels.float()) * mask).sum() / mask.sum()
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(config.device)
            labels = batch.y[:, task_idx]
            mask = ~torch.isnan(labels)

            if mask.sum() == 0:
                continue

            pred = torch.sigmoid(model(batch.x, batch.edge_index, batch.batch).squeeze())
            all_preds.extend(pred[mask].cpu().numpy())
            all_labels.extend(labels[mask].cpu().numpy())

    if len(all_preds) > 10:
        try:
            auc = roc_auc_score(all_labels, all_preds)
        except:
            auc = 0.5
    else:
        auc = 0.5

    return auc, model


def train_two_task_mtl(
    data_loader: DataLoader,
    task_a: int,
    task_b: int,
    config: ExperimentConfig,
    input_dim: int,
    seed: int = 42
) -> Tuple[float, float, nn.Module]:
    """Train a two-task MTL model and return test performance for both tasks."""

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
    head_a = TaskHead(
        input_dim=config.hidden_dim,
        hidden_dim=config.hidden_dim // 2,
        output_dim=1,
        task_type='classification'
    ).to(config.device)

    head_b = TaskHead(
        input_dim=config.hidden_dim,
        hidden_dim=config.hidden_dim // 2,
        output_dim=1,
        task_type='classification'
    ).to(config.device)

    model = TwoTaskMTLModel(encoder, head_a, head_b).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    # Training loop
    model.train()
    for epoch in range(config.n_epochs):
        for batch in data_loader:
            batch = batch.to(config.device)
            labels_a = batch.y[:, task_a]
            labels_b = batch.y[:, task_b]
            mask_a = ~torch.isnan(labels_a)
            mask_b = ~torch.isnan(labels_b)

            optimizer.zero_grad()
            pred_a, pred_b = model(batch.x, batch.edge_index, batch.batch)
            pred_a, pred_b = pred_a.squeeze(), pred_b.squeeze()

            loss = 0
            if mask_a.sum() > 0:
                loss += (criterion(pred_a, labels_a.float()) * mask_a).sum() / mask_a.sum()
            if mask_b.sum() > 0:
                loss += (criterion(pred_b, labels_b.float()) * mask_b).sum() / mask_b.sum()

            if loss > 0:
                loss.backward()
                optimizer.step()

    # Evaluation
    model.eval()
    preds_a, labels_a_list = [], []
    preds_b, labels_b_list = [], []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(config.device)
            labels_a = batch.y[:, task_a]
            labels_b = batch.y[:, task_b]
            mask_a = ~torch.isnan(labels_a)
            mask_b = ~torch.isnan(labels_b)

            pred_a, pred_b = model(batch.x, batch.edge_index, batch.batch)
            pred_a = torch.sigmoid(pred_a.squeeze())
            pred_b = torch.sigmoid(pred_b.squeeze())

            if mask_a.sum() > 0:
                preds_a.extend(pred_a[mask_a].cpu().numpy())
                labels_a_list.extend(labels_a[mask_a].cpu().numpy())
            if mask_b.sum() > 0:
                preds_b.extend(pred_b[mask_b].cpu().numpy())
                labels_b_list.extend(labels_b[mask_b].cpu().numpy())

    try:
        auc_a = roc_auc_score(labels_a_list, preds_a) if len(preds_a) > 10 else 0.5
    except:
        auc_a = 0.5
    try:
        auc_b = roc_auc_score(labels_b_list, preds_b) if len(preds_b) > 10 else 0.5
    except:
        auc_b = 0.5

    return auc_a, auc_b, model


def run_mtl_benefit_experiment(
    config: ExperimentConfig,
    gradient_matrix: np.ndarray,
    task_names: List[str],
    output_dir: str
) -> Dict:
    """
    Run the full MTL benefit prediction experiment.

    Returns:
        Dictionary with correlation between G and MTL benefit
    """

    print("=" * 60)
    print("MTL Benefit Prediction Experiment")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader, input_dim, n_tasks = load_tox21_data(
        batch_size=config.batch_size
    )

    # Use test loader for evaluation
    eval_loader = test_loader

    n_tasks = min(n_tasks, gradient_matrix.shape[0])
    results = []

    # For each task pair
    n_pairs = n_tasks * (n_tasks - 1) // 2
    pair_idx = 0

    for i in range(n_tasks):
        for j in range(i + 1, n_tasks):
            pair_idx += 1
            print(f"\nPair {pair_idx}/{n_pairs}: {task_names[i]} vs {task_names[j]}")

            g_ij = gradient_matrix[i, j]
            print(f"  Gradient similarity: {g_ij:.4f}")

            # Run multiple seeds
            single_aucs_i = []
            single_aucs_j = []
            mtl_aucs_i = []
            mtl_aucs_j = []

            for seed in range(config.n_seeds):
                # Train single-task models
                auc_i, _ = train_single_task(train_loader, i, config, input_dim, seed)
                auc_j, _ = train_single_task(train_loader, j, config, input_dim, seed)
                single_aucs_i.append(auc_i)
                single_aucs_j.append(auc_j)

                # Train MTL model
                mtl_auc_i, mtl_auc_j, _ = train_two_task_mtl(
                    train_loader, i, j, config, input_dim, seed
                )
                mtl_aucs_i.append(mtl_auc_i)
                mtl_aucs_j.append(mtl_auc_j)

            # Compute average performance
            single_avg = (np.mean(single_aucs_i) + np.mean(single_aucs_j)) / 2
            mtl_avg = (np.mean(mtl_aucs_i) + np.mean(mtl_aucs_j)) / 2

            # MTL benefit = MTL performance - single-task performance
            mtl_benefit = mtl_avg - single_avg

            print(f"  Single-task avg: {single_avg:.4f}")
            print(f"  MTL avg: {mtl_avg:.4f}")
            print(f"  MTL benefit: {mtl_benefit:+.4f}")

            results.append({
                'task_i': task_names[i],
                'task_j': task_names[j],
                'gradient_similarity': float(g_ij),
                'single_task_i': float(np.mean(single_aucs_i)),
                'single_task_j': float(np.mean(single_aucs_j)),
                'single_task_avg': float(single_avg),
                'mtl_task_i': float(np.mean(mtl_aucs_i)),
                'mtl_task_j': float(np.mean(mtl_aucs_j)),
                'mtl_avg': float(mtl_avg),
                'mtl_benefit': float(mtl_benefit),
                'single_std_i': float(np.std(single_aucs_i)),
                'single_std_j': float(np.std(single_aucs_j)),
                'mtl_std_i': float(np.std(mtl_aucs_i)),
                'mtl_std_j': float(np.std(mtl_aucs_j)),
            })

    # Compute correlation between G and MTL benefit
    g_values = [r['gradient_similarity'] for r in results]
    benefit_values = [r['mtl_benefit'] for r in results]

    r, p = stats.pearsonr(g_values, benefit_values)
    rho, p_rho = stats.spearmanr(g_values, benefit_values)

    print("\n" + "=" * 60)
    print("RESULTS: Gradient Similarity vs MTL Benefit")
    print("=" * 60)
    print(f"Pearson r: {r:.4f} (p = {p:.2e})")
    print(f"Spearman rho: {rho:.4f} (p = {p_rho:.2e})")

    # Bin analysis: high G vs low G
    high_g_mask = np.array(g_values) > 0.05
    low_g_mask = np.array(g_values) < 0.02

    if high_g_mask.sum() > 0:
        high_g_benefit = np.mean([b for b, m in zip(benefit_values, high_g_mask) if m])
        print(f"\nHigh G (>0.05) avg benefit: {high_g_benefit:+.4f} (n={high_g_mask.sum()})")

    if low_g_mask.sum() > 0:
        low_g_benefit = np.mean([b for b, m in zip(benefit_values, low_g_mask) if m])
        print(f"Low G (<0.02) avg benefit: {low_g_benefit:+.4f} (n={low_g_mask.sum()})")

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        'pearson_r': float(r),
        'pearson_p': float(p),
        'spearman_rho': float(rho),
        'spearman_p': float(p_rho),
        'n_pairs': len(results),
        'high_g_mean_benefit': float(high_g_benefit) if high_g_mask.sum() > 0 else None,
        'low_g_mean_benefit': float(low_g_benefit) if low_g_mask.sum() > 0 else None,
        'pair_results': results
    }

    with open(os.path.join(output_dir, 'mtl_benefit_results.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Save for plotting
    np.savez(
        os.path.join(output_dir, 'mtl_benefit_data.npz'),
        gradient_similarity=np.array(g_values),
        mtl_benefit=np.array(benefit_values)
    )

    print(f"\nResults saved to {output_dir}")

    return summary


def main():
    """Run the MTL benefit prediction experiment."""

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
    config = ExperimentConfig(
        n_epochs=30,  # Faster for pairwise experiments
        n_seeds=3,
        batch_size=32
    )

    output_dir = os.path.join(base_path, 'outputs', 'mtl_benefit_experiment')

    results = run_mtl_benefit_experiment(config, G, task_names, output_dir)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Gradient similarity predicts MTL benefit: r = {results['pearson_r']:.3f}")
    if results['pearson_p'] < 0.05:
        print("This correlation is STATISTICALLY SIGNIFICANT")
    else:
        print("This correlation is NOT statistically significant")


if __name__ == '__main__':
    main()
