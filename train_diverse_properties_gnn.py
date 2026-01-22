#!/usr/bin/env python3
"""
Train GNN on Diverse Properties Dataset.

Trains a multi-task GNN on the combined ADME + Toxicity dataset to compute
gradient conflicts across truly diverse property types.

Key difference from Tox21:
- Mix of ADME (Lipophilicity, Solubility, Permeability) and Toxicity (hERG, DILI, AMES)
- Cross-category analysis: Do ADME-Toxicity gradients show meaningful relationships?
- Validation of gradient method on diverse properties (not just panel assays)

Expected findings:
- Cross-category trade-offs (e.g., lipophilicity vs hERG liability)
- Within-category synergies (e.g., lipophilicity-permeability)

Usage:
    python train_diverse_properties_gnn.py
    python train_diverse_properties_gnn.py --epochs 100 --batch-size 32
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# PyTorch Geometric imports
try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from torch_geometric.nn import GCNConv, global_mean_pool
except ImportError:
    print("Error: PyTorch Geometric not installed")
    print("Install with: pip install torch-geometric")
    sys.exit(1)

# RDKit
try:
    from rdkit import Chem
except ImportError:
    print("Error: RDKit not installed")
    sys.exit(1)

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


# =============================================================================
# Molecular Graph Construction
# =============================================================================

ATOM_FEATURES = {
    'atomic_num': list(range(1, 119)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-2, -1, 0, 1, 2],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
    'is_aromatic': [False, True],
}


def one_hot(value, choices):
    encoding = [0] * (len(choices) + 1)
    try:
        idx = choices.index(value)
        encoding[idx] = 1
    except ValueError:
        encoding[-1] = 1
    return encoding


def atom_features(atom):
    features = []
    features.extend(one_hot(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num']))
    features.extend(one_hot(atom.GetDegree(), ATOM_FEATURES['degree']))
    features.extend(one_hot(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']))
    features.extend(one_hot(atom.GetHybridization(), ATOM_FEATURES['hybridization']))
    features.extend(one_hot(atom.GetIsAromatic(), ATOM_FEATURES['is_aromatic']))
    return features


def smiles_to_graph(smiles: str) -> Optional[Data]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_feats = [atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_feats, dtype=torch.float)

    edge_indices = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices.extend([[i, j], [j, i]])

    if len(edge_indices) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)


# =============================================================================
# Dataset
# =============================================================================

class DiversePropertiesDataset:
    """Dataset for diverse ADME + Toxicity properties."""

    def __init__(
        self,
        data_path: str,
        tasks: List[str],
        split: str = 'train',
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42
    ):
        self.tasks = tasks
        self.split = split

        df = pd.read_csv(data_path)

        # Filter to rows with at least some task values
        task_cols = [t for t in tasks if t in df.columns]
        df = df.dropna(subset=['smiles'])
        df = df[df[task_cols].notna().any(axis=1)]

        # Random split (scaffold split would be better for production)
        np.random.seed(seed)
        n = len(df)
        indices = np.random.permutation(n)

        train_end = int(split_ratio[0] * n)
        val_end = train_end + int(split_ratio[1] * n)

        if split == 'train':
            df = df.iloc[indices[:train_end]]
        elif split == 'val':
            df = df.iloc[indices[train_end:val_end]]
        else:
            df = df.iloc[indices[val_end:]]

        self.smiles = df['smiles'].tolist()

        # Get labels and masks
        self.labels = {}
        self.masks = {}

        for task in tasks:
            if task in df.columns:
                values = df[task].values.astype(np.float32)
                mask = ~np.isnan(values)
                values = np.nan_to_num(values, nan=0.0)
                self.labels[task] = values
                self.masks[task] = mask
            else:
                self.labels[task] = np.zeros(len(df), dtype=np.float32)
                self.masks[task] = np.zeros(len(df), dtype=bool)

        # Convert to graphs
        self.graphs = []
        self.valid_indices = []
        for i, smi in enumerate(self.smiles):
            graph = smiles_to_graph(smi)
            if graph is not None:
                self.graphs.append(graph)
                self.valid_indices.append(i)

        # Filter labels/masks
        for task in tasks:
            self.labels[task] = self.labels[task][self.valid_indices]
            self.masks[task] = self.masks[task][self.valid_indices]

        print(f"  {split}: {len(self.graphs)} molecules")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx].clone()
        labels = torch.tensor([self.labels[t][idx] for t in self.tasks], dtype=torch.float)
        masks = torch.tensor([self.masks[t][idx] for t in self.tasks], dtype=torch.bool)
        graph.y = labels
        graph.mask = masks
        return graph


# =============================================================================
# Model
# =============================================================================

class GNNEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        return global_mean_pool(x, batch)


class TaskHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, is_classification: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.2)
        self.is_classification = is_classification

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(-1)


class MultiTaskGNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, tasks: List[str], task_types: Dict[str, str]):
        super().__init__()
        self.encoder = GNNEncoder(input_dim, hidden_dim)
        self.task_heads = nn.ModuleDict({
            task: TaskHead(hidden_dim, is_classification=(task_types.get(task, 'regression') == 'classification'))
            for task in tasks
        })
        self.tasks = tasks
        self.task_types = task_types

    def forward(self, batch):
        h = self.encoder(batch.x, batch.edge_index, batch.batch)
        return {task: head(h) for task, head in self.task_heads.items()}

    def get_encoder_params(self):
        return list(self.encoder.parameters())


# =============================================================================
# Gradient Logger
# =============================================================================

class GradientConflictLogger:
    def __init__(self, tasks: List[str]):
        self.tasks = tasks
        self.K = len(tasks)
        self.gradient_history = []

    def compute_conflicts(
        self,
        task_losses: Dict[str, torch.Tensor],
        encoder_params: List[torch.nn.Parameter]
    ) -> np.ndarray:
        task_gradients = {}

        for i, (task, loss) in enumerate(task_losses.items()):
            if loss is None or loss.item() == 0:
                continue

            retain = (i < len(task_losses) - 1)
            grads = torch.autograd.grad(
                outputs=loss,
                inputs=encoder_params,
                retain_graph=retain,
                allow_unused=True
            )

            grad_vec = torch.cat([
                g.flatten() if g is not None else torch.zeros_like(p.flatten())
                for g, p in zip(grads, encoder_params)
            ])
            task_gradients[task] = grad_vec

        G = np.eye(self.K)

        for i, task_i in enumerate(self.tasks):
            for j, task_j in enumerate(self.tasks):
                if i >= j:
                    continue
                if task_i not in task_gradients or task_j not in task_gradients:
                    G[i, j] = G[j, i] = np.nan
                    continue

                g_i, g_j = task_gradients[task_i], task_gradients[task_j]
                norm_i, norm_j = torch.norm(g_i), torch.norm(g_j)

                if norm_i > 1e-8 and norm_j > 1e-8:
                    cos_sim = torch.dot(g_i, g_j) / (norm_i * norm_j)
                    G[i, j] = G[j, i] = cos_sim.item()
                else:
                    G[i, j] = G[j, i] = np.nan

        self.gradient_history.append(G)
        return G

    def get_average_matrix(self) -> np.ndarray:
        if not self.gradient_history:
            return np.eye(self.K)
        return np.nanmean(np.array(self.gradient_history), axis=0)

    def save(self, path: str):
        np.savez(
            path,
            average_matrix=self.get_average_matrix(),
            history=np.array(self.gradient_history),
            tasks=self.tasks
        )


# =============================================================================
# Training
# =============================================================================

def train_epoch(
    model: MultiTaskGNN,
    loader: PyGDataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_logger: GradientConflictLogger,
    log_interval: int = 10
) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0
    task_losses_sum = {task: 0.0 for task in model.tasks}
    task_counts = {task: 0 for task in model.tasks}

    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        optimizer.zero_grad()

        outputs = model(batch)

        task_losses = {}
        for i, task in enumerate(model.tasks):
            mask = batch.mask[:, i]
            if mask.sum() == 0:
                continue

            pred = outputs[task][mask]
            target = batch.y[:, i][mask]

            # Use appropriate loss
            if model.task_types.get(task, 'regression') == 'classification':
                loss = F.binary_cross_entropy_with_logits(pred, target)
            else:
                loss = F.mse_loss(pred, target)

            task_losses[task] = loss
            task_losses_sum[task] += loss.item() * mask.sum().item()
            task_counts[task] += mask.sum().item()

        if not task_losses:
            continue

        # Log gradients
        if batch_idx % log_interval == 0 and len(task_losses) > 1:
            encoder_params = model.get_encoder_params()
            gradient_logger.compute_conflicts(task_losses, encoder_params)

        combined_loss = sum(task_losses.values()) / len(task_losses)
        combined_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += combined_loss.item()

    avg_loss = total_loss / len(loader)
    avg_task_losses = {
        task: task_losses_sum[task] / max(task_counts[task], 1)
        for task in model.tasks
    }

    return avg_loss, avg_task_losses


@torch.no_grad()
def evaluate(model: MultiTaskGNN, loader: PyGDataLoader, device: torch.device):
    model.eval()

    all_preds = {task: [] for task in model.tasks}
    all_targets = {task: [] for task in model.tasks}

    for batch in loader:
        batch = batch.to(device)
        outputs = model(batch)

        for i, task in enumerate(model.tasks):
            mask = batch.mask[:, i]
            if mask.sum() == 0:
                continue

            pred = outputs[task][mask].cpu().numpy()
            target = batch.y[:, i][mask].cpu().numpy()

            all_preds[task].extend(pred.flatten().tolist())
            all_targets[task].extend(target.flatten().tolist())

    metrics = {}
    for task in model.tasks:
        if len(all_preds[task]) == 0:
            metrics[task] = {'metric': float('nan'), 'metric_name': 'N/A'}
            continue

        preds = np.array(all_preds[task])
        targets = np.array(all_targets[task])

        if model.task_types.get(task, 'regression') == 'classification':
            # AUC for classification
            from sklearn.metrics import roc_auc_score
            try:
                auc = roc_auc_score(targets, preds)
                metrics[task] = {'metric': auc, 'metric_name': 'AUC'}
            except:
                metrics[task] = {'metric': float('nan'), 'metric_name': 'AUC'}
        else:
            # RMSE and R² for regression
            rmse = np.sqrt(np.mean((preds - targets) ** 2))
            ss_res = np.sum((targets - preds) ** 2)
            ss_tot = np.sum((targets - np.mean(targets)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            metrics[task] = {'metric': rmse, 'r2': r2, 'metric_name': 'RMSE'}

    return metrics


def compute_empirical_correlation(
    merged_df: pd.DataFrame,
    task_cols: List[str]
) -> np.ndarray:
    """Compute empirical label correlations."""
    n = len(task_cols)
    corr = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):
            mask = merged_df[[task_cols[i], task_cols[j]]].notna().all(axis=1)
            if mask.sum() < 10:
                corr[i, j] = corr[j, i] = np.nan
                continue

            x = merged_df.loc[mask, task_cols[i]]
            y = merged_df.loc[mask, task_cols[j]]
            r, _ = stats.pearsonr(x, y)
            corr[i, j] = corr[j, i] = r

    return corr


def main():
    parser = argparse.ArgumentParser(description='Train GNN on diverse properties')
    parser.add_argument('--data-path', type=str,
                       default='outputs/diverse_data/diverse_properties.csv')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='outputs/diverse_results')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check data exists
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Data not found at {data_path}")
        print("Please run: python scripts/curate_diverse_properties_v2.py")
        sys.exit(1)

    # Load data to get task columns
    df = pd.read_csv(data_path)
    task_cols = [c for c in df.columns if c != 'smiles']

    # Determine task types
    task_types = {}
    for task in task_cols:
        if task.startswith('Tox_'):
            # Toxicity tasks are usually classification
            unique_vals = df[task].dropna().unique()
            if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                task_types[task] = 'classification'
            else:
                task_types[task] = 'regression'
        else:
            task_types[task] = 'regression'

    print("\n" + "=" * 60)
    print("Training GNN on Diverse Properties Dataset")
    print("=" * 60)
    print(f"\nTasks ({len(task_cols)}):")
    adme_tasks = [t for t in task_cols if t.startswith('ADME_')]
    tox_tasks = [t for t in task_cols if t.startswith('Tox_')]
    print(f"  ADME tasks: {adme_tasks}")
    print(f"  Tox tasks: {tox_tasks}")

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = DiversePropertiesDataset(str(data_path), task_cols, 'train', seed=args.seed)
    val_dataset = DiversePropertiesDataset(str(data_path), task_cols, 'val', seed=args.seed)
    test_dataset = DiversePropertiesDataset(str(data_path), task_cols, 'test', seed=args.seed)

    train_loader = PyGDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = PyGDataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = PyGDataLoader(test_dataset, batch_size=args.batch_size)

    # Get input dim
    sample_graph = train_dataset[0]
    input_dim = sample_graph.x.shape[1]
    print(f"\nInput dimension: {input_dim}")

    # Model
    model = MultiTaskGNN(input_dim, args.hidden_dim, task_cols, task_types).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    gradient_logger = GradientConflictLogger(task_cols)

    # Training
    print("\nTraining...")
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0

    for epoch in range(args.epochs):
        start_time = time.time()

        train_loss, _ = train_epoch(
            model, train_loader, optimizer, device,
            gradient_logger, log_interval=10
        )

        val_metrics = evaluate(model, val_loader, device)

        # Compute average validation loss proxy
        val_losses = []
        for task, m in val_metrics.items():
            if not np.isnan(m['metric']):
                val_losses.append(m['metric'])
        avg_val = np.mean(val_losses) if val_losses else float('inf')

        scheduler.step(avg_val)
        elapsed = time.time() - start_time

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"\nEpoch {epoch+1}/{args.epochs} ({elapsed:.1f}s)")
            print(f"  Train loss: {train_loss:.4f}")
            for task, m in val_metrics.items():
                print(f"    {task}: {m['metric_name']}={m['metric']:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(torch.load(output_dir / 'best_model.pt'))

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    test_metrics = evaluate(model, test_loader, device)
    print("\nTest metrics:")
    for task, m in test_metrics.items():
        print(f"  {task}: {m['metric_name']}={m['metric']:.4f}")

    # Save gradient matrix
    gradient_path = output_dir / 'gradient_conflict_matrices.npz'
    gradient_logger.save(str(gradient_path))

    G = gradient_logger.get_average_matrix()
    print("\n" + "=" * 60)
    print("GRADIENT CONFLICT MATRIX")
    print("=" * 60)
    print("\nTasks:", task_cols)
    np.set_printoptions(precision=3, suppress=True)
    print(G)

    # Compute empirical correlation
    print("\n" + "=" * 60)
    print("EMPIRICAL LABEL CORRELATIONS")
    print("=" * 60)

    empirical_corr = compute_empirical_correlation(df, task_cols)
    print(empirical_corr)

    # Compare G vs empirical
    print("\n" + "=" * 60)
    print("VALIDATION: Gradient vs Empirical")
    print("=" * 60)

    G_flat = G[np.triu_indices(len(task_cols), k=1)]
    emp_flat = empirical_corr[np.triu_indices(len(task_cols), k=1)]

    # Remove NaN pairs
    valid_mask = ~(np.isnan(G_flat) | np.isnan(emp_flat))
    if valid_mask.sum() >= 3:
        r, p = stats.pearsonr(G_flat[valid_mask], emp_flat[valid_mask])
        print(f"\nPearson correlation: r={r:.4f}, p={p:.4e}")
        print(f"Number of valid pairs: {valid_mask.sum()}")

        if r > 0.5:
            print("\n✓ Strong agreement between gradient conflicts and empirical correlations")
        elif r > 0.3:
            print("\n~ Moderate agreement between gradient conflicts and empirical correlations")
        else:
            print("\n✗ Weak agreement - may indicate different mechanisms")
    else:
        print("\nInsufficient valid pairs for correlation analysis")

    # Highlight cross-category relationships
    print("\n" + "=" * 60)
    print("CROSS-CATEGORY ANALYSIS (ADME vs Tox)")
    print("=" * 60)

    cross_pairs = []
    for i, task_i in enumerate(task_cols):
        for j, task_j in enumerate(task_cols):
            if i >= j:
                continue
            is_cross = (task_i.startswith('ADME_') and task_j.startswith('Tox_')) or \
                       (task_i.startswith('Tox_') and task_j.startswith('ADME_'))
            if is_cross and not np.isnan(G[i, j]):
                cross_pairs.append({
                    'task1': task_i,
                    'task2': task_j,
                    'G': G[i, j],
                    'empirical': empirical_corr[i, j]
                })

    if cross_pairs:
        print("\nCross-category gradient relationships:")
        for pair in sorted(cross_pairs, key=lambda x: abs(x['G']), reverse=True):
            rel = "synergy" if pair['G'] > 0.1 else ("conflict" if pair['G'] < -0.1 else "neutral")
            print(f"  {pair['task1']} vs {pair['task2']}: G={pair['G']:.3f} ({rel})")
    else:
        print("\nNo cross-category pairs with valid gradient data")

    # Save results
    results = {
        'test_metrics': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in test_metrics.items()},
        'tasks': task_cols,
        'task_types': task_types,
        'gradient_matrix': G.tolist(),
        'empirical_correlation': empirical_corr.tolist(),
        'n_gradient_samples': len(gradient_logger.gradient_history),
        'cross_category_pairs': cross_pairs,
        'config': vars(args),
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}")

    return G, test_metrics


if __name__ == '__main__':
    main()
