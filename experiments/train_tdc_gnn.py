#!/usr/bin/env python3
"""
Train GNN on TDC Multi-Property Dataset.

Uses compound-aligned ADME dataset from TDC to compute gradient conflicts
across diverse physicochemical and ADME properties.

Expected trade-offs:
- Lipophilicity vs Solubility: inverse relationship (G < -0.3)
- Lipophilicity vs PPB: positive (both driven by hydrophobicity, G > 0.3)
- Permeability vs Solubility: inverse (G < -0.2)

Usage:
    python train_tdc_gnn.py
    python train_tdc_gnn.py --epochs 100 --batch-size 32
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# PyTorch Geometric
try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from torch_geometric.nn import GCNConv, global_mean_pool
except ImportError:
    print("Error: PyTorch Geometric not installed")
    sys.exit(1)


def collate_tdc_batch(batch):
    """Custom collate that properly stacks graph-level y and mask as 2D tensors."""
    ys = torch.stack([g.y for g in batch])
    masks = torch.stack([g.mask for g in batch])
    batched = Batch.from_data_list(batch)
    batched.y = ys
    batched.mask = masks
    return batched

# RDKit
try:
    from rdkit import Chem
except ImportError:
    print("Error: RDKit not installed")
    sys.exit(1)

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


# =============================================================================
# Graph Construction (same as ChEMBL version)
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
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.extend([[i, j], [j, i]])

    if len(edge_indices) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)


# =============================================================================
# Dataset
# =============================================================================

class TDCMultiPropertyDataset:
    """Dataset for TDC multi-property data."""

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
        df = df.dropna(subset=['smiles'])

        # Normalize task values (z-score)
        self.task_means = {}
        self.task_stds = {}
        for task in tasks:
            if task in df.columns:
                vals = df[task].dropna()
                self.task_means[task] = vals.mean()
                self.task_stds[task] = vals.std()
                if self.task_stds[task] > 0:
                    df.loc[df[task].notna(), task] = (df[task] - self.task_means[task]) / self.task_stds[task]

        # Random split
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

        # Labels and masks
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

        for task in tasks:
            self.labels[task] = self.labels[task][self.valid_indices]
            self.masks[task] = self.masks[task][self.valid_indices]

        print(f"  {split}: {len(self.graphs)} molecules")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx].clone()
        labels = torch.tensor([self.labels[t][idx] for t in self.tasks], dtype=torch.float)
        masks = torch.tensor([bool(self.masks[t][idx]) for t in self.tasks], dtype=torch.bool)
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
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        return self.fc2(x).squeeze(-1)


class MultiTaskGNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, tasks: List[str]):
        super().__init__()
        self.encoder = GNNEncoder(input_dim, hidden_dim)
        self.task_heads = nn.ModuleDict({task: TaskHead(hidden_dim) for task in tasks})
        self.tasks = tasks

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

            # Always retain graph so we can still call backward() after
            grads = torch.autograd.grad(
                outputs=loss,
                inputs=encoder_params,
                retain_graph=True,
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
                    G[i, j] = np.nan
                    G[j, i] = np.nan
                    continue

                g_i = task_gradients[task_i]
                g_j = task_gradients[task_j]

                norm_i = torch.norm(g_i)
                norm_j = torch.norm(g_j)

                if norm_i > 1e-8 and norm_j > 1e-8:
                    cos_sim = torch.dot(g_i, g_j) / (norm_i * norm_j)
                    G[i, j] = cos_sim.item()
                    G[j, i] = cos_sim.item()
                else:
                    G[i, j] = np.nan
                    G[j, i] = np.nan

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

def train_epoch(model, loader, optimizer, device, gradient_logger, log_interval=10):
    model.train()
    total_loss = 0

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
            loss = F.mse_loss(pred, target)
            task_losses[task] = loss

        if not task_losses:
            continue

        if batch_idx % log_interval == 0 and len(task_losses) > 1:
            gradient_logger.compute_conflicts(task_losses, model.get_encoder_params())

        combined_loss = sum(task_losses.values()) / len(task_losses)
        combined_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += combined_loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device):
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

            all_preds[task].extend(outputs[task][mask].cpu().numpy().flatten().tolist())
            all_targets[task].extend(batch.y[:, i][mask].cpu().numpy().flatten().tolist())

    metrics = {}
    for task in model.tasks:
        if len(all_preds[task]) == 0:
            metrics[task] = {'rmse': float('nan'), 'r2': float('nan')}
            continue

        preds = np.array(all_preds[task])
        targets = np.array(all_targets[task])

        rmse = np.sqrt(np.mean((preds - targets) ** 2))
        ss_res = np.sum((targets - preds) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        metrics[task] = {'rmse': rmse, 'r2': r2}

    valid_rmses = [m['rmse'] for m in metrics.values() if not np.isnan(m['rmse'])]
    avg_rmse = np.mean(valid_rmses) if valid_rmses else float('nan')

    return avg_rmse, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='outputs/tdc_data/tdc_multiproperty.csv')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='outputs/tdc_results')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Data not found at {data_path}")
        print("Run: python scripts/curate_tdc_multiproperty.py")
        sys.exit(1)

    # Determine available tasks
    df = pd.read_csv(data_path)
    available_tasks = [c for c in df.columns if c != 'smiles' and df[c].notna().sum() > 100]
    tasks = available_tasks[:6]  # Max 6 tasks

    print("\n" + "=" * 60)
    print("Training GNN on TDC Multi-Property Dataset")
    print("=" * 60)
    print(f"\nTasks: {tasks}")

    # Load data
    print("\nLoading datasets...")
    train_dataset = TDCMultiPropertyDataset(str(data_path), tasks, 'train', seed=args.seed)
    val_dataset = TDCMultiPropertyDataset(str(data_path), tasks, 'val', seed=args.seed)
    test_dataset = TDCMultiPropertyDataset(str(data_path), tasks, 'test', seed=args.seed)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate_tdc_batch)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_tdc_batch)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_tdc_batch)

    input_dim = train_dataset[0].x.shape[1]
    print(f"\nInput dimension: {input_dim}")

    model = MultiTaskGNN(input_dim, args.hidden_dim, tasks).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    gradient_logger = GradientConflictLogger(tasks)

    print("\nTraining...")
    best_val_rmse = float('inf')
    patience = 20
    patience_counter = 0

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, gradient_logger)
        val_rmse, val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_rmse)

        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            print(f"  Train loss: {train_loss:.4f}, Val RMSE: {val_rmse:.4f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(output_dir / 'best_model.pt'))

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    test_rmse, test_metrics = evaluate(model, test_loader, device)
    print(f"\nTest RMSE: {test_rmse:.4f}")
    for task, m in test_metrics.items():
        print(f"  {task}: RMSE={m['rmse']:.4f}, R²={m['r2']:.3f}")

    gradient_logger.save(str(output_dir / 'gradient_matrices.npz'))

    G = gradient_logger.get_average_matrix()
    print("\n" + "=" * 60)
    print("GRADIENT CONFLICT MATRIX")
    print("=" * 60)
    print("\nTasks:", tasks)
    np.set_printoptions(precision=3, suppress=True)
    print(G)

    print("\nKey relationships:")
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            val = G[i, j]
            if np.isnan(val):
                continue
            if abs(val) > 0.15:
                relation = "synergy" if val > 0 else "conflict"
                print(f"  {tasks[i]} vs {tasks[j]}: G={val:.3f} ({relation})")

    # Check expected trade-offs
    print("\n" + "=" * 60)
    print("VALIDATION OF EXPECTED TRADE-OFFS")
    print("=" * 60)

    expected = [
        ('Lipophilicity', 'Solubility', 'negative', 'physicochemical inverse'),
        ('Lipophilicity', 'PPB', 'positive', 'both hydrophobicity-driven'),
        ('PAMPA', 'Solubility', 'negative', 'permeability-solubility trade-off'),
    ]

    for task1, task2, expected_sign, reason in expected:
        if task1 in tasks and task2 in tasks:
            i = tasks.index(task1)
            j = tasks.index(task2)
            val = G[i, j]

            if np.isnan(val):
                print(f"  {task1} vs {task2}: NO DATA")
            else:
                actual = "positive" if val > 0 else "negative"
                match = "✓" if actual == expected_sign else "✗"
                print(f"  {match} {task1} vs {task2}: G={val:.3f} (expected {expected_sign}, {reason})")

    results = {
        'test_rmse': float(test_rmse),
        'test_metrics': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in test_metrics.items()},
        'tasks': tasks,
        'gradient_matrix': G.tolist(),
        'config': vars(args),
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
