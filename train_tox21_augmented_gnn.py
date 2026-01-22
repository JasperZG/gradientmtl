#!/usr/bin/env python3
"""
Train GNN on Tox21 Augmented Dataset (Toxicity + Physicochemical).

Trains multi-task GNN on combined toxicity endpoints and computed molecular
descriptors to analyze gradient conflicts across diverse property types.

Key analysis:
1. Cross-category conflicts: Do physicochemical features conflict with toxicity?
2. Within-category patterns: Confirm expected Tox21 relationships
3. Feature importance: Which molecular features most conflict with which toxicity?

Usage:
    python train_tox21_augmented_gnn.py
    python train_tox21_augmented_gnn.py --epochs 50 --batch-size 64
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
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from torch_geometric.nn import GCNConv, global_mean_pool
except ImportError:
    print("Error: PyTorch Geometric not installed")
    sys.exit(1)

try:
    from rdkit import Chem
except ImportError:
    print("Error: RDKit not installed")
    sys.exit(1)

from sklearn.metrics import roc_auc_score

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
        encoding[choices.index(value)] = 1
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

class Tox21AugmentedDataset:
    """Dataset for Tox21 + physicochemical properties."""

    def __init__(
        self,
        data_path: str,
        tasks: List[str],
        task_types: Dict[str, str],
        split: str = 'train',
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42
    ):
        self.tasks = tasks
        self.task_types = task_types
        self.split = split

        df = pd.read_csv(data_path)
        df = df.dropna(subset=['smiles'])

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

        # Normalize regression tasks
        self.normalizers = {}
        for task in tasks:
            if task_types.get(task) == 'regression' and task in df.columns:
                values = df[task].dropna()
                self.normalizers[task] = {'mean': values.mean(), 'std': values.std() + 1e-8}

        # Get labels and masks
        self.labels = {}
        self.masks = {}

        for task in tasks:
            if task in df.columns:
                values = df[task].values.astype(np.float32)
                mask = ~np.isnan(values)

                # Normalize regression tasks
                if task in self.normalizers:
                    norm = self.normalizers[task]
                    values = (values - norm['mean']) / norm['std']

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
            x = F.relu(bn(conv(x, edge_index)))
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
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        return self.fc2(x).squeeze(-1)


class MultiTaskGNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, tasks: List[str], task_types: Dict[str, str]):
        super().__init__()
        self.encoder = GNNEncoder(input_dim, hidden_dim)
        self.task_heads = nn.ModuleDict({task: TaskHead(hidden_dim) for task in tasks})
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
                outputs=loss, inputs=encoder_params,
                retain_graph=retain, allow_unused=True
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
        np.savez(path, average_matrix=self.get_average_matrix(),
                 history=np.array(self.gradient_history), tasks=self.tasks)


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, loader, optimizer, device, gradient_logger, log_interval=10):
    model.train()
    total_loss = 0
    n_tasks = len(model.tasks)

    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        optimizer.zero_grad()

        outputs = model(batch)
        task_losses = {}

        # Reshape mask and labels: PyG concatenates them, so we need to reshape
        batch_size = batch.num_graphs
        masks = batch.mask.view(batch_size, n_tasks)
        labels = batch.y.view(batch_size, n_tasks)

        for i, task in enumerate(model.tasks):
            mask = masks[:, i]
            if mask.sum() == 0:
                continue

            pred = outputs[task][mask]
            target = labels[:, i][mask]

            if model.task_types.get(task) == 'classification':
                loss = F.binary_cross_entropy_with_logits(pred, target)
            else:
                loss = F.mse_loss(pred, target)

            task_losses[task] = loss

        if not task_losses:
            continue

        # Log gradients
        if batch_idx % log_interval == 0 and len(task_losses) > 1:
            encoder_params = model.get_encoder_params()
            gradient_logger.compute_conflicts(task_losses, encoder_params)
            continue  # Skip training update to avoid graph issues

        combined_loss = sum(task_losses.values()) / len(task_losses)
        combined_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += combined_loss.item()

    return total_loss / max(len(loader) - len(loader) // log_interval, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    n_tasks = len(model.tasks)

    all_preds = {task: [] for task in model.tasks}
    all_targets = {task: [] for task in model.tasks}

    for batch in loader:
        batch = batch.to(device)
        outputs = model(batch)

        # Reshape mask and labels
        batch_size = batch.num_graphs
        masks = batch.mask.view(batch_size, n_tasks)
        labels = batch.y.view(batch_size, n_tasks)

        for i, task in enumerate(model.tasks):
            mask = masks[:, i]
            if mask.sum() == 0:
                continue

            pred = outputs[task][mask].cpu().numpy()
            target = labels[:, i][mask].cpu().numpy()

            all_preds[task].extend(pred.flatten().tolist())
            all_targets[task].extend(target.flatten().tolist())

    metrics = {}
    for task in model.tasks:
        if len(all_preds[task]) == 0:
            metrics[task] = {'metric': float('nan'), 'name': 'N/A'}
            continue

        preds = np.array(all_preds[task])
        targets = np.array(all_targets[task])

        if model.task_types.get(task) == 'classification':
            try:
                auc = roc_auc_score(targets, preds)
                metrics[task] = {'metric': auc, 'name': 'AUC'}
            except:
                metrics[task] = {'metric': float('nan'), 'name': 'AUC'}
        else:
            rmse = np.sqrt(np.mean((preds - targets) ** 2))
            metrics[task] = {'metric': rmse, 'name': 'RMSE'}

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train GNN on Tox21 augmented data')
    parser.add_argument('--data-path', type=str,
                       default='outputs/tox21_augmented/tox21_augmented.csv')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='outputs/tox21_augmented_results')
    parser.add_argument('--max-tox-tasks', type=int, default=6,
                       help='Max toxicity tasks to use (for speed)')
    parser.add_argument('--max-phys-tasks', type=int, default=6,
                       help='Max physicochemical tasks to use')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Data not found at {data_path}")
        print("Please run: python scripts/create_tox21_augmented.py")
        sys.exit(1)

    df = pd.read_csv(data_path)

    # Select tasks
    tox_tasks = [c for c in df.columns if c.startswith('Tox_')][:args.max_tox_tasks]
    phys_tasks = [c for c in df.columns if c.startswith('Phys_')][:args.max_phys_tasks]
    tasks = tox_tasks + phys_tasks

    task_types = {**{t: 'classification' for t in tox_tasks},
                  **{t: 'regression' for t in phys_tasks}}

    print("\n" + "=" * 60)
    print("Training GNN on Tox21 Augmented Dataset")
    print("=" * 60)
    print(f"\nToxicity tasks ({len(tox_tasks)}): {tox_tasks}")
    print(f"Physicochemical tasks ({len(phys_tasks)}): {phys_tasks}")
    print(f"Total tasks: {len(tasks)}")

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = Tox21AugmentedDataset(str(data_path), tasks, task_types, 'train', seed=args.seed)
    val_dataset = Tox21AugmentedDataset(str(data_path), tasks, task_types, 'val', seed=args.seed)
    test_dataset = Tox21AugmentedDataset(str(data_path), tasks, task_types, 'test', seed=args.seed)

    train_loader = PyGDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = PyGDataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = PyGDataLoader(test_dataset, batch_size=args.batch_size)

    # Model
    sample_graph = train_dataset[0]
    input_dim = sample_graph.x.shape[1]
    print(f"\nInput dimension: {input_dim}")

    model = MultiTaskGNN(input_dim, args.hidden_dim, tasks, task_types).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    gradient_logger = GradientConflictLogger(tasks)

    # Training
    print("\nTraining...")
    best_val_loss = float('inf')
    patience, patience_counter = 10, 0

    for epoch in range(args.epochs):
        start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device, gradient_logger)
        val_metrics = evaluate(model, val_loader, device)

        # Average validation metric
        val_losses = [m['metric'] for m in val_metrics.values() if not np.isnan(m['metric'])]
        avg_val = np.mean(val_losses) if val_losses else float('inf')

        scheduler.step(avg_val)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"\nEpoch {epoch+1}/{args.epochs} ({time.time()-start:.1f}s)")
            print(f"  Train loss: {train_loss:.4f}, Val avg: {avg_val:.4f}")

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
        print(f"  {task}: {m['name']}={m['metric']:.4f}")

    # Save gradients
    gradient_logger.save(str(output_dir / 'gradient_matrices.npz'))

    G = gradient_logger.get_average_matrix()
    print("\n" + "=" * 60)
    print("GRADIENT CONFLICT MATRIX")
    print("=" * 60)
    np.set_printoptions(precision=3, suppress=True)
    print("\nTasks:", tasks)
    print(G)

    # Analyze cross-category conflicts
    print("\n" + "=" * 60)
    print("CROSS-CATEGORY ANALYSIS (Tox vs Phys)")
    print("=" * 60)

    cross_pairs = []
    for i, t_i in enumerate(tasks):
        for j, t_j in enumerate(tasks):
            if i >= j:
                continue
            is_cross = (t_i.startswith('Tox_') and t_j.startswith('Phys_')) or \
                       (t_i.startswith('Phys_') and t_j.startswith('Tox_'))
            if is_cross and not np.isnan(G[i, j]):
                cross_pairs.append({'tox': t_i if t_i.startswith('Tox_') else t_j,
                                    'phys': t_j if t_j.startswith('Phys_') else t_i,
                                    'G': G[i, j]})

    if cross_pairs:
        print("\nTop cross-category relationships (by magnitude):")
        for p in sorted(cross_pairs, key=lambda x: abs(x['G']), reverse=True)[:10]:
            rel = "synergy" if p['G'] > 0.1 else ("conflict" if p['G'] < -0.1 else "neutral")
            print(f"  {p['tox']} vs {p['phys']}: G={p['G']:.3f} ({rel})")

    # Within-category analysis
    print("\n" + "=" * 60)
    print("WITHIN-CATEGORY ANALYSIS")
    print("=" * 60)

    # Tox-Tox
    print("\nToxicity (Tox vs Tox):")
    for i, t_i in enumerate(tox_tasks):
        for j, t_j in enumerate(tox_tasks):
            if i >= j:
                continue
            idx_i, idx_j = tasks.index(t_i), tasks.index(t_j)
            if not np.isnan(G[idx_i, idx_j]) and abs(G[idx_i, idx_j]) > 0.1:
                print(f"  {t_i} vs {t_j}: G={G[idx_i, idx_j]:.3f}")

    # Phys-Phys
    print("\nPhysicochemical (Phys vs Phys):")
    for i, t_i in enumerate(phys_tasks):
        for j, t_j in enumerate(phys_tasks):
            if i >= j:
                continue
            idx_i, idx_j = tasks.index(t_i), tasks.index(t_j)
            if not np.isnan(G[idx_i, idx_j]) and abs(G[idx_i, idx_j]) > 0.1:
                print(f"  {t_i} vs {t_j}: G={G[idx_i, idx_j]:.3f}")

    # Save results
    results = {
        'test_metrics': {k: {kk: (float(vv) if isinstance(vv, (int, float)) else str(vv)) for kk, vv in v.items()} for k, v in test_metrics.items()},
        'tasks': tasks,
        'tox_tasks': tox_tasks,
        'phys_tasks': phys_tasks,
        'gradient_matrix': G.tolist(),
        'cross_category_pairs': cross_pairs,
        'n_gradient_samples': len(gradient_logger.gradient_history),
        'config': vars(args),
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}")

    return G, test_metrics


if __name__ == '__main__':
    main()
