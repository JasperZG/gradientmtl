#!/usr/bin/env python3
"""
Train GNN on Kinase Selectivity Panel Data.

Uses the kinase selectivity dataset from ChEMBL to validate gradient conflict
analysis on data with known antagonistic relationships.

Key features:
- 21 kinases across 5 families (CDK, JAK, EGFR, Aurora, SRC)
- Mix of positive (within-family) and negative (cross-family) correlations
- Expected to find selectivity trade-offs as negative gradient correlations

Usage:
    python experiments/train_kinase_gnn.py
    python experiments/train_kinase_gnn.py --family jak
    python experiments/train_kinase_gnn.py --epochs 50 --batch-size 64
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

# RDKit for molecular graphs
try:
    from rdkit import Chem
except ImportError:
    print("Error: RDKit not installed")
    print("Install with: pip install rdkit")
    sys.exit(1)

# Add project root
project_root = Path(__file__).parent.parent
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
    """One-hot encode a value."""
    encoding = [0] * (len(choices) + 1)
    try:
        idx = choices.index(value)
        encoding[idx] = 1
    except ValueError:
        encoding[-1] = 1
    return encoding


def atom_features(atom):
    """Compute features for a single atom."""
    features = []
    features.extend(one_hot(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num']))
    features.extend(one_hot(atom.GetDegree(), ATOM_FEATURES['degree']))
    features.extend(one_hot(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']))
    features.extend(one_hot(atom.GetHybridization(), ATOM_FEATURES['hybridization']))
    features.extend(one_hot(atom.GetIsAromatic(), ATOM_FEATURES['is_aromatic']))
    return features


def smiles_to_graph(smiles: str) -> Optional[Data]:
    """Convert SMILES to PyG Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_feats = []
    for atom in mol.GetAtoms():
        atom_feats.append(atom_features(atom))

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

class KinaseSelectivityDataset(torch.utils.data.Dataset):
    """Dataset for kinase selectivity data."""

    def __init__(
        self,
        data_path: str,
        tasks: List[str] = None,
        split: str = 'train',
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42
    ):
        self.split = split

        # Load data
        df = pd.read_csv(data_path)

        # Auto-detect tasks (columns ending in _pIC50)
        if tasks is None:
            tasks = [c for c in df.columns if c.endswith('_pIC50')]

        self.tasks = tasks
        self.num_tasks = len(tasks)
        print(f"Tasks: {tasks}")

        # Filter to rows with at least one valid measurement
        task_mask = df[tasks].notna().any(axis=1)
        df = df[task_mask].reset_index(drop=True)

        # Split data
        np.random.seed(seed)
        n = len(df)
        indices = np.random.permutation(n)

        train_end = int(n * split_ratio[0])
        val_end = int(n * (split_ratio[0] + split_ratio[1]))

        if split == 'train':
            df = df.iloc[indices[:train_end]]
        elif split == 'val':
            df = df.iloc[indices[train_end:val_end]]
        else:  # test
            df = df.iloc[indices[val_end:]]

        # Convert to graphs and store labels/masks
        self.graphs = []
        self.labels_list = []
        self.masks_list = []

        print(f"Converting {len(df)} molecules to graphs...")
        for idx, row in df.iterrows():
            smiles = row['smiles']
            if pd.isna(smiles):
                continue

            graph = smiles_to_graph(smiles)
            if graph is None:
                continue

            self.graphs.append(graph)

            labels = []
            masks = []
            for task in tasks:
                value = row[task]
                if pd.isna(value):
                    labels.append(0.0)
                    masks.append(0)
                else:
                    labels.append(float(value))
                    masks.append(1)

            self.labels_list.append(labels)
            self.masks_list.append(masks)

        # Convert to tensors
        self.all_labels = torch.tensor(self.labels_list, dtype=torch.float)  # [N, num_tasks]
        self.all_masks = torch.tensor(self.masks_list, dtype=torch.bool)     # [N, num_tasks]

        print(f"  {split}: {len(self.graphs)} molecules")

        # Print coverage stats
        for i, task in enumerate(tasks):
            coverage = self.all_masks[:, i].float().mean() * 100
            print(f"    {task}: {coverage:.1f}% coverage")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx].clone()
        # Store as graph-level attributes with proper shape for PyG batching
        graph.y = self.all_labels[idx]      # [num_tasks]
        graph.mask = self.all_masks[idx]    # [num_tasks]
        return graph


def collate_kinase_batch(batch):
    """Custom collate that properly stacks graph-level y and mask."""
    from torch_geometric.data import Batch

    # Extract y and mask before batching
    ys = torch.stack([g.y for g in batch])      # [batch_size, num_tasks]
    masks = torch.stack([g.mask for g in batch]) # [batch_size, num_tasks]

    # Create batch (this will try to cat y and mask, but we'll override)
    batched = Batch.from_data_list(batch)

    # Override with properly stacked tensors
    batched.y = ys
    batched.mask = masks

    return batched


# =============================================================================
# Model
# =============================================================================

class GNNEncoder(nn.Module):
    """Graph Neural Network encoder."""

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
        x = global_mean_pool(x, batch)
        return x


class TaskHead(nn.Module):
    """Task-specific prediction head for regression."""

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
        x = self.fc2(x)
        return x.squeeze(-1)


class MultiTaskGNN(nn.Module):
    """Multi-task GNN for kinase selectivity."""

    def __init__(self, input_dim: int, hidden_dim: int, tasks: List[str]):
        super().__init__()
        self.encoder = GNNEncoder(input_dim, hidden_dim)
        self.task_heads = nn.ModuleDict({
            task.replace('-', '_'): TaskHead(hidden_dim) for task in tasks
        })
        self.tasks = tasks
        self.task_key_map = {task: task.replace('-', '_') for task in tasks}

    def forward(self, batch):
        h = self.encoder(batch.x, batch.edge_index, batch.batch)
        outputs = {task: self.task_heads[self.task_key_map[task]](h) for task in self.tasks}
        return outputs

    def get_encoder_params(self):
        return list(self.encoder.parameters())


# =============================================================================
# Gradient Logger
# =============================================================================

class GradientConflictLogger:
    """Log gradient conflicts between tasks."""

    def __init__(self, tasks: List[str]):
        self.tasks = tasks
        self.K = len(tasks)
        self.gradient_history = []

    def compute_conflicts(
        self,
        task_losses: Dict[str, torch.Tensor],
        encoder_params: List[torch.nn.Parameter],
        retain_graph: bool = True
    ) -> np.ndarray:
        """Compute pairwise gradient conflicts.

        Args:
            task_losses: Dict of task name to loss tensor
            encoder_params: List of encoder parameters
            retain_graph: Whether to retain graph (True if backward will be called after)
        """
        task_gradients = {}

        task_list = list(task_losses.items())
        for i, (task, loss) in enumerate(task_list):
            if loss is None or loss.item() == 0:
                continue

            # Always retain graph since we need it for backward() after this
            grads = torch.autograd.grad(
                outputs=loss,
                inputs=encoder_params,
                retain_graph=True,  # Always retain for subsequent operations
                allow_unused=True
            )

            grad_vec = torch.cat([
                g.flatten() if g is not None else torch.zeros_like(p.flatten())
                for g, p in zip(grads, encoder_params)
            ])
            task_gradients[task] = grad_vec.detach()  # Detach to avoid memory leak

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
        """Get time-averaged conflict matrix."""
        if not self.gradient_history:
            return np.eye(self.K)
        history = np.array(self.gradient_history)
        return np.nanmean(history, axis=0)

    def save(self, path: str):
        """Save gradient matrices."""
        avg_matrix = self.get_average_matrix()
        history = np.array(self.gradient_history)
        np.savez(
            path,
            average_matrix=avg_matrix,
            history=history,
            tasks=self.tasks
        )
        print(f"Saved gradient matrices to {path}")


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
    """Train for one epoch."""
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

            loss = F.mse_loss(pred, target)
            task_losses[task] = loss
            task_losses_sum[task] += loss.item() * mask.sum().item()
            task_counts[task] += mask.sum().item()

        if not task_losses:
            continue

        # Log gradients periodically
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
def evaluate(
    model: MultiTaskGNN,
    loader: PyGDataLoader,
    device: torch.device
) -> Tuple[float, Dict[str, dict]]:
    """Evaluate model."""
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


def validate_gradient_vs_empirical(
    gradient_matrix: np.ndarray,
    empirical_corr_path: str,
    tasks: List[str],
    output_dir: Path
) -> Dict:
    """Validate gradient conflict matrix against empirical correlations."""
    # Load empirical correlations
    empirical_df = pd.read_csv(empirical_corr_path, index_col=0)

    # Match task order
    task_cols = [t for t in tasks if t in empirical_df.columns]

    if len(task_cols) < 2:
        print("Warning: Not enough matching tasks for validation")
        return {}

    empirical_matrix = empirical_df.loc[task_cols, task_cols].values

    # Extract upper triangular (excluding diagonal)
    n = len(task_cols)
    task_indices = [tasks.index(t) for t in task_cols]

    g_values = []
    e_values = []

    for i in range(n):
        for j in range(i+1, n):
            g = gradient_matrix[task_indices[i], task_indices[j]]
            e = empirical_matrix[i, j]
            if not np.isnan(g) and not np.isnan(e):
                g_values.append(g)
                e_values.append(e)

    g_values = np.array(g_values)
    e_values = np.array(e_values)

    if len(g_values) < 3:
        print("Warning: Not enough valid pairs for correlation")
        return {}

    # Compute correlations
    pearson_r, pearson_p = stats.pearsonr(g_values, e_values)
    spearman_r, spearman_p = stats.spearmanr(g_values, e_values)

    # Count sign agreement
    sign_agreement = np.mean(np.sign(g_values) == np.sign(e_values))

    # Count negative correlations
    n_negative_g = (g_values < 0).sum()
    n_negative_e = (e_values < 0).sum()

    results = {
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'sign_agreement': float(sign_agreement),
        'n_pairs': len(g_values),
        'n_negative_gradient': int(n_negative_g),
        'n_negative_empirical': int(n_negative_e),
    }

    # Save validation results
    with open(output_dir / 'gradient_validation.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("GRADIENT vs EMPIRICAL VALIDATION")
    print("=" * 60)
    print(f"  Pearson r:  {pearson_r:.3f} (p = {pearson_p:.2e})")
    print(f"  Spearman r: {spearman_r:.3f} (p = {spearman_p:.2e})")
    print(f"  Sign agreement: {sign_agreement*100:.1f}%")
    print(f"  Pairs analyzed: {len(g_values)}")
    print(f"  Negative G: {n_negative_g} / {len(g_values)}")
    print(f"  Negative Empirical: {n_negative_e} / {len(g_values)}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Train GNN on kinase selectivity data')
    parser.add_argument('--data', type=str, default='outputs/kinase_data/kinase_all_activity_matrix.csv',
                       help='Path to kinase activity matrix')
    parser.add_argument('--family', type=str, default='all',
                       choices=['cdk', 'jak', 'egfr', 'aurora', 'src', 'all'],
                       help='Kinase family')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='Hidden dimension')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Output directory
    output_dir = project_root / 'outputs' / f'kinase_{args.family}_results'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data_path = project_root / args.data
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        print("Run: python scripts/curate_kinase_selectivity.py --family all")
        sys.exit(1)

    print("=" * 60)
    print("Loading Kinase Selectivity Dataset")
    print("=" * 60)

    train_dataset = KinaseSelectivityDataset(data_path, split='train', seed=args.seed)
    val_dataset = KinaseSelectivityDataset(data_path, tasks=train_dataset.tasks, split='val', seed=args.seed)
    test_dataset = KinaseSelectivityDataset(data_path, tasks=train_dataset.tasks, split='test', seed=args.seed)

    tasks = train_dataset.tasks
    print(f"\nTasks: {len(tasks)}")

    # Data loaders with custom collate
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_kinase_batch)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_kinase_batch)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_kinase_batch)

    # Model
    input_dim = train_dataset[0].x.shape[1]
    model = MultiTaskGNN(input_dim, args.hidden_dim, tasks).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Gradient logger
    gradient_logger = GradientConflictLogger(tasks)

    # Training
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(args.epochs):
        start = time.time()

        train_loss, train_task_losses = train_epoch(
            model, train_loader, optimizer, device, gradient_logger
        )

        val_loss, val_metrics = evaluate(model, val_loader, device)

        scheduler.step(val_loss)

        elapsed = time.time() - start

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val RMSE: {val_loss:.4f} | "
              f"Time: {elapsed:.1f}s")

    # Load best model
    model.load_state_dict(best_model_state)

    # Test evaluation
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)

    test_loss, test_metrics = evaluate(model, test_loader, device)
    print(f"\nTest RMSE: {test_loss:.4f}")

    print("\nPer-task metrics:")
    for task, m in test_metrics.items():
        if not np.isnan(m['rmse']):
            print(f"  {task}: RMSE={m['rmse']:.3f}, R²={m['r2']:.3f}")

    # Save gradient matrix
    gradient_logger.save(str(output_dir / 'gradient_matrices.npz'))

    # Get averaged gradient matrix
    avg_gradient_matrix = gradient_logger.get_average_matrix()

    # Save gradient matrix as CSV for inspection
    gradient_df = pd.DataFrame(avg_gradient_matrix, index=tasks, columns=tasks)
    gradient_df.to_csv(output_dir / 'gradient_matrix.csv')
    print(f"\nSaved gradient matrix to {output_dir / 'gradient_matrix.csv'}")

    # Validate against empirical correlations
    empirical_corr_path = project_root / 'outputs' / 'kinase_data' / f'kinase_{args.family}_empirical_correlations.csv'
    if empirical_corr_path.exists():
        validation_results = validate_gradient_vs_empirical(
            avg_gradient_matrix,
            str(empirical_corr_path),
            tasks,
            output_dir
        )

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'tasks': tasks,
        'args': vars(args),
    }, output_dir / 'model.pt')

    print(f"\nAll results saved to: {output_dir}")


if __name__ == '__main__':
    main()
