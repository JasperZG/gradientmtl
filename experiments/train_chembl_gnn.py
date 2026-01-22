#!/usr/bin/env python3
"""
Train GNN on ChEMBL Multi-Property Dataset.

Uses the compound-aligned dataset from ChEMBL curation to compute
interpretable gradient conflicts across diverse property types.

Key difference from Tox21:
- Mix of binding (hERG, BACE1, CYP3A4) and physicochemical (LogP, PSA, MW) tasks
- Regression tasks (pIC50 values, continuous properties)
- Expect to find real trade-offs like lipophilicity vs solubility

Usage:
    python train_chembl_gnn.py
    python train_chembl_gnn.py --epochs 100 --batch-size 32
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
    from rdkit.Chem import AllChem
except ImportError:
    print("Error: RDKit not installed")
    print("Install with: pip install rdkit")
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
    """One-hot encode a value."""
    encoding = [0] * (len(choices) + 1)
    try:
        idx = choices.index(value)
        encoding[idx] = 1
    except ValueError:
        encoding[-1] = 1  # Unknown
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

    # Atom features
    atom_feats = []
    for atom in mol.GetAtoms():
        atom_feats.append(atom_features(atom))

    x = torch.tensor(atom_feats, dtype=torch.float)

    # Edge indices (bonds)
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.extend([[i, j], [j, i]])

    if len(edge_indices) == 0:
        # Single atom molecule
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)


# =============================================================================
# Dataset
# =============================================================================

class ChEMBLMultiPropertyDataset:
    """Dataset for ChEMBL multi-property data."""

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

        # Load data
        df = pd.read_csv(data_path)

        # Filter to rows with at least some task values
        task_cols = [t for t in tasks if t in df.columns]
        df = df.dropna(subset=['smiles'])
        df = df[df[task_cols].notna().any(axis=1)]

        # Scaffold split
        np.random.seed(seed)
        n = len(df)
        indices = np.random.permutation(n)

        train_end = int(split_ratio[0] * n)
        val_end = train_end + int(split_ratio[1] * n)

        if split == 'train':
            df = df.iloc[indices[:train_end]]
        elif split == 'val':
            df = df.iloc[indices[train_end:val_end]]
        else:  # test
            df = df.iloc[indices[val_end:]]

        self.smiles = df['smiles'].tolist()

        # Get labels for each task
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
                # Task not in dataset
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

        # Filter labels/masks to valid indices
        for task in tasks:
            self.labels[task] = self.labels[task][self.valid_indices]
            self.masks[task] = self.masks[task][self.valid_indices]

        print(f"  {split}: {len(self.graphs)} molecules")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx].clone()

        # Add labels and masks
        labels = torch.tensor([self.labels[t][idx] for t in self.tasks], dtype=torch.float)
        masks = torch.tensor([self.masks[t][idx] for t in self.tasks], dtype=torch.bool)

        graph.y = labels
        graph.mask = masks

        return graph


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

        # Global pooling
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
    """Multi-task GNN for ChEMBL properties."""

    def __init__(self, input_dim: int, hidden_dim: int, tasks: List[str]):
        super().__init__()

        self.encoder = GNNEncoder(input_dim, hidden_dim)
        self.task_heads = nn.ModuleDict({
            task: TaskHead(hidden_dim) for task in tasks
        })
        self.tasks = tasks

    def forward(self, batch):
        h = self.encoder(batch.x, batch.edge_index, batch.batch)
        outputs = {task: head(h) for task, head in self.task_heads.items()}
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
        encoder_params: List[torch.nn.Parameter]
    ) -> np.ndarray:
        """Compute pairwise gradient conflicts."""
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

            # Concatenate gradients
            grad_vec = torch.cat([
                g.flatten() if g is not None else torch.zeros_like(p.flatten())
                for g, p in zip(grads, encoder_params)
            ])
            task_gradients[task] = grad_vec

        # Compute conflict matrix
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

                # Cosine similarity
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

        # Compute per-task losses
        task_losses = {}
        for i, task in enumerate(model.tasks):
            mask = batch.mask[:, i]
            if mask.sum() == 0:
                continue

            pred = outputs[task][mask]
            target = batch.y[:, i][mask]

            # MSE loss for regression
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

        # Combined loss
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
) -> Tuple[float, Dict[str, float], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
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

    # Compute metrics (RMSE and R² for regression)
    metrics = {}
    for task in model.tasks:
        if len(all_preds[task]) == 0:
            metrics[task] = {'rmse': float('nan'), 'r2': float('nan')}
            continue

        preds = np.array(all_preds[task])
        targets = np.array(all_targets[task])

        rmse = np.sqrt(np.mean((preds - targets) ** 2))

        # R²
        ss_res = np.sum((targets - preds) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        metrics[task] = {'rmse': rmse, 'r2': r2}

    # Average RMSE
    valid_rmses = [m['rmse'] for m in metrics.values() if not np.isnan(m['rmse'])]
    avg_rmse = np.mean(valid_rmses) if valid_rmses else float('nan')

    predictions = {
        task: (np.array(all_preds[task]), np.array(all_targets[task]))
        for task in model.tasks
    }

    return avg_rmse, metrics, predictions


def main():
    parser = argparse.ArgumentParser(description='Train GNN on ChEMBL multi-property data')
    parser.add_argument('--data-path', type=str,
                       default='outputs/chembl_data/chembl_multiproperty_filtered.csv',
                       help='Path to ChEMBL dataset')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='outputs/chembl_results')
    args = parser.parse_args()

    # Setup
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
        print("Please run: python scripts/curate_chembl_multiproperty.py")
        sys.exit(1)

    # Define tasks
    tasks = ['hERG_pIC50', 'BACE1_pIC50', 'CYP3A4_pIC50', 'alogp', 'psa', 'mw']

    print("\n" + "=" * 60)
    print("Training GNN on ChEMBL Multi-Property Dataset")
    print("=" * 60)
    print(f"\nTasks: {tasks}")
    print(f"Data: {data_path}")

    # Load data
    print("\nLoading datasets...")
    train_dataset = ChEMBLMultiPropertyDataset(str(data_path), tasks, 'train', seed=args.seed)
    val_dataset = ChEMBLMultiPropertyDataset(str(data_path), tasks, 'val', seed=args.seed)
    test_dataset = ChEMBLMultiPropertyDataset(str(data_path), tasks, 'test', seed=args.seed)

    train_loader = PyGDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = PyGDataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = PyGDataLoader(test_dataset, batch_size=args.batch_size)

    # Get input dimension from first graph
    sample_graph = train_dataset[0]
    input_dim = sample_graph.x.shape[1]
    print(f"\nInput dimension: {input_dim}")

    # Model
    model = MultiTaskGNN(input_dim, args.hidden_dim, tasks).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # Gradient logger
    gradient_logger = GradientConflictLogger(tasks)

    # Training loop
    print("\nTraining...")
    best_val_rmse = float('inf')
    patience = 20
    patience_counter = 0

    for epoch in range(args.epochs):
        start_time = time.time()

        # Train
        train_loss, train_task_losses = train_epoch(
            model, train_loader, optimizer, device,
            gradient_logger, log_interval=10
        )

        # Validate
        val_rmse, val_metrics, _ = evaluate(model, val_loader, device)

        scheduler.step(val_rmse)

        elapsed = time.time() - start_time

        # Logging
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"\nEpoch {epoch+1}/{args.epochs} ({elapsed:.1f}s)")
            print(f"  Train loss: {train_loss:.4f}")
            print(f"  Val RMSE: {val_rmse:.4f}")
            for task, metrics in val_metrics.items():
                print(f"    {task}: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.3f}")

        # Early stopping
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
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

    test_rmse, test_metrics, test_preds = evaluate(model, test_loader, device)

    print(f"\nTest RMSE: {test_rmse:.4f}")
    print("\nPer-task results:")
    for task, metrics in test_metrics.items():
        print(f"  {task}: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.3f}")

    # Save gradient matrix
    gradient_path = output_dir / 'gradient_conflict_matrices.npz'
    gradient_logger.save(str(gradient_path))

    # Print gradient conflict matrix
    G = gradient_logger.get_average_matrix()
    print("\n" + "=" * 60)
    print("GRADIENT CONFLICT MATRIX")
    print("=" * 60)
    print("\nTasks:", tasks)
    print("\nMatrix (positive = synergy, negative = conflict):")
    np.set_printoptions(precision=3, suppress=True)
    print(G)

    # Highlight key findings
    print("\nKey gradient relationships:")
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            val = G[i, j]
            if np.isnan(val):
                continue
            relation = "synergy" if val > 0.1 else ("conflict" if val < -0.1 else "neutral")
            if abs(val) > 0.1:
                print(f"  {tasks[i]} vs {tasks[j]}: G={val:.3f} ({relation})")

    # Save results summary
    results = {
        'test_rmse': float(test_rmse),
        'test_metrics': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in test_metrics.items()},
        'tasks': tasks,
        'gradient_matrix': G.tolist(),
        'n_gradient_samples': len(gradient_logger.gradient_history),
        'config': vars(args),
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")

    return G, test_metrics


if __name__ == '__main__':
    main()
