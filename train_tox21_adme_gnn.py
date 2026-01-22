#!/usr/bin/env python3
"""
Train GNN on Tox21 + ADME Cross-Domain Dataset.

This is the KEY experiment for cross-domain validation:
- Toxicity endpoints: 12 Tox21 assays (classification)
- ADME endpoints: Measured experimental data (mixed)

Key hypothesis: Gradient conflicts between Tox and ADME reveal
mechanistic relationships not visible in single-domain analysis.

Expected patterns:
- Solubility vs Tox_NR-*: Lipophilic compounds often bind nuclear receptors
- CYP inhibition vs Tox_*: CYP inhibitors may have different toxicity profiles
- Permeability (HIA, Caco2) vs Tox_*: Transport affects exposure

Usage:
    python train_tox21_adme_gnn.py
    python train_tox21_adme_gnn.py --epochs 50 --batch-size 64
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
# Task Configuration
# =============================================================================

# ADME task types: True = classification, False = regression
ADME_TASK_TYPES = {
    # Regression tasks (continuous values)
    'ADME_Solubility': 'regression',
    'ADME_Lipophilicity': 'regression',
    'ADME_Caco2': 'regression',
    'ADME_PPB': 'regression',
    'ADME_VDss': 'regression',
    'ADME_Clearance_Hepatocyte': 'regression',
    'ADME_Clearance_Microsome': 'regression',
    'MN_ESOL': 'regression',
    'MN_FreeSolv': 'regression',
    'MN_Lipophilicity_MN': 'regression',
    'ADME_PAMPA': 'regression',
    'ADME_HLM': 'regression',

    # Classification tasks (binary)
    'ADME_Bioavailability': 'classification',
    'ADME_HIA': 'classification',
    'ADME_CYP2D6_Inhibitor': 'classification',
    'ADME_CYP3A4_Inhibitor': 'classification',
    'ADME_CYP2C9_Inhibitor': 'classification',
}

# Duplicate ADME datasets to exclude (keep originals)
ADME_DUPLICATES = ['MN_ESOL_raw', 'MN_Lipophilicity_raw']


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

class Tox21ADMEDataset:
    """Dataset for Tox21 + ADME cross-domain learning."""

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
                if len(values) > 0:
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
                    values = np.where(mask, (values - norm['mean']) / norm['std'], 0.0)

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

        # Reshape mask and labels: PyG concatenates them
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
    parser = argparse.ArgumentParser(description='Train GNN on Tox21 + ADME cross-domain data')
    parser.add_argument('--data-path', type=str,
                       default='outputs/tox21_adme_augmented/tox21_adme_augmented.csv')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='outputs/tox21_adme_results')
    parser.add_argument('--max-tox-tasks', type=int, default=8,
                       help='Max toxicity tasks to use')
    parser.add_argument('--max-adme-tasks', type=int, default=8,
                       help='Max ADME tasks to use')
    parser.add_argument('--min-samples', type=int, default=100,
                       help='Minimum samples required for a task')
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
        print("Please run: python scripts/augment_tox21_with_adme.py")
        sys.exit(1)

    df = pd.read_csv(data_path)

    print("\n" + "=" * 60)
    print("TOX21 + ADME CROSS-DOMAIN GRADIENT ANALYSIS")
    print("=" * 60)
    print(f"\nTotal compounds: {len(df)}")

    # Identify toxicity tasks (Tox21 endpoints)
    tox_tasks = [c for c in df.columns if c.startswith('NR-') or c.startswith('SR-')]

    # Filter by sample count
    tox_task_counts = {t: df[t].notna().sum() for t in tox_tasks}
    tox_tasks = sorted([t for t in tox_tasks if tox_task_counts[t] >= args.min_samples],
                       key=lambda x: -tox_task_counts[x])[:args.max_tox_tasks]

    # Identify ADME tasks (exclude duplicates)
    adme_cols = [c for c in df.columns
                 if c.startswith('ADME_') or c.startswith('MN_')]
    adme_cols = [c for c in adme_cols if c not in ADME_DUPLICATES]

    # Filter by sample count
    adme_task_counts = {t: df[t].notna().sum() for t in adme_cols}
    adme_tasks = sorted([t for t in adme_cols if adme_task_counts[t] >= args.min_samples],
                        key=lambda x: -adme_task_counts[x])[:args.max_adme_tasks]

    tasks = tox_tasks + adme_tasks

    # Build task types
    task_types = {}
    for t in tox_tasks:
        task_types[t] = 'classification'
    for t in adme_tasks:
        task_types[t] = ADME_TASK_TYPES.get(t, 'regression')

    print(f"\nToxicity tasks ({len(tox_tasks)}):")
    for t in tox_tasks:
        print(f"  {t}: {tox_task_counts[t]} samples ({task_types[t]})")

    print(f"\nADME tasks ({len(adme_tasks)}):")
    for t in adme_tasks:
        print(f"  {t}: {adme_task_counts[t]} samples ({task_types[t]})")

    print(f"\nTotal tasks: {len(tasks)}")

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = Tox21ADMEDataset(str(data_path), tasks, task_types, 'train', seed=args.seed)
    val_dataset = Tox21ADMEDataset(str(data_path), tasks, task_types, 'val', seed=args.seed)
    test_dataset = Tox21ADMEDataset(str(data_path), tasks, task_types, 'test', seed=args.seed)

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
    print("FINAL TEST RESULTS")
    print("=" * 60)

    test_metrics = evaluate(model, test_loader, device)

    print("\nToxicity metrics:")
    for task in tox_tasks:
        m = test_metrics[task]
        print(f"  {task}: {m['name']}={m['metric']:.4f}")

    print("\nADME metrics:")
    for task in adme_tasks:
        m = test_metrics[task]
        print(f"  {task}: {m['name']}={m['metric']:.4f}")

    # Save gradients
    gradient_logger.save(str(output_dir / 'gradient_matrices.npz'))

    G = gradient_logger.get_average_matrix()

    print("\n" + "=" * 60)
    print("CROSS-DOMAIN GRADIENT ANALYSIS (Tox vs ADME)")
    print("=" * 60)

    cross_pairs = []
    for i, t_i in enumerate(tasks):
        for j, t_j in enumerate(tasks):
            if i >= j:
                continue
            is_tox_i = t_i in tox_tasks
            is_tox_j = t_j in tox_tasks
            is_cross = is_tox_i != is_tox_j

            if is_cross and not np.isnan(G[i, j]):
                tox_task = t_i if is_tox_i else t_j
                adme_task = t_j if is_tox_i else t_i
                cross_pairs.append({
                    'tox': tox_task,
                    'adme': adme_task,
                    'G': G[i, j]
                })

    if cross_pairs:
        print("\nCross-domain relationships (sorted by magnitude):")
        for p in sorted(cross_pairs, key=lambda x: abs(x['G']), reverse=True)[:15]:
            if p['G'] > 0.3:
                rel = "STRONG SYNERGY"
            elif p['G'] > 0.1:
                rel = "synergy"
            elif p['G'] < -0.3:
                rel = "STRONG CONFLICT"
            elif p['G'] < -0.1:
                rel = "conflict"
            else:
                rel = "neutral"
            print(f"  {p['tox']} vs {p['adme']}: G={p['G']:.3f} ({rel})")

    # Within-domain analysis
    print("\n" + "=" * 60)
    print("WITHIN-DOMAIN ANALYSIS")
    print("=" * 60)

    # Tox-Tox
    print("\nToxicity (Tox vs Tox):")
    tox_tox_pairs = []
    for i, t_i in enumerate(tox_tasks):
        for j, t_j in enumerate(tox_tasks):
            if i >= j:
                continue
            idx_i, idx_j = tasks.index(t_i), tasks.index(t_j)
            if not np.isnan(G[idx_i, idx_j]):
                tox_tox_pairs.append((t_i, t_j, G[idx_i, idx_j]))

    for t_i, t_j, g in sorted(tox_tox_pairs, key=lambda x: abs(x[2]), reverse=True)[:10]:
        print(f"  {t_i} vs {t_j}: G={g:.3f}")

    # ADME-ADME
    print("\nADME (ADME vs ADME):")
    adme_adme_pairs = []
    for i, t_i in enumerate(adme_tasks):
        for j, t_j in enumerate(adme_tasks):
            if i >= j:
                continue
            idx_i, idx_j = tasks.index(t_i), tasks.index(t_j)
            if not np.isnan(G[idx_i, idx_j]):
                adme_adme_pairs.append((t_i, t_j, G[idx_i, idx_j]))

    for t_i, t_j, g in sorted(adme_adme_pairs, key=lambda x: abs(x[2]), reverse=True)[:10]:
        print(f"  {t_i} vs {t_j}: G={g:.3f}")

    # Validation statistics
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    cross_g_values = [p['G'] for p in cross_pairs]
    within_tox = [g for _, _, g in tox_tox_pairs]
    within_adme = [g for _, _, g in adme_adme_pairs]

    print(f"\nCross-domain (Tox vs ADME):")
    print(f"  N pairs: {len(cross_g_values)}")
    if cross_g_values:
        print(f"  Mean G: {np.mean(cross_g_values):.3f}")
        print(f"  Std G: {np.std(cross_g_values):.3f}")
        print(f"  Range: [{min(cross_g_values):.3f}, {max(cross_g_values):.3f}]")

    print(f"\nWithin-Toxicity:")
    print(f"  N pairs: {len(within_tox)}")
    if within_tox:
        print(f"  Mean G: {np.mean(within_tox):.3f}")
        print(f"  Std G: {np.std(within_tox):.3f}")

    print(f"\nWithin-ADME:")
    print(f"  N pairs: {len(within_adme)}")
    if within_adme:
        print(f"  Mean G: {np.mean(within_adme):.3f}")
        print(f"  Std G: {np.std(within_adme):.3f}")

    # Compare cross vs within
    if cross_g_values and (within_tox or within_adme):
        within_all = within_tox + within_adme
        if len(cross_g_values) >= 5 and len(within_all) >= 5:
            t_stat, p_val = stats.ttest_ind(cross_g_values, within_all)
            print(f"\nCross vs Within comparison:")
            print(f"  t-statistic: {t_stat:.3f}")
            print(f"  p-value: {p_val:.4f}")
            if p_val < 0.05:
                print("  -> Significant difference between cross-domain and within-domain patterns!")

    # Save results
    results = {
        'test_metrics': {k: {kk: (float(vv) if isinstance(vv, (int, float)) else str(vv))
                            for kk, vv in v.items()}
                        for k, v in test_metrics.items()},
        'tasks': tasks,
        'tox_tasks': tox_tasks,
        'adme_tasks': adme_tasks,
        'task_types': task_types,
        'gradient_matrix': G.tolist(),
        'cross_domain_pairs': cross_pairs,
        'n_gradient_samples': len(gradient_logger.gradient_history),
        'validation': {
            'cross_domain_mean': float(np.mean(cross_g_values)) if cross_g_values else None,
            'within_tox_mean': float(np.mean(within_tox)) if within_tox else None,
            'within_adme_mean': float(np.mean(within_adme)) if within_adme else None,
        },
        'config': vars(args),
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}")

    return G, test_metrics


if __name__ == '__main__':
    main()
