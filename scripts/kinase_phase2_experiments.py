#!/usr/bin/env python3
"""
Phase 2 Experiments on Kinase Selectivity Data.

Three experiments to validate gradient-based methods on kinase data:
- 2A: Transfer Learning - does G predict transfer success?
- 2B: PCGrad - does PCGrad help negative-G pairs?
- 2C: Task Selection - can we select informative kinase subsets?

Usage:
    python scripts/kinase_phase2_experiments.py --exp transfer --gpu 0
    python scripts/kinase_phase2_experiments.py --exp pcgrad --gpu 0
    python scripts/kinase_phase2_experiments.py --exp selection
    python scripts/kinase_phase2_experiments.py --exp all --gpu 0
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem


# =============================================================================
# Utilities
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


def smiles_to_graph(smiles):
    """Convert SMILES to PyG Data object."""
    if pd.isna(smiles) or not isinstance(smiles, str):
        return None

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


class SimpleGNN(nn.Module):
    """Simple GNN for single-task prediction."""
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x).squeeze(-1)


def load_task_data(data_path: str, task: str, max_samples: int = None):
    """Load and prepare data for a single task."""
    df = pd.read_csv(data_path)

    # Filter to valid rows
    valid_mask = df[task].notna() & df['smiles'].notna()
    df_valid = df[valid_mask].reset_index(drop=True)

    if max_samples and len(df_valid) > max_samples:
        df_valid = df_valid.sample(n=max_samples, random_state=42)

    graphs = []
    labels = []

    for _, row in df_valid.iterrows():
        g = smiles_to_graph(row['smiles'])
        if g is not None:
            graphs.append(g)
            labels.append(float(row[task]))

    return graphs, labels


# =============================================================================
# Experiment 2A: Transfer Learning
# =============================================================================

def train_and_evaluate(
    train_graphs, train_labels,
    test_graphs, test_labels,
    device, epochs=50, lr=1e-3
):
    """Train model and return test correlation."""
    if len(train_graphs) < 10 or len(test_graphs) < 10:
        return 0.0

    input_dim = train_graphs[0].x.shape[1]
    model = SimpleGNN(input_dim).to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    # Training
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_graphs))

        for i in range(0, len(train_graphs), 32):
            idx = perm[i:i+32].tolist()
            batch_graphs = [train_graphs[j] for j in idx]
            batch_labels = torch.tensor([train_labels[j] for j in idx], dtype=torch.float, device=device)

            batch = Batch.from_data_list(batch_graphs).to(device)

            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.batch)
            loss = F.mse_loss(pred, batch_labels)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        batch = Batch.from_data_list(test_graphs).to(device)
        pred = model(batch.x, batch.edge_index, batch.batch).cpu().numpy()

    test_labels_np = np.array(test_labels)

    # Pearson correlation
    if np.std(pred) < 1e-6 or np.std(test_labels_np) < 1e-6:
        return 0.0

    r, _ = stats.pearsonr(pred, test_labels_np)
    return float(r) if not np.isnan(r) else 0.0


def run_transfer_experiment(
    data_path: str,
    gradient_matrix_path: str,
    output_dir: Path,
    device: torch.device,
    seed: int = 42
):
    """Test if gradient similarity predicts transfer learning success."""
    print("=" * 60)
    print("EXPERIMENT 2A: Transfer Learning on Kinase Data")
    print("=" * 60)

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load gradient matrix
    grad_data = np.load(gradient_matrix_path, allow_pickle=True)
    G = grad_data['average_matrix']
    tasks = [str(t) for t in grad_data['tasks']]

    print(f"Loaded gradient matrix: {G.shape}")
    print(f"Tasks: {tasks[:5]}...")

    # Load data
    df = pd.read_csv(data_path)
    available_tasks = [t for t in tasks if t in df.columns]
    print(f"Available tasks in data: {len(available_tasks)}")

    # Build task index mapping
    task_to_idx = {t: i for i, t in enumerate(tasks)}

    # Select pairs to test (mix of high and low G)
    pairs = []
    for i, t1 in enumerate(available_tasks):
        for j, t2 in enumerate(available_tasks):
            if i < j and t1 in task_to_idx and t2 in task_to_idx:
                g = G[task_to_idx[t1], task_to_idx[t2]]
                if not np.isnan(g):
                    pairs.append((t1, t2, g))

    pairs.sort(key=lambda x: x[2])
    print(f"Total pairs: {len(pairs)}")

    # Select diverse pairs: 5 lowest, 5 highest, 5 middle
    n_select = min(5, len(pairs) // 3)
    if n_select < 2:
        print("Not enough pairs to test")
        return {}

    test_pairs = pairs[:n_select] + pairs[-n_select:] + pairs[len(pairs)//2:len(pairs)//2 + n_select]
    print(f"Testing {len(test_pairs)} pairs")

    results = []

    for source, target, g_value in test_pairs:
        print(f"\n  {source} → {target} (G={g_value:.3f})")

        # Load source data for pretraining
        source_graphs, source_labels = load_task_data(data_path, source, max_samples=500)

        # Load target data
        target_graphs, target_labels = load_task_data(data_path, target, max_samples=500)

        if len(target_graphs) < 150:
            print(f"    Skipping: insufficient target data ({len(target_graphs)})")
            continue

        for n_train in [50, 100, 200]:
            if len(target_graphs) < n_train + 50:
                continue

            # Split target data
            perm = np.random.permutation(len(target_graphs))
            train_idx = perm[:n_train]
            test_idx = perm[n_train:n_train+100]

            train_graphs = [target_graphs[i] for i in train_idx]
            train_labels = [target_labels[i] for i in train_idx]
            test_graphs_split = [target_graphs[i] for i in test_idx]
            test_labels_split = [target_labels[i] for i in test_idx]

            # From scratch
            scratch_r = train_and_evaluate(
                train_graphs, train_labels,
                test_graphs_split, test_labels_split,
                device, epochs=50
            )

            # With transfer: pretrain on source, then fine-tune
            if len(source_graphs) >= 100:
                # Pretrain
                input_dim = source_graphs[0].x.shape[1]
                model = SimpleGNN(input_dim).to(device)
                optimizer = Adam(model.parameters(), lr=1e-3)

                model.train()
                for epoch in range(30):
                    perm_s = torch.randperm(min(200, len(source_graphs)))
                    for i in range(0, len(perm_s), 32):
                        idx = perm_s[i:i+32].tolist()
                        batch_g = [source_graphs[j] for j in idx]
                        batch_l = torch.tensor([source_labels[j] for j in idx], dtype=torch.float, device=device)
                        batch = Batch.from_data_list(batch_g).to(device)

                        optimizer.zero_grad()
                        pred = model(batch.x, batch.edge_index, batch.batch)
                        loss = F.mse_loss(pred, batch_l)
                        loss.backward()
                        optimizer.step()

                # Fine-tune on target
                optimizer = Adam(model.parameters(), lr=5e-4)
                for epoch in range(30):
                    perm_t = torch.randperm(len(train_graphs))
                    for i in range(0, len(perm_t), 32):
                        idx = perm_t[i:i+32].tolist()
                        batch_g = [train_graphs[j] for j in idx]
                        batch_l = torch.tensor([train_labels[j] for j in idx], dtype=torch.float, device=device)
                        batch = Batch.from_data_list(batch_g).to(device)

                        optimizer.zero_grad()
                        pred = model(batch.x, batch.edge_index, batch.batch)
                        loss = F.mse_loss(pred, batch_l)
                        loss.backward()
                        optimizer.step()

                # Evaluate
                model.eval()
                with torch.no_grad():
                    batch = Batch.from_data_list(test_graphs_split).to(device)
                    pred = model(batch.x, batch.edge_index, batch.batch).cpu().numpy()

                test_np = np.array(test_labels_split)
                if np.std(pred) > 1e-6 and np.std(test_np) > 1e-6:
                    transfer_r, _ = stats.pearsonr(pred, test_np)
                    transfer_r = float(transfer_r) if not np.isnan(transfer_r) else 0.0
                else:
                    transfer_r = 0.0
            else:
                transfer_r = scratch_r

            benefit = transfer_r - scratch_r

            results.append({
                'source': source,
                'target': target,
                'n_train': n_train,
                'gradient_similarity': g_value,
                'scratch_r': scratch_r,
                'transfer_r': transfer_r,
                'benefit': benefit
            })

            print(f"    n={n_train}: scratch={scratch_r:.3f}, transfer={transfer_r:.3f}, benefit={benefit:+.3f}")

    # Save results
    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_dir / 'transfer_results.csv', index=False)

        # Compute correlation
        if len(df_results) > 3 and df_results['benefit'].std() > 1e-6:
            r, p = stats.pearsonr(df_results['gradient_similarity'], df_results['benefit'])
        else:
            r, p = np.nan, np.nan

        summary = {
            'pearson_r': float(r) if not np.isnan(r) else None,
            'pearson_p': float(p) if not np.isnan(p) else None,
            'n_experiments': len(results),
            'mean_benefit': float(df_results['benefit'].mean()),
            'pct_positive': float((df_results['benefit'] > 0).mean() * 100)
        }

        with open(output_dir / 'transfer_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"Correlation (G vs benefit): r={r:.3f}, p={p:.4f}" if not np.isnan(r) else "Correlation: N/A")
        print(f"Mean benefit: {summary['mean_benefit']:.3f}")
        print(f"% positive transfer: {summary['pct_positive']:.1f}%")

        return summary

    return {}


# =============================================================================
# Experiment 2B: PCGrad
# =============================================================================

def run_pcgrad_experiment(
    data_path: str,
    gradient_matrix_path: str,
    output_dir: Path,
    device: torch.device,
    seed: int = 42
):
    """Test if PCGrad helps negative-G pairs."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2B: PCGrad on Kinase Data")
    print("=" * 60)

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load gradient matrix
    grad_data = np.load(gradient_matrix_path, allow_pickle=True)
    G = grad_data['average_matrix']
    tasks = [str(t) for t in grad_data['tasks']]

    task_to_idx = {t: i for i, t in enumerate(tasks)}

    # Find negative and positive G pairs
    df = pd.read_csv(data_path)
    available_tasks = [t for t in tasks if t in df.columns]

    negative_pairs = []
    positive_pairs = []

    for i, t1 in enumerate(available_tasks):
        for j, t2 in enumerate(available_tasks):
            if i < j and t1 in task_to_idx and t2 in task_to_idx:
                g = G[task_to_idx[t1], task_to_idx[t2]]
                if not np.isnan(g):
                    if g < -0.05:
                        negative_pairs.append((t1, t2, g))
                    elif g > 0.15:
                        positive_pairs.append((t1, t2, g))

    print(f"Negative-G pairs (G<-0.05): {len(negative_pairs)}")
    print(f"Positive-G pairs (G>0.15): {len(positive_pairs)}")

    # Test pairs
    test_pairs = negative_pairs[:5] + positive_pairs[:5]

    if len(test_pairs) == 0:
        print("No suitable pairs found")
        return {}

    results = []

    for t1, t2, g_value in test_pairs:
        print(f"\n  {t1} vs {t2} (G={g_value:.3f})")

        # Get overlapping data
        valid_mask = df[t1].notna() & df[t2].notna() & df['smiles'].notna()
        df_both = df[valid_mask].reset_index(drop=True)

        if len(df_both) < 200:
            print(f"    Skipping: insufficient overlap ({len(df_both)})")
            continue

        # Prepare data
        graphs = []
        labels1 = []
        labels2 = []

        for _, row in df_both.iterrows():
            g = smiles_to_graph(row['smiles'])
            if g is not None:
                graphs.append(g)
                labels1.append(float(row[t1]))
                labels2.append(float(row[t2]))

        if len(graphs) < 150:
            continue

        # Split
        perm = np.random.permutation(len(graphs))
        train_idx = perm[:200]
        test_idx = perm[200:300]

        train_graphs = [graphs[i] for i in train_idx]
        train_l1 = [labels1[i] for i in train_idx]
        train_l2 = [labels2[i] for i in train_idx]

        test_graphs = [graphs[i] for i in test_idx]
        test_l1 = np.array([labels1[i] for i in test_idx])
        test_l2 = np.array([labels2[i] for i in test_idx])

        # Train WITHOUT PCGrad
        baseline_r = train_two_tasks(
            train_graphs, train_l1, train_l2,
            test_graphs, test_l1, test_l2,
            device, use_pcgrad=False
        )

        # Train WITH PCGrad
        pcgrad_r = train_two_tasks(
            train_graphs, train_l1, train_l2,
            test_graphs, test_l1, test_l2,
            device, use_pcgrad=True
        )

        improvement = pcgrad_r - baseline_r
        category = 'negative' if g_value < 0 else 'positive'

        results.append({
            'task1': t1,
            'task2': t2,
            'gradient_similarity': g_value,
            'category': category,
            'baseline_r': baseline_r,
            'pcgrad_r': pcgrad_r,
            'improvement': improvement
        })

        print(f"    Baseline: {baseline_r:.3f}, PCGrad: {pcgrad_r:.3f}, Δ={improvement:+.3f}")

    # Summarize
    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_dir / 'pcgrad_results.csv', index=False)

        neg_results = df_results[df_results['category'] == 'negative']
        pos_results = df_results[df_results['category'] == 'positive']

        summary = {
            'negative_pairs': {
                'n': len(neg_results),
                'mean_improvement': float(neg_results['improvement'].mean()) if len(neg_results) > 0 else None
            },
            'positive_pairs': {
                'n': len(pos_results),
                'mean_improvement': float(pos_results['improvement'].mean()) if len(pos_results) > 0 else None
            }
        }

        with open(output_dir / 'pcgrad_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        if summary['negative_pairs']['n'] > 0:
            print(f"Negative-G pairs: mean improvement = {summary['negative_pairs']['mean_improvement']:.3f}")
        if summary['positive_pairs']['n'] > 0:
            print(f"Positive-G pairs: mean improvement = {summary['positive_pairs']['mean_improvement']:.3f}")

        return summary

    return {}


def train_two_tasks(
    train_graphs, train_l1, train_l2,
    test_graphs, test_l1, test_l2,
    device, use_pcgrad=False, epochs=50
):
    """Train on two tasks and return average test correlation."""
    input_dim = train_graphs[0].x.shape[1]

    class TwoHeadGNN(nn.Module):
        def __init__(self, input_dim, hidden_dim=128):
            super().__init__()
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.head1 = nn.Linear(hidden_dim, 1)
            self.head2 = nn.Linear(hidden_dim, 1)

        def forward(self, x, edge_index, batch):
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.2, training=self.training)
            x = F.relu(self.conv2(x, edge_index))
            x = global_mean_pool(x, batch)
            return self.head1(x).squeeze(-1), self.head2(x).squeeze(-1)

        def shared_params(self):
            return list(self.conv1.parameters()) + list(self.conv2.parameters())

    model = TwoHeadGNN(input_dim).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_graphs))

        for i in range(0, len(train_graphs), 32):
            idx = perm[i:i+32].tolist()
            batch_g = [train_graphs[j] for j in idx]
            l1 = torch.tensor([train_l1[j] for j in idx], dtype=torch.float, device=device)
            l2 = torch.tensor([train_l2[j] for j in idx], dtype=torch.float, device=device)

            batch = Batch.from_data_list(batch_g).to(device)

            optimizer.zero_grad()
            pred1, pred2 = model(batch.x, batch.edge_index, batch.batch)

            loss1 = F.mse_loss(pred1, l1)
            loss2 = F.mse_loss(pred2, l2)

            if use_pcgrad:
                # PCGrad: project conflicting gradients
                shared = model.shared_params()

                g1 = torch.autograd.grad(loss1, shared, retain_graph=True, allow_unused=True)
                g2 = torch.autograd.grad(loss2, shared, retain_graph=True, allow_unused=True)

                g1_flat = torch.cat([g.flatten() if g is not None else torch.zeros(1, device=device) for g in g1])
                g2_flat = torch.cat([g.flatten() if g is not None else torch.zeros(1, device=device) for g in g2])

                # Project if conflicting
                dot = torch.dot(g1_flat, g2_flat)
                if dot < 0:
                    g1_flat = g1_flat - (dot / (g2_flat.norm()**2 + 1e-8)) * g2_flat
                    g2_flat = g2_flat - (dot / (g1_flat.norm()**2 + 1e-8)) * g1_flat

                combined = (g1_flat + g2_flat) / 2

                # Apply gradients
                idx_g = 0
                for p in shared:
                    numel = p.numel()
                    if p.grad is None:
                        p.grad = combined[idx_g:idx_g+numel].view(p.shape)
                    else:
                        p.grad = combined[idx_g:idx_g+numel].view(p.shape)
                    idx_g += numel

                # Backward for heads only
                (loss1 + loss2).backward()
            else:
                (loss1 + loss2).backward()

            optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        batch = Batch.from_data_list(test_graphs).to(device)
        pred1, pred2 = model(batch.x, batch.edge_index, batch.batch)
        pred1, pred2 = pred1.cpu().numpy(), pred2.cpu().numpy()

    # Correlations
    r1 = stats.pearsonr(pred1, test_l1)[0] if np.std(pred1) > 1e-6 else 0
    r2 = stats.pearsonr(pred2, test_l2)[0] if np.std(pred2) > 1e-6 else 0

    r1 = float(r1) if not np.isnan(r1) else 0
    r2 = float(r2) if not np.isnan(r2) else 0

    return (r1 + r2) / 2


# =============================================================================
# Experiment 2C: Task Selection
# =============================================================================

def run_selection_experiment(
    gradient_matrix_path: str,
    empirical_corr_path: str,
    output_dir: Path
):
    """Test gradient-based task selection."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2C: Task Selection on Kinase Data")
    print("=" * 60)

    # Load matrices
    grad_data = np.load(gradient_matrix_path, allow_pickle=True)
    G = grad_data['average_matrix']
    tasks = [str(t) for t in grad_data['tasks']]

    empirical_df = pd.read_csv(empirical_corr_path, index_col=0)

    # Align
    common = [t for t in tasks if t in empirical_df.columns]
    task_idx = {t: i for i, t in enumerate(tasks)}

    G_sub = np.array([[G[task_idx[t1], task_idx[t2]] for t2 in common] for t1 in common])
    n = len(common)

    print(f"Tasks: {n}")

    results = []

    for budget in range(2, min(n, 12)):
        # Greedy selection
        selected = []
        remaining = list(range(n))

        for _ in range(budget):
            best = None
            best_cov = -1

            for t in remaining:
                test = selected + [t]
                cov = np.mean([max(abs(G_sub[i, j]) for j in test) for i in range(n) if i not in test]) if len(test) < n else 0
                if cov > best_cov:
                    best_cov = cov
                    best = t

            if best is not None:
                selected.append(best)
                remaining.remove(best)

        greedy_cov = best_cov

        # Random baseline
        random_covs = []
        for _ in range(100):
            rand_sel = np.random.choice(n, budget, replace=False).tolist()
            cov = np.mean([max(abs(G_sub[i, j]) for j in rand_sel) for i in range(n) if i not in rand_sel]) if len(rand_sel) < n else 0
            random_covs.append(cov)

        results.append({
            'budget': budget,
            'greedy': greedy_cov,
            'random_mean': np.mean(random_covs),
            'random_std': np.std(random_covs),
            'improvement': greedy_cov - np.mean(random_covs),
            'selected': [common[i] for i in selected]
        })

        print(f"  Budget {budget}: greedy={greedy_cov:.3f}, random={np.mean(random_covs):.3f} (Δ={greedy_cov - np.mean(random_covs):+.3f})")

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_dir / 'selection_results.csv', index=False)

    summary = {
        'mean_improvement': float(df_results['improvement'].mean()),
        'max_coverage': float(df_results['greedy'].max()),
        'best_budget': int(df_results.loc[df_results['greedy'].idxmax(), 'budget'])
    }

    with open(output_dir / 'selection_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Mean improvement over random: {summary['mean_improvement']:.3f}")
    print(f"Max coverage: {summary['max_coverage']:.3f} at budget {summary['best_budget']}")

    return summary


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', required=True, choices=['transfer', 'pcgrad', 'selection', 'all'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data', default='outputs/kinase_data/kinase_all_activity_matrix.csv')
    parser.add_argument('--gradients', default='outputs/kinase_all_results/gradient_matrices.npz')
    parser.add_argument('--empirical', default='outputs/kinase_data/kinase_all_empirical_correlations.csv')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = project_root / 'outputs' / 'kinase_phase2'
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.exp in ['transfer', 'all']:
        run_transfer_experiment(args.data, args.gradients, output_dir, device, args.seed)

    if args.exp in ['pcgrad', 'all']:
        run_pcgrad_experiment(args.data, args.gradients, output_dir, device, args.seed)

    if args.exp in ['selection', 'all']:
        run_selection_experiment(args.gradients, args.empirical, output_dir)

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
