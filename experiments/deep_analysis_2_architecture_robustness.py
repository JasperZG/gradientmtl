#!/usr/bin/env python3
"""
Deep Analysis 2: Architecture Robustness

Question: Do gradient conflict patterns depend on model architecture?
Method: Compare ECFP+MLP, GCN, GAT, 1D-CNN on SMILES
Expected: Correlation > 0.8 across architectures (method is architecture-agnostic)
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.loader import DataLoader as PyGDataLoader
import urllib.request
import gzip
import io
import json
from scipy import stats
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem

from data.graph_preprocessing import MoleculeGraphPreprocessor, get_atom_feature_dim
from data.splitting import scaffold_split
from data.graph_dataset import MultiTaskGraphDataset
from models.gnn_multitask import GNNMultiTaskModel
from training.losses import MultiTaskLoss
from training.gradient_logger import GradientConflictLogger


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


class ECFPEncoder(nn.Module):
    """ECFP fingerprint + MLP encoder."""
    def __init__(self, input_dim=2048, hidden_dims=[512, 256], output_dim=256, dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class CNNEncoder(nn.Module):
    """1D CNN on SMILES character sequence."""
    def __init__(self, vocab_size=100, embed_dim=64, output_dim=256, max_len=200, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, output_dim)

    def forward(self, x):
        # x: [batch, seq_len]
        x = self.embedding(x)  # [batch, seq_len, embed]
        x = x.transpose(1, 2)  # [batch, embed, seq_len]
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)  # [batch, 256]
        x = self.dropout(x)
        return self.fc(x)


class TaskHead(nn.Module):
    """Simple task head."""
    def __init__(self, input_dim=256, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x).squeeze(-1)


class MLPMultiTaskModel(nn.Module):
    """Multi-task model with MLP encoder (for ECFP)."""
    def __init__(self, task_names, input_dim=2048, encoder_hidden=[512, 256],
                 encoder_output=256, head_hidden=128, dropout=0.3):
        super().__init__()
        self.encoder = ECFPEncoder(input_dim, encoder_hidden, encoder_output, dropout)
        self.heads = nn.ModuleDict({
            task: TaskHead(encoder_output, head_hidden, dropout)
            for task in task_names
        })
        self.task_names = task_names

    def forward(self, x):
        h = self.encoder(x)
        return {task: head(h) for task, head in self.heads.items()}


class CNNMultiTaskModel(nn.Module):
    """Multi-task model with 1D CNN encoder (for SMILES)."""
    def __init__(self, task_names, vocab_size=100, embed_dim=64, encoder_output=256,
                 head_hidden=128, dropout=0.3):
        super().__init__()
        self.encoder = CNNEncoder(vocab_size, embed_dim, encoder_output, dropout=dropout)
        self.heads = nn.ModuleDict({
            task: TaskHead(encoder_output, head_hidden, dropout)
            for task in task_names
        })
        self.task_names = task_names

    def forward(self, x):
        h = self.encoder(x)
        return {task: head(h) for task, head in self.heads.items()}


def download_tox21():
    """Download Tox21 dataset."""
    output_dir = Path('outputs/raw_data')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'tox21.csv'

    if output_path.exists():
        return output_path

    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz'
    print(f"Downloading Tox21...")

    request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(request, timeout=60) as response:
        compressed = response.read()

    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as f:
        data = f.read()

    with open(output_path, 'wb') as f:
        f.write(data)

    return output_path


def smiles_to_ecfp(smiles_list, radius=2, n_bits=2048):
    """Convert SMILES to ECFP fingerprints."""
    fps = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                fps.append(np.array(fp))
                valid_idx.append(i)
        except:
            pass
    return np.array(fps), valid_idx


def smiles_to_tokens(smiles_list, max_len=200):
    """Convert SMILES to token sequences."""
    # Extended character-level tokenization covering common SMILES characters
    smiles_chars = (
        'CNOSPFIHBcnospfihb'  # Elements (upper and lower case)
        'lrae'                # Parts of Cl, Br, Se (lowercase continuations)
        '=#+-'                # Bond types
        '()[]'                # Brackets
        '0123456789'          # Numbers for rings
        '@/\\.%'              # Stereochemistry and other
        ':*'                  # Additional valid SMILES chars
    )
    char_to_idx = {c: i+1 for i, c in enumerate(smiles_chars)}
    char_to_idx['<pad>'] = 0
    char_to_idx['<unk>'] = len(char_to_idx)  # Unknown character token

    vocab_size = len(char_to_idx)
    unk_idx = char_to_idx['<unk>']

    tokens = []
    for smi in smiles_list:
        seq = [char_to_idx.get(c, unk_idx) for c in smi[:max_len]]
        seq = seq + [0] * (max_len - len(seq))
        tokens.append(seq)
    return np.array(tokens), vocab_size


def compute_gradient_matrix_mlp(model, loader, task_types, task_names, device, n_batches=50):
    """Compute gradient matrix for MLP model."""
    model.train()
    loss_fn = MultiTaskLoss(task_types)

    # Get encoder parameters
    encoder_params = list(model.encoder.parameters())

    all_conflicts = []

    for batch_idx, (x, labels_t, masks_t) in enumerate(loader):
        if batch_idx >= n_batches:
            break

        x = x.to(device)
        labels_t = labels_t.to(device)
        masks_t = masks_t.to(device)

        labels = {task: labels_t[:, i] for i, task in enumerate(task_names)}
        masks = {task: masks_t[:, i] for i, task in enumerate(task_names)}

        predictions = model(x)
        task_losses = loss_fn.get_individual_losses(predictions, labels, masks)

        # Compute gradients for each task
        task_grads = {}
        for task, loss in task_losses.items():
            if loss.requires_grad:
                grads = torch.autograd.grad(
                    loss, encoder_params, retain_graph=True, allow_unused=True
                )
                grad_vec = torch.cat([
                    g.flatten() if g is not None else torch.zeros_like(p).flatten()
                    for g, p in zip(grads, encoder_params)
                ])
                task_grads[task] = grad_vec

        if len(task_grads) >= 2:
            # Compute conflict matrix
            n_tasks = len(task_names)
            conflict = np.zeros((n_tasks, n_tasks))

            for i, t1 in enumerate(task_names):
                for j, t2 in enumerate(task_names):
                    if t1 in task_grads and t2 in task_grads:
                        g1, g2 = task_grads[t1], task_grads[t2]
                        sim = torch.dot(g1, g2) / (g1.norm() * g2.norm() + 1e-8)
                        conflict[i, j] = sim.item()

            all_conflicts.append(conflict)

    return np.mean(all_conflicts, axis=0) if all_conflicts else np.eye(len(task_names))


def compute_gradient_matrix_gnn(model, loader, task_types, task_names, device, n_batches=50):
    """Compute gradient matrix for GNN model."""
    model.train()
    loss_fn = MultiTaskLoss(task_types)

    gradient_logger = GradientConflictLogger(
        model=model,
        task_names=task_names,
        log_interval=1,
        device=device,
    )

    for batch_idx, batch_graph in enumerate(loader):
        if batch_idx >= n_batches:
            break

        batch_graph = batch_graph.to(device)

        labels_tensor = batch_graph.y
        masks_tensor = batch_graph.mask
        batch_size = batch_graph.num_graphs
        n_tasks = len(task_names)

        if labels_tensor.dim() == 1:
            labels_tensor = labels_tensor.view(batch_size, n_tasks)
            masks_tensor = masks_tensor.view(batch_size, n_tasks)

        labels = {task: labels_tensor[:, i] for i, task in enumerate(task_names)}
        masks = {task: masks_tensor[:, i] for i, task in enumerate(task_names)}

        predictions = model(batch_graph)
        task_losses = loss_fn.get_individual_losses(predictions, labels, masks)
        gradient_logger.log_step(batch_idx, task_losses)

    return gradient_logger.get_averaged_conflict_matrix()


def train_mlp_model(model, train_loader, task_names, device, epochs=50, lr=1e-3):
    """Train MLP model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = MultiTaskLoss(TOX21_TASKS)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, labels_t, masks_t in train_loader:
            x = x.to(device)
            labels_t = labels_t.to(device)
            masks_t = masks_t.to(device)

            labels = {task: labels_t[:, i] for i, task in enumerate(task_names)}
            masks = {task: masks_t[:, i] for i, task in enumerate(task_names)}

            optimizer.zero_grad()
            predictions = model(x)
            loss, _ = loss_fn(predictions, labels, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss = {total_loss/len(train_loader):.4f}")


def train_gnn_model(model, train_loader, task_names, device, epochs=50, lr=1e-3):
    """Train GNN model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = MultiTaskLoss(TOX21_TASKS)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_graph in train_loader:
            batch_graph = batch_graph.to(device)

            labels_tensor = batch_graph.y
            masks_tensor = batch_graph.mask
            batch_size = batch_graph.num_graphs
            n_tasks = len(task_names)

            if labels_tensor.dim() == 1:
                labels_tensor = labels_tensor.view(batch_size, n_tasks)
                masks_tensor = masks_tensor.view(batch_size, n_tasks)

            labels = {task: labels_tensor[:, i] for i, task in enumerate(task_names)}
            masks = {task: masks_tensor[:, i] for i, task in enumerate(task_names)}

            optimizer.zero_grad()
            predictions = model(batch_graph)
            loss, _ = loss_fn(predictions, labels, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss = {total_loss/len(train_loader):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    print("\n" + "=" * 60)
    print("Loading Tox21 for Architecture Robustness Analysis")
    print("=" * 60)

    tox21_path = download_tox21()
    df = pd.read_csv(tox21_path)

    smiles_list = df['smiles'].tolist()
    task_names = list(TOX21_TASKS.keys())

    # Prepare ECFP fingerprints
    print("\nPreparing ECFP fingerprints...")
    ecfp_fps, ecfp_valid_idx = smiles_to_ecfp(smiles_list)
    print(f"  Valid: {len(ecfp_valid_idx)}/{len(smiles_list)}")

    # Prepare SMILES tokens
    print("Preparing SMILES tokens...")
    smiles_tokens, vocab_size = smiles_to_tokens(smiles_list)

    # Prepare labels
    raw_labels = {}
    for task in TOX21_TASKS:
        if task in df.columns:
            raw_labels[task] = df[task].values.astype(np.float32)

    # Filter to ECFP valid indices
    ecfp_labels = {task: arr[ecfp_valid_idx] for task, arr in raw_labels.items()}
    ecfp_smiles = [smiles_list[i] for i in ecfp_valid_idx]
    ecfp_tokens = smiles_tokens[ecfp_valid_idx]

    # Filter to molecules with 10+ task labels
    n_labels_per_mol = np.zeros(len(ecfp_valid_idx))
    for task, values in ecfp_labels.items():
        n_labels_per_mol += ~np.isnan(values)

    mask = n_labels_per_mol >= 10
    keep_idx = np.where(mask)[0]

    ecfp_fps = ecfp_fps[keep_idx]
    ecfp_smiles = [ecfp_smiles[i] for i in keep_idx]
    ecfp_tokens = ecfp_tokens[keep_idx]
    ecfp_labels = {task: arr[keep_idx] for task, arr in ecfp_labels.items()}

    print(f"Using {len(keep_idx)} molecules with 10+ task labels")

    # Scaffold split
    train_idx, val_idx, test_idx = scaffold_split(
        ecfp_smiles, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=seed
    )

    # Prepare training data
    train_fps = torch.FloatTensor(ecfp_fps[train_idx])
    train_tokens = torch.LongTensor(ecfp_tokens[train_idx])

    train_labels_arr = np.stack([ecfp_labels[t][train_idx] for t in task_names], axis=1)
    train_labels_t = torch.FloatTensor(np.nan_to_num(train_labels_arr, nan=0.0))
    train_masks_t = torch.FloatTensor(~np.isnan(train_labels_arr))

    # Create data loaders
    mlp_loader = DataLoader(
        TensorDataset(train_fps, train_labels_t, train_masks_t),
        batch_size=args.batch_size, shuffle=True
    )

    cnn_loader = DataLoader(
        TensorDataset(train_tokens, train_labels_t, train_masks_t),
        batch_size=args.batch_size, shuffle=True
    )

    # Prepare graph data for GNN
    print("\nPreparing graph data...")
    preprocessor = MoleculeGraphPreprocessor()
    valid_gnn_smiles, graphs, gnn_valid_idx = preprocessor.process_smiles_list(
        ecfp_smiles, show_progress=True
    )

    # Align GNN data with ECFP data
    gnn_train_idx = [i for i in range(len(graphs)) if gnn_valid_idx[i] in train_idx]
    gnn_train_graphs = [graphs[i] for i in gnn_train_idx]

    # Get labels for GNN training indices
    gnn_labels = {task: ecfp_labels[task][gnn_valid_idx][gnn_train_idx] for task in task_names}

    gnn_dataset = MultiTaskGraphDataset(gnn_train_graphs, gnn_labels, TOX21_TASKS)
    gnn_loader = PyGDataLoader(gnn_dataset, batch_size=args.batch_size, shuffle=True)

    # Results storage
    gradient_matrices = {}

    print("\n" + "=" * 60)
    print("Testing Architecture 1: ECFP + MLP")
    print("=" * 60)

    mlp_model = MLPMultiTaskModel(
        task_names=task_names,
        input_dim=2048,
        encoder_hidden=[512, 256],
        encoder_output=256,
        head_hidden=128,
        dropout=0.3,
    ).to(device)

    train_mlp_model(mlp_model, mlp_loader, task_names, device, epochs=args.epochs)
    gradient_matrices['ECFP_MLP'] = compute_gradient_matrix_mlp(
        mlp_model, mlp_loader, TOX21_TASKS, task_names, device
    )
    print(f"  Gradient matrix computed")

    print("\n" + "=" * 60)
    print("Testing Architecture 2: GCN")
    print("=" * 60)

    gcn_model = GNNMultiTaskModel(
        task_names=task_names,
        atom_feature_dim=get_atom_feature_dim(),
        encoder_type='gcn',
        encoder_hidden_dims=[256, 256, 256],
        encoder_output_dim=256,
        head_hidden_dim=128,
        dropout=0.3,
    ).to(device)

    train_gnn_model(gcn_model, gnn_loader, task_names, device, epochs=args.epochs)
    gradient_matrices['GCN'] = compute_gradient_matrix_gnn(
        gcn_model, gnn_loader, TOX21_TASKS, task_names, device
    )
    print(f"  Gradient matrix computed")

    print("\n" + "=" * 60)
    print("Testing Architecture 3: GAT")
    print("=" * 60)

    gat_model = GNNMultiTaskModel(
        task_names=task_names,
        atom_feature_dim=get_atom_feature_dim(),
        encoder_type='gat',
        encoder_hidden_dims=[256, 256, 256],
        encoder_output_dim=256,
        head_hidden_dim=128,
        dropout=0.3,
    ).to(device)

    train_gnn_model(gat_model, gnn_loader, task_names, device, epochs=args.epochs)
    gradient_matrices['GAT'] = compute_gradient_matrix_gnn(
        gat_model, gnn_loader, TOX21_TASKS, task_names, device
    )
    print(f"  Gradient matrix computed")

    print("\n" + "=" * 60)
    print("Testing Architecture 4: 1D-CNN on SMILES")
    print("=" * 60)

    cnn_model = CNNMultiTaskModel(
        task_names=task_names,
        vocab_size=vocab_size,
        embed_dim=64,
        encoder_output=256,
        head_hidden=128,
        dropout=0.3,
    ).to(device)

    train_mlp_model(cnn_model, cnn_loader, task_names, device, epochs=args.epochs)
    gradient_matrices['CNN_SMILES'] = compute_gradient_matrix_mlp(
        cnn_model, cnn_loader, TOX21_TASKS, task_names, device
    )
    print(f"  Gradient matrix computed")

    # Compute cross-architecture correlations
    print("\n" + "=" * 60)
    print("Cross-Architecture Correlation Analysis")
    print("=" * 60)

    architectures = list(gradient_matrices.keys())
    results = {
        'architectures': architectures,
        'gradient_matrices': {k: v.tolist() for k, v in gradient_matrices.items()},
        'task_names': task_names,
        'correlations': {},
    }

    print(f"\n{'Arch 1':>15} {'Arch 2':>15} {'Pearson r':>12} {'p-value':>12}")
    print("-" * 60)

    for i, a1 in enumerate(architectures):
        for a2 in architectures[i+1:]:
            g1 = gradient_matrices[a1][np.triu_indices(len(task_names), k=1)]
            g2 = gradient_matrices[a2][np.triu_indices(len(task_names), k=1)]
            r, p = stats.pearsonr(g1, g2)
            results['correlations'][f'{a1}_vs_{a2}'] = {'r': r, 'p': p}
            print(f"{a1:>15} {a2:>15} {r:>12.4f} {p:>12.2e}")

    # Summary
    all_correlations = [v['r'] for v in results['correlations'].values()]
    mean_r = np.mean(all_correlations)
    min_r = np.min(all_correlations)

    print(f"\nSummary:")
    print(f"  Mean cross-architecture correlation: {mean_r:.4f}")
    print(f"  Minimum correlation: {min_r:.4f}")

    if min_r > 0.8:
        print(f"\n>>> PASS: Gradient patterns are architecture-agnostic (min r > 0.8)")
    elif min_r > 0.6:
        print(f"\n>>> PARTIAL: Moderate architecture dependence (min r = {min_r:.2f})")
    else:
        print(f"\n>>> FAIL: Gradient patterns are architecture-dependent (min r = {min_r:.2f})")

    results['summary'] = {
        'mean_correlation': float(mean_r),
        'min_correlation': float(min_r),
        'pass': bool(min_r > 0.8),
    }

    # Save results
    output_dir = Path('outputs/deep_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'architecture_robustness_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    np.savez(
        output_dir / 'architecture_gradient_matrices.npz',
        **gradient_matrices,
        task_names=task_names,
    )

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, (arch, matrix) in enumerate(gradient_matrices.items()):
        ax = axes[idx // 2, idx % 2]
        im = ax.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title(arch)
        ax.set_xticks(range(len(task_names)))
        ax.set_yticks(range(len(task_names)))
        ax.set_xticklabels([t[:6] for t in task_names], rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels([t[:6] for t in task_names], fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle('Gradient Conflict Matrices Across Architectures', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'architecture_robustness_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
