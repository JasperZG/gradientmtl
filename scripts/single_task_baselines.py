#!/usr/bin/env python3
"""
Single-Task Baselines

Trains individual single-task models for each property to establish
baseline performance without multi-task learning.

This provides:
1. Upper bound on individual task performance
2. Baseline for measuring negative transfer in MTL
3. Comparison for transfer learning experiments

Usage:
    python scripts/single_task_baselines.py --task NR-AR --seed 42
    python scripts/single_task_baselines.py --all --output-dir outputs/baselines
"""

import os
import sys
import json
import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, mean_squared_error
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.graph_preprocessing import MoleculeGraphPreprocessor
from data.graph_dataset import MultiTaskGraphDataset
from data.splitting import scaffold_split
from models.gnn_encoder import GCNEncoder
from models.heads import TaskHead
from torch_geometric.loader import DataLoader as PyGDataLoader


# Tox21 tasks
TOX21_TASKS = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
    'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
    'SR-HSE', 'SR-MMP', 'SR-p53'
]


class SingleTaskGNNModel(nn.Module):
    """GNN model for single-task prediction."""

    def __init__(
        self,
        atom_feature_dim: int,
        hidden_dim: int = 256,
        head_hidden_dim: int = 128,
        dropout: float = 0.2,
        task_type: str = 'classification'
    ):
        super().__init__()
        self.task_type = task_type

        # GCN encoder
        self.encoder = GCNEncoder(
            input_dim=atom_feature_dim,
            hidden_dims=[hidden_dim, hidden_dim, hidden_dim],
            output_dim=hidden_dim,
            dropout=dropout
        )

        # Task head
        self.head = TaskHead(
            input_dim=hidden_dim,
            hidden_dim=head_hidden_dim
        )

    def forward(self, batch):
        """Forward pass."""
        h = self.encoder(batch)
        return self.head(h)


def load_tox21_data(data_dir: Path, min_tasks: int = 1):
    """Load Tox21 data and convert to graphs."""
    import urllib.request
    import gzip
    import io

    tox21_path = data_dir / 'tox21.csv'
    if not tox21_path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz'
        print("Downloading Tox21...")
        request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(request, timeout=60) as response:
            compressed = response.read()
        with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as f:
            data = f.read()
        with open(tox21_path, 'wb') as f:
            f.write(data)

    df = pd.read_csv(tox21_path)
    smiles_list = df['smiles'].tolist()

    # Get labels
    raw_labels = {}
    for task in TOX21_TASKS:
        if task in df.columns:
            raw_labels[task] = df[task].values.astype(np.float32)

    # Convert to graphs
    preprocessor = MoleculeGraphPreprocessor()
    valid_smiles, graphs, valid_indices = preprocessor.process_smiles_list(
        smiles_list, show_progress=True
    )

    # Filter labels
    labels = {task: values[valid_indices] for task, values in raw_labels.items()}

    # Filter by label coverage
    n_labels = np.zeros(len(valid_smiles))
    for task, values in labels.items():
        n_labels += ~np.isnan(values)

    mask = n_labels >= min_tasks
    keep_indices = np.where(mask)[0]

    graphs = [graphs[i] for i in keep_indices]
    labels = {task: arr[keep_indices] for task, arr in labels.items()}
    valid_smiles = [valid_smiles[i] for i in keep_indices]

    return valid_smiles, graphs, labels, preprocessor.atom_feature_dim


def train_single_task_model(
    task_name: str,
    graphs: list,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    atom_feature_dim: int,
    config: dict,
    device: str = 'cuda'
) -> dict:
    """
    Train a single-task model.

    Returns:
        Dict with training results and metrics
    """
    # Filter to samples with labels for this task
    task_labels = labels
    valid_mask = ~np.isnan(task_labels)

    train_valid = np.isin(np.arange(len(graphs)), train_idx) & valid_mask
    val_valid = np.isin(np.arange(len(graphs)), val_idx) & valid_mask
    test_valid = np.isin(np.arange(len(graphs)), test_idx) & valid_mask

    train_indices = np.where(train_valid)[0]
    val_indices = np.where(val_valid)[0]
    test_indices = np.where(test_valid)[0]

    print(f"  Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

    if len(train_indices) < 50 or len(val_indices) < 10:
        return {'error': 'Insufficient data', 'task': task_name}

    # Create single-task dataset
    single_labels = {task_name: task_labels}
    single_types = {task_name: 'classification'}

    # Create datasets for each split
    train_graphs = [graphs[i] for i in train_indices]
    val_graphs = [graphs[i] for i in val_indices]
    test_graphs = [graphs[i] for i in test_indices]

    train_labels = {task_name: task_labels[train_indices]}
    val_labels = {task_name: task_labels[val_indices]}
    test_labels = {task_name: task_labels[test_indices]}

    train_dataset = MultiTaskGraphDataset(train_graphs, train_labels, single_types)
    val_dataset = MultiTaskGraphDataset(val_graphs, val_labels, single_types)
    test_dataset = MultiTaskGraphDataset(test_graphs, test_labels, single_types)

    train_loader = PyGDataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = PyGDataLoader(val_dataset, batch_size=config['batch_size'])
    test_loader = PyGDataLoader(test_dataset, batch_size=config['batch_size'])

    # Create model
    model = SingleTaskGNNModel(
        atom_feature_dim=atom_feature_dim,
        hidden_dim=config.get('hidden_dim', 256),
        head_hidden_dim=config.get('head_hidden_dim', 128),
        dropout=config.get('dropout', 0.2)
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    loss_fn = nn.BCEWithLogitsLoss()

    # Training loop
    best_val_auc = 0
    best_model_state = None
    patience_counter = 0
    train_losses = []

    for epoch in range(config['epochs']):
        model.train()
        epoch_losses = []

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            outputs = model(batch)

            # Get labels - single task, so simpler
            y = batch.y
            mask = batch.mask

            if y.dim() == 1:
                # May need reshaping based on dataset structure
                pass

            # Ensure we're working with valid samples only
            if mask.sum() == 0:
                continue

            pred = outputs[mask.bool()].squeeze()
            target = y[mask.bool()]

            if len(pred) == 0:
                continue

            loss = loss_fn(pred, target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        train_losses.append(np.mean(epoch_losses) if epoch_losses else 0)

        # Validation
        model.eval()
        val_preds = []
        val_labels_list = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                outputs = model(batch)
                y = batch.y
                mask = batch.mask

                if mask.sum() == 0:
                    continue

                pred = torch.sigmoid(outputs[mask.bool()].squeeze())
                target = y[mask.bool()]

                val_preds.extend(pred.cpu().numpy())
                val_labels_list.extend(target.cpu().numpy())

        if len(val_preds) > 0 and len(np.unique(val_labels_list)) > 1:
            val_auc = roc_auc_score(val_labels_list, val_preds)
        else:
            val_auc = 0.5

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.get('early_stopping_patience', 20):
                break

    # Load best model and evaluate on test set
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()
    test_preds = []
    test_labels_list = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            outputs = model(batch)
            y = batch.y
            mask = batch.mask

            if mask.sum() == 0:
                continue

            pred = torch.sigmoid(outputs[mask.bool()].squeeze())
            target = y[mask.bool()]

            test_preds.extend(pred.cpu().numpy())
            test_labels_list.extend(target.cpu().numpy())

    if len(test_preds) > 0 and len(np.unique(test_labels_list)) > 1:
        test_auc = roc_auc_score(test_labels_list, test_preds)
    else:
        test_auc = 0.5

    return {
        'task': task_name,
        'best_val_auc': float(best_val_auc),
        'test_auc': float(test_auc),
        'epochs_trained': epoch + 1,
        'train_samples': len(train_indices),
        'val_samples': len(val_indices),
        'test_samples': len(test_indices),
    }


def run_all_baselines(
    output_dir: str = 'outputs/baselines',
    seed: int = 42,
    epochs: int = 100
) -> dict:
    """Run baselines for all Tox21 tasks."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("SINGLE-TASK BASELINES")
    print("=" * 70)

    np.random.seed(seed)
    torch.manual_seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load data
    print("\nLoading Tox21 data...")
    data_dir = project_root / 'outputs' / 'raw_data'
    valid_smiles, graphs, labels, atom_feature_dim = load_tox21_data(data_dir)
    print(f"Loaded {len(graphs)} molecules")

    # Scaffold split
    train_idx, val_idx, test_idx = scaffold_split(
        valid_smiles, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=seed
    )

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)

    # Training config
    config = {
        'batch_size': 32,
        'learning_rate': 1e-3,
        'weight_decay': 0.01,
        'epochs': epochs,
        'early_stopping_patience': 20,
        'hidden_dim': 256,
        'head_hidden_dim': 128,
        'dropout': 0.2,
    }

    # Train each task
    all_results = {}

    for task in tqdm(TOX21_TASKS, desc="Training baselines"):
        print(f"\n{'='*50}")
        print(f"Training: {task}")
        print('='*50)

        if task not in labels:
            print(f"  Skipping {task}: not in dataset")
            continue

        result = train_single_task_model(
            task_name=task,
            graphs=graphs,
            labels=labels[task],
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            atom_feature_dim=atom_feature_dim,
            config=config,
            device=device
        )

        all_results[task] = result

        if 'error' not in result:
            print(f"  Val AUC: {result['best_val_auc']:.4f}")
            print(f"  Test AUC: {result['test_auc']:.4f}")

    # Save results
    results_file = output_dir / 'single_task_baselines.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Summary
    print("\n" + "=" * 70)
    print("BASELINE SUMMARY")
    print("=" * 70)
    print(f"{'Task':<20} {'Val AUC':>10} {'Test AUC':>10} {'Train N':>10}")
    print("-" * 50)

    valid_aucs = []
    for task, result in sorted(all_results.items()):
        if 'error' in result:
            print(f"{task:<20} {'ERROR':>10}")
        else:
            print(f"{task:<20} {result['best_val_auc']:>10.4f} {result['test_auc']:>10.4f} {result['train_samples']:>10}")
            valid_aucs.append(result['test_auc'])

    if valid_aucs:
        print("-" * 50)
        print(f"{'Mean'::<20} {np.mean(valid_aucs):>10.4f}")
        print(f"{'Std'::<20} {np.std(valid_aucs):>10.4f}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Single-Task Baselines')
    parser.add_argument('--task', type=str, default=None,
                       help='Specific task to train (default: all)')
    parser.add_argument('--all', action='store_true',
                       help='Train all tasks')
    parser.add_argument('--output-dir', type=str, default='outputs/baselines',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)

    args = parser.parse_args()

    if args.task:
        # Single task mode
        print(f"Training single task: {args.task}")
        # Would need to implement single-task mode here
        pass
    else:
        # All tasks mode
        results = run_all_baselines(
            output_dir=args.output_dir,
            seed=args.seed,
            epochs=args.epochs
        )


if __name__ == '__main__':
    main()
