#!/usr/bin/env python3
"""
Improved Tox21 training with:
1. Longer training (100 epochs)
2. Only use molecules with labels for ALL 12 tasks
3. Larger encoder for more capacity
4. Learning rate warmup
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import urllib.request
import gzip
import io

from data.preprocessing import MoleculePreprocessor
from data.splitting import scaffold_split
from data.dataset import MultiTaskMoleculeDataset, multitask_collate_fn
from models.multitask import MultiTaskModel
from training.trainer import MultiTaskTrainer
from analysis.visualization import (
    plot_conflict_heatmap,
    plot_conflict_evolution,
    print_conflict_summary,
)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--complete_only', action='store_true',
                       help='Only use molecules with ALL 12 task labels')
    parser.add_argument('--min_tasks', type=int, default=8,
                       help='Minimum number of tasks with labels per molecule')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    print("\n" + "=" * 60)
    print("Loading Tox21")
    print("=" * 60)

    tox21_path = download_tox21()
    df = pd.read_csv(tox21_path)

    smiles_list = df['smiles'].tolist()
    raw_labels = {}
    for task in TOX21_TASKS:
        if task in df.columns:
            raw_labels[task] = df[task].values.astype(np.float32)

    print(f"Total molecules: {len(smiles_list)}")

    # Preprocess
    print("\n" + "=" * 60)
    print("Preprocessing")
    print("=" * 60)

    preprocessor = MoleculePreprocessor(fp_bits=2048, fp_radius=2)
    valid_smiles, fingerprints, valid_indices = preprocessor.process_smiles_list(
        smiles_list, show_progress=True
    )

    # Filter labels
    labels = {task: values[valid_indices] for task, values in raw_labels.items()}

    # Filter to molecules with enough task labels
    n_labels_per_mol = np.zeros(len(valid_smiles))
    for task, values in labels.items():
        n_labels_per_mol += ~np.isnan(values)

    if args.complete_only:
        # Only molecules with ALL 12 labels
        mask = n_labels_per_mol >= 12
        threshold = 12
    else:
        # Molecules with at least min_tasks labels
        mask = n_labels_per_mol >= args.min_tasks
        threshold = args.min_tasks

    keep_indices = np.where(mask)[0]

    print(f"\nFiltering to molecules with {threshold}+ task labels:")
    print(f"  Before: {len(valid_smiles)} molecules")
    print(f"  After: {len(keep_indices)} molecules")

    # Filter data
    fingerprints = fingerprints[keep_indices]
    valid_smiles = [valid_smiles[i] for i in keep_indices]
    labels = {task: arr[keep_indices] for task, arr in labels.items()}

    # Check label coverage after filtering
    print("\nLabel coverage after filtering:")
    for task in labels:
        n_valid = np.sum(~np.isnan(labels[task]))
        n_pos = np.nansum(labels[task])
        print(f"  {task}: {n_valid} labels ({100*n_valid/len(fingerprints):.1f}%), "
              f"{n_pos:.0f} positive ({100*n_pos/n_valid:.1f}%)")

    # Scaffold split
    print("\n" + "=" * 60)
    print("Scaffold Split")
    print("=" * 60)

    train_idx, val_idx, test_idx = scaffold_split(
        valid_smiles, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=seed
    )

    # Create datasets
    print("\n" + "=" * 60)
    print("Creating Data Loaders")
    print("=" * 60)

    train_fp = fingerprints[train_idx]
    val_fp = fingerprints[val_idx]
    train_labels = {task: arr[train_idx] for task, arr in labels.items()}
    val_labels = {task: arr[val_idx] for task, arr in labels.items()}

    print("\nTraining dataset:")
    train_dataset = MultiTaskMoleculeDataset(train_fp, train_labels, TOX21_TASKS)

    print("\nValidation dataset:")
    val_dataset = MultiTaskMoleculeDataset(val_fp, val_labels, TOX21_TASKS)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=multitask_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=multitask_collate_fn
    )

    # Create model with LARGER encoder
    print("\n" + "=" * 60)
    print("Creating Model (Larger Encoder)")
    print("=" * 60)

    task_names = list(TOX21_TASKS.keys())
    model = MultiTaskModel(
        task_names=task_names,
        input_dim=2048,
        encoder_hidden_dims=[1024, 512, 256],  # Deeper encoder
        head_hidden_dim=128,
        dropout=0.3,  # More dropout
    )

    print(model.summary())

    # Train
    print("\n" + "=" * 60)
    print(f"Training ({args.epochs} epochs)")
    print("=" * 60)

    config = {
        'learning_rate': args.lr,
        'weight_decay': 0.01,
        'epochs': args.epochs,
        'early_stopping_patience': 25,  # More patience
        'gradient_log_interval': 5,  # Log more frequently
        'gradient_clip_norm': 1.0,
    }

    trainer = MultiTaskTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        task_types=TOX21_TASKS,
        config=config,
        device=device,
        output_dir=Path('outputs'),
    )

    results = trainer.train()

    # Analysis
    print("\n" + "=" * 60)
    print("Gradient Conflict Analysis")
    print("=" * 60)

    expected_patterns = {
        'NR-AR_NR-AR-LBD': 0.3,
        'NR-ER_NR-ER-LBD': 0.3,
        'SR-ARE_SR-MMP': 0.2,
    }

    print_conflict_summary(
        results['conflict_matrix'],
        results['task_names'],
        expected_patterns,
    )

    # Visualize
    output_dir = Path('outputs/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_conflict_heatmap(
        results['conflict_matrix'],
        results['task_names'],
        output_dir / 'tox21_improved_heatmap.png',
        title='Tox21 Gradient Conflicts (Improved)',
        cluster=True,
    )

    # Compute within-group vs between-group statistics
    matrix = results['conflict_matrix']
    task_names = results['task_names']

    nr_tasks = [t for t in task_names if t.startswith('NR-')]
    sr_tasks = [t for t in task_names if t.startswith('SR-')]

    def get_pairs(group1, group2):
        vals = []
        for t1 in group1:
            for t2 in group2:
                if t1 != t2:
                    i, j = task_names.index(t1), task_names.index(t2)
                    vals.append(matrix[i, j])
        return vals

    within_nr = get_pairs(nr_tasks, nr_tasks)
    within_sr = get_pairs(sr_tasks, sr_tasks)
    between = get_pairs(nr_tasks, sr_tasks)

    print("\n" + "=" * 60)
    print("MECHANISTIC GROUPING RESULTS")
    print("=" * 60)
    print(f"Within NR group: mean={np.mean(within_nr):+.4f} +/- {np.std(within_nr):.4f}")
    print(f"Within SR group: mean={np.mean(within_sr):+.4f} +/- {np.std(within_sr):.4f}")
    print(f"Between NR-SR:   mean={np.mean(between):+.4f} +/- {np.std(between):.4f}")

    # Statistical test
    from scipy import stats
    t_nr, p_nr = stats.ttest_ind(within_nr, between)
    t_sr, p_sr = stats.ttest_ind(within_sr, between)

    print(f"\nStatistical significance:")
    print(f"  NR within vs between: t={t_nr:.2f}, p={p_nr:.4f} {'***' if p_nr < 0.001 else '**' if p_nr < 0.01 else '*' if p_nr < 0.05 else ''}")
    print(f"  SR within vs between: t={t_sr:.2f}, p={p_sr:.4f} {'***' if p_sr < 0.001 else '**' if p_sr < 0.01 else '*' if p_sr < 0.05 else ''}")

    # Key pairs
    print("\nKey relationships:")
    pairs = [
        ('NR-AR', 'NR-AR-LBD'),
        ('NR-ER', 'NR-ER-LBD'),
        ('SR-ARE', 'SR-MMP'),
        ('SR-ATAD5', 'SR-p53'),
    ]
    for t1, t2 in pairs:
        if t1 in task_names and t2 in task_names:
            i, j = task_names.index(t1), task_names.index(t2)
            print(f"  {t1} vs {t2}: {matrix[i, j]:+.4f}")


if __name__ == '__main__':
    main()
