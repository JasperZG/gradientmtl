#!/usr/bin/env python3
"""
Training script using Tox21 dataset - 12 toxicity endpoints for the SAME molecules.

This provides a proper test of the gradient conflict hypothesis because:
- All molecules have labels for multiple tasks
- Gradients are computed on the same chemical space
- Known relationships exist between toxicity endpoints

Expected patterns:
- Nuclear receptor endpoints (NR-*) should cluster together
- Stress response endpoints (SR-*) should cluster together
- NR vs SR endpoints may show different patterns
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import urllib.request

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


# Tox21 task definitions - 12 toxicity endpoints
TOX21_TASKS = {
    'NR-AR': 'classification',      # Nuclear Receptor - Androgen Receptor
    'NR-AR-LBD': 'classification',  # Nuclear Receptor - AR Ligand Binding Domain
    'NR-AhR': 'classification',     # Nuclear Receptor - Aryl Hydrocarbon Receptor
    'NR-Aromatase': 'classification',  # Nuclear Receptor - Aromatase
    'NR-ER': 'classification',      # Nuclear Receptor - Estrogen Receptor
    'NR-ER-LBD': 'classification',  # Nuclear Receptor - ER Ligand Binding Domain
    'NR-PPAR-gamma': 'classification',  # Nuclear Receptor - PPAR gamma
    'SR-ARE': 'classification',     # Stress Response - Antioxidant Response Element
    'SR-ATAD5': 'classification',   # Stress Response - ATAD5
    'SR-HSE': 'classification',     # Stress Response - Heat Shock Element
    'SR-MMP': 'classification',     # Stress Response - Mitochondrial Membrane Potential
    'SR-p53': 'classification',     # Stress Response - p53
}


def download_tox21():
    """Download Tox21 dataset from MoleculeNet."""
    output_dir = Path('outputs/raw_data')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'tox21.csv'

    if output_path.exists():
        print(f"Tox21 already downloaded: {output_path}")
        return output_path

    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz'
    print(f"Downloading Tox21 from {url}...")

    import gzip
    import io

    request = urllib.request.Request(
        url,
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        compressed = response.read()

    # Decompress
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as f:
        data = f.read()

    with open(output_path, 'wb') as f:
        f.write(data)

    print(f"Downloaded to {output_path}")
    return output_path


def load_tox21(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    """Load Tox21 dataset and extract SMILES + labels."""
    df = pd.read_csv(path)

    # SMILES column
    smiles_col = 'smiles'
    smiles_list = df[smiles_col].tolist()

    # Task columns
    labels = {}
    for task in TOX21_TASKS:
        if task in df.columns:
            # Convert to float, keeping NaN for missing
            values = df[task].values.astype(np.float32)
            labels[task] = values
            n_valid = np.sum(~np.isnan(values))
            n_pos = np.nansum(values)
            print(f"  {task}: {n_valid} labels, {n_pos:.0f} positive ({100*n_pos/n_valid:.1f}%)")

    return smiles_list, labels


def main():
    parser = argparse.ArgumentParser(description='Train MTL on Tox21 with gradient logging')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set seed
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # =========================================================================
    # Load Tox21
    # =========================================================================
    print("\n" + "=" * 60)
    print("Loading Tox21 Dataset (12 toxicity endpoints)")
    print("=" * 60)

    tox21_path = download_tox21()
    smiles_list, raw_labels = load_tox21(tox21_path)

    print(f"\nTotal molecules: {len(smiles_list)}")

    # =========================================================================
    # Preprocess
    # =========================================================================
    print("\n" + "=" * 60)
    print("Preprocessing")
    print("=" * 60)

    preprocessor = MoleculePreprocessor(fp_bits=2048, fp_radius=2)

    valid_smiles, fingerprints, valid_indices = preprocessor.process_smiles_list(
        smiles_list, show_progress=True
    )

    print(f"Valid molecules: {len(valid_smiles)}")
    print(f"Fingerprints shape: {fingerprints.shape}")

    # Filter labels to valid indices
    labels = {}
    for task, values in raw_labels.items():
        labels[task] = values[valid_indices]

    # Check label coverage
    print("\nLabel coverage after filtering:")
    for task in labels:
        n_valid = np.sum(~np.isnan(labels[task]))
        print(f"  {task}: {n_valid} labels ({100*n_valid/len(labels[task]):.1f}%)")

    # Check how many molecules have MULTIPLE task labels
    n_labels_per_mol = np.zeros(len(valid_smiles))
    for task, values in labels.items():
        n_labels_per_mol += ~np.isnan(values)

    print(f"\nMolecules with labels for:")
    for n in [1, 2, 5, 10, 12]:
        count = np.sum(n_labels_per_mol >= n)
        print(f"  {n}+ tasks: {count} ({100*count/len(valid_smiles):.1f}%)")

    # =========================================================================
    # Scaffold split
    # =========================================================================
    print("\n" + "=" * 60)
    print("Scaffold Splitting")
    print("=" * 60)

    train_idx, val_idx, test_idx = scaffold_split(
        valid_smiles,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_seed=seed,
    )

    # =========================================================================
    # Create data loaders
    # =========================================================================
    print("\n" + "=" * 60)
    print("Creating Data Loaders")
    print("=" * 60)

    # Create subset arrays
    train_fp = fingerprints[train_idx]
    val_fp = fingerprints[val_idx]
    test_fp = fingerprints[test_idx]

    train_labels = {task: arr[train_idx] for task, arr in labels.items()}
    val_labels = {task: arr[val_idx] for task, arr in labels.items()}
    test_labels = {task: arr[test_idx] for task, arr in labels.items()}

    # Create datasets
    print("\nTraining dataset:")
    train_dataset = MultiTaskMoleculeDataset(train_fp, train_labels, TOX21_TASKS)

    print("\nValidation dataset:")
    val_dataset = MultiTaskMoleculeDataset(val_fp, val_labels, TOX21_TASKS)

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=multitask_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=multitask_collate_fn,
    )

    # =========================================================================
    # Create model
    # =========================================================================
    print("\n" + "=" * 60)
    print("Creating Model")
    print("=" * 60)

    task_names = list(TOX21_TASKS.keys())
    model = MultiTaskModel(
        task_names=task_names,
        input_dim=2048,
        encoder_hidden_dims=[512, 256],
        head_hidden_dim=128,
        dropout=0.2,
    )

    print(model.summary())

    # =========================================================================
    # Train
    # =========================================================================
    print("\n" + "=" * 60)
    print("Training with Gradient Logging")
    print("=" * 60)

    config = {
        'learning_rate': args.lr,
        'weight_decay': 0.01,
        'epochs': args.epochs,
        'early_stopping_patience': 15,
        'gradient_log_interval': 10,
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

    # =========================================================================
    # Analysis
    # =========================================================================
    print("\n" + "=" * 60)
    print("Gradient Conflict Analysis")
    print("=" * 60)

    # Expected patterns for Tox21:
    # - NR-AR and NR-AR-LBD should be highly correlated (same receptor)
    # - NR-ER and NR-ER-LBD should be highly correlated (same receptor)
    # - NR endpoints may cluster together (nuclear receptor pathway)
    # - SR endpoints may cluster together (stress response pathway)
    expected_patterns = {
        'NR-AR_NR-AR-LBD': 0.5,    # Same receptor, different assays - should be positive
        'NR-ER_NR-ER-LBD': 0.5,    # Same receptor, different assays - should be positive
        'SR-ARE_SR-p53': 0.3,      # Both stress response - might be positive
    }

    print_conflict_summary(
        results['conflict_matrix'],
        results['task_names'],
        expected_patterns,
    )

    # Create visualizations
    output_dir = Path('outputs/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_conflict_heatmap(
        results['conflict_matrix'],
        results['task_names'],
        output_dir / 'tox21_gradient_conflict_heatmap.png',
        title='Tox21 Gradient Conflict Matrix',
        cluster=True,
    )

    # Load history for evolution plot
    from training.gradient_logger import GradientConflictLogger
    gradient_data = GradientConflictLogger.load(
        Path('outputs/gradients/conflict_matrices.npz')
    )

    if len(gradient_data['history']) > 0:
        # Plot NR vs SR pairs
        key_pairs = [
            ('NR-AR', 'NR-AR-LBD'),
            ('NR-ER', 'NR-ER-LBD'),
            ('NR-AR', 'SR-p53'),
            ('SR-ARE', 'SR-p53'),
        ]
        key_pairs = [(a, b) for a, b in key_pairs
                    if a in results['task_names'] and b in results['task_names']]

        if key_pairs:
            plot_conflict_evolution(
                gradient_data['history'],
                gradient_data['task_names'],
                task_pairs=key_pairs,
                output_path=output_dir / 'tox21_gradient_evolution.png',
            )

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nOutputs saved to:")
    print(f"  - Gradient data: outputs/gradients/conflict_matrices.npz")
    print(f"  - Heatmap: outputs/figures/tox21_gradient_conflict_heatmap.png")

    # Print key findings
    matrix = results['conflict_matrix']
    task_names = results['task_names']

    print("\nKey Findings:")
    print("-" * 60)

    # Check NR-AR vs NR-AR-LBD
    if 'NR-AR' in task_names and 'NR-AR-LBD' in task_names:
        i, j = task_names.index('NR-AR'), task_names.index('NR-AR-LBD')
        val = matrix[i, j]
        print(f"NR-AR vs NR-AR-LBD: {val:+.3f} (expected positive - same receptor)")

    # Check NR-ER vs NR-ER-LBD
    if 'NR-ER' in task_names and 'NR-ER-LBD' in task_names:
        i, j = task_names.index('NR-ER'), task_names.index('NR-ER-LBD')
        val = matrix[i, j]
        print(f"NR-ER vs NR-ER-LBD: {val:+.3f} (expected positive - same receptor)")

    # Find strongest correlations
    print("\nStrongest task relationships:")
    pairs = []
    for i in range(len(task_names)):
        for j in range(i+1, len(task_names)):
            pairs.append((task_names[i], task_names[j], matrix[i, j]))

    pairs.sort(key=lambda x: -abs(x[2]))
    for t1, t2, val in pairs[:5]:
        relation = "synergistic" if val > 0.1 else "conflicting" if val < -0.1 else "independent"
        print(f"  {t1} vs {t2}: {val:+.3f} ({relation})")


if __name__ == '__main__':
    main()
