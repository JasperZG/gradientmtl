#!/usr/bin/env python3
"""
Main training script for Gradient-Based Causal Discovery in Molecular Property Prediction.

Phase 1 Proof of Concept: 5 tasks (BACE, BBBP, ESOL, Lipophilicity, hERG)

Usage:
    python train.py [--config config.yaml]

Expected Results:
    - ESOL vs Lipophilicity: G < -0.5 (physicochemical trade-off)
    - BBBP vs hERG: G > 0.3 (ADME cluster)
    - BACE vs BBBP: G < -0.2 (selectivity vs permeability)
"""

import argparse
from pathlib import Path
import yaml
import torch
import pandas as pd
import numpy as np

from data.download import download_datasets, load_dataset
from data.preprocessing import MoleculePreprocessor
from data.splitting import scaffold_split, verify_scaffold_split
from data.dataset import create_data_loaders
from models.multitask import MultiTaskModel
from training.trainer import MultiTaskTrainer
from analysis.visualization import (
    plot_conflict_heatmap,
    plot_conflict_evolution,
    print_conflict_summary,
)


def main():
    parser = argparse.ArgumentParser(
        description='Train multi-task model with gradient conflict logging'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('config.yaml'),
        help='Path to configuration file'
    )
    args = parser.parse_args()

    # Load config
    print(f"Loading config from {args.config}")
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set random seed for reproducibility
    seed = config['splitting']['random_seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # =========================================================================
    # Step 1: Download and load datasets
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 1: Loading Datasets")
    print("=" * 60)

    raw_data_dir = Path('outputs/raw_data')
    downloaded_files = download_datasets(raw_data_dir, config)

    # Load each dataset
    datasets = {}
    for task_name, file_path in downloaded_files.items():
        task_config = config['tasks'][task_name]
        df = load_dataset(file_path, task_config)
        datasets[task_name] = df
        print(f"Loaded {task_name}: {len(df)} samples")

    # =========================================================================
    # Step 2: Preprocess molecules and compute fingerprints
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Preprocessing Molecules")
    print("=" * 60)

    preprocessor = MoleculePreprocessor(
        fp_bits=config['model']['fingerprint_bits'],
        fp_radius=config['model']['fingerprint_radius'],
    )

    # Collect all unique SMILES across datasets
    all_smiles = set()
    for task_name, df in datasets.items():
        all_smiles.update(df['smiles'].tolist())

    print(f"Total unique SMILES across all tasks: {len(all_smiles)}")

    # Standardize all SMILES and compute fingerprints
    smiles_to_data = {}  # canonical_smiles -> (fingerprint, original_smiles)
    failed_count = 0

    for smi in all_smiles:
        std_smi = preprocessor.standardize_smiles(smi)
        if std_smi is None:
            failed_count += 1
            continue

        if std_smi in smiles_to_data:
            continue

        fp = preprocessor.compute_fingerprint(std_smi)
        if fp is None:
            failed_count += 1
            continue

        smiles_to_data[std_smi] = (fp, smi)

    print(f"Valid molecules: {len(smiles_to_data)}")
    print(f"Failed to parse: {failed_count}")

    # Create unified molecule list
    canonical_smiles_list = list(smiles_to_data.keys())
    smiles_to_idx = {smi: i for i, smi in enumerate(canonical_smiles_list)}
    fingerprints = np.stack([smiles_to_data[smi][0] for smi in canonical_smiles_list])

    print(f"Fingerprint matrix shape: {fingerprints.shape}")

    # Create label matrix (with NaN for missing)
    labels = {}
    task_types = {}

    for task_name, df in datasets.items():
        task_labels = np.full(len(canonical_smiles_list), np.nan, dtype=np.float32)

        for _, row in df.iterrows():
            std_smi = preprocessor.standardize_smiles(row['smiles'])
            if std_smi in smiles_to_idx:
                idx = smiles_to_idx[std_smi]
                task_labels[idx] = row['label']

        n_valid = np.sum(~np.isnan(task_labels))
        print(f"  {task_name}: {n_valid} labels ({100*n_valid/len(task_labels):.1f}%)")

        labels[task_name] = task_labels
        task_types[task_name] = config['tasks'][task_name]['type']

    # =========================================================================
    # Step 3: Scaffold-based splitting
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Scaffold Splitting")
    print("=" * 60)

    train_idx, val_idx, test_idx = scaffold_split(
        canonical_smiles_list,
        train_ratio=config['splitting']['train_ratio'],
        val_ratio=config['splitting']['val_ratio'],
        test_ratio=config['splitting']['test_ratio'],
        random_seed=config['splitting']['random_seed'],
    )

    # Verify no scaffold leakage
    verify_scaffold_split(canonical_smiles_list, train_idx, val_idx, test_idx)

    # =========================================================================
    # Step 4: Create data loaders
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Creating Data Loaders")
    print("=" * 60)

    train_loader, val_loader, test_loader = create_data_loaders(
        fingerprints=fingerprints,
        labels=labels,
        task_types=task_types,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        batch_size=config['training']['batch_size'],
    )

    print(f"\nData loaders created:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val: {len(val_loader)} batches")
    print(f"  Test: {len(test_loader)} batches")

    # =========================================================================
    # Step 5: Create model
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 5: Creating Model")
    print("=" * 60)

    task_names = list(config['tasks'].keys())
    model = MultiTaskModel(
        task_names=task_names,
        input_dim=config['model']['input_dim'],
        encoder_hidden_dims=config['model']['encoder_hidden_dims'],
        head_hidden_dim=config['model']['head_hidden_dim'],
        dropout=config['model']['dropout'],
    )

    print(model.summary())

    # =========================================================================
    # Step 6: Train with gradient logging
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 6: Training with Gradient Logging")
    print("=" * 60)

    trainer = MultiTaskTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        task_types=task_types,
        config=config['training'],
        device=device,
        output_dir=Path('outputs'),
    )

    results = trainer.train()

    # =========================================================================
    # Step 7: Analysis and visualization
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 7: Analysis and Visualization")
    print("=" * 60)

    # Get expected patterns from config
    expected_patterns = config.get('expected_patterns', {})

    # Print detailed summary
    print_conflict_summary(
        results['conflict_matrix'],
        results['task_names'],
        expected_patterns,
    )

    # Create visualizations
    output_dir = Path('outputs/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Heatmap
    plot_conflict_heatmap(
        results['conflict_matrix'],
        results['task_names'],
        output_dir / 'gradient_conflict_heatmap.png',
        cluster=True,
    )

    # Load history for evolution plot
    from training.gradient_logger import GradientConflictLogger
    gradient_data = GradientConflictLogger.load(
        Path('outputs/gradients/conflict_matrices.npz')
    )

    if len(gradient_data['history']) > 0:
        # Plot key pairs
        key_pairs = [
            ('esol', 'lipophilicity'),
            ('bbbp', 'herg'),
            ('bace', 'bbbp'),
        ]
        # Filter to existing tasks
        key_pairs = [(a, b) for a, b in key_pairs
                    if a in results['task_names'] and b in results['task_names']]

        plot_conflict_evolution(
            gradient_data['history'],
            gradient_data['task_names'],
            task_pairs=key_pairs,
            output_path=output_dir / 'gradient_conflict_evolution.png',
            logged_steps=gradient_data.get('logged_steps'),
        )

    # =========================================================================
    # Step 8: Final test evaluation
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 8: Test Set Evaluation")
    print("=" * 60)

    # Quick test evaluation
    model.eval()
    from sklearn.metrics import roc_auc_score

    with torch.no_grad():
        all_preds = {task: [] for task in task_names}
        all_labels = {task: [] for task in task_names}
        all_masks = {task: [] for task in task_names}

        for batch in test_loader:
            fingerprints_batch = batch['fingerprint'].to(device)
            predictions = model(fingerprints_batch)

            for task in task_names:
                all_preds[task].append(predictions[task].cpu())
                all_labels[task].append(batch['labels'][task])
                all_masks[task].append(batch['masks'][task])

    print("\nTest Set Results:")
    for task in task_names:
        preds = torch.cat(all_preds[task]).numpy()
        labels_arr = torch.cat(all_labels[task]).numpy()
        masks = torch.cat(all_masks[task]).numpy()

        mask = masks > 0
        if mask.sum() == 0:
            continue

        preds = preds[mask]
        labels_arr = labels_arr[mask]

        if task_types[task] == 'classification':
            probs = 1 / (1 + np.exp(-preds))
            try:
                auc = roc_auc_score(labels_arr, probs)
                print(f"  {task}: ROC-AUC = {auc:.3f} (n={mask.sum()})")
            except ValueError:
                print(f"  {task}: ROC-AUC = N/A (single class)")
        else:
            rmse = np.sqrt(np.mean((preds - labels_arr) ** 2))
            print(f"  {task}: RMSE = {rmse:.3f} (n={mask.sum()})")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nOutputs saved to:")
    print(f"  - Model checkpoint: outputs/checkpoints/best_model.pt")
    print(f"  - Gradient data: outputs/gradients/conflict_matrices.npz")
    print(f"  - Heatmap: outputs/figures/gradient_conflict_heatmap.png")
    print(f"  - Evolution plot: outputs/figures/gradient_conflict_evolution.png")

    print("\n" + "=" * 60)
    print("HYPOTHESIS VALIDATION SUMMARY")
    print("=" * 60)

    matrix = results['conflict_matrix']
    task_names = results['task_names']

    # Check key hypotheses
    def get_value(t1, t2):
        if t1 in task_names and t2 in task_names:
            i, j = task_names.index(t1), task_names.index(t2)
            return matrix[i, j]
        return None

    checks = [
        ('ESOL vs Lipophilicity', 'esol', 'lipophilicity', '<', -0.5),
        ('BBBP vs hERG', 'bbbp', 'herg', '>', 0.3),
        ('BACE vs BBBP', 'bace', 'bbbp', '<', -0.2),
    ]

    all_pass = True
    for name, t1, t2, op, threshold in checks:
        val = get_value(t1, t2)
        if val is None:
            print(f"  ? {name}: Tasks not found")
            continue

        if op == '<':
            passed = val < threshold
        else:
            passed = val > threshold

        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        all_pass = all_pass and passed

        print(f"  {symbol} {name}: {val:+.3f} (expected {op} {threshold}) [{status}]")

    print()
    if all_pass:
        print("🎉 ALL HYPOTHESES VALIDATED! Core findings confirmed.")
        print("   Gradient conflicts reveal mechanistic task relationships.")
    else:
        print("⚠️  Some hypotheses not validated. Consider:")
        print("   - Training longer (more epochs)")
        print("   - Adjusting architecture")
        print("   - Checking data quality")


if __name__ == '__main__':
    main()
