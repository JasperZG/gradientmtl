#!/usr/bin/env python3
"""
Tox21 training with Graph Neural Network encoder.

This experiment tests whether learned molecular representations (via GNN)
show stronger gradient conflict patterns than fixed ECFP fingerprints.

Hypothesis: GNN-learned representations should reveal stronger mechanistic
relationships between tasks because the encoder learns task-relevant features.

Success criteria: NR receptor pairs G > 0.2 (4x improvement over ECFP baseline)
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
import urllib.request
import gzip
import io

from data.graph_preprocessing import MoleculeGraphPreprocessor, get_atom_feature_dim
from data.splitting import scaffold_split
from data.graph_dataset import MultiTaskGraphDataset, graph_collate_fn
from models.gnn_multitask import GNNMultiTaskModel
from training.gnn_trainer import GNNMultiTaskTrainer
from analysis.visualization import (
    plot_conflict_heatmap,
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
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--encoder_type', type=str, default='gcn',
                       choices=['gcn', 'gat'], help='GNN encoder type')
    parser.add_argument('--min_tasks', type=int, default=10,
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

    # Convert SMILES to graphs
    print("\n" + "=" * 60)
    print("Converting SMILES to Molecular Graphs")
    print("=" * 60)

    preprocessor = MoleculeGraphPreprocessor()
    valid_smiles, graphs, valid_indices = preprocessor.process_smiles_list(
        smiles_list, show_progress=True
    )

    # Filter labels to valid indices
    labels = {task: values[valid_indices] for task, values in raw_labels.items()}

    # Filter to molecules with enough task labels
    n_labels_per_mol = np.zeros(len(valid_smiles))
    for task, values in labels.items():
        n_labels_per_mol += ~np.isnan(values)

    mask = n_labels_per_mol >= args.min_tasks
    keep_indices = np.where(mask)[0]

    print(f"\nFiltering to molecules with {args.min_tasks}+ task labels:")
    print(f"  Before: {len(valid_smiles)} molecules")
    print(f"  After: {len(keep_indices)} molecules")

    # Filter data
    graphs = [graphs[i] for i in keep_indices]
    valid_smiles = [valid_smiles[i] for i in keep_indices]
    labels = {task: arr[keep_indices] for task, arr in labels.items()}

    # Check label coverage
    print("\nLabel coverage after filtering:")
    for task in labels:
        n_valid = np.sum(~np.isnan(labels[task]))
        n_pos = np.nansum(labels[task])
        print(f"  {task}: {n_valid} labels ({100*n_valid/len(graphs):.1f}%), "
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

    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    train_labels = {task: arr[train_idx] for task, arr in labels.items()}
    val_labels = {task: arr[val_idx] for task, arr in labels.items()}

    print("\nTraining dataset:")
    train_dataset = MultiTaskGraphDataset(train_graphs, train_labels, TOX21_TASKS)

    print("\nValidation dataset:")
    val_dataset = MultiTaskGraphDataset(val_graphs, val_labels, TOX21_TASKS)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
    )

    # Create GNN model
    print("\n" + "=" * 60)
    print(f"Creating GNN Model ({args.encoder_type.upper()})")
    print("=" * 60)

    atom_feature_dim = get_atom_feature_dim()
    task_names = list(TOX21_TASKS.keys())

    model = GNNMultiTaskModel(
        task_names=task_names,
        atom_feature_dim=atom_feature_dim,
        encoder_type=args.encoder_type,
        encoder_hidden_dims=[256, 256, 256],
        encoder_output_dim=256,
        head_hidden_dim=128,
        dropout=0.3,
    )

    print(model.summary())

    # Train
    print("\n" + "=" * 60)
    print(f"Training GNN ({args.epochs} epochs)")
    print("=" * 60)

    config = {
        'learning_rate': args.lr,
        'weight_decay': 0.01,
        'epochs': args.epochs,
        'early_stopping_patience': 25,
        'gradient_log_interval': 5,
        'gradient_clip_norm': 1.0,
    }

    trainer = GNNMultiTaskTrainer(
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
    print("Gradient Conflict Analysis (GNN)")
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
        output_dir / f'tox21_gnn_{args.encoder_type}_heatmap.png',
        title=f'Tox21 Gradient Conflicts ({args.encoder_type.upper()})',
        cluster=True,
    )

    # Compare with ECFP baseline
    print("\n" + "=" * 60)
    print("COMPARISON: GNN vs ECFP Baseline")
    print("=" * 60)

    matrix = results['conflict_matrix']
    task_names = results['task_names']

    # Load ECFP baseline if available
    ecfp_path = Path('outputs/gradients/conflict_matrices.npz')
    if ecfp_path.exists():
        ecfp_data = np.load(ecfp_path, allow_pickle=True)
        ecfp_matrix = ecfp_data['averaged']
        ecfp_tasks = ecfp_data['task_names'].tolist()

        print("\nKey pair comparisons:")
        pairs = [
            ('NR-AR', 'NR-AR-LBD'),
            ('NR-ER', 'NR-ER-LBD'),
            ('SR-ARE', 'SR-MMP'),
            ('SR-ATAD5', 'SR-p53'),
        ]

        for t1, t2 in pairs:
            if t1 in task_names and t2 in task_names:
                i, j = task_names.index(t1), task_names.index(t2)
                gnn_val = matrix[i, j]

                ecfp_val = None
                if t1 in ecfp_tasks and t2 in ecfp_tasks:
                    ei, ej = ecfp_tasks.index(t1), ecfp_tasks.index(t2)
                    ecfp_val = ecfp_matrix[ei, ej]

                if ecfp_val is not None:
                    improvement = gnn_val / ecfp_val if ecfp_val != 0 else float('inf')
                    print(f"  {t1} vs {t2}:")
                    print(f"    ECFP: {ecfp_val:+.4f}")
                    print(f"    GNN:  {gnn_val:+.4f}")
                    print(f"    Ratio: {improvement:.1f}x")
                else:
                    print(f"  {t1} vs {t2}: GNN = {gnn_val:+.4f}")
    else:
        print("(No ECFP baseline found for comparison)")

    # Mechanistic grouping analysis
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
    print("MECHANISTIC GROUPING RESULTS (GNN)")
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

    # Success criteria check
    print("\n" + "=" * 60)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 60)

    success_threshold = 0.2
    nr_ar_pass = False
    nr_er_pass = False

    if 'NR-AR' in task_names and 'NR-AR-LBD' in task_names:
        i, j = task_names.index('NR-AR'), task_names.index('NR-AR-LBD')
        val = matrix[i, j]
        nr_ar_pass = val > success_threshold
        status = "PASS" if nr_ar_pass else "FAIL"
        print(f"NR-AR vs NR-AR-LBD: {val:+.4f} (threshold: >{success_threshold}) [{status}]")

    if 'NR-ER' in task_names and 'NR-ER-LBD' in task_names:
        i, j = task_names.index('NR-ER'), task_names.index('NR-ER-LBD')
        val = matrix[i, j]
        nr_er_pass = val > success_threshold
        status = "PASS" if nr_er_pass else "FAIL"
        print(f"NR-ER vs NR-ER-LBD: {val:+.4f} (threshold: >{success_threshold}) [{status}]")

    if nr_ar_pass or nr_er_pass:
        print("\n[+] SUCCESS: GNN shows meaningful gradient conflict patterns!")
        print("    Proceed to Experiments 3-6 (transfer learning, selection, PCGrad)")
    else:
        print("\n[!] Signal still weak. Consider:")
        print("    - Trying GAT instead of GCN (--encoder_type gat)")
        print("    - Increasing model capacity")
        print("    - Pivoting to Option B (direct mechanistic modeling)")


if __name__ == '__main__':
    main()
