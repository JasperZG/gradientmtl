#!/usr/bin/env python3
"""
Experiment 3: Transfer Learning Validation (Full Matrix)

Hypothesis: Tasks with positive gradient correlations should benefit from transfer learning.

Full experimental design:
- 12 source tasks × 11 target tasks = 132 directed pairs
- 3 data regimes: n=50, 100, 200 (target task training samples)
- 2 conditions: transfer (pretrained encoder) vs scratch (random init)

Total: 132 × 3 × 2 = 792 independent runs

OPTIMIZED: Uses cached pretrained encoders to avoid redundant pretraining.
  Phase 1: python experiment3_transfer_learning.py --pretrain  (~3 hours, 1 GPU)
  Phase 2: python experiment3_transfer_learning.py --job-index N  (parallel across GPUs)
  Phase 3: python experiment3_transfer_learning.py --aggregate

Uses GNN encoder for stronger gradient signal.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
import urllib.request
import gzip
import io
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.graph_preprocessing import MoleculeGraphPreprocessor, get_atom_feature_dim
from data.splitting import scaffold_split
from data.graph_dataset import MultiTaskGraphDataset
from models.gnn_multitask import GNNMultiTaskModel
from training.gnn_trainer import GNNMultiTaskTrainer


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

TASK_NAMES = list(TOX21_TASKS.keys())
DATA_REGIMES = [50, 100, 200]  # Number of target task training samples
CONDITIONS = ['transfer', 'scratch']


def get_all_transfer_pairs():
    """
    Generate all 132 directed transfer pairs.

    Returns:
        List of (source_task, target_task) tuples
    """
    pairs = []
    for source in TASK_NAMES:
        for target in TASK_NAMES:
            if source != target:
                pairs.append((source, target))
    return pairs


def get_job_config(job_index: int) -> dict:
    """
    Map a SLURM array index to specific experiment configuration.

    Total jobs: 132 pairs × 3 regimes × 2 conditions = 792

    Index mapping:
        job_index = pair_idx * 6 + regime_idx * 2 + condition_idx

    Args:
        job_index: SLURM_ARRAY_TASK_ID (0-791)

    Returns:
        Dict with source_task, target_task, data_regime, condition
    """
    all_pairs = get_all_transfer_pairs()
    n_pairs = len(all_pairs)  # 132
    n_regimes = len(DATA_REGIMES)  # 3
    n_conditions = len(CONDITIONS)  # 2

    # Decode indices
    pair_idx = job_index // (n_regimes * n_conditions)
    remainder = job_index % (n_regimes * n_conditions)
    regime_idx = remainder // n_conditions
    condition_idx = remainder % n_conditions

    source, target = all_pairs[pair_idx]
    regime = DATA_REGIMES[regime_idx]
    condition = CONDITIONS[condition_idx]

    return {
        'source_task': source,
        'target_task': target,
        'data_regime': regime,
        'condition': condition,
        'pair_idx': pair_idx,
        'job_index': job_index,
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


def load_and_preprocess_data(min_tasks=10):
    """Load Tox21 and preprocess to graphs."""
    tox21_path = download_tox21()
    df = pd.read_csv(tox21_path)

    smiles_list = df['smiles'].tolist()
    raw_labels = {}
    for task in TOX21_TASKS:
        if task in df.columns:
            raw_labels[task] = df[task].values.astype(np.float32)

    preprocessor = MoleculeGraphPreprocessor()
    valid_smiles, graphs, valid_indices = preprocessor.process_smiles_list(
        smiles_list, show_progress=True
    )

    labels = {task: values[valid_indices] for task, values in raw_labels.items()}

    n_labels_per_mol = np.zeros(len(valid_smiles))
    for task, values in labels.items():
        n_labels_per_mol += ~np.isnan(values)

    mask = n_labels_per_mol >= min_tasks
    keep_indices = np.where(mask)[0]

    graphs = [graphs[i] for i in keep_indices]
    valid_smiles = [valid_smiles[i] for i in keep_indices]
    labels = {task: arr[keep_indices] for task, arr in labels.items()}

    return valid_smiles, graphs, labels


def load_gradient_matrix(path: Path) -> tuple[np.ndarray, list]:
    """Load gradient conflict matrix from GNN experiment."""
    data = np.load(path, allow_pickle=True)
    matrix = data['averaged']
    task_names = data['task_names'].tolist()
    return matrix, task_names


def subsample_training_data(
    train_idx: np.ndarray,
    labels: dict,
    task_name: str,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """
    Subsample training indices to n_samples for a specific task.

    Stratifies to maintain class balance.
    """
    rng = np.random.RandomState(seed)

    # Ensure train_idx is numpy array
    train_idx = np.array(train_idx)

    # Get valid indices for this task
    task_labels = labels[task_name][train_idx]
    valid_mask = ~np.isnan(task_labels)
    valid_train_idx = train_idx[valid_mask]
    valid_labels = task_labels[valid_mask]

    if len(valid_train_idx) <= n_samples:
        return valid_train_idx

    # Stratified sampling
    pos_idx = valid_train_idx[valid_labels == 1]
    neg_idx = valid_train_idx[valid_labels == 0]

    # Maintain class ratio
    pos_ratio = len(pos_idx) / len(valid_train_idx)
    n_pos = max(1, int(n_samples * pos_ratio))
    n_neg = n_samples - n_pos

    # Sample
    sampled_pos = rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False)
    sampled_neg = rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False)

    subsampled_idx = np.concatenate([sampled_pos, sampled_neg])
    rng.shuffle(subsampled_idx)

    return subsampled_idx


def train_single_task(
    task_name: str,
    graphs: list,
    labels: dict,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    config: dict,
    device: torch.device,
    output_dir: Path,
    pretrained_encoder: dict = None,
    verbose: bool = True,
):
    """Train a model on a single task."""
    single_task_labels = {task_name: labels[task_name]}
    single_task_types = {task_name: 'classification'}

    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    train_labels = {task_name: single_task_labels[task_name][train_idx]}
    val_labels = {task_name: single_task_labels[task_name][val_idx]}

    train_dataset = MultiTaskGraphDataset(train_graphs, train_labels, single_task_types)
    val_dataset = MultiTaskGraphDataset(val_graphs, val_labels, single_task_types)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    atom_feature_dim = get_atom_feature_dim()
    model = GNNMultiTaskModel(
        task_names=[task_name],
        atom_feature_dim=atom_feature_dim,
        encoder_type=config['encoder_type'],
        encoder_hidden_dims=config['encoder_hidden_dims'],
        encoder_output_dim=config['encoder_output_dim'],
        head_hidden_dim=config['head_hidden_dim'],
        dropout=config['dropout'],
    )

    if pretrained_encoder is not None:
        model.encoder.load_state_dict(pretrained_encoder)
        if verbose:
            print(f"  Loaded pretrained encoder weights")

    trainer = GNNMultiTaskTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        task_types=single_task_types,
        config=config,
        device=device,
        output_dir=output_dir,
    )

    results = trainer.train()
    return results, model


def run_single_transfer_experiment(
    source_task: str,
    target_task: str,
    data_regime: int,
    condition: str,
    graphs: list,
    labels: dict,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    gradient_matrix: np.ndarray,
    gradient_task_names: list,
    config: dict,
    device: torch.device,
    output_dir: Path,
    seed: int,
    checkpoint_dir: Path = None,
):
    """
    Run a single transfer learning experiment.

    Args:
        source_task: Task to pretrain on
        target_task: Task to fine-tune on
        data_regime: Number of target task training samples (50, 100, 200)
        condition: 'transfer' or 'scratch'
        ...

    Returns:
        Dict with experiment results
    """
    exp_name = f"{source_task}_to_{target_task}_n{data_regime}_{condition}"
    exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"{'='*60}")

    # Get gradient correlation for this pair
    if source_task in gradient_task_names and target_task in gradient_task_names:
        src_idx = gradient_task_names.index(source_task)
        tgt_idx = gradient_task_names.index(target_task)
        gradient_correlation = gradient_matrix[src_idx, tgt_idx]
    else:
        gradient_correlation = np.nan

    print(f"Gradient correlation G({source_task}, {target_task}) = {gradient_correlation:.4f}")

    # Subsample training data for target task
    subsampled_train_idx = subsample_training_data(
        train_idx, labels, target_task, data_regime, seed
    )
    print(f"Training samples: {len(subsampled_train_idx)} (regime: n={data_regime})")

    if condition == 'transfer':
        # Try to load cached pretrained encoder first
        pretrained_encoder = None
        if checkpoint_dir is not None:
            checkpoint_path = Path(checkpoint_dir) / f"{source_task}_encoder.pt"
            if checkpoint_path.exists():
                print(f"\n[1/2] Loading cached encoder from {checkpoint_path}")
                pretrained_encoder = torch.load(checkpoint_path, map_location=device)

        # If no cached checkpoint, train from scratch
        if pretrained_encoder is None:
            print(f"\n[1/2] Pretraining on {source_task} (full data)...")
            _, source_model = train_single_task(
                task_name=source_task,
                graphs=graphs,
                labels=labels,
                train_idx=train_idx,
                val_idx=val_idx,
                config=config,
                device=device,
                output_dir=exp_dir / 'pretrain',
            )
            pretrained_encoder = source_model.encoder.state_dict()

        # Step 2: Fine-tune on target task with pretrained encoder
        print(f"\n[2/2] Fine-tuning on {target_task} (n={data_regime})...")
        finetune_config = config.copy()
        finetune_config['learning_rate'] = config['learning_rate'] / 10

        target_results, _ = train_single_task(
            task_name=target_task,
            graphs=graphs,
            labels=labels,
            train_idx=subsampled_train_idx,
            val_idx=val_idx,
            config=finetune_config,
            device=device,
            output_dir=exp_dir / 'finetune',
            pretrained_encoder=pretrained_encoder,
        )

    else:  # scratch
        # Train from scratch on target task
        print(f"\n[1/1] Training from scratch on {target_task} (n={data_regime})...")
        target_results, _ = train_single_task(
            task_name=target_task,
            graphs=graphs,
            labels=labels,
            train_idx=subsampled_train_idx,
            val_idx=val_idx,
            config=config,
            device=device,
            output_dir=exp_dir / 'scratch',
        )

    # Extract metrics
    target_auc = target_results['final_metrics'][target_task]['roc_auc']

    result = {
        'source_task': source_task,
        'target_task': target_task,
        'data_regime': data_regime,
        'condition': condition,
        'gradient_correlation': float(gradient_correlation),
        'target_auc': float(target_auc),
        'best_val_loss': float(target_results['best_val_loss']),
        'seed': seed,
    }

    print(f"\nResult: {target_task} AUC = {target_auc:.4f}")

    # Save result
    with open(exp_dir / 'result.json', 'w') as f:
        json.dump(result, f, indent=2)

    return result


def aggregate_transfer_results(output_dir: Path):
    """Aggregate all transfer learning results from individual jobs."""
    print("\n" + "=" * 60)
    print("Aggregating Transfer Learning Results")
    print("=" * 60)

    results = []

    # Find all result.json files in subdirectories
    for result_file in output_dir.glob('**/result.json'):
        with open(result_file) as f:
            result = json.load(f)
            results.append(result)

    if not results:
        print("No results found!")
        return

    print(f"Found {len(results)} experiment results")

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save combined results
    df.to_csv(output_dir / 'all_transfer_results.csv', index=False)
    print(f"Saved combined results to {output_dir}/all_transfer_results.csv")

    # Analyze: transfer vs scratch by gradient correlation
    print("\n" + "=" * 60)
    print("TRANSFER VS SCRATCH ANALYSIS")
    print("=" * 60)

    for regime in DATA_REGIMES:
        regime_df = df[df['data_regime'] == regime]
        if len(regime_df) == 0:
            continue

        transfer_df = regime_df[regime_df['condition'] == 'transfer']
        scratch_df = regime_df[regime_df['condition'] == 'scratch']

        if len(transfer_df) > 0 and len(scratch_df) > 0:
            # Merge on source/target pair
            merged = transfer_df.merge(
                scratch_df[['source_task', 'target_task', 'target_auc']],
                on=['source_task', 'target_task'],
                suffixes=('_transfer', '_scratch')
            )

            if len(merged) > 0:
                merged['improvement'] = merged['target_auc_transfer'] - merged['target_auc_scratch']

                # Correlation with gradient similarity
                from scipy import stats
                corr, p_value = stats.pearsonr(
                    merged['gradient_correlation'].dropna(),
                    merged['improvement'].loc[merged['gradient_correlation'].notna()]
                )

                print(f"\nData regime n={regime}:")
                print(f"  Pairs analyzed: {len(merged)}")
                print(f"  Mean improvement: {merged['improvement'].mean():.4f}")
                print(f"  Correlation (G vs improvement): {corr:.4f} (p={p_value:.4f})")

                # How often does high G predict positive transfer?
                high_g = merged[merged['gradient_correlation'] > 0.1]
                if len(high_g) > 0:
                    pct_positive = (high_g['improvement'] > 0).mean() * 100
                    print(f"  High-G pairs (G>0.1): {len(high_g)}, positive transfer: {pct_positive:.1f}%")

    print(f"\nAggregation complete.")


def pretrain_all_sources(graphs, labels, train_idx, val_idx, config, device, checkpoint_dir, seed):
    """
    Phase 1: Pretrain all 12 source models and save encoder checkpoints.
    Only needs to run ONCE. Takes ~3 hours on single GPU.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("PHASE 1: Pretraining all source models")
    print("=" * 60)

    for i, task in enumerate(TASK_NAMES):
        checkpoint_path = checkpoint_dir / f"{task}_encoder.pt"

        if checkpoint_path.exists():
            print(f"\n[{i+1}/12] {task}: checkpoint exists, skipping")
            continue

        print(f"\n[{i+1}/12] Pretraining on {task}...")

        torch.manual_seed(seed)
        np.random.seed(seed)

        _, model = train_single_task(
            task_name=task,
            graphs=graphs,
            labels=labels,
            train_idx=train_idx,
            val_idx=val_idx,
            config=config,
            device=device,
            output_dir=checkpoint_dir / f"{task}_pretrain",
        )

        torch.save(model.encoder.state_dict(), checkpoint_path)
        print(f"  Saved checkpoint to {checkpoint_path}")

    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE: All source models pretrained")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Transfer Learning Validation (Full Matrix)')
    parser.add_argument('--pretrain', action='store_true',
                       help='Phase 1: Pretrain all 12 source models and cache checkpoints')
    parser.add_argument('--run-all', action='store_true',
                       help='Phase 2: Run all 792 jobs sequentially')
    parser.add_argument('--job-index', type=int, default=None,
                       help='SLURM array task ID (0-791).')
    parser.add_argument('--source-task', type=str, default=None,
                       help='Source task (overrides job-index)')
    parser.add_argument('--target-task', type=str, default=None,
                       help='Target task (overrides job-index)')
    parser.add_argument('--data-regime', type=int, default=None,
                       help='Data regime n (overrides job-index)')
    parser.add_argument('--condition', type=str, default=None,
                       choices=['transfer', 'scratch'],
                       help='Condition (overrides job-index)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--encoder-type', type=str, default='gcn')
    parser.add_argument('--min-tasks', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gradient-matrix', type=str,
                       default='outputs/gradients/gnn_conflict_matrices.npz')
    parser.add_argument('--output-dir', type=str, default='outputs/experiment3')
    parser.add_argument('--checkpoint-dir', type=str, default='outputs/experiment3/checkpoints',
                       help='Directory for cached pretrained encoders')
    parser.add_argument('--aggregate', action='store_true',
                       help='Aggregate all results instead of running experiment')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir)

    # Aggregate mode
    if args.aggregate:
        aggregate_transfer_results(output_dir)
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load gradient matrix
    print("\nLoading gradient conflict matrix...")
    gradient_path = Path(args.gradient_matrix)
    if gradient_path.exists():
        gradient_matrix, gradient_task_names = load_gradient_matrix(gradient_path)
        print(f"Loaded gradient matrix for {len(gradient_task_names)} tasks")
    else:
        print(f"WARNING: Gradient matrix not found at {gradient_path}")
        gradient_matrix = np.zeros((12, 12))
        gradient_task_names = TASK_NAMES

    # Load data
    print("\nLoading and preprocessing data...")
    valid_smiles, graphs, labels = load_and_preprocess_data(args.min_tasks)
    print(f"Loaded {len(graphs)} molecules")

    train_idx, val_idx, test_idx = scaffold_split(
        valid_smiles, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=args.seed
    )

    # Training config
    config = {
        'learning_rate': args.lr,
        'weight_decay': 0.01,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'early_stopping_patience': 15,
        'gradient_log_interval': 50,  # Less frequent logging for speed
        'gradient_clip_norm': 1.0,
        'encoder_type': args.encoder_type,
        'encoder_hidden_dims': [256, 256, 256],
        'encoder_output_dim': 256,
        'head_hidden_dim': 128,
        'dropout': 0.3,
    }

    # Phase 1: Pretrain all source models
    if args.pretrain:
        pretrain_all_sources(
            graphs=graphs,
            labels=labels,
            train_idx=train_idx,
            val_idx=val_idx,
            config=config,
            device=device,
            checkpoint_dir=checkpoint_dir,
            seed=args.seed,
        )
        return

    # Determine what to run
    if args.source_task and args.target_task and args.data_regime and args.condition:
        # Manual override
        jobs_to_run = [{
            'source_task': args.source_task,
            'target_task': args.target_task,
            'data_regime': args.data_regime,
            'condition': args.condition,
        }]
    elif args.job_index is not None:
        # Single job from array
        jobs_to_run = [get_job_config(args.job_index)]
    elif args.run_all:
        # Run all 792 jobs sequentially
        print("Running all 792 jobs sequentially...")
        jobs_to_run = [get_job_config(i) for i in range(792)]
    else:
        print("Usage:")
        print("  Step 1: python experiment3_transfer_learning.py --pretrain")
        print("  Step 2: python experiment3_transfer_learning.py --run-all")
        print("          OR python experiment3_transfer_learning.py --job-index N")
        print("  Step 3: python experiment3_transfer_learning.py --aggregate")
        return

    # Run experiments
    all_results = []
    for job_config in jobs_to_run:
        result = run_single_transfer_experiment(
            source_task=job_config['source_task'],
            target_task=job_config['target_task'],
            data_regime=job_config['data_regime'],
            condition=job_config['condition'],
            graphs=graphs,
            labels=labels,
            train_idx=train_idx,
            val_idx=val_idx,
            gradient_matrix=gradient_matrix,
            gradient_task_names=gradient_task_names,
            config=config,
            device=device,
            output_dir=output_dir,
            seed=args.seed,
            checkpoint_dir=checkpoint_dir,
        )
        all_results.append(result)

    # Save combined results (for single-job runs)
    if len(all_results) == 1:
        print(f"\nResult saved to {output_dir}/")
    else:
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(output_dir / 'all_transfer_results.csv', index=False)
        print(f"\nAll results saved to {output_dir}/all_transfer_results.csv")


if __name__ == '__main__':
    main()
