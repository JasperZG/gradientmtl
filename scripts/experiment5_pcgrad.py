#!/usr/bin/env python3
"""
Experiment 5: PCGrad Validation

Validates that gradient conflicts detected by our G matrix are real
by showing PCGrad helps conflicting pairs but not synergistic ones.

Hypothesis:
- High-conflict pairs (G < -0.2): PCGrad should HELP (reduce interference)
- Synergistic pairs (G > 0.2): PCGrad should NOT help (no interference to fix)

This validates that our gradient conflict measurements are meaningful.

SLURM Array Job:
    - 15 jobs: 5 high-conflict pairs + 5 synergistic pairs + 5 random pairs
    - Each job: train with PCGrad ON vs OFF
    - Output: improvement delta for each condition
"""

import sys
import os
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, mean_squared_error

# Project imports
from data.graph_dataset import MultiTaskGraphDataset
from data.graph_preprocessing import MoleculeGraphPreprocessor, get_atom_feature_dim
from models.gnn_encoder import GCNEncoder
from models.heads import TaskHead
from models.gnn_multitask import GNNMultiTaskModel
from training.pcgrad import PCGrad
from torch_geometric.loader import DataLoader as PyGDataLoader


# =============================================================================
# Task Pair Definitions
# =============================================================================

# Tox21 task names
TOX21_TASKS = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
    'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]

# Pre-selected pairs based on expected gradient relationships
# These are hypothesized based on biological/chemical knowledge
HIGH_CONFLICT_PAIRS = [
    # Different mechanisms expected to conflict
    ('NR-AR', 'NR-ER'),           # Androgen vs Estrogen receptors
    ('NR-AR', 'NR-Aromatase'),    # AR signaling vs aromatase inhibition
    ('NR-PPAR-gamma', 'SR-MMP'),  # Metabolic vs stress response
    ('NR-AhR', 'SR-HSE'),         # Xenobiotic vs heat shock
    ('NR-Aromatase', 'SR-ARE'),   # Enzyme vs oxidative stress
]

SYNERGISTIC_PAIRS = [
    # Same receptor family - should be synergistic
    ('NR-AR', 'NR-AR-LBD'),       # Same receptor, different binding sites
    ('NR-ER', 'NR-ER-LBD'),       # Same receptor, different binding sites
    ('SR-ARE', 'SR-HSE'),         # Both stress response
    ('SR-MMP', 'SR-p53'),         # Both stress/apoptosis related
    ('SR-ATAD5', 'SR-p53'),       # Both DNA damage related
]

# Random control pairs
RANDOM_PAIRS = [
    ('NR-AR-LBD', 'SR-MMP'),
    ('NR-AhR', 'NR-ER-LBD'),
    ('NR-PPAR-gamma', 'SR-ARE'),
    ('NR-Aromatase', 'SR-ATAD5'),
    ('NR-ER', 'SR-HSE'),
]

ALL_PAIRS = HIGH_CONFLICT_PAIRS + SYNERGISTIC_PAIRS + RANDOM_PAIRS
PAIR_CATEGORIES = (
    ['high_conflict'] * len(HIGH_CONFLICT_PAIRS) +
    ['synergistic'] * len(SYNERGISTIC_PAIRS) +
    ['random'] * len(RANDOM_PAIRS)
)


def get_job_config(job_index: int) -> dict:
    """
    Map SLURM array index to experiment configuration.

    Jobs 0-14: Each pair trained with PCGrad ON and OFF (both conditions run in same job)
    """
    if job_index >= len(ALL_PAIRS):
        raise ValueError(f"Job index {job_index} out of range (max {len(ALL_PAIRS)-1})")

    task1, task2 = ALL_PAIRS[job_index]
    category = PAIR_CATEGORIES[job_index]

    return {
        'task1': task1,
        'task2': task2,
        'category': category,
        'job_index': job_index
    }


# =============================================================================
# Model and Training
# =============================================================================

class TwoTaskModel(nn.Module):
    """Simple two-task model for PCGrad validation."""

    def __init__(self, task_names: list, atom_feature_dim: int = 37, hidden_dim: int = 256):
        super().__init__()
        self.task_names = task_names

        # Shared GCN encoder
        self.encoder = GCNEncoder(
            input_dim=atom_feature_dim,
            hidden_dims=[hidden_dim, hidden_dim, hidden_dim],
            output_dim=hidden_dim,
            dropout=0.2
        )

        # Task-specific heads
        self.heads = nn.ModuleDict({
            task: TaskHead(
                input_dim=hidden_dim,
                hidden_dim=128
            )
            for task in task_names
        })

    def forward(self, batch):
        """Forward pass returning dict of task outputs."""
        # Encode - GCNEncoder expects the full batch object
        graph_embeddings = self.encoder(batch)

        # Task heads
        outputs = {}
        for task in self.task_names:
            outputs[task] = self.heads[task](graph_embeddings)

        return outputs


def train_two_task_model(
    task1: str,
    task2: str,
    use_pcgrad: bool,
    train_dataset,
    val_dataset,
    device: str,
    atom_feature_dim: int = 37,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 10
) -> dict:
    """
    Train a two-task model with or without PCGrad.

    Returns:
        Dict with training results and metrics
    """
    task_names = [task1, task2]

    # Create data loaders using PyG DataLoader (handles batching automatically)
    train_loader = PyGDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = PyGDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # Create model with correct atom feature dimension
    model = TwoTaskModel(task_names, atom_feature_dim=atom_feature_dim).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    # PCGrad wrapper if needed
    if use_pcgrad:
        pcgrad = PCGrad(optimizer)
        shared_params = list(model.encoder.parameters())
        head_params = {task: list(model.heads[task].parameters()) for task in task_names}
    else:
        pcgrad = None

    # Get task indices
    task_idx = {task: train_dataset.task_names.index(task) for task in task_names}

    # Training loop
    best_val_auc = 0
    patience_counter = 0
    train_losses = []
    val_aucs = []

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_losses = []

        for batch in train_loader:
            batch = batch.to(device)

            if use_pcgrad:
                pcgrad.zero_grad()
            else:
                optimizer.zero_grad()

            # Forward pass
            outputs = model(batch)

            # Handle PyG batching - reshape if needed
            batch_size = batch.num_graphs
            n_tasks = len(task_names)
            y_tensor = batch.y
            mask_tensor = batch.mask
            if y_tensor.dim() == 1:
                y_tensor = y_tensor.view(batch_size, n_tasks)
                mask_tensor = mask_tensor.view(batch_size, n_tasks)

            # Compute per-task losses
            task_losses = {}
            for task in task_names:
                idx = task_idx[task]
                labels = y_tensor[:, idx]
                masks = mask_tensor[:, idx]

                if masks.sum() == 0:
                    continue

                pred = outputs[task][masks.bool()].squeeze()
                target = labels[masks.bool()]

                if len(pred) == 0:
                    continue

                loss = nn.functional.binary_cross_entropy_with_logits(
                    pred, target.float()
                )
                task_losses[task] = loss

            if len(task_losses) == 0:
                continue

            # Backward
            if use_pcgrad:
                pcgrad.backward(
                    task_losses,
                    shared_params=shared_params,
                    head_params=head_params
                )
                pcgrad.step()
            else:
                total_loss = sum(task_losses.values())
                total_loss.backward()
                optimizer.step()

            epoch_losses.append(sum(l.item() for l in task_losses.values()))

        avg_train_loss = np.mean(epoch_losses) if epoch_losses else 0
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        task_preds = {task: [] for task in task_names}
        task_labels = {task: [] for task in task_names}
        n_tasks_val = len(task_names)

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                outputs = model(batch)

                # Handle PyG batching - reshape if needed
                batch_size = batch.num_graphs
                y_tensor = batch.y
                mask_tensor = batch.mask
                if y_tensor.dim() == 1:
                    y_tensor = y_tensor.view(batch_size, n_tasks_val)
                    mask_tensor = mask_tensor.view(batch_size, n_tasks_val)

                for task in task_names:
                    idx = task_idx[task]
                    labels = y_tensor[:, idx]
                    masks = mask_tensor[:, idx]

                    if masks.sum() == 0:
                        continue

                    pred = torch.sigmoid(outputs[task][masks.bool()].squeeze())
                    target = labels[masks.bool()]

                    task_preds[task].extend(pred.cpu().numpy())
                    task_labels[task].extend(target.cpu().numpy())

        # Compute AUC for each task
        task_aucs = {}
        for task in task_names:
            if len(task_preds[task]) > 0 and len(np.unique(task_labels[task])) > 1:
                try:
                    auc = roc_auc_score(task_labels[task], task_preds[task])
                    task_aucs[task] = auc
                except:
                    task_aucs[task] = 0.5
            else:
                task_aucs[task] = 0.5

        avg_auc = np.mean(list(task_aucs.values()))
        val_aucs.append(avg_auc)

        # Early stopping
        if avg_auc > best_val_auc:
            best_val_auc = avg_auc
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    return {
        'best_val_auc': best_val_auc,
        'final_val_auc': avg_auc,
        'task_aucs': task_aucs,
        'train_losses': train_losses,
        'val_aucs': val_aucs,
        'epochs_trained': epoch + 1
    }


# =============================================================================
# Main Experiment
# =============================================================================

def load_tox21_graphs(data_dir: Path, min_tasks: int = 10):
    """Load Tox21 data and convert to graphs."""
    import pandas as pd
    import urllib.request
    import gzip
    import io

    # Download if needed
    tox21_path = data_dir / 'tox21.csv'
    if not tox21_path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz'
        print(f"Downloading Tox21...")
        request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(request, timeout=60) as response:
            compressed = response.read()
        with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as f:
            data = f.read()
        with open(tox21_path, 'wb') as f:
            f.write(data)

    # Load CSV
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

    # Filter labels to valid indices
    labels = {task: values[valid_indices] for task, values in raw_labels.items()}

    # Filter to molecules with enough labels
    n_labels_per_mol = np.zeros(len(valid_smiles))
    for task, values in labels.items():
        n_labels_per_mol += ~np.isnan(values)

    mask = n_labels_per_mol >= min_tasks
    keep_indices = np.where(mask)[0]

    graphs = [graphs[i] for i in keep_indices]
    labels = {task: arr[keep_indices] for task, arr in labels.items()}

    return graphs, labels, preprocessor.atom_feature_dim


def run_pcgrad_experiment(config: dict, seed: int = 42, output_dir: str = 'outputs/pcgrad'):
    """
    Run PCGrad validation for a single task pair.

    Trains the pair with PCGrad ON and OFF, compares performance.
    """
    os.makedirs(output_dir, exist_ok=True)

    task1 = config['task1']
    task2 = config['task2']
    category = config['category']

    print(f"\n{'='*60}")
    print(f"Experiment 5: PCGrad Validation")
    print(f"{'='*60}")
    print(f"Task pair: {task1} vs {task2}")
    print(f"Category: {category}")
    print(f"Seed: {seed}")

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load dataset
    print(f"\nLoading Tox21 dataset...")
    data_dir = project_root / 'outputs' / 'raw_data'

    try:
        graphs, labels, atom_feature_dim = load_tox21_graphs(data_dir)
        print(f"Loaded {len(graphs)} molecules with atom_feature_dim={atom_feature_dim}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Create task types dict for the two tasks we care about
    task_types = {task1: 'classification', task2: 'classification'}
    two_task_labels = {task1: labels[task1], task2: labels[task2]}

    # Create full dataset
    full_dataset = MultiTaskGraphDataset(graphs, two_task_labels, task_types)

    # Simple train/val split (80/20)
    n_samples = len(full_dataset)
    indices = list(range(n_samples))
    np.random.shuffle(indices)

    train_size = int(0.8 * n_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    # Make subset datasets inherit task_names
    train_dataset.task_names = full_dataset.task_names
    val_dataset.task_names = full_dataset.task_names

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    results = {
        'task1': task1,
        'task2': task2,
        'category': category,
        'seed': seed,
        'n_train': len(train_dataset),
        'n_val': len(val_dataset)
    }

    # Train WITHOUT PCGrad (baseline)
    print(f"\n{'='*40}")
    print("Training WITHOUT PCGrad (baseline)...")
    print(f"{'='*40}")

    torch.manual_seed(seed)  # Reset seed for fair comparison
    np.random.seed(seed)

    start_time = time.time()
    baseline_results = train_two_task_model(
        task1=task1,
        task2=task2,
        use_pcgrad=False,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        atom_feature_dim=atom_feature_dim
    )
    baseline_time = time.time() - start_time

    print(f"  Best AUC: {baseline_results['best_val_auc']:.4f}")
    print(f"  Task AUCs: {baseline_results['task_aucs']}")
    print(f"  Epochs: {baseline_results['epochs_trained']}")
    print(f"  Time: {baseline_time:.1f}s")

    results['baseline'] = {
        'best_auc': baseline_results['best_val_auc'],
        'task_aucs': baseline_results['task_aucs'],
        'epochs': baseline_results['epochs_trained'],
        'time': baseline_time
    }

    # Train WITH PCGrad
    print(f"\n{'='*40}")
    print("Training WITH PCGrad...")
    print(f"{'='*40}")

    torch.manual_seed(seed)  # Reset seed for fair comparison
    np.random.seed(seed)

    start_time = time.time()
    pcgrad_results = train_two_task_model(
        task1=task1,
        task2=task2,
        use_pcgrad=True,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        atom_feature_dim=atom_feature_dim
    )
    pcgrad_time = time.time() - start_time

    print(f"  Best AUC: {pcgrad_results['best_val_auc']:.4f}")
    print(f"  Task AUCs: {pcgrad_results['task_aucs']}")
    print(f"  Epochs: {pcgrad_results['epochs_trained']}")
    print(f"  Time: {pcgrad_time:.1f}s")

    results['pcgrad'] = {
        'best_auc': pcgrad_results['best_val_auc'],
        'task_aucs': pcgrad_results['task_aucs'],
        'epochs': pcgrad_results['epochs_trained'],
        'time': pcgrad_time
    }

    # Compute improvement
    improvement = pcgrad_results['best_val_auc'] - baseline_results['best_val_auc']
    results['pcgrad_improvement'] = improvement
    results['pcgrad_relative_improvement'] = improvement / max(baseline_results['best_val_auc'], 0.001)

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Category: {category}")
    print(f"Baseline AUC: {baseline_results['best_val_auc']:.4f}")
    print(f"PCGrad AUC:   {pcgrad_results['best_val_auc']:.4f}")
    print(f"Improvement:  {improvement:+.4f} ({results['pcgrad_relative_improvement']*100:+.2f}%)")

    if category == 'high_conflict':
        if improvement > 0.01:
            print("✓ PCGrad HELPS as expected for high-conflict pair")
        else:
            print("✗ PCGrad did NOT help for high-conflict pair")
    elif category == 'synergistic':
        if improvement < 0.01:
            print("✓ PCGrad does NOT help as expected for synergistic pair")
        else:
            print("? PCGrad unexpectedly helps synergistic pair")

    # Save results
    output_file = os.path.join(
        output_dir,
        f"pcgrad_{task1}_{task2}_seed{seed}.json"
    )
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


def aggregate_results(output_dir: str = 'outputs/pcgrad'):
    """Aggregate all PCGrad validation results."""
    results_by_category = {
        'high_conflict': [],
        'synergistic': [],
        'random': []
    }

    # Load all result files
    for f in Path(output_dir).glob('pcgrad_*.json'):
        with open(f) as fp:
            result = json.load(fp)
            category = result['category']
            if category in results_by_category:
                results_by_category[category].append(result)

    print("\n" + "="*60)
    print("PCGrad Validation: Aggregated Results")
    print("="*60)

    for category, results in results_by_category.items():
        if not results:
            continue

        improvements = [r['pcgrad_improvement'] for r in results]
        avg_improvement = np.mean(improvements)
        std_improvement = np.std(improvements)

        n_helped = sum(1 for imp in improvements if imp > 0.01)

        print(f"\n{category.upper()} pairs ({len(results)} pairs):")
        print(f"  Average improvement: {avg_improvement:+.4f} ± {std_improvement:.4f}")
        print(f"  Pairs where PCGrad helped: {n_helped}/{len(results)}")

        for r in results:
            print(f"    {r['task1']} vs {r['task2']}: {r['pcgrad_improvement']:+.4f}")

    # Statistical test: high-conflict should have higher improvement than synergistic
    hc_improvements = [r['pcgrad_improvement'] for r in results_by_category['high_conflict']]
    syn_improvements = [r['pcgrad_improvement'] for r in results_by_category['synergistic']]

    if hc_improvements and syn_improvements:
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(hc_improvements, syn_improvements)

        print(f"\nStatistical comparison:")
        print(f"  High-conflict avg: {np.mean(hc_improvements):+.4f}")
        print(f"  Synergistic avg:   {np.mean(syn_improvements):+.4f}")
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value: {p_value:.4f}")

        if p_value < 0.05 and np.mean(hc_improvements) > np.mean(syn_improvements):
            print("  ✓ VALIDATED: PCGrad helps more for high-conflict than synergistic pairs")
        else:
            print("  ✗ Not statistically significant")


def main():
    parser = argparse.ArgumentParser(
        description='Experiment 5: PCGrad Validation'
    )
    parser.add_argument('--job-index', type=int, default=None,
                        help='SLURM array task ID (0-14)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output-dir', type=str, default='outputs/pcgrad',
                        help='Output directory')
    parser.add_argument('--aggregate', action='store_true',
                        help='Aggregate all results instead of running experiment')

    args = parser.parse_args()

    if args.aggregate:
        aggregate_results(args.output_dir)
        return

    # Get job config from environment or argument
    job_index = args.job_index
    if job_index is None:
        job_index = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))

    config = get_job_config(job_index)

    # Run experiment
    run_pcgrad_experiment(
        config=config,
        seed=args.seed,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
