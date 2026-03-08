#!/usr/bin/env python3
"""
Deep Analysis 5: Transfer Learning Validation

Question: Do gradient conflicts predict transfer learning success?
Method:
  1. Run 132 pairwise transfer experiments on Tox21 (12 tasks, 12*11 pairs)
  2. Compute correlation between gradient similarity and transfer gain
Expected: r > 0.5 between gradient similarity and transfer gain
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
import json
from scipy import stats
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from data.graph_preprocessing import MoleculeGraphPreprocessor, get_atom_feature_dim
from data.splitting import scaffold_split
from data.graph_dataset import MultiTaskGraphDataset
from models.gnn_multitask import GNNMultiTaskModel
from training.losses import MultiTaskLoss


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


def load_gradient_matrix(search_paths):
    """Load gradient matrix from saved files."""
    for path in search_paths:
        if path.exists():
            try:
                data = np.load(path, allow_pickle=True)
                if 'conflict_matrix' in data:
                    return data['conflict_matrix'], data['task_names'].tolist()
                elif 'averaged' in data:
                    return data['averaged'], data['task_names'].tolist()
            except:
                pass
    return None, None


def train_single_task_model(model, loader, target_task, task_names, device, epochs=30, lr=1e-3):
    """Train model on single source task."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    for epoch in range(epochs):
        model.train()
        for batch_graph in loader:
            batch_graph = batch_graph.to(device)

            labels_tensor = batch_graph.y
            masks_tensor = batch_graph.mask
            batch_size = batch_graph.num_graphs
            n_tasks = len(task_names)

            if labels_tensor.dim() == 1:
                labels_tensor = labels_tensor.view(batch_size, n_tasks)
                masks_tensor = masks_tensor.view(batch_size, n_tasks)

            task_idx = task_names.index(target_task)
            labels = labels_tensor[:, task_idx]
            masks = masks_tensor[:, task_idx]

            # Only train on samples with labels for this task
            if masks.sum() == 0:
                continue

            optimizer.zero_grad()
            predictions = model(batch_graph)

            # BCE loss for single task
            pred = predictions[target_task][masks > 0]
            target = labels[masks > 0]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


def evaluate_task(model, loader, target_task, task_names, device):
    """Evaluate model on target task."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_graph in loader:
            batch_graph = batch_graph.to(device)

            labels_tensor = batch_graph.y
            masks_tensor = batch_graph.mask
            batch_size = batch_graph.num_graphs
            n_tasks = len(task_names)

            if labels_tensor.dim() == 1:
                labels_tensor = labels_tensor.view(batch_size, n_tasks)
                masks_tensor = masks_tensor.view(batch_size, n_tasks)

            task_idx = task_names.index(target_task)
            labels = labels_tensor[:, task_idx]
            masks = masks_tensor[:, task_idx]

            predictions = model(batch_graph)
            pred = predictions[target_task]

            # Filter to valid samples
            valid_idx = masks > 0
            all_preds.extend(pred[valid_idx].cpu().numpy())
            all_labels.extend(labels[valid_idx].cpu().numpy())

    if len(all_preds) == 0:
        return 0.5

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Convert to probabilities
    probs = 1 / (1 + np.exp(-all_preds))

    try:
        if len(np.unique(all_labels)) > 1:
            return roc_auc_score(all_labels, probs)
        else:
            return 0.5
    except:
        return 0.5


def fine_tune_model(model, loader, target_task, task_names, device, epochs=10, lr=1e-4):
    """Fine-tune pretrained model on target task."""
    # Only fine-tune the task head, freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False

    head_params = model.heads[target_task].parameters()
    optimizer = torch.optim.Adam(head_params, lr=lr)

    for epoch in range(epochs):
        model.train()
        for batch_graph in loader:
            batch_graph = batch_graph.to(device)

            labels_tensor = batch_graph.y
            masks_tensor = batch_graph.mask
            batch_size = batch_graph.num_graphs
            n_tasks = len(task_names)

            if labels_tensor.dim() == 1:
                labels_tensor = labels_tensor.view(batch_size, n_tasks)
                masks_tensor = masks_tensor.view(batch_size, n_tasks)

            task_idx = task_names.index(target_task)
            labels = labels_tensor[:, task_idx]
            masks = masks_tensor[:, task_idx]

            if masks.sum() == 0:
                continue

            optimizer.zero_grad()
            predictions = model(batch_graph)

            pred = predictions[target_task][masks > 0]
            target = labels[masks > 0]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, target)

            loss.backward()
            optimizer.step()

    # Unfreeze for next use
    for param in model.encoder.parameters():
        param.requires_grad = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_epochs', type=int, default=30)
    parser.add_argument('--finetune_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_pairs', type=int, default=None,
                       help='Number of transfer pairs to test (default: all 132)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load gradient matrix
    print("\n" + "=" * 60)
    print("Loading Gradient Matrix")
    print("=" * 60)

    search_paths = [
        Path('outputs/tox21_gnn_gcn/gradient_matrices.npz'),
        Path('outputs/gradients/gnn_conflict_matrices.npz'),
        Path('outputs/gradients/conflict_matrices.npz'),
    ]

    G, g_task_names = load_gradient_matrix(search_paths)

    if G is None:
        print("No gradient matrix found. Will compute during experiments.")
        G = None

    # Load data
    print("\n" + "=" * 60)
    print("Loading Tox21 for Transfer Learning")
    print("=" * 60)

    tox21_path = download_tox21()
    df = pd.read_csv(tox21_path)

    smiles_list = df['smiles'].tolist()
    task_names = list(TOX21_TASKS.keys())

    raw_labels = {}
    for task in TOX21_TASKS:
        if task in df.columns:
            raw_labels[task] = df[task].values.astype(np.float32)

    # Convert to graphs
    preprocessor = MoleculeGraphPreprocessor()
    valid_smiles, graphs, valid_indices = preprocessor.process_smiles_list(
        smiles_list, show_progress=True
    )

    labels = {task: values[valid_indices] for task, values in raw_labels.items()}

    # Filter to molecules with 10+ task labels
    n_labels_per_mol = np.zeros(len(valid_smiles))
    for task, values in labels.items():
        n_labels_per_mol += ~np.isnan(values)

    mask = n_labels_per_mol >= 10
    keep_idx = np.where(mask)[0]

    graphs = [graphs[i] for i in keep_idx]
    valid_smiles = [valid_smiles[i] for i in keep_idx]
    labels = {task: arr[keep_idx] for task, arr in labels.items()}

    print(f"Using {len(graphs)} molecules")

    # Split
    train_idx, val_idx, test_idx = scaffold_split(
        valid_smiles, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=seed
    )

    train_graphs = [graphs[i] for i in train_idx]
    test_graphs = [graphs[i] for i in test_idx]
    train_labels = {task: arr[train_idx] for task, arr in labels.items()}
    test_labels = {task: arr[test_idx] for task, arr in labels.items()}

    train_dataset = MultiTaskGraphDataset(train_graphs, train_labels, TOX21_TASKS)
    test_dataset = MultiTaskGraphDataset(test_graphs, test_labels, TOX21_TASKS)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # First, train baseline models (from scratch for each task)
    print("\n" + "=" * 60)
    print("Training Baseline Models (Scratch)")
    print("=" * 60)

    baseline_scores = {}

    for target_task in task_names:
        print(f"\nTarget: {target_task}")

        model = GNNMultiTaskModel(
            task_names=task_names,
            atom_feature_dim=get_atom_feature_dim(),
            encoder_type='gcn',
            encoder_hidden_dims=[256, 256],
            encoder_output_dim=256,
            head_hidden_dim=128,
            dropout=0.3,
        ).to(device)

        train_single_task_model(model, train_loader, target_task, task_names, device,
                               epochs=args.source_epochs)
        auc = evaluate_task(model, test_loader, target_task, task_names, device)
        baseline_scores[target_task] = auc
        print(f"  Baseline AUC: {auc:.4f}")

    # Generate transfer pairs
    transfer_pairs = []
    for source in task_names:
        for target in task_names:
            if source != target:
                transfer_pairs.append((source, target))

    if args.n_pairs:
        np.random.shuffle(transfer_pairs)
        transfer_pairs = transfer_pairs[:args.n_pairs]

    print(f"\nTotal transfer pairs: {len(transfer_pairs)}")

    # Run transfer experiments
    print("\n" + "=" * 60)
    print("Running Transfer Learning Experiments")
    print("=" * 60)

    transfer_results = []

    for i, (source, target) in enumerate(transfer_pairs):
        print(f"\n[{i+1}/{len(transfer_pairs)}] {source} -> {target}")

        # Train on source task
        model = GNNMultiTaskModel(
            task_names=task_names,
            atom_feature_dim=get_atom_feature_dim(),
            encoder_type='gcn',
            encoder_hidden_dims=[256, 256],
            encoder_output_dim=256,
            head_hidden_dim=128,
            dropout=0.3,
        ).to(device)

        train_single_task_model(model, train_loader, source, task_names, device,
                               epochs=args.source_epochs)

        # Fine-tune on target task
        fine_tune_model(model, train_loader, target, task_names, device,
                       epochs=args.finetune_epochs)

        # Evaluate
        transfer_auc = evaluate_task(model, test_loader, target, task_names, device)
        baseline_auc = baseline_scores[target]
        transfer_gain = transfer_auc - baseline_auc

        # Get gradient similarity if available
        gradient_sim = np.nan
        if G is not None and g_task_names:
            if source in g_task_names and target in g_task_names:
                si = g_task_names.index(source)
                ti = g_task_names.index(target)
                gradient_sim = G[si, ti]

        transfer_results.append({
            'source': source,
            'target': target,
            'baseline_auc': baseline_auc,
            'transfer_auc': transfer_auc,
            'transfer_gain': transfer_gain,
            'gradient_similarity': gradient_sim,
        })

        print(f"  Baseline: {baseline_auc:.4f}, Transfer: {transfer_auc:.4f}, Gain: {transfer_gain:+.4f}")
        if not np.isnan(gradient_sim):
            print(f"  Gradient similarity: {gradient_sim:.4f}")

    # Analyze correlation
    print("\n" + "=" * 60)
    print("Transfer Learning Analysis")
    print("=" * 60)

    results_df = pd.DataFrame(transfer_results)

    # Filter to pairs with gradient similarity
    valid_mask = ~results_df['gradient_similarity'].isna()
    valid_df = results_df[valid_mask]

    if len(valid_df) > 5:
        grad_sim = valid_df['gradient_similarity'].values
        transfer_gain = valid_df['transfer_gain'].values

        r, p = stats.pearsonr(grad_sim, transfer_gain)
        spearman_r, spearman_p = stats.spearmanr(grad_sim, transfer_gain)

        print(f"\nGradient Similarity vs Transfer Gain:")
        print(f"  Pearson r: {r:.4f} (p = {p:.2e})")
        print(f"  Spearman r: {spearman_r:.4f} (p = {spearman_p:.2e})")
        print(f"  N pairs: {len(valid_df)}")

        # Breakdown by gain direction
        positive_gains = valid_df[valid_df['transfer_gain'] > 0]
        negative_gains = valid_df[valid_df['transfer_gain'] < 0]

        print(f"\nTransfer gain statistics:")
        print(f"  Positive transfers: {len(positive_gains)} ({100*len(positive_gains)/len(valid_df):.1f}%)")
        print(f"  Negative transfers: {len(negative_gains)} ({100*len(negative_gains)/len(valid_df):.1f}%)")
        print(f"  Mean positive gain: {positive_gains['transfer_gain'].mean():.4f}")
        print(f"  Mean negative gain: {negative_gains['transfer_gain'].mean():.4f}")

        # Gradient similarity for positive vs negative transfers
        if len(positive_gains) > 0 and len(negative_gains) > 0:
            pos_sim = positive_gains['gradient_similarity'].mean()
            neg_sim = negative_gains['gradient_similarity'].mean()
            t_stat, t_p = stats.ttest_ind(
                positive_gains['gradient_similarity'],
                negative_gains['gradient_similarity']
            )
            print(f"\nGradient similarity by transfer outcome:")
            print(f"  Positive transfers: mean G = {pos_sim:.4f}")
            print(f"  Negative transfers: mean G = {neg_sim:.4f}")
            print(f"  t-test: t = {t_stat:.2f}, p = {t_p:.4f}")

        conclusion = "PASS" if r > 0.5 else ("PARTIAL" if r > 0.3 else "FAIL")
    else:
        r, p = np.nan, np.nan
        spearman_r, spearman_p = np.nan, np.nan
        conclusion = "INSUFFICIENT_DATA"

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    if r > 0.5:
        print(f"\n>>> PASS: Gradient conflicts predict transfer success (r = {r:.2f})")
    elif r > 0.3:
        print(f"\n>>> PARTIAL: Moderate predictive power (r = {r:.2f})")
    else:
        print(f"\n>>> FAIL or INSUFFICIENT: r = {r:.2f}")

    # Save results
    output_dir = Path('outputs/deep_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    final_results = {
        'n_transfer_pairs': len(transfer_results),
        'baseline_scores': baseline_scores,
        'transfer_results': transfer_results,
        'analysis': {
            'pearson_r': float(r) if not np.isnan(r) else None,
            'pearson_p': float(p) if not np.isnan(p) else None,
            'spearman_r': float(spearman_r) if not np.isnan(spearman_r) else None,
            'spearman_p': float(spearman_p) if not np.isnan(spearman_p) else None,
            'conclusion': conclusion,
        },
        'task_names': task_names,
    }

    with open(output_dir / 'transfer_learning_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)

    results_df.to_csv(output_dir / 'transfer_learning_pairs.csv', index=False)

    # Visualization
    if len(valid_df) > 5:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Scatter plot
        ax1 = axes[0]
        colors = ['#2E86AB' if g > 0 else '#A23B72' for g in valid_df['transfer_gain']]
        ax1.scatter(valid_df['gradient_similarity'], valid_df['transfer_gain'],
                   c=colors, alpha=0.7, s=50, edgecolors='white', linewidth=0.5)

        # Add regression line
        z = np.polyfit(grad_sim, transfer_gain, 1)
        p_line = np.poly1d(z)
        x_line = np.linspace(grad_sim.min(), grad_sim.max(), 100)
        ax1.plot(x_line, p_line(x_line), 'r--', linewidth=2, alpha=0.8)

        ax1.axhline(0, color='gray', linestyle='-', alpha=0.3)
        ax1.axvline(0, color='gray', linestyle='-', alpha=0.3)
        ax1.set_xlabel('Gradient Similarity')
        ax1.set_ylabel('Transfer Gain (AUC)')
        ax1.set_title(f'Transfer Learning Prediction\n(r = {r:.3f}, p = {p:.2e})')

        # Heatmap of transfer gains
        ax2 = axes[1]
        gain_matrix = np.zeros((len(task_names), len(task_names)))
        gain_matrix[:] = np.nan

        for _, row in results_df.iterrows():
            si = task_names.index(row['source'])
            ti = task_names.index(row['target'])
            gain_matrix[si, ti] = row['transfer_gain']

        im = ax2.imshow(gain_matrix, cmap='RdBu_r', vmin=-0.1, vmax=0.1)
        ax2.set_xticks(range(len(task_names)))
        ax2.set_yticks(range(len(task_names)))
        ax2.set_xticklabels([t[:6] for t in task_names], rotation=45, ha='right', fontsize=8)
        ax2.set_yticklabels([t[:6] for t in task_names], fontsize=8)
        ax2.set_xlabel('Target Task')
        ax2.set_ylabel('Source Task')
        ax2.set_title('Transfer Gain Matrix')
        plt.colorbar(im, ax=ax2, shrink=0.8)

        plt.tight_layout()
        plt.savefig(output_dir / 'transfer_learning_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
