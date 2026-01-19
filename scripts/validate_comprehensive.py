#!/usr/bin/env python3
"""
COMPREHENSIVE Validation - Tests every experiment end-to-end with mini batches.

This validates:
1. Experiment 3: Transfer learning (single job)
2. Experiment 4: Task selection (single job)
3. Experiment 5: PCGrad (single job)

Each test runs the actual experiment code with minimal data.
"""

import sys
import os
from pathlib import Path
import traceback
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import torch
import numpy as np


def test_experiment3_mini():
    """Test Experiment 3 transfer learning with mini data."""
    print("\n" + "=" * 70)
    print("TEST: Experiment 3 - Transfer Learning (Mini)")
    print("=" * 70)

    sys.path.insert(0, str(project_root / 'scripts'))

    try:
        from experiment3_transfer_learning import (
            get_job_config, load_gradient_matrix, load_and_preprocess_data,
            scaffold_split, run_single_transfer_experiment
        )

        # Get config for job 0
        config = get_job_config(0)
        print(f"  Job config: {config}")

        # Load gradient matrix
        gradient_path = project_root / 'outputs' / 'gradients' / 'gnn_conflict_matrices.npz'
        gradient_matrix, gradient_task_names = load_gradient_matrix(gradient_path)
        print(f"  Gradient matrix: {gradient_matrix.shape}")

        # Load data (this takes a moment)
        print("  Loading data...")
        valid_smiles, graphs, labels = load_and_preprocess_data(min_tasks=10)
        print(f"  Loaded {len(graphs)} molecules")

        # Use only first 100 for speed
        graphs = graphs[:100]
        labels = {k: v[:100] for k, v in labels.items()}
        valid_smiles = valid_smiles[:100]

        # Split
        train_idx, val_idx, test_idx = scaffold_split(
            valid_smiles, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42
        )
        print(f"  Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

        # Training config (minimal)
        mini_config = {
            'learning_rate': 1e-3,
            'weight_decay': 0.01,
            'epochs': 2,  # Very few epochs
            'batch_size': 16,
            'early_stopping_patience': 5,
            'gradient_log_interval': 50,
            'gradient_clip_norm': 1.0,
            'encoder_type': 'gcn',
            'encoder_hidden_dims': [64, 64],  # Smaller model
            'encoder_output_dim': 64,
            'head_hidden_dim': 32,
            'dropout': 0.2,
        }

        # Create temp output dir
        with tempfile.TemporaryDirectory() as tmp_dir:
            device = torch.device('cpu')  # Use CPU for test

            print(f"  Running transfer experiment...")
            result = run_single_transfer_experiment(
                source_task=config['source_task'],
                target_task=config['target_task'],
                data_regime=20,  # Very small for speed
                condition='transfer',
                graphs=graphs,
                labels=labels,
                train_idx=train_idx,
                val_idx=val_idx,
                gradient_matrix=gradient_matrix,
                gradient_task_names=gradient_task_names,
                config=mini_config,
                device=device,
                output_dir=Path(tmp_dir),
                seed=42,
            )

            print(f"  Result keys: {list(result.keys())}")
            print(f"  Source AUC: {result.get('source_auc', 'N/A')}")
            print(f"  Target AUC: {result.get('target_auc', 'N/A')}")

        print("\n  [PASS] Experiment 3 mini test completed successfully!")
        return True

    except Exception as e:
        print(f"\n  [FAIL] Experiment 3 error: {e}")
        traceback.print_exc()
        return False


def test_experiment4_mini():
    """Test Experiment 4 task selection with mini data."""
    print("\n" + "=" * 70)
    print("TEST: Experiment 4 - Task Selection (Mini)")
    print("=" * 70)

    sys.path.insert(0, str(project_root / 'scripts'))

    try:
        from experiment4_task_selection import (
            get_job_config, load_gradient_conflicts,
            run_single_selection_experiment, SELECTION_METHODS, BUDGETS
        )

        # Load gradient matrix
        gradient_path = project_root / 'outputs' / 'gradients' / 'gnn_conflict_matrices.npz'
        gradient_matrix, task_names = load_gradient_conflicts(gradient_path)
        print(f"  Gradient matrix: {gradient_matrix.shape}")
        print(f"  Tasks: {task_names}")

        # Test each method with budget=3
        for method in SELECTION_METHODS:
            print(f"\n  Testing {method} selection...")
            result = run_single_selection_experiment(
                method=method,
                budget=3,
                gradient_matrix=gradient_matrix,
                task_names=task_names,
                n_random_draws=10,  # Few draws for speed
                seed=42
            )

            if method == 'random':
                print(f"    Mean coverage: {result['mean_coverage']:.4f}")
            else:
                print(f"    Selected: {result['selected_tasks']}")
                print(f"    Coverage: {result['coverage']:.4f}")

        # Test job config mapping
        print("\n  Testing job configs...")
        for job_idx in [0, 5, 10, 15, 19]:
            config = get_job_config(job_idx)
            print(f"    Job {job_idx}: {config['method']}, budget={config['budget']}")

        print("\n  [PASS] Experiment 4 mini test completed successfully!")
        return True

    except Exception as e:
        print(f"\n  [FAIL] Experiment 4 error: {e}")
        traceback.print_exc()
        return False


def test_experiment5_mini():
    """Test Experiment 5 PCGrad with mini data."""
    print("\n" + "=" * 70)
    print("TEST: Experiment 5 - PCGrad Validation (Mini)")
    print("=" * 70)

    sys.path.insert(0, str(project_root / 'scripts'))

    try:
        from experiment5_pcgrad import (
            get_job_config, load_tox21_graphs, train_two_task_model,
            ALL_PAIRS, PAIR_CATEGORIES
        )
        from data.graph_dataset import MultiTaskGraphDataset
        from torch.utils.data import Subset

        # Test job configs
        print("  Testing job configs...")
        for job_idx in [0, 5, 10]:
            config = get_job_config(job_idx)
            print(f"    Job {job_idx}: {config['task1']} vs {config['task2']} ({config['category']})")

        # Load data
        print("\n  Loading Tox21 graphs...")
        data_dir = project_root / 'outputs' / 'raw_data'
        graphs, labels, atom_dim = load_tox21_graphs(data_dir, min_tasks=10)
        print(f"  Loaded {len(graphs)} graphs, atom_dim={atom_dim}")

        # Use only 50 molecules for speed
        graphs = graphs[:50]
        labels = {k: v[:50] for k, v in labels.items()}

        # Create dataset for 2 tasks
        task1, task2 = 'NR-AR', 'NR-ER'
        task_types = {task1: 'classification', task2: 'classification'}
        two_task_labels = {task1: labels[task1], task2: labels[task2]}

        dataset = MultiTaskGraphDataset(graphs, two_task_labels, task_types)
        print(f"  Dataset: {len(dataset)} samples")

        # Split
        train_dataset = Subset(dataset, list(range(40)))
        val_dataset = Subset(dataset, list(range(40, 50)))
        train_dataset.task_names = dataset.task_names
        val_dataset.task_names = dataset.task_names

        # Test WITHOUT PCGrad
        print("\n  Training WITHOUT PCGrad (2 epochs)...")
        torch.manual_seed(42)
        np.random.seed(42)
        result_baseline = train_two_task_model(
            task1=task1,
            task2=task2,
            use_pcgrad=False,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device='cpu',
            atom_feature_dim=atom_dim,
            epochs=2,
            batch_size=8,
            patience=10
        )
        print(f"    Epochs trained: {result_baseline['epochs_trained']}")
        print(f"    Best AUC: {result_baseline['best_val_auc']:.4f}")
        print(f"    Task AUCs: {result_baseline['task_aucs']}")

        # Test WITH PCGrad
        print("\n  Training WITH PCGrad (2 epochs)...")
        torch.manual_seed(42)
        np.random.seed(42)
        result_pcgrad = train_two_task_model(
            task1=task1,
            task2=task2,
            use_pcgrad=True,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device='cpu',
            atom_feature_dim=atom_dim,
            epochs=2,
            batch_size=8,
            patience=10
        )
        print(f"    Epochs trained: {result_pcgrad['epochs_trained']}")
        print(f"    Best AUC: {result_pcgrad['best_val_auc']:.4f}")
        print(f"    Task AUCs: {result_pcgrad['task_aucs']}")

        improvement = result_pcgrad['best_val_auc'] - result_baseline['best_val_auc']
        print(f"\n  PCGrad improvement: {improvement:+.4f}")

        print("\n  [PASS] Experiment 5 mini test completed successfully!")
        return True

    except Exception as e:
        print(f"\n  [FAIL] Experiment 5 error: {e}")
        traceback.print_exc()
        return False


def test_data_pipeline():
    """Test the full data pipeline."""
    print("\n" + "=" * 70)
    print("TEST: Data Pipeline")
    print("=" * 70)

    try:
        from data.graph_preprocessing import MoleculeGraphPreprocessor, get_atom_feature_dim
        from data.graph_dataset import MultiTaskGraphDataset
        from torch_geometric.loader import DataLoader

        # Test preprocessing
        print("  Testing MoleculeGraphPreprocessor...")
        preprocessor = MoleculeGraphPreprocessor()

        test_smiles = ['CCO', 'c1ccccc1', 'CC(=O)O', 'CCN', 'CCCC']
        valid_smiles, graphs, valid_indices = preprocessor.process_smiles_list(
            test_smiles, show_progress=False
        )
        print(f"    Processed {len(graphs)}/{len(test_smiles)} molecules")
        print(f"    Atom feature dim: {preprocessor.atom_feature_dim}")
        print(f"    Bond feature dim: {preprocessor.bond_feature_dim}")

        # Check graph structure
        g = graphs[0]
        print(f"    Sample graph: {g.num_nodes} nodes, {g.num_edges} edges")
        print(f"    Node features shape: {g.x.shape}")
        print(f"    Edge features shape: {g.edge_attr.shape}")

        # Test dataset
        print("\n  Testing MultiTaskGraphDataset...")
        labels = {
            'task1': np.array([1.0, 0.0, np.nan, 1.0, 0.0], dtype=np.float32),
            'task2': np.array([0.0, 1.0, 1.0, np.nan, 0.0], dtype=np.float32),
        }
        task_types = {'task1': 'classification', 'task2': 'classification'}

        dataset = MultiTaskGraphDataset(graphs, labels, task_types)
        print(f"    Dataset size: {len(dataset)}")
        print(f"    Task names: {dataset.task_names}")

        # Test single item
        item = dataset[0]
        print(f"    Item type: {type(item)}")
        print(f"    Item.y shape: {item.y.shape}")
        print(f"    Item.mask shape: {item.mask.shape}")

        # Test dataloader
        print("\n  Testing DataLoader batching...")
        loader = DataLoader(dataset, batch_size=3, shuffle=False)
        batch = next(iter(loader))
        print(f"    Batch num_graphs: {batch.num_graphs}")
        print(f"    Batch.y shape: {batch.y.shape}")
        print(f"    Batch.mask shape: {batch.mask.shape}")

        # Test reshaping (the critical fix)
        if batch.y.dim() == 1:
            batch_size = batch.num_graphs
            n_tasks = len(dataset.task_names)
            y_reshaped = batch.y.view(batch_size, n_tasks)
            mask_reshaped = batch.mask.view(batch_size, n_tasks)
            print(f"    Reshaped y: {y_reshaped.shape}")
            print(f"    Reshaped mask: {mask_reshaped.shape}")

        print("\n  [PASS] Data pipeline test completed successfully!")
        return True

    except Exception as e:
        print(f"\n  [FAIL] Data pipeline error: {e}")
        traceback.print_exc()
        return False


def test_model_forward():
    """Test model forward pass with various batch sizes."""
    print("\n" + "=" * 70)
    print("TEST: Model Forward Pass")
    print("=" * 70)

    try:
        from data.graph_preprocessing import MoleculeGraphPreprocessor
        from models.gnn_encoder import GCNEncoder, GATEncoder
        from models.gnn_multitask import GNNMultiTaskModel
        from torch_geometric.data import Batch

        # Create test graphs
        preprocessor = MoleculeGraphPreprocessor()
        smiles = ['CCO', 'c1ccccc1', 'CC(=O)O', 'CCN', 'CCCC', 'c1ccc(O)cc1']
        _, graphs, _ = preprocessor.process_smiles_list(smiles, show_progress=False)

        atom_dim = preprocessor.atom_feature_dim
        print(f"  Atom feature dim: {atom_dim}")

        # Test GCNEncoder
        print("\n  Testing GCNEncoder...")
        gcn = GCNEncoder(
            input_dim=atom_dim,
            hidden_dims=[64, 64],
            output_dim=64,
            dropout=0.2
        )

        # Skip batch_size=1 in training mode (BatchNorm requires >1)
        for batch_size in [2, 4]:
            batch = Batch.from_data_list(graphs[:batch_size])
            output = gcn(batch)
            print(f"    Batch size {batch_size}: input graphs={batch.num_graphs}, output={output.shape}")

        # Test batch_size=1 in eval mode
        gcn.eval()
        batch = Batch.from_data_list(graphs[:1])
        output = gcn(batch)
        print(f"    Batch size 1 (eval mode): input graphs={batch.num_graphs}, output={output.shape}")
        gcn.train()

        # Test GATEncoder
        print("\n  Testing GATEncoder...")
        gat = GATEncoder(
            input_dim=atom_dim,
            hidden_dims=[64, 64],
            output_dim=64,
            dropout=0.2,
            heads=4
        )

        # Skip batch_size=1 in training mode (BatchNorm requires >1)
        for batch_size in [2, 4]:
            batch = Batch.from_data_list(graphs[:batch_size])
            output = gat(batch)
            print(f"    Batch size {batch_size}: input graphs={batch.num_graphs}, output={output.shape}")

        # Test batch_size=1 in eval mode
        gat.eval()
        batch = Batch.from_data_list(graphs[:1])
        output = gat(batch)
        print(f"    Batch size 1 (eval mode): input graphs={batch.num_graphs}, output={output.shape}")
        gat.train()

        # Test full multi-task model
        print("\n  Testing GNNMultiTaskModel...")
        model = GNNMultiTaskModel(
            task_names=['task1', 'task2', 'task3'],
            atom_feature_dim=atom_dim,
            encoder_type='gcn',
            encoder_hidden_dims=[64, 64],
            encoder_output_dim=64,
            head_hidden_dim=32,
            dropout=0.2
        )

        # Skip batch_size=1 in training mode (BatchNorm requires >1)
        for batch_size in [2, 4]:
            batch = Batch.from_data_list(graphs[:batch_size])
            outputs = model(batch)
            print(f"    Batch size {batch_size}: outputs = {[(k, v.shape) for k, v in outputs.items()]}")

        # Test batch_size=1 in eval mode
        model.eval()
        batch = Batch.from_data_list(graphs[:1])
        outputs = model(batch)
        print(f"    Batch size 1 (eval mode): outputs = {[(k, v.shape) for k, v in outputs.items()]}")

        print("\n  [PASS] Model forward pass test completed successfully!")
        return True

    except Exception as e:
        print(f"\n  [FAIL] Model forward pass error: {e}")
        traceback.print_exc()
        return False


def test_pcgrad_optimizer():
    """Test PCGrad optimizer in detail."""
    print("\n" + "=" * 70)
    print("TEST: PCGrad Optimizer")
    print("=" * 70)

    try:
        import torch.nn as nn
        from training.pcgrad import PCGrad, compute_gradient_conflict_stats

        # Create simple model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        pcgrad = PCGrad(optimizer)

        # Test multiple iterations
        print("  Testing PCGrad iterations...")
        for i in range(3):
            x = torch.randn(8, 10)
            output = model(x)

            # Create conflicting gradients (different task heads)
            task_losses = {
                'task1': output[:, 0].mean(),
                'task2': -output[:, 1].mean(),  # Negative to create conflict
                'task3': output[:, 2].mean(),
            }

            pcgrad.zero_grad()
            pcgrad.backward(
                task_losses,
                shared_params=list(model.parameters())
            )
            pcgrad.step()

            # Check gradients exist
            has_grad = any(p.grad is not None for p in model.parameters())
            grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
            print(f"    Iteration {i+1}: has_grad={has_grad}, grad_norm={grad_norm:.4f}")

        # Test gradient conflict stats
        print("\n  Testing gradient conflict stats...")
        task_grads = {
            'task1': torch.randn(100),
            'task2': torch.randn(100),
            'task3': torch.randn(100),
        }
        stats = compute_gradient_conflict_stats(task_grads)
        print(f"    Stats: {stats}")

        print("\n  [PASS] PCGrad optimizer test completed successfully!")
        return True

    except Exception as e:
        print(f"\n  [FAIL] PCGrad optimizer error: {e}")
        traceback.print_exc()
        return False


def test_gradient_logger():
    """Test gradient conflict logger."""
    print("\n" + "=" * 70)
    print("TEST: Gradient Conflict Logger")
    print("=" * 70)

    try:
        import torch.nn as nn
        from training.gradient_logger import GradientConflictLogger

        # Create simple model with encoder
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(10, 20),
                    nn.ReLU(),
                    nn.Linear(20, 10)
                )
                self.head1 = nn.Linear(10, 1)
                self.head2 = nn.Linear(10, 1)

            def get_encoder_parameters(self):
                return self.encoder.parameters()

        model = SimpleModel()
        task_names = ['task1', 'task2']

        logger = GradientConflictLogger(
            model=model,
            task_names=task_names,
            log_interval=1,
            device='cpu'
        )

        print("  Testing gradient logging...")
        for step in range(5):
            x = torch.randn(4, 10)
            enc = model.encoder(x)
            out1 = model.head1(enc)
            out2 = model.head2(enc)

            task_losses = {
                'task1': out1.mean(),
                'task2': out2.mean(),
            }

            matrix = logger.log_step(step, task_losses)
            if matrix is not None:
                print(f"    Step {step}: logged, diagonal={matrix.diagonal()}")

        # Get averaged matrix
        avg_matrix = logger.get_averaged_conflict_matrix()
        print(f"\n  Averaged matrix shape: {avg_matrix.shape}")
        print(f"  Diagonal: {avg_matrix.diagonal()}")

        # Test summary
        summary = logger.summary()
        print(f"\n  Summary:\n{summary[:500]}...")

        print("\n  [PASS] Gradient logger test completed successfully!")
        return True

    except Exception as e:
        print(f"\n  [FAIL] Gradient logger error: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all comprehensive tests."""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE VALIDATION SUITE")
    print("=" * 70)
    print("This tests every experiment end-to-end with mini batches.")

    results = {}

    # Run all tests
    results['data_pipeline'] = test_data_pipeline()
    results['model_forward'] = test_model_forward()
    results['gradient_logger'] = test_gradient_logger()
    results['pcgrad_optimizer'] = test_pcgrad_optimizer()
    results['experiment4'] = test_experiment4_mini()
    results['experiment5'] = test_experiment5_mini()
    results['experiment3'] = test_experiment3_mini()

    # Summary
    print("\n" + "=" * 70)
    print("COMPREHENSIVE VALIDATION SUMMARY")
    print("=" * 70)

    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        symbol = "[OK]" if passed else "[XX]"
        print(f"  {symbol} {test_name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  *** ALL COMPREHENSIVE TESTS PASSED ***")
        print("  Ready for HPC submission!")
    else:
        print("  *** SOME TESTS FAILED ***")
        print("  Fix errors before HPC submission!")

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
