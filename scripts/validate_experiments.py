#!/usr/bin/env python3
"""
Validation script to test all experiments before HPC submission.

Runs mini-batch tests to ensure:
1. All imports work
2. Models can be instantiated
3. Training loop runs without errors
4. Results can be saved

Usage:
    python scripts/validate_experiments.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import traceback

def test_imports():
    """Test all required imports."""
    print("\n" + "=" * 60)
    print("Testing imports...")
    print("=" * 60)

    errors = []

    # Core imports
    try:
        import torch
        print(f"  [OK] torch {torch.__version__}")
    except ImportError as e:
        errors.append(f"torch: {e}")

    try:
        import torch_geometric
        print(f"  [OK] torch_geometric {torch_geometric.__version__}")
    except ImportError as e:
        errors.append(f"torch_geometric: {e}")

    try:
        import numpy as np
        print(f"  [OK] numpy {np.__version__}")
    except ImportError as e:
        errors.append(f"numpy: {e}")

    try:
        import pandas as pd
        print(f"  [OK] pandas {pd.__version__}")
    except ImportError as e:
        errors.append(f"pandas: {e}")

    try:
        from rdkit import Chem
        print(f"  [OK] rdkit")
    except ImportError as e:
        errors.append(f"rdkit: {e}")

    try:
        from sklearn.metrics import roc_auc_score
        print(f"  [OK] sklearn")
    except ImportError as e:
        errors.append(f"sklearn: {e}")

    # Project imports
    try:
        from data.graph_preprocessing import MoleculeGraphPreprocessor, get_atom_feature_dim
        print(f"  [OK] data.graph_preprocessing")
    except ImportError as e:
        errors.append(f"data.graph_preprocessing: {e}")

    try:
        from data.graph_dataset import MultiTaskGraphDataset
        print(f"  [OK] data.graph_dataset")
    except ImportError as e:
        errors.append(f"data.graph_dataset: {e}")

    try:
        from data.splitting import scaffold_split
        print(f"  [OK] data.splitting")
    except ImportError as e:
        errors.append(f"data.splitting: {e}")

    try:
        from models.gnn_encoder import GCNEncoder, GATEncoder, create_gnn_encoder
        print(f"  [OK] models.gnn_encoder")
    except ImportError as e:
        errors.append(f"models.gnn_encoder: {e}")

    try:
        from models.heads import TaskHead
        print(f"  [OK] models.heads")
    except ImportError as e:
        errors.append(f"models.heads: {e}")

    try:
        from models.gnn_multitask import GNNMultiTaskModel
        print(f"  [OK] models.gnn_multitask")
    except ImportError as e:
        errors.append(f"models.gnn_multitask: {e}")

    try:
        from training.losses import MultiTaskLoss
        print(f"  [OK] training.losses")
    except ImportError as e:
        errors.append(f"training.losses: {e}")

    try:
        from training.gradient_logger import GradientConflictLogger
        print(f"  [OK] training.gradient_logger")
    except ImportError as e:
        errors.append(f"training.gradient_logger: {e}")

    try:
        from training.pcgrad import PCGrad
        print(f"  [OK] training.pcgrad")
    except ImportError as e:
        errors.append(f"training.pcgrad: {e}")

    if errors:
        print("\n  IMPORT ERRORS:")
        for err in errors:
            print(f"    [FAIL] {err}")
        return False

    print("\n  All imports successful!")
    return True


def test_prerequisites():
    """Test that prerequisite files exist."""
    print("\n" + "=" * 60)
    print("Testing prerequisites...")
    print("=" * 60)

    errors = []

    # Check gradient matrix
    gradient_path = project_root / 'outputs' / 'gradients' / 'gnn_conflict_matrices.npz'
    if gradient_path.exists():
        import numpy as np
        data = np.load(gradient_path, allow_pickle=True)
        matrix = data['averaged']
        task_names = data['task_names'].tolist()
        print(f"  [OK] Gradient matrix: {gradient_path}")
        print(f"       Shape: {matrix.shape}")
        print(f"       Tasks: {len(task_names)}")
        print(f"       Diagonal mean: {matrix.diagonal().mean():.4f}")
    else:
        errors.append(f"Gradient matrix not found: {gradient_path}")

    # Check model checkpoint
    model_path = project_root / 'outputs' / 'checkpoints' / 'best_gnn_model.pt'
    if model_path.exists():
        print(f"  [OK] Model checkpoint: {model_path}")
    else:
        errors.append(f"Model checkpoint not found: {model_path}")

    # Check Tox21 data
    tox21_path = project_root / 'outputs' / 'raw_data' / 'tox21.csv'
    if tox21_path.exists():
        import pandas as pd
        df = pd.read_csv(tox21_path)
        print(f"  [OK] Tox21 data: {tox21_path}")
        print(f"       Molecules: {len(df)}")
    else:
        errors.append(f"Tox21 data not found: {tox21_path}")

    if errors:
        print("\n  PREREQUISITE ERRORS:")
        for err in errors:
            print(f"    [FAIL] {err}")
        return False

    print("\n  All prerequisites found!")
    return True


def test_model_instantiation():
    """Test that models can be created."""
    print("\n" + "=" * 60)
    print("Testing model instantiation...")
    print("=" * 60)

    import torch
    from data.graph_preprocessing import get_atom_feature_dim
    from models.gnn_encoder import GCNEncoder, GATEncoder
    from models.heads import TaskHead
    from models.gnn_multitask import GNNMultiTaskModel

    errors = []
    atom_dim = get_atom_feature_dim()
    print(f"  Atom feature dim: {atom_dim}")

    # Test GCN encoder
    try:
        encoder = GCNEncoder(
            input_dim=atom_dim,
            hidden_dims=[256, 256],
            output_dim=256,
            dropout=0.2
        )
        print(f"  [OK] GCNEncoder created ({sum(p.numel() for p in encoder.parameters())} params)")
    except Exception as e:
        errors.append(f"GCNEncoder: {e}")

    # Test GAT encoder
    try:
        encoder = GATEncoder(
            input_dim=atom_dim,
            hidden_dims=[256, 256],
            output_dim=256,
            dropout=0.2
        )
        print(f"  [OK] GATEncoder created ({sum(p.numel() for p in encoder.parameters())} params)")
    except Exception as e:
        errors.append(f"GATEncoder: {e}")

    # Test TaskHead
    try:
        head = TaskHead(input_dim=256, hidden_dim=128, dropout=0.2)
        print(f"  [OK] TaskHead created ({sum(p.numel() for p in head.parameters())} params)")
    except Exception as e:
        errors.append(f"TaskHead: {e}")

    # Test full model
    try:
        model = GNNMultiTaskModel(
            task_names=['task1', 'task2'],
            atom_feature_dim=atom_dim,
            encoder_type='gcn',
            encoder_hidden_dims=[256, 256],
            encoder_output_dim=256,
            head_hidden_dim=128,
            dropout=0.2
        )
        print(f"  [OK] GNNMultiTaskModel created ({sum(p.numel() for p in model.parameters())} params)")
    except Exception as e:
        errors.append(f"GNNMultiTaskModel: {e}")

    if errors:
        print("\n  MODEL ERRORS:")
        for err in errors:
            print(f"    [FAIL] {err}")
            traceback.print_exc()
        return False

    print("\n  All models created successfully!")
    return True


def test_mini_training():
    """Test a mini training run."""
    print("\n" + "=" * 60)
    print("Testing mini training run...")
    print("=" * 60)

    import torch
    import numpy as np
    from torch_geometric.loader import DataLoader
    from data.graph_preprocessing import MoleculeGraphPreprocessor, get_atom_feature_dim
    from data.graph_dataset import MultiTaskGraphDataset
    from models.gnn_multitask import GNNMultiTaskModel
    from training.losses import MultiTaskLoss

    # Create some fake molecules for testing
    test_smiles = [
        'CCO', 'CCCO', 'CCCCO', 'CC(C)O', 'c1ccccc1',
        'c1ccccc1O', 'CC(=O)O', 'CC(=O)OC', 'CCN', 'CCNC'
    ]

    print(f"  Processing {len(test_smiles)} test molecules...")

    try:
        # Process molecules
        preprocessor = MoleculeGraphPreprocessor()
        valid_smiles, graphs, valid_indices = preprocessor.process_smiles_list(
            test_smiles, show_progress=False
        )

        if len(graphs) < 5:
            print(f"  [FAIL] Only {len(graphs)} valid molecules, need at least 5")
            return False

        print(f"  [OK] Processed {len(graphs)} molecules")

        # Create fake labels
        np.random.seed(42)
        labels = {
            'task1': np.random.rand(len(graphs)).astype(np.float32),
            'task2': np.random.rand(len(graphs)).astype(np.float32),
        }
        # Add some NaN for missing labels
        labels['task1'][0] = np.nan
        labels['task2'][1] = np.nan

        task_types = {'task1': 'regression', 'task2': 'regression'}

        # Create dataset
        dataset = MultiTaskGraphDataset(graphs, labels, task_types)
        print(f"  [OK] Created dataset with {len(dataset)} samples")

        # Create dataloader
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        print(f"  [OK] Created dataloader")

        # Create model
        atom_dim = get_atom_feature_dim()
        model = GNNMultiTaskModel(
            task_names=['task1', 'task2'],
            atom_feature_dim=atom_dim,
            encoder_type='gcn',
            encoder_hidden_dims=[64, 64],
            encoder_output_dim=64,
            head_hidden_dim=32,
            dropout=0.2
        )
        print(f"  [OK] Created model")

        # Create loss
        loss_fn = MultiTaskLoss(task_types)

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Run a few training steps
        model.train()
        for batch_idx, batch in enumerate(loader):
            optimizer.zero_grad()

            # Forward
            outputs = model(batch)

            # Get labels and masks - handle PyG batching
            batch_labels = {}
            batch_masks = {}
            batch_size = batch.num_graphs
            n_tasks = 2

            # PyG may stack graph-level attributes as [batch_size * n_tasks]
            y_tensor = batch.y
            mask_tensor = batch.mask

            if y_tensor.dim() == 1:
                y_tensor = y_tensor.view(batch_size, n_tasks)
                mask_tensor = mask_tensor.view(batch_size, n_tasks)

            for i, task in enumerate(['task1', 'task2']):
                batch_labels[task] = y_tensor[:, i]
                batch_masks[task] = mask_tensor[:, i]

            # Loss
            loss, task_losses = loss_fn(outputs, batch_labels, batch_masks)

            # Backward
            loss.backward()
            optimizer.step()

            print(f"  [OK] Training step {batch_idx+1}: loss = {loss.item():.4f}")

            if batch_idx >= 2:
                break

        print("\n  Mini training run successful!")
        return True

    except Exception as e:
        print(f"\n  [FAIL] Training error: {e}")
        traceback.print_exc()
        return False


def test_experiment3_config():
    """Test experiment 3 job configuration."""
    print("\n" + "=" * 60)
    print("Testing Experiment 3 configuration...")
    print("=" * 60)

    sys.path.insert(0, str(project_root / 'scripts'))
    from experiment3_transfer_learning import get_job_config, get_all_transfer_pairs

    try:
        pairs = get_all_transfer_pairs()
        print(f"  [OK] Total transfer pairs: {len(pairs)}")

        # Test job configs
        for job_index in [0, 100, 500, 791]:
            config = get_job_config(job_index)
            print(f"  [OK] Job {job_index}: {config['source_task']} -> {config['target_task']}, "
                  f"n={config['data_regime']}, {config['condition']}")

        print("\n  Experiment 3 configuration OK!")
        return True

    except Exception as e:
        print(f"\n  [FAIL] {e}")
        traceback.print_exc()
        return False


def test_experiment4_config():
    """Test experiment 4 job configuration."""
    print("\n" + "=" * 60)
    print("Testing Experiment 4 configuration...")
    print("=" * 60)

    sys.path.insert(0, str(project_root / 'scripts'))
    from experiment4_task_selection import get_job_config, SELECTION_METHODS, BUDGETS

    try:
        print(f"  [OK] Selection methods: {SELECTION_METHODS}")
        print(f"  [OK] Budgets: {BUDGETS}")

        # Test job configs
        for job_index in [0, 5, 10, 15, 19]:
            config = get_job_config(job_index)
            print(f"  [OK] Job {job_index}: method={config['method']}, budget={config['budget']}")

        print("\n  Experiment 4 configuration OK!")
        return True

    except Exception as e:
        print(f"\n  [FAIL] {e}")
        traceback.print_exc()
        return False


def test_experiment5_config():
    """Test experiment 5 job configuration."""
    print("\n" + "=" * 60)
    print("Testing Experiment 5 configuration...")
    print("=" * 60)

    sys.path.insert(0, str(project_root / 'scripts'))
    from experiment5_pcgrad import get_job_config, ALL_PAIRS, PAIR_CATEGORIES

    try:
        print(f"  [OK] Total pairs: {len(ALL_PAIRS)}")
        print(f"  [OK] Categories: {set(PAIR_CATEGORIES)}")

        # Test job configs
        for job_index in [0, 5, 10, 14]:
            config = get_job_config(job_index)
            print(f"  [OK] Job {job_index}: {config['task1']} vs {config['task2']}, "
                  f"category={config['category']}")

        print("\n  Experiment 5 configuration OK!")
        return True

    except Exception as e:
        print(f"\n  [FAIL] {e}")
        traceback.print_exc()
        return False


def test_pcgrad():
    """Test PCGrad optimizer."""
    print("\n" + "=" * 60)
    print("Testing PCGrad...")
    print("=" * 60)

    import torch
    import torch.nn as nn
    from training.pcgrad import PCGrad

    try:
        # Simple test model
        model = nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        pcgrad = PCGrad(optimizer)

        # Fake input
        x = torch.randn(5, 10)
        output = model(x)

        # Fake per-task losses
        task_losses = {
            'task1': output[:, 0].mean(),
            'task2': output[:, 1].mean(),
        }

        # Test PCGrad backward
        pcgrad.zero_grad()
        pcgrad.backward(
            task_losses,
            shared_params=list(model.parameters())
        )
        pcgrad.step()

        print(f"  [OK] PCGrad backward successful")

        # Check gradients were set
        has_grad = any(p.grad is not None for p in model.parameters())
        if has_grad:
            print(f"  [OK] Gradients were computed")
        else:
            print(f"  [FAIL] No gradients computed")
            return False

        print("\n  PCGrad test passed!")
        return True

    except Exception as e:
        print(f"\n  [FAIL] {e}")
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("\n" + "=" * 60)
    print("GRADIENT PROJECT VALIDATION")
    print("=" * 60)

    results = {}

    # Run tests
    results['imports'] = test_imports()
    results['prerequisites'] = test_prerequisites()
    results['models'] = test_model_instantiation()
    results['training'] = test_mini_training()
    results['pcgrad'] = test_pcgrad()
    results['exp3_config'] = test_experiment3_config()
    results['exp4_config'] = test_experiment4_config()
    results['exp5_config'] = test_experiment5_config()

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  [OK] ALL TESTS PASSED - Ready for HPC submission!")
    else:
        print("  [FAIL] SOME TESTS FAILED - Fix errors before HPC submission!")

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
