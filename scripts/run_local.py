#!/usr/bin/env python3
"""
Local runner for pre-training and analysis experiments.

Runs locally:
    - Pre-train GNN model (generates gradient matrix)
    - Experiment 2: SAR Validation
    - Experiment 6: Novel Discovery
    - Experiment 7: Representation Generalization (ECFP vs GNN)
    - Single-task baselines

After local runs complete, submit HPC jobs for:
    - Experiment 3: Transfer learning (792 jobs)
    - Experiment 4: Task selection (20 jobs)
    - Experiment 5: PCGrad validation (15 jobs)

Usage:
    python scripts/run_local.py                    # Run all local experiments
    python scripts/run_local.py --pretrain         # Only pretrain
    python scripts/run_local.py --analysis         # Only exp 2 and 6 (needs gradient matrix)
    python scripts/run_local.py --exp 2            # Run specific experiment
    python scripts/run_local.py --exp 7            # Run experiment 7
    python scripts/run_local.py --baselines        # Run single-task baselines
    python scripts/run_local.py --check            # Check prerequisites
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_prerequisites(need_gradient_matrix=True):
    """Check if required files exist."""
    print("Checking prerequisites...")

    all_good = True

    # Check for required Python packages
    try:
        import torch
        print(f"  [OK] PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  [OK] CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("  [WARN] CUDA not available - will use CPU (slower)")
    except ImportError:
        print("  [MISSING] PyTorch not installed")
        all_good = False

    try:
        import torch_geometric
        print(f"  [OK] PyTorch Geometric {torch_geometric.__version__}")
    except ImportError:
        print("  [MISSING] PyTorch Geometric not installed")
        all_good = False

    try:
        import rdkit
        print(f"  [OK] RDKit available")
    except ImportError:
        print("  [MISSING] RDKit not installed")
        all_good = False

    # Check gradient matrix if needed
    if need_gradient_matrix:
        gradient_matrix = project_root / 'outputs' / 'gradients' / 'gnn_conflict_matrices.npz'
        if gradient_matrix.exists():
            print(f"  [OK] Gradient matrix found")
        else:
            print(f"  [MISSING] Gradient matrix not found")
            print(f"           Run pretrain first: python scripts/run_local.py --pretrain")
            all_good = False

    return all_good


def run_pretrain(epochs=100):
    """Run GNN pre-training to generate gradient matrix."""
    print("\n" + "=" * 60)
    print("STEP 0: Pre-training GNN Model")
    print("=" * 60)
    print("This generates the gradient conflict matrix needed for analysis.")
    print("")

    script = project_root / 'train_tox21_gnn.py'

    cmd = [
        sys.executable, str(script),
        '--epochs', str(epochs),
        '--batch_size', '32',
        '--lr', '1e-3',
        '--encoder_type', 'gcn',
        '--min_tasks', '10',
    ]

    print(f"Command: {' '.join(cmd)}")
    print("")

    start_time = time.time()
    result = subprocess.run(cmd, cwd=str(project_root))
    elapsed = time.time() - start_time

    if result.returncode == 0:
        print(f"\n[SUCCESS] Pre-training completed in {elapsed/60:.1f} minutes")

        # Verify output
        gradient_matrix = project_root / 'outputs' / 'gradients' / 'gnn_conflict_matrices.npz'
        if gradient_matrix.exists():
            print(f"  Gradient matrix saved to: {gradient_matrix}")
        else:
            print("  [WARN] Gradient matrix not found - check training logs")
    else:
        print(f"\n[ERROR] Pre-training failed with code {result.returncode}")

    return result.returncode == 0


def run_experiment_2():
    """Run SAR Validation experiment."""
    print("\n" + "=" * 60)
    print("Experiment 2: SAR Validation")
    print("=" * 60)

    script = project_root / 'scripts' / 'experiment2_sar_validation.py'
    output_dir = project_root / 'outputs' / 'sar_validation'
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(script),
        '--gradient-matrix', str(project_root / 'outputs' / 'gradients' / 'gnn_conflict_matrices.npz'),
        '--output-dir', str(output_dir),
    ]

    print(f"Command: {' '.join(cmd)}")
    print("")

    result = subprocess.run(cmd, cwd=str(project_root))

    if result.returncode == 0:
        print(f"\n[SUCCESS] Experiment 2 completed")
        print(f"  Results saved to: {output_dir}")
    else:
        print(f"\n[ERROR] Experiment 2 failed with code {result.returncode}")

    return result.returncode == 0


def run_experiment_6():
    """Run Novel Discovery experiment."""
    print("\n" + "=" * 60)
    print("Experiment 6: Novel Trade-off Discovery")
    print("=" * 60)

    script = project_root / 'scripts' / 'experiment6_novel_discovery.py'
    output_dir = project_root / 'outputs' / 'novel_discovery'
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(script),
        '--gradient-matrix', str(project_root / 'outputs' / 'gradients' / 'gnn_conflict_matrices.npz'),
        '--output-dir', str(output_dir),
    ]

    print(f"Command: {' '.join(cmd)}")
    print("")

    result = subprocess.run(cmd, cwd=str(project_root))

    if result.returncode == 0:
        print(f"\n[SUCCESS] Experiment 6 completed")
        print(f"  Results saved to: {output_dir}")
    else:
        print(f"\n[ERROR] Experiment 6 failed with code {result.returncode}")

    return result.returncode == 0


def run_experiment_7():
    """Run Representation Generalization experiment (ECFP vs GNN)."""
    print("\n" + "=" * 60)
    print("Experiment 7: Representation Generalization")
    print("=" * 60)
    print("Compares gradient patterns between ECFP and GNN representations.")
    print("")

    script = project_root / 'scripts' / 'experiment7_representation.py'
    output_dir = project_root / 'outputs' / 'representation'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run both ECFP (index 0) and GNN (index 1)
    results = []
    for idx, rep_type in [(0, 'ECFP'), (1, 'GNN')]:
        print(f"\n--- Running {rep_type} model ---")

        cmd = [
            sys.executable, str(script),
            '--index', str(idx),
            '--output-dir', str(output_dir),
        ]

        print(f"Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(project_root))
        results.append(result.returncode == 0)

    if all(results):
        print(f"\n[SUCCESS] Experiment 7 completed")
        print(f"  Results saved to: {output_dir}")

        # Run aggregation
        print("\n--- Aggregating results ---")
        cmd = [sys.executable, str(script), '--aggregate', '--output-dir', str(output_dir)]
        subprocess.run(cmd, cwd=str(project_root))
    else:
        print(f"\n[ERROR] Experiment 7 had failures")

    return all(results)


def run_baselines():
    """Run single-task baselines."""
    print("\n" + "=" * 60)
    print("Single-Task Baselines")
    print("=" * 60)
    print("Training individual models for each Tox21 task.")
    print("")

    script = project_root / 'scripts' / 'single_task_baselines.py'
    output_dir = project_root / 'outputs' / 'baselines'
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(script),
        '--output-dir', str(output_dir),
        '--epochs', '100',
    ]

    print(f"Command: {' '.join(cmd)}")
    print("")

    start_time = time.time()
    result = subprocess.run(cmd, cwd=str(project_root))
    elapsed = time.time() - start_time

    if result.returncode == 0:
        print(f"\n[SUCCESS] Baselines completed in {elapsed/60:.1f} minutes")
        print(f"  Results saved to: {output_dir}")
    else:
        print(f"\n[ERROR] Baselines failed with code {result.returncode}")

    return result.returncode == 0


def run_all():
    """Run all local experiments in order."""
    results = {}

    # Step 0: Pretrain
    results['pretrain'] = run_pretrain()

    if not results['pretrain']:
        print("\n[ABORT] Pretrain failed - cannot continue with analysis experiments")
        return results

    # Analysis experiments (need gradient matrix)
    results['exp2'] = run_experiment_2()
    results['exp6'] = run_experiment_6()

    # Representation experiment
    results['exp7'] = run_experiment_7()

    # Baselines
    results['baselines'] = run_baselines()

    return results


def main():
    parser = argparse.ArgumentParser(description='Run local experiments')
    parser.add_argument('--pretrain', action='store_true',
                       help='Run only pre-training')
    parser.add_argument('--analysis', action='store_true',
                       help='Run only analysis experiments (2 and 6)')
    parser.add_argument('--exp', type=int, choices=[2, 6, 7],
                       help='Run specific experiment')
    parser.add_argument('--baselines', action='store_true',
                       help='Run only single-task baselines')
    parser.add_argument('--check', action='store_true',
                       help='Only check prerequisites')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs for training (default: 100)')

    args = parser.parse_args()

    # Create output directories
    for subdir in ['gradients', 'checkpoints', 'sar_validation', 'novel_discovery',
                   'representation', 'baselines', 'raw_data']:
        (project_root / 'outputs' / subdir).mkdir(parents=True, exist_ok=True)

    # Check prerequisites
    if args.check:
        need_gradient = not args.pretrain
        success = check_prerequisites(need_gradient_matrix=need_gradient)
        sys.exit(0 if success else 1)

    # Determine what to run
    if args.pretrain:
        # Just pretrain
        if not check_prerequisites(need_gradient_matrix=False):
            sys.exit(1)
        success = run_pretrain(epochs=args.epochs)
        sys.exit(0 if success else 1)

    elif args.analysis:
        # Just analysis (exp 2 and 6)
        if not check_prerequisites(need_gradient_matrix=True):
            sys.exit(1)
        results = {
            'exp2': run_experiment_2(),
            'exp6': run_experiment_6(),
        }

    elif args.exp == 2:
        if not check_prerequisites(need_gradient_matrix=True):
            sys.exit(1)
        success = run_experiment_2()
        sys.exit(0 if success else 1)

    elif args.exp == 6:
        if not check_prerequisites(need_gradient_matrix=True):
            sys.exit(1)
        success = run_experiment_6()
        sys.exit(0 if success else 1)

    elif args.exp == 7:
        if not check_prerequisites(need_gradient_matrix=False):
            sys.exit(1)
        success = run_experiment_7()
        sys.exit(0 if success else 1)

    elif args.baselines:
        if not check_prerequisites(need_gradient_matrix=False):
            sys.exit(1)
        success = run_baselines()
        sys.exit(0 if success else 1)

    else:
        # Run everything
        if not check_prerequisites(need_gradient_matrix=False):
            sys.exit(1)
        results = run_all()

    # Summary
    print("\n" + "=" * 60)
    print("LOCAL EXPERIMENTS SUMMARY")
    print("=" * 60)

    for exp, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {exp}: {status}")

    if all(results.values()):
        print("\nAll local experiments completed successfully!")
        print("")
        print("Next steps:")
        print("  1. Upload gradient matrix to HPC:")
        print("     scp outputs/gradients/gnn_conflict_matrices.npz $USER@hpc:gradient/outputs/gradients/")
        print("")
        print("  2. Submit HPC jobs:")
        print("     bash scripts/submit_all_cshl.sh")
        print("")
        sys.exit(0)
    else:
        print("\nSome experiments failed.")
        sys.exit(1)


if __name__ == '__main__':
    main()
