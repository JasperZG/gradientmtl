#!/usr/bin/env python3
"""
Local runner for lightweight experiments.

Runs experiments 2 (SAR Validation) and 6 (Novel Discovery) locally.
These only require the gradient matrix - no GPU training needed.

Prerequisites:
    - Gradient matrix must exist at outputs/gradients/gnn_conflict_matrices.npz
    - Either run pretrain locally first, or download from HPC after pretrain completes

Usage:
    python scripts/run_local.py                    # Run both exp 2 and 6
    python scripts/run_local.py --exp 2            # Run only exp 2
    python scripts/run_local.py --exp 6            # Run only exp 6
    python scripts/run_local.py --check            # Check if prerequisites exist
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_prerequisites():
    """Check if required files exist."""
    gradient_matrix = project_root / 'outputs' / 'gradients' / 'gnn_conflict_matrices.npz'

    print("Checking prerequisites...")
    print(f"  Gradient matrix: {gradient_matrix}")

    if gradient_matrix.exists():
        print("  [OK] Gradient matrix found")
        return True
    else:
        print("  [MISSING] Gradient matrix not found")
        print("")
        print("To generate the gradient matrix, either:")
        print("  1. Run pretrain locally:  python train_tox21_gnn.py --epochs 100")
        print("  2. Run pretrain on HPC and download outputs/gradients/gnn_conflict_matrices.npz")
        return False


def run_experiment_2():
    """Run SAR Validation experiment."""
    print("\n" + "=" * 60)
    print("Running Experiment 2: SAR Validation")
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
        print("\n[SUCCESS] Experiment 2 completed")
        print(f"Results saved to: {output_dir}")
    else:
        print(f"\n[ERROR] Experiment 2 failed with code {result.returncode}")

    return result.returncode == 0


def run_experiment_6():
    """Run Novel Discovery experiment."""
    print("\n" + "=" * 60)
    print("Running Experiment 6: Novel Trade-off Discovery")
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
        print("\n[SUCCESS] Experiment 6 completed")
        print(f"Results saved to: {output_dir}")
    else:
        print(f"\n[ERROR] Experiment 6 failed with code {result.returncode}")

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='Run local experiments')
    parser.add_argument('--exp', type=int, choices=[2, 6], default=None,
                       help='Specific experiment to run (default: both)')
    parser.add_argument('--check', action='store_true',
                       help='Only check prerequisites')
    parser.add_argument('--skip-check', action='store_true',
                       help='Skip prerequisite check')

    args = parser.parse_args()

    # Check prerequisites
    if args.check:
        success = check_prerequisites()
        sys.exit(0 if success else 1)

    if not args.skip_check:
        if not check_prerequisites():
            print("\nPrerequisites not met. Use --skip-check to run anyway.")
            sys.exit(1)

    # Create output directories
    (project_root / 'outputs' / 'sar_validation').mkdir(parents=True, exist_ok=True)
    (project_root / 'outputs' / 'novel_discovery').mkdir(parents=True, exist_ok=True)

    # Run experiments
    results = {}

    if args.exp is None or args.exp == 2:
        results['exp2'] = run_experiment_2()

    if args.exp is None or args.exp == 6:
        results['exp6'] = run_experiment_6()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for exp, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {exp}: {status}")

    if all(results.values()):
        print("\nAll experiments completed successfully!")
        sys.exit(0)
    else:
        print("\nSome experiments failed.")
        sys.exit(1)


if __name__ == '__main__':
    main()
