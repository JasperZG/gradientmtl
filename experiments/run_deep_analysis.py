#!/usr/bin/env python3
"""
Master script to run all deep analyses and generate summary report.

Analyses:
1. Gradient Dynamics During Training - When do patterns stabilize?
2. Architecture Robustness - Are patterns architecture-independent?
3. Biological Pathway Recovery - Do gradients match known biology?
4. Threshold Characterization - What overlap % is required?
5. Transfer Learning Validation - Do gradients predict transfer success?

Also trains on SIDER dataset (27 tasks, 100% overlap).
"""

import argparse
import subprocess
import sys
from pathlib import Path
import json
from datetime import datetime


def run_script(script_path, args_list=None, timeout=7200):
    """Run a Python script and capture output."""
    cmd = [sys.executable, str(script_path)]
    if args_list:
        cmd.extend(args_list)

    print(f"\n{'='*60}")
    print(f"Running: {script_path.name}")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent.parent
        )

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        return result.returncode == 0, result.stdout
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {script_path.name} exceeded {timeout}s")
        return False, "TIMEOUT"
    except Exception as e:
        print(f"ERROR: {e}")
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Run all deep analyses")
    parser.add_argument('--skip-sider', action='store_true',
                       help='Skip SIDER training')
    parser.add_argument('--skip-dynamics', action='store_true',
                       help='Skip gradient dynamics analysis')
    parser.add_argument('--skip-architecture', action='store_true',
                       help='Skip architecture robustness')
    parser.add_argument('--skip-pathway', action='store_true',
                       help='Skip pathway recovery')
    parser.add_argument('--skip-threshold', action='store_true',
                       help='Skip threshold characterization')
    parser.add_argument('--skip-transfer', action='store_true',
                       help='Skip transfer learning')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode (fewer epochs)')
    args = parser.parse_args()

    experiments_dir = Path(__file__).parent
    output_dir = Path('outputs/deep_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'analyses': {},
    }

    quick_args = ['--epochs', '20'] if args.quick else []

    # 1. Train SIDER
    if not args.skip_sider:
        success, output = run_script(
            experiments_dir / 'train_sider_gnn.py',
            ['--epochs', '50' if args.quick else '100']
        )
        results_summary['analyses']['sider'] = {
            'success': success,
            'output_dir': 'outputs/sider_gnn',
        }

    # 2. Gradient Dynamics
    if not args.skip_dynamics:
        success, output = run_script(
            experiments_dir / 'deep_analysis_1_gradient_dynamics.py',
            ['--max_epochs', '50' if args.quick else '100']
        )
        results_summary['analyses']['gradient_dynamics'] = {
            'success': success,
        }

    # 3. Architecture Robustness
    if not args.skip_architecture:
        success, output = run_script(
            experiments_dir / 'deep_analysis_2_architecture_robustness.py',
            quick_args
        )
        results_summary['analyses']['architecture_robustness'] = {
            'success': success,
        }

    # 4. Pathway Recovery
    if not args.skip_pathway:
        success, output = run_script(
            experiments_dir / 'deep_analysis_3_pathway_recovery.py',
            []
        )
        results_summary['analyses']['pathway_recovery'] = {
            'success': success,
        }

    # 5. Threshold Characterization
    if not args.skip_threshold:
        success, output = run_script(
            experiments_dir / 'deep_analysis_4_threshold_characterization.py',
            ['--epochs', '20' if args.quick else '30']
        )
        results_summary['analyses']['threshold_characterization'] = {
            'success': success,
        }

    # 6. Transfer Learning
    if not args.skip_transfer:
        n_pairs = '20' if args.quick else None
        transfer_args = ['--source_epochs', '20' if args.quick else '30']
        if n_pairs:
            transfer_args.extend(['--n_pairs', n_pairs])

        success, output = run_script(
            experiments_dir / 'deep_analysis_5_transfer_learning.py',
            transfer_args
        )
        results_summary['analyses']['transfer_learning'] = {
            'success': success,
        }

    # Load individual results and compile summary
    print("\n" + "=" * 60)
    print("DEEP ANALYSIS SUMMARY")
    print("=" * 60)

    # Load SIDER results
    sider_results_path = Path('outputs/sider_gnn/validation_results.json')
    if sider_results_path.exists():
        with open(sider_results_path) as f:
            sider_results = json.load(f)
        print(f"\n1. SIDER (27 tasks, 100% overlap):")
        print(f"   Gradient-Empirical r = {sider_results['pearson_r']:.4f}")
        print(f"   p-value = {sider_results['pearson_p']:.2e}")
        results_summary['analyses']['sider']['results'] = sider_results

    # Load gradient dynamics
    dynamics_path = output_dir / 'gradient_dynamics_results.json'
    if dynamics_path.exists():
        with open(dynamics_path) as f:
            dynamics_results = json.load(f)
        print(f"\n2. Gradient Dynamics:")
        if 'stability_metrics' in dynamics_results:
            stab_epoch = dynamics_results['stability_metrics'].get('stabilization_epoch')
            if stab_epoch:
                print(f"   Patterns stabilize by epoch {stab_epoch}")
            else:
                print(f"   Patterns may not have fully stabilized")
        results_summary['analyses']['gradient_dynamics']['results'] = dynamics_results.get('stability_metrics')

    # Load architecture results
    arch_path = output_dir / 'architecture_robustness_results.json'
    if arch_path.exists():
        with open(arch_path) as f:
            arch_results = json.load(f)
        print(f"\n3. Architecture Robustness:")
        if 'summary' in arch_results:
            print(f"   Mean cross-arch correlation: {arch_results['summary']['mean_correlation']:.4f}")
            print(f"   Min correlation: {arch_results['summary']['min_correlation']:.4f}")
            print(f"   Conclusion: {arch_results['summary']['pass']}")
        results_summary['analyses']['architecture_robustness']['results'] = arch_results.get('summary')

    # Load pathway results
    pathway_path = output_dir / 'pathway_recovery_results.json'
    if pathway_path.exists():
        with open(pathway_path) as f:
            pathway_results = json.load(f)
        print(f"\n4. Biological Pathway Recovery:")
        if 'summary' in pathway_results:
            print(f"   Best ARI: {pathway_results['summary']['max_ari']:.4f}")
            print(f"   Conclusion: {pathway_results['summary']['conclusion']}")
        results_summary['analyses']['pathway_recovery']['results'] = pathway_results.get('summary')

    # Load threshold results
    threshold_path = output_dir / 'threshold_characterization_results.json'
    if threshold_path.exists():
        with open(threshold_path) as f:
            threshold_results = json.load(f)
        print(f"\n5. Threshold Characterization:")
        if 'summary' in threshold_results:
            infl = threshold_results['summary'].get('inflection_point')
            if infl:
                print(f"   Sigmoid inflection point: {infl:.1f}%")
            sig_thresh = threshold_results['summary'].get('significant_threshold')
            if sig_thresh:
                print(f"   r > 0.5 at: {sig_thresh}% overlap")
        results_summary['analyses']['threshold_characterization']['results'] = threshold_results.get('summary')

    # Load transfer results
    transfer_path = output_dir / 'transfer_learning_results.json'
    if transfer_path.exists():
        with open(transfer_path) as f:
            transfer_results = json.load(f)
        print(f"\n6. Transfer Learning Validation:")
        if 'analysis' in transfer_results:
            r = transfer_results['analysis'].get('pearson_r')
            if r:
                print(f"   Gradient-Transfer correlation: {r:.4f}")
            print(f"   Conclusion: {transfer_results['analysis']['conclusion']}")
        results_summary['analyses']['transfer_learning']['results'] = transfer_results.get('analysis')

    # Overall conclusion
    print("\n" + "=" * 60)
    print("OVERALL CONCLUSION")
    print("=" * 60)

    passes = 0
    total = 0

    conclusions = {
        'sider': lambda r: r.get('pearson_r', 0) > 0.6,
        'architecture_robustness': lambda r: r.get('min_correlation', 0) > 0.6,
        'pathway_recovery': lambda r: r.get('max_ari', 0) > 0.3,
        'threshold_characterization': lambda r: r.get('inflection_point') is not None,
        'transfer_learning': lambda r: (r.get('pearson_r') or 0) > 0.3,
    }

    for analysis, check_fn in conclusions.items():
        if analysis in results_summary['analyses']:
            res = results_summary['analyses'][analysis].get('results', {})
            if res:
                total += 1
                if check_fn(res):
                    passes += 1

    if total > 0:
        print(f"\nPassed {passes}/{total} analyses")
        if passes == total:
            print("\n>>> ALL ANALYSES PASSED")
        elif passes >= total * 0.6:
            print("\n>>> MAJORITY PASSED - Results are promising")
        else:
            print("\n>>> MORE WORK NEEDED")

    # Save summary
    with open(output_dir / 'deep_analysis_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nFull summary saved to: {output_dir / 'deep_analysis_summary.json'}")


if __name__ == '__main__':
    main()
