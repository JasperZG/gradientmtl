#!/usr/bin/env python3
"""
Experiment 14: Negative Transfer Prediction

Tests whether gradient similarity can predict negative transfer
before it occurs. Uses existing transfer learning results from
Tox21 and kinase experiments.

Key question: Can G serve as a screening tool to avoid bad transfers?
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import json


def load_transfer_results():
    """Load all available transfer learning results."""
    all_results = []

    # Tox21 transfer results
    exp3_dir = Path('outputs/experiment3')
    if exp3_dir.exists():
        for csv_path in exp3_dir.glob('**/transfer_results*.csv'):
            try:
                df = pd.read_csv(csv_path)
                df['dataset'] = 'Tox21'
                all_results.append(df)
            except:
                continue

        # Also try aggregated results
        agg_path = exp3_dir / 'aggregated_results.csv'
        if agg_path.exists():
            try:
                df = pd.read_csv(agg_path)
                df['dataset'] = 'Tox21'
                all_results.append(df)
            except:
                pass

    # Kinase transfer results
    kinase_path = Path('outputs/kinase_phase2/transfer_results.csv')
    if kinase_path.exists():
        try:
            df = pd.read_csv(kinase_path)
            df['dataset'] = 'Kinase'
            all_results.append(df)
        except:
            pass

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        return combined
    return None


def load_gradient_matrices():
    """Load gradient matrices for all datasets."""
    matrices = {}

    # Tox21
    for path in ['outputs/gradients/gnn_conflict_matrices.npz',
                  'outputs/gradients/gradient_matrices.npz']:
        if Path(path).exists():
            data = np.load(path, allow_pickle=True)
            G_key = 'average_matrix' if 'average_matrix' in data else 'averaged'
            t_key = 'tasks' if 'tasks' in data else 'task_names'
            matrices['Tox21'] = {
                'G': data[G_key],
                'tasks': list(data[t_key])
            }
            break

    # Kinase
    kinase_path = 'outputs/kinase_all_results/gradient_matrices.npz'
    if Path(kinase_path).exists():
        data = np.load(kinase_path, allow_pickle=True)
        matrices['Kinase'] = {
            'G': data['average_matrix'],
            'tasks': list(data['tasks'])
        }

    return matrices


def analyze_negative_transfer(transfer_df, gradient_matrices, output_dir):
    """Analyze G as predictor of negative transfer."""

    print("\n--- Negative Transfer Prediction ---")

    # Identify columns
    benefit_col = None
    for col in ['benefit', 'transfer_benefit', 'improvement', 'delta']:
        if col in transfer_df.columns:
            benefit_col = col
            break

    g_col = None
    for col in ['gradient_G', 'gradient_similarity', 'G', 'gradient_correlation']:
        if col in transfer_df.columns:
            g_col = col
            break

    if benefit_col is None or g_col is None:
        print(f"Available columns: {list(transfer_df.columns)}")
        print("Cannot find benefit or gradient columns. Trying to compute from raw data...")

        # Try to compute from source/target columns
        source_col = None
        target_col = None
        for col in transfer_df.columns:
            if 'source' in col.lower():
                source_col = col
            if 'target' in col.lower():
                target_col = col

        if source_col and target_col and benefit_col is None:
            # Look for scratch/transfer columns to compute benefit
            for s_col in ['scratch_score', 'scratch_r', 'scratch']:
                for t_col in ['transfer_score', 'transfer_r', 'transfer']:
                    if s_col in transfer_df.columns and t_col in transfer_df.columns:
                        transfer_df['benefit'] = transfer_df[t_col] - transfer_df[s_col]
                        benefit_col = 'benefit'
                        break

        if source_col and target_col and g_col is None:
            # Look up G from gradient matrix
            g_values = []
            for _, row in transfer_df.iterrows():
                src = row[source_col]
                tgt = row[target_col]
                ds = row.get('dataset', 'Tox21')
                if ds in gradient_matrices:
                    tasks = gradient_matrices[ds]['tasks']
                    G = gradient_matrices[ds]['G']
                    # Match task names
                    src_match = [i for i, t in enumerate(tasks) if src in t or t in src]
                    tgt_match = [i for i, t in enumerate(tasks) if tgt in t or t in tgt]
                    if src_match and tgt_match:
                        g_values.append(G[src_match[0], tgt_match[0]])
                    else:
                        g_values.append(np.nan)
                else:
                    g_values.append(np.nan)
            transfer_df['gradient_G'] = g_values
            g_col = 'gradient_G'

    if benefit_col is None or g_col is None:
        print("ERROR: Could not determine benefit or gradient columns")
        print(f"Columns available: {list(transfer_df.columns)}")
        return None

    # Filter valid rows
    valid = transfer_df.dropna(subset=[benefit_col, g_col])
    print(f"Valid transfer experiments: {len(valid)}")

    if len(valid) < 10:
        print("Too few valid experiments for analysis")
        return None

    benefits = valid[benefit_col].values
    g_values = valid[g_col].values

    # Basic correlation
    r, p = stats.pearsonr(g_values, benefits)
    print(f"\nCorrelation: r(G, benefit) = {r:.3f}, p = {p:.2e}")

    # Binary classification: can G predict negative transfer?
    is_negative = (benefits < 0).astype(int)
    n_negative = is_negative.sum()
    n_positive = len(is_negative) - n_negative
    print(f"\nNegative transfers: {n_negative}/{len(is_negative)} ({n_negative/len(is_negative):.1%})")

    results = {
        'n_experiments': len(valid),
        'n_negative_transfer': int(n_negative),
        'n_positive_transfer': int(n_positive),
        'pct_negative': round(n_negative / len(is_negative), 3),
        'correlation_r': round(r, 4),
        'correlation_p': float(f'{p:.2e}'),
    }

    # ROC-AUC: higher G → less likely negative transfer
    # So we predict "positive transfer" with G, and compute AUC
    if n_negative > 0 and n_positive > 0:
        is_positive = 1 - is_negative
        auc = roc_auc_score(is_positive, g_values)
        print(f"AUC (G predicts positive transfer): {auc:.3f}")
        results['auc_positive_transfer'] = round(auc, 4)

        # Precision-recall for negative transfer avoidance
        # "If G < threshold, predict negative transfer"
        ap = average_precision_score(is_negative, -g_values)
        print(f"Average Precision (detecting negative transfer): {ap:.3f}")
        results['avg_precision_negative'] = round(ap, 4)

        # Threshold analysis: what if we only transfer when G > threshold?
        print("\n--- Threshold Analysis ---")
        print(f"{'Threshold':>10} {'N kept':>8} {'Neg avoided':>12} {'Pos kept':>10} {'Precision':>10}")

        thresholds = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]
        threshold_results = []
        for thresh in thresholds:
            kept = valid[g_values >= thresh]
            n_kept = len(kept)
            if n_kept == 0:
                continue
            kept_benefits = kept[benefit_col].values
            n_neg_kept = (kept_benefits < 0).sum()
            n_pos_kept = (kept_benefits >= 0).sum()
            precision = n_pos_kept / n_kept if n_kept > 0 else 0

            n_neg_avoided = n_negative - n_neg_kept
            print(f"{thresh:>10.2f} {n_kept:>8d} {n_neg_avoided:>12d} {n_pos_kept:>10d} {precision:>10.3f}")

            threshold_results.append({
                'threshold': thresh,
                'n_kept': n_kept,
                'n_negative_avoided': n_neg_avoided,
                'n_positive_kept': n_pos_kept,
                'precision': round(precision, 3),
            })

        results['threshold_analysis'] = threshold_results

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Panel A: G vs benefit
        ax = axes[0]
        colors = ['#e41a1c' if b < 0 else '#4daf4a' for b in benefits]
        ax.scatter(g_values, benefits, c=colors, alpha=0.5, edgecolors='k', linewidths=0.3)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Gradient Similarity (G)')
        ax.set_ylabel('Transfer Benefit')
        ax.set_title(f'A) G vs Transfer Benefit (r={r:.3f})')

        # Panel B: Threshold analysis
        ax = axes[1]
        if threshold_results:
            t_vals = [t['threshold'] for t in threshold_results]
            p_vals = [t['precision'] for t in threshold_results]
            k_vals = [t['n_kept'] / len(valid) for t in threshold_results]
            ax.plot(t_vals, p_vals, 'b-o', linewidth=2, label='Precision (positive transfer)')
            ax.plot(t_vals, k_vals, 'r--s', linewidth=2, label='Fraction of experiments kept')
            ax.set_xlabel('G Threshold')
            ax.set_ylabel('Rate')
            ax.set_title('B) Screening with G threshold')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'negative_transfer_analysis.png', dpi=150)
        plt.close()

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='outputs/experiment14_negative_transfer')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Experiment 14: Negative Transfer Prediction")
    print("=" * 60)

    # Load data
    print("\nLoading transfer results...")
    transfer_df = load_transfer_results()

    if transfer_df is None or len(transfer_df) == 0:
        print("No transfer learning results found.")
        print("Expected locations:")
        print("  outputs/experiment3/")
        print("  outputs/kinase_phase2/transfer_results.csv")
        return

    print(f"Loaded {len(transfer_df)} transfer experiments")

    print("\nLoading gradient matrices...")
    grad_matrices = load_gradient_matrices()
    print(f"Loaded matrices for: {list(grad_matrices.keys())}")

    # Analyze
    results = analyze_negative_transfer(transfer_df, grad_matrices, output_dir)

    if results:
        with open(output_dir / 'summary.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSaved to {output_dir}/")


if __name__ == '__main__':
    main()
