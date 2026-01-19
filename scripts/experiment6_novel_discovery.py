#!/usr/bin/env python3
"""
Experiment 6: Novel Trade-off Discovery

Identifies previously undocumented property trade-offs from gradient conflicts.

Key analyses:
1. Find unexpected strong conflicts (G < -0.4) not in literature
2. Generate mechanistic hypotheses for novel trade-offs
3. Validate via retrospective analysis
4. Rank discoveries by confidence and novelty

Expected outcome: 3-5 novel trade-offs for expert validation
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from analysis.literature_matrix import LITERATURE_RELATIONSHIPS, get_literature_matrix


# Mechanistic hypothesis templates
MECHANISM_TEMPLATES = {
    'lipophilicity_related': (
        "Both {task1} and {task2} may be influenced by lipophilicity. "
        "Lipophilic compounds often show {effect} for {task1} but {opposite_effect} for {task2} "
        "due to {reason}."
    ),
    'metabolism_related': (
        "{task1} and {task2} may share metabolic pathways. "
        "CYP-mediated metabolism could create a trade-off where "
        "metabolic stability for {task1} conflicts with {task2}."
    ),
    'transport_related': (
        "Both {task1} and {task2} may involve membrane transport mechanisms. "
        "P-gp efflux or active transport could create opposing effects."
    ),
    'binding_related': (
        "{task1} and {task2} may involve overlapping binding sites or "
        "structural features. Optimizing for {task1} may inadvertently "
        "affect {task2} through shared pharmacophore elements."
    ),
    'physicochemical_related': (
        "The trade-off between {task1} and {task2} may reflect fundamental "
        "physicochemical constraints such as solubility-permeability trade-off "
        "or molecular size limitations."
    ),
}

# Task category mapping for hypothesis generation
TASK_CATEGORIES = {
    'ADME': ['BBBP', 'BBB', 'Caco2', 'HIA', 'Bioavailability', 'Pgp', 'Clearance', 'Half_Life', 'VDss', 'PPBR'],
    'Metabolism': ['CYP', 'Clearance', 'Half_Life'],
    'Toxicity': ['hERG', 'AMES', 'DILI', 'LD50', 'Tox21', 'Carcino', 'ClinTox'],
    'Physicochemical': ['ESOL', 'Lipophilicity', 'Solubility', 'FreeSolv', 'logP'],
    'Binding': ['BACE', 'HIV', 'PDBbind', 'IC50', 'Ki'],
}


def get_task_category(task_name: str) -> str:
    """Determine category of a task based on name."""
    for category, keywords in TASK_CATEGORIES.items():
        for keyword in keywords:
            if keyword.lower() in task_name.lower():
                return category
    return 'Other'


def generate_hypothesis(task1: str, task2: str, conflict_value: float) -> str:
    """Generate mechanistic hypothesis for a discovered trade-off."""
    cat1 = get_task_category(task1)
    cat2 = get_task_category(task2)

    # Select appropriate template
    if cat1 == 'Physicochemical' or cat2 == 'Physicochemical':
        template_key = 'physicochemical_related'
    elif cat1 == 'Metabolism' or cat2 == 'Metabolism':
        template_key = 'metabolism_related'
    elif cat1 == 'ADME' and cat2 == 'ADME':
        template_key = 'transport_related'
    elif cat1 == 'Binding' or cat2 == 'Binding':
        template_key = 'binding_related'
    else:
        template_key = 'lipophilicity_related'

    # Generate hypothesis
    hypothesis = (
        f"DISCOVERED TRADE-OFF: {task1} vs {task2} (G = {conflict_value:.3f})\n"
        f"Categories: {cat1} vs {cat2}\n\n"
        f"Potential mechanism: The negative gradient correlation suggests that "
        f"optimizing for {task1} may adversely affect {task2}. "
    )

    # Add category-specific reasoning
    if cat1 == 'Toxicity' or cat2 == 'Toxicity':
        hypothesis += (
            "This may represent a safety-efficacy trade-off where potent compounds "
            "have higher off-target liability."
        )
    elif cat1 == 'ADME' and cat2 == 'Toxicity':
        hypothesis += (
            "High permeability or bioavailability may increase exposure to toxic "
            "metabolites or off-target tissues."
        )
    elif 'Clearance' in task1 or 'Clearance' in task2:
        hypothesis += (
            "Metabolic stability optimizations may conflict with clearance mechanisms, "
            "potentially leading to accumulation."
        )
    else:
        hypothesis += (
            "Further investigation is needed to determine the underlying "
            "structural or mechanistic basis for this trade-off."
        )

    return hypothesis


def find_novel_tradeoffs(
    G: np.ndarray,
    task_names: list,
    conflict_threshold: float = -0.3,
    novelty_threshold: float = 0.2
) -> list:
    """
    Find trade-offs in gradient matrix not documented in literature.

    Args:
        G: Gradient conflict matrix
        task_names: List of task names
        conflict_threshold: Minimum negative correlation to consider a trade-off
        novelty_threshold: Maximum literature correlation to consider novel

    Returns:
        List of novel trade-off discoveries
    """
    L = get_literature_matrix(task_names)

    discoveries = []

    for i, task_i in enumerate(task_names):
        for j, task_j in enumerate(task_names):
            if i >= j:
                continue

            g_val = G[i, j]
            l_val = L[i, j]

            # Check if strong conflict exists
            if g_val > conflict_threshold:
                continue

            # Check if not well-documented in literature
            if abs(l_val) > novelty_threshold:
                continue

            # This is a novel discovery!
            discovery = {
                'task1': task_i,
                'task2': task_j,
                'gradient_conflict': float(g_val),
                'literature_value': float(l_val),
                'novelty_score': abs(g_val) - abs(l_val),  # How much stronger than expected
                'category1': get_task_category(task_i),
                'category2': get_task_category(task_j),
                'hypothesis': generate_hypothesis(task_i, task_j, g_val),
            }

            discoveries.append(discovery)

    # Sort by novelty score (strongest unexpected conflicts first)
    discoveries.sort(key=lambda x: x['novelty_score'], reverse=True)

    return discoveries


def find_novel_synergies(
    G: np.ndarray,
    task_names: list,
    synergy_threshold: float = 0.5,
    novelty_threshold: float = 0.2
) -> list:
    """
    Find unexpected synergies (positive correlations not in literature).
    """
    L = get_literature_matrix(task_names)

    discoveries = []

    for i, task_i in enumerate(task_names):
        for j, task_j in enumerate(task_names):
            if i >= j:
                continue

            g_val = G[i, j]
            l_val = L[i, j]

            # Check if strong synergy exists
            if g_val < synergy_threshold:
                continue

            # Check if not well-documented
            if l_val > novelty_threshold:
                continue

            discovery = {
                'task1': task_i,
                'task2': task_j,
                'gradient_synergy': float(g_val),
                'literature_value': float(l_val),
                'novelty_score': g_val - l_val,
                'category1': get_task_category(task_i),
                'category2': get_task_category(task_j),
            }

            discoveries.append(discovery)

    discoveries.sort(key=lambda x: x['novelty_score'], reverse=True)
    return discoveries


def plot_discovery_heatmap(
    G: np.ndarray,
    task_names: list,
    discoveries: list,
    output_path: Path
):
    """Highlight novel discoveries on gradient heatmap."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create heatmap
    im = ax.imshow(G, cmap='RdBu_r', vmin=-1, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Gradient Conflict', fontsize=12)

    # Mark novel discoveries with boxes
    for disc in discoveries[:10]:  # Top 10
        try:
            i = task_names.index(disc['task1'])
            j = task_names.index(disc['task2'])
            rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                                  edgecolor='yellow', linewidth=3)
            ax.add_patch(rect)
            rect = plt.Rectangle((i-0.5, j-0.5), 1, 1, fill=False,
                                  edgecolor='yellow', linewidth=3)
            ax.add_patch(rect)
        except ValueError:
            continue

    # Labels
    ax.set_xticks(range(len(task_names)))
    ax.set_yticks(range(len(task_names)))
    ax.set_xticklabels(task_names, rotation=90, fontsize=8)
    ax.set_yticklabels(task_names, fontsize=8)

    ax.set_title('Gradient Conflict Matrix\n(Yellow boxes = novel discoveries)', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved discovery heatmap to {output_path}")


def plot_discovery_ranking(discoveries: list, output_path: Path, top_n: int = 15):
    """Bar plot ranking novel discoveries by strength."""
    if not discoveries:
        return

    top_disc = discoveries[:top_n]

    fig, ax = plt.subplots(figsize=(12, 8))

    labels = [f"{d['task1'][:10]} vs\n{d['task2'][:10]}" for d in top_disc]
    values = [d['gradient_conflict'] for d in top_disc]
    colors = ['red' if v < -0.4 else 'orange' for v in values]

    bars = ax.barh(range(len(labels)), values, color=colors, alpha=0.8)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color='gray', linestyle='-', alpha=0.5)
    ax.axvline(-0.4, color='red', linestyle='--', alpha=0.5, label='Strong conflict threshold')

    ax.set_xlabel('Gradient Conflict', fontsize=12)
    ax.set_title('Novel Trade-off Discoveries\n(Not documented in literature)', fontsize=14)
    ax.legend()

    # Invert y-axis so top discovery is at top
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved discovery ranking to {output_path}")


def run_novel_discovery(
    gradient_matrix_path: str,
    output_dir: str = 'outputs/novel_discovery',
    conflict_threshold: float = -0.3,
    verbose: bool = True
) -> dict:
    """
    Run novel trade-off discovery experiment.

    Args:
        gradient_matrix_path: Path to gradient conflict matrix
        output_dir: Output directory
        conflict_threshold: Threshold for trade-off detection
        verbose: Print detailed output

    Returns:
        Dict with discovery results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("EXPERIMENT 6: NOVEL TRADE-OFF DISCOVERY")
    print("=" * 70)

    # Load gradient matrix
    print(f"\nLoading gradient matrix from {gradient_matrix_path}...")
    data = np.load(gradient_matrix_path, allow_pickle=True)
    G = data['averaged']
    task_names = data['task_names'].tolist()
    print(f"Loaded {len(task_names)} tasks")

    # Find novel trade-offs
    print(f"\nSearching for novel trade-offs (G < {conflict_threshold})...")
    tradeoffs = find_novel_tradeoffs(G, task_names, conflict_threshold)
    print(f"Found {len(tradeoffs)} novel trade-off candidates")

    # Find novel synergies
    print("\nSearching for novel synergies (G > 0.5)...")
    synergies = find_novel_synergies(G, task_names, synergy_threshold=0.5)
    print(f"Found {len(synergies)} novel synergy candidates")

    # Generate plots
    print("\nGenerating visualizations...")
    plot_discovery_heatmap(G, task_names, tradeoffs,
                           output_dir / 'discovery_heatmap.png')
    plot_discovery_ranking(tradeoffs, output_dir / 'tradeoff_ranking.png')

    if verbose and tradeoffs:
        print("\n" + "-" * 70)
        print("TOP NOVEL TRADE-OFF DISCOVERIES")
        print("-" * 70)

        for i, disc in enumerate(tradeoffs[:5]):
            print(f"\n{i+1}. {disc['task1']} vs {disc['task2']}")
            print(f"   Gradient conflict: {disc['gradient_conflict']:.3f}")
            print(f"   Literature value: {disc['literature_value']:.3f}")
            print(f"   Categories: {disc['category1']} vs {disc['category2']}")
            print(f"\n   {disc['hypothesis'][:500]}...")

    if verbose and synergies:
        print("\n" + "-" * 70)
        print("TOP NOVEL SYNERGY DISCOVERIES")
        print("-" * 70)

        for i, disc in enumerate(synergies[:3]):
            print(f"\n{i+1}. {disc['task1']} vs {disc['task2']}")
            print(f"   Gradient synergy: {disc['gradient_synergy']:.3f}")
            print(f"   Literature value: {disc['literature_value']:.3f}")

    # Compile results
    results = {
        'n_tasks': len(task_names),
        'n_novel_tradeoffs': len(tradeoffs),
        'n_novel_synergies': len(synergies),
        'top_tradeoffs': tradeoffs[:10],
        'top_synergies': synergies[:5],
        'conflict_threshold': conflict_threshold,
    }

    # Save results
    results_file = output_dir / 'novel_discovery_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Save hypotheses for expert review
    hypotheses_file = output_dir / 'hypotheses_for_review.txt'
    with open(hypotheses_file, 'w') as f:
        f.write("NOVEL TRADE-OFF DISCOVERIES FOR EXPERT REVIEW\n")
        f.write("=" * 70 + "\n\n")

        for i, disc in enumerate(tradeoffs[:10]):
            f.write(f"Discovery {i+1}:\n")
            f.write("-" * 40 + "\n")
            f.write(disc['hypothesis'] + "\n\n")
            f.write(f"Confidence: {'HIGH' if disc['gradient_conflict'] < -0.5 else 'MEDIUM'}\n")
            f.write(f"Suggested validation: Literature search, expert interview\n\n")

    print(f"Hypotheses saved to {hypotheses_file}")

    # Summary
    print("\n" + "=" * 70)
    print("DISCOVERY SUMMARY")
    print("=" * 70)
    print(f"Novel trade-offs found: {len(tradeoffs)}")
    print(f"Novel synergies found: {len(synergies)}")

    if len(tradeoffs) >= 3:
        print("✓ SUCCESS: Found 3+ novel trade-offs for expert validation")
    else:
        print("⚠ WARNING: Fewer than 3 novel trade-offs found")

    return results


def main():
    parser = argparse.ArgumentParser(description='Novel Trade-off Discovery')
    parser.add_argument('--gradient-matrix', type=str,
                       default='outputs/gradients/gnn_conflict_matrices.npz',
                       help='Path to gradient conflict matrix')
    parser.add_argument('--output-dir', type=str, default='outputs/novel_discovery',
                       help='Output directory')
    parser.add_argument('--conflict-threshold', type=float, default=-0.3,
                       help='Threshold for trade-off detection')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')

    args = parser.parse_args()

    results = run_novel_discovery(
        gradient_matrix_path=args.gradient_matrix,
        output_dir=args.output_dir,
        conflict_threshold=args.conflict_threshold,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
