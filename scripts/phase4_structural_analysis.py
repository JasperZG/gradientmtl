#!/usr/bin/env python3
"""
Phase 4B: Structural Analysis

Cluster kinases by gradient similarity and compare to:
1. Kinase phylogenetic tree / sequence similarity
2. Binding site structural similarity
3. Known kinase classification (Manning et al.)

Key outputs:
1. Gradient-based clustering dendrogram
2. Comparison to kinase phylogeny
3. Cluster composition analysis
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import seaborn as sns


# Manning kinase classification (simplified)
# Based on Manning et al. Science 2002 "The Protein Kinase Complement of the Human Genome"
KINASE_GROUPS = {
    'CMGC': ['CDK1', 'CDK2', 'CDK4', 'CDK5', 'CDK6', 'CDK7', 'CDK9', 'GSK3A', 'GSK3B', 'DYRK1A', 'CLK1'],
    'AGC': ['AKT1', 'AKT2', 'PKA', 'PKC', 'RSK1', 'RSK2', 'SGK1', 'ROCK1', 'ROCK2'],
    'TK': ['EGFR', 'ERBB2', 'ERBB3', 'ERBB4', 'SRC', 'FYN', 'LCK', 'LYN', 'ABL1', 'JAK1', 'JAK2', 'JAK3', 'TYK2', 'KIT', 'FLT3', 'PDGFRA', 'VEGFR1', 'VEGFR2'],
    'TKL': ['BRAF', 'RAF1', 'ARAF', 'MAP3K1', 'MAP3K7'],
    'STE': ['MAP2K1', 'MAP2K2', 'PAK1', 'PAK2'],
    'CK1': ['CK1A', 'CK1D', 'CK1E'],
    'Other': ['AURKA', 'AURKB', 'AURKC', 'PLK1', 'PLK2', 'PLK3', 'PLK4', 'NEK1', 'NEK2'],
}

# Reverse mapping
KINASE_TO_GROUP = {}
for group, kinases in KINASE_GROUPS.items():
    for k in kinases:
        KINASE_TO_GROUP[k] = group


def load_gradient_matrix(results_dir: str) -> tuple:
    """Load gradient matrix and task names."""
    data = np.load(f"{results_dir}/gradient_matrices.npz", allow_pickle=True)
    G = data['average_matrix']
    tasks = list(data['tasks'])
    return G, tasks


def cluster_by_gradient(G: np.ndarray, tasks: list, output_dir: str):
    """Perform hierarchical clustering based on gradient similarity."""

    # Convert correlation to distance (1 - correlation)
    # Ensure symmetric and valid distance matrix
    G_sym = (G + G.T) / 2
    np.fill_diagonal(G_sym, 1.0)

    # Clip to valid range
    G_clipped = np.clip(G_sym, -1, 1)
    distance_matrix = 1 - G_clipped

    # Ensure non-negative distances
    distance_matrix = np.maximum(distance_matrix, 0)

    # Convert to condensed form for hierarchical clustering
    condensed = squareform(distance_matrix, checks=False)

    # Perform hierarchical clustering
    linkage = hierarchy.linkage(condensed, method='average')

    # Clean task names
    clean_tasks = [t.replace('_pIC50', '') for t in tasks]

    # Create dendrogram
    plt.figure(figsize=(12, 8))
    hierarchy.dendrogram(
        linkage,
        labels=clean_tasks,
        leaf_rotation=45,
        leaf_font_size=10
    )
    plt.title('Kinase Clustering by Gradient Similarity', fontsize=14)
    plt.xlabel('Kinase', fontsize=12)
    plt.ylabel('Distance (1 - Gradient Correlation)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/gradient_clustering_dendrogram.png", dpi=150)
    plt.close()

    print(f"Saved dendrogram to {output_dir}/gradient_clustering_dendrogram.png")

    return linkage, clean_tasks


def get_clusters(linkage: np.ndarray, tasks: list, n_clusters: int = 4) -> dict:
    """Extract cluster assignments at specified number of clusters."""
    cluster_ids = hierarchy.fcluster(linkage, n_clusters, criterion='maxclust')

    clusters = {}
    for i, (task, cid) in enumerate(zip(tasks, cluster_ids)):
        if cid not in clusters:
            clusters[cid] = []
        clusters[cid].append(task)

    return clusters


def compare_to_phylogeny(clusters: dict, output_dir: str) -> dict:
    """Compare gradient clusters to Manning kinase classification."""

    comparison = {
        'cluster_composition': {},
        'group_distribution': {},
        'consistency_score': 0
    }

    total_pairs = 0
    same_group_same_cluster = 0

    for cid, kinases in clusters.items():
        # Get Manning groups for each kinase in cluster
        groups = [KINASE_TO_GROUP.get(k, 'Unknown') for k in kinases]
        group_counts = pd.Series(groups).value_counts().to_dict()

        comparison['cluster_composition'][f'Cluster_{cid}'] = {
            'kinases': kinases,
            'manning_groups': group_counts,
            'dominant_group': max(group_counts, key=group_counts.get) if group_counts else 'Unknown'
        }

        # Count pairs within cluster from same Manning group
        for i, k1 in enumerate(kinases):
            for k2 in kinases[i+1:]:
                total_pairs += 1
                g1 = KINASE_TO_GROUP.get(k1, 'Unknown')
                g2 = KINASE_TO_GROUP.get(k2, 'Unknown')
                if g1 == g2 and g1 != 'Unknown':
                    same_group_same_cluster += 1

    if total_pairs > 0:
        comparison['consistency_score'] = round(same_group_same_cluster / total_pairs, 3)

    # Save comparison
    with open(f"{output_dir}/phylogeny_comparison.json", 'w') as f:
        json.dump(comparison, f, indent=2)

    return comparison


def generate_clustered_heatmap(G: np.ndarray, tasks: list, linkage: np.ndarray, output_dir: str):
    """Generate heatmap with hierarchical clustering."""

    clean_tasks = [t.replace('_pIC50', '') for t in tasks]

    # Get cluster order from dendrogram
    dendro = hierarchy.dendrogram(linkage, no_plot=True)
    order = dendro['leaves']

    # Reorder matrix
    G_ordered = G[np.ix_(order, order)]
    tasks_ordered = [clean_tasks[i] for i in order]

    # Get Manning groups for coloring
    groups = [KINASE_TO_GROUP.get(t, 'Unknown') for t in tasks_ordered]
    group_colors = {
        'CMGC': '#e41a1c',
        'TK': '#377eb8',
        'Other': '#4daf4a',
        'AGC': '#984ea3',
        'TKL': '#ff7f00',
        'Unknown': '#999999'
    }
    row_colors = [group_colors.get(g, '#999999') for g in groups]

    # Create clustermap
    plt.figure(figsize=(14, 12))

    g = sns.clustermap(
        G_ordered,
        xticklabels=tasks_ordered,
        yticklabels=tasks_ordered,
        cmap='RdYlGn',
        center=0,
        row_cluster=False,
        col_cluster=False,
        row_colors=row_colors,
        col_colors=row_colors,
        figsize=(14, 12),
        cbar_kws={'label': 'Gradient Correlation'}
    )

    g.ax_heatmap.set_xlabel('Kinase', fontsize=12)
    g.ax_heatmap.set_ylabel('Kinase', fontsize=12)
    plt.suptitle('Gradient Correlation Matrix (Clustered)\nColors: Manning kinase groups', y=1.02, fontsize=14)

    plt.savefig(f"{output_dir}/gradient_clustered_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved clustered heatmap to {output_dir}/gradient_clustered_heatmap.png")


def generate_structural_report(clusters: dict, comparison: dict, output_dir: str):
    """Generate structural analysis report."""

    lines = [
        "# Structural Analysis Report",
        "",
        "## Overview",
        "",
        "This report compares gradient-based kinase clustering to established",
        "kinase phylogeny (Manning et al. 2002 classification).",
        "",
        "## Cluster Composition",
        ""
    ]

    for cid, info in comparison['cluster_composition'].items():
        lines.extend([
            f"### {cid}",
            "",
            f"**Kinases:** {', '.join(info['kinases'])}",
            "",
            f"**Dominant Manning group:** {info['dominant_group']}",
            "",
            "Manning group breakdown:",
            ""
        ])
        for group, count in info['manning_groups'].items():
            lines.append(f"- {group}: {count}")
        lines.extend(["", "---", ""])

    # Consistency analysis
    lines.extend([
        "## Consistency with Kinase Phylogeny",
        "",
        f"**Consistency score:** {comparison['consistency_score']:.1%}",
        "",
        "The consistency score measures how often kinases from the same Manning",
        "group are placed in the same gradient cluster.",
        "",
        "| Score | Interpretation |",
        "|-------|----------------|",
        "| > 0.7 | Strong agreement with phylogeny |",
        "| 0.4-0.7 | Moderate agreement |",
        "| < 0.4 | Gradient patterns diverge from sequence similarity |",
        "",
    ])

    # Interpretation
    score = comparison['consistency_score']
    if score > 0.7:
        lines.append("**Interpretation:** Gradient clustering strongly agrees with kinase phylogeny,")
        lines.append("suggesting ligand binding preferences follow evolutionary relationships.")
    elif score > 0.4:
        lines.append("**Interpretation:** Moderate agreement with phylogeny. Some clusters")
        lines.append("group kinases by functional similarity rather than sequence identity.")
    else:
        lines.append("**Interpretation:** Low phylogeny consistency suggests gradient patterns")
        lines.append("capture ligand-binding features that diverge from sequence similarity.")
        lines.append("This may indicate convergent evolution of binding sites.")

    with open(f"{output_dir}/structural_report.md", 'w') as f:
        f.write('\n'.join(lines))

    print(f"Saved structural report to {output_dir}/structural_report.md")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Phase 4B: Structural Analysis')
    parser.add_argument('--results-dir', default='outputs/kinase_all_results',
                        help='Directory with gradient matrices')
    parser.add_argument('--output-dir', default='outputs/phase4_structural',
                        help='Output directory')
    parser.add_argument('--n-clusters', type=int, default=4,
                        help='Number of clusters for analysis')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Phase 4B: Structural Analysis")
    print("=" * 50)

    # Load gradient matrix
    print("\nLoading gradient matrix...")
    G, tasks = load_gradient_matrix(args.results_dir)
    print(f"Loaded {len(tasks)} kinases")

    # Cluster by gradient
    print("\nPerforming hierarchical clustering...")
    linkage, clean_tasks = cluster_by_gradient(G, tasks, str(output_dir))

    # Extract clusters
    print(f"\nExtracting {args.n_clusters} clusters...")
    clusters = get_clusters(linkage, clean_tasks, args.n_clusters)

    for cid, kinases in clusters.items():
        print(f"  Cluster {cid}: {', '.join(kinases)}")

    # Compare to phylogeny
    print("\nComparing to Manning kinase classification...")
    comparison = compare_to_phylogeny(clusters, str(output_dir))

    # Generate clustered heatmap
    print("\nGenerating clustered heatmap...")
    generate_clustered_heatmap(G, tasks, linkage, str(output_dir))

    # Generate report
    print("\nGenerating structural report...")
    generate_structural_report(clusters, comparison, str(output_dir))

    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"\nPhylogeny consistency score: {comparison['consistency_score']:.1%}")


if __name__ == '__main__':
    main()
