"""Visualization tools for gradient conflict analysis."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


def plot_conflict_heatmap(
    conflict_matrix: np.ndarray,
    task_names: list[str],
    output_path: Path | str,
    title: str = "Gradient Conflict Matrix (Cosine Similarity)",
    cluster: bool = True,
    figsize: tuple[int, int] = (10, 8),
    annotate: bool = True,
):
    """
    Plot heatmap of gradient conflict matrix with optional hierarchical clustering.

    Args:
        conflict_matrix: (N, N) matrix of cosine similarities
        task_names: Names for axis labels
        output_path: Where to save figure
        title: Plot title
        cluster: Whether to reorder by hierarchical clustering
        figsize: Figure size
        annotate: Whether to show values in cells
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Make a copy for reordering
    matrix = conflict_matrix.copy()
    names = list(task_names)

    if cluster and len(matrix) > 2:
        # Convert similarity to distance for clustering
        # distance = (1 - similarity) / 2 maps [-1, 1] to [0, 1]
        distance_matrix = (1 - matrix) / 2

        # Ensure diagonal is exactly 0
        np.fill_diagonal(distance_matrix, 0)

        # Make symmetric (average with transpose for numerical stability)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2

        # Perform hierarchical clustering
        condensed = squareform(distance_matrix)
        linkage_matrix = linkage(condensed, method='average')

        # Get reordering
        order = leaves_list(linkage_matrix)

        # Reorder matrix and labels
        matrix = matrix[order][:, order]
        names = [names[i] for i in order]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    sns.heatmap(
        matrix,
        xticklabels=names,
        yticklabels=names,
        annot=annotate,
        fmt='.2f',
        cmap='RdBu_r',  # Red=conflict (negative), Blue=synergy (positive)
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
        cbar_kws={'label': 'Cosine Similarity'},
    )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('Task', fontsize=12)

    # Rotate x labels for readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved heatmap to {output_path}")


def plot_conflict_evolution(
    conflict_history: np.ndarray,
    task_names: list[str],
    task_pairs: list[tuple[str, str]] | None = None,
    output_path: Path | str | None = None,
    logged_steps: np.ndarray | None = None,
    title: str = "Gradient Conflict Evolution",
    figsize: tuple[int, int] = (12, 6),
):
    """
    Plot how gradient conflicts evolve over training.

    Args:
        conflict_history: (T, N, N) array of conflict matrices over time
        task_names: List of task names
        task_pairs: List of (task_a, task_b) pairs to plot (default: all pairs)
        output_path: Where to save figure (None = display)
        logged_steps: Array of step numbers (for x-axis)
        title: Plot title
        figsize: Figure size
    """
    if len(conflict_history) == 0:
        print("No history to plot")
        return

    # Default: plot all pairs
    if task_pairs is None:
        task_pairs = [
            (task_names[i], task_names[j])
            for i in range(len(task_names))
            for j in range(i + 1, len(task_names))
        ]

    # X-axis
    if logged_steps is not None:
        x = logged_steps
        xlabel = "Training Step"
    else:
        x = np.arange(len(conflict_history))
        xlabel = "Logging Step"

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each pair
    for task_a, task_b in task_pairs:
        i = task_names.index(task_a)
        j = task_names.index(task_b)
        values = conflict_history[:, i, j]

        label = f"{task_a} vs {task_b}"
        ax.plot(x, values, label=label, alpha=0.8)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0.3, color='blue', linestyle=':', alpha=0.3, label='Synergy threshold')
    ax.axhline(y=-0.3, color='red', linestyle=':', alpha=0.3, label='Conflict threshold')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(-1.1, 1.1)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved evolution plot to {output_path}")
    else:
        plt.show()


def print_conflict_summary(
    conflict_matrix: np.ndarray,
    task_names: list[str],
    expected_patterns: dict[str, float] | None = None,
):
    """
    Print summary of gradient conflicts with hypothesis testing.

    Args:
        conflict_matrix: (N, N) average conflict matrix
        task_names: List of task names
        expected_patterns: Optional dict of expected values, e.g.,
            {'esol_lipophilicity': -0.5, 'bbbp_herg': 0.3}
    """
    print("\n" + "=" * 60)
    print("GRADIENT CONFLICT ANALYSIS RESULTS")
    print("=" * 60)

    # Print matrix
    print("\nConflict Matrix (Cosine Similarity):")
    print("-" * 60)

    # Header
    header = "          " + "  ".join(f"{t[:8]:>8}" for t in task_names)
    print(header)

    # Rows
    for i, task_i in enumerate(task_names):
        row = f"{task_i[:8]:>8}  "
        row += "  ".join(f"{conflict_matrix[i,j]:>8.3f}" for j in range(len(task_names)))
        print(row)

    # Identify key relationships
    print("\n" + "-" * 60)
    print("Key Relationships (sorted by |similarity|):")

    pairs = []
    for i, task_i in enumerate(task_names):
        for j, task_j in enumerate(task_names):
            if i < j:
                val = conflict_matrix[i, j]
                key = f"{task_i}_{task_j}"
                pairs.append((task_i, task_j, val, key))

    # Sort by absolute value
    pairs.sort(key=lambda x: -abs(x[2]))

    for task_i, task_j, val, key in pairs:
        # Determine relationship type
        if val > 0.5:
            relation = "STRONGLY SYNERGISTIC"
            symbol = "++"
        elif val > 0.3:
            relation = "Synergistic"
            symbol = "+"
        elif val < -0.5:
            relation = "STRONGLY CONFLICTING"
            symbol = "--"
        elif val < -0.3:
            relation = "Conflicting"
            symbol = "-"
        else:
            relation = "Independent"
            symbol = "~"

        print(f"  [{symbol}] {task_i} vs {task_j}: {val:+.3f} ({relation})")

    # Hypothesis validation
    if expected_patterns:
        print("\n" + "-" * 60)
        print("Hypothesis Validation:")

        for pattern_key, expected in expected_patterns.items():
            # Parse pattern key (e.g., 'esol_lipophilicity')
            parts = pattern_key.lower().split('_')

            # Find matching tasks
            found = False
            for task_i, task_j, val, key in pairs:
                key_lower = key.lower()
                if all(p in key_lower for p in parts):
                    found = True

                    # Check if expectation is met
                    if expected < 0:
                        success = val < expected * 0.5  # Within 50% of expected
                        direction = "negative"
                    else:
                        success = val > expected * 0.5
                        direction = "positive"

                    status = "PASS" if success else "FAIL"
                    symbol = "[+]" if success else "[X]"

                    print(f"  {symbol} {pattern_key}: expected {expected:+.2f} ({direction}), "
                          f"got {val:+.3f} [{status}]")
                    break

            if not found:
                print(f"  ? {pattern_key}: tasks not found in matrix")

    print("=" * 60 + "\n")


def create_full_analysis(
    conflict_data_path: Path | str,
    output_dir: Path | str,
    expected_patterns: dict[str, float] | None = None,
):
    """
    Create full analysis from saved gradient conflict data.

    Args:
        conflict_data_path: Path to conflict_matrices.npz
        output_dir: Directory for output figures
        expected_patterns: Expected conflict patterns for validation
    """
    from .visualization import plot_conflict_heatmap, plot_conflict_evolution, print_conflict_summary
    from ..training.gradient_logger import GradientConflictLogger

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data = GradientConflictLogger.load(conflict_data_path)
    matrix = data['averaged']
    history = data['history']
    task_names = data['task_names']
    logged_steps = data.get('logged_steps')

    # Print summary
    print_conflict_summary(matrix, task_names, expected_patterns)

    # Create heatmap
    plot_conflict_heatmap(
        matrix,
        task_names,
        output_dir / 'gradient_conflict_heatmap.png',
        cluster=True,
    )

    # Create evolution plot
    if len(history) > 0:
        plot_conflict_evolution(
            history,
            task_names,
            output_path=output_dir / 'gradient_conflict_evolution.png',
            logged_steps=logged_steps,
        )

    print(f"\nAnalysis complete. Outputs saved to {output_dir}")
