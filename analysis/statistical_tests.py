"""
Statistical Analysis Module for Gradient Conflict Validation.

Implements:
- Permutation tests for significance testing
- Bootstrap confidence intervals
- Silhouette score analysis for clustering quality
- ROC analysis for transfer learning prediction
"""

import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score, roc_auc_score, roc_curve
from typing import Dict, List, Tuple, Optional
import warnings


def permutation_test(
    G: np.ndarray,
    L: np.ndarray,
    n_permutations: int = 10000,
    statistic: str = 'pearson'
) -> Dict:
    """
    Permutation test for significance of gradient-literature correlation.

    Tests null hypothesis that gradient conflicts are unrelated to
    literature-documented mechanisms.

    Args:
        G: Gradient conflict matrix (K×K)
        L: Literature relationship matrix (K×K)
        n_permutations: Number of random permutations
        statistic: 'pearson' or 'spearman'

    Returns:
        Dict with observed statistic, p-value, and null distribution
    """
    K = G.shape[0]

    # Extract off-diagonal elements
    mask = ~np.eye(K, dtype=bool)
    g_flat = G[mask]
    l_flat = L[mask]

    # Filter to documented pairs
    nonzero = l_flat != 0
    g_doc = g_flat[nonzero]
    l_doc = l_flat[nonzero]

    if len(g_doc) < 3:
        return {'error': 'Insufficient documented pairs for permutation test'}

    # Compute observed statistic
    if statistic == 'pearson':
        obs_stat, _ = stats.pearsonr(g_doc, l_doc)
    else:
        obs_stat, _ = stats.spearmanr(g_doc, l_doc)

    # Permutation test
    null_distribution = []
    for _ in range(n_permutations):
        # Permute task order
        perm = np.random.permutation(K)
        G_perm = G[perm][:, perm]
        g_perm_flat = G_perm[mask]
        g_perm_doc = g_perm_flat[nonzero]

        if statistic == 'pearson':
            perm_stat, _ = stats.pearsonr(g_perm_doc, l_doc)
        else:
            perm_stat, _ = stats.spearmanr(g_perm_doc, l_doc)

        null_distribution.append(perm_stat)

    null_distribution = np.array(null_distribution)

    # Compute p-value (two-tailed)
    p_value = np.mean(np.abs(null_distribution) >= np.abs(obs_stat))

    return {
        'observed_statistic': obs_stat,
        'p_value': p_value,
        'null_mean': np.mean(null_distribution),
        'null_std': np.std(null_distribution),
        'null_distribution': null_distribution,
        'n_permutations': n_permutations,
        'statistic_type': statistic,
    }


def bootstrap_confidence_interval(
    G: np.ndarray,
    L: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    statistic: str = 'pearson'
) -> Dict:
    """
    Bootstrap confidence interval for gradient-literature correlation.

    Args:
        G: Gradient conflict matrix
        L: Literature relationship matrix
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        statistic: 'pearson' or 'spearman'

    Returns:
        Dict with point estimate, CI bounds, and bootstrap distribution
    """
    K = G.shape[0]

    # Extract off-diagonal documented pairs
    mask = ~np.eye(K, dtype=bool)
    g_flat = G[mask]
    l_flat = L[mask]
    nonzero = l_flat != 0
    g_doc = g_flat[nonzero]
    l_doc = l_flat[nonzero]

    n = len(g_doc)
    if n < 3:
        return {'error': 'Insufficient pairs for bootstrap'}

    # Point estimate
    if statistic == 'pearson':
        point_est, _ = stats.pearsonr(g_doc, l_doc)
    else:
        point_est, _ = stats.spearmanr(g_doc, l_doc)

    # Bootstrap
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        g_boot = g_doc[idx]
        l_boot = l_doc[idx]

        try:
            if statistic == 'pearson':
                stat, _ = stats.pearsonr(g_boot, l_boot)
            else:
                stat, _ = stats.spearmanr(g_boot, l_boot)
            bootstrap_stats.append(stat)
        except:
            continue

    bootstrap_stats = np.array(bootstrap_stats)

    # Confidence interval
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return {
        'point_estimate': point_est,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'confidence': confidence,
        'bootstrap_std': np.std(bootstrap_stats),
        'bootstrap_distribution': bootstrap_stats,
        'n_bootstrap': len(bootstrap_stats),
    }


def compute_silhouette_score(
    G: np.ndarray,
    task_names: List[str],
    n_clusters: int = None,
    method: str = 'average'
) -> Dict:
    """
    Compute silhouette score for task clustering based on gradient conflicts.

    Args:
        G: Gradient conflict matrix (K×K)
        task_names: List of task names
        n_clusters: Number of clusters (if None, optimizes)
        method: Linkage method for hierarchical clustering

    Returns:
        Dict with silhouette score, cluster assignments, and optimal n_clusters
    """
    K = len(task_names)

    # Convert similarity to distance
    D = 1 - G
    np.fill_diagonal(D, 0)

    # Ensure symmetry and non-negative
    D = (D + D.T) / 2
    D = np.maximum(D, 0)

    # Condensed distance matrix for linkage
    from scipy.spatial.distance import squareform
    condensed = squareform(D)

    # Hierarchical clustering
    Z = linkage(condensed, method=method)

    if n_clusters is None:
        # Find optimal number of clusters
        best_score = -1
        best_n = 2

        for n in range(2, min(K-1, 8)):
            labels = fcluster(Z, n, criterion='maxclust')
            try:
                score = silhouette_score(D, labels, metric='precomputed')
                if score > best_score:
                    best_score = score
                    best_n = n
            except:
                continue

        n_clusters = best_n

    # Final clustering
    labels = fcluster(Z, n_clusters, criterion='maxclust')
    score = silhouette_score(D, labels, metric='precomputed')

    # Get cluster assignments
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(task_names[i])

    return {
        'silhouette_score': score,
        'n_clusters': n_clusters,
        'cluster_labels': labels,
        'clusters': clusters,
        'linkage_matrix': Z,
    }


def transfer_learning_roc_analysis(
    G: np.ndarray,
    transfer_results: Dict[Tuple[str, str], float],
    task_names: List[str],
    threshold: float = 0.05
) -> Dict:
    """
    ROC analysis for predicting beneficial transfer learning from gradient conflicts.

    Args:
        G: Gradient conflict matrix
        transfer_results: Dict mapping (source, target) -> AUROC improvement
        task_names: List of task names
        threshold: Minimum improvement to count as beneficial transfer

    Returns:
        Dict with ROC-AUC, optimal threshold, and predictions
    """
    # Extract pairs with transfer results
    g_values = []
    transfer_benefits = []
    pairs = []

    for (source, target), benefit in transfer_results.items():
        if source in task_names and target in task_names:
            i = task_names.index(source)
            j = task_names.index(target)
            g_values.append(G[i, j])
            transfer_benefits.append(benefit)
            pairs.append((source, target))

    g_values = np.array(g_values)
    transfer_benefits = np.array(transfer_benefits)

    if len(g_values) < 5:
        return {'error': 'Insufficient transfer learning results'}

    # Binary classification: beneficial vs non-beneficial transfer
    labels = (transfer_benefits > threshold).astype(int)

    if len(np.unique(labels)) < 2:
        return {'error': 'All transfers are in same class'}

    # ROC analysis
    try:
        auc = roc_auc_score(labels, g_values)
        fpr, tpr, thresholds = roc_curve(labels, g_values)
    except Exception as e:
        return {'error': str(e)}

    # Find optimal threshold (Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]

    # Predictions at optimal threshold
    predictions = (g_values >= optimal_threshold).astype(int)
    accuracy = np.mean(predictions == labels)
    precision = np.sum((predictions == 1) & (labels == 1)) / max(np.sum(predictions == 1), 1)
    recall = np.sum((predictions == 1) & (labels == 1)) / max(np.sum(labels == 1), 1)

    return {
        'roc_auc': auc,
        'optimal_g_threshold': optimal_threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'n_pairs': len(g_values),
        'n_beneficial': np.sum(labels),
    }


def compute_cluster_recovery_score(
    predicted_clusters: Dict[int, List[str]],
    true_clusters: Dict[str, List[str]]
) -> Dict:
    """
    Compute cluster recovery score comparing predicted to known clusters.

    Args:
        predicted_clusters: Dict mapping cluster_id -> list of tasks
        true_clusters: Dict mapping cluster_name -> list of tasks

    Returns:
        Dict with various cluster quality metrics
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    # Get all tasks
    all_tasks = set()
    for tasks in predicted_clusters.values():
        all_tasks.update(tasks)
    for tasks in true_clusters.values():
        all_tasks.update(tasks)
    all_tasks = sorted(all_tasks)

    # Create label arrays
    pred_labels = []
    true_labels = []

    # Reverse mapping
    task_to_pred = {}
    for cluster_id, tasks in predicted_clusters.items():
        for task in tasks:
            task_to_pred[task] = cluster_id

    task_to_true = {}
    for cluster_name, tasks in true_clusters.items():
        for task in tasks:
            task_to_true[task] = cluster_name

    # Build label arrays for overlapping tasks
    common_tasks = []
    for task in all_tasks:
        if task in task_to_pred and task in task_to_true:
            common_tasks.append(task)
            pred_labels.append(task_to_pred[task])
            true_labels.append(task_to_true[task])

    if len(common_tasks) < 3:
        return {'error': 'Insufficient overlap between predicted and true clusters'}

    # Convert to numeric labels for sklearn
    pred_label_map = {l: i for i, l in enumerate(set(pred_labels))}
    true_label_map = {l: i for i, l in enumerate(set(true_labels))}

    pred_numeric = [pred_label_map[l] for l in pred_labels]
    true_numeric = [true_label_map[l] for l in true_labels]

    # Compute metrics
    ari = adjusted_rand_score(true_numeric, pred_numeric)
    nmi = normalized_mutual_info_score(true_numeric, pred_numeric)

    return {
        'adjusted_rand_index': ari,
        'normalized_mutual_info': nmi,
        'n_common_tasks': len(common_tasks),
        'n_predicted_clusters': len(predicted_clusters),
        'n_true_clusters': len(true_clusters),
    }


def comprehensive_statistical_report(
    G: np.ndarray,
    L: np.ndarray,
    task_names: List[str],
    transfer_results: Dict = None,
    true_clusters: Dict = None,
    verbose: bool = True
) -> Dict:
    """
    Generate comprehensive statistical report for gradient conflict matrix.

    Args:
        G: Gradient conflict matrix
        L: Literature relationship matrix
        task_names: List of task names
        transfer_results: Optional transfer learning results
        true_clusters: Optional known mechanistic clusters
        verbose: Print detailed report

    Returns:
        Dict with all statistical analyses
    """
    results = {}

    # 1. Permutation test
    if verbose:
        print("\n" + "=" * 60)
        print("STATISTICAL ANALYSIS REPORT")
        print("=" * 60)

    perm_result = permutation_test(G, L, n_permutations=10000)
    results['permutation_test'] = perm_result

    if verbose and 'error' not in perm_result:
        print(f"\n1. Permutation Test (H0: no correlation)")
        print(f"   Observed r = {perm_result['observed_statistic']:.4f}")
        print(f"   p-value = {perm_result['p_value']:.4f}")
        sig = "***" if perm_result['p_value'] < 0.001 else "**" if perm_result['p_value'] < 0.01 else "*" if perm_result['p_value'] < 0.05 else "n.s."
        print(f"   Significance: {sig}")

    # 2. Bootstrap CI
    boot_result = bootstrap_confidence_interval(G, L, n_bootstrap=10000)
    results['bootstrap_ci'] = boot_result

    if verbose and 'error' not in boot_result:
        print(f"\n2. Bootstrap 95% Confidence Interval")
        print(f"   Point estimate: r = {boot_result['point_estimate']:.4f}")
        print(f"   95% CI: [{boot_result['ci_lower']:.4f}, {boot_result['ci_upper']:.4f}]")

    # 3. Silhouette score
    sil_result = compute_silhouette_score(G, task_names)
    results['silhouette'] = sil_result

    if verbose and 'error' not in sil_result:
        print(f"\n3. Clustering Quality")
        print(f"   Silhouette score: {sil_result['silhouette_score']:.4f}")
        print(f"   Optimal clusters: {sil_result['n_clusters']}")
        print(f"   Clusters:")
        for cid, tasks in sil_result['clusters'].items():
            print(f"     Cluster {cid}: {', '.join(tasks[:3])}{'...' if len(tasks) > 3 else ''}")

    # 4. Transfer learning ROC (if available)
    if transfer_results is not None:
        roc_result = transfer_learning_roc_analysis(G, transfer_results, task_names)
        results['transfer_roc'] = roc_result

        if verbose and 'error' not in roc_result:
            print(f"\n4. Transfer Learning Prediction")
            print(f"   ROC-AUC: {roc_result['roc_auc']:.4f}")
            print(f"   Optimal G threshold: {roc_result['optimal_g_threshold']:.4f}")
            print(f"   Accuracy: {roc_result['accuracy']:.1%}")

    # 5. Cluster recovery (if true clusters available)
    if true_clusters is not None:
        recovery = compute_cluster_recovery_score(sil_result['clusters'], true_clusters)
        results['cluster_recovery'] = recovery

        if verbose and 'error' not in recovery:
            print(f"\n5. Cluster Recovery")
            print(f"   Adjusted Rand Index: {recovery['adjusted_rand_index']:.4f}")
            print(f"   Normalized MI: {recovery['normalized_mutual_info']:.4f}")

    if verbose:
        print("\n" + "=" * 60)

    return results


if __name__ == '__main__':
    # Test with synthetic data
    np.random.seed(42)
    K = 10
    task_names = [f'Task_{i}' for i in range(K)]

    # Synthetic gradient matrix with cluster structure
    G = np.random.randn(K, K) * 0.2
    G = (G + G.T) / 2
    # Add cluster structure
    G[:4, :4] += 0.5  # Cluster 1
    G[4:7, 4:7] += 0.5  # Cluster 2
    G[7:, 7:] += 0.5  # Cluster 3
    np.fill_diagonal(G, 1.0)
    G = np.clip(G, -1, 1)

    # Synthetic literature matrix
    L = np.zeros((K, K))
    L[:4, :4] = 0.6  # Cluster 1
    L[4:7, 4:7] = 0.5  # Cluster 2
    L[7:, 7:] = 0.4  # Cluster 3
    # Add some cross-cluster relationships
    L[0, 5] = L[5, 0] = -0.3
    np.fill_diagonal(L, 1.0)

    # True clusters for validation
    true_clusters = {
        'Cluster_A': task_names[:4],
        'Cluster_B': task_names[4:7],
        'Cluster_C': task_names[7:],
    }

    # Run comprehensive analysis
    results = comprehensive_statistical_report(
        G, L, task_names,
        true_clusters=true_clusters,
        verbose=True
    )
