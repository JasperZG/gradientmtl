"""
Empirical Correlation Computation for SAR Validation.

Computes pairwise Pearson correlations from actual measured endpoint data.
This provides a data-driven ground truth for validating gradient conflicts,
replacing semi-synthetic literature values with empirical correlations.

Key principle: If gradient conflicts reflect true biological relationships,
they should correlate with empirical correlations in the measured data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings


def compute_empirical_correlations(
    data_path: str,
    endpoints: List[str],
    min_samples: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute pairwise Pearson correlations from measured endpoint data.

    For each pair of endpoints, computes the correlation using only
    molecules that have both measurements (non-missing labels).

    Args:
        data_path: Path to CSV file with endpoint columns
        endpoints: List of endpoint column names
        min_samples: Minimum number of co-measured molecules required
                     for a valid correlation estimate

    Returns:
        Tuple of (correlation_matrix, pvalue_matrix)
        - correlation_matrix: K×K matrix of Pearson correlations
        - pvalue_matrix: K×K matrix of p-values (NaN where insufficient data)
    """
    df = pd.read_csv(data_path)

    K = len(endpoints)
    corr_matrix = np.eye(K)  # Diagonal is 1 (perfect self-correlation)
    pval_matrix = np.zeros((K, K))

    # Map endpoint names to actual column names in dataframe
    endpoint_cols = []
    for ep in endpoints:
        # Try exact match first
        if ep in df.columns:
            endpoint_cols.append(ep)
        else:
            # Try partial match (e.g., "NR-AR" for Tox21)
            matches = [c for c in df.columns if ep in c or c in ep]
            if matches:
                endpoint_cols.append(matches[0])
            else:
                print(f"Warning: Endpoint '{ep}' not found in data")
                endpoint_cols.append(None)

    for i in range(K):
        for j in range(i + 1, K):
            col_i = endpoint_cols[i]
            col_j = endpoint_cols[j]

            if col_i is None or col_j is None:
                corr_matrix[i, j] = np.nan
                corr_matrix[j, i] = np.nan
                pval_matrix[i, j] = np.nan
                pval_matrix[j, i] = np.nan
                continue

            # Get values for both endpoints
            vals_i = df[col_i].values
            vals_j = df[col_j].values

            # Find molecules with both measurements (non-NaN)
            valid_mask = ~(pd.isna(vals_i) | pd.isna(vals_j))

            # Convert to numeric (handle string labels)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x = pd.to_numeric(vals_i[valid_mask], errors='coerce')
                y = pd.to_numeric(vals_j[valid_mask], errors='coerce')

            # Remove any remaining NaN from conversion
            valid_numeric = ~(np.isnan(x) | np.isnan(y))
            x = x[valid_numeric]
            y = y[valid_numeric]

            n_samples = len(x)

            if n_samples < min_samples:
                # Insufficient co-measured samples
                corr_matrix[i, j] = np.nan
                corr_matrix[j, i] = np.nan
                pval_matrix[i, j] = np.nan
                pval_matrix[j, i] = np.nan
            elif np.std(x) < 1e-10 or np.std(y) < 1e-10:
                # Constant values - no meaningful correlation
                corr_matrix[i, j] = 0.0
                corr_matrix[j, i] = 0.0
                pval_matrix[i, j] = 1.0
                pval_matrix[j, i] = 1.0
            else:
                r, p = stats.pearsonr(x, y)
                corr_matrix[i, j] = r
                corr_matrix[j, i] = r
                pval_matrix[i, j] = p
                pval_matrix[j, i] = p

    return corr_matrix, pval_matrix


def compute_empirical_matrix_tox21(
    data_path: str = 'outputs/raw_data/tox21.csv'
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Compute empirical correlation matrix specifically for Tox21 endpoints.

    Returns:
        Tuple of (correlation_matrix, pvalue_matrix, endpoint_names)
    """
    # Standard Tox21 endpoints
    endpoints = [
        'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
        'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
        'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
    ]

    corr, pval = compute_empirical_correlations(data_path, endpoints)
    return corr, pval, endpoints


def get_empirical_matrix_for_tasks(
    task_names: List[str],
    data_path: str = 'outputs/raw_data/tox21.csv'
) -> np.ndarray:
    """
    Get empirical correlation matrix matching the order of task_names.

    This function maps the gradient matrix task names to the empirical
    correlation matrix, handling naming variations.

    Args:
        task_names: Task names from gradient matrix (e.g., 'Tox21_NR-AR' or 'NR-AR')
        data_path: Path to Tox21 CSV file

    Returns:
        K×K empirical correlation matrix aligned with task_names order
    """
    # Compute full empirical matrix
    full_corr, full_pval, full_endpoints = compute_empirical_matrix_tox21(data_path)

    K = len(task_names)
    empirical_matrix = np.zeros((K, K))
    np.fill_diagonal(empirical_matrix, 1.0)

    # Create mapping from task names to endpoint indices
    def find_endpoint_idx(task_name: str) -> Optional[int]:
        """Find matching endpoint index for a task name."""
        # Remove common prefixes
        clean_name = task_name.replace('Tox21_', '').replace('tox21_', '')

        for idx, ep in enumerate(full_endpoints):
            if ep == clean_name or clean_name == ep:
                return idx
            if ep in task_name or task_name in ep:
                return idx
        return None

    # Build matrix
    for i, task_i in enumerate(task_names):
        for j, task_j in enumerate(task_names):
            if i >= j:
                continue

            idx_i = find_endpoint_idx(task_i)
            idx_j = find_endpoint_idx(task_j)

            if idx_i is not None and idx_j is not None:
                empirical_matrix[i, j] = full_corr[idx_i, idx_j]
                empirical_matrix[j, i] = full_corr[idx_i, idx_j]

    return empirical_matrix


def print_correlation_summary(
    corr_matrix: np.ndarray,
    endpoints: List[str],
    threshold: float = 0.1
):
    """Print summary of significant correlations."""
    print("\n" + "=" * 60)
    print("EMPIRICAL CORRELATION SUMMARY")
    print("=" * 60)

    K = len(endpoints)

    # Find strong correlations
    strong_pos = []
    strong_neg = []

    for i in range(K):
        for j in range(i + 1, K):
            r = corr_matrix[i, j]
            if np.isnan(r):
                continue
            if r > threshold:
                strong_pos.append((endpoints[i], endpoints[j], r))
            elif r < -threshold:
                strong_neg.append((endpoints[i], endpoints[j], r))

    # Sort by absolute value
    strong_pos.sort(key=lambda x: -x[2])
    strong_neg.sort(key=lambda x: x[2])

    print(f"\nStrong positive correlations (r > {threshold}):")
    for t1, t2, r in strong_pos[:10]:
        print(f"  {t1} <-> {t2}: r = {r:.3f}")

    print(f"\nStrong negative correlations (r < -{threshold}):")
    for t1, t2, r in strong_neg[:10]:
        print(f"  {t1} <-> {t2}: r = {r:.3f}")

    # Print matrix statistics
    off_diag = corr_matrix[~np.eye(K, dtype=bool)]
    valid = off_diag[~np.isnan(off_diag)]

    print(f"\nMatrix statistics:")
    print(f"  Valid pairs: {len(valid)} / {K * (K-1) // 2}")
    print(f"  Mean correlation: {np.mean(valid):.3f}")
    print(f"  Std correlation: {np.std(valid):.3f}")
    print(f"  Range: [{np.min(valid):.3f}, {np.max(valid):.3f}]")


if __name__ == '__main__':
    import sys
    from pathlib import Path

    # Add project root
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    data_path = project_root / 'outputs' / 'raw_data' / 'tox21.csv'

    if not data_path.exists():
        print(f"Error: Tox21 data not found at {data_path}")
        print("Please run training first to download the data.")
        sys.exit(1)

    print("Computing empirical correlations from Tox21 data...")
    corr, pval, endpoints = compute_empirical_matrix_tox21(str(data_path))

    print_correlation_summary(corr, endpoints)

    # Print full matrix
    print("\nFull Correlation Matrix:")
    print(f"Endpoints: {endpoints}")
    np.set_printoptions(precision=3, suppress=True)
    print(corr)
