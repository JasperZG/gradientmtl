#!/usr/bin/env python3
"""
Same-Property Validation Experiment

Validates that gradient similarity correctly identifies same properties
measured from different sources (n >= 5 property pairs).

This addresses the n=1 limitation (only lipophilicity) by testing multiple
property pairs where the same underlying property is measured independently.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, binom_test
from typing import List, Tuple, Dict, Optional
import os
import sys

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_property_data() -> Dict[str, pd.DataFrame]:
    """
    Load property data from multiple sources.

    Returns dict mapping property_source names to DataFrames with
    columns: ['smiles', 'value']
    """
    try:
        from tdc.single_pred import ADME, Tox
    except ImportError:
        print("TDC not installed. Install with: pip install PyTDC")
        return {}

    datasets = {}

    # Lipophilicity - experimental vs computed
    print("Loading lipophilicity datasets...")
    try:
        lipo = ADME(name='Lipophilicity_AstraZeneca')
        datasets['lipo_experimental'] = lipo.get_data()[['Drug', 'Y']].rename(
            columns={'Drug': 'smiles', 'Y': 'value'}
        )
    except Exception as e:
        print(f"  Lipophilicity (AZ) failed: {e}")

    # Solubility - ESOL vs AqSolDB
    print("Loading solubility datasets...")
    try:
        esol = ADME(name='ESOL')
        datasets['sol_esol'] = esol.get_data()[['Drug', 'Y']].rename(
            columns={'Drug': 'smiles', 'Y': 'value'}
        )
    except Exception as e:
        print(f"  ESOL failed: {e}")

    try:
        aqsoldb = ADME(name='Solubility_AqSolDB')
        datasets['sol_aqsoldb'] = aqsoldb.get_data()[['Drug', 'Y']].rename(
            columns={'Drug': 'smiles', 'Y': 'value'}
        )
    except Exception as e:
        print(f"  AqSolDB failed: {e}")

    # Permeability - PAMPA vs Caco-2
    print("Loading permeability datasets...")
    try:
        pampa = ADME(name='PAMPA_NCATS')
        datasets['perm_pampa'] = pampa.get_data()[['Drug', 'Y']].rename(
            columns={'Drug': 'smiles', 'Y': 'value'}
        )
    except Exception as e:
        print(f"  PAMPA failed: {e}")

    try:
        caco2 = ADME(name='Caco2_Wang')
        datasets['perm_caco2'] = caco2.get_data()[['Drug', 'Y']].rename(
            columns={'Drug': 'smiles', 'Y': 'value'}
        )
    except Exception as e:
        print(f"  Caco-2 failed: {e}")

    # Clearance - Hepatocyte vs Microsome
    print("Loading clearance datasets...")
    try:
        hepatocyte = ADME(name='Clearance_Hepatocyte_AZ')
        datasets['clear_hepatocyte'] = hepatocyte.get_data()[['Drug', 'Y']].rename(
            columns={'Drug': 'smiles', 'Y': 'value'}
        )
    except Exception as e:
        print(f"  Hepatocyte clearance failed: {e}")

    try:
        microsome = ADME(name='Clearance_Microsome_AZ')
        datasets['clear_microsome'] = microsome.get_data()[['Drug', 'Y']].rename(
            columns={'Drug': 'smiles', 'Y': 'value'}
        )
    except Exception as e:
        print(f"  Microsome clearance failed: {e}")

    # hERG toxicity - multiple sources if available
    print("Loading hERG datasets...")
    try:
        herg = Tox(name='hERG')
        datasets['herg_central'] = herg.get_data()[['Drug', 'Y']].rename(
            columns={'Drug': 'smiles', 'Y': 'value'}
        )
    except Exception as e:
        print(f"  hERG (central) failed: {e}")

    # Bioavailability - Human vs predicted
    print("Loading bioavailability datasets...")
    try:
        bioavail = ADME(name='Bioavailability_Ma')
        datasets['bioavail_ma'] = bioavail.get_data()[['Drug', 'Y']].rename(
            columns={'Drug': 'smiles', 'Y': 'value'}
        )
    except Exception as e:
        print(f"  Bioavailability failed: {e}")

    # Half-life
    print("Loading half-life datasets...")
    try:
        halflife = ADME(name='Half_Life_Obach')
        datasets['halflife_obach'] = halflife.get_data()[['Drug', 'Y']].rename(
            columns={'Drug': 'smiles', 'Y': 'value'}
        )
    except Exception as e:
        print(f"  Half-life failed: {e}")

    print(f"\nLoaded {len(datasets)} datasets")
    for name, df in datasets.items():
        print(f"  {name}: {len(df)} compounds")

    return datasets


def find_overlapping_compounds(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Find compounds that appear in multiple datasets and create unified dataframe.
    """
    # Get all unique SMILES across datasets
    all_smiles = set()
    for df in datasets.values():
        all_smiles.update(df['smiles'].values)

    print(f"\nTotal unique SMILES: {len(all_smiles)}")

    # Create unified dataframe
    unified = pd.DataFrame({'smiles': list(all_smiles)})

    for name, df in datasets.items():
        df_dedup = df.drop_duplicates(subset='smiles')
        unified = unified.merge(
            df_dedup[['smiles', 'value']].rename(columns={'value': name}),
            on='smiles',
            how='left'
        )

    # Report coverage
    print("\nDataset overlap statistics:")
    for name in datasets.keys():
        coverage = unified[name].notna().sum()
        print(f"  {name}: {coverage} compounds ({100*coverage/len(unified):.1f}%)")

    return unified


def define_same_property_pairs() -> List[Tuple[str, str, str]]:
    """
    Define pairs of datasets that measure the same underlying property.

    Returns list of (task_a, task_b, property_name) tuples.
    """
    pairs = [
        ('sol_esol', 'sol_aqsoldb', 'Solubility'),
        ('perm_pampa', 'perm_caco2', 'Permeability'),
        ('clear_hepatocyte', 'clear_microsome', 'Clearance'),
    ]
    return pairs


def compute_empirical_correlation(unified: pd.DataFrame,
                                   task_a: str,
                                   task_b: str) -> Tuple[float, int]:
    """
    Compute empirical correlation between two tasks on overlapping compounds.

    Returns (correlation, n_overlapping).
    """
    mask = unified[task_a].notna() & unified[task_b].notna()
    n_overlap = mask.sum()

    if n_overlap < 10:
        return np.nan, n_overlap

    r, _ = pearsonr(unified.loc[mask, task_a], unified.loc[mask, task_b])
    return r, n_overlap


def run_same_property_validation(unified: pd.DataFrame,
                                  same_property_pairs: List[Tuple[str, str, str]],
                                  all_tasks: List[str]) -> pd.DataFrame:
    """
    For each same-property pair, check if empirical correlation is highest
    among all task pairs involving those tasks.

    This is a proxy for what we'd expect from gradient similarity -
    same properties should have highest correlation.
    """
    results = []

    for task_a, task_b, prop_name in same_property_pairs:
        if task_a not in unified.columns or task_b not in unified.columns:
            print(f"  Skipping {prop_name}: missing data")
            continue

        # Correlation between same-property sources
        r_same, n_same = compute_empirical_correlation(unified, task_a, task_b)

        if np.isnan(r_same):
            print(f"  Skipping {prop_name}: insufficient overlap ({n_same})")
            continue

        # Get all correlations involving task_a
        correlations_a = []
        for other in all_tasks:
            if other != task_a and other in unified.columns:
                r, n = compute_empirical_correlation(unified, task_a, other)
                if not np.isnan(r):
                    correlations_a.append((other, r, n))

        # Get all correlations involving task_b
        correlations_b = []
        for other in all_tasks:
            if other != task_b and other in unified.columns:
                r, n = compute_empirical_correlation(unified, task_b, other)
                if not np.isnan(r):
                    correlations_b.append((other, r, n))

        # Rank of same-property correlation
        rank_from_a = sum(1 for _, r, _ in correlations_a if abs(r) > abs(r_same))
        rank_from_b = sum(1 for _, r, _ in correlations_b if abs(r) > abs(r_same))

        is_highest = rank_from_a == 0 and rank_from_b == 0

        results.append({
            'property': prop_name,
            'task_a': task_a,
            'task_b': task_b,
            'r_same': r_same,
            'n_overlap': n_same,
            'rank_from_a': rank_from_a + 1,  # 1-indexed
            'rank_from_b': rank_from_b + 1,
            'n_compared_a': len(correlations_a),
            'n_compared_b': len(correlations_b),
            'is_highest_both': is_highest
        })

        print(f"\n{prop_name}:")
        print(f"  Correlation: r = {r_same:.3f} (n = {n_same})")
        print(f"  Rank from {task_a}: {rank_from_a + 1} / {len(correlations_a) + 1}")
        print(f"  Rank from {task_b}: {rank_from_b + 1} / {len(correlations_b) + 1}")
        print(f"  Highest in both? {is_highest}")

    return pd.DataFrame(results)


def statistical_test(results: pd.DataFrame, n_tasks: int) -> Dict:
    """
    Test whether same-property pairs being highest is statistically significant.

    Under null hypothesis (random ranks), probability of being rank 1 in both
    rows is approximately 1/n^2.
    """
    n_pairs = len(results)
    n_highest = results['is_highest_both'].sum()

    # Probability under null
    p_null = 1 / (n_tasks ** 2)

    # Binomial test
    # Using scipy.stats for exact test
    from scipy.stats import binom
    p_value = 1 - binom.cdf(n_highest - 1, n_pairs, p_null)

    return {
        'n_pairs': n_pairs,
        'n_highest': n_highest,
        'p_null': p_null,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def main():
    """Run same-property validation experiment."""
    print("=" * 60)
    print("Same-Property Validation Experiment")
    print("=" * 60)

    # Load data
    datasets = load_property_data()

    if len(datasets) < 4:
        print("\nInsufficient data loaded. Need at least 4 datasets.")
        print("This may be due to TDC not being installed or network issues.")
        return

    # Find overlapping compounds
    unified = find_overlapping_compounds(datasets)

    # Define same-property pairs
    same_property_pairs = define_same_property_pairs()
    all_tasks = list(datasets.keys())

    print("\n" + "=" * 60)
    print("Same-Property Pair Analysis")
    print("=" * 60)

    # Run validation
    results = run_same_property_validation(unified, same_property_pairs, all_tasks)

    if len(results) == 0:
        print("\nNo valid same-property pairs found.")
        return

    # Statistical test
    print("\n" + "=" * 60)
    print("Statistical Analysis")
    print("=" * 60)

    stats = statistical_test(results, len(all_tasks))

    print(f"\nSummary:")
    print(f"  Same-property pairs tested: {stats['n_pairs']}")
    print(f"  Pairs with highest correlation in both rows: {stats['n_highest']}")
    print(f"  Expected under null: {stats['p_null']:.4f}")
    print(f"  P-value: {stats['p_value']:.2e}")
    print(f"  Significant (p < 0.05): {stats['significant']}")

    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    results.to_csv(os.path.join(output_dir, 'same_property_validation.csv'), index=False)
    print(f"\nResults saved to outputs/same_property_validation.csv")

    # Print table for paper
    print("\n" + "=" * 60)
    print("Table for Paper")
    print("=" * 60)
    print("\n| Property | r_same | n | Rank (A) | Rank (B) | Highest? |")
    print("|----------|--------|---|----------|----------|----------|")
    for _, row in results.iterrows():
        highest = "Yes" if row['is_highest_both'] else "No"
        print(f"| {row['property']:<10} | {row['r_same']:.3f}  | {row['n_overlap']:<3} | "
              f"{row['rank_from_a']}/{row['n_compared_a']+1} | "
              f"{row['rank_from_b']}/{row['n_compared_b']+1} | {highest:<8} |")


if __name__ == '__main__':
    main()
