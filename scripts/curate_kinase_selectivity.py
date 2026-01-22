#!/usr/bin/env python3
"""
ChEMBL Kinase Selectivity Panel Curation.

Creates a compound-aligned dataset for kinase selectivity analysis where
compounds are tested against multiple kinases. This enables discovery of
selectivity trade-offs via gradient conflict analysis.

Key principle: Kinase inhibitors often show selectivity trade-offs - high affinity
for one kinase may come at the cost of affinity for related kinases. These
antagonistic relationships should appear as negative gradient correlations.

Target kinase families:
- CDK family: CDK1, CDK2, CDK4, CDK6, CDK7, CDK9 (cell cycle kinases)
- JAK family: JAK1, JAK2, JAK3, TYK2 (immune signaling)
- EGFR family: EGFR, HER2, HER3, HER4 (receptor tyrosine kinases)

Usage:
    python scripts/curate_kinase_selectivity.py
    python scripts/curate_kinase_selectivity.py --family cdk
    python scripts/curate_kinase_selectivity.py --family jak
    python scripts/curate_kinase_selectivity.py --min-overlap 0.5
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import requests
import pandas as pd
import numpy as np
from collections import defaultdict

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# ChEMBL API Configuration
# =============================================================================

CHEMBL_API_BASE = "https://www.ebi.ac.uk/chembl/api/data"

# Kinase ChEMBL Target IDs
# Source: ChEMBL target search for human kinases
KINASE_TARGETS = {
    # CDK family - Cell Cycle Kinases
    'cdk': {
        'CDK1': 'CHEMBL308',      # Cyclin-dependent kinase 1
        'CDK2': 'CHEMBL301',      # Cyclin-dependent kinase 2
        'CDK4': 'CHEMBL461',      # Cyclin-dependent kinase 4
        'CDK6': 'CHEMBL2508',     # Cyclin-dependent kinase 6
        'CDK7': 'CHEMBL3267',     # Cyclin-dependent kinase 7
        'CDK9': 'CHEMBL3116',     # Cyclin-dependent kinase 9
    },
    # JAK family - Immune Signaling
    'jak': {
        'JAK1': 'CHEMBL2835',     # Janus kinase 1
        'JAK2': 'CHEMBL2971',     # Janus kinase 2
        'JAK3': 'CHEMBL2148',     # Janus kinase 3
        'TYK2': 'CHEMBL3905',     # Tyrosine kinase 2
    },
    # EGFR/HER family - Receptor Tyrosine Kinases
    'egfr': {
        'EGFR': 'CHEMBL203',      # Epidermal growth factor receptor
        'HER2': 'CHEMBL1824',     # Receptor tyrosine-protein kinase erbB-2
        'HER3': 'CHEMBL5838',     # Receptor tyrosine-protein kinase erbB-3
        'HER4': 'CHEMBL3009',     # Receptor tyrosine-protein kinase erbB-4
    },
    # Aurora kinases - Mitotic regulators
    'aurora': {
        'AURKA': 'CHEMBL4722',    # Aurora kinase A
        'AURKB': 'CHEMBL3267',    # Aurora kinase B
        'AURKC': 'CHEMBL5606',    # Aurora kinase C
    },
    # SRC family - Non-receptor tyrosine kinases
    'src': {
        'SRC': 'CHEMBL267',       # Proto-oncogene tyrosine-protein kinase Src
        'ABL1': 'CHEMBL1862',     # Tyrosine-protein kinase ABL1
        'LCK': 'CHEMBL258',       # Tyrosine-protein kinase Lck
        'FYN': 'CHEMBL1841',      # Tyrosine-protein kinase Fyn
        'YES1': 'CHEMBL2073',     # Tyrosine-protein kinase Yes
    },
}

# Known selectivity relationships (for validation)
KNOWN_SELECTIVITY = {
    'cdk': [
        ('CDK4', 'CDK6', 'synergy'),     # Palbociclib targets both
        ('CDK2', 'CDK1', 'synergy'),     # Often co-inhibited
        ('CDK4', 'CDK2', 'selectivity'), # CDK4/6 inhibitors selective over CDK2
        ('CDK7', 'CDK9', 'selectivity'), # Transcriptional CDKs distinct
    ],
    'jak': [
        ('JAK1', 'JAK2', 'synergy'),     # Often co-targeted (ruxolitinib)
        ('JAK1', 'JAK3', 'selectivity'), # Tofacitinib JAK1/3 > JAK2
        ('JAK2', 'TYK2', 'selectivity'), # Different substrate specificity
    ],
    'egfr': [
        ('EGFR', 'HER2', 'selectivity'), # Lapatinib dual, gefitinib EGFR-selective
        ('HER2', 'HER3', 'synergy'),     # Often co-expressed
        ('EGFR', 'HER4', 'selectivity'), # Different tissue distribution
    ],
}


def query_kinase_activities(
    target_id: str,
    target_name: str,
    activity_types: List[str] = ['IC50', 'Ki', 'Kd'],
    limit: int = 10000,
    max_nm: float = 100000,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Query ChEMBL for kinase inhibitor activities.

    Args:
        target_id: ChEMBL target ID
        target_name: Human-readable name for logging
        activity_types: Types of activity to query
        limit: Max results per activity type
        max_nm: Max activity value in nM
        verbose: Print progress

    Returns:
        DataFrame with kinase activities
    """
    all_activities = []

    for activity_type in activity_types:
        url = f"{CHEMBL_API_BASE}/activity.json"

        params = {
            'target_chembl_id': target_id,
            'standard_type': activity_type,
            'standard_units': 'nM',
            'limit': limit,
            'offset': 0,
        }

        type_activities = []

        while True:
            try:
                response = requests.get(url, params=params, timeout=120)
                response.raise_for_status()
                data = response.json()
            except requests.RequestException as e:
                if verbose:
                    print(f"    Error querying {target_name} {activity_type}: {e}")
                break

            activities = data.get('activities', [])
            if not activities:
                break

            for act in activities:
                # Data quality filters
                if act.get('data_validity_comment'):
                    continue
                if act.get('standard_value') is None:
                    continue

                value = float(act['standard_value'])
                if value <= 0 or value > max_nm:
                    continue

                # Prefer activities with pChEMBL values (quality indicator)
                pchembl = act.get('pchembl_value')

                type_activities.append({
                    'chembl_id': act.get('molecule_chembl_id'),
                    'smiles': act.get('canonical_smiles'),
                    'activity_value_nM': value,
                    'activity_type': activity_type,
                    'pchembl_value': float(pchembl) if pchembl else None,
                    'assay_chembl_id': act.get('assay_chembl_id'),
                    'target_chembl_id': target_id,
                    'target_name': target_name,
                })

            # Check for next page
            if data.get('page_meta', {}).get('next'):
                params['offset'] += limit
                time.sleep(0.3)
            else:
                break

        all_activities.extend(type_activities)

        if verbose and type_activities:
            print(f"    {activity_type}: {len(type_activities)} activities")

    if not all_activities:
        return pd.DataFrame()

    df = pd.DataFrame(all_activities)

    # Deduplicate: keep entry with best pChEMBL or median value per compound
    def aggregate_compound(group):
        # Prefer entry with pChEMBL value
        has_pchembl = group[group['pchembl_value'].notna()]
        if len(has_pchembl) > 0:
            # Use median of entries with pChEMBL
            return pd.Series({
                'smiles': has_pchembl['smiles'].iloc[0],
                'activity_value_nM': has_pchembl['activity_value_nM'].median(),
                'pchembl_value': has_pchembl['pchembl_value'].median(),
                'target_chembl_id': group['target_chembl_id'].iloc[0],
                'target_name': group['target_name'].iloc[0],
            })
        else:
            return pd.Series({
                'smiles': group['smiles'].iloc[0],
                'activity_value_nM': group['activity_value_nM'].median(),
                'pchembl_value': None,
                'target_chembl_id': group['target_chembl_id'].iloc[0],
                'target_name': group['target_name'].iloc[0],
            })

    df = df.groupby('chembl_id').apply(aggregate_compound).reset_index()

    return df


def compute_overlap_matrix(kinase_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute pairwise compound overlap between kinases.

    Args:
        kinase_dfs: Dict mapping kinase name to activity DataFrame

    Returns:
        DataFrame with pairwise overlap counts and percentages
    """
    kinases = list(kinase_dfs.keys())
    n = len(kinases)

    # Get compound sets
    compound_sets = {k: set(df['chembl_id'].dropna()) for k, df in kinase_dfs.items()}

    # Compute overlap matrix
    overlap_counts = np.zeros((n, n), dtype=int)
    overlap_pcts = np.zeros((n, n))

    for i, k1 in enumerate(kinases):
        for j, k2 in enumerate(kinases):
            intersection = len(compound_sets[k1] & compound_sets[k2])
            overlap_counts[i, j] = intersection
            min_size = min(len(compound_sets[k1]), len(compound_sets[k2]))
            overlap_pcts[i, j] = intersection / min_size if min_size > 0 else 0

    return pd.DataFrame(overlap_pcts, index=kinases, columns=kinases), overlap_counts


def filter_by_overlap(
    kinase_dfs: Dict[str, pd.DataFrame],
    min_overlap: float = 0.5,
    min_compounds_per_kinase: int = 100,
    verbose: bool = True
) -> Tuple[Dict[str, pd.DataFrame], Set[str]]:
    """
    Filter to compounds tested on multiple kinases with sufficient overlap.

    Args:
        kinase_dfs: Dict mapping kinase name to activity DataFrame
        min_overlap: Minimum pairwise overlap fraction required
        min_compounds_per_kinase: Minimum compounds per kinase after filtering
        verbose: Print progress

    Returns:
        Tuple of (filtered kinase DataFrames, set of compound IDs to keep)
    """
    # Get all compound IDs
    all_compounds = set()
    for df in kinase_dfs.values():
        all_compounds.update(df['chembl_id'].dropna())

    if verbose:
        print(f"\n  Total unique compounds across all kinases: {len(all_compounds)}")

    # Count how many kinases each compound is tested on
    compound_counts = defaultdict(int)
    for df in kinase_dfs.values():
        for cid in df['chembl_id'].dropna():
            compound_counts[cid] += 1

    # Filter to compounds tested on at least 2 kinases
    multi_kinase_compounds = {c for c, count in compound_counts.items() if count >= 2}

    if verbose:
        print(f"  Compounds tested on 2+ kinases: {len(multi_kinase_compounds)}")

    # Further filter based on overlap requirement
    kinases = list(kinase_dfs.keys())
    compound_sets = {k: set(df['chembl_id'].dropna()) for k, df in kinase_dfs.items()}

    # Find kinases with sufficient pairwise overlap
    valid_kinases = []
    for k1 in kinases:
        has_overlap = False
        for k2 in kinases:
            if k1 != k2:
                intersection = len(compound_sets[k1] & compound_sets[k2])
                min_size = min(len(compound_sets[k1]), len(compound_sets[k2]))
                if min_size > 0 and intersection / min_size >= min_overlap:
                    has_overlap = True
                    break
        if has_overlap:
            valid_kinases.append(k1)

    if verbose:
        print(f"  Kinases with {min_overlap*100:.0f}%+ pairwise overlap: {len(valid_kinases)}")
        print(f"    {valid_kinases}")

    # Get compounds in the overlap region
    if len(valid_kinases) >= 2:
        # Find compounds tested on multiple valid kinases
        overlap_compounds = set()
        for cid in multi_kinase_compounds:
            kinases_with_cid = [k for k in valid_kinases if cid in compound_sets[k]]
            if len(kinases_with_cid) >= 2:
                overlap_compounds.add(cid)
    else:
        overlap_compounds = multi_kinase_compounds

    if verbose:
        print(f"  Compounds in high-overlap region: {len(overlap_compounds)}")

    # Filter DataFrames
    filtered_dfs = {}
    for k, df in kinase_dfs.items():
        if k in valid_kinases:
            filtered = df[df['chembl_id'].isin(overlap_compounds)].copy()
            if len(filtered) >= min_compounds_per_kinase:
                filtered_dfs[k] = filtered

    if verbose:
        print(f"\n  Final kinases with {min_compounds_per_kinase}+ compounds: {len(filtered_dfs)}")
        for k, df in filtered_dfs.items():
            print(f"    {k}: {len(df)} compounds")

    return filtered_dfs, overlap_compounds


def build_activity_matrix(
    kinase_dfs: Dict[str, pd.DataFrame],
    compound_ids: Set[str]
) -> pd.DataFrame:
    """
    Build a compound x kinase activity matrix.

    Args:
        kinase_dfs: Dict mapping kinase name to activity DataFrame
        compound_ids: Set of compound IDs to include

    Returns:
        DataFrame with compounds as rows, kinases as columns, pIC50 as values
    """
    # Initialize matrix
    compound_list = sorted(compound_ids)
    kinases = sorted(kinase_dfs.keys())

    # Create base DataFrame with SMILES
    smiles_dict = {}
    for df in kinase_dfs.values():
        for _, row in df.iterrows():
            if row['chembl_id'] in compound_ids and pd.notna(row['smiles']):
                smiles_dict[row['chembl_id']] = row['smiles']

    matrix_data = {
        'chembl_id': compound_list,
        'smiles': [smiles_dict.get(c, '') for c in compound_list],
    }

    # Add activity columns (convert to pIC50)
    for kinase in kinases:
        df = kinase_dfs[kinase]
        activity_dict = dict(zip(df['chembl_id'], df['activity_value_nM']))

        pIC50_values = []
        for cid in compound_list:
            if cid in activity_dict:
                # Convert nM to pIC50: pIC50 = -log10(IC50_M) = -log10(IC50_nM * 1e-9)
                ic50_nm = activity_dict[cid]
                pIC50 = -np.log10(ic50_nm * 1e-9)
                pIC50_values.append(pIC50)
            else:
                pIC50_values.append(np.nan)

        matrix_data[f'{kinase}_pIC50'] = pIC50_values

    return pd.DataFrame(matrix_data)


def curate_kinase_selectivity_dataset(
    family: str = 'cdk',
    min_overlap: float = 0.3,
    min_compounds: int = 500,
    output_dir: Path = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Curate a kinase selectivity panel dataset from ChEMBL.

    Args:
        family: Kinase family to query ('cdk', 'jak', 'egfr', 'all')
        min_overlap: Minimum pairwise compound overlap fraction
        min_compounds: Target minimum number of compounds
        output_dir: Output directory
        verbose: Print progress

    Returns:
        Tuple of (activity matrix DataFrame, metadata dict)
    """
    if output_dir is None:
        output_dir = project_root / 'outputs' / 'kinase_data'
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("ChEMBL Kinase Selectivity Panel Curation")
        print("=" * 60)
        print()

    # Select kinase targets
    if family == 'all':
        targets = {}
        for fam_targets in KINASE_TARGETS.values():
            targets.update(fam_targets)
    else:
        targets = KINASE_TARGETS.get(family, {})

    if not targets:
        raise ValueError(f"Unknown kinase family: {family}")

    if verbose:
        print(f"Kinase family: {family}")
        print(f"Targets: {list(targets.keys())}")
        print()

    # Step 1: Query activities for each kinase
    if verbose:
        print("Step 1: Querying ChEMBL for kinase activities...")

    kinase_dfs = {}
    for kinase_name, target_id in targets.items():
        if verbose:
            print(f"\n  {kinase_name} ({target_id}):")

        df = query_kinase_activities(target_id, kinase_name, verbose=verbose)

        if len(df) > 0:
            kinase_dfs[kinase_name] = df
            if verbose:
                print(f"    Total unique compounds: {len(df)}")
        else:
            if verbose:
                print(f"    No activities found!")

    if len(kinase_dfs) < 2:
        raise ValueError(f"Need at least 2 kinases with data, found {len(kinase_dfs)}")

    # Step 2: Analyze overlap
    if verbose:
        print("\n" + "=" * 60)
        print("Step 2: Analyzing compound overlap...")

    overlap_pct, overlap_counts = compute_overlap_matrix(kinase_dfs)

    if verbose:
        print("\n  Pairwise overlap matrix (% of smaller set):")
        print(overlap_pct.round(2).to_string())

    # Step 3: Filter by overlap
    if verbose:
        print("\n" + "=" * 60)
        print(f"Step 3: Filtering for {min_overlap*100:.0f}%+ overlap...")

    filtered_dfs, overlap_compounds = filter_by_overlap(
        kinase_dfs,
        min_overlap=min_overlap,
        min_compounds_per_kinase=50,
        verbose=verbose
    )

    if len(filtered_dfs) < 2:
        if verbose:
            print("\n  Warning: Insufficient overlap. Relaxing constraints...")
        filtered_dfs, overlap_compounds = filter_by_overlap(
            kinase_dfs,
            min_overlap=0.1,  # Relax to 10%
            min_compounds_per_kinase=30,
            verbose=verbose
        )

    # Step 4: Build activity matrix
    if verbose:
        print("\n" + "=" * 60)
        print("Step 4: Building activity matrix...")

    activity_matrix = build_activity_matrix(filtered_dfs, overlap_compounds)

    if verbose:
        print(f"  Matrix shape: {activity_matrix.shape}")
        print(f"  Compounds: {len(activity_matrix)}")
        print(f"  Kinases: {len(filtered_dfs)}")

    # Compute coverage statistics
    kinase_cols = [c for c in activity_matrix.columns if c.endswith('_pIC50')]
    coverage = activity_matrix[kinase_cols].notna().mean()

    if verbose:
        print("\n  Per-kinase coverage:")
        for col in kinase_cols:
            print(f"    {col}: {coverage[col]*100:.1f}%")

    # Step 5: Compute empirical correlations (for validation)
    if verbose:
        print("\n" + "=" * 60)
        print("Step 5: Computing empirical activity correlations...")

    empirical_corr = activity_matrix[kinase_cols].corr()

    if verbose:
        print("\n  Empirical pIC50 correlation matrix:")
        print(empirical_corr.round(3).to_string())

    # Check for negative correlations (selectivity trade-offs)
    n_negative = (empirical_corr.values < 0).sum()
    n_positive = (empirical_corr.values > 0).sum()
    n_pairs = len(kinase_cols) * (len(kinase_cols) - 1)

    if verbose:
        print(f"\n  Positive correlations: {n_positive - len(kinase_cols)} / {n_pairs} pairs")
        print(f"  Negative correlations: {n_negative} / {n_pairs} pairs")

        if n_negative > 0:
            print("\n  Negative correlation pairs (selectivity trade-offs):")
            for i, k1 in enumerate(kinase_cols):
                for j, k2 in enumerate(kinase_cols):
                    if i < j and empirical_corr.loc[k1, k2] < 0:
                        print(f"    {k1} vs {k2}: r = {empirical_corr.loc[k1, k2]:.3f}")

    # Step 6: Save outputs
    if verbose:
        print("\n" + "=" * 60)
        print("Step 6: Saving outputs...")

    # Save activity matrix
    output_file = output_dir / f'kinase_{family}_activity_matrix.csv'
    activity_matrix.to_csv(output_file, index=False)

    # Save correlation matrix
    corr_file = output_dir / f'kinase_{family}_empirical_correlations.csv'
    empirical_corr.to_csv(corr_file)

    # Save metadata
    metadata = {
        'family': family,
        'kinases': list(filtered_dfs.keys()),
        'n_compounds': len(activity_matrix),
        'n_kinases': len(filtered_dfs),
        'min_overlap': min_overlap,
        'coverage': {col: float(coverage[col]) for col in kinase_cols},
        'empirical_correlation_stats': {
            'mean': float(empirical_corr.values[np.triu_indices(len(kinase_cols), 1)].mean()),
            'min': float(empirical_corr.values[np.triu_indices(len(kinase_cols), 1)].min()),
            'max': float(empirical_corr.values[np.triu_indices(len(kinase_cols), 1)].max()),
            'n_negative': int(n_negative),
            'n_positive': int(n_positive - len(kinase_cols)),
        },
        'target_ids': {k: targets[k] for k in filtered_dfs.keys()},
        'curation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    with open(output_dir / f'kinase_{family}_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"\n  Saved to: {output_dir}")
        print(f"    - kinase_{family}_activity_matrix.csv")
        print(f"    - kinase_{family}_empirical_correlations.csv")
        print(f"    - kinase_{family}_metadata.json")

    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print("CURATION SUMMARY")
        print("=" * 60)
        print(f"\nFamily: {family}")
        print(f"Kinases: {list(filtered_dfs.keys())}")
        print(f"Compounds: {len(activity_matrix)}")
        print(f"Mean coverage: {coverage.mean()*100:.1f}%")
        print(f"Negative correlations: {n_negative} (selectivity trade-offs)")

        if len(activity_matrix) >= min_compounds:
            print(f"\n[OK] Dataset meets minimum requirement ({min_compounds} compounds)")
        else:
            print(f"\n[WARN] Dataset below target ({len(activity_matrix)} < {min_compounds})")
            print("  Consider combining families with --family all")

    return activity_matrix, metadata


def check_chembl_availability():
    """Check if ChEMBL API is accessible."""
    print("Checking ChEMBL API availability...")

    try:
        response = requests.get(f"{CHEMBL_API_BASE}/status.json", timeout=30)
        if response.status_code == 200:
            print("  [OK] ChEMBL API is accessible")
            return True
        else:
            print(f"  [ERROR] ChEMBL API returned status {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"  [ERROR] Cannot reach ChEMBL API: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Curate kinase selectivity panel dataset from ChEMBL'
    )
    parser.add_argument('--family', type=str, default='cdk',
                       choices=['cdk', 'jak', 'egfr', 'aurora', 'src', 'all'],
                       help='Kinase family to query (default: cdk)')
    parser.add_argument('--min-overlap', type=float, default=0.3,
                       help='Minimum pairwise compound overlap (default: 0.3)')
    parser.add_argument('--min-compounds', type=int, default=500,
                       help='Target minimum compounds (default: 500)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check ChEMBL API availability')
    args = parser.parse_args()

    if args.check_only:
        success = check_chembl_availability()
        sys.exit(0 if success else 1)

    # Check API
    if not check_chembl_availability():
        print("\nCannot proceed without ChEMBL API access.")
        sys.exit(1)

    print()

    # Run curation
    output_dir = Path(args.output_dir) if args.output_dir else None

    try:
        df, metadata = curate_kinase_selectivity_dataset(
            family=args.family,
            min_overlap=args.min_overlap,
            min_compounds=args.min_compounds,
            output_dir=output_dir,
            verbose=True
        )

        print("\n" + "=" * 60)
        print("NEXT STEPS")
        print("=" * 60)
        print("\n1. Train GNN on kinase selectivity data:")
        print(f"   python experiments/train_kinase_gnn.py --data outputs/kinase_data/kinase_{args.family}_activity_matrix.csv")
        print("\n2. Extract gradient conflict matrix")
        print("\n3. Compare gradient matrix to empirical correlations:")
        print("   - Positive G + positive empirical = synergy (same binding site)")
        print("   - Negative G + negative empirical = selectivity trade-off")
        print("   - This validates the method can discover antagonistic relationships")

    except KeyboardInterrupt:
        print("\n\nCuration interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during curation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
