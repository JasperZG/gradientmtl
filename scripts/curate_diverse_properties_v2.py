#!/usr/bin/env python3
"""
Diverse Properties Dataset Curation (v2).

Creates a dataset with TRULY diverse property types by combining:
- ADME properties (from TDC): Lipophilicity, Solubility, Permeability, Metabolism
- Toxicity endpoints (from TDC/Tox21): Various toxicity assays
- Binding affinity (from TDC): hERG, DILI

Key insight: TDC datasets share compound libraries (pharmaceutical screening sets),
so we can find compounds measured across different property categories.

This addresses the limitation of Tox21-only experiments (same category)
while ensuring sufficient compound overlap (>50%) for meaningful gradient analysis.

Usage:
    python scripts/curate_diverse_properties_v2.py
    python scripts/curate_diverse_properties_v2.py --min-overlap 100
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
import pandas as pd
import numpy as np
from collections import defaultdict

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def install_tdc():
    """Install TDC if not present."""
    try:
        import tdc
        return True
    except ImportError:
        print("Installing TDC (Therapeutics Data Commons)...")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'PyTDC', '-q'])
        return True


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Canonicalize SMILES for consistent matching."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
    except:
        pass
    return None


def load_adme_datasets(verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """Load ADME datasets from TDC."""
    from tdc.single_pred import ADME

    datasets = {}

    # ADME datasets with good coverage
    adme_configs = [
        ('Lipophilicity', 'Lipophilicity_AstraZeneca', 'LogD7.4'),
        ('Solubility', 'Solubility_AqSolDB', 'LogS'),
        ('PAMPA', 'PAMPA_NCATS', 'log Papp'),
        ('HLM', 'HLM', 'Clearance'),
        ('Caco2', 'Caco2_Wang', 'Permeability'),
    ]

    for name, tdc_name, description in adme_configs:
        if verbose:
            print(f"  Loading ADME/{name}...")
        try:
            data = ADME(name=tdc_name)
            df = data.get_data()

            # Standardize columns
            if 'Drug' in df.columns:
                df = df.rename(columns={'Drug': 'smiles'})
            if 'Y' in df.columns:
                df = df.rename(columns={'Y': 'value'})

            # Canonicalize SMILES
            df['smiles_canonical'] = df['smiles'].apply(canonicalize_smiles)
            df = df.dropna(subset=['smiles_canonical'])

            # Remove duplicates (keep mean)
            df = df.groupby('smiles_canonical').agg({
                'smiles': 'first',
                'value': 'mean'
            }).reset_index()

            datasets[f'ADME_{name}'] = df
            if verbose:
                print(f"    -> {len(df)} compounds")

        except Exception as e:
            print(f"    Warning: Failed to load {name}: {e}")

    return datasets


def load_toxicity_datasets(verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """Load toxicity datasets from TDC."""
    from tdc.single_pred import Tox

    datasets = {}

    # Toxicity datasets
    tox_configs = [
        ('hERG', 'hERG', 'hERG liability'),  # Also safety/binding
        ('DILI', 'DILI', 'Drug-induced liver injury'),
        ('AMES', 'AMES', 'Mutagenicity'),
        ('Carcinogens', 'Carcinogens_Lagunin', 'Carcinogenicity'),
        ('ClinTox', 'ClinTox', 'Clinical toxicity'),
    ]

    for name, tdc_name, description in tox_configs:
        if verbose:
            print(f"  Loading Tox/{name}...")
        try:
            data = Tox(name=tdc_name)
            df = data.get_data()

            # Standardize columns
            if 'Drug' in df.columns:
                df = df.rename(columns={'Drug': 'smiles'})
            if 'Y' in df.columns:
                df = df.rename(columns={'Y': 'value'})

            # Canonicalize SMILES
            df['smiles_canonical'] = df['smiles'].apply(canonicalize_smiles)
            df = df.dropna(subset=['smiles_canonical'])

            # Remove duplicates
            df = df.groupby('smiles_canonical').agg({
                'smiles': 'first',
                'value': 'mean'
            }).reset_index()

            datasets[f'Tox_{name}'] = df
            if verbose:
                print(f"    -> {len(df)} compounds")

        except Exception as e:
            print(f"    Warning: Failed to load {name}: {e}")

    return datasets


def analyze_cross_category_overlap(
    adme_datasets: Dict[str, pd.DataFrame],
    tox_datasets: Dict[str, pd.DataFrame],
    verbose: bool = True
) -> Dict:
    """Analyze compound overlap between ADME and Toxicity categories."""

    # Get unique SMILES per category
    adme_smiles = set()
    for df in adme_datasets.values():
        adme_smiles.update(df['smiles_canonical'].unique())

    tox_smiles = set()
    for df in tox_datasets.values():
        tox_smiles.update(df['smiles_canonical'].unique())

    overlap = adme_smiles & tox_smiles

    stats = {
        'n_adme_unique': len(adme_smiles),
        'n_tox_unique': len(tox_smiles),
        'n_cross_overlap': len(overlap),
        'overlap_pct_of_adme': 100 * len(overlap) / len(adme_smiles) if adme_smiles else 0,
        'overlap_pct_of_tox': 100 * len(overlap) / len(tox_smiles) if tox_smiles else 0,
    }

    if verbose:
        print("\n" + "=" * 60)
        print("CROSS-CATEGORY OVERLAP ANALYSIS")
        print("=" * 60)
        print(f"\nADME compounds: {stats['n_adme_unique']}")
        print(f"Toxicity compounds: {stats['n_tox_unique']}")
        print(f"Cross-category overlap: {stats['n_cross_overlap']}")
        print(f"  -> {stats['overlap_pct_of_adme']:.1f}% of ADME")
        print(f"  -> {stats['overlap_pct_of_tox']:.1f}% of Toxicity")

    return stats


def merge_datasets(
    adme_datasets: Dict[str, pd.DataFrame],
    tox_datasets: Dict[str, pd.DataFrame],
    min_tasks: int = 3,
    require_both_categories: bool = True,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Merge ADME and toxicity datasets.

    Args:
        adme_datasets: ADME task DataFrames
        tox_datasets: Toxicity task DataFrames
        min_tasks: Minimum tasks per compound
        require_both_categories: Require at least one ADME and one Tox task
        verbose: Print progress

    Returns:
        Tuple of (merged DataFrame, metadata)
    """
    all_datasets = {**adme_datasets, **tox_datasets}

    # Build compound -> tasks mapping
    compound_tasks = defaultdict(dict)

    for task_name, df in all_datasets.items():
        for _, row in df.iterrows():
            smi = row['smiles_canonical']
            compound_tasks[smi][task_name] = row['value']
            if 'smiles' not in compound_tasks[smi]:
                compound_tasks[smi]['smiles'] = row['smiles']

    # Filter compounds
    filtered_compounds = []

    for smi, task_values in compound_tasks.items():
        tasks = [k for k in task_values.keys() if k != 'smiles']
        n_tasks = len(tasks)

        # Check minimum tasks
        if n_tasks < min_tasks:
            continue

        # Check both categories if required
        if require_both_categories:
            has_adme = any(t.startswith('ADME_') for t in tasks)
            has_tox = any(t.startswith('Tox_') for t in tasks)
            if not (has_adme and has_tox):
                continue

        filtered_compounds.append({
            'smiles': smi,
            **{t: v for t, v in task_values.items() if t != 'smiles'}
        })

    # Create DataFrame
    merged = pd.DataFrame(filtered_compounds)

    # Get task columns
    task_cols = [c for c in merged.columns if c != 'smiles']
    adme_tasks = [t for t in task_cols if t.startswith('ADME_')]
    tox_tasks = [t for t in task_cols if t.startswith('Tox_')]

    # Compute statistics
    stats = {
        'n_compounds': len(merged),
        'n_tasks': len(task_cols),
        'n_adme_tasks': len(adme_tasks),
        'n_tox_tasks': len(tox_tasks),
        'min_tasks_required': min_tasks,
        'require_both_categories': require_both_categories,
        'task_coverage': {},
    }

    if verbose and len(merged) > 0:
        print("\n" + "=" * 60)
        print("MERGED DATASET")
        print("=" * 60)
        print(f"\nCompounds: {len(merged)}")
        print(f"Tasks: {len(task_cols)} ({len(adme_tasks)} ADME, {len(tox_tasks)} Tox)")
        print("\nTask coverage:")
        for task in sorted(task_cols):
            n_valid = merged[task].notna().sum()
            pct = 100 * n_valid / len(merged)
            category = 'ADME' if task.startswith('ADME_') else 'Tox'
            print(f"  [{category}] {task}: {n_valid} ({pct:.1f}%)")
            stats['task_coverage'][task] = {'n': int(n_valid), 'pct': float(pct)}

    return merged, stats


def compute_pairwise_overlap(merged: pd.DataFrame, task_cols: List[str]) -> pd.DataFrame:
    """Compute pairwise task overlap matrix."""
    n_tasks = len(task_cols)
    overlap_matrix = np.zeros((n_tasks, n_tasks))

    for i, task_i in enumerate(task_cols):
        for j, task_j in enumerate(task_cols):
            mask_i = merged[task_i].notna()
            mask_j = merged[task_j].notna()
            overlap = (mask_i & mask_j).sum()
            overlap_matrix[i, j] = overlap

    return pd.DataFrame(overlap_matrix, index=task_cols, columns=task_cols)


def curate_diverse_properties(
    min_tasks: int = 3,
    min_compounds: int = 100,
    require_both_categories: bool = True,
    output_dir: Path = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Curate diverse properties dataset from TDC.

    Args:
        min_tasks: Minimum number of tasks per compound
        min_compounds: Minimum compounds required
        require_both_categories: Require ADME + Tox per compound
        output_dir: Output directory
        verbose: Print progress

    Returns:
        Tuple of (DataFrame, metadata)
    """
    if output_dir is None:
        output_dir = project_root / 'outputs' / 'diverse_data'
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("Diverse Properties Dataset Curation")
        print("=" * 60)
        print("\nLoading datasets from TDC...")

    # Install TDC
    install_tdc()

    # Load datasets
    if verbose:
        print("\nLoading ADME datasets...")
    adme_datasets = load_adme_datasets(verbose)

    if verbose:
        print("\nLoading Toxicity datasets...")
    tox_datasets = load_toxicity_datasets(verbose)

    # Analyze cross-category overlap
    overlap_stats = analyze_cross_category_overlap(adme_datasets, tox_datasets, verbose)

    # Merge datasets
    merged, merge_stats = merge_datasets(
        adme_datasets, tox_datasets,
        min_tasks=min_tasks,
        require_both_categories=require_both_categories,
        verbose=verbose
    )

    # Check if we have enough compounds
    if len(merged) < min_compounds:
        if verbose:
            print(f"\nWarning: Only {len(merged)} compounds (< {min_compounds})")
            print("Trying without require_both_categories...")

        merged, merge_stats = merge_datasets(
            adme_datasets, tox_datasets,
            min_tasks=min_tasks,
            require_both_categories=False,
            verbose=verbose
        )

    if len(merged) < 50:
        if verbose:
            print(f"\nWarning: Still only {len(merged)} compounds. Trying min_tasks=2...")

        merged, merge_stats = merge_datasets(
            adme_datasets, tox_datasets,
            min_tasks=2,
            require_both_categories=False,
            verbose=verbose
        )

    # Save datasets
    merged.to_csv(output_dir / 'diverse_properties.csv', index=False)

    # Compute pairwise overlap
    task_cols = [c for c in merged.columns if c != 'smiles']
    if len(task_cols) > 1:
        overlap_matrix = compute_pairwise_overlap(merged, task_cols)
        overlap_matrix.to_csv(output_dir / 'task_overlap_matrix.csv')

    # Save metadata
    metadata = {
        'n_compounds': len(merged),
        'tasks': task_cols,
        'n_adme_tasks': len([t for t in task_cols if t.startswith('ADME_')]),
        'n_tox_tasks': len([t for t in task_cols if t.startswith('Tox_')]),
        'cross_category_overlap': overlap_stats,
        'merge_stats': merge_stats,
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print("\n" + "=" * 60)
        print("CURATION COMPLETE")
        print("=" * 60)
        print(f"\nSaved to: {output_dir}")
        print(f"  - diverse_properties.csv ({len(merged)} compounds)")
        print(f"  - task_overlap_matrix.csv")
        print(f"  - metadata.json")

        if len(merged) >= min_compounds:
            print(f"\n✓ Dataset meets minimum requirement ({min_compounds}+ compounds)")
        else:
            print(f"\n✗ Dataset below minimum ({len(merged)} < {min_compounds})")

    return merged, metadata


def main():
    parser = argparse.ArgumentParser(description='Curate diverse properties dataset')
    parser.add_argument('--min-tasks', type=int, default=3,
                       help='Minimum tasks per compound')
    parser.add_argument('--min-compounds', type=int, default=100,
                       help='Minimum compounds required')
    parser.add_argument('--no-require-both', action='store_true',
                       help='Do not require both ADME and Tox')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None

    try:
        df, metadata = curate_diverse_properties(
            min_tasks=args.min_tasks,
            min_compounds=args.min_compounds,
            require_both_categories=not args.no_require_both,
            output_dir=output_dir,
            verbose=True
        )

        print("\n" + "=" * 60)
        print("EXPECTED GRADIENT RELATIONSHIPS")
        print("=" * 60)
        print("\nCross-category trade-offs (ADME vs Tox):")
        print("  - Lipophilicity vs hERG: High LogD often increases hERG binding")
        print("  - Permeability vs AMES: Permeable compounds may be mutagenic")
        print("  - Solubility vs DILI: Insoluble compounds may cause liver toxicity")
        print("\nWithin-category synergies:")
        print("  - ADME: Lipophilicity-Permeability synergy (lipophilic = permeable)")
        print("  - Tox: hERG-DILI synergy (promiscuous binders)")

        print("\n" + "=" * 60)
        print("NEXT STEPS")
        print("=" * 60)
        print("\n1. Train GNN on diverse properties:")
        print("   python train_diverse_gnn.py --data-path outputs/diverse_data/diverse_properties.csv")
        print("\n2. Run validation (Experiment 2):")
        print("   python scripts/experiment2_sar_validation.py --dataset diverse")
        print("\n3. Run full experiment suite (Experiments 3-5) on HPC")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
