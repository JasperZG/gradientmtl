#!/usr/bin/env python3
"""
TDC (Therapeutics Data Commons) Multi-Property Dataset Curation.

Creates a compound-aligned dataset from TDC ADME datasets.
TDC provides pre-curated, downloadable datasets - more reliable than ChEMBL API.

Key insight: Many ADME properties are measured on the same compound libraries
(e.g., Pfizer/AstraZeneca internal libraries). This gives us compound overlap.

Selected datasets (diverse property types):
- Lipophilicity (physicochemical)
- Solubility (AqSolDB) (physicochemical)
- PAMPA (permeability)
- HLM (metabolism - Human Liver Microsome stability)
- PPB (Plasma Protein Binding)
- Caco2 (permeability)

Usage:
    python scripts/curate_tdc_multiproperty.py
    python scripts/curate_tdc_multiproperty.py --min-overlap 100
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set
import pandas as pd
import numpy as np

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


def canonicalize_smiles(smiles: str) -> str:
    """Canonicalize SMILES for consistent matching."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
    except:
        pass
    return smiles


def load_tdc_datasets(output_dir: Path, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Load ADME datasets from TDC.

    Returns:
        Dictionary of task_name -> DataFrame with 'smiles' and 'value' columns
    """
    from tdc.single_pred import ADME

    datasets = {}

    # Define datasets to load
    # Format: (name, TDC_name, is_regression, description)
    dataset_configs = [
        ('Lipophilicity', 'Lipophilicity_AstraZeneca', True, 'LogD7.4 - lipophilicity'),
        ('Solubility', 'Solubility_AqSolDB', True, 'Aqueous solubility (LogS)'),
        ('PAMPA', 'PAMPA_NCATS', True, 'Permeability (log Papp)'),
        ('HLM', 'HLM', True, 'Human liver microsome stability'),
        ('PPB', 'PPBR_AZ', True, 'Plasma protein binding (%)'),
        ('Caco2', 'Caco2_Wang', True, 'Caco-2 permeability'),
    ]

    for name, tdc_name, is_regression, description in dataset_configs:
        if verbose:
            print(f"  Loading {name} ({tdc_name})...")

        try:
            data = ADME(name=tdc_name)
            df = data.get_data()

            # Standardize column names
            if 'Drug' in df.columns:
                df = df.rename(columns={'Drug': 'smiles'})
            if 'Y' in df.columns:
                df = df.rename(columns={'Y': 'value'})

            # Canonicalize SMILES for matching
            df['smiles_canonical'] = df['smiles'].apply(canonicalize_smiles)

            # Remove duplicates (keep mean)
            df = df.groupby('smiles_canonical').agg({
                'smiles': 'first',
                'value': 'mean'
            }).reset_index()

            datasets[name] = df

            if verbose:
                print(f"    Loaded {len(df)} compounds")

        except Exception as e:
            print(f"    Warning: Failed to load {name}: {e}")

    return datasets


def find_overlapping_compounds(
    datasets: Dict[str, pd.DataFrame],
    min_datasets: int = 3
) -> Tuple[pd.DataFrame, Dict]:
    """
    Find compounds that appear in multiple datasets.

    Args:
        datasets: Dictionary of task_name -> DataFrame
        min_datasets: Minimum number of datasets a compound must appear in

    Returns:
        Tuple of (merged DataFrame, overlap statistics)
    """
    # Build compound -> datasets mapping
    compound_presence = {}

    for name, df in datasets.items():
        for smi in df['smiles_canonical'].unique():
            if smi not in compound_presence:
                compound_presence[smi] = set()
            compound_presence[smi].add(name)

    # Count overlaps
    overlap_counts = {i: 0 for i in range(1, len(datasets) + 1)}
    for smi, present_in in compound_presence.items():
        overlap_counts[len(present_in)] = overlap_counts.get(len(present_in), 0) + 1

    print("\nCompound overlap statistics:")
    for n_datasets, count in sorted(overlap_counts.items()):
        print(f"  In {n_datasets} datasets: {count} compounds")

    # Filter to compounds in at least min_datasets
    overlapping_smiles = {
        smi for smi, present_in in compound_presence.items()
        if len(present_in) >= min_datasets
    }

    print(f"\nCompounds in {min_datasets}+ datasets: {len(overlapping_smiles)}")

    # Build merged dataset
    # Start with first dataset
    task_names = list(datasets.keys())
    first_task = task_names[0]
    merged = datasets[first_task][['smiles_canonical', 'smiles']].copy()
    merged = merged.rename(columns={'smiles': 'smiles_original'})
    merged[first_task] = datasets[first_task]['value']

    # Merge other datasets
    for name in task_names[1:]:
        df = datasets[name][['smiles_canonical', 'value']].copy()
        df = df.rename(columns={'value': name})
        merged = merged.merge(df, on='smiles_canonical', how='outer')

    # Filter to overlapping compounds
    merged = merged[merged['smiles_canonical'].isin(overlapping_smiles)]

    # Use canonical SMILES as main column
    merged = merged.rename(columns={'smiles_canonical': 'smiles'})
    merged = merged.drop(columns=['smiles_original'], errors='ignore')

    stats = {
        'total_unique_compounds': len(compound_presence),
        'overlap_counts': overlap_counts,
        'n_overlapping': len(overlapping_smiles),
        'min_datasets_required': min_datasets,
    }

    return merged, stats


def curate_tdc_dataset(
    min_overlap: int = 3,
    output_dir: Path = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Curate multi-property dataset from TDC.

    Args:
        min_overlap: Minimum number of tasks each compound must have
        output_dir: Directory to save outputs
        verbose: Print progress

    Returns:
        Tuple of (DataFrame, metadata)
    """
    if output_dir is None:
        output_dir = project_root / 'outputs' / 'tdc_data'
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("TDC Multi-Property Dataset Curation")
        print("=" * 60)
        print()

    # Install TDC if needed
    install_tdc()

    # Load datasets
    if verbose:
        print("Loading TDC ADME datasets...")
    datasets = load_tdc_datasets(output_dir, verbose)

    if len(datasets) < 2:
        raise RuntimeError("Failed to load enough datasets")

    # Find overlapping compounds
    if verbose:
        print("\nFinding compound overlaps...")
    merged, overlap_stats = find_overlapping_compounds(datasets, min_overlap)

    if len(merged) < 50:
        print(f"\nWarning: Only {len(merged)} compounds with {min_overlap}+ tasks")
        print("Trying with min_overlap=2...")
        merged, overlap_stats = find_overlapping_compounds(datasets, 2)

    # Summary of final dataset
    task_cols = [c for c in merged.columns if c != 'smiles']

    if verbose:
        print("\n" + "=" * 60)
        print("FINAL DATASET")
        print("=" * 60)
        print(f"\nCompounds: {len(merged)}")
        print(f"Tasks: {len(task_cols)}")
        print("\nTask coverage:")
        for task in task_cols:
            n_valid = merged[task].notna().sum()
            print(f"  {task}: {n_valid} ({100*n_valid/len(merged):.1f}%)")

    # Save datasets
    merged.to_csv(output_dir / 'tdc_multiproperty.csv', index=False)

    # Also save version with only complete cases
    complete = merged.dropna()
    if len(complete) >= 20:
        complete.to_csv(output_dir / 'tdc_multiproperty_complete.csv', index=False)
        if verbose:
            print(f"\nComplete cases (all tasks): {len(complete)}")

    # Metadata
    metadata = {
        'n_compounds': len(merged),
        'n_complete': len(complete),
        'tasks': task_cols,
        'min_overlap': min_overlap,
        'overlap_stats': overlap_stats,
        'task_coverage': {
            task: int(merged[task].notna().sum())
            for task in task_cols
        }
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"\nSaved to {output_dir}")

    return merged, metadata


def main():
    parser = argparse.ArgumentParser(description='Curate multi-property dataset from TDC')
    parser.add_argument('--min-overlap', type=int, default=3,
                       help='Minimum number of tasks per compound')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None

    try:
        df, metadata = curate_tdc_dataset(
            min_overlap=args.min_overlap,
            output_dir=output_dir,
            verbose=True
        )

        print("\n" + "=" * 60)
        print("NEXT STEPS")
        print("=" * 60)
        print("\n1. Train GNN on this dataset:")
        print("   python train_tdc_gnn.py")
        print("\n2. Expected trade-offs to find:")
        print("   - Lipophilicity vs Solubility: physicochemical inverse")
        print("   - Lipophilicity vs PPB: both driven by hydrophobicity")
        print("   - Permeability vs Solubility: lipophilic compounds permeable but insoluble")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
