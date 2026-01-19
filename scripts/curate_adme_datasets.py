#!/usr/bin/env python3
"""
Direct ADME Dataset Curation (No TDC dependency).

Downloads ADME datasets directly from public sources and merges them
to create a compound-aligned multi-property dataset.

Sources:
- Lipophilicity: MoleculeNet (AstraZeneca)
- Solubility: AqSolDB (public)
- FreeSolv: MoleculeNet (hydration free energy)
- ESOL: MoleculeNet (solubility)

These datasets have significant compound overlap due to common benchmarking.

Usage:
    python scripts/curate_adme_datasets.py
"""

import os
import sys
import json
import urllib.request
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def canonicalize_smiles(smiles: str) -> str:
    """Canonicalize SMILES for matching."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
    except:
        pass
    return smiles


def download_file(url: str, path: Path) -> bool:
    """Download a file from URL."""
    try:
        print(f"  Downloading from {url[:60]}...")
        urllib.request.urlretrieve(url, path)
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        return False


def load_moleculenet_datasets(data_dir: Path) -> dict:
    """
    Load MoleculeNet datasets.

    MoleculeNet datasets available from DeepChem:
    - Lipophilicity
    - FreeSolv
    - ESOL
    """
    datasets = {}

    # MoleculeNet dataset URLs (from DeepChem)
    moleculenet_urls = {
        'Lipophilicity': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv',
        'FreeSolv': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv',
        'ESOL': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv',
    }

    # Column mappings: dataset -> (smiles_col, value_col)
    column_maps = {
        'Lipophilicity': ('smiles', 'exp'),
        'FreeSolv': ('smiles', 'expt'),
        'ESOL': ('smiles', 'measured log solubility in mols per litre'),
    }

    for name, url in moleculenet_urls.items():
        file_path = data_dir / f'{name.lower()}.csv'

        if not file_path.exists():
            if not download_file(url, file_path):
                continue

        try:
            df = pd.read_csv(file_path)

            # Get column names for this dataset
            smiles_col, value_col = column_maps[name]

            if smiles_col not in df.columns:
                smiles_cols = [c for c in df.columns if 'smiles' in c.lower()]
                if smiles_cols:
                    smiles_col = smiles_cols[0]
                else:
                    print(f"  Warning: No SMILES column found in {name}")
                    continue

            if value_col not in df.columns:
                print(f"  Warning: Value column '{value_col}' not found in {name}")
                print(f"  Available: {df.columns.tolist()}")
                continue

            df = df[[smiles_col, value_col]]
            df.columns = ['smiles', 'value']

            # Canonicalize SMILES
            df['smiles_canonical'] = df['smiles'].apply(canonicalize_smiles)

            # Remove duplicates
            df = df.groupby('smiles_canonical').agg({
                'smiles': 'first',
                'value': 'mean'
            }).reset_index()

            datasets[name] = df
            print(f"  Loaded {name}: {len(df)} compounds")

        except Exception as e:
            print(f"  Error loading {name}: {e}")

    return datasets


def load_pubchem_properties(smiles_list: list, data_dir: Path) -> pd.DataFrame:
    """
    Calculate molecular properties using RDKit.

    Properties:
    - MolWt: Molecular weight
    - LogP: Calculated octanol/water partition coefficient
    - TPSA: Topological polar surface area
    - NumHDonors: Hydrogen bond donors
    - NumHAcceptors: Hydrogen bond acceptors
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski

    properties = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            properties.append({
                'smiles_canonical': smi,
                'MolWt': np.nan,
                'cLogP': np.nan,
                'TPSA': np.nan,
                'NumHDonors': np.nan,
                'NumHAcceptors': np.nan,
            })
            continue

        properties.append({
            'smiles_canonical': smi,
            'MolWt': Descriptors.MolWt(mol),
            'cLogP': Descriptors.MolLogP(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumHDonors': Lipinski.NumHDonors(mol),
            'NumHAcceptors': Lipinski.NumHAcceptors(mol),
        })

    return pd.DataFrame(properties)


def curate_adme_dataset(
    output_dir: Path = None,
    verbose: bool = True
) -> tuple:
    """
    Curate multi-property ADME dataset.

    Strategy:
    1. Load MoleculeNet ADME datasets
    2. Find overlapping compounds
    3. Add calculated properties for all compounds
    4. Create merged dataset

    Returns:
        Tuple of (DataFrame, metadata dict)
    """
    if output_dir is None:
        output_dir = project_root / 'outputs' / 'adme_data'
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("ADME Multi-Property Dataset Curation")
        print("=" * 60)
        print()

    # Step 1: Load MoleculeNet datasets
    if verbose:
        print("Step 1: Loading MoleculeNet datasets...")
    datasets = load_moleculenet_datasets(output_dir)

    if len(datasets) < 2:
        raise RuntimeError("Failed to load enough datasets")

    # Step 2: Find all unique compounds
    if verbose:
        print("\nStep 2: Finding unique compounds...")

    all_smiles = set()
    compound_datasets = {}

    for name, df in datasets.items():
        for smi in df['smiles_canonical']:
            all_smiles.add(smi)
            if smi not in compound_datasets:
                compound_datasets[smi] = set()
            compound_datasets[smi].add(name)

    if verbose:
        print(f"  Total unique compounds: {len(all_smiles)}")

    # Count overlaps
    overlap_counts = {}
    for smi, present_in in compound_datasets.items():
        n = len(present_in)
        overlap_counts[n] = overlap_counts.get(n, 0) + 1

    if verbose:
        print("\n  Overlap statistics:")
        for n in sorted(overlap_counts.keys()):
            print(f"    In {n} datasets: {overlap_counts[n]} compounds")

    # Step 3: Merge datasets
    if verbose:
        print("\nStep 3: Merging datasets...")

    # Start with SMILES as base
    merged = pd.DataFrame({'smiles_canonical': list(all_smiles)})

    # Add each dataset
    for name, df in datasets.items():
        subset = df[['smiles_canonical', 'value', 'smiles']].copy()
        subset = subset.rename(columns={'value': name, 'smiles': f'{name}_smiles'})
        merged = merged.merge(subset, on='smiles_canonical', how='left')

    # Use first available SMILES as canonical
    smiles_cols = [c for c in merged.columns if c.endswith('_smiles')]
    merged['smiles'] = merged[smiles_cols].bfill(axis=1).iloc[:, 0]
    merged = merged.drop(columns=smiles_cols)

    # Step 4: Add calculated properties
    if verbose:
        print("\nStep 4: Calculating molecular properties...")

    props_df = load_pubchem_properties(merged['smiles_canonical'].tolist(), output_dir)
    merged = merged.merge(props_df, on='smiles_canonical', how='left')

    # Step 5: Reorder columns
    task_cols = list(datasets.keys()) + ['MolWt', 'cLogP', 'TPSA', 'NumHDonors', 'NumHAcceptors']
    merged = merged[['smiles'] + [c for c in task_cols if c in merged.columns]]

    # Step 6: Summary
    if verbose:
        print("\n" + "=" * 60)
        print("FINAL DATASET")
        print("=" * 60)
        print(f"\nTotal compounds: {len(merged)}")
        print("\nTask coverage:")
        for col in merged.columns:
            if col != 'smiles':
                n = merged[col].notna().sum()
                print(f"  {col}: {n} ({100*n/len(merged):.1f}%)")

    # Save
    merged.to_csv(output_dir / 'adme_multiproperty.csv', index=False)

    # Filter to compounds with 3+ measured properties
    measured_tasks = list(datasets.keys())
    merged['n_measured'] = merged[measured_tasks].notna().sum(axis=1)
    filtered = merged[merged['n_measured'] >= 2].drop(columns=['n_measured'])
    filtered.to_csv(output_dir / 'adme_multiproperty_overlap.csv', index=False)

    if verbose:
        print(f"\nCompounds with 2+ measured properties: {len(filtered)}")
        print(f"\nSaved to {output_dir}")

    metadata = {
        'n_compounds_total': len(merged),
        'n_compounds_overlap': len(filtered),
        'datasets': list(datasets.keys()),
        'calculated_properties': ['MolWt', 'cLogP', 'TPSA', 'NumHDonors', 'NumHAcceptors'],
        'overlap_counts': overlap_counts,
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    return filtered, metadata


def main():
    try:
        df, metadata = curate_adme_dataset(verbose=True)

        print("\n" + "=" * 60)
        print("NEXT STEPS")
        print("=" * 60)
        print("\n1. Train GNN on this dataset:")
        print("   python train_adme_gnn.py")
        print("\n2. Expected trade-offs:")
        print("   - Lipophilicity vs ESOL: physicochemical inverse (G < -0.3)")
        print("   - cLogP vs TPSA: inverse (lipophilic = low polarity)")
        print("   - FreeSolv vs Lipophilicity: related (hydrophobicity)")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
