#!/usr/bin/env python3
"""
Strategy B: Curate diverse property dataset from overlapping MoleculeNet compounds.

Goal: Find compounds with DIFFERENT property types measured:
- Physical Chemistry: Lipophilicity, ESOL (solubility), FreeSolv
- Physiology: BBBP (permeability)
- Biophysics: BACE (binding), HIV (activity)

Approach:
1. Load all MoleculeNet datasets
2. Find compounds present in 3+ datasets from DIFFERENT categories
3. Create merged dataset with diverse property coverage
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def canonicalize_smiles(smiles: str) -> str:
    """Canonicalize SMILES for matching."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(str(smiles))
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
    except:
        pass
    return None


def load_all_datasets():
    """Load all available MoleculeNet datasets."""
    data_dir = project_root / 'outputs' / 'moleculenet_data'

    # Dataset info: (file, smiles_col, value_col, property_type)
    datasets_info = {
        'Lipophilicity': ('lipophilicity.csv', 'smiles', 'exp', 'Physical'),
        'ESOL': ('esol.csv', 'smiles', 'measured log solubility in mols per litre', 'Physical'),
        'FreeSolv': ('freesolv.csv', 'smiles', 'expt', 'Physical'),
        'BBBP': ('bbbp.csv', 'smiles', 'p_np', 'Physiology'),
        'BACE': ('bace.csv', 'mol', 'Class', 'Biophysics'),
        'HIV': ('hiv.csv', 'smiles', 'HIV_active', 'Biophysics'),
    }

    datasets = {}
    for name, (filename, smiles_col, value_col, prop_type) in datasets_info.items():
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"  {name}: not found")
            continue

        try:
            df = pd.read_csv(filepath)

            # Find SMILES column
            if smiles_col not in df.columns:
                smiles_cols = [c for c in df.columns if 'smiles' in c.lower() or 'mol' in c.lower()]
                if smiles_cols:
                    smiles_col = smiles_cols[0]
                else:
                    print(f"  {name}: no SMILES column")
                    continue

            # Canonicalize
            df['smiles_canonical'] = df[smiles_col].apply(canonicalize_smiles)
            df = df.dropna(subset=['smiles_canonical'])

            # Get value column
            if value_col in df.columns:
                values = df.set_index('smiles_canonical')[value_col].to_dict()
            else:
                # Try to find it
                for col in df.columns:
                    if col not in [smiles_col, 'smiles_canonical', 'mol_id', 'smiles']:
                        values = df.set_index('smiles_canonical')[col].to_dict()
                        value_col = col
                        break
                else:
                    print(f"  {name}: no value column")
                    continue

            datasets[name] = {
                'values': values,
                'type': prop_type,
                'n_compounds': len(values)
            }
            print(f"  {name} ({prop_type}): {len(values)} compounds")

        except Exception as e:
            print(f"  {name}: error - {e}")

    return datasets


def find_diverse_overlap(datasets: dict):
    """Find compounds with coverage across diverse property types."""

    # Build compound -> properties mapping
    compound_properties = defaultdict(dict)
    compound_types = defaultdict(set)

    for name, data in datasets.items():
        prop_type = data['type']
        for smiles, value in data['values'].items():
            if pd.notna(value):
                compound_properties[smiles][name] = value
                compound_types[smiles].add(prop_type)

    # Count compounds by type diversity
    diversity_counts = defaultdict(int)
    for smiles, types in compound_types.items():
        n_types = len(types)
        diversity_counts[n_types] += 1

    print("\nCompound diversity:")
    for n, count in sorted(diversity_counts.items()):
        print(f"  {n} property types: {count} compounds")

    return compound_properties, compound_types


def create_diverse_dataset(compound_properties, compound_types, datasets, min_types=2):
    """Create dataset with compounds having diverse property coverage."""

    # Filter to compounds with min_types different property types
    diverse_compounds = [
        smi for smi, types in compound_types.items()
        if len(types) >= min_types
    ]

    print(f"\nCompounds with {min_types}+ property types: {len(diverse_compounds)}")

    if len(diverse_compounds) < 100:
        print("Warning: Very few compounds with diverse properties")
        min_types = 1
        diverse_compounds = list(compound_types.keys())
        print(f"Relaxed to {min_types}+ types: {len(diverse_compounds)} compounds")

    # Build merged dataframe
    rows = []
    for smiles in diverse_compounds:
        row = {'smiles': smiles}
        row.update(compound_properties[smiles])
        rows.append(row)

    df = pd.DataFrame(rows)

    # Reorder columns: smiles first, then by property type
    property_order = []
    for prop_type in ['Physical', 'Physiology', 'Biophysics']:
        for name, data in datasets.items():
            if data['type'] == prop_type and name in df.columns:
                property_order.append(name)

    df = df[['smiles'] + property_order]

    return df


def analyze_overlap_quality(df, datasets):
    """Analyze the quality of overlap for gradient conflict analysis."""
    print("\n" + "=" * 60)
    print("OVERLAP QUALITY ANALYSIS")
    print("=" * 60)

    task_cols = [c for c in df.columns if c != 'smiles']

    # Pairwise overlap
    print("\nPairwise overlap matrix:")
    overlap_matrix = {}
    for i, task1 in enumerate(task_cols):
        for task2 in task_cols[i+1:]:
            mask1 = df[task1].notna()
            mask2 = df[task2].notna()
            overlap = (mask1 & mask2).sum()

            type1 = datasets[task1]['type']
            type2 = datasets[task2]['type']
            diverse = type1 != type2

            key = f"{task1[:8]} x {task2[:8]}"
            overlap_matrix[key] = {
                'overlap': overlap,
                'diverse': diverse,
                'types': f"{type1} x {type2}"
            }

    # Sort by diversity then overlap
    sorted_pairs = sorted(
        overlap_matrix.items(),
        key=lambda x: (-x[1]['diverse'], -x[1]['overlap'])
    )

    print("\nDiverse property pairs (for testing):")
    for key, data in sorted_pairs[:10]:
        if data['diverse']:
            print(f"  {key}: {data['overlap']} compounds ({data['types']})")

    print("\nSame property pairs (for comparison):")
    for key, data in sorted_pairs:
        if not data['diverse']:
            print(f"  {key}: {data['overlap']} compounds ({data['types']})")

    # Calculate expected utility
    diverse_pairs = [(k, v) for k, v in sorted_pairs if v['diverse'] and v['overlap'] >= 50]
    print(f"\nUsable diverse pairs (>= 50 overlap): {len(diverse_pairs)}")

    return overlap_matrix


def main():
    print("=" * 60)
    print("Strategy B: Diverse Property Dataset Curation")
    print("=" * 60)

    print("\nLoading datasets...")
    datasets = load_all_datasets()

    if len(datasets) < 3:
        print("Error: Need at least 3 datasets")
        return

    # Find overlapping compounds
    compound_properties, compound_types = find_diverse_overlap(datasets)

    # Create diverse dataset
    df = create_diverse_dataset(compound_properties, compound_types, datasets, min_types=2)

    # Analyze overlap quality
    overlap_matrix = analyze_overlap_quality(df, datasets)

    # Save
    output_dir = project_root / 'outputs' / 'diverse_properties'
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / 'diverse_properties.csv', index=False)
    print(f"\nSaved to {output_dir / 'diverse_properties.csv'}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    task_cols = [c for c in df.columns if c != 'smiles']
    print(f"\nDataset: {len(df)} compounds, {len(task_cols)} properties")
    print("\nProperty coverage:")
    for col in task_cols:
        n = df[col].notna().sum()
        prop_type = datasets[col]['type']
        print(f"  {col} ({prop_type}): {n} ({100*n/len(df):.1f}%)")

    # Check if we have enough diverse overlap
    diverse_pairs = [
        (k, v) for k, v in overlap_matrix.items()
        if v['diverse'] and v['overlap'] >= 50
    ]

    if len(diverse_pairs) >= 3:
        print("\n[READY] Dataset has sufficient diverse property overlap for testing")
        print("\nNext: Run train_diverse_properties_gnn.py")
    else:
        print("\n[WARNING] Limited diverse property overlap")
        print("Consider using ToxCast validation as primary evidence")


if __name__ == '__main__':
    main()
