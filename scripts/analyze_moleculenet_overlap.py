#!/usr/bin/env python3
"""
Analyze compound overlap across MoleculeNet datasets.

Goal: Find which datasets share enough compounds to enable
gradient conflict analysis across DIVERSE property types.

MoleculeNet Categories:
- Physiology: BBBP, Tox21, ToxCast, SIDER, ClinTox
- Biophysics: BACE, HIV
- Physical Chemistry: ESOL, FreeSolv, Lipophilicity
"""

import os
import sys
import urllib.request
from pathlib import Path
import pandas as pd
import numpy as np

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


def download_moleculenet_datasets(data_dir: Path) -> dict:
    """Download and load MoleculeNet datasets."""

    # Dataset URLs and column mappings
    datasets_info = {
        # Physical Chemistry (regression)
        'ESOL': {
            'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv',
            'smiles_col': 'smiles',
            'type': 'Physical Chemistry'
        },
        'FreeSolv': {
            'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv',
            'smiles_col': 'smiles',
            'type': 'Physical Chemistry'
        },
        'Lipophilicity': {
            'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv',
            'smiles_col': 'smiles',
            'type': 'Physical Chemistry'
        },
        # Biophysics (classification)
        'BACE': {
            'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv',
            'smiles_col': 'mol',
            'type': 'Biophysics'
        },
        'BBBP': {
            'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv',
            'smiles_col': 'smiles',
            'type': 'Physiology'
        },
        'HIV': {
            'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv',
            'smiles_col': 'smiles',
            'type': 'Biophysics'
        },
        # Physiology (classification, multi-task)
        'SIDER': {
            'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv',
            'smiles_col': 'smiles',
            'type': 'Physiology'
        },
        'ClinTox': {
            'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv',
            'smiles_col': 'smiles',
            'type': 'Physiology'
        },
        'Tox21': {
            'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv',
            'smiles_col': 'smiles',
            'type': 'Physiology'
        },
    }

    datasets = {}
    data_dir.mkdir(parents=True, exist_ok=True)

    for name, info in datasets_info.items():
        file_path = data_dir / f'{name.lower()}.csv'

        # Download if needed
        if not file_path.exists():
            print(f"Downloading {name}...")
            try:
                urllib.request.urlretrieve(info['url'], file_path)
            except Exception as e:
                print(f"  Failed: {e}")
                continue

        # Load and extract SMILES
        try:
            df = pd.read_csv(file_path)
            smiles_col = info['smiles_col']

            if smiles_col not in df.columns:
                # Try to find SMILES column
                smiles_cols = [c for c in df.columns if 'smiles' in c.lower() or 'mol' in c.lower()]
                if smiles_cols:
                    smiles_col = smiles_cols[0]
                else:
                    print(f"  Warning: No SMILES column in {name}")
                    continue

            # Canonicalize SMILES
            smiles_list = df[smiles_col].dropna().tolist()
            canonical = set()
            for smi in smiles_list:
                can = canonicalize_smiles(str(smi))
                if can:
                    canonical.add(can)

            datasets[name] = {
                'smiles': canonical,
                'count': len(canonical),
                'type': info['type']
            }
            print(f"  {name}: {len(canonical)} compounds ({info['type']})")

        except Exception as e:
            print(f"  Error loading {name}: {e}")

    return datasets


def compute_overlap_matrix(datasets: dict) -> pd.DataFrame:
    """Compute pairwise overlap between datasets."""
    names = list(datasets.keys())
    n = len(names)

    overlap_count = np.zeros((n, n), dtype=int)
    overlap_pct = np.zeros((n, n))

    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            set_i = datasets[name_i]['smiles']
            set_j = datasets[name_j]['smiles']

            intersection = len(set_i & set_j)
            overlap_count[i, j] = intersection

            # Percentage of smaller set that overlaps
            min_size = min(len(set_i), len(set_j))
            overlap_pct[i, j] = 100 * intersection / min_size if min_size > 0 else 0

    return pd.DataFrame(overlap_count, index=names, columns=names), \
           pd.DataFrame(overlap_pct, index=names, columns=names)


def find_high_overlap_combinations(datasets: dict, min_overlap: int = 100) -> list:
    """Find dataset combinations with significant overlap."""
    names = list(datasets.keys())
    combinations = []

    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            if i >= j:
                continue

            set_i = datasets[name_i]['smiles']
            set_j = datasets[name_j]['smiles']
            intersection = len(set_i & set_j)

            if intersection >= min_overlap:
                type_i = datasets[name_i]['type']
                type_j = datasets[name_j]['type']
                diverse = type_i != type_j

                combinations.append({
                    'dataset_1': name_i,
                    'dataset_2': name_j,
                    'overlap': intersection,
                    'type_1': type_i,
                    'type_2': type_j,
                    'diverse_types': diverse
                })

    return sorted(combinations, key=lambda x: (-x['diverse_types'], -x['overlap']))


def main():
    print("=" * 70)
    print("MoleculeNet Compound Overlap Analysis")
    print("=" * 70)
    print()

    data_dir = project_root / 'outputs' / 'moleculenet_data'

    # Download and load datasets
    print("Loading MoleculeNet datasets...")
    datasets = download_moleculenet_datasets(data_dir)

    if len(datasets) < 2:
        print("Error: Not enough datasets loaded")
        return

    # Compute overlap matrix
    print("\n" + "=" * 70)
    print("OVERLAP MATRIX (compound counts)")
    print("=" * 70)

    overlap_count, overlap_pct = compute_overlap_matrix(datasets)
    print("\nAbsolute overlap (number of shared compounds):")
    print(overlap_count.to_string())

    print("\n\nPercentage overlap (% of smaller dataset):")
    print(overlap_pct.round(1).to_string())

    # Find high-overlap combinations
    print("\n" + "=" * 70)
    print("HIGH-OVERLAP COMBINATIONS (>= 100 compounds)")
    print("=" * 70)

    combinations = find_high_overlap_combinations(datasets, min_overlap=100)

    print("\nDiverse property types (best for validation):")
    diverse = [c for c in combinations if c['diverse_types']]
    for c in diverse[:10]:
        print(f"  {c['dataset_1']:15} x {c['dataset_2']:15}: "
              f"{c['overlap']:5} compounds  ({c['type_1']} x {c['type_2']})")

    print("\nSame property type:")
    same = [c for c in combinations if not c['diverse_types']]
    for c in same[:10]:
        print(f"  {c['dataset_1']:15} x {c['dataset_2']:15}: "
              f"{c['overlap']:5} compounds  ({c['type_1']})")

    # Find best multi-dataset overlap
    print("\n" + "=" * 70)
    print("MULTI-DATASET OVERLAP ANALYSIS")
    print("=" * 70)

    # Check overlap across 3+ datasets
    names = list(datasets.keys())
    for size in [3, 4, 5]:
        from itertools import combinations as iter_comb

        best_overlap = 0
        best_combo = None
        best_types = None

        for combo in iter_comb(names, size):
            # Find intersection of all datasets
            intersection = datasets[combo[0]]['smiles'].copy()
            for name in combo[1:]:
                intersection &= datasets[name]['smiles']

            types = set(datasets[name]['type'] for name in combo)

            if len(intersection) > best_overlap:
                best_overlap = len(intersection)
                best_combo = combo
                best_types = types

        if best_combo:
            print(f"\nBest {size}-dataset combination:")
            print(f"  Datasets: {', '.join(best_combo)}")
            print(f"  Overlap: {best_overlap} compounds")
            print(f"  Property types: {', '.join(best_types)}")

    # Recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    print("""
Based on the overlap analysis:

1. SIDER + ClinTox + Tox21 combination:
   - All share clinical drug molecules
   - Covers: side effects, clinical toxicity, in-vitro toxicity
   - Expected overlap: significant (FDA-approved drugs)

2. BBBP + BACE + Lipophilicity:
   - CNS drugs often tested for all three
   - Covers: permeability, enzyme inhibition, physicochemical

3. Use SIDER's 27 side effect tasks as primary dataset:
   - All 1,427 compounds have all 27 labels
   - Diverse side effects (hepatotoxicity, cardiotoxicity, etc.)
   - Similar structure to Tox21 but clinical outcomes

Next step: Create merged multi-property dataset from best combination.
""")


if __name__ == '__main__':
    main()
