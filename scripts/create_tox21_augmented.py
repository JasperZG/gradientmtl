#!/usr/bin/env python3
"""
Create Tox21 Augmented Dataset with Computed Molecular Descriptors.

Augments Tox21 toxicity data with RDKit-computed physicochemical properties.
This creates a truly diverse property dataset with 100% compound overlap.

Property types included:
- Toxicity (12 Tox21 endpoints) - classification
- Physicochemical (LogP, TPSA, MW, etc.) - regression

This allows analysis of:
- How molecular features (LogP, MW) relate to toxicity endpoints
- Cross-property-type gradient conflicts
- Validation of gradient method on diverse property types

Usage:
    python scripts/create_tox21_augmented.py
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski
except ImportError:
    print("Error: RDKit not installed")
    sys.exit(1)


def compute_molecular_descriptors(smiles: str) -> dict:
    """
    Compute molecular descriptors for a SMILES string.

    Returns dict of descriptor name -> value, or None if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    descriptors = {
        'MolWeight': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'HBD': Descriptors.NumHDonors(mol),
        'HBA': Descriptors.NumHAcceptors(mol),
        'RotatableBonds': Descriptors.NumRotatableBonds(mol),
        'RingCount': Descriptors.RingCount(mol),
        'AromaticRings': Descriptors.NumAromaticRings(mol),
        'FractionCSP3': Descriptors.FractionCSP3(mol),
        'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
    }

    return descriptors


def create_augmented_dataset(
    tox21_path: str,
    output_dir: Path,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Create augmented Tox21 dataset with computed descriptors.
    """
    if verbose:
        print("=" * 60)
        print("Creating Tox21 Augmented Dataset")
        print("=" * 60)

    # Load Tox21 data
    if verbose:
        print("\nLoading Tox21 data...")

    tox21 = pd.read_csv(tox21_path)

    # Check for smiles column
    if 'smiles' not in tox21.columns:
        # Look for it
        possible_cols = [c for c in tox21.columns if 'smiles' in c.lower()]
        if possible_cols:
            tox21 = tox21.rename(columns={possible_cols[0]: 'smiles'})
        else:
            print("Error: No SMILES column found")
            print("Columns:", list(tox21.columns))
            sys.exit(1)

    if verbose:
        print(f"  Loaded {len(tox21)} compounds")

    # Identify Tox21 task columns
    tox_tasks = [c for c in tox21.columns if c != 'smiles' and c not in ['mol_id', 'compound_id']]
    if verbose:
        print(f"  Tox21 tasks: {tox_tasks}")

    # Compute descriptors for each compound
    if verbose:
        print("\nComputing molecular descriptors...")

    descriptor_data = []
    failed = 0

    for i, row in tox21.iterrows():
        smi = row['smiles']
        desc = compute_molecular_descriptors(smi)

        if desc is None:
            failed += 1
            descriptor_data.append({k: np.nan for k in ['MolWeight', 'LogP', 'TPSA', 'HBD',
                                                         'HBA', 'RotatableBonds', 'RingCount',
                                                         'AromaticRings', 'FractionCSP3',
                                                         'NumHeteroatoms']})
        else:
            descriptor_data.append(desc)

        if verbose and (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(tox21)} compounds...")

    desc_df = pd.DataFrame(descriptor_data)

    if verbose:
        print(f"  Computed descriptors for {len(tox21) - failed}/{len(tox21)} compounds")
        if failed > 0:
            print(f"  Warning: {failed} compounds failed RDKit parsing")

    # Combine Tox21 + descriptors
    augmented = pd.concat([tox21, desc_df], axis=1)

    # Filter out rows with failed descriptors
    augmented = augmented.dropna(subset=['MolWeight'])

    if verbose:
        print(f"\nFinal dataset: {len(augmented)} compounds")

    # Rename columns for clarity
    # Prefix Tox21 tasks with 'Tox_'
    rename_map = {task: f'Tox_{task}' for task in tox_tasks}
    # Prefix descriptors with 'Phys_'
    desc_cols = ['MolWeight', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotatableBonds',
                 'RingCount', 'AromaticRings', 'FractionCSP3', 'NumHeteroatoms']
    rename_map.update({d: f'Phys_{d}' for d in desc_cols})

    augmented = augmented.rename(columns=rename_map)

    # Ensure smiles column is first
    cols = ['smiles'] + [c for c in augmented.columns if c != 'smiles']
    augmented = augmented[cols]

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    augmented.to_csv(output_dir / 'tox21_augmented.csv', index=False)

    # Get final task lists
    tox_tasks_final = [c for c in augmented.columns if c.startswith('Tox_')]
    phys_tasks_final = [c for c in augmented.columns if c.startswith('Phys_')]

    # Save metadata
    metadata = {
        'n_compounds': len(augmented),
        'n_tox_tasks': len(tox_tasks_final),
        'n_phys_tasks': len(phys_tasks_final),
        'tox_tasks': tox_tasks_final,
        'phys_tasks': phys_tasks_final,
        'task_types': {
            **{t: 'classification' for t in tox_tasks_final},
            **{t: 'regression' for t in phys_tasks_final}
        },
        'overlap': '100% (computed properties always available)',
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print("\n" + "=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)
        print(f"\nCompounds: {len(augmented)}")
        print(f"\nToxicity tasks ({len(tox_tasks_final)}):")
        for t in tox_tasks_final:
            n = augmented[t].notna().sum()
            print(f"  {t}: {n} ({100*n/len(augmented):.1f}%)")
        print(f"\nPhysicochemical tasks ({len(phys_tasks_final)}):")
        for t in phys_tasks_final:
            n = augmented[t].notna().sum()
            print(f"  {t}: {n} ({100*n/len(augmented):.1f}%)")

        print(f"\nSaved to: {output_dir}")
        print("  - tox21_augmented.csv")
        print("  - metadata.json")

    return augmented


def main():
    # Find Tox21 data
    tox21_paths = [
        project_root / 'outputs' / 'raw_data' / 'tox21.csv',
        project_root / 'data' / 'tox21.csv',
    ]

    tox21_path = None
    for p in tox21_paths:
        if p.exists():
            tox21_path = str(p)
            break

    if tox21_path is None:
        print("Error: Tox21 data not found")
        print("Expected locations:")
        for p in tox21_paths:
            print(f"  - {p}")
        sys.exit(1)

    output_dir = project_root / 'outputs' / 'tox21_augmented'

    df = create_augmented_dataset(tox21_path, output_dir, verbose=True)

    print("\n" + "=" * 60)
    print("EXPECTED FINDINGS")
    print("=" * 60)
    print("\nCross-category relationships (Tox vs Phys):")
    print("  - LogP vs Tox_NR-*: Lipophilic compounds often bind nuclear receptors")
    print("  - TPSA vs Tox_SR-*: Polar compounds may have stress response activity")
    print("  - MolWeight vs Tox_*: Larger molecules may have different tox profiles")
    print("\nWithin-category relationships:")
    print("  - Tox_NR-AR vs Tox_NR-AR-LBD: Same receptor, different assays")
    print("  - Phys_LogP vs Phys_TPSA: Inverse relationship (lipophilic = less polar)")

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("\n1. Train GNN on augmented dataset:")
    print("   python train_tox21_augmented_gnn.py")
    print("\n2. Analyze cross-property gradient conflicts")
    print("\n3. Compare to Tox21-only results (within-category)")


if __name__ == '__main__':
    main()
