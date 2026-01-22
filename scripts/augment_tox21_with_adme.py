#!/usr/bin/env python3
"""
Augment Tox21 with Measured ADME Properties.

Strategy: Find Tox21 compounds that ALSO have experimental ADME measurements
in public databases. This gives us:
- 12 toxicity properties (Tox21)
- 2-5 ADME properties (from TDC/MoleculeNet)
- 100% overlap by construction (same compounds)

This is the key insight: Instead of finding compounds measured across domains,
we take Tox21 and find which compounds ALSO have ADME data.

Sources:
- TDC ADME datasets (already downloaded)
- MoleculeNet ADME (ESOL, Lipophilicity, FreeSolv)
- Any other sources with SMILES matching

Usage:
    python scripts/augment_tox21_with_adme.py
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Optional
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


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


def load_tox21_data() -> pd.DataFrame:
    """Load Tox21 data with canonical SMILES."""
    print("Loading Tox21 data...")

    # Try multiple paths
    paths = [
        project_root / 'outputs' / 'raw_data' / 'tox21.csv',
        project_root / 'data' / 'tox21.csv',
    ]

    tox21 = None
    for path in paths:
        if path.exists():
            tox21 = pd.read_csv(path)
            print(f"  Loaded from: {path}")
            break

    if tox21 is None:
        raise FileNotFoundError("Tox21 data not found")

    # Find SMILES column
    smiles_col = None
    for col in tox21.columns:
        if 'smiles' in col.lower():
            smiles_col = col
            break

    if smiles_col is None:
        # Assume SMILES is last column or there's a mol_id
        print("  Warning: No SMILES column found, checking structure...")
        print(f"  Columns: {list(tox21.columns)}")
        return None

    # Rename to standard
    if smiles_col != 'smiles':
        tox21 = tox21.rename(columns={smiles_col: 'smiles'})

    # Canonicalize
    print("  Canonicalizing SMILES...")
    tox21['smiles_canonical'] = tox21['smiles'].apply(canonicalize_smiles)
    tox21 = tox21.dropna(subset=['smiles_canonical'])

    # Get task columns (exclude smiles columns)
    task_cols = [c for c in tox21.columns
                 if c not in ['smiles', 'smiles_canonical', 'mol_id', 'compound_id']]

    print(f"  Tox21: {len(tox21)} compounds, {len(task_cols)} tasks")
    print(f"  Tasks: {task_cols}")

    return tox21


def load_tdc_adme_data() -> Dict[str, pd.DataFrame]:
    """Load TDC ADME datasets."""
    print("\nLoading TDC ADME datasets...")

    try:
        from tdc.single_pred import ADME
    except ImportError:
        print("  Warning: TDC not installed, skipping TDC datasets")
        return {}

    datasets = {}

    adme_configs = [
        ('Solubility', 'Solubility_AqSolDB'),
        ('Lipophilicity', 'Lipophilicity_AstraZeneca'),
        ('Caco2', 'Caco2_Wang'),
        ('PAMPA', 'PAMPA_NCATS'),
        ('PPB', 'PPBR_AZ'),
        ('HLM', 'HLM'),
        ('Bioavailability', 'Bioavailability_Ma'),
        ('HIA', 'HIA_Hou'),
        ('VDss', 'VDss_Lombardo'),
        ('Clearance_Hepatocyte', 'Clearance_Hepatocyte_AZ'),
        ('Clearance_Microsome', 'Clearance_Microsome_AZ'),
        ('CYP2D6_Inhibitor', 'CYP2D6_Veith'),
        ('CYP3A4_Inhibitor', 'CYP3A4_Veith'),
        ('CYP2C9_Inhibitor', 'CYP2C9_Veith'),
    ]

    for name, tdc_name in adme_configs:
        try:
            data = ADME(name=tdc_name)
            df = data.get_data()

            if 'Drug' in df.columns:
                df = df.rename(columns={'Drug': 'smiles'})
            if 'Y' in df.columns:
                df = df.rename(columns={'Y': 'value'})

            df['smiles_canonical'] = df['smiles'].apply(canonicalize_smiles)
            df = df.dropna(subset=['smiles_canonical'])
            df = df.drop_duplicates(subset=['smiles_canonical'])

            datasets[f'ADME_{name}'] = df
            print(f"  [OK] ADME_{name}: {len(df)} compounds")

        except Exception as e:
            print(f"  [FAIL] {name}: {str(e)[:40]}")

    return datasets


def load_moleculenet_adme() -> Dict[str, pd.DataFrame]:
    """Load MoleculeNet ADME datasets from local files."""
    print("\nLoading MoleculeNet ADME datasets...")

    datasets = {}

    # Define specific value columns for each dataset
    moleculenet_configs = {
        'ESOL': {
            'path': project_root / 'outputs' / 'moleculenet_data' / 'esol.csv',
            'value_col': 'measured log solubility in mols per litre',
        },
        'FreeSolv': {
            'path': project_root / 'outputs' / 'moleculenet_data' / 'freesolv.csv',
            'value_col': 'expt',  # experimental hydration free energy
        },
        'Lipophilicity_MN': {
            'path': project_root / 'outputs' / 'moleculenet_data' / 'lipophilicity.csv',
            'value_col': 'exp',  # experimental logD
        },
    }

    for name, config in moleculenet_configs.items():
        path = config['path']
        if not path.exists():
            continue

        try:
            df = pd.read_csv(path)

            # Find SMILES column
            smiles_col = None
            for col in df.columns:
                if 'smiles' in col.lower():
                    smiles_col = col
                    break

            if smiles_col is None:
                print(f"  [FAIL] {name}: No SMILES column")
                continue

            # Get value column
            value_col = config.get('value_col')
            if value_col and value_col in df.columns:
                df = df.rename(columns={smiles_col: 'smiles', value_col: 'value'})
            else:
                # Try to find a numeric column
                found = False
                for col in df.columns:
                    if col != smiles_col and df[col].dtype in [np.float64, np.int64]:
                        df = df.rename(columns={smiles_col: 'smiles', col: 'value'})
                        found = True
                        break
                if not found:
                    print(f"  [FAIL] {name}: No numeric value column")
                    continue

            df['smiles_canonical'] = df['smiles'].apply(canonicalize_smiles)
            df = df.dropna(subset=['smiles_canonical', 'value'])
            df = df.drop_duplicates(subset=['smiles_canonical'])

            datasets[f'MN_{name}'] = df
            print(f"  [OK] MN_{name}: {len(df)} compounds")

        except Exception as e:
            print(f"  [FAIL] {name}: {str(e)[:40]}")

    return datasets


def match_tox21_to_adme(
    tox21: pd.DataFrame,
    adme_datasets: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Match Tox21 compounds to ADME measurements.
    """
    print("\n" + "=" * 60)
    print("MATCHING TOX21 TO ADME")
    print("=" * 60)

    # Get Tox21 canonical SMILES
    tox21_smiles = set(tox21['smiles_canonical'].unique())
    print(f"\nTox21 compounds: {len(tox21_smiles)}")

    # Build lookup: canonical SMILES -> ADME values
    compound_adme = defaultdict(dict)

    for dataset_name, df in adme_datasets.items():
        adme_smiles = set(df['smiles_canonical'].unique())

        # Find overlap
        overlap = tox21_smiles & adme_smiles
        overlap_pct = len(overlap) / len(tox21_smiles) * 100

        print(f"  {dataset_name}: {len(overlap)} matches ({overlap_pct:.1f}%)")

        # Store values
        for smi in overlap:
            match = df[df['smiles_canonical'] == smi]
            if len(match) > 0:
                compound_adme[smi][dataset_name] = match.iloc[0]['value']

    # Count compounds by number of ADME properties
    adme_counts = defaultdict(int)
    for smi, props in compound_adme.items():
        adme_counts[len(props)] += 1

    print("\n" + "=" * 60)
    print("ADME PROPERTY COVERAGE")
    print("=" * 60)

    print("\nTox21 compounds by ADME property count:")
    cumulative = 0
    for n in sorted(adme_counts.keys(), reverse=True):
        cumulative += adme_counts[n]
        print(f"  {n}+ ADME properties: {cumulative} compounds")

    # Filter to compounds with at least 1 ADME property
    compounds_with_adme = [smi for smi, props in compound_adme.items() if len(props) >= 1]

    print(f"\nTotal Tox21 compounds with 1+ ADME properties: {len(compounds_with_adme)}")

    # Create augmented dataset
    print("\n" + "=" * 60)
    print("CREATING AUGMENTED DATASET")
    print("=" * 60)

    # Get Tox21 task columns
    tox_cols = [c for c in tox21.columns
                if c not in ['smiles', 'smiles_canonical', 'mol_id', 'compound_id']]

    # Filter Tox21 to compounds with ADME
    augmented = tox21[tox21['smiles_canonical'].isin(compounds_with_adme)].copy()

    # Add ADME columns
    adme_cols = list(adme_datasets.keys())
    for col in adme_cols:
        augmented[col] = augmented['smiles_canonical'].map(
            lambda x: compound_adme.get(x, {}).get(col, np.nan)
        )

    # Summary
    print(f"\nAugmented dataset:")
    print(f"  Compounds: {len(augmented)}")
    print(f"  Toxicity tasks: {len(tox_cols)}")
    print(f"  ADME tasks: {len(adme_cols)}")
    print(f"  Total tasks: {len(tox_cols) + len(adme_cols)}")

    # Coverage
    print("\nADME property coverage in augmented dataset:")
    for col in adme_cols:
        n = augmented[col].notna().sum()
        if n > 0:
            print(f"  {col}: {n} ({100*n/len(augmented):.1f}%)")

    return augmented


def decision_gate(augmented: pd.DataFrame, adme_cols: List[str]) -> str:
    """
    Apply decision gate for Tox21 + ADME augmentation.
    """
    print("\n" + "=" * 60)
    print("DECISION GATE")
    print("=" * 60)

    n_compounds = len(augmented)

    # Count compounds with 2+ ADME properties
    adme_count = augmented[adme_cols].notna().sum(axis=1)
    compounds_2plus_adme = (adme_count >= 2).sum()
    compounds_3plus_adme = (adme_count >= 3).sum()

    # ADME columns with >100 compounds
    valid_adme_cols = [col for col in adme_cols
                       if augmented[col].notna().sum() >= 100]

    print()
    print(f"Total compounds with any ADME: {n_compounds}")
    print(f"Compounds with 2+ ADME properties: {compounds_2plus_adme}")
    print(f"Compounds with 3+ ADME properties: {compounds_3plus_adme}")
    print(f"ADME properties with >100 compounds: {len(valid_adme_cols)}")
    print()

    # Decision criteria
    if n_compounds >= 1000 and len(valid_adme_cols) >= 3:
        decision = 'SUCCESS'
        print("*** DECISION: SUCCESS ***")
        print()
        print("Tox21 + ADME augmentation provides sufficient data!")
        print(f"  - {n_compounds} compounds (12 tox + {len(valid_adme_cols)} ADME)")
        print(f"  - True cross-domain: Toxicity + ADME")
        print(f"  - 100% overlap by construction")
        print()
        print("Proceed with training and validation!")

    elif n_compounds >= 500 and len(valid_adme_cols) >= 2:
        decision = 'MARGINAL'
        print("--- DECISION: MARGINAL ---")
        print()
        print("Tox21 + ADME has borderline data.")
        print("Options:")
        print("  1. Proceed with reduced ADME (2 properties)")
        print("  2. Try DrugBank augmentation")

    else:
        decision = 'FAILED'
        print("XXX DECISION: FAILED XXX")
        print()
        print("Insufficient ADME coverage for Tox21 compounds.")
        print("Pivot to DrugBank or accept limitation.")

    return decision


def main():
    output_dir = project_root / 'outputs' / 'tox21_adme_augmented'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TOX21 + ADME AUGMENTATION")
    print("=" * 60)
    print()

    # Step 1: Load Tox21
    tox21 = load_tox21_data()
    if tox21 is None:
        print("Failed to load Tox21 data")
        return

    # Step 2: Load ADME datasets
    adme_datasets = {}

    # TDC ADME
    tdc_adme = load_tdc_adme_data()
    adme_datasets.update(tdc_adme)

    # MoleculeNet ADME
    mn_adme = load_moleculenet_adme()
    adme_datasets.update(mn_adme)

    if not adme_datasets:
        print("No ADME datasets found!")
        return

    print(f"\nTotal ADME datasets loaded: {len(adme_datasets)}")

    # Step 3: Match Tox21 to ADME
    augmented = match_tox21_to_adme(tox21, adme_datasets)

    # Step 4: Decision gate
    adme_cols = list(adme_datasets.keys())
    decision = decision_gate(augmented, adme_cols)

    # Step 5: Save results
    # Filter to useful columns
    tox_cols = [c for c in tox21.columns
                if c not in ['smiles', 'smiles_canonical', 'mol_id', 'compound_id']]

    # Keep only ADME columns with data
    useful_adme = [col for col in adme_cols if augmented[col].notna().sum() > 0]

    save_cols = ['smiles_canonical'] + tox_cols + useful_adme
    save_cols = [c for c in save_cols if c in augmented.columns]

    augmented_save = augmented[save_cols].copy()
    augmented_save = augmented_save.rename(columns={'smiles_canonical': 'smiles'})

    augmented_save.to_csv(output_dir / 'tox21_adme_augmented.csv', index=False)

    # Metadata
    metadata = {
        'n_compounds': len(augmented_save),
        'n_tox_tasks': len(tox_cols),
        'n_adme_tasks': len(useful_adme),
        'tox_tasks': tox_cols,
        'adme_tasks': useful_adme,
        'decision': decision,
        'coverage': {
            col: int(augmented_save[col].notna().sum())
            for col in useful_adme
        }
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print(f"  - tox21_adme_augmented.csv ({len(augmented_save)} compounds)")
    print(f"  - metadata.json")

    return decision


if __name__ == '__main__':
    main()
