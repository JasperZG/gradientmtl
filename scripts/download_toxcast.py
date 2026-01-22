#!/usr/bin/env python3
"""
Download ToxCast dataset for Strategy A validation.

ToxCast: EPA's Toxicity ForeCaster
- ~8,000 compounds
- 617 assay endpoints
- 100% compound overlap (panel assay design)
- Shares compounds with Tox21 (can verify overlap)

This validates: Method generalizes across panel assays (not Tox21-specific)
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

data_dir = project_root / 'outputs' / 'toxcast_data'
data_dir.mkdir(parents=True, exist_ok=True)


def download_toxcast():
    """Download ToxCast from available sources."""

    # Try multiple sources
    sources = [
        # DeepChem S3 (MoleculeNet)
        ('deepchem_s3', 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/toxcast_data.csv.gz'),
        # Alternative: direct CSV
        ('deepchem_csv', 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/toxcast.csv'),
    ]

    for name, url in sources:
        output_path = data_dir / f'toxcast_{name}.csv'
        if output_path.exists():
            print(f"Already have {name}")
            return output_path

        try:
            print(f"Trying {name}: {url[:60]}...")

            if url.endswith('.gz'):
                import gzip
                temp_path = data_dir / 'temp.csv.gz'
                urllib.request.urlretrieve(url, temp_path)
                with gzip.open(temp_path, 'rt') as f:
                    content = f.read()
                with open(output_path, 'w') as f:
                    f.write(content)
                temp_path.unlink()
            else:
                urllib.request.urlretrieve(url, output_path)

            # Verify it's valid
            df = pd.read_csv(output_path, nrows=5)
            print(f"  Success: {len(df.columns)} columns")
            return output_path

        except Exception as e:
            print(f"  Failed: {e}")
            if output_path.exists():
                output_path.unlink()

    return None


def analyze_toxcast(path: Path):
    """Analyze ToxCast dataset structure."""
    print("\n" + "=" * 60)
    print("ToxCast Dataset Analysis")
    print("=" * 60)

    df = pd.read_csv(path)
    print(f"\nShape: {df.shape}")
    print(f"Columns: {len(df.columns)}")

    # Find SMILES column
    smiles_col = None
    for col in df.columns:
        if 'smiles' in col.lower():
            smiles_col = col
            break

    if smiles_col:
        print(f"SMILES column: {smiles_col}")
        print(f"Compounds: {df[smiles_col].nunique()}")

    # Count task columns
    task_cols = [c for c in df.columns if c != smiles_col and c != 'mol_id']
    print(f"Task columns: {len(task_cols)}")

    # Check missing data pattern
    if len(task_cols) > 0:
        missing_pct = df[task_cols].isna().sum().sum() / (len(df) * len(task_cols)) * 100
        print(f"Missing data: {missing_pct:.1f}%")

        # Task coverage
        task_coverage = df[task_cols].notna().sum()
        print(f"\nTask coverage range: {task_coverage.min()} - {task_coverage.max()} compounds")

        # Show some task names
        print(f"\nSample task names:")
        for task in task_cols[:10]:
            n = df[task].notna().sum()
            print(f"  {task[:50]}: {n} compounds")

    return df


def compare_with_tox21(toxcast_df: pd.DataFrame, tox21_path: Path):
    """Compare compound overlap with Tox21."""
    print("\n" + "=" * 60)
    print("ToxCast vs Tox21 Overlap")
    print("=" * 60)

    from rdkit import Chem

    def canonicalize(smi):
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol:
                return Chem.MolToSmiles(mol, canonical=True)
        except:
            pass
        return None

    # Get ToxCast SMILES
    toxcast_smiles_col = None
    for col in toxcast_df.columns:
        if 'smiles' in col.lower():
            toxcast_smiles_col = col
            break

    if not toxcast_smiles_col:
        print("No SMILES column found in ToxCast")
        return

    print("Canonicalizing ToxCast SMILES...")
    toxcast_smiles = set()
    for smi in toxcast_df[toxcast_smiles_col].dropna():
        can = canonicalize(smi)
        if can:
            toxcast_smiles.add(can)

    print(f"ToxCast unique compounds: {len(toxcast_smiles)}")

    # Load Tox21
    if not tox21_path.exists():
        print(f"Tox21 not found at {tox21_path}")
        return

    tox21_df = pd.read_csv(tox21_path)

    print("Canonicalizing Tox21 SMILES...")
    tox21_smiles = set()
    for smi in tox21_df['smiles'].dropna():
        can = canonicalize(smi)
        if can:
            tox21_smiles.add(can)

    print(f"Tox21 unique compounds: {len(tox21_smiles)}")

    # Overlap
    overlap = toxcast_smiles & tox21_smiles
    print(f"\nOverlap: {len(overlap)} compounds")
    print(f"  {100*len(overlap)/len(tox21_smiles):.1f}% of Tox21")
    print(f"  {100*len(overlap)/len(toxcast_smiles):.1f}% of ToxCast")


def select_diverse_tasks(df: pd.DataFrame, n_tasks: int = 20):
    """Select diverse subset of ToxCast tasks for training."""
    print("\n" + "=" * 60)
    print(f"Selecting {n_tasks} Diverse Tasks")
    print("=" * 60)

    # Find SMILES and task columns
    smiles_col = None
    for col in df.columns:
        if 'smiles' in col.lower():
            smiles_col = col
            break

    task_cols = [c for c in df.columns if c != smiles_col and c != 'mol_id']

    # Filter tasks with sufficient data
    min_samples = 500
    good_tasks = []
    for task in task_cols:
        n = df[task].notna().sum()
        if n >= min_samples:
            good_tasks.append((task, n))

    print(f"Tasks with >= {min_samples} samples: {len(good_tasks)}")

    if len(good_tasks) == 0:
        print("No tasks meet minimum sample requirement")
        return None

    # Sort by coverage and select diverse set
    good_tasks.sort(key=lambda x: -x[1])

    # Select top n_tasks
    selected = [t[0] for t in good_tasks[:n_tasks]]

    print(f"\nSelected tasks:")
    for i, task in enumerate(selected):
        n = df[task].notna().sum()
        print(f"  {i+1}. {task[:50]}: {n} compounds")

    # Create filtered dataset
    output_df = df[[smiles_col] + selected].copy()
    output_df = output_df.rename(columns={smiles_col: 'smiles'})

    # Remove rows with all NaN
    task_data = output_df[selected]
    output_df = output_df[task_data.notna().any(axis=1)]

    print(f"\nFiltered dataset: {len(output_df)} compounds, {len(selected)} tasks")

    # Save
    output_path = data_dir / 'toxcast_selected.csv'
    output_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    return output_df


def main():
    print("=" * 60)
    print("ToxCast Dataset Download and Analysis")
    print("=" * 60)

    # Download
    path = download_toxcast()

    if path is None:
        print("\nFailed to download ToxCast. Trying alternative approach...")

        # Try to create from EPA ToxCast dashboard export
        # or use a subset approach
        print("\nAlternative: Use ToxCast subset from Tox21 compound library")
        return

    # Analyze
    df = analyze_toxcast(path)

    # Compare with Tox21
    tox21_path = project_root / 'outputs' / 'raw_data' / 'tox21.csv'
    compare_with_tox21(df, tox21_path)

    # Select diverse tasks
    selected_df = select_diverse_tasks(df, n_tasks=20)

    if selected_df is not None:
        print("\n" + "=" * 60)
        print("NEXT STEPS")
        print("=" * 60)
        print("""
1. Train GNN on ToxCast:
   python train_toxcast_gnn.py

2. Compare gradient matrix correlation with empirical:
   Expected: r > 0.6 (validates method isn't Tox21-specific)

3. Compare ToxCast G matrix with Tox21 G matrix:
   For overlapping tasks, G values should be consistent
""")


if __name__ == '__main__':
    main()
