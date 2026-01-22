#!/usr/bin/env python3
"""
Prepare diverse ToxCast subset excluding Tox21 overlap.

Goal: Select tasks from DIFFERENT assay families to test method
generalization beyond Tox21-like assays.

ToxCast assay families:
- TOX21_*: Tox21 assays (exclude for independence)
- ACEA_*: Cell proliferation
- APR_*: High-content imaging
- ATG_*: Gene expression (reporter assays)
- BSK_*: BioSeek immune panel
- NVS_*: Various GPCR, kinase assays
- Tanguay_*: Zebrafish developmental
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
data_dir = project_root / 'outputs' / 'toxcast_data'


def select_diverse_toxcast():
    """Select diverse non-Tox21 ToxCast assays."""

    # Load full ToxCast
    toxcast_path = data_dir / 'toxcast_deepchem_s3.csv'
    if not toxcast_path.exists():
        print(f"ToxCast not found at {toxcast_path}")
        return None

    df = pd.read_csv(toxcast_path)
    print(f"Loaded ToxCast: {df.shape}")

    # Find task columns
    smiles_col = 'smiles'
    task_cols = [c for c in df.columns if c != smiles_col and c != 'mol_id']

    # Group by assay family (prefix before first underscore)
    assay_families = {}
    for task in task_cols:
        prefix = task.split('_')[0]
        if prefix not in assay_families:
            assay_families[prefix] = []
        assay_families[prefix].append(task)

    print(f"\nAssay families:")
    for prefix, tasks in sorted(assay_families.items(), key=lambda x: -len(x[1])):
        print(f"  {prefix}: {len(tasks)} assays")

    # Select tasks from diverse families (exclude TOX21 for independence test)
    target_families = {
        'ATG': 3,    # Gene expression reporters
        'BSK': 3,    # BioSeek immune panel
        'NVS': 3,    # GPCR, kinase assays
        'APR': 3,    # High-content imaging
        'ACEA': 2,   # Cell proliferation
        'CEETOX': 2, # Cytotoxicity
        'OT': 2,     # Odyssey Thera
        'Tanguay': 2, # Zebrafish
    }

    selected_tasks = []
    min_samples = 1000

    for family, n_select in target_families.items():
        if family not in assay_families:
            continue

        # Get tasks with sufficient data
        family_tasks = []
        for task in assay_families[family]:
            n = df[task].notna().sum()
            if n >= min_samples:
                family_tasks.append((task, n))

        # Sort by coverage
        family_tasks.sort(key=lambda x: -x[1])

        # Select top n
        for task, n in family_tasks[:n_select]:
            selected_tasks.append(task)
            print(f"  Selected {task}: {n} compounds")

    print(f"\nTotal selected: {len(selected_tasks)} tasks")

    if len(selected_tasks) < 10:
        print("\nNot enough tasks. Adding TOX21 assays for comparison...")
        # Add some TOX21 assays
        tox21_tasks = []
        for task in assay_families.get('TOX21', []):
            n = df[task].notna().sum()
            if n >= min_samples:
                tox21_tasks.append((task, n))
        tox21_tasks.sort(key=lambda x: -x[1])
        for task, n in tox21_tasks[:12]:
            if task not in selected_tasks:
                selected_tasks.append(task)
                print(f"  Added TOX21 {task}: {n} compounds")

    # Create filtered dataset
    output_df = df[[smiles_col] + selected_tasks].copy()

    # Remove rows with all NaN
    task_data = output_df[selected_tasks]
    output_df = output_df[task_data.notna().any(axis=1)]

    print(f"\nFiltered dataset: {len(output_df)} compounds, {len(selected_tasks)} tasks")

    # Check overlap pattern
    print("\nCompound coverage per task:")
    for task in selected_tasks[:10]:
        n = output_df[task].notna().sum()
        pct = 100 * n / len(output_df)
        print(f"  {task[:40]}: {n} ({pct:.1f}%)")

    # Save
    output_path = data_dir / 'toxcast_diverse.csv'
    output_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    # Also create a version with only non-TOX21 tasks
    non_tox21_tasks = [t for t in selected_tasks if not t.startswith('TOX21')]
    if len(non_tox21_tasks) >= 8:
        non_tox21_df = df[[smiles_col] + non_tox21_tasks].copy()
        non_tox21_df = non_tox21_df[non_tox21_df[non_tox21_tasks].notna().any(axis=1)]
        non_tox21_path = data_dir / 'toxcast_non_tox21.csv'
        non_tox21_df.to_csv(non_tox21_path, index=False)
        print(f"Saved non-TOX21 subset: {len(non_tox21_df)} compounds, {len(non_tox21_tasks)} tasks")

    return output_df


def main():
    print("=" * 60)
    print("Preparing Diverse ToxCast Subset")
    print("=" * 60)

    df = select_diverse_toxcast()

    if df is not None:
        print("\n" + "=" * 60)
        print("VALIDATION PLAN")
        print("=" * 60)
        print("""
Strategy A Test (Dataset Generalization):
1. Train GNN on ToxCast diverse subset
2. Compute gradient conflict matrix G
3. Compute empirical correlation matrix
4. Compare G vs empirical: expect r > 0.6

If successful: Method generalizes to different panel assays
If fails: Method may be Tox21-specific
""")


if __name__ == '__main__':
    main()
