#!/usr/bin/env python3
"""
ChEMBL Multi-Property Dataset Curation.

Creates a compound-aligned dataset where every molecule has measurements
for ALL selected properties. This enables interpretable gradient conflict
analysis - if tasks share molecules, gradient conflicts reflect true
biological/chemical relationships.

Key principle: MoleculeNet datasets have ~0% molecule overlap between tasks.
ChEMBL allows us to find compounds measured across diverse assay types.

Target properties (diverse types):
- Binding: hERG IC50, BACE1 IC50
- ADME: Solubility, LogD, Permeability (PAMPA/Caco-2)
- Toxicity: Ames, hERG liability
- Physicochemical: LogP, pKa

Usage:
    python scripts/curate_chembl_multiproperty.py
    python scripts/curate_chembl_multiproperty.py --min-compounds 500
    python scripts/curate_chembl_multiproperty.py --check-only
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

# Target ChEMBL IDs for key proteins
TARGETS = {
    'hERG': 'CHEMBL240',      # hERG potassium channel
    'BACE1': 'CHEMBL4822',    # Beta-secretase 1
    'CYP3A4': 'CHEMBL340',    # Cytochrome P450 3A4
    'P-gp': 'CHEMBL4302',     # P-glycoprotein (MDR1)
}

# Assay types for ADME/physicochemical properties
ASSAY_TYPES = {
    'solubility': ['Solubility', 'Aqueous solubility', 'kinetic solubility'],
    'logd': ['LogD', 'Distribution coefficient'],
    'permeability': ['PAMPA', 'Caco-2', 'permeability'],
    'plasma_protein_binding': ['PPB', 'Plasma protein binding', 'fu'],
    'microsomal_stability': ['HLM', 'microsomal', 'Clint'],
}


def query_chembl_activities(
    target_id: str,
    activity_type: str = 'IC50',
    limit: int = 10000,
    max_nm: float = 100000
) -> pd.DataFrame:
    """
    Query ChEMBL for activities against a specific target.

    Args:
        target_id: ChEMBL target ID (e.g., 'CHEMBL240')
        activity_type: Type of activity (IC50, Ki, EC50, etc.)
        limit: Maximum number of results
        max_nm: Maximum activity value in nM to include

    Returns:
        DataFrame with columns: chembl_id, smiles, activity_value, activity_type
    """
    url = f"{CHEMBL_API_BASE}/activity.json"

    params = {
        'target_chembl_id': target_id,
        'standard_type': activity_type,
        'standard_units': 'nM',
        'limit': limit,
        'offset': 0,
    }

    all_activities = []

    while True:
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            print(f"  Error querying ChEMBL: {e}")
            break

        activities = data.get('activities', [])
        if not activities:
            break

        for act in activities:
            # Filter by data quality
            if act.get('data_validity_comment'):
                continue  # Skip flagged data
            if act.get('standard_value') is None:
                continue
            if float(act['standard_value']) > max_nm:
                continue

            all_activities.append({
                'chembl_id': act.get('molecule_chembl_id'),
                'smiles': act.get('canonical_smiles'),
                'activity_value': float(act['standard_value']),
                'activity_type': activity_type,
                'assay_chembl_id': act.get('assay_chembl_id'),
                'pchembl_value': act.get('pchembl_value'),
            })

        # Check if more pages
        if data.get('page_meta', {}).get('next'):
            params['offset'] += limit
            time.sleep(0.5)  # Rate limiting
        else:
            break

    df = pd.DataFrame(all_activities)

    if len(df) > 0:
        # Remove duplicates - keep median value per compound
        df = df.groupby('chembl_id').agg({
            'smiles': 'first',
            'activity_value': 'median',
            'activity_type': 'first',
            'pchembl_value': 'first',
        }).reset_index()

    return df


def query_chembl_assays_by_description(
    search_terms: List[str],
    limit: int = 5000
) -> pd.DataFrame:
    """
    Query ChEMBL for assays matching description terms.

    Args:
        search_terms: List of terms to search in assay descriptions
        limit: Maximum results per term

    Returns:
        DataFrame with assay activities
    """
    all_activities = []

    for term in search_terms:
        url = f"{CHEMBL_API_BASE}/assay.json"
        params = {
            'assay_type': 'B',  # Binding assays
            'description__icontains': term,
            'limit': 100,
        }

        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            print(f"  Error searching for '{term}': {e}")
            continue

        assays = data.get('assays', [])

        for assay in assays[:10]:  # Limit assays per term
            assay_id = assay.get('assay_chembl_id')

            # Get activities for this assay
            act_url = f"{CHEMBL_API_BASE}/activity.json"
            act_params = {
                'assay_chembl_id': assay_id,
                'limit': limit,
            }

            try:
                act_response = requests.get(act_url, params=act_params, timeout=60)
                act_response.raise_for_status()
                act_data = act_response.json()
            except requests.RequestException:
                continue

            for act in act_data.get('activities', []):
                if act.get('standard_value') is None:
                    continue

                all_activities.append({
                    'chembl_id': act.get('molecule_chembl_id'),
                    'smiles': act.get('canonical_smiles'),
                    'activity_value': float(act['standard_value']),
                    'activity_type': act.get('standard_type'),
                    'assay_description': assay.get('description', '')[:100],
                })

            time.sleep(0.3)  # Rate limiting

    return pd.DataFrame(all_activities)


def get_compound_properties(chembl_ids: List[str]) -> pd.DataFrame:
    """
    Get calculated molecular properties for compounds.

    Args:
        chembl_ids: List of ChEMBL compound IDs

    Returns:
        DataFrame with molecular properties
    """
    properties = []

    # Process in batches
    batch_size = 50
    for i in range(0, len(chembl_ids), batch_size):
        batch = chembl_ids[i:i+batch_size]

        url = f"{CHEMBL_API_BASE}/molecule.json"
        params = {
            'molecule_chembl_id__in': ','.join(batch),
            'limit': batch_size,
        }

        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            print(f"  Error fetching properties batch {i//batch_size}: {e}")
            continue

        for mol in data.get('molecules', []):
            props = mol.get('molecule_properties', {}) or {}

            properties.append({
                'chembl_id': mol.get('molecule_chembl_id'),
                'smiles': mol.get('molecule_structures', {}).get('canonical_smiles'),
                'alogp': props.get('alogp'),
                'psa': props.get('psa'),
                'hba': props.get('hba'),
                'hbd': props.get('hbd'),
                'mw': props.get('mw_freebase'),
                'rtb': props.get('rtb'),
                'aromatic_rings': props.get('aromatic_rings'),
                'heavy_atoms': props.get('heavy_atoms'),
            })

        time.sleep(0.3)

        if (i + batch_size) % 500 == 0:
            print(f"    Fetched properties for {min(i + batch_size, len(chembl_ids))}/{len(chembl_ids)} compounds")

    return pd.DataFrame(properties)


def curate_multiproperty_dataset(
    min_compounds: int = 500,
    output_dir: Path = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Curate a multi-property dataset from ChEMBL.

    Strategy:
    1. Query hERG activities (most measured target)
    2. Find which of those compounds also have BACE1 data
    3. Get calculated properties (LogP, PSA, etc.) for all
    4. Filter to compounds with complete data

    Args:
        min_compounds: Minimum number of compounds required
        output_dir: Directory to save output files
        verbose: Print progress

    Returns:
        Tuple of (DataFrame with all properties, metadata dict)
    """
    if output_dir is None:
        output_dir = project_root / 'outputs' / 'chembl_data'
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("ChEMBL Multi-Property Dataset Curation")
        print("=" * 60)
        print()

    # Step 1: Query hERG IC50 data
    if verbose:
        print("Step 1: Querying hERG IC50 activities...")
    herg_df = query_chembl_activities(TARGETS['hERG'], 'IC50', limit=20000)
    if verbose:
        print(f"  Found {len(herg_df)} unique compounds with hERG IC50")

    # Step 2: Query BACE1 IC50 data
    if verbose:
        print("\nStep 2: Querying BACE1 IC50 activities...")
    bace_df = query_chembl_activities(TARGETS['BACE1'], 'IC50', limit=20000)
    if verbose:
        print(f"  Found {len(bace_df)} unique compounds with BACE1 IC50")

    # Step 3: Query CYP3A4 inhibition data
    if verbose:
        print("\nStep 3: Querying CYP3A4 inhibition activities...")
    cyp_df = query_chembl_activities(TARGETS['CYP3A4'], 'IC50', limit=20000)
    if verbose:
        print(f"  Found {len(cyp_df)} unique compounds with CYP3A4 IC50")

    # Step 4: Find overlapping compounds
    if verbose:
        print("\nStep 4: Finding compound overlaps...")

    herg_ids = set(herg_df['chembl_id'].dropna())
    bace_ids = set(bace_df['chembl_id'].dropna())
    cyp_ids = set(cyp_df['chembl_id'].dropna())

    # Find compounds with at least 2 activity measurements
    overlap_2 = (herg_ids & bace_ids) | (herg_ids & cyp_ids) | (bace_ids & cyp_ids)
    overlap_3 = herg_ids & bace_ids & cyp_ids

    if verbose:
        print(f"  hERG ∩ BACE1: {len(herg_ids & bace_ids)}")
        print(f"  hERG ∩ CYP3A4: {len(herg_ids & cyp_ids)}")
        print(f"  BACE1 ∩ CYP3A4: {len(bace_ids & cyp_ids)}")
        print(f"  All three: {len(overlap_3)}")

    # Use compounds with at least 2 bioactivity measurements
    selected_ids = list(overlap_2)

    if len(selected_ids) < min_compounds:
        if verbose:
            print(f"\n  Warning: Only {len(selected_ids)} compounds with 2+ activities")
            print(f"  Expanding to include compounds with hERG + properties...")
        # Fall back to just hERG compounds (we'll add calculated properties)
        selected_ids = list(herg_ids)[:min_compounds * 2]

    # Step 5: Get calculated molecular properties
    if verbose:
        print(f"\nStep 5: Fetching molecular properties for {len(selected_ids)} compounds...")
    props_df = get_compound_properties(selected_ids)
    if verbose:
        print(f"  Retrieved properties for {len(props_df)} compounds")

    # Step 6: Merge all data
    if verbose:
        print("\nStep 6: Merging datasets...")

    # Start with properties as base
    merged = props_df.copy()

    # Add hERG
    herg_subset = herg_df[['chembl_id', 'activity_value']].copy()
    herg_subset = herg_subset.rename(columns={'activity_value': 'hERG_IC50_nM'})
    merged = merged.merge(herg_subset, on='chembl_id', how='left')

    # Add BACE1
    bace_subset = bace_df[['chembl_id', 'activity_value']].copy()
    bace_subset = bace_subset.rename(columns={'activity_value': 'BACE1_IC50_nM'})
    merged = merged.merge(bace_subset, on='chembl_id', how='left')

    # Add CYP3A4
    cyp_subset = cyp_df[['chembl_id', 'activity_value']].copy()
    cyp_subset = cyp_subset.rename(columns={'activity_value': 'CYP3A4_IC50_nM'})
    merged = merged.merge(cyp_subset, on='chembl_id', how='left')

    # Convert IC50 to pIC50 (more ML-friendly)
    for col in ['hERG_IC50_nM', 'BACE1_IC50_nM', 'CYP3A4_IC50_nM']:
        if col in merged.columns:
            pic50_col = col.replace('IC50_nM', 'pIC50')
            merged[pic50_col] = -np.log10(merged[col] * 1e-9)

    # Step 7: Filter and finalize
    if verbose:
        print("\nStep 7: Filtering dataset...")

    # Define tasks (mix of bioactivity and physicochemical)
    tasks = {
        'hERG_pIC50': 'binding',        # Safety: hERG liability
        'BACE1_pIC50': 'binding',       # Efficacy: Alzheimer's target
        'CYP3A4_pIC50': 'ADME',         # Metabolism: drug interaction
        'alogp': 'physicochemical',      # Lipophilicity
        'psa': 'physicochemical',        # Polar surface area
        'mw': 'physicochemical',         # Molecular weight
    }

    # Count non-null values per task
    task_cols = list(tasks.keys())
    available_tasks = [t for t in task_cols if t in merged.columns]

    if verbose:
        print("\n  Available tasks and coverage:")
        for task in available_tasks:
            n_valid = merged[task].notna().sum()
            print(f"    {task}: {n_valid} compounds ({100*n_valid/len(merged):.1f}%)")

    # Filter to compounds with at least N properties measured
    min_properties = 4
    merged['n_properties'] = merged[available_tasks].notna().sum(axis=1)

    filtered = merged[merged['n_properties'] >= min_properties].copy()

    if verbose:
        print(f"\n  Compounds with {min_properties}+ properties: {len(filtered)}")

    # For the "complete overlap" subset, filter to compounds with ALL properties
    complete = merged[merged['n_properties'] == len(available_tasks)].copy()

    if verbose:
        print(f"  Compounds with ALL {len(available_tasks)} properties: {len(complete)}")

    # Step 8: Save datasets
    if verbose:
        print("\nStep 8: Saving datasets...")

    # Save full merged dataset
    merged.to_csv(output_dir / 'chembl_multiproperty_full.csv', index=False)

    # Save filtered dataset (4+ properties)
    filtered.to_csv(output_dir / 'chembl_multiproperty_filtered.csv', index=False)

    # Save complete-overlap dataset (all properties)
    if len(complete) >= 50:
        complete.to_csv(output_dir / 'chembl_multiproperty_complete.csv', index=False)

    # Save metadata
    metadata = {
        'n_compounds_full': len(merged),
        'n_compounds_filtered': len(filtered),
        'n_compounds_complete': len(complete),
        'tasks': tasks,
        'available_tasks': available_tasks,
        'min_properties_for_filtered': min_properties,
        'sources': {
            'hERG': TARGETS['hERG'],
            'BACE1': TARGETS['BACE1'],
            'CYP3A4': TARGETS['CYP3A4'],
        },
        'curation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"\n  Saved to: {output_dir}")
        print(f"    - chembl_multiproperty_full.csv ({len(merged)} compounds)")
        print(f"    - chembl_multiproperty_filtered.csv ({len(filtered)} compounds)")
        if len(complete) >= 50:
            print(f"    - chembl_multiproperty_complete.csv ({len(complete)} compounds)")
        print(f"    - metadata.json")

    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print("CURATION SUMMARY")
        print("=" * 60)
        print(f"\nTotal compounds: {len(merged)}")
        print(f"With 4+ properties: {len(filtered)}")
        print(f"With complete data: {len(complete)}")
        print(f"\nTasks ({len(available_tasks)}):")
        for task in available_tasks:
            print(f"  - {task} ({tasks.get(task, 'unknown')})")

        if len(filtered) >= min_compounds:
            print(f"\n✓ Dataset meets minimum requirement ({min_compounds} compounds)")
        else:
            print(f"\n✗ Dataset below minimum ({len(filtered)} < {min_compounds})")
            print("  Consider reducing min_properties or using filtered dataset")

    return filtered, metadata


def check_chembl_availability():
    """Check if ChEMBL API is accessible."""
    print("Checking ChEMBL API availability...")

    try:
        response = requests.get(f"{CHEMBL_API_BASE}/status.json", timeout=10)
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
    parser = argparse.ArgumentParser(description='Curate multi-property dataset from ChEMBL')
    parser.add_argument('--min-compounds', type=int, default=500,
                       help='Minimum number of compounds required (default: 500)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check ChEMBL API availability')
    args = parser.parse_args()

    if args.check_only:
        success = check_chembl_availability()
        sys.exit(0 if success else 1)

    # Check API first
    if not check_chembl_availability():
        print("\nCannot proceed without ChEMBL API access.")
        sys.exit(1)

    print()

    # Run curation
    output_dir = Path(args.output_dir) if args.output_dir else None

    try:
        df, metadata = curate_multiproperty_dataset(
            min_compounds=args.min_compounds,
            output_dir=output_dir,
            verbose=True
        )

        print("\n" + "=" * 60)
        print("NEXT STEPS")
        print("=" * 60)
        print("\n1. Train GNN on this dataset:")
        print("   python train_chembl_gnn.py")
        print("\n2. Analyze gradient conflicts:")
        print("   python scripts/analyze_chembl_gradients.py")
        print("\n3. Expected trade-offs to find:")
        print("   - hERG vs BACE1: selectivity challenge")
        print("   - alogp vs psa: lipophilicity-polarity trade-off")
        print("   - mw vs permeability: size-absorption relationship")

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
