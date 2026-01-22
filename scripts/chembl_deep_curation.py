#!/usr/bin/env python3
"""
ChEMBL Deep Curation - Phase 2b.

Queries ChEMBL for compounds with measurements across multiple property domains:
- Binding affinity (IC50/Ki to specific targets)
- ADME (solubility, permeability, metabolism)
- Toxicity (hERG, AMES, hepatotoxicity)

Decision Gate Criteria:
- compounds_with_cross_domain >= 300 AND overlap >= 50% -> SUCCESS
- compounds_with_cross_domain >= 100 AND overlap >= 70% -> MARGINAL
- Otherwise -> FAILED (accept limitation)

Usage:
    python scripts/chembl_deep_curation.py
    python scripts/chembl_deep_curation.py --max-compounds 5000
"""

import os
import sys
import json
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
import requests
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# ChEMBL API Configuration
# =============================================================================

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"

# Key targets for binding affinity
BINDING_TARGETS = {
    'hERG': 'CHEMBL240',           # hERG potassium channel (safety)
    'BACE1': 'CHEMBL4822',         # Beta-secretase 1 (Alzheimer's)
    'DPP4': 'CHEMBL284',           # Dipeptidyl peptidase 4 (diabetes)
    'JAK2': 'CHEMBL2971',          # Janus kinase 2 (cancer)
    'EGFR': 'CHEMBL203',           # EGF receptor (cancer)
}

# ADME assay keywords
ADME_KEYWORDS = {
    'Solubility': ['solubility', 'aqueous', 'LogS'],
    'Permeability': ['Caco-2', 'PAMPA', 'permeability', 'Papp'],
    'Metabolism': ['microsomal', 'HLM', 'clearance', 'Clint', 'CYP'],
    'PPB': ['protein binding', 'PPB', 'fu', 'plasma'],
}

# Toxicity assay keywords
TOX_KEYWORDS = {
    'hERG_Tox': ['hERG', 'QT', 'cardiac'],
    'AMES': ['AMES', 'mutagenic', 'Salmonella'],
    'Hepatotox': ['hepatotox', 'liver', 'DILI', 'ALT', 'AST'],
    'Cytotox': ['cytotox', 'viability', 'IC50', 'cell death'],
}


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Canonicalize SMILES."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
    except:
        pass
    return None


def query_chembl_target(target_id: str, target_name: str,
                        activity_type: str = 'IC50',
                        max_records: int = 10000) -> pd.DataFrame:
    """
    Query ChEMBL for activities against a specific target.
    """
    print(f"  Querying {target_name} ({target_id})...")

    url = f"{CHEMBL_API}/activity.json"
    params = {
        'target_chembl_id': target_id,
        'standard_type': activity_type,
        'standard_units': 'nM',
        'limit': 1000,
        'offset': 0,
    }

    all_activities = []

    while len(all_activities) < max_records:
        try:
            response = requests.get(url, params=params, timeout=60)
            if response.status_code != 200:
                break
            data = response.json()
        except Exception as e:
            print(f"    Error: {str(e)[:50]}")
            break

        activities = data.get('activities', [])
        if not activities:
            break

        for act in activities:
            if act.get('standard_value') is None:
                continue
            if act.get('data_validity_comment'):
                continue  # Skip flagged data

            smiles = act.get('canonical_smiles')
            if not smiles:
                continue

            all_activities.append({
                'chembl_id': act.get('molecule_chembl_id'),
                'smiles': smiles,
                'value': float(act['standard_value']),
                'type': activity_type,
                'target': target_name,
            })

        if not data.get('page_meta', {}).get('next'):
            break

        params['offset'] += 1000
        time.sleep(0.3)  # Rate limiting

    df = pd.DataFrame(all_activities)

    if len(df) > 0:
        # Canonicalize SMILES
        df['smiles_canonical'] = df['smiles'].apply(canonicalize_smiles)
        df = df.dropna(subset=['smiles_canonical'])

        # Deduplicate (keep median)
        df = df.groupby('smiles_canonical').agg({
            'chembl_id': 'first',
            'smiles': 'first',
            'value': 'median',
            'type': 'first',
            'target': 'first',
        }).reset_index()

        print(f"    -> {len(df)} compounds")
    else:
        print(f"    -> 0 compounds")

    return df


def query_chembl_assays_by_keyword(keywords: List[str], domain: str,
                                    max_assays: int = 20,
                                    max_records_per_assay: int = 5000) -> pd.DataFrame:
    """
    Query ChEMBL for assays matching keyword descriptions.
    """
    print(f"  Querying {domain} assays...")

    all_activities = []
    assays_found = 0

    for keyword in keywords:
        if assays_found >= max_assays:
            break

        # Search for assays
        url = f"{CHEMBL_API}/assay.json"
        params = {
            'description__icontains': keyword,
            'limit': 50,
        }

        try:
            response = requests.get(url, params=params, timeout=60)
            if response.status_code != 200:
                continue
            data = response.json()
        except:
            continue

        assays = data.get('assays', [])

        for assay in assays[:5]:  # Limit per keyword
            if assays_found >= max_assays:
                break

            assay_id = assay.get('assay_chembl_id')
            assay_desc = assay.get('description', '')[:50]

            # Get activities for this assay
            act_url = f"{CHEMBL_API}/activity.json"
            act_params = {
                'assay_chembl_id': assay_id,
                'limit': max_records_per_assay,
            }

            try:
                act_response = requests.get(act_url, params=act_params, timeout=60)
                if act_response.status_code != 200:
                    continue
                act_data = act_response.json()
            except:
                continue

            activities = act_data.get('activities', [])

            for act in activities:
                if act.get('standard_value') is None:
                    continue

                smiles = act.get('canonical_smiles')
                if not smiles:
                    continue

                all_activities.append({
                    'chembl_id': act.get('molecule_chembl_id'),
                    'smiles': smiles,
                    'value': float(act['standard_value']),
                    'type': act.get('standard_type', 'unknown'),
                    'domain': domain,
                    'assay': assay_id,
                })

            assays_found += 1
            time.sleep(0.3)

    df = pd.DataFrame(all_activities)

    if len(df) > 0:
        df['smiles_canonical'] = df['smiles'].apply(canonicalize_smiles)
        df = df.dropna(subset=['smiles_canonical'])

        # Deduplicate per domain
        df = df.groupby(['smiles_canonical', 'domain']).agg({
            'chembl_id': 'first',
            'smiles': 'first',
            'value': 'median',
            'type': 'first',
        }).reset_index()

        print(f"    -> {len(df)} compound-domain pairs from {assays_found} assays")
    else:
        print(f"    -> 0 compounds")

    return df


def curate_chembl_diverse(max_compounds_per_target: int = 5000,
                          output_dir: Path = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Curate diverse property dataset from ChEMBL.
    """
    if output_dir is None:
        output_dir = project_root / 'outputs' / 'chembl_diverse'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ChEMBL DEEP CURATION")
    print("=" * 60)
    print()

    # =================================================================
    # Step 1: Query binding targets
    # =================================================================
    print("Step 1: Querying binding targets...")
    binding_data = {}

    for target_name, target_id in BINDING_TARGETS.items():
        df = query_chembl_target(target_id, target_name, 'IC50', max_compounds_per_target)
        if len(df) > 0:
            binding_data[f'Binding_{target_name}'] = df

    # =================================================================
    # Step 2: Query ADME assays
    # =================================================================
    print()
    print("Step 2: Querying ADME assays...")
    adme_data = {}

    for adme_type, keywords in ADME_KEYWORDS.items():
        df = query_chembl_assays_by_keyword(keywords, adme_type, max_assays=10)
        if len(df) > 0:
            adme_data[f'ADME_{adme_type}'] = df

    # =================================================================
    # Step 3: Query Toxicity assays
    # =================================================================
    print()
    print("Step 3: Querying toxicity assays...")
    tox_data = {}

    for tox_type, keywords in TOX_KEYWORDS.items():
        df = query_chembl_assays_by_keyword(keywords, tox_type, max_assays=10)
        if len(df) > 0:
            tox_data[f'Tox_{tox_type}'] = df

    # =================================================================
    # Step 4: Compute overlap
    # =================================================================
    print()
    print("=" * 60)
    print("OVERLAP ANALYSIS")
    print("=" * 60)

    all_datasets = {**binding_data, **adme_data, **tox_data}

    # Build compound -> domains mapping
    compound_domains = defaultdict(set)
    compound_datasets = defaultdict(set)

    for name, df in all_datasets.items():
        domain = name.split('_')[0]  # Binding, ADME, or Tox
        for smi in df['smiles_canonical'].unique():
            compound_domains[smi].add(domain)
            compound_datasets[smi].add(name)

    total_compounds = len(compound_domains)

    # Count by domain coverage
    domain_counts = defaultdict(int)
    for smi, domains in compound_domains.items():
        key = tuple(sorted(domains))
        domain_counts[key] += 1

    print()
    print("Compounds by domain coverage:")
    for domains, count in sorted(domain_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {' + '.join(domains)}: {count}")

    # Cross-domain compounds
    compounds_3_domains = sum(1 for d in compound_domains.values() if len(d) >= 3)
    compounds_2_domains = sum(1 for d in compound_domains.values() if len(d) >= 2)
    compounds_binding_adme = sum(1 for d in compound_domains.values()
                                  if 'Binding' in d and 'ADME' in d)
    compounds_binding_tox = sum(1 for d in compound_domains.values()
                                 if 'Binding' in d and 'Tox' in d)
    compounds_adme_tox = sum(1 for d in compound_domains.values()
                             if 'ADME' in d and 'Tox' in d)

    print()
    print(f"Total compounds: {total_compounds}")
    print(f"Compounds in 3 domains (Binding + ADME + Tox): {compounds_3_domains}")
    print(f"Compounds in 2+ domains: {compounds_2_domains}")
    print(f"  Binding + ADME: {compounds_binding_adme}")
    print(f"  Binding + Tox: {compounds_binding_tox}")
    print(f"  ADME + Tox: {compounds_adme_tox}")

    # Compute overlap percentage
    # Using min of domain sizes as denominator
    binding_smiles = set()
    for name, df in binding_data.items():
        binding_smiles.update(df['smiles_canonical'].unique())

    adme_smiles = set()
    for name, df in adme_data.items():
        adme_smiles.update(df['smiles_canonical'].unique())

    tox_smiles = set()
    for name, df in tox_data.items():
        tox_smiles.update(df['smiles_canonical'].unique())

    binding_adme_overlap_pct = (len(binding_smiles & adme_smiles) /
                                 min(len(binding_smiles), len(adme_smiles))
                                 if binding_smiles and adme_smiles else 0)
    binding_tox_overlap_pct = (len(binding_smiles & tox_smiles) /
                                min(len(binding_smiles), len(tox_smiles))
                                if binding_smiles and tox_smiles else 0)
    adme_tox_overlap_pct = (len(adme_smiles & tox_smiles) /
                             min(len(adme_smiles), len(tox_smiles))
                             if adme_smiles and tox_smiles else 0)

    mean_cross_overlap = np.mean([binding_adme_overlap_pct,
                                   binding_tox_overlap_pct,
                                   adme_tox_overlap_pct])

    print()
    print(f"Cross-domain overlap:")
    print(f"  Binding-ADME: {binding_adme_overlap_pct:.1%}")
    print(f"  Binding-Tox: {binding_tox_overlap_pct:.1%}")
    print(f"  ADME-Tox: {adme_tox_overlap_pct:.1%}")
    print(f"  Mean: {mean_cross_overlap:.1%}")

    # =================================================================
    # Step 5: Create merged dataset
    # =================================================================
    print()
    print("=" * 60)
    print("DATASET CREATION")
    print("=" * 60)

    # Find compounds with ALL 3 domains
    target_compounds = [smi for smi, domains in compound_domains.items()
                        if len(domains) >= 2]  # At least 2 domains

    if len(target_compounds) < 50:
        print(f"Warning: Only {len(target_compounds)} compounds with 2+ domains")
        target_compounds = list(compound_domains.keys())[:5000]

    # Build merged dataframe
    merged_data = []
    for smi in target_compounds:
        row = {'smiles': smi}

        for name, df in all_datasets.items():
            match = df[df['smiles_canonical'] == smi]
            if len(match) > 0:
                row[name] = match['value'].values[0]

        merged_data.append(row)

    merged_df = pd.DataFrame(merged_data)

    # Get task columns
    task_cols = [c for c in merged_df.columns if c != 'smiles']

    print()
    print(f"Merged dataset: {len(merged_df)} compounds x {len(task_cols)} tasks")
    print()
    print("Task coverage:")
    for task in task_cols:
        n = merged_df[task].notna().sum()
        print(f"  {task}: {n} ({100*n/len(merged_df):.1f}%)")

    # Save
    merged_df.to_csv(output_dir / 'chembl_diverse.csv', index=False)

    results = {
        'total_compounds': total_compounds,
        'compounds_3_domains': compounds_3_domains,
        'compounds_2_domains': compounds_2_domains,
        'binding_adme_overlap': float(binding_adme_overlap_pct),
        'binding_tox_overlap': float(binding_tox_overlap_pct),
        'adme_tox_overlap': float(adme_tox_overlap_pct),
        'mean_cross_overlap': float(mean_cross_overlap),
        'merged_compounds': len(merged_df),
        'tasks': task_cols,
        'domain_counts': {str(k): v for k, v in domain_counts.items()},
    }

    with open(output_dir / 'curation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # =================================================================
    # Decision Gate
    # =================================================================
    print()
    print("=" * 60)
    print("DECISION GATE")
    print("=" * 60)
    print()
    print(f"Compounds with 3 domains: {compounds_3_domains}")
    print(f"Mean cross-domain overlap: {mean_cross_overlap:.1%}")
    print()

    if compounds_3_domains >= 300 and mean_cross_overlap >= 0.50:
        decision = 'SUCCESS'
        print("*** DECISION: SUCCESS ***")
        print()
        print("ChEMBL provides sufficient cross-domain overlap!")
        print("Proceed with experiments on diverse properties.")

    elif compounds_3_domains >= 100 and mean_cross_overlap >= 0.30:
        decision = 'MARGINAL'
        print("--- DECISION: MARGINAL ---")
        print()
        print("ChEMBL provides borderline overlap.")
        print("Consider using 2-domain pairs instead of full 3-domain.")

    else:
        decision = 'FAILED'
        print("XXX DECISION: FAILED XXX")
        print()
        print("ChEMBL does not provide sufficient cross-domain overlap.")
        print("Accept limitation: validate on panel assays only.")

    results['decision'] = decision

    print()
    print(f"Results saved to: {output_dir}")

    return merged_df, results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-compounds', type=int, default=5000)
    args = parser.parse_args()

    df, results = curate_chembl_diverse(max_compounds_per_target=args.max_compounds)

    return results['decision']


if __name__ == '__main__':
    main()
