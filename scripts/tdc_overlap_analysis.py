#!/usr/bin/env python3
"""
TDC Overlap Analysis - Phase 1 Decision Gate.

Analyzes compound overlap across ALL TDC datasets to determine
if sufficient overlap exists for diverse property validation.

Decision Gate Criteria:
- compounds_with_5plus_properties >= 500 AND mean_overlap >= 0.60 → SUCCESS
- compounds_with_5plus_properties >= 300 AND mean_overlap >= 0.70 → MARGINAL
- Otherwise → FAILED (pivot to ChEMBL or accept limitation)

Usage:
    python scripts/tdc_overlap_analysis.py
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


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


def load_tdc_datasets() -> Dict[str, pd.DataFrame]:
    """
    Load ALL relevant TDC datasets.

    Returns:
        Dictionary of dataset_name -> DataFrame with 'smiles' and 'value' columns
    """
    try:
        from tdc.single_pred import ADME, Tox
    except ImportError:
        print("ERROR: TDC not installed")
        print("\nTo install TDC, create a clean conda environment:")
        print("  conda create -n tdc python=3.9")
        print("  conda activate tdc")
        print("  pip install PyTDC")
        print("  pip install rdkit torch torch-geometric")
        return {}

    datasets = {}

    # ADME datasets (regression tasks)
    adme_datasets = [
        ('ADME_Lipophilicity', 'Lipophilicity_AstraZeneca'),
        ('ADME_Solubility', 'Solubility_AqSolDB'),
        ('ADME_Caco2', 'Caco2_Wang'),
        ('ADME_HIA', 'HIA_Hou'),
        ('ADME_Bioavailability', 'Bioavailability_Ma'),
        ('ADME_PPBR', 'PPBR_AZ'),
        ('ADME_VDss', 'VDss_Lombardo'),
        ('ADME_CYP2D6_Inhibitor', 'CYP2D6_Veith'),
        ('ADME_CYP3A4_Inhibitor', 'CYP3A4_Veith'),
        ('ADME_CYP2C9_Inhibitor', 'CYP2C9_Veith'),
        ('ADME_CYP2D6_Substrate', 'CYP2D6_Substrate_CarbonMangels'),
        ('ADME_CYP3A4_Substrate', 'CYP3A4_Substrate_CarbonMangels'),
        ('ADME_CYP2C9_Substrate', 'CYP2C9_Substrate_CarbonMangels'),
        ('ADME_HalfLife', 'Half_Life_Obach'),
        ('ADME_Clearance_Hepatocyte', 'Clearance_Hepatocyte_AZ'),
        ('ADME_Clearance_Microsome', 'Clearance_Microsome_AZ'),
    ]

    # Toxicity datasets (classification tasks)
    tox_datasets = [
        ('Tox_hERG', 'hERG'),
        ('Tox_AMES', 'AMES'),
        ('Tox_DILI', 'DILI'),
        ('Tox_LD50', 'LD50_Zhu'),
        ('Tox_Carcinogens', 'Carcinogens_Lagunin'),
        ('Tox_ClinTox', 'ClinTox'),
        ('Tox_SkinReaction', 'Skin_Reaction'),
    ]

    print("Loading TDC datasets...")
    print()

    # Load ADME
    print("ADME Datasets:")
    for name, tdc_name in adme_datasets:
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

            datasets[name] = df
            print(f"  [OK] {name}: {len(df)} compounds")

        except Exception as e:
            print(f"  [FAIL] {name}: {str(e)[:50]}")

    print()
    print("Toxicity Datasets:")
    for name, tdc_name in tox_datasets:
        try:
            data = Tox(name=tdc_name)
            df = data.get_data()

            if 'Drug' in df.columns:
                df = df.rename(columns={'Drug': 'smiles'})
            if 'Y' in df.columns:
                df = df.rename(columns={'Y': 'value'})

            df['smiles_canonical'] = df['smiles'].apply(canonicalize_smiles)
            df = df.dropna(subset=['smiles_canonical'])
            df = df.drop_duplicates(subset=['smiles_canonical'])

            datasets[name] = df
            print(f"  [OK] {name}: {len(df)} compounds")

        except Exception as e:
            print(f"  [FAIL] {name}: {str(e)[:50]}")

    return datasets


def compute_overlap_analysis(datasets: Dict[str, pd.DataFrame]) -> Dict:
    """
    Compute comprehensive overlap analysis.

    Returns:
        Dictionary with overlap statistics
    """
    if not datasets:
        return {}

    print()
    print("=" * 60)
    print("OVERLAP ANALYSIS")
    print("=" * 60)

    # Build compound -> datasets mapping
    compound_datasets = defaultdict(set)

    for name, df in datasets.items():
        for smi in df['smiles_canonical'].unique():
            compound_datasets[smi].add(name)

    total_compounds = len(compound_datasets)

    # Count compounds by number of datasets
    counts_by_n = defaultdict(int)
    for smi, ds_set in compound_datasets.items():
        counts_by_n[len(ds_set)] += 1

    print()
    print("Compounds by number of datasets:")
    cumulative = 0
    for n in sorted(counts_by_n.keys(), reverse=True):
        cumulative += counts_by_n[n]
        print(f"  {n}+ datasets: {cumulative} compounds")

    # Key metrics
    compounds_5plus = sum(counts_by_n[n] for n in counts_by_n if n >= 5)
    compounds_4plus = sum(counts_by_n[n] for n in counts_by_n if n >= 4)
    compounds_3plus = sum(counts_by_n[n] for n in counts_by_n if n >= 3)

    # Pairwise overlap matrix
    dataset_names = list(datasets.keys())
    n_datasets = len(dataset_names)
    overlap_matrix = np.zeros((n_datasets, n_datasets))

    for i, name_i in enumerate(dataset_names):
        smiles_i = set(datasets[name_i]['smiles_canonical'].unique())
        for j, name_j in enumerate(dataset_names):
            smiles_j = set(datasets[name_j]['smiles_canonical'].unique())
            overlap = len(smiles_i & smiles_j)
            min_size = min(len(smiles_i), len(smiles_j))
            overlap_matrix[i, j] = overlap / max(min_size, 1)

    # Mean pairwise overlap (excluding diagonal)
    mask = ~np.eye(n_datasets, dtype=bool)
    mean_overlap = overlap_matrix[mask].mean()

    print()
    print(f"Mean pairwise overlap: {mean_overlap:.2%}")

    # Cross-domain analysis
    adme_names = [n for n in dataset_names if n.startswith('ADME_')]
    tox_names = [n for n in dataset_names if n.startswith('Tox_')]

    cross_overlaps = []
    for adme in adme_names:
        for tox in tox_names:
            i, j = dataset_names.index(adme), dataset_names.index(tox)
            cross_overlaps.append(overlap_matrix[i, j])

    mean_cross_overlap = np.mean(cross_overlaps) if cross_overlaps else 0

    print(f"Mean ADME-Tox overlap: {mean_cross_overlap:.2%}")

    # Find best overlapping pairs across domains
    print()
    print("Top cross-domain overlaps:")
    cross_pairs = []
    for adme in adme_names:
        for tox in tox_names:
            smiles_adme = set(datasets[adme]['smiles_canonical'].unique())
            smiles_tox = set(datasets[tox]['smiles_canonical'].unique())
            overlap = len(smiles_adme & smiles_tox)
            overlap_pct = overlap / min(len(smiles_adme), len(smiles_tox))
            cross_pairs.append((adme, tox, overlap, overlap_pct))

    cross_pairs.sort(key=lambda x: x[2], reverse=True)
    for adme, tox, n, pct in cross_pairs[:10]:
        print(f"  {adme} & {tox}: {n} ({pct:.1%})")

    # Identify best subset for experiments
    print()
    print("=" * 60)
    print("BEST SUBSET IDENTIFICATION")
    print("=" * 60)

    # Find datasets with most cross-domain overlap
    dataset_cross_overlap = {}
    for name in dataset_names:
        smiles_set = set(datasets[name]['smiles_canonical'].unique())

        if name.startswith('ADME_'):
            # Count overlap with all Tox datasets
            total_overlap = 0
            for tox_name in tox_names:
                tox_smiles = set(datasets[tox_name]['smiles_canonical'].unique())
                total_overlap += len(smiles_set & tox_smiles)
            dataset_cross_overlap[name] = total_overlap / len(tox_names)
        else:
            # Count overlap with all ADME datasets
            total_overlap = 0
            for adme_name in adme_names:
                adme_smiles = set(datasets[adme_name]['smiles_canonical'].unique())
                total_overlap += len(smiles_set & adme_smiles)
            dataset_cross_overlap[name] = total_overlap / len(adme_names)

    # Top ADME and Tox datasets by cross-overlap
    adme_ranked = sorted([(n, dataset_cross_overlap[n]) for n in adme_names],
                         key=lambda x: x[1], reverse=True)
    tox_ranked = sorted([(n, dataset_cross_overlap[n]) for n in tox_names],
                        key=lambda x: x[1], reverse=True)

    print()
    print("Best ADME datasets (by Tox overlap):")
    for name, score in adme_ranked[:5]:
        print(f"  {name}: avg overlap = {score:.0f}")

    print()
    print("Best Tox datasets (by ADME overlap):")
    for name, score in tox_ranked[:5]:
        print(f"  {name}: avg overlap = {score:.0f}")

    # Recommend best subset
    best_adme = [n for n, _ in adme_ranked[:3]]
    best_tox = [n for n, _ in tox_ranked[:3]]
    best_subset = best_adme + best_tox

    # Compute overlap for best subset
    subset_compounds = None
    for name in best_subset:
        smiles_set = set(datasets[name]['smiles_canonical'].unique())
        if subset_compounds is None:
            subset_compounds = smiles_set
        else:
            subset_compounds = subset_compounds & smiles_set

    print()
    print(f"Recommended subset: {best_subset}")
    print(f"Compounds with ALL {len(best_subset)} properties: {len(subset_compounds)}")

    results = {
        'total_datasets': len(datasets),
        'total_compounds': total_compounds,
        'compounds_5plus': compounds_5plus,
        'compounds_4plus': compounds_4plus,
        'compounds_3plus': compounds_3plus,
        'mean_pairwise_overlap': float(mean_overlap),
        'mean_cross_domain_overlap': float(mean_cross_overlap),
        'best_subset': best_subset,
        'best_subset_overlap': len(subset_compounds),
        'top_cross_pairs': [(a, t, n, float(p)) for a, t, n, p in cross_pairs[:20]],
        'dataset_names': dataset_names,
    }

    return results


def decision_gate(results: Dict) -> str:
    """
    Apply decision gate criteria.

    Returns:
        'SUCCESS', 'MARGINAL', or 'FAILED'
    """
    if not results:
        return 'FAILED'

    n_5plus = results.get('compounds_5plus', 0)
    mean_overlap = results.get('mean_pairwise_overlap', 0)
    cross_overlap = results.get('mean_cross_domain_overlap', 0)

    print()
    print("=" * 60)
    print("DECISION GATE")
    print("=" * 60)
    print()
    print(f"Compounds with 5+ properties: {n_5plus}")
    print(f"Mean pairwise overlap: {mean_overlap:.2%}")
    print(f"Mean ADME-Tox overlap: {cross_overlap:.2%}")
    print()

    # Criteria
    if n_5plus >= 500 and mean_overlap >= 0.60:
        decision = 'SUCCESS'
        print("*** DECISION: SUCCESS ***")
        print()
        print("TDC provides sufficient overlap for diverse property validation.")
        print("Proceed with experiments:")
        print("  1. Train GNN MTL on diverse TDC properties")
        print("  2. Compute gradient conflicts")
        print("  3. Validate with empirical correlations")
        print("  4. Look for cross-domain trade-offs")

    elif n_5plus >= 300 and mean_overlap >= 0.70:
        decision = 'MARGINAL'
        print("--- DECISION: MARGINAL ---")
        print()
        print("TDC provides borderline overlap. Small dataset but acceptable quality.")
        print("Options:")
        print("  1. Proceed with caution (small N)")
        print("  2. Focus on best-overlapping subset only")
        print("  3. Consider ChEMBL augmentation")

    else:
        decision = 'FAILED'
        print("XXX DECISION: FAILED XXX")
        print()
        print("TDC does not provide sufficient overlap for diverse validation.")
        print("Options:")
        print("  1. ChEMBL deep curation (5 days effort)")
        print("  2. Accept limitation - validate on panel assays only")
        print("  3. Frame paper as toxicity-focused (Tox21 + ToxCast)")

    return decision


def main():
    output_dir = project_root / 'outputs' / 'tdc_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TDC OVERLAP ANALYSIS - PHASE 1 DECISION GATE")
    print("=" * 60)
    print()

    # Load datasets
    datasets = load_tdc_datasets()

    if not datasets:
        print()
        print("Cannot proceed without TDC.")
        print("Please install TDC in a clean environment.")
        return

    # Compute overlap
    results = compute_overlap_analysis(datasets)

    # Decision gate
    decision = decision_gate(results)
    results['decision'] = decision

    # Save results
    with open(output_dir / 'overlap_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Results saved to: {output_dir / 'overlap_analysis.json'}")

    return decision


if __name__ == '__main__':
    main()
