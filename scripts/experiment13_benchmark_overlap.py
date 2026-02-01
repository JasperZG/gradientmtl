#!/usr/bin/env python3
"""
Experiment 13: Benchmark Dataset Overlap Measurement

Computes exact pairwise compound overlap for standard benchmarks:
- MoleculeNet (ADME tasks)
- TDC ADMET datasets
- Tox21 / ToxCast (for comparison)

Validates the claim that standard benchmarks have insufficient overlap
for gradient-based analysis.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import json


def canonicalize_smiles(smi):
    """Canonicalize SMILES for matching."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            return Chem.MolToSmiles(mol)
    except:
        pass
    return None


def load_moleculenet_datasets():
    """Load MoleculeNet datasets and return compound sets."""
    datasets = {}

    # Try loading from MoleculeNet via DeepChem or direct files
    try:
        from tdc.single_pred import ADME, Tox
        print("Loading via TDC...")

        adme_datasets = {
            'Caco2': ('Caco2_Wang', ADME),
            'HIA': ('HIA_Hou', ADME),
            'Pgp': ('Pgp_Broccatelli', ADME),
            'Bioavailability': ('Bioavailability_Ma', ADME),
            'Solubility': ('Solubility_AqSolDB', ADME),
            'Lipophilicity': ('Lipophilicity_AstraZeneca', ADME),
            'PPBR': ('PPBR_AZ', ADME),
            'VDss': ('VDss_Lombardo', ADME),
            'BBB': ('BBB_Martins', ADME),
            'CYP2C9': ('CYP2C9_Veith', ADME),
            'CYP2D6': ('CYP2D6_Veith', ADME),
            'CYP3A4': ('CYP3A4_Veith', ADME),
            'CYP2C19': ('CYP2C19_Veith', ADME),
            'CYP1A2': ('CYP1A2_Veith', ADME),
            'HLM': ('Clearance_Hepatocyte_AZ', ADME),
            'MLM': ('Clearance_Microsome_AZ', ADME),
            'Half_Life': ('Half_Life_Obach', ADME),
        }

        tox_datasets = {
            'hERG': ('hERG', Tox),
            'AMES': ('AMES', Tox),
            'DILI': ('DILI', Tox),
            'LD50': ('LD50_Zhu', Tox),
        }

        all_datasets = {**adme_datasets, **tox_datasets}

        for name, (tdc_name, cls) in all_datasets.items():
            try:
                data = cls(name=tdc_name)
                df = data.get_data()
                smiles_col = 'Drug' if 'Drug' in df.columns else 'smiles'
                compounds = set()
                for smi in df[smiles_col]:
                    canon = canonicalize_smiles(smi)
                    if canon:
                        compounds.add(canon)
                datasets[name] = compounds
                print(f"  {name}: {len(compounds)} compounds")
            except Exception as e:
                print(f"  {name}: FAILED ({e})")

    except ImportError:
        print("TDC not installed. Trying local files...")

        # Try local MoleculeNet files
        for csv_path in Path('data').glob('*.csv'):
            try:
                df = pd.read_csv(csv_path)
                smi_col = None
                for col in ['smiles', 'SMILES', 'Drug', 'mol']:
                    if col in df.columns:
                        smi_col = col
                        break
                if smi_col:
                    compounds = set()
                    for smi in df[smi_col].dropna():
                        canon = canonicalize_smiles(smi)
                        if canon:
                            compounds.add(canon)
                    datasets[csv_path.stem] = compounds
                    print(f"  {csv_path.stem}: {len(compounds)} compounds")
            except:
                continue

    return datasets


def compute_overlap_matrix(datasets):
    """Compute pairwise overlap between all datasets."""
    names = sorted(datasets.keys())
    n = len(names)
    overlap_counts = np.zeros((n, n), dtype=int)
    overlap_fracs = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            shared = datasets[names[i]] & datasets[names[j]]
            overlap_counts[i, j] = len(shared)
            min_size = min(len(datasets[names[i]]), len(datasets[names[j]]))
            overlap_fracs[i, j] = len(shared) / min_size if min_size > 0 else 0

    return names, overlap_counts, overlap_fracs


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='outputs/experiment13_benchmark_overlap')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Experiment 13: Benchmark Dataset Overlap Measurement")
    print("=" * 60)

    # Load datasets
    print("\nLoading benchmark datasets...")
    datasets = load_moleculenet_datasets()

    if len(datasets) < 2:
        print("ERROR: Not enough datasets loaded. Install TDC: pip install PyTDC")
        return

    # Compute overlap
    print(f"\nComputing pairwise overlap for {len(datasets)} datasets...")
    names, counts, fracs = compute_overlap_matrix(datasets)

    # Save overlap matrix
    overlap_df = pd.DataFrame(fracs, index=names, columns=names)
    overlap_df.to_csv(output_dir / 'overlap_matrix_fraction.csv')

    counts_df = pd.DataFrame(counts, index=names, columns=names)
    counts_df.to_csv(output_dir / 'overlap_matrix_counts.csv')

    # Compute statistics
    n = len(names)
    upper_fracs = fracs[np.triu_indices(n, 1)]
    upper_counts = counts[np.triu_indices(n, 1)]

    print(f"\n--- Overlap Statistics ---")
    print(f"Mean overlap fraction: {upper_fracs.mean():.3f}")
    print(f"Median overlap fraction: {np.median(upper_fracs):.3f}")
    print(f"Max overlap fraction: {upper_fracs.max():.3f}")
    print(f"Min overlap fraction: {upper_fracs.min():.3f}")
    print(f"Pairs with >30% overlap: {(upper_fracs > 0.30).sum()} / {len(upper_fracs)}")
    print(f"Pairs with >50% overlap: {(upper_fracs > 0.50).sum()} / {len(upper_fracs)}")

    # Categorize by ADME group
    cyp_names = [n for n in names if n.startswith('CYP')]
    adme_names = [n for n in names if n not in cyp_names and n not in ['hERG', 'AMES', 'DILI', 'LD50']]
    tox_names = [n for n in names if n in ['hERG', 'AMES', 'DILI', 'LD50']]

    print(f"\n--- By Category ---")
    print(f"CYP datasets: {cyp_names}")
    print(f"ADME datasets: {adme_names}")
    print(f"Tox datasets: {tox_names}")

    # CYP within-group overlap
    if len(cyp_names) >= 2:
        cyp_overlaps = []
        for i, n1 in enumerate(cyp_names):
            for n2 in cyp_names[i+1:]:
                idx1 = names.index(n1)
                idx2 = names.index(n2)
                cyp_overlaps.append(fracs[idx1, idx2])
        print(f"\nCYP within-group overlap: mean={np.mean(cyp_overlaps):.3f}")

    # Cross-category overlap
    cross_overlaps = []
    for n1 in adme_names:
        for n2 in tox_names:
            if n1 in names and n2 in names:
                idx1 = names.index(n1)
                idx2 = names.index(n2)
                cross_overlaps.append(fracs[idx1, idx2])
    if cross_overlaps:
        print(f"ADME-Tox cross overlap: mean={np.mean(cross_overlaps):.3f}")

    # Summary
    summary = {
        'n_datasets': len(names),
        'n_pairs': len(upper_fracs),
        'mean_overlap': round(float(upper_fracs.mean()), 4),
        'median_overlap': round(float(np.median(upper_fracs)), 4),
        'max_overlap': round(float(upper_fracs.max()), 4),
        'pairs_above_30pct': int((upper_fracs > 0.30).sum()),
        'pairs_above_50pct': int((upper_fracs > 0.50).sum()),
        'dataset_sizes': {name: len(datasets[name]) for name in names},
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {output_dir}/")


if __name__ == '__main__':
    main()
