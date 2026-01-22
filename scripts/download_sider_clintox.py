#!/usr/bin/env python3
"""
Download SIDER and ClinTox from alternative sources.

SIDER: 27 side effect categories on 1,427 drugs
ClinTox: 2 clinical trial toxicity endpoints on 1,491 drugs

These are ideal for gradient conflict analysis because:
1. Same compounds have ALL labels (no missing data)
2. Diverse outcomes (hepatotoxicity, cardiotoxicity, etc.)
"""

import os
import sys
import urllib.request
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent.parent
data_dir = project_root / 'outputs' / 'moleculenet_data'
data_dir.mkdir(parents=True, exist_ok=True)


def download_from_github():
    """Download from MoleculeNet GitHub mirror."""

    # Alternative URLs (GitHub raw, various mirrors)
    sources = {
        'sider': [
            'https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/sider.csv',
            'https://raw.githubusercontent.com/deepchem/deepchem/master/examples/tutorials/sider.csv',
        ],
        'clintox': [
            'https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/clintox.csv',
        ],
        'tox21': [
            'https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/tox21.csv',
        ],
    }

    for name, urls in sources.items():
        output_path = data_dir / f'{name}.csv'
        if output_path.exists():
            print(f"{name}: already exists")
            continue

        for url in urls:
            try:
                print(f"Trying {name} from {url[:50]}...")
                urllib.request.urlretrieve(url, output_path)

                # Verify it's valid CSV
                df = pd.read_csv(output_path)
                print(f"  Success: {len(df)} rows, {len(df.columns)} columns")
                break
            except Exception as e:
                print(f"  Failed: {e}")
                if output_path.exists():
                    output_path.unlink()
        else:
            print(f"  Could not download {name} from any source")


def try_deepchem_loader():
    """Try loading via DeepChem if installed."""
    try:
        import deepchem as dc

        print("\nUsing DeepChem to load datasets...")

        # SIDER
        try:
            print("Loading SIDER...")
            tasks, datasets, transformers = dc.molnet.load_sider()
            train, valid, test = datasets

            # Combine all splits
            all_X = list(train.ids) + list(valid.ids) + list(test.ids)
            all_y = list(train.y) + list(valid.y) + list(test.y)

            df = pd.DataFrame({'smiles': all_X})
            for i, task in enumerate(tasks):
                df[task] = [y[i] for y in all_y]

            df.to_csv(data_dir / 'sider.csv', index=False)
            print(f"  Saved SIDER: {len(df)} compounds, {len(tasks)} tasks")
        except Exception as e:
            print(f"  SIDER failed: {e}")

        # ClinTox
        try:
            print("Loading ClinTox...")
            tasks, datasets, transformers = dc.molnet.load_clintox()
            train, valid, test = datasets

            all_X = list(train.ids) + list(valid.ids) + list(test.ids)
            all_y = list(train.y) + list(valid.y) + list(test.y)

            df = pd.DataFrame({'smiles': all_X})
            for i, task in enumerate(tasks):
                df[task] = [y[i] for y in all_y]

            df.to_csv(data_dir / 'clintox.csv', index=False)
            print(f"  Saved ClinTox: {len(df)} compounds, {len(tasks)} tasks")
        except Exception as e:
            print(f"  ClinTox failed: {e}")

    except ImportError:
        print("\nDeepChem not installed. Install with: pip install deepchem")
        return False

    return True


def analyze_sider():
    """Analyze SIDER dataset structure."""
    sider_path = data_dir / 'sider.csv'
    if not sider_path.exists():
        print("SIDER not found")
        return

    df = pd.read_csv(sider_path)

    print("\n" + "=" * 60)
    print("SIDER Dataset Analysis")
    print("=" * 60)
    print(f"\nCompounds: {len(df)}")
    print(f"Tasks: {len(df.columns) - 1}")

    # Find SMILES column
    smiles_col = 'smiles' if 'smiles' in df.columns else df.columns[0]
    task_cols = [c for c in df.columns if c != smiles_col]

    print(f"\nSide effect categories:")
    for col in task_cols[:10]:
        n_pos = (df[col] == 1).sum()
        n_neg = (df[col] == 0).sum()
        n_missing = df[col].isna().sum()
        print(f"  {col}: {n_pos} positive, {n_neg} negative, {n_missing} missing")

    if len(task_cols) > 10:
        print(f"  ... and {len(task_cols) - 10} more tasks")

    # Check missing data
    missing_pct = df[task_cols].isna().sum().sum() / (len(df) * len(task_cols)) * 100
    print(f"\nMissing data: {missing_pct:.1f}%")


def main():
    print("=" * 60)
    print("Downloading SIDER/ClinTox/Tox21 Multi-Task Datasets")
    print("=" * 60)

    # Try direct download first
    print("\n1. Trying direct download...")
    download_from_github()

    # Check what we have
    for name in ['sider', 'clintox', 'tox21']:
        path = data_dir / f'{name}.csv'
        if path.exists():
            df = pd.read_csv(path)
            print(f"\n{name.upper()}: {len(df)} compounds, {len(df.columns)} columns")

    # Try DeepChem if direct failed
    if not (data_dir / 'sider.csv').exists():
        print("\n2. Trying DeepChem loader...")
        try_deepchem_loader()

    # Analyze SIDER
    analyze_sider()


if __name__ == '__main__':
    main()
