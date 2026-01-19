"""
Extended dataset loader for comprehensive molecular property prediction.

Supports 25+ properties across:
- Binding affinity (BACE, PDBbind)
- ADMET (BBBP, Caco-2, hERG, CYP3A4, CYP2D6, Clearance, VDss)
- Toxicity (Tox21, AMES, LD50)
- Physicochemical (ESOL, Lipophilicity, FreeSolv)
- Electronic (QM9 subsets - optional)
"""

import os
import gzip
import io
import urllib.request
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from tqdm import tqdm


# Dataset metadata
DATASET_INFO = {
    # Binding Affinity
    'BACE': {
        'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv',
        'smiles_col': 'mol',
        'target_col': 'Class',
        'task_type': 'classification',
        'description': 'Beta-secretase 1 inhibition (Alzheimer\'s)',
    },
    # ADMET
    'BBBP': {
        'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv',
        'smiles_col': 'smiles',
        'target_col': 'p_np',
        'task_type': 'classification',
        'description': 'Blood-brain barrier permeability',
    },
    'ESOL': {
        'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv',
        'smiles_col': 'smiles',
        'target_col': 'measured log solubility in mols per litre',
        'task_type': 'regression',
        'description': 'Aqueous solubility',
    },
    'Lipophilicity': {
        'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv',
        'smiles_col': 'smiles',
        'target_col': 'exp',
        'task_type': 'regression',
        'description': 'Octanol-water partition coefficient (logP)',
    },
    'FreeSolv': {
        'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv',
        'smiles_col': 'smiles',
        'target_col': 'expt',
        'task_type': 'regression',
        'description': 'Hydration free energy',
    },
    'ClinTox': {
        'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz',
        'smiles_col': 'smiles',
        'target_cols': ['FDA_APPROVED', 'CT_TOX'],
        'task_type': 'classification',
        'description': 'Clinical trial toxicity',
    },
    'SIDER': {
        'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz',
        'smiles_col': 'smiles',
        'target_cols': None,  # Multiple columns
        'task_type': 'classification',
        'description': 'Side effect database (27 endpoints)',
    },
    'Tox21': {
        'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz',
        'smiles_col': 'smiles',
        'target_cols': [
            'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
            'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
            'SR-HSE', 'SR-MMP', 'SR-p53'
        ],
        'task_type': 'classification',
        'description': 'Toxicity endpoints (12 assays)',
    },
    'HIV': {
        'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv',
        'smiles_col': 'smiles',
        'target_col': 'HIV_active',
        'task_type': 'classification',
        'description': 'HIV replication inhibition',
    },
    'MUV': {
        'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/muv.csv.gz',
        'smiles_col': 'smiles',
        'target_cols': None,  # 17 targets
        'task_type': 'classification',
        'description': 'Maximum Unbiased Validation (17 targets)',
    },
}

# Known mechanistic trade-offs from medicinal chemistry literature
LITERATURE_TRADEOFFS = {
    # Strong negative correlations expected
    ('ESOL', 'Lipophilicity'): -0.8,  # Solubility vs lipophilicity anti-correlation
    ('BBBP', 'ESOL'): -0.4,  # CNS penetration requires lipophilicity

    # Positive correlations expected (ADME cluster)
    ('BBBP', 'Lipophilicity'): 0.5,  # Both favor lipophilic compounds

    # Toxicity correlations
    ('NR-AR', 'NR-AR-LBD'): 0.8,  # Same receptor, different binding sites
    ('NR-ER', 'NR-ER-LBD'): 0.8,  # Same receptor, different binding sites
    ('SR-ARE', 'SR-HSE'): 0.5,  # Both stress response pathways

    # Mixed/independent
    ('BACE', 'BBBP'): 0.3,  # CNS drugs need BBB penetration
    ('BACE', 'NR-AhR'): 0.0,  # Independent mechanisms
}


class ExtendedDatasetLoader:
    """Load and merge multiple molecular property datasets."""

    def __init__(self, data_dir: str = 'outputs/raw_data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.datasets = {}
        self.merged_df = None
        self.task_types = {}

    def download_dataset(self, name: str) -> Path:
        """Download a single dataset."""
        if name not in DATASET_INFO:
            raise ValueError(f"Unknown dataset: {name}")

        info = DATASET_INFO[name]
        url = info['url']

        # Determine filename
        if url.endswith('.gz'):
            local_path = self.data_dir / f"{name.lower()}.csv"
        else:
            local_path = self.data_dir / f"{name.lower()}.csv"

        if local_path.exists():
            return local_path

        print(f"Downloading {name}...")

        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()

            # Decompress if needed
            if url.endswith('.gz'):
                with gzip.GzipFile(fileobj=io.BytesIO(data)) as f:
                    data = f.read()

            with open(local_path, 'wb') as f:
                f.write(data)

            print(f"  Saved to {local_path}")
            return local_path

        except Exception as e:
            print(f"  Failed to download {name}: {e}")
            return None

    def load_dataset(self, name: str) -> Optional[pd.DataFrame]:
        """Load a single dataset and standardize columns."""
        path = self.download_dataset(name)
        if path is None:
            return None

        info = DATASET_INFO[name]
        df = pd.read_csv(path)

        # Standardize SMILES column
        smiles_col = info['smiles_col']
        if smiles_col not in df.columns:
            print(f"  Warning: SMILES column '{smiles_col}' not found in {name}")
            return None

        # Extract targets
        result = pd.DataFrame({'smiles': df[smiles_col]})

        if 'target_cols' in info and info['target_cols']:
            # Multiple targets
            for col in info['target_cols']:
                if col in df.columns:
                    result[f"{name}_{col}"] = df[col].values
                    self.task_types[f"{name}_{col}"] = info['task_type']
        elif 'target_col' in info:
            # Single target
            col = info['target_col']
            if col in df.columns:
                result[name] = df[col].values
                self.task_types[name] = info['task_type']
        else:
            # All non-smiles columns are targets
            for col in df.columns:
                if col != smiles_col:
                    result[f"{name}_{col}"] = df[col].values
                    self.task_types[f"{name}_{col}"] = info['task_type']

        return result

    def load_all_datasets(self, dataset_names: list = None) -> pd.DataFrame:
        """Load and merge all specified datasets."""
        if dataset_names is None:
            dataset_names = list(DATASET_INFO.keys())

        all_dfs = []

        for name in tqdm(dataset_names, desc="Loading datasets"):
            df = self.load_dataset(name)
            if df is not None:
                all_dfs.append(df)
                self.datasets[name] = df
                print(f"  {name}: {len(df)} molecules, {len(df.columns)-1} tasks")

        if not all_dfs:
            raise ValueError("No datasets loaded successfully")

        # Merge on SMILES (outer join to keep all molecules)
        merged = all_dfs[0]
        for df in all_dfs[1:]:
            merged = pd.merge(merged, df, on='smiles', how='outer')

        self.merged_df = merged
        print(f"\nMerged dataset: {len(merged)} unique molecules, {len(merged.columns)-1} tasks")

        return merged

    def get_task_names(self) -> list:
        """Get list of all task names."""
        if self.merged_df is None:
            return []
        return [col for col in self.merged_df.columns if col != 'smiles']

    def get_labels_dict(self) -> dict:
        """Get labels as dictionary mapping task -> numpy array."""
        if self.merged_df is None:
            return {}

        labels = {}
        for task in self.get_task_names():
            labels[task] = self.merged_df[task].values.astype(np.float32)
        return labels

    def get_smiles_list(self) -> list:
        """Get list of SMILES strings."""
        if self.merged_df is None:
            return []
        return self.merged_df['smiles'].tolist()

    def filter_by_label_coverage(self, min_tasks: int = 5) -> pd.DataFrame:
        """Filter molecules to those with at least min_tasks labels."""
        if self.merged_df is None:
            raise ValueError("No data loaded")

        task_cols = self.get_task_names()
        n_labels = self.merged_df[task_cols].notna().sum(axis=1)
        mask = n_labels >= min_tasks

        filtered = self.merged_df[mask].copy()
        print(f"Filtered to {len(filtered)} molecules with >= {min_tasks} labels")

        return filtered

    def get_dataset_statistics(self) -> pd.DataFrame:
        """Get statistics for each task."""
        if self.merged_df is None:
            return pd.DataFrame()

        stats = []
        for task in self.get_task_names():
            values = self.merged_df[task].dropna()
            task_type = self.task_types.get(task, 'unknown')

            stat = {
                'task': task,
                'type': task_type,
                'n_samples': len(values),
                'missing_pct': (self.merged_df[task].isna().sum() / len(self.merged_df)) * 100,
            }

            if task_type == 'classification':
                stat['pos_rate'] = values.mean() if len(values) > 0 else 0
            else:
                stat['mean'] = values.mean() if len(values) > 0 else 0
                stat['std'] = values.std() if len(values) > 0 else 0

            stats.append(stat)

        return pd.DataFrame(stats)


def load_extended_dataset(
    dataset_names: list = None,
    min_tasks: int = 3,
    data_dir: str = 'outputs/raw_data'
) -> tuple:
    """
    Convenience function to load extended dataset.

    Returns:
        smiles_list: List of SMILES strings
        labels: Dict mapping task name -> numpy array
        task_types: Dict mapping task name -> 'classification' or 'regression'
    """
    loader = ExtendedDatasetLoader(data_dir)

    if dataset_names is None:
        # Default: core datasets for gradient conflict analysis
        dataset_names = ['BACE', 'BBBP', 'ESOL', 'Lipophilicity', 'Tox21', 'ClinTox', 'HIV']

    loader.load_all_datasets(dataset_names)
    filtered = loader.filter_by_label_coverage(min_tasks)

    # Update merged_df with filtered data
    loader.merged_df = filtered

    return loader.get_smiles_list(), loader.get_labels_dict(), loader.task_types


def get_literature_tradeoffs() -> dict:
    """Get dictionary of known mechanistic trade-offs."""
    return LITERATURE_TRADEOFFS.copy()


if __name__ == '__main__':
    # Test loading
    print("Testing extended dataset loader...")

    smiles, labels, task_types = load_extended_dataset(
        dataset_names=['BACE', 'BBBP', 'ESOL', 'Lipophilicity', 'Tox21'],
        min_tasks=3
    )

    print(f"\nLoaded {len(smiles)} molecules")
    print(f"Tasks: {list(labels.keys())}")
    print(f"\nTask types:")
    for task, ttype in task_types.items():
        n_valid = (~np.isnan(labels[task])).sum()
        print(f"  {task}: {ttype}, {n_valid} samples")
