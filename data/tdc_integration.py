"""
Therapeutics Data Commons (TDC) integration for comprehensive ADMET property coverage.

TDC provides 25+ ADMET properties with standardized benchmarks:
- Absorption: Caco2, HIA, Pgp, Bioavailability
- Distribution: BBB, PPBR, VDss
- Metabolism: CYP2C9, CYP2D6, CYP3A4, CYP2C19, CYP1A2
- Excretion: Clearance, Half_Life
- Toxicity: hERG, AMES, DILI, LD50, Carcinogenicity

Install: pip install PyTDC
"""

import os
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd


# TDC dataset configurations
TDC_ADMET_DATASETS = {
    # Absorption
    'Caco2_Wang': {'group': 'Absorption', 'type': 'regression', 'description': 'Caco-2 cell permeability'},
    'HIA_Hou': {'group': 'Absorption', 'type': 'classification', 'description': 'Human intestinal absorption'},
    'Pgp_Broccatelli': {'group': 'Absorption', 'type': 'classification', 'description': 'P-glycoprotein inhibition'},
    'Bioavailability_Ma': {'group': 'Absorption', 'type': 'classification', 'description': 'Oral bioavailability'},
    'Solubility_AqSolDB': {'group': 'Absorption', 'type': 'regression', 'description': 'Aqueous solubility'},

    # Distribution
    'BBB_Martins': {'group': 'Distribution', 'type': 'classification', 'description': 'Blood-brain barrier penetration'},
    'PPBR_AZ': {'group': 'Distribution', 'type': 'regression', 'description': 'Plasma protein binding rate'},
    'VDss_Lombardo': {'group': 'Distribution', 'type': 'regression', 'description': 'Volume of distribution'},

    # Metabolism
    'CYP2C9_Veith': {'group': 'Metabolism', 'type': 'classification', 'description': 'CYP2C9 inhibition'},
    'CYP2D6_Veith': {'group': 'Metabolism', 'type': 'classification', 'description': 'CYP2D6 inhibition'},
    'CYP3A4_Veith': {'group': 'Metabolism', 'type': 'classification', 'description': 'CYP3A4 inhibition'},
    'CYP2C19_Veith': {'group': 'Metabolism', 'type': 'classification', 'description': 'CYP2C19 inhibition'},
    'CYP1A2_Veith': {'group': 'Metabolism', 'type': 'classification', 'description': 'CYP1A2 inhibition'},

    # Excretion
    'Clearance_Hepatocyte_AZ': {'group': 'Excretion', 'type': 'regression', 'description': 'Hepatocyte clearance'},
    'Clearance_Microsome_AZ': {'group': 'Excretion', 'type': 'regression', 'description': 'Microsomal clearance'},
    'Half_Life_Obach': {'group': 'Excretion', 'type': 'regression', 'description': 'Human half-life'},

    # Toxicity
    'hERG': {'group': 'Toxicity', 'type': 'classification', 'description': 'hERG channel inhibition (cardiotoxicity)'},
    'AMES': {'group': 'Toxicity', 'type': 'classification', 'description': 'AMES mutagenicity'},
    'DILI': {'group': 'Toxicity', 'type': 'classification', 'description': 'Drug-induced liver injury'},
    'LD50_Zhu': {'group': 'Toxicity', 'type': 'regression', 'description': 'Acute toxicity LD50'},
    'Carcinogens_Lagunin': {'group': 'Toxicity', 'type': 'classification', 'description': 'Carcinogenicity'},
    'ClinTox': {'group': 'Toxicity', 'type': 'classification', 'description': 'Clinical trial toxicity'},
    'Skin_Reaction': {'group': 'Toxicity', 'type': 'classification', 'description': 'Skin sensitization'},
}

# Known mechanistic relationships from TDC properties
TDC_MECHANISTIC_CLUSTERS = {
    'CYP_enzymes': ['CYP2C9_Veith', 'CYP2D6_Veith', 'CYP3A4_Veith', 'CYP2C19_Veith', 'CYP1A2_Veith'],
    'Clearance': ['Clearance_Hepatocyte_AZ', 'Clearance_Microsome_AZ', 'Half_Life_Obach'],
    'Permeability': ['Caco2_Wang', 'HIA_Hou', 'BBB_Martins', 'Bioavailability_Ma'],
    'Toxicity': ['hERG', 'AMES', 'DILI', 'LD50_Zhu', 'Carcinogens_Lagunin'],
}


def check_tdc_available() -> bool:
    """Check if TDC is installed."""
    try:
        import tdc
        return True
    except ImportError:
        return False


class TDCDatasetLoader:
    """Load ADMET datasets from Therapeutics Data Commons."""

    def __init__(self, data_dir: str = 'outputs/tdc_data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.datasets = {}
        self.merged_df = None
        self.task_types = {}

        if not check_tdc_available():
            print("WARNING: TDC not installed. Install with: pip install PyTDC")

    def load_single_dataset(self, name: str) -> Optional[pd.DataFrame]:
        """Load a single TDC ADMET dataset."""
        if name not in TDC_ADMET_DATASETS:
            print(f"Unknown TDC dataset: {name}")
            return None

        if not check_tdc_available():
            print("TDC not installed")
            return None

        try:
            from tdc.single_pred import ADME, Tox

            info = TDC_ADMET_DATASETS[name]

            # Determine which TDC class to use
            if info['group'] in ['Absorption', 'Distribution', 'Metabolism', 'Excretion']:
                data = ADME(name=name)
            else:  # Toxicity
                data = Tox(name=name)

            df = data.get_data()

            # Standardize columns
            result = pd.DataFrame({
                'smiles': df['Drug'],
                name: df['Y'].values.astype(np.float32)
            })

            self.task_types[name] = info['type']

            print(f"  Loaded {name}: {len(result)} molecules")
            return result

        except Exception as e:
            print(f"  Failed to load {name}: {e}")
            return None

    def load_datasets(self, dataset_names: list = None) -> pd.DataFrame:
        """Load and merge multiple TDC datasets."""
        if dataset_names is None:
            # Default: key ADMET properties
            dataset_names = [
                'Caco2_Wang', 'BBB_Martins', 'Solubility_AqSolDB',
                'CYP2D6_Veith', 'CYP3A4_Veith',
                'Clearance_Hepatocyte_AZ', 'hERG', 'AMES', 'DILI'
            ]

        all_dfs = []

        print("Loading TDC datasets...")
        for name in dataset_names:
            df = self.load_single_dataset(name)
            if df is not None:
                all_dfs.append(df)
                self.datasets[name] = df

        if not all_dfs:
            raise ValueError("No TDC datasets loaded")

        # Merge on SMILES
        merged = all_dfs[0]
        for df in all_dfs[1:]:
            merged = pd.merge(merged, df, on='smiles', how='outer')

        self.merged_df = merged
        print(f"\nMerged TDC dataset: {len(merged)} molecules, {len(merged.columns)-1} tasks")

        return merged

    def get_task_names(self) -> list:
        """Get list of task names."""
        if self.merged_df is None:
            return []
        return [col for col in self.merged_df.columns if col != 'smiles']

    def get_labels_dict(self) -> dict:
        """Get labels as dictionary."""
        if self.merged_df is None:
            return {}

        labels = {}
        for task in self.get_task_names():
            labels[task] = self.merged_df[task].values.astype(np.float32)
        return labels

    def get_smiles_list(self) -> list:
        """Get list of SMILES."""
        if self.merged_df is None:
            return []
        return self.merged_df['smiles'].tolist()

    def filter_by_coverage(self, min_tasks: int = 3) -> pd.DataFrame:
        """Filter to molecules with minimum task coverage."""
        if self.merged_df is None:
            raise ValueError("No data loaded")

        task_cols = self.get_task_names()
        n_labels = self.merged_df[task_cols].notna().sum(axis=1)
        mask = n_labels >= min_tasks

        filtered = self.merged_df[mask].copy()
        print(f"Filtered to {len(filtered)} molecules with >= {min_tasks} labels")

        return filtered


def load_tdc_admet(
    dataset_names: list = None,
    min_tasks: int = 2,
    data_dir: str = 'outputs/tdc_data'
) -> tuple:
    """
    Load TDC ADMET datasets.

    Returns:
        smiles_list, labels_dict, task_types
    """
    loader = TDCDatasetLoader(data_dir)
    loader.load_datasets(dataset_names)
    filtered = loader.filter_by_coverage(min_tasks)
    loader.merged_df = filtered

    return loader.get_smiles_list(), loader.get_labels_dict(), loader.task_types


def get_tdc_mechanistic_clusters() -> dict:
    """Get known mechanistic clusters from TDC properties."""
    return TDC_MECHANISTIC_CLUSTERS.copy()


def get_tdc_dataset_info() -> pd.DataFrame:
    """Get information about all TDC ADMET datasets."""
    rows = []
    for name, info in TDC_ADMET_DATASETS.items():
        rows.append({
            'dataset': name,
            'group': info['group'],
            'type': info['type'],
            'description': info['description']
        })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    print("TDC ADMET Dataset Information:")
    print("=" * 60)

    info_df = get_tdc_dataset_info()
    for group in ['Absorption', 'Distribution', 'Metabolism', 'Excretion', 'Toxicity']:
        group_df = info_df[info_df['group'] == group]
        print(f"\n{group}:")
        for _, row in group_df.iterrows():
            print(f"  {row['dataset']}: {row['description']} ({row['type']})")

    if check_tdc_available():
        print("\n\nTesting TDC loader...")
        smiles, labels, task_types = load_tdc_admet(
            dataset_names=['BBB_Martins', 'hERG', 'AMES'],
            min_tasks=2
        )
        print(f"\nLoaded {len(smiles)} molecules with {len(labels)} tasks")
    else:
        print("\n\nTDC not installed. Install with: pip install PyTDC")
