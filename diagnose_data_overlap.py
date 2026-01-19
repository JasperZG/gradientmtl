"""Check overlap between task labels - do molecules have multiple task labels?"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml

# Load config
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Load raw data files
from data.download import load_dataset

datasets = {}
raw_dir = Path('outputs/raw_data')

files = {
    'bace': raw_dir / 'bace_raw.csv',
    'bbbp': raw_dir / 'bbbp_raw.csv',
    'esol': raw_dir / 'esol_raw.csv',
    'lipophilicity': raw_dir / 'lipophilicity_raw.csv',
    'herg': raw_dir / 'herg_raw.tab',
}

for task, path in files.items():
    datasets[task] = load_dataset(path, config['tasks'][task])

print("Dataset Overlap Analysis")
print("=" * 60)

# Check SMILES overlap between datasets
task_names = list(datasets.keys())

# Create standardized SMILES sets
from rdkit import Chem

def canonicalize(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None

smiles_sets = {}
for task, df in datasets.items():
    canonical = set()
    for smi in df['smiles']:
        can_smi = canonicalize(smi)
        if can_smi:
            canonical.add(can_smi)
    smiles_sets[task] = canonical
    print(f"{task}: {len(canonical)} unique molecules")

print()
print("Pairwise Molecule Overlap:")
print("-" * 60)

for i, t1 in enumerate(task_names):
    for j, t2 in enumerate(task_names):
        if i < j:
            overlap = len(smiles_sets[t1] & smiles_sets[t2])
            pct1 = 100 * overlap / len(smiles_sets[t1]) if smiles_sets[t1] else 0
            pct2 = 100 * overlap / len(smiles_sets[t2]) if smiles_sets[t2] else 0
            print(f"{t1} & {t2}: {overlap} molecules ({pct1:.1f}% of {t1}, {pct2:.1f}% of {t2})")

print()
print("Key Insight:")
print("-" * 60)
print("""
If datasets have little overlap, gradient conflicts may be noisy because:
- Different molecules contribute to different task gradients
- The shared encoder sees different regions of chemical space for each task
- True mechanistic relationships might not be captured

For meaningful gradient conflict detection, we need molecules with
MULTIPLE task labels so gradients are computed on the SAME molecules.
""")

# Check how many molecules have labels for multiple tasks
# (After merging)
from data.preprocessing import MoleculePreprocessor

preprocessor = MoleculePreprocessor()

# Find molecules in ALL datasets
all_canonical = set.intersection(*smiles_sets.values())
print(f"Molecules with labels for ALL 5 tasks: {len(all_canonical)}")

# Find molecules in at least 2 tasks
in_multiple = set()
for smi in set.union(*smiles_sets.values()):
    count = sum(1 for s in smiles_sets.values() if smi in s)
    if count >= 2:
        in_multiple.add(smi)
print(f"Molecules with labels for 2+ tasks: {len(in_multiple)}")
