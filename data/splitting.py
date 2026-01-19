"""Scaffold-based splitting for molecular datasets."""

from collections import defaultdict
from typing import Optional
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def get_scaffold(smiles: str) -> str:
    """
    Extract Murcko scaffold from a molecule.

    The Murcko scaffold is the core ring structure of a molecule,
    with all side chains removed.

    Args:
        smiles: SMILES string of molecule

    Returns:
        Canonical SMILES of scaffold, or empty string if parsing fails
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ''

        # Get the core scaffold (generic scaffold without side chains)
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, canonical=True)

    except Exception:
        return ''


def scaffold_split(
    smiles_list: list[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """
    Split molecules by scaffold to prevent data leakage.

    This ensures that molecules with the same core scaffold are all
    in the same split, preventing the model from memorizing scaffold
    patterns and achieving unrealistically high test performance.

    Args:
        smiles_list: List of SMILES strings
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1"

    # Group molecules by scaffold
    scaffold_to_indices = defaultdict(list)
    for idx, smi in enumerate(smiles_list):
        scaffold = get_scaffold(smi)
        scaffold_to_indices[scaffold].append(idx)

    # Sort scaffolds by size (largest first) for reproducibility
    scaffold_groups = list(scaffold_to_indices.values())
    scaffold_groups.sort(key=lambda x: (-len(x), min(x)))  # Size desc, then min index

    # Shuffle with seed for randomness within size classes
    rng = np.random.default_rng(random_seed)

    # Group scaffolds by size and shuffle within each size class
    size_to_scaffolds = defaultdict(list)
    for group in scaffold_groups:
        size_to_scaffolds[len(group)].append(group)

    shuffled_groups = []
    for size in sorted(size_to_scaffolds.keys(), reverse=True):
        groups = size_to_scaffolds[size]
        rng.shuffle(groups)
        shuffled_groups.extend(groups)

    # Calculate target sizes
    n_total = len(smiles_list)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    # n_test is the remainder

    # Greedy assignment: add entire scaffold groups to splits
    train_idx = []
    val_idx = []
    test_idx = []

    for group in shuffled_groups:
        if len(train_idx) < n_train:
            train_idx.extend(group)
        elif len(val_idx) < n_val:
            val_idx.extend(group)
        else:
            test_idx.extend(group)

    # Shuffle indices within each split
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    print(f"Scaffold split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")
    print(f"  ({len(train_idx)/n_total:.1%} / {len(val_idx)/n_total:.1%} / {len(test_idx)/n_total:.1%})")
    print(f"  {len(scaffold_to_indices)} unique scaffolds")

    return train_idx, val_idx, test_idx


def random_split(
    n_samples: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """
    Random split (for comparison/debugging, not recommended for molecules).

    Args:
        n_samples: Total number of samples
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    rng = np.random.default_rng(random_seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)

    train_idx = indices[:n_train].tolist()
    val_idx = indices[n_train:n_train + n_val].tolist()
    test_idx = indices[n_train + n_val:].tolist()

    return train_idx, val_idx, test_idx


def verify_scaffold_split(
    smiles_list: list[str],
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int]
) -> bool:
    """
    Verify that no scaffold appears in multiple splits.

    Args:
        smiles_list: List of SMILES strings
        train_idx: Training indices
        val_idx: Validation indices
        test_idx: Test indices

    Returns:
        True if split is valid (no scaffold leakage), False otherwise
    """
    train_scaffolds = set(get_scaffold(smiles_list[i]) for i in train_idx)
    val_scaffolds = set(get_scaffold(smiles_list[i]) for i in val_idx)
    test_scaffolds = set(get_scaffold(smiles_list[i]) for i in test_idx)

    train_val_overlap = train_scaffolds & val_scaffolds
    train_test_overlap = train_scaffolds & test_scaffolds
    val_test_overlap = val_scaffolds & test_scaffolds

    if train_val_overlap or train_test_overlap or val_test_overlap:
        print(f"Warning: Scaffold leakage detected!")
        print(f"  Train-Val overlap: {len(train_val_overlap)} scaffolds")
        print(f"  Train-Test overlap: {len(train_test_overlap)} scaffolds")
        print(f"  Val-Test overlap: {len(val_test_overlap)} scaffolds")
        return False

    print("Scaffold split verified: No leakage detected")
    return True
