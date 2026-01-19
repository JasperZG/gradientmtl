"""SMILES standardization and fingerprint generation using RDKit."""

from typing import Optional
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def smiles_to_ecfp4(
    smiles: str,
    n_bits: int = 2048,
    radius: int = 2
) -> Optional[np.ndarray]:
    """
    Convert SMILES string to ECFP fingerprint.

    Args:
        smiles: SMILES string
        n_bits: Number of bits in fingerprint (default 2048)
        radius: Radius for Morgan fingerprint (2 = ECFP4, 3 = ECFP6)

    Returns:
        Numpy array of shape (n_bits,) or None if parsing fails
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


class MoleculePreprocessor:
    """Preprocess molecules: standardize SMILES and compute fingerprints."""

    def __init__(
        self,
        fp_bits: int = 2048,
        fp_radius: int = 2,
        canonicalize: bool = True,
        remove_salts: bool = True,
    ):
        """
        Args:
            fp_bits: Number of bits in fingerprint
            fp_radius: Radius for Morgan fingerprint (2 = ECFP4)
            canonicalize: Whether to canonicalize SMILES
            remove_salts: Whether to keep only largest fragment (removes salts)
        """
        self.fp_bits = fp_bits
        self.fp_radius = fp_radius
        self.canonicalize = canonicalize
        self.remove_salts = remove_salts

    def standardize_smiles(self, smiles: str) -> Optional[str]:
        """
        Standardize a SMILES string.

        Args:
            smiles: Input SMILES string

        Returns:
            Canonical SMILES or None if parsing fails
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # Remove salts by keeping largest fragment
            if self.remove_salts:
                frags = Chem.GetMolFrags(mol, asMols=True)
                if len(frags) > 1:
                    mol = max(frags, key=lambda m: m.GetNumAtoms())

            # Sanitize
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                return None

            # Canonicalize
            if self.canonicalize:
                return Chem.MolToSmiles(mol, canonical=True)
            else:
                return Chem.MolToSmiles(mol)

        except Exception:
            return None

    def compute_fingerprint(self, smiles: str) -> Optional[np.ndarray]:
        """
        Compute ECFP fingerprint for a SMILES string.

        Args:
            smiles: SMILES string (should be standardized)

        Returns:
            Fingerprint array or None if parsing fails
        """
        return smiles_to_ecfp4(smiles, self.fp_bits, self.fp_radius)

    def process_smiles_list(
        self,
        smiles_list: list[str],
        show_progress: bool = True
    ) -> tuple[list[str], np.ndarray, list[int]]:
        """
        Process a list of SMILES strings.

        Args:
            smiles_list: List of SMILES strings
            show_progress: Whether to print progress

        Returns:
            Tuple of (valid_smiles, fingerprints, valid_indices)
            - valid_smiles: List of standardized SMILES (only valid molecules)
            - fingerprints: Array of shape (n_valid, fp_bits)
            - valid_indices: Original indices of valid molecules
        """
        valid_smiles = []
        fingerprints = []
        valid_indices = []

        n_failed = 0
        for i, smi in enumerate(smiles_list):
            # Standardize
            std_smi = self.standardize_smiles(smi)
            if std_smi is None:
                n_failed += 1
                continue

            # Compute fingerprint
            fp = self.compute_fingerprint(std_smi)
            if fp is None:
                n_failed += 1
                continue

            valid_smiles.append(std_smi)
            fingerprints.append(fp)
            valid_indices.append(i)

            if show_progress and (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(smiles_list)} molecules...")

        if show_progress and n_failed > 0:
            print(f"Warning: {n_failed}/{len(smiles_list)} molecules failed to parse")

        fingerprints_array = np.stack(fingerprints) if fingerprints else np.array([])

        return valid_smiles, fingerprints_array, valid_indices


def merge_datasets_by_smiles(
    datasets: dict[str, tuple[list[str], np.ndarray]],
    labels_dict: dict[str, np.ndarray],
    preprocessor: MoleculePreprocessor
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str]]:
    """
    Merge multiple datasets by standardized SMILES.

    This creates a unified molecule list where each molecule has labels
    for the tasks where it appears, and NaN for tasks where it doesn't.

    Args:
        datasets: Dict mapping task -> (smiles_list, labels_array)
        labels_dict: Already processed labels for each task
        preprocessor: Preprocessor for standardization

    Returns:
        Tuple of (fingerprints, labels_dict, canonical_smiles)
        - fingerprints: (N, fp_bits) array for all unique molecules
        - labels_dict: task -> (N,) array with NaN for missing labels
        - canonical_smiles: List of canonical SMILES for all molecules
    """
    # Collect all unique standardized SMILES
    smiles_to_idx = {}
    all_smiles = []
    all_fingerprints = []

    for task, (smiles_list, _) in datasets.items():
        for smi in smiles_list:
            std_smi = preprocessor.standardize_smiles(smi)
            if std_smi is not None and std_smi not in smiles_to_idx:
                fp = preprocessor.compute_fingerprint(std_smi)
                if fp is not None:
                    smiles_to_idx[std_smi] = len(all_smiles)
                    all_smiles.append(std_smi)
                    all_fingerprints.append(fp)

    n_molecules = len(all_smiles)
    print(f"Total unique molecules: {n_molecules}")

    # Create label arrays with NaN for missing
    merged_labels = {}
    for task, (smiles_list, labels) in datasets.items():
        task_labels = np.full(n_molecules, np.nan, dtype=np.float32)

        for smi, label in zip(smiles_list, labels):
            std_smi = preprocessor.standardize_smiles(smi)
            if std_smi in smiles_to_idx:
                idx = smiles_to_idx[std_smi]
                task_labels[idx] = label

        n_valid = np.sum(~np.isnan(task_labels))
        print(f"  {task}: {n_valid} labels")
        merged_labels[task] = task_labels

    fingerprints = np.stack(all_fingerprints)

    return fingerprints, merged_labels, all_smiles
