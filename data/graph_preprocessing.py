"""
Graph-based molecular preprocessing for GNN models.

Converts SMILES strings to PyTorch Geometric Data objects with:
- Node features: Atom properties (element, degree, charge, etc.)
- Edge features: Bond properties (type, conjugation, ring membership)
- Edge index: Adjacency in COO format
"""

import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm


# Atom feature dimensions
ATOM_FEATURES = {
    'atomic_num': list(range(1, 119)),  # H to Og
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-2, -1, 0, 1, 2],
    'num_hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

# Bond feature dimensions
BOND_FEATURES = {
    'bond_type': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    'is_conjugated': [False, True],
    'is_in_ring': [False, True],
    'stereo': [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
    ],
}


def one_hot_encode(value, choices):
    """One-hot encode a value given a list of choices."""
    encoding = [0] * (len(choices) + 1)  # +1 for unknown
    try:
        idx = choices.index(value)
        encoding[idx] = 1
    except ValueError:
        encoding[-1] = 1  # Unknown category
    return encoding


def get_atom_features(atom):
    """Extract features for a single atom."""
    features = []

    # Atomic number (one-hot, but we'll use a subset of common elements)
    common_atoms = [6, 7, 8, 9, 15, 16, 17, 35, 53]  # C, N, O, F, P, S, Cl, Br, I
    features.extend(one_hot_encode(atom.GetAtomicNum(), common_atoms))

    # Degree
    features.extend(one_hot_encode(atom.GetDegree(), ATOM_FEATURES['degree']))

    # Formal charge
    features.extend(one_hot_encode(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']))

    # Number of hydrogens
    features.extend(one_hot_encode(atom.GetTotalNumHs(), ATOM_FEATURES['num_hs']))

    # Hybridization
    features.extend(one_hot_encode(atom.GetHybridization(), ATOM_FEATURES['hybridization']))

    # Aromaticity
    features.append(1 if atom.GetIsAromatic() else 0)

    # In ring
    features.append(1 if atom.IsInRing() else 0)

    return features


def get_bond_features(bond):
    """Extract features for a single bond."""
    features = []

    # Bond type
    features.extend(one_hot_encode(bond.GetBondType(), BOND_FEATURES['bond_type']))

    # Conjugation
    features.append(1 if bond.GetIsConjugated() else 0)

    # In ring
    features.append(1 if bond.IsInRing() else 0)

    # Stereo
    features.extend(one_hot_encode(bond.GetStereo(), BOND_FEATURES['stereo']))

    return features


def smiles_to_graph(smiles: str) -> Data | None:
    """
    Convert a SMILES string to a PyTorch Geometric Data object.

    Args:
        smiles: SMILES string

    Returns:
        PyG Data object or None if parsing fails
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Get atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))

    if len(atom_features) == 0:
        return None

    x = torch.tensor(atom_features, dtype=torch.float)

    # Get bond features and edge index
    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bond_feat = get_bond_features(bond)

        # Add both directions (undirected graph)
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append(bond_feat)
        edge_attr.append(bond_feat)

    if len(edge_index) == 0:
        # Single atom molecule - add self-loop
        edge_index = [[0, 0]]
        edge_attr = [[0] * 12]  # Zero features for self-loop

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class MoleculeGraphPreprocessor:
    """Preprocessor for converting SMILES to molecular graphs."""

    def __init__(self):
        # Compute feature dimensions
        self.atom_feature_dim = self._compute_atom_dim()
        self.bond_feature_dim = self._compute_bond_dim()

    def _compute_atom_dim(self) -> int:
        """Compute the dimension of atom features."""
        # Common atoms (9) + 1 unknown
        # Degree (6) + 1 unknown
        # Formal charge (5) + 1 unknown
        # Num Hs (5) + 1 unknown
        # Hybridization (5) + 1 unknown
        # Aromatic (1)
        # In ring (1)
        return 10 + 7 + 6 + 6 + 6 + 1 + 1  # = 37

    def _compute_bond_dim(self) -> int:
        """Compute the dimension of bond features."""
        # Bond type (4) + 1 unknown
        # Conjugated (1)
        # In ring (1)
        # Stereo (4) + 1 unknown
        return 5 + 1 + 1 + 5  # = 12

    def process_smiles(self, smiles: str) -> Data | None:
        """Convert a single SMILES to a graph."""
        return smiles_to_graph(smiles)

    def process_smiles_list(
        self,
        smiles_list: list[str],
        show_progress: bool = True,
    ) -> tuple[list[str], list[Data], list[int]]:
        """
        Process a list of SMILES strings to molecular graphs.

        Args:
            smiles_list: List of SMILES strings
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (valid_smiles, graph_list, valid_indices)
        """
        valid_smiles = []
        graphs = []
        valid_indices = []

        iterator = tqdm(enumerate(smiles_list), total=len(smiles_list), desc="Converting to graphs") \
                   if show_progress else enumerate(smiles_list)

        for idx, smi in iterator:
            graph = self.process_smiles(smi)
            if graph is not None:
                valid_smiles.append(smi)
                graphs.append(graph)
                valid_indices.append(idx)

        print(f"Converted {len(graphs)}/{len(smiles_list)} SMILES to graphs")
        print(f"Atom feature dim: {self.atom_feature_dim}")
        print(f"Bond feature dim: {self.bond_feature_dim}")

        return valid_smiles, graphs, valid_indices


def get_atom_feature_dim() -> int:
    """Get the dimension of atom features."""
    return MoleculeGraphPreprocessor().atom_feature_dim


def get_bond_feature_dim() -> int:
    """Get the dimension of bond features."""
    return MoleculeGraphPreprocessor().bond_feature_dim
