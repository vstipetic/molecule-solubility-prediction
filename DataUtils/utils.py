"""Utility functions for molecular data processing.

This module provides shared utilities for SMILES processing, fingerprint computation,
scaffold splitting, and molecular featurization.
"""

from typing import List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


# Atom feature constants
ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'Si', 'B', 'Na', 'K', 'Ca', 'Mg', 'Fe', 'Zn', 'Cu', 'Unknown']
HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    """Convert a SMILES string to an RDKit Mol object.

    Args:
        smiles: SMILES string representation of a molecule.

    Returns:
        RDKit Mol object, or None if parsing fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    return mol


def compute_ecfp(
    mol: Chem.Mol,
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Compute Extended Connectivity Fingerprints (ECFP) for a molecule.

    Args:
        mol: RDKit Mol object.
        radius: Radius for Morgan fingerprint (2 = ECFP4, 3 = ECFP6).
        n_bits: Number of bits in the fingerprint vector.

    Returns:
        Numpy array of shape (n_bits,) containing the fingerprint.
    """
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def compute_ecfp_from_smiles(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> Optional[np.ndarray]:
    """Compute ECFP fingerprint directly from SMILES string.

    Args:
        smiles: SMILES string representation of a molecule.
        radius: Radius for Morgan fingerprint.
        n_bits: Number of bits in the fingerprint vector.

    Returns:
        Numpy array of fingerprint, or None if SMILES parsing fails.
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    return compute_ecfp(mol, radius, n_bits)


def get_scaffold(smiles: str) -> str:
    """Get the Murcko scaffold for a molecule.

    Args:
        smiles: SMILES string representation of a molecule.

    Returns:
        SMILES string of the Murcko scaffold.
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return ""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)


def scaffold_split(
    smiles_list: List[str],
    labels: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    random_state: int = 42,
) -> Tuple[
    Tuple[List[str], np.ndarray],
    Tuple[List[str], np.ndarray],
    Tuple[List[str], np.ndarray],
]:
    """Split data based on molecular scaffolds.

    Molecules with the same scaffold will be in the same split to prevent
    data leakage between train/val/test sets.

    Args:
        smiles_list: List of SMILES strings.
        labels: Numpy array of labels (solubility values).
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        random_state: Random seed for reproducibility.

    Returns:
        Three tuples of (smiles_list, labels) for train, val, and test sets.
    """
    np.random.seed(random_state)

    # Group molecules by scaffold
    scaffold_to_indices: dict[str, List[int]] = {}
    for idx, smiles in enumerate(smiles_list):
        scaffold = get_scaffold(smiles)
        if scaffold not in scaffold_to_indices:
            scaffold_to_indices[scaffold] = []
        scaffold_to_indices[scaffold].append(idx)

    # Sort scaffolds by size (largest first) for more balanced splits
    scaffold_sets = list(scaffold_to_indices.values())
    scaffold_sets.sort(key=lambda x: len(x), reverse=True)

    # Assign scaffolds to splits
    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    n_total = len(smiles_list)
    train_size = int(n_total * train_ratio)
    val_size = int(n_total * val_ratio)

    for scaffold_indices in scaffold_sets:
        if len(train_indices) < train_size:
            train_indices.extend(scaffold_indices)
        elif len(val_indices) < val_size:
            val_indices.extend(scaffold_indices)
        else:
            test_indices.extend(scaffold_indices)

    # Create output arrays
    train_smiles = [smiles_list[i] for i in train_indices]
    val_smiles = [smiles_list[i] for i in val_indices]
    test_smiles = [smiles_list[i] for i in test_indices]

    train_labels = labels[train_indices]
    val_labels = labels[val_indices]
    test_labels = labels[test_indices]

    return (
        (train_smiles, train_labels),
        (val_smiles, val_labels),
        (test_smiles, test_labels),
    )


def one_hot_encode(value: int, choices: List) -> List[float]:
    """One-hot encode a value given a list of choices.

    Args:
        value: The value to encode.
        choices: List of possible values.

    Returns:
        One-hot encoded list.
    """
    encoding = [0.0] * len(choices)
    if value in choices:
        encoding[choices.index(value)] = 1.0
    return encoding


def get_atom_features(atom: Chem.Atom) -> List[float]:
    """Get feature vector for an atom.

    Features include:
    - Atom type (one-hot)
    - Degree
    - Formal charge
    - Number of hydrogens
    - Hybridization (one-hot)
    - Aromaticity
    - Mass

    Args:
        atom: RDKit Atom object.

    Returns:
        List of atom features.
    """
    features: List[float] = []

    # Atom type (one-hot)
    atom_symbol = atom.GetSymbol()
    if atom_symbol not in ATOM_TYPES[:-1]:
        atom_symbol = 'Unknown'
    features.extend(one_hot_encode(atom_symbol, ATOM_TYPES))

    # Degree (number of bonds)
    features.append(float(atom.GetDegree()))

    # Formal charge
    features.append(float(atom.GetFormalCharge()))

    # Number of hydrogens
    features.append(float(atom.GetTotalNumHs()))

    # Hybridization (one-hot)
    hybridization = atom.GetHybridization()
    features.extend(one_hot_encode(hybridization, HYBRIDIZATION_TYPES))

    # Aromaticity
    features.append(1.0 if atom.GetIsAromatic() else 0.0)

    # Atomic mass (normalized)
    features.append(float(atom.GetMass()) / 100.0)

    return features


def get_bond_features(bond: Chem.Bond) -> List[float]:
    """Get feature vector for a bond.

    Features include:
    - Bond type (one-hot)
    - Conjugation
    - Ring membership
    - Stereo configuration

    Args:
        bond: RDKit Bond object.

    Returns:
        List of bond features.
    """
    features: List[float] = []

    # Bond type (one-hot)
    bond_type = bond.GetBondType()
    features.extend(one_hot_encode(bond_type, BOND_TYPES))

    # Conjugation
    features.append(1.0 if bond.GetIsConjugated() else 0.0)

    # Ring membership
    features.append(1.0 if bond.IsInRing() else 0.0)

    # Stereo (simplified)
    stereo = bond.GetStereo()
    features.append(1.0 if stereo != Chem.rdchem.BondStereo.STEREONONE else 0.0)

    return features


def get_atom_feature_dim() -> int:
    """Get the dimension of atom feature vectors.

    Returns:
        Dimension of atom features.
    """
    # Atom type + degree + charge + H count + hybridization + aromatic + mass
    return len(ATOM_TYPES) + 1 + 1 + 1 + len(HYBRIDIZATION_TYPES) + 1 + 1


def get_bond_feature_dim() -> int:
    """Get the dimension of bond feature vectors.

    Returns:
        Dimension of bond features.
    """
    # Bond type + conjugation + ring + stereo
    return len(BOND_TYPES) + 1 + 1 + 1


def load_data(
    data_path: str,
    smiles_column: str = "SMILES",
    target_column: str = "Solubility",
) -> Tuple[List[str], np.ndarray]:
    """Load molecular data from CSV file.

    Args:
        data_path: Path to CSV file.
        smiles_column: Name of column containing SMILES strings.
        target_column: Name of column containing target values.

    Returns:
        Tuple of (smiles_list, targets_array).
    """
    import pandas as pd

    df = pd.read_csv(data_path)
    smiles_list = df[smiles_column].tolist()
    targets = df[target_column].values.astype(np.float32)
    return smiles_list, targets


def load_zinc_data(
    data_path: str,
    max_samples: Optional[int] = None,
) -> List[str]:
    """Load ZINC SMILES data from CSV or text file.

    Automatically detects SMILES column by trying common column names.

    Args:
        data_path: Path to ZINC data file (CSV or txt with SMILES).
        max_samples: Maximum number of samples to load.

    Returns:
        List of SMILES strings.
    """
    import pandas as pd

    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
        # Try common column names
        for col in ['smiles', 'SMILES', 'Smiles', 'canonical_smiles']:
            if col in df.columns:
                smiles_list = df[col].tolist()
                break
        else:
            # Fall back to first column
            smiles_list = df.iloc[:, 0].tolist()
    else:
        with open(data_path, 'r') as f:
            smiles_list = [line.strip() for line in f if line.strip()]

    if max_samples is not None:
        smiles_list = smiles_list[:max_samples]

    return smiles_list
