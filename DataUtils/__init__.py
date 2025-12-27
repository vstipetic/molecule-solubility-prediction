"""DataUtils module for molecular solubility prediction.

This module provides utilities for:
- SMILES processing and molecular featurization
- Dataset classes for AqSolDB and ESOL
- Graph preprocessing for GNN-based approaches
"""

from DataUtils.utils import (
    smiles_to_mol,
    compute_ecfp,
    scaffold_split,
    get_atom_features,
    get_bond_features,
)
from DataUtils.datasets import AqSolDBDataset, ESOLDataset
from DataUtils.graph_preprocessing import smiles_to_graph, MoleculeGraphDataset

__all__ = [
    "smiles_to_mol",
    "compute_ecfp",
    "scaffold_split",
    "get_atom_features",
    "get_bond_features",
    "AqSolDBDataset",
    "ESOLDataset",
    "smiles_to_graph",
    "MoleculeGraphDataset",
]
