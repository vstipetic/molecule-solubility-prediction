"""DataUtils module for molecular solubility prediction.

This module provides utilities for:
- SMILES processing and molecular featurization
- Dataset classes for AqSolDB, ESOL, and transformer training
- Graph preprocessing for GNN-based approaches
- Metrics computation for model evaluation
- Data loading and preparation utilities
"""

from DataUtils.utils import (
    smiles_to_mol,
    compute_ecfp,
    compute_ecfp_from_smiles,
    scaffold_split,
    get_atom_features,
    get_bond_features,
    get_atom_feature_dim,
    get_bond_feature_dim,
    load_data,
    load_zinc_data,
)
from DataUtils.datasets import (
    AqSolDBDataset,
    ESOLDataset,
    SubsetDataset,
    ZINCDataset,
    TransformerSolubilityDataset,
    create_subset_dataset,
)
from DataUtils.graph_preprocessing import (
    smiles_to_graph,
    smiles_to_dmpnn_graph,
    MoleculeGraphDataset,
    collate_dmpnn_batch,
    get_graph_feature_dims,
)
from DataUtils.metrics import (
    compute_metrics,
    compute_calibration_metrics,
)
from DataUtils.collate import (
    transformer_collate_fn,
)

__all__ = [
    # utils
    "smiles_to_mol",
    "compute_ecfp",
    "compute_ecfp_from_smiles",
    "scaffold_split",
    "get_atom_features",
    "get_bond_features",
    "get_atom_feature_dim",
    "get_bond_feature_dim",
    "load_data",
    "load_zinc_data",
    # datasets
    "AqSolDBDataset",
    "ESOLDataset",
    "SubsetDataset",
    "ZINCDataset",
    "TransformerSolubilityDataset",
    "create_subset_dataset",
    # graph_preprocessing
    "smiles_to_graph",
    "smiles_to_dmpnn_graph",
    "MoleculeGraphDataset",
    "collate_dmpnn_batch",
    "get_graph_feature_dims",
    # metrics
    "compute_metrics",
    "compute_calibration_metrics",
    # collate
    "transformer_collate_fn",
]
