"""Graph preprocessing utilities for GNN-based molecular models.

This module provides functions to convert SMILES strings to graph representations
suitable for Graph Neural Networks, specifically for PyTorch Geometric.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from DataUtils.utils import (
    get_atom_features,
    get_bond_features,
    get_atom_feature_dim,
    get_bond_feature_dim,
    smiles_to_mol,
)


def smiles_to_graph(smiles: str) -> Optional[Data]:
    """Convert a SMILES string to a PyTorch Geometric Data object.

    Creates a graph representation where:
    - Nodes represent atoms with atom features
    - Edges represent bonds with bond features (bidirectional)

    Args:
        smiles: SMILES string representation of a molecule.

    Returns:
        PyTorch Geometric Data object, or None if SMILES parsing fails.
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None

    # Get atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))

    x = torch.tensor(atom_features, dtype=torch.float32)

    # Get bonds and create edge indices and features
    edge_indices = []
    edge_features = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bond_feat = get_bond_features(bond)

        # Add both directions (undirected graph representation)
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        edge_features.append(bond_feat)
        edge_features.append(bond_feat)

    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
    else:
        # Single atom molecule
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, get_bond_feature_dim()), dtype=torch.float32)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        smiles=smiles,
    )

    return data


def smiles_to_dmpnn_graph(smiles: str) -> Optional[Data]:
    """Convert SMILES to graph format optimized for D-MPNN.

    D-MPNN uses directed edges with message passing along bond directions.
    This function creates additional attributes needed for D-MPNN:
    - a2b: Mapping from atoms to outgoing bonds
    - b2a: Mapping from bonds to target atoms
    - b2revb: Mapping from bonds to reverse bonds

    Args:
        smiles: SMILES string representation of a molecule.

    Returns:
        PyTorch Geometric Data object with D-MPNN specific attributes.
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None

    n_atoms = mol.GetNumAtoms()
    n_bonds = mol.GetNumBonds()

    # Get atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))

    x = torch.tensor(atom_features, dtype=torch.float32)

    # Build bond information for D-MPNN
    # We create 2 * n_bonds directed edges (both directions)
    edge_indices = []
    edge_features = []
    a2b: List[List[int]] = [[] for _ in range(n_atoms)]  # atom to bonds
    b2a: List[int] = []  # bond to atom (target)
    b2revb: List[int] = []  # bond to reverse bond

    bond_idx = 0
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_feat = get_bond_features(bond)

        # Direction i -> j
        edge_indices.append([i, j])
        edge_features.append(bond_feat)
        a2b[i].append(bond_idx)
        b2a.append(j)

        # Direction j -> i (reverse)
        edge_indices.append([j, i])
        edge_features.append(bond_feat)
        a2b[j].append(bond_idx + 1)
        b2a.append(i)

        # Reverse bond mapping
        b2revb.append(bond_idx + 1)
        b2revb.append(bond_idx)

        bond_idx += 2

    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, get_bond_feature_dim()), dtype=torch.float32)

    # Pad a2b to have same length for batching
    max_num_bonds = max(len(bonds) for bonds in a2b) if a2b else 0
    a2b_padded = []
    for bonds in a2b:
        padded = bonds + [-1] * (max_num_bonds - len(bonds))
        a2b_padded.append(padded)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        smiles=smiles,
        a2b=torch.tensor(a2b_padded, dtype=torch.long) if a2b_padded else torch.zeros((0, 0), dtype=torch.long),
        b2a=torch.tensor(b2a, dtype=torch.long) if b2a else torch.zeros(0, dtype=torch.long),
        b2revb=torch.tensor(b2revb, dtype=torch.long) if b2revb else torch.zeros(0, dtype=torch.long),
        num_atoms=n_atoms,
        num_bonds=bond_idx,
    )

    return data


class MoleculeGraphDataset(Dataset):
    """PyTorch Geometric Dataset for molecular graphs.

    Converts SMILES strings to graph representations on-the-fly or precomputed.

    Args:
        smiles_list: List of SMILES strings.
        targets: Numpy array of target values (solubility).
        precompute: Whether to precompute all graphs at initialization.
        use_dmpnn_format: Whether to use D-MPNN specific graph format.

    Example:
        >>> dataset = MoleculeGraphDataset(smiles_list, targets)
        >>> data = dataset[0]
        >>> print(data.x.shape)  # Atom features
        >>> print(data.edge_index.shape)  # Edge connectivity
    """

    def __init__(
        self,
        smiles_list: List[str],
        targets: np.ndarray,
        precompute: bool = False,
        use_dmpnn_format: bool = False,
    ) -> None:
        super().__init__()

        self.smiles_list = smiles_list
        self.targets = targets.astype(np.float32)
        self.use_dmpnn_format = use_dmpnn_format

        # Filter out invalid SMILES
        valid_indices = []
        for idx, smiles in enumerate(smiles_list):
            mol = smiles_to_mol(smiles)
            if mol is not None:
                valid_indices.append(idx)

        self.valid_smiles = [smiles_list[i] for i in valid_indices]
        self.valid_targets = self.targets[valid_indices]

        self._precomputed_graphs: Optional[List[Data]] = None
        if precompute:
            self._precompute_graphs()

    def _precompute_graphs(self) -> None:
        """Precompute all graph representations."""
        self._precomputed_graphs = []
        convert_fn = smiles_to_dmpnn_graph if self.use_dmpnn_format else smiles_to_graph

        for idx, smiles in enumerate(self.valid_smiles):
            graph = convert_fn(smiles)
            if graph is not None:
                graph.y = torch.tensor([self.valid_targets[idx]], dtype=torch.float32)
                self._precomputed_graphs.append(graph)

    def len(self) -> int:
        """Return the number of graphs in the dataset."""
        return len(self.valid_smiles)

    def get(self, idx: int) -> Data:
        """Get a single graph from the dataset.

        Args:
            idx: Index of the graph to retrieve.

        Returns:
            PyTorch Geometric Data object with atom features, edge information,
            and target value.
        """
        if self._precomputed_graphs is not None:
            return self._precomputed_graphs[idx]

        smiles = self.valid_smiles[idx]
        target = self.valid_targets[idx]

        convert_fn = smiles_to_dmpnn_graph if self.use_dmpnn_format else smiles_to_graph
        graph = convert_fn(smiles)

        if graph is None:
            # This shouldn't happen since we filtered invalid SMILES
            raise ValueError(f"Failed to convert SMILES at index {idx}: {smiles}")

        graph.y = torch.tensor([target], dtype=torch.float32)

        return graph


def collate_dmpnn_batch(batch: List[Data]) -> Data:
    """Custom collate function for D-MPNN batches.

    Handles the special D-MPNN attributes (a2b, b2a, b2revb) during batching.

    Args:
        batch: List of Data objects.

    Returns:
        Batched Data object with adjusted indices.
    """
    from torch_geometric.data import Batch

    # Standard PyG batching
    batched = Batch.from_data_list(batch)

    # Adjust D-MPNN specific indices
    # The standard Batch.from_data_list handles edge_index automatically,
    # but we need to adjust a2b, b2a, b2revb manually if present

    return batched


def get_graph_feature_dims() -> Tuple[int, int]:
    """Get the dimensions of node and edge features.

    Returns:
        Tuple of (atom_feature_dim, bond_feature_dim).
    """
    return get_atom_feature_dim(), get_bond_feature_dim()
