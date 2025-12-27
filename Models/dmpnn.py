"""Directed Message Passing Neural Network (D-MPNN) for molecular property prediction.

This module implements the D-MPNN architecture from Chemprop, which performs
message passing on directed edges of molecular graphs.

Reference:
    Yang et al. "Analyzing Learned Molecular Representations for Property Prediction"
    Journal of Chemical Information and Modeling, 2019.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_add_pool, global_mean_pool

from Models.layers import MCDropout, MCDropoutMixin
from DataUtils.utils import get_atom_feature_dim, get_bond_feature_dim


class DMPNNEncoder(nn.Module):
    """D-MPNN encoder that converts molecular graphs to fixed-size embeddings.

    Messages are passed along directed edges (bonds) and aggregated at atoms.
    The final molecule embedding is obtained by summing atom hidden states.

    Args:
        atom_fdim: Dimension of atom features.
        bond_fdim: Dimension of bond features.
        hidden_size: Size of hidden layers.
        depth: Number of message passing iterations.
        dropout: Dropout probability.
        aggregation: Aggregation method ('sum' or 'mean').

    Attributes:
        W_i: Initial message transformation.
        W_h: Message update transformation.
        W_o: Output transformation.
    """

    def __init__(
        self,
        atom_fdim: int,
        bond_fdim: int,
        hidden_size: int = 300,
        depth: int = 3,
        dropout: float = 0.1,
        aggregation: str = "sum",
    ) -> None:
        super().__init__()

        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = hidden_size
        self.depth = depth
        self.aggregation = aggregation

        # Initial message from bond features
        self.W_i = nn.Linear(atom_fdim + bond_fdim, hidden_size, bias=False)

        # Message update
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)

        # Output transformation (atom hidden state)
        self.W_o = nn.Linear(atom_fdim + hidden_size, hidden_size)

        # Activation and dropout
        self.act = nn.ReLU()
        self.dropout = MCDropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Encode molecular graphs into fixed-size embeddings.

        Args:
            x: Atom features [num_atoms, atom_fdim].
            edge_index: Edge connectivity [2, num_edges].
            edge_attr: Edge features [num_edges, bond_fdim].
            batch: Batch assignment [num_atoms].

        Returns:
            Molecule embeddings [batch_size, hidden_size].
        """
        # Get source and target atom features for each edge
        source_idx = edge_index[0]
        target_idx = edge_index[1]

        # Initialize messages from atom + bond features
        source_features = x[source_idx]
        init_messages = torch.cat([source_features, edge_attr], dim=-1)
        messages = self.act(self.W_i(init_messages))

        # Message passing iterations
        for _ in range(self.depth - 1):
            # Aggregate incoming messages for each edge's source atom
            # We need messages coming into the source atom (excluding the reverse edge)
            nei_messages = self._aggregate_messages(
                messages, edge_index, x.size(0)
            )

            # Get aggregated messages for source atoms of each edge
            source_nei_messages = nei_messages[source_idx]

            # Update messages
            messages = self.act(messages + self.W_h(source_nei_messages))
            messages = self.dropout(messages)

        # Aggregate messages to atoms
        atom_messages = self._aggregate_to_atoms(
            messages, edge_index, x.size(0)
        )

        # Final atom hidden states
        atom_hiddens = self.act(self.W_o(torch.cat([x, atom_messages], dim=-1)))
        atom_hiddens = self.dropout(atom_hiddens)

        # Aggregate to molecule level
        if self.aggregation == "sum":
            mol_embeddings = global_add_pool(atom_hiddens, batch)
        else:
            mol_embeddings = global_mean_pool(atom_hiddens, batch)

        return mol_embeddings

    def _aggregate_messages(
        self,
        messages: torch.Tensor,
        edge_index: torch.Tensor,
        num_atoms: int,
    ) -> torch.Tensor:
        """Aggregate messages to atoms.

        Args:
            messages: Edge messages [num_edges, hidden_size].
            edge_index: Edge connectivity [2, num_edges].
            num_atoms: Number of atoms.

        Returns:
            Aggregated messages per atom [num_atoms, hidden_size].
        """
        target_idx = edge_index[1]

        # Sum messages arriving at each atom
        aggregated = torch.zeros(
            num_atoms, messages.size(-1),
            device=messages.device, dtype=messages.dtype
        )
        aggregated.scatter_add_(0, target_idx.unsqueeze(-1).expand_as(messages), messages)

        return aggregated

    def _aggregate_to_atoms(
        self,
        messages: torch.Tensor,
        edge_index: torch.Tensor,
        num_atoms: int,
    ) -> torch.Tensor:
        """Aggregate final messages to atoms for output.

        Args:
            messages: Edge messages [num_edges, hidden_size].
            edge_index: Edge connectivity [2, num_edges].
            num_atoms: Number of atoms.

        Returns:
            Atom-level message aggregation [num_atoms, hidden_size].
        """
        target_idx = edge_index[1]

        aggregated = torch.zeros(
            num_atoms, messages.size(-1),
            device=messages.device, dtype=messages.dtype
        )
        aggregated.scatter_add_(0, target_idx.unsqueeze(-1).expand_as(messages), messages)

        return aggregated


class DMPNN(nn.Module, MCDropoutMixin):
    """Complete D-MPNN model for molecular property prediction.

    Combines the D-MPNN encoder with a feedforward neural network
    for property prediction.

    Args:
        atom_fdim: Dimension of atom features. If None, uses default.
        bond_fdim: Dimension of bond features. If None, uses default.
        hidden_size: Size of hidden layers in encoder.
        depth: Number of message passing iterations.
        ffn_hidden_size: Size of hidden layers in FFN.
        ffn_num_layers: Number of layers in FFN.
        dropout: Dropout probability.
        aggregation: Aggregation method ('sum' or 'mean').

    Example:
        >>> model = DMPNN()
        >>> data = smiles_to_graph("CCO")
        >>> batch = Batch.from_data_list([data])
        >>> output = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    """

    def __init__(
        self,
        atom_fdim: Optional[int] = None,
        bond_fdim: Optional[int] = None,
        hidden_size: int = 300,
        depth: int = 3,
        ffn_hidden_size: int = 300,
        ffn_num_layers: int = 2,
        dropout: float = 0.1,
        aggregation: str = "sum",
    ) -> None:
        super().__init__()

        if atom_fdim is None:
            atom_fdim = get_atom_feature_dim()
        if bond_fdim is None:
            bond_fdim = get_bond_feature_dim()

        self.encoder = DMPNNEncoder(
            atom_fdim=atom_fdim,
            bond_fdim=bond_fdim,
            hidden_size=hidden_size,
            depth=depth,
            dropout=dropout,
            aggregation=aggregation,
        )

        # Build FFN
        ffn_layers: List[nn.Module] = []

        input_size = hidden_size
        for _ in range(ffn_num_layers - 1):
            ffn_layers.extend([
                nn.Linear(input_size, ffn_hidden_size),
                nn.ReLU(),
                MCDropout(dropout),
            ])
            input_size = ffn_hidden_size

        # Final output layer
        ffn_layers.append(nn.Linear(input_size, 1))

        self.ffn = nn.Sequential(*ffn_layers)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Predict molecular property.

        Args:
            x: Atom features [num_atoms, atom_fdim].
            edge_index: Edge connectivity [2, num_edges].
            edge_attr: Edge features [num_edges, bond_fdim].
            batch: Batch assignment [num_atoms].

        Returns:
            Predictions [batch_size, 1].
        """
        mol_embeddings = self.encoder(x, edge_index, edge_attr, batch)
        output = self.ffn(mol_embeddings)
        return output

    def forward_from_data(self, data: Data) -> torch.Tensor:
        """Predict from a PyTorch Geometric Data object.

        Args:
            data: PyG Data object with x, edge_index, edge_attr, and batch.

        Returns:
            Predictions [batch_size, 1].
        """
        return self.forward(
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Get molecule embeddings without final prediction.

        Args:
            x: Atom features.
            edge_index: Edge connectivity.
            edge_attr: Edge features.
            batch: Batch assignment.

        Returns:
            Molecule embeddings [batch_size, hidden_size].
        """
        return self.encoder(x, edge_index, edge_attr, batch)


def create_dmpnn_ensemble(
    n_models: int = 5,
    **kwargs,
) -> List[DMPNN]:
    """Create an ensemble of D-MPNN models for deep ensemble uncertainty.

    Args:
        n_models: Number of models in the ensemble.
        **kwargs: Arguments passed to DMPNN constructor.

    Returns:
        List of DMPNN models.
    """
    models = []
    for _ in range(n_models):
        model = DMPNN(**kwargs)
        models.append(model)
    return models


def ensemble_predict(
    models: List[DMPNN],
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    batch: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Make predictions with an ensemble of D-MPNN models.

    Args:
        models: List of trained DMPNN models.
        x: Atom features.
        edge_index: Edge connectivity.
        edge_attr: Edge features.
        batch: Batch assignment.

    Returns:
        Tuple of (mean_predictions, std_predictions).
    """
    predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            pred = model(x, edge_index, edge_attr, batch)
            predictions.append(pred)

    predictions_tensor = torch.stack(predictions, dim=0)
    mean = predictions_tensor.mean(dim=0)
    std = predictions_tensor.std(dim=0)

    return mean, std
