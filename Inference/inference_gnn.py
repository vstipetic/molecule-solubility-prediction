"""Inference script for D-MPNN (GNN) model.

This script loads a trained D-MPNN model and runs inference on an input
dataset. Predictions are saved in the standard dataset format (SMILES, Solubility).

Output columns:
- SMILES: Input molecular structure
- Solubility: Predicted solubility (log mol/L)
- Uncertainty: Standard deviation across MC dropout samples (same units as Solubility)
"""

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

from DataUtils.graph_preprocessing import smiles_to_dmpnn_graph
from DataUtils.utils import smiles_to_mol
from Models.dmpnn import DMPNN


class InferenceGraphDataset(torch.utils.data.Dataset):
    """Simple dataset for inference that tracks valid indices."""

    def __init__(self, smiles_list: List[str]) -> None:
        self.smiles_list = smiles_list
        self.valid_indices: List[int] = []
        self.graphs: List = []

        for idx, smiles in enumerate(smiles_list):
            mol = smiles_to_mol(smiles)
            if mol is not None:
                graph = smiles_to_dmpnn_graph(smiles)
                if graph is not None:
                    # Add dummy target
                    graph.y = torch.tensor([0.0], dtype=torch.float32)
                    self.graphs.append(graph)
                    self.valid_indices.append(idx)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int):
        return self.graphs[idx]


def run_inference_gnn(
    model_path: str,
    input_path: str,
    output_path: str,
    smiles_column: str = "SMILES",
    hidden_size: int = 300,
    depth: int = 3,
    ffn_hidden_size: int = 300,
    ffn_num_layers: int = 2,
    dropout: float = 0.1,
    batch_size: int = 32,
    include_uncertainty: bool = True,
    n_mc_samples: int = 100,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """Run inference with a trained D-MPNN model.

    Args:
        model_path: Path to the trained model (state_dict .pt file).
        input_path: Path to input CSV with SMILES column.
        output_path: Path to save predictions CSV.
        smiles_column: Name of the SMILES column in input CSV.
        hidden_size: Hidden size used in training.
        depth: Message passing depth used in training.
        ffn_hidden_size: FFN hidden size used in training.
        ffn_num_layers: Number of FFN layers used in training.
        dropout: Dropout used in training.
        batch_size: Batch size for inference.
        include_uncertainty: Whether to include MC dropout uncertainty.
        n_mc_samples: Number of MC dropout samples.
        device: Device to use (None for auto-detect).

    Returns:
        DataFrame with predictions.
    """
    # Device setup
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {model_path}...")
    model = DMPNN(
        hidden_size=hidden_size,
        depth=depth,
        ffn_hidden_size=ffn_hidden_size,
        ffn_num_layers=ffn_num_layers,
        dropout=dropout,
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Load input data
    print(f"Loading input data from {input_path}...")
    df = pd.read_csv(input_path)

    if smiles_column not in df.columns:
        raise ValueError(f"Column '{smiles_column}' not found in input CSV")

    smiles_list = df[smiles_column].tolist()
    print(f"Running inference on {len(smiles_list)} molecules...")

    # Create graph dataset that tracks valid indices
    dataset = InferenceGraphDataset(smiles_list)
    dataloader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Run inference
    all_predictions: List[float] = []
    all_uncertainties: List[float] = []

    if include_uncertainty:
        # MC dropout inference
        model.enable_mc_dropout()

        for batch in dataloader:
            batch = batch.to(device)
            batch_preds = []

            with torch.no_grad():
                for _ in range(n_mc_samples):
                    output = model(
                        batch.x, batch.edge_index, batch.edge_attr, batch.batch
                    )
                    batch_preds.append(output.squeeze(-1).cpu().numpy())

            batch_preds = np.stack(batch_preds, axis=0)
            mean_preds = batch_preds.mean(axis=0)
            std_preds = batch_preds.std(axis=0)

            all_predictions.extend(mean_preds.tolist())
            all_uncertainties.extend(std_preds.tolist())

        model.disable_mc_dropout()
    else:
        # Standard inference
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(device)
                output = model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch
                )
                all_predictions.extend(output.squeeze(-1).cpu().numpy().tolist())

    # Map predictions back to original indices
    predictions = np.full(len(smiles_list), np.nan)
    uncertainties = np.full(len(smiles_list), np.nan) if include_uncertainty else None

    for idx, valid_idx in enumerate(dataset.valid_indices):
        if idx < len(all_predictions):
            predictions[valid_idx] = all_predictions[idx]
            if include_uncertainty and uncertainties is not None:
                uncertainties[valid_idx] = all_uncertainties[idx]

    # Create output DataFrame
    output_df = pd.DataFrame({
        'SMILES': smiles_list,
        'Solubility': predictions,
    })

    if include_uncertainty:
        output_df['Uncertainty'] = uncertainties

    # Save predictions
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

    # Print summary
    valid_preds = predictions[~np.isnan(predictions)]
    print(f"\nSummary:")
    print(f"  Total molecules: {len(smiles_list)}")
    print(f"  Valid predictions: {len(valid_preds)}")
    print(f"  Failed molecules: {len(smiles_list) - len(valid_preds)}")
    if len(valid_preds) > 0:
        print(f"  Prediction range: [{valid_preds.min():.3f}, {valid_preds.max():.3f}]")
        print(f"  Mean prediction: {valid_preds.mean():.3f}")

    return output_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference with trained D-MPNN model"
    )
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to trained model (.pt file)"
    )
    parser.add_argument(
        "--input-path", type=str, required=True,
        help="Path to input CSV with SMILES"
    )
    parser.add_argument(
        "--output-path", type=str, required=True,
        help="Path to save predictions CSV"
    )
    parser.add_argument(
        "--smiles-column", type=str, default="SMILES",
        help="Name of SMILES column in input CSV"
    )
    parser.add_argument(
        "--hidden-size", type=int, default=300,
        help="Hidden size (must match training)"
    )
    parser.add_argument(
        "--depth", type=int, default=3,
        help="Message passing depth (must match training)"
    )
    parser.add_argument(
        "--ffn-hidden-size", type=int, default=300,
        help="FFN hidden size (must match training)"
    )
    parser.add_argument(
        "--ffn-num-layers", type=int, default=2,
        help="Number of FFN layers (must match training)"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1,
        help="Dropout rate (must match training)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--no-uncertainty", action="store_true",
        help="Disable MC dropout uncertainty estimation"
    )
    parser.add_argument(
        "--n-mc-samples", type=int, default=100,
        help="Number of MC dropout samples"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use (cuda/cpu)"
    )

    args = parser.parse_args()

    run_inference_gnn(
        model_path=args.model_path,
        input_path=args.input_path,
        output_path=args.output_path,
        smiles_column=args.smiles_column,
        hidden_size=args.hidden_size,
        depth=args.depth,
        ffn_hidden_size=args.ffn_hidden_size,
        ffn_num_layers=args.ffn_num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        include_uncertainty=not args.no_uncertainty,
        n_mc_samples=args.n_mc_samples,
        device=args.device,
    )


if __name__ == "__main__":
    main()
