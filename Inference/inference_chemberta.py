"""Inference script for ChemBERTa model.

This script loads a fine-tuned ChemBERTa model and runs inference on an input
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
from torch.utils.data import DataLoader

from DataUtils.datasets import TransformerSolubilityDataset
from DataUtils.collate import transformer_collate_fn
from Models.chemberta import ChemBERTaForSolubility


def run_inference_chemberta(
    model_path: str,
    input_path: str,
    output_path: str,
    smiles_column: str = "SMILES",
    batch_size: int = 32,
    max_length: int = 512,
    include_uncertainty: bool = True,
    n_mc_samples: int = 100,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """Run inference with a fine-tuned ChemBERTa model.

    Args:
        model_path: Path to the fine-tuned model weights.
        input_path: Path to input CSV with SMILES column.
        output_path: Path to save predictions CSV.
        smiles_column: Name of the SMILES column in input CSV.
        batch_size: Batch size for inference.
        max_length: Maximum sequence length.
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
    model = ChemBERTaForSolubility.from_pretrained(model_path)
    model = model.to(device)
    model.eval()

    # Get tokenizer from model
    tokenizer = model.tokenizer

    # Load input data
    print(f"Loading input data from {input_path}...")
    df = pd.read_csv(input_path)

    if smiles_column not in df.columns:
        raise ValueError(f"Column '{smiles_column}' not found in input CSV")

    smiles_list = df[smiles_column].tolist()
    print(f"Running inference on {len(smiles_list)} molecules...")

    # Create dummy targets for dataset
    dummy_targets = np.zeros(len(smiles_list))

    # Create dataset and dataloader
    dataset = TransformerSolubilityDataset(
        smiles_list, dummy_targets, tokenizer, max_length, is_huggingface=True
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=transformer_collate_fn
    )

    # Run inference
    all_predictions: List[float] = []
    all_uncertainties: List[float] = []

    if include_uncertainty:
        # MC dropout inference
        model.enable_mc_dropout()

        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_preds = []

            with torch.no_grad():
                for _ in range(n_mc_samples):
                    outputs = model(input_ids, attention_mask)
                    predictions = outputs['logits'].squeeze(-1)
                    batch_preds.append(predictions.cpu().numpy())

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
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                outputs = model(input_ids, attention_mask)
                predictions = outputs['logits'].squeeze(-1)
                all_predictions.extend(predictions.cpu().numpy().tolist())

    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    uncertainties = np.array(all_uncertainties) if include_uncertainty else None

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
    print(f"\nSummary:")
    print(f"  Total molecules: {len(smiles_list)}")
    print(f"  Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    print(f"  Mean prediction: {predictions.mean():.3f}")
    if include_uncertainty and uncertainties is not None:
        print(f"  Mean uncertainty: {uncertainties.mean():.3f}")

    return output_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference with fine-tuned ChemBERTa model"
    )
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to fine-tuned model weights"
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
        "--batch-size", type=int, default=32,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--max-length", type=int, default=512,
        help="Maximum sequence length"
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

    run_inference_chemberta(
        model_path=args.model_path,
        input_path=args.input_path,
        output_path=args.output_path,
        smiles_column=args.smiles_column,
        batch_size=args.batch_size,
        max_length=args.max_length,
        include_uncertainty=not args.no_uncertainty,
        n_mc_samples=args.n_mc_samples,
        device=args.device,
    )


if __name__ == "__main__":
    main()
