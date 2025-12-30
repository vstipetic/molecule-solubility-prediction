"""Inference script for Random Forest model.

This script loads a trained Random Forest model and runs inference on an input
dataset. Predictions are saved in the standard dataset format (SMILES, Solubility).

Output columns:
- SMILES: Input molecular structure
- Solubility: Predicted solubility (log mol/L)
- Uncertainty: Standard deviation across tree predictions (same units as Solubility)
"""

import argparse
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from Models.random_forest import ECFPRandomForest


def run_inference_random_forest(
    model_path: str,
    input_path: str,
    output_path: str,
    smiles_column: str = "SMILES",
    include_uncertainty: bool = True,
) -> pd.DataFrame:
    """Run inference with a trained Random Forest model.

    Args:
        model_path: Path to the trained model (pickle file).
        input_path: Path to input CSV with SMILES column.
        output_path: Path to save predictions CSV.
        smiles_column: Name of the SMILES column in input CSV.
        include_uncertainty: Whether to include uncertainty estimates.

    Returns:
        DataFrame with predictions.
    """
    # Load model
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model: ECFPRandomForest = pickle.load(f)

    if not model.is_fitted:
        raise RuntimeError("Model has not been trained")

    # Load input data
    print(f"Loading input data from {input_path}...")
    df = pd.read_csv(input_path)

    if smiles_column not in df.columns:
        raise ValueError(f"Column '{smiles_column}' not found in input CSV")

    smiles_list = df[smiles_column].tolist()
    print(f"Running inference on {len(smiles_list)} molecules...")

    # Run inference
    if include_uncertainty:
        predictions, uncertainties = model.predict_with_uncertainty(smiles_list)
    else:
        predictions = model.predict(smiles_list)
        uncertainties = None

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
        description="Run inference with trained Random Forest model"
    )
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to trained model (pickle file)"
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
        "--no-uncertainty", action="store_true",
        help="Disable uncertainty estimation"
    )

    args = parser.parse_args()

    run_inference_random_forest(
        model_path=args.model_path,
        input_path=args.input_path,
        output_path=args.output_path,
        smiles_column=args.smiles_column,
        include_uncertainty=not args.no_uncertainty,
    )


if __name__ == "__main__":
    main()
