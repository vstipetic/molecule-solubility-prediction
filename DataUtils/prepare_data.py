"""Script to prepare and split datasets for reproducible experiments.

This script takes a raw dataset CSV file and creates train/val/test CSV files
using scaffold splitting for proper data separation.

Usage:
    python -m DataUtils.prepare_data --input data/aqsoldb.csv --output-dir data/splits/
"""

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from DataUtils.utils import scaffold_split, smiles_to_mol


def filter_valid_smiles(
    df: pd.DataFrame,
    smiles_column: str,
    target_column: str,
) -> pd.DataFrame:
    """Filter dataframe to only include rows with valid SMILES and targets.

    Args:
        df: Input dataframe.
        smiles_column: Name of SMILES column.
        target_column: Name of target column.

    Returns:
        Filtered dataframe with only valid rows.
    """
    valid_indices = []

    for idx in range(len(df)):
        smiles = df.iloc[idx][smiles_column]
        target = df.iloc[idx][target_column]

        mol = smiles_to_mol(smiles)
        if mol is not None and pd.notna(target):
            valid_indices.append(idx)

    return df.iloc[valid_indices].copy()


def prepare_splits(
    input_path: str,
    output_dir: str,
    smiles_column: str = "SMILES",
    target_column: str = "Solubility",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    calib_ratio: float = 0.0,
    random_state: int = 42,
    standardize_columns: bool = True,
) -> None:
    """Prepare train/val/(calib)/test CSV files from input dataset.

    Creates reproducible splits using scaffold-based splitting to prevent
    data leakage between sets. When ``calib_ratio`` is greater than zero, an
    additional ``calib.csv`` is written for conformal calibration.

    Args:
        input_path: Path to input CSV file.
        output_dir: Directory to save output CSV files.
        smiles_column: Name of SMILES column in input.
        target_column: Name of target column in input.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        calib_ratio: Fraction for the conformal calibration set. If 0.0
            (default), no calib.csv is written.
        random_state: Random seed for reproducibility.
        standardize_columns: If True, output columns are renamed to
                           'SMILES' and 'Solubility'.
    """
    # Load data
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")

    # Filter valid SMILES
    df = filter_valid_smiles(df, smiles_column, target_column)
    print(f"After filtering invalid SMILES: {len(df)} valid molecules")

    # Extract data
    smiles_list = df[smiles_column].tolist()
    targets = df[target_column].values.astype(np.float32)

    # Scaffold split
    print(
        f"Performing scaffold split "
        f"(train={train_ratio}, val={val_ratio}, calib={calib_ratio})..."
    )
    splits = scaffold_split(
        smiles_list, targets, train_ratio, val_ratio, calib_ratio, random_state
    )

    if calib_ratio > 0.0:
        split_names = ["train", "val", "calib", "test"]
    else:
        split_names = ["train", "val", "test"]

    named_splits = list(zip(split_names, splits))

    size_summary = ", ".join(
        f"{name.capitalize()}={len(split_smiles)}"
        for name, (split_smiles, _) in named_splits
    )
    print(f"Split sizes: {size_summary}")

    # Determine output column names
    out_smiles_col = "SMILES" if standardize_columns else smiles_column
    out_target_col = "Solubility" if standardize_columns else target_column

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save splits
    for split_name, (split_smiles, split_targets) in named_splits:
        split_df = pd.DataFrame({
            out_smiles_col: split_smiles,
            out_target_col: split_targets,
        })
        output_path = output_dir / f"{split_name}.csv"
        split_df.to_csv(output_path, index=False)
        print(f"Saved {output_path} ({len(split_df)} samples)")

    # Save split metadata
    metadata = {
        'input_file': str(input_path),
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'calib_ratio': calib_ratio,
        'test_ratio': 1 - train_ratio - val_ratio - calib_ratio,
        'random_state': random_state,
        'total_valid': len(smiles_list),
        'smiles_column': out_smiles_col,
        'target_column': out_target_col,
    }
    for split_name, (split_smiles, _) in named_splits:
        metadata[f'{split_name}_size'] = len(split_smiles)

    metadata_path = output_dir / "split_metadata.txt"
    with open(metadata_path, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    print(f"Saved metadata to {metadata_path}")


def main() -> None:
    """Main entry point for the prepare_data script."""
    parser = argparse.ArgumentParser(
        description="Prepare dataset splits for reproducible experiments"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input CSV file path"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        required=True,
        help="Output directory for split CSV files"
    )
    parser.add_argument(
        "--smiles-column",
        type=str,
        default="SMILES",
        help="Name of SMILES column in input (default: SMILES)"
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="Solubility",
        help="Name of target column in input (default: Solubility)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction for training set (default: 0.8)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction for validation set (default: 0.1)"
    )
    parser.add_argument(
        "--calib-ratio",
        type=float,
        default=0.0,
        help="Fraction for conformal calibration set (default: 0.0, no calib split)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Keep original column names (don't rename to SMILES/Solubility)"
    )

    args = parser.parse_args()

    # Validate ratios
    if args.train_ratio + args.val_ratio + args.calib_ratio >= 1.0:
        parser.error("train_ratio + val_ratio + calib_ratio must be less than 1.0")

    prepare_splits(
        input_path=args.input,
        output_dir=args.output_dir,
        smiles_column=args.smiles_column,
        target_column=args.target_column,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        calib_ratio=args.calib_ratio,
        random_state=args.random_state,
        standardize_columns=not args.no_standardize,
    )

    print("\nDone! You can now use the split files for training:")
    print(f"  Train: {args.output_dir}/train.csv")
    print(f"  Val:   {args.output_dir}/val.csv")
    if args.calib_ratio > 0.0:
        print(f"  Calib: {args.output_dir}/calib.csv")
    print(f"  Test:  {args.output_dir}/test.csv")


if __name__ == "__main__":
    main()
