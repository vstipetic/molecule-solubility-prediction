"""Training script for Random Forest model with ECFP fingerprints.

This script trains a Random Forest model on molecular solubility data
with wandb logging and hyperparameter tuning support.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import wandb

from DataUtils.datasets import AqSolDBDataset
from DataUtils.utils import scaffold_split
from Models.random_forest import ECFPRandomForest
from Train.validate_random_forest import (
    validate_random_forest,
    validate_random_forest_with_uncertainty,
)


def load_data(
    data_path: str,
    smiles_column: str = "SMILES",
    target_column: str = "Solubility",
) -> Tuple[List[str], np.ndarray]:
    """Load data from CSV file.

    Args:
        data_path: Path to CSV file.
        smiles_column: Name of SMILES column.
        target_column: Name of target column.

    Returns:
        Tuple of (smiles_list, targets).
    """
    df = pd.read_csv(data_path)
    smiles_list = df[smiles_column].tolist()
    targets = df[target_column].values.astype(np.float32)
    return smiles_list, targets


def train_random_forest(
    train_smiles: List[str],
    train_targets: np.ndarray,
    val_smiles: List[str],
    val_targets: np.ndarray,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    fingerprint_radius: int = 2,
    fingerprint_bits: int = 2048,
    log_to_wandb: bool = True,
) -> ECFPRandomForest:
    """Train Random Forest model.

    Args:
        train_smiles: Training SMILES strings.
        train_targets: Training targets.
        val_smiles: Validation SMILES strings.
        val_targets: Validation targets.
        n_estimators: Number of trees.
        max_depth: Maximum tree depth.
        min_samples_split: Minimum samples for split.
        min_samples_leaf: Minimum samples per leaf.
        fingerprint_radius: ECFP radius.
        fingerprint_bits: ECFP bits.
        log_to_wandb: Whether to log to wandb.

    Returns:
        Trained ECFPRandomForest model.
    """
    # Create model
    model = ECFPRandomForest(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        fingerprint_radius=fingerprint_radius,
        fingerprint_bits=fingerprint_bits,
    )

    # Log hyperparameters
    if log_to_wandb and wandb.run is not None:
        wandb.config.update(model.get_params())

    # Train model
    print("Training Random Forest...")
    model.fit(train_smiles, train_targets)
    print("Training complete.")

    # Validate
    print("Validating...")
    train_metrics = validate_random_forest_with_uncertainty(
        model, train_smiles, train_targets,
        log_to_wandb=log_to_wandb, prefix='train'
    )
    val_metrics = validate_random_forest_with_uncertainty(
        model, val_smiles, val_targets,
        log_to_wandb=log_to_wandb, prefix='val'
    )

    print(f"Train RMSE: {train_metrics['train/rmse']:.4f}")
    print(f"Val RMSE: {val_metrics['val/rmse']:.4f}")

    return model


def hyperparameter_search(
    train_smiles: List[str],
    train_targets: np.ndarray,
    val_smiles: List[str],
    val_targets: np.ndarray,
    n_trials: int = 20,
    log_to_wandb: bool = True,
) -> Dict:
    """Random hyperparameter search for Random Forest.

    Args:
        train_smiles: Training SMILES strings.
        train_targets: Training targets.
        val_smiles: Validation SMILES strings.
        val_targets: Validation targets.
        n_trials: Number of hyperparameter configurations to try.
        log_to_wandb: Whether to log to wandb.

    Returns:
        Best hyperparameters dictionary.
    """
    np.random.seed(42)

    best_rmse = float('inf')
    best_params = {}

    for trial in range(n_trials):
        # Sample hyperparameters
        params = {
            'n_estimators': int(np.random.choice([50, 100, 200, 500])),
            'max_depth': np.random.choice([None, 10, 20, 30, 50]),
            'min_samples_split': int(np.random.choice([2, 5, 10])),
            'min_samples_leaf': int(np.random.choice([1, 2, 4])),
            'fingerprint_radius': int(np.random.choice([2, 3])),
            'fingerprint_bits': int(np.random.choice([1024, 2048, 4096])),
        }

        print(f"\nTrial {trial + 1}/{n_trials}")
        print(f"Params: {params}")

        # Train and evaluate
        model = ECFPRandomForest(**params)
        model.fit(train_smiles, train_targets)

        val_metrics = validate_random_forest(
            model, val_smiles, val_targets,
            log_to_wandb=False, prefix='val'
        )

        rmse = val_metrics['val/rmse']
        print(f"Val RMSE: {rmse:.4f}")

        if log_to_wandb and wandb.run is not None:
            wandb.log({
                'trial': trial,
                'trial/val_rmse': rmse,
                **{f'trial/{k}': v for k, v in params.items() if v is not None},
            })

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params.copy()

    print(f"\nBest RMSE: {best_rmse:.4f}")
    print(f"Best params: {best_params}")

    if log_to_wandb and wandb.run is not None:
        wandb.log({
            'best/val_rmse': best_rmse,
            **{f'best/{k}': v for k, v in best_params.items() if v is not None},
        })

    return best_params


def main():
    parser = argparse.ArgumentParser(description="Train Random Forest model")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to training data CSV")
    parser.add_argument("--smiles-column", type=str, default="SMILES")
    parser.add_argument("--target-column", type=str, default="Solubility")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--fingerprint-radius", type=int, default=2)
    parser.add_argument("--fingerprint-bits", type=int, default=2048)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--hyperparam-search", action="store_true",
                        help="Run hyperparameter search")
    parser.add_argument("--n-trials", type=int, default=20,
                        help="Number of trials for hyperparameter search")
    parser.add_argument("--save-path", type=str, default=None,
                        help="Path to save trained model")
    parser.add_argument("--wandb-project", type=str, default="mol-solubility")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")

    args = parser.parse_args()

    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
            job_type="train",
        )

    # Load data
    print(f"Loading data from {args.data_path}...")
    smiles_list, targets = load_data(
        args.data_path,
        args.smiles_column,
        args.target_column,
    )
    print(f"Loaded {len(smiles_list)} molecules")

    # Scaffold split
    print("Performing scaffold split...")
    (train_smiles, train_targets), (val_smiles, val_targets), (test_smiles, test_targets) = \
        scaffold_split(smiles_list, targets, args.train_ratio, args.val_ratio)

    print(f"Train: {len(train_smiles)}, Val: {len(val_smiles)}, Test: {len(test_smiles)}")

    if args.hyperparam_search:
        # Hyperparameter search
        best_params = hyperparameter_search(
            train_smiles, train_targets,
            val_smiles, val_targets,
            n_trials=args.n_trials,
            log_to_wandb=not args.no_wandb,
        )

        # Train final model with best params
        model = train_random_forest(
            train_smiles, train_targets,
            val_smiles, val_targets,
            log_to_wandb=not args.no_wandb,
            **best_params,
        )
    else:
        # Train with specified params
        model = train_random_forest(
            train_smiles, train_targets,
            val_smiles, val_targets,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            fingerprint_radius=args.fingerprint_radius,
            fingerprint_bits=args.fingerprint_bits,
            log_to_wandb=not args.no_wandb,
        )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = validate_random_forest_with_uncertainty(
        model, test_smiles, test_targets,
        log_to_wandb=not args.no_wandb, prefix='test'
    )
    print(f"Test RMSE: {test_metrics['test/rmse']:.4f}")
    print(f"Test MAE: {test_metrics['test/mae']:.4f}")

    # Save model
    if args.save_path:
        import pickle
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {save_path}")

        if not args.no_wandb and wandb.run is not None:
            wandb.save(str(save_path))

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
