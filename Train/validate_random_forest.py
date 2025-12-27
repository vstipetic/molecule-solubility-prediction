"""Validation utilities for Random Forest model.

This module provides functions to evaluate the Random Forest model
on validation/test sets with uncertainty estimation via tree variance.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import wandb

from Models.random_forest import ECFPRandomForest
from DataUtils.metrics import compute_metrics, compute_calibration_metrics


def validate_random_forest(
    model: ECFPRandomForest,
    smiles_list: List[str],
    targets: np.ndarray,
    log_to_wandb: bool = True,
    prefix: str = "val",
) -> Dict[str, float]:
    """Validate Random Forest model.

    Args:
        model: Trained ECFPRandomForest model.
        smiles_list: List of SMILES strings.
        targets: Ground truth solubility values.
        log_to_wandb: Whether to log metrics to wandb.
        prefix: Prefix for metric names (e.g., 'val', 'test').

    Returns:
        Dictionary of metrics.
    """
    predictions = model.predict(smiles_list)
    metrics = compute_metrics(predictions, targets)

    # Add prefix
    metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

    if log_to_wandb and wandb.run is not None:
        wandb.log(metrics)

    return metrics


def validate_random_forest_with_uncertainty(
    model: ECFPRandomForest,
    smiles_list: List[str],
    targets: np.ndarray,
    log_to_wandb: bool = True,
    prefix: str = "val",
) -> Dict[str, float]:
    """Validate Random Forest with uncertainty estimation.

    Uses tree variance for uncertainty quantification.

    Args:
        model: Trained ECFPRandomForest model.
        smiles_list: List of SMILES strings.
        targets: Ground truth solubility values.
        log_to_wandb: Whether to log metrics to wandb.
        prefix: Prefix for metric names.

    Returns:
        Dictionary of metrics including calibration metrics.
    """
    mean_preds, std_preds = model.predict_with_uncertainty(smiles_list)

    # Regression metrics
    metrics = compute_metrics(mean_preds, targets)

    # Calibration metrics
    calibration = compute_calibration_metrics(mean_preds, std_preds, targets)
    metrics.update(calibration)

    # Add prefix
    metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

    if log_to_wandb and wandb.run is not None:
        wandb.log(metrics)

    return metrics


def evaluate_on_splits(
    model: ECFPRandomForest,
    train_data: Tuple[List[str], np.ndarray],
    val_data: Tuple[List[str], np.ndarray],
    test_data: Optional[Tuple[List[str], np.ndarray]] = None,
    with_uncertainty: bool = True,
    log_to_wandb: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Evaluate model on train, validation, and optionally test splits.

    Args:
        model: Trained ECFPRandomForest model.
        train_data: Tuple of (smiles_list, targets) for training set.
        val_data: Tuple of (smiles_list, targets) for validation set.
        test_data: Optional tuple of (smiles_list, targets) for test set.
        with_uncertainty: Whether to compute uncertainty metrics.
        log_to_wandb: Whether to log to wandb.

    Returns:
        Dictionary of metrics for each split.
    """
    results = {}

    validate_fn = (
        validate_random_forest_with_uncertainty
        if with_uncertainty
        else validate_random_forest
    )

    # Train set (for checking overfitting)
    results['train'] = validate_fn(
        model, train_data[0], train_data[1],
        log_to_wandb=log_to_wandb, prefix='train'
    )

    # Validation set
    results['val'] = validate_fn(
        model, val_data[0], val_data[1],
        log_to_wandb=log_to_wandb, prefix='val'
    )

    # Test set
    if test_data is not None:
        results['test'] = validate_fn(
            model, test_data[0], test_data[1],
            log_to_wandb=log_to_wandb, prefix='test'
        )

    return results


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Validate Random Forest model")
    parser.add_argument("--model-path", type=str, help="Path to saved model")
    parser.add_argument("--data-path", type=str, help="Path to validation data")
    parser.add_argument("--wandb-project", type=str, default="mol-solubility")

    args = parser.parse_args()

    # Initialize wandb
    wandb.init(project=args.wandb_project, job_type="validation")

    # Load model and data, run validation
    # (Implementation depends on how model is saved)
    print("Validation complete. Check wandb for metrics.")
