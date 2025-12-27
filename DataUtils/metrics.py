"""Metrics functions for model evaluation.

This module provides shared metrics computation for regression tasks
with uncertainty quantification.
"""

from typing import Dict

import numpy as np


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    filter_nan: bool = True,
) -> Dict[str, float]:
    """Compute regression metrics (RMSE, MAE).

    Args:
        predictions: Model predictions.
        targets: Ground truth values.
        filter_nan: Whether to filter out NaN predictions.

    Returns:
        Dictionary with 'rmse' and 'mae' metrics.
    """
    if filter_nan:
        valid_mask = ~np.isnan(predictions)
        preds = predictions[valid_mask]
        targs = targets[valid_mask]
    else:
        preds = predictions
        targs = targets

    if len(preds) == 0:
        return {'rmse': float('inf'), 'mae': float('inf')}

    rmse = float(np.sqrt(np.mean((preds - targs) ** 2)))
    mae = float(np.mean(np.abs(preds - targs)))

    return {'rmse': rmse, 'mae': mae}


def compute_calibration_metrics(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    targets: np.ndarray,
    filter_nan: bool = True,
) -> Dict[str, float]:
    """Compute uncertainty calibration metrics.

    Computes coverage at different confidence levels and negative log-likelihood.
    For well-calibrated predictions, coverage should match the confidence level.

    Args:
        predictions: Mean predictions.
        uncertainties: Standard deviation estimates.
        targets: Ground truth values.
        filter_nan: Whether to filter out NaN predictions.

    Returns:
        Dictionary with calibration metrics:
        - coverage_50: Fraction within 50% CI
        - coverage_90: Fraction within 90% CI
        - coverage_95: Fraction within 95% CI
        - nll: Negative log-likelihood (Gaussian)
    """
    if filter_nan:
        valid_mask = ~np.isnan(predictions)
        preds = predictions[valid_mask]
        stds = uncertainties[valid_mask]
        targs = targets[valid_mask]
    else:
        preds = predictions
        stds = uncertainties
        targs = targets

    if len(preds) == 0:
        return {}

    # Normalized error (should be ~N(0,1) if well-calibrated)
    normalized_errors = (targs - preds) / (stds + 1e-8)

    # Calibration: percentage of points within different confidence intervals
    z_scores = {0.5: 0.674, 0.9: 1.645, 0.95: 1.96}
    calibration = {}

    for confidence, z_score in z_scores.items():
        within = np.abs(normalized_errors) <= z_score
        calibration[f'coverage_{int(confidence * 100)}'] = float(within.mean())

    # Negative log-likelihood (assuming Gaussian)
    nll = 0.5 * np.mean(
        np.log(2 * np.pi * stds ** 2 + 1e-8)
        + (targs - preds) ** 2 / (stds ** 2 + 1e-8)
    )
    calibration['nll'] = float(nll)

    return calibration
