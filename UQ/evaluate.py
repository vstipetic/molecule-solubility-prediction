"""Conformal evaluation helpers.

Bridge between a calibrated :class:`UQ.conformal.ConformalRegressor` and the
shared metric utilities in :mod:`DataUtils.metrics`, so validation scripts can
report empirical coverage and interval widths alongside the existing
calibration metrics in a single call.
"""

from typing import Dict, Optional

import numpy as np

from DataUtils.metrics import compute_conformal_metrics
from UQ.conformal import ConformalRegressor


def evaluate_conformal_intervals(
    conformal: ConformalRegressor,
    predictions: np.ndarray,
    targets: np.ndarray,
    uncertainties: Optional[np.ndarray] = None,
    alpha: float = 0.1,
) -> Dict[str, float]:
    """Evaluate conformal prediction intervals against ground truth.

    Builds prediction intervals at confidence ``1 - alpha`` from the calibrated
    regressor and computes empirical coverage and interval widths. The
    resulting metric keys are namespaced by the confidence level (e.g.
    ``"coverage_90"``, ``"mean_interval_width_90"``) so multiple alpha levels
    can be logged without collision.

    Args:
        conformal: A calibrated :class:`ConformalRegressor`.
        predictions: Point predictions on the evaluation set.
        targets: Ground truth values for the evaluation set.
        uncertainties: Per-sample uncertainty estimates. Required when the
            regressor was calibrated with ``score_type='normalized'``, ignored
            otherwise.
        alpha: Miscoverage level (e.g. 0.1 for 90% intervals).

    Returns:
        Dictionary with namespaced conformal metrics. Returns an empty dict if
        no valid intervals remain after NaN filtering.
    """
    lower, upper = conformal.predict_interval(
        predictions, alpha=alpha, uncertainties=uncertainties
    )
    base = compute_conformal_metrics(lower, upper, targets)

    suffix = int(round((1.0 - alpha) * 100))
    return {f"{key}_{suffix}": value for key, value in base.items()}
