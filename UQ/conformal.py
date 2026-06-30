"""Split conformal prediction for molecular solubility regression.

This module provides a model-agnostic conformal regressor that turns point
predictions (optionally paired with an uncertainty estimate) into prediction
intervals with a finite-sample marginal coverage guarantee.

Two nonconformity scores are supported:

- ``"absolute"``: ``s = |y - y_hat|`` produces constant-width intervals
  ``y_hat +/- q``.
- ``"normalized"``: ``s = |y - y_hat| / (sigma + epsilon)`` produces
  variable-width intervals ``y_hat +/- q * (sigma + epsilon)`` where ``sigma``
  is a per-molecule uncertainty estimate (e.g. MC dropout or tree variance).
  This ties conformal prediction to the existing uncertainty estimates and
  yields locally adaptive intervals.

Reference:
    Vovk et al. "Algorithmic Learning in a Random World", 2005.
    Lei et al. "Distribution-Free Predictive Inference for Regression", JASA 2018.
"""

import json
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import numpy as np

ScoreType = Literal["absolute", "normalized"]


class ConformalRegressor:
    """Split conformal predictor for regression.

    The regressor is fit on a held-out calibration set by storing the
    nonconformity scores. Prediction intervals can then be produced for any
    confidence level ``1 - alpha`` without recalibration.

    Args:
        score_type: Nonconformity score, ``"absolute"`` or ``"normalized"``.
        epsilon: Small constant added to uncertainties to avoid division by
            zero in the normalized score.

    Attributes:
        score_type: The configured nonconformity score.
        epsilon: Numerical stabilizer for the normalized score.
        scores_: Sorted calibration nonconformity scores (set after calibrate).
        is_calibrated: Whether the regressor has been calibrated.

    Example:
        >>> conformal = ConformalRegressor(score_type="normalized")
        >>> conformal.calibrate(calib_preds, calib_targets, calib_stds)
        >>> lower, upper = conformal.predict_interval(
        ...     test_preds, alpha=0.1, uncertainties=test_stds
        ... )
    """

    def __init__(
        self,
        score_type: ScoreType = "absolute",
        epsilon: float = 1e-8,
    ) -> None:
        if score_type not in ("absolute", "normalized"):
            raise ValueError(
                f"score_type must be 'absolute' or 'normalized', got {score_type}"
            )
        self.score_type: ScoreType = score_type
        self.epsilon = epsilon
        self.scores_: Optional[np.ndarray] = None

    @property
    def is_calibrated(self) -> bool:
        """Whether the regressor has been calibrated."""
        return self.scores_ is not None

    def _compute_scores(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        uncertainties: Optional[np.ndarray],
    ) -> np.ndarray:
        """Compute nonconformity scores for the configured score type."""
        residuals = np.abs(targets - predictions)

        if self.score_type == "absolute":
            return residuals

        # normalized
        if uncertainties is None:
            raise ValueError(
                "uncertainties are required for score_type='normalized'"
            )
        return residuals / (uncertainties + self.epsilon)

    def calibrate(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        uncertainties: Optional[np.ndarray] = None,
    ) -> "ConformalRegressor":
        """Calibrate the conformal regressor on a held-out set.

        Rows with NaN predictions, targets, or (for the normalized score)
        uncertainties are dropped before computing scores.

        Args:
            predictions: Point predictions on the calibration set.
            targets: Ground truth values for the calibration set.
            uncertainties: Per-sample uncertainty estimates. Required when
                ``score_type='normalized'``, ignored otherwise.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If no valid calibration samples remain.
        """
        predictions = np.asarray(predictions, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)

        valid_mask = ~(np.isnan(predictions) | np.isnan(targets))
        if self.score_type == "normalized":
            if uncertainties is None:
                raise ValueError(
                    "uncertainties are required for score_type='normalized'"
                )
            uncertainties = np.asarray(uncertainties, dtype=np.float64)
            valid_mask &= ~np.isnan(uncertainties)

        preds = predictions[valid_mask]
        targs = targets[valid_mask]
        uncs = uncertainties[valid_mask] if uncertainties is not None else None

        if len(preds) == 0:
            raise ValueError("No valid calibration samples after filtering NaNs")

        scores = self._compute_scores(preds, targs, uncs)
        self.scores_ = np.sort(scores)
        return self

    def quantile(self, alpha: float = 0.1) -> float:
        """Return the conformal quantile for confidence ``1 - alpha``.

        Uses the finite-sample correction: the ``ceil((n + 1) * (1 - alpha))``-th
        smallest calibration score. When that rank exceeds ``n`` (i.e. ``alpha``
        is too small for the calibration set size), the largest score is used
        and exact coverage cannot be guaranteed.

        Args:
            alpha: Miscoverage level (e.g. 0.1 for 90% intervals).

        Returns:
            The conformal quantile value.

        Raises:
            RuntimeError: If the regressor has not been calibrated.
            ValueError: If alpha is not in (0, 1).
        """
        if not self.is_calibrated:
            raise RuntimeError("ConformalRegressor must be calibrated first")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        assert self.scores_ is not None  # for type checkers
        n = len(self.scores_)
        rank = int(np.ceil((n + 1) * (1.0 - alpha)))

        if rank > n:
            return float(self.scores_[-1])
        return float(self.scores_[rank - 1])

    def predict_interval(
        self,
        predictions: np.ndarray,
        alpha: float = 0.1,
        uncertainties: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Produce prediction intervals for confidence ``1 - alpha``.

        Args:
            predictions: Point predictions.
            alpha: Miscoverage level (e.g. 0.1 for 90% intervals).
            uncertainties: Per-sample uncertainty estimates. Required when
                ``score_type='normalized'``, ignored otherwise.

        Returns:
            Tuple of (lower, upper) interval bound arrays. NaN predictions
            (or NaN uncertainties for the normalized score) yield NaN bounds.

        Raises:
            RuntimeError: If the regressor has not been calibrated.
        """
        if not self.is_calibrated:
            raise RuntimeError("ConformalRegressor must be calibrated first")

        predictions = np.asarray(predictions, dtype=np.float64)
        q = self.quantile(alpha)

        if self.score_type == "absolute":
            half_width = np.full_like(predictions, q)
        else:
            if uncertainties is None:
                raise ValueError(
                    "uncertainties are required for score_type='normalized'"
                )
            uncertainties = np.asarray(uncertainties, dtype=np.float64)
            half_width = q * (uncertainties + self.epsilon)

        lower = predictions - half_width
        upper = predictions + half_width
        return lower, upper

    def save(self, path: Union[str, Path]) -> None:
        """Save the calibrated regressor to a JSON artifact.

        Args:
            path: Destination file path.

        Raises:
            RuntimeError: If the regressor has not been calibrated.
        """
        if not self.is_calibrated:
            raise RuntimeError("ConformalRegressor must be calibrated before saving")

        assert self.scores_ is not None
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "score_type": self.score_type,
            "epsilon": self.epsilon,
            "n_calibration": len(self.scores_),
            "scores": self.scores_.tolist(),
        }
        with open(path, "w") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ConformalRegressor":
        """Load a calibrated regressor from a JSON artifact.

        Args:
            path: Path to a JSON artifact written by :meth:`save`.

        Returns:
            A calibrated ConformalRegressor.
        """
        with open(path, "r") as f:
            payload = json.load(f)

        regressor = cls(
            score_type=payload["score_type"],
            epsilon=payload["epsilon"],
        )
        regressor.scores_ = np.sort(np.asarray(payload["scores"], dtype=np.float64))
        return regressor
