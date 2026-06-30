"""UQ module for uncertainty quantification.

This module provides uncertainty quantification utilities that complement the
per-model uncertainty estimates (MC dropout, tree variance):

- ConformalRegressor: distribution-free prediction intervals with a
  finite-sample marginal coverage guarantee.
"""

from UQ.conformal import ConformalRegressor
from UQ.evaluate import evaluate_conformal_intervals

__all__ = [
    "ConformalRegressor",
    "evaluate_conformal_intervals",
]
