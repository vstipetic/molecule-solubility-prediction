"""Random Forest model with ECFP fingerprints for molecular solubility prediction.

This module provides a wrapper around scikit-learn's RandomForestRegressor
that handles SMILES to fingerprint conversion and provides uncertainty
estimation via tree variance.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from DataUtils.utils import compute_ecfp_from_smiles


class ECFPRandomForest:
    """Random Forest regressor using ECFP fingerprints.

    This class wraps sklearn's RandomForestRegressor and handles:
    - SMILES to ECFP fingerprint conversion
    - Training and prediction
    - Uncertainty estimation via tree variance

    Args:
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of trees. None for unlimited.
        min_samples_split: Minimum samples required to split a node.
        min_samples_leaf: Minimum samples required at a leaf node.
        fingerprint_radius: Radius for Morgan fingerprint (2 = ECFP4).
        fingerprint_bits: Number of bits in fingerprint vector.
        n_jobs: Number of parallel jobs. -1 uses all processors.
        random_state: Random seed for reproducibility.

    Attributes:
        model: The underlying RandomForestRegressor.
        fingerprint_radius: ECFP radius used.
        fingerprint_bits: ECFP bit size used.
        is_fitted: Whether the model has been trained.

    Example:
        >>> model = ECFPRandomForest(n_estimators=100)
        >>> model.fit(train_smiles, train_labels)
        >>> predictions = model.predict(test_smiles)
        >>> mean, std = model.predict_with_uncertainty(test_smiles)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        fingerprint_radius: int = 2,
        fingerprint_bits: int = 2048,
        n_jobs: int = -1,
        random_state: int = 42,
    ) -> None:
        self.fingerprint_radius = fingerprint_radius
        self.fingerprint_bits = fingerprint_bits
        self.is_fitted = False

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def _smiles_to_fingerprints(
        self, smiles_list: List[str]
    ) -> Tuple[np.ndarray, List[int]]:
        """Convert a list of SMILES to fingerprint array.

        Args:
            smiles_list: List of SMILES strings.

        Returns:
            Tuple of (fingerprints array, valid indices).
            Invalid SMILES are excluded from the output.
        """
        fingerprints = []
        valid_indices = []

        for idx, smiles in enumerate(smiles_list):
            fp = compute_ecfp_from_smiles(
                smiles,
                radius=self.fingerprint_radius,
                n_bits=self.fingerprint_bits,
            )
            if fp is not None:
                fingerprints.append(fp)
                valid_indices.append(idx)

        return np.array(fingerprints), valid_indices

    def fit(
        self,
        smiles_list: List[str],
        labels: Union[List[float], np.ndarray],
    ) -> "ECFPRandomForest":
        """Train the Random Forest model.

        Args:
            smiles_list: List of SMILES strings.
            labels: Target values (solubility).

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If all SMILES are invalid.
        """
        labels_arr = np.array(labels)

        fingerprints, valid_indices = self._smiles_to_fingerprints(smiles_list)

        if len(fingerprints) == 0:
            raise ValueError("No valid SMILES found in training data")

        valid_labels = labels_arr[valid_indices]

        self.model.fit(fingerprints, valid_labels)
        self.is_fitted = True

        return self

    def predict(self, smiles_list: List[str]) -> np.ndarray:
        """Predict solubility for a list of SMILES.

        Args:
            smiles_list: List of SMILES strings.

        Returns:
            Array of predictions. Invalid SMILES get NaN.

        Raises:
            RuntimeError: If model hasn't been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        predictions = np.full(len(smiles_list), np.nan)
        fingerprints, valid_indices = self._smiles_to_fingerprints(smiles_list)

        if len(fingerprints) > 0:
            preds = self.model.predict(fingerprints)
            predictions[valid_indices] = preds

        return predictions

    def predict_with_uncertainty(
        self, smiles_list: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict solubility with uncertainty estimation.

        Uncertainty is estimated using the variance of predictions across
        individual trees in the forest.

        Args:
            smiles_list: List of SMILES strings.

        Returns:
            Tuple of (mean_predictions, std_predictions).
            Invalid SMILES get NaN for both.

        Raises:
            RuntimeError: If model hasn't been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        mean_predictions = np.full(len(smiles_list), np.nan)
        std_predictions = np.full(len(smiles_list), np.nan)

        fingerprints, valid_indices = self._smiles_to_fingerprints(smiles_list)

        if len(fingerprints) > 0:
            # Get predictions from each tree
            tree_predictions = np.array([
                tree.predict(fingerprints) for tree in self.model.estimators_
            ])

            # Calculate mean and std
            mean_preds = tree_predictions.mean(axis=0)
            std_preds = tree_predictions.std(axis=0)

            mean_predictions[valid_indices] = mean_preds
            std_predictions[valid_indices] = std_preds

        return mean_predictions, std_predictions

    def get_feature_importances(self) -> np.ndarray:
        """Get feature importances from the trained model.

        Returns:
            Array of feature importances.

        Raises:
            RuntimeError: If model hasn't been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting importances")

        return self.model.feature_importances_

    def get_params(self) -> dict:
        """Get model parameters.

        Returns:
            Dictionary of model parameters.
        """
        return {
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth,
            "min_samples_split": self.model.min_samples_split,
            "min_samples_leaf": self.model.min_samples_leaf,
            "fingerprint_radius": self.fingerprint_radius,
            "fingerprint_bits": self.fingerprint_bits,
        }

    def set_params(self, **params) -> "ECFPRandomForest":
        """Set model parameters.

        Args:
            **params: Parameters to set.

        Returns:
            Self for method chaining.
        """
        if "fingerprint_radius" in params:
            self.fingerprint_radius = params.pop("fingerprint_radius")
        if "fingerprint_bits" in params:
            self.fingerprint_bits = params.pop("fingerprint_bits")

        self.model.set_params(**params)
        return self
