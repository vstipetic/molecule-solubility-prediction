"""Calibrate a conformal predictor from model predictions and labels.

This script is model-agnostic. It consumes the standard inference output
(SMILES, Solubility, Uncertainty) produced by any of the ``Inference.*``
scripts on the calibration split, joins it with the ground-truth labels from
``calib.csv``, fits a :class:`UQ.conformal.ConformalRegressor`, and saves the
calibration artifact.

Typical workflow:

    # 1. Produce predictions on the calibration split
    python -m Inference.inference_random_forest \
        --model-path models/rf.pkl \
        --input-path data/splits/calib.csv \
        --output-path preds/rf_calib.csv

    # 2. Fit and save the conformal artifact
    python -m Train.calibrate_conformal \
        --predictions-path preds/rf_calib.csv \
        --labels-path data/splits/calib.csv \
        --output-path models/rf_conformal.json \
        --score-type normalized
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from UQ.conformal import ConformalRegressor


def fit_conformal(
    predictions: np.ndarray,
    targets: np.ndarray,
    uncertainties: Optional[np.ndarray] = None,
    score_type: str = "normalized",
    save_path: Optional[str] = None,
) -> ConformalRegressor:
    """Fit a conformal regressor from prediction arrays.

    Args:
        predictions: Point predictions on the calibration set.
        targets: Ground truth values for the calibration set.
        uncertainties: Per-sample uncertainty estimates. Required for the
            normalized score.
        score_type: Nonconformity score, ``"absolute"`` or ``"normalized"``.
        save_path: Optional path to save the calibration artifact.

    Returns:
        The calibrated ConformalRegressor.
    """
    conformal = ConformalRegressor(score_type=score_type)
    conformal.calibrate(predictions, targets, uncertainties)

    if save_path is not None:
        conformal.save(save_path)

    return conformal


def calibrate_from_files(
    predictions_path: str,
    labels_path: str,
    output_path: str,
    score_type: str = "normalized",
    smiles_column: str = "SMILES",
    pred_column: str = "Solubility",
    uncertainty_column: str = "Uncertainty",
    label_column: str = "Solubility",
    report_alpha: float = 0.1,
) -> ConformalRegressor:
    """Fit a conformal regressor by joining a predictions file with labels.

    Args:
        predictions_path: CSV with SMILES, predictions, and (for the
            normalized score) an uncertainty column.
        labels_path: CSV with SMILES and ground-truth labels (e.g. calib.csv).
        output_path: Where to save the calibration artifact (JSON).
        score_type: Nonconformity score, ``"absolute"`` or ``"normalized"``.
        smiles_column: Name of the SMILES column in both files.
        pred_column: Name of the prediction column in the predictions file.
        uncertainty_column: Name of the uncertainty column in the predictions
            file (only used for the normalized score).
        label_column: Name of the ground-truth column in the labels file.
        report_alpha: Miscoverage level used for the printed summary.

    Returns:
        The calibrated ConformalRegressor.
    """
    preds_df = pd.read_csv(predictions_path)
    labels_df = pd.read_csv(labels_path)

    if score_type == "normalized" and uncertainty_column not in preds_df.columns:
        raise ValueError(
            f"score_type='normalized' requires an '{uncertainty_column}' column "
            f"in {predictions_path}. Re-run inference without --no-uncertainty."
        )

    # Join predictions with labels on SMILES. Keep only the columns we need to
    # avoid collisions when both files share a 'Solubility' column.
    pred_cols = [smiles_column, pred_column]
    if score_type == "normalized":
        pred_cols.append(uncertainty_column)
    merged = preds_df[pred_cols].merge(
        labels_df[[smiles_column, label_column]],
        on=smiles_column,
        how="inner",
        suffixes=("_pred", "_label"),
    )

    print(
        f"Matched {len(merged)} molecules "
        f"(predictions: {len(preds_df)}, labels: {len(labels_df)})"
    )

    # Resolve column names after the merge (pandas suffixes collisions).
    pred_col = pred_column if pred_column != label_column else f"{pred_column}_pred"
    label_col = label_column if pred_column != label_column else f"{label_column}_label"

    predictions = merged[pred_col].to_numpy(dtype=np.float64)
    targets = merged[label_col].to_numpy(dtype=np.float64)
    uncertainties = (
        merged[uncertainty_column].to_numpy(dtype=np.float64)
        if score_type == "normalized"
        else None
    )

    conformal = fit_conformal(
        predictions=predictions,
        targets=targets,
        uncertainties=uncertainties,
        score_type=score_type,
        save_path=output_path,
    )

    q = conformal.quantile(report_alpha)
    print(f"Calibrated conformal regressor (score_type='{score_type}')")
    print(f"  Calibration samples: {len(conformal.scores_)}")
    print(f"  Quantile at alpha={report_alpha} (i.e. {int((1 - report_alpha) * 100)}% intervals): {q:.4f}")
    print(f"  Saved artifact to {output_path}")

    return conformal


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate a conformal predictor from predictions and labels"
    )
    parser.add_argument(
        "--predictions-path", type=str, required=True,
        help="CSV with SMILES, predictions, and (for normalized) uncertainty"
    )
    parser.add_argument(
        "--labels-path", type=str, required=True,
        help="CSV with SMILES and ground-truth labels (e.g. calib.csv)"
    )
    parser.add_argument(
        "--output-path", type=str, required=True,
        help="Path to save the conformal calibration artifact (JSON)"
    )
    parser.add_argument(
        "--score-type", type=str, default="normalized",
        choices=["absolute", "normalized"],
        help="Nonconformity score (default: normalized)"
    )
    parser.add_argument("--smiles-column", type=str, default="SMILES")
    parser.add_argument("--pred-column", type=str, default="Solubility")
    parser.add_argument("--uncertainty-column", type=str, default="Uncertainty")
    parser.add_argument("--label-column", type=str, default="Solubility")
    parser.add_argument(
        "--report-alpha", type=float, default=0.1,
        help="Miscoverage level for the printed summary (default: 0.1)"
    )

    args = parser.parse_args()

    calibrate_from_files(
        predictions_path=args.predictions_path,
        labels_path=args.labels_path,
        output_path=args.output_path,
        score_type=args.score_type,
        smiles_column=args.smiles_column,
        pred_column=args.pred_column,
        uncertainty_column=args.uncertainty_column,
        label_column=args.label_column,
        report_alpha=args.report_alpha,
    )


if __name__ == "__main__":
    main()
