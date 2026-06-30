"""Tests for split conformal prediction and conformal metric helpers."""

import json

import numpy as np
import pytest

from DataUtils.metrics import compute_conformal_metrics
from UQ.conformal import ConformalRegressor
from UQ.evaluate import evaluate_conformal_intervals


# ---------------------------------------------------------------------------
# ConformalRegressor
# ---------------------------------------------------------------------------


def test_absolute_score_constant_width_interval():
    preds = np.array([1.0, 2.0, 3.0, 4.0])
    targets = np.array([1.5, 1.5, 3.5, 3.5])  # residuals = [0.5, 0.5, 0.5, 0.5]

    cr = ConformalRegressor(score_type="absolute")
    cr.calibrate(preds, targets)

    q = cr.quantile(alpha=0.1)
    # n=4, rank = ceil(5 * 0.9) = 5 -> exceeds n, so largest score (0.5)
    assert q == pytest.approx(0.5)

    lower, upper = cr.predict_interval(np.array([10.0, -3.0]), alpha=0.1)
    assert np.allclose(upper - lower, 2 * 0.5)
    assert np.allclose(lower, [9.5, -3.5])
    assert np.allclose(upper, [10.5, -2.5])


def test_quantile_finite_sample_correction():
    # residuals 1..10, sorted
    preds = np.zeros(10)
    targets = np.arange(1, 11).astype(float)
    cr = ConformalRegressor(score_type="absolute")
    cr.calibrate(preds, targets)

    # n=10, alpha=0.1 -> rank = ceil(11 * 0.9) = ceil(9.9) = 10 -> 10th smallest = 10
    assert cr.quantile(alpha=0.1) == pytest.approx(10.0)
    # alpha=0.2 -> rank = ceil(11 * 0.8) = ceil(8.8) = 9 -> 9th smallest = 9
    assert cr.quantile(alpha=0.2) == pytest.approx(9.0)


def test_normalized_score_scales_with_uncertainty():
    preds = np.array([0.0, 0.0, 0.0])
    targets = np.array([2.0, 4.0, 6.0])
    sigmas = np.array([1.0, 2.0, 3.0])  # normalized scores = [2, 2, 2]

    cr = ConformalRegressor(score_type="normalized")
    cr.calibrate(preds, targets, uncertainties=sigmas)

    q = cr.quantile(alpha=0.1)
    assert q == pytest.approx(2.0)

    test_preds = np.array([0.0, 0.0])
    test_sigmas = np.array([1.0, 5.0])
    lower, upper = cr.predict_interval(test_preds, alpha=0.1, uncertainties=test_sigmas)
    # half-width = q * (sigma + eps) ~ 2 * sigma
    assert np.allclose(upper - lower, 2 * 2 * test_sigmas, atol=1e-6)


def test_normalized_requires_uncertainties():
    cr = ConformalRegressor(score_type="normalized")
    with pytest.raises(ValueError):
        cr.calibrate(np.array([1.0]), np.array([1.0]))
    cr.calibrate(np.array([1.0, 2.0]), np.array([1.0, 2.0]),
                 uncertainties=np.array([0.5, 0.5]))
    with pytest.raises(ValueError):
        cr.predict_interval(np.array([1.0]), alpha=0.1)


def test_invalid_score_type_raises():
    with pytest.raises(ValueError):
        ConformalRegressor(score_type="bogus")  # type: ignore[arg-type]


def test_quantile_before_calibrate_raises():
    cr = ConformalRegressor()
    with pytest.raises(RuntimeError):
        cr.quantile(alpha=0.1)


def test_predict_interval_before_calibrate_raises():
    cr = ConformalRegressor()
    with pytest.raises(RuntimeError):
        cr.predict_interval(np.array([1.0]), alpha=0.1)


def test_alpha_bounds():
    cr = ConformalRegressor()
    cr.calibrate(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        cr.quantile(alpha=0.0)
    with pytest.raises(ValueError):
        cr.quantile(alpha=1.0)


def test_nan_filtering_in_calibrate():
    preds = np.array([1.0, np.nan, 3.0, 4.0])
    targets = np.array([1.0, 2.0, 3.0, np.nan])
    cr = ConformalRegressor(score_type="absolute")
    cr.calibrate(preds, targets)
    # Only rows 0 and 2 survive -> residuals [0, 0]
    assert len(cr.scores_) == 2
    assert cr.quantile(alpha=0.1) == pytest.approx(0.0)


def test_empty_calibration_raises():
    cr = ConformalRegressor(score_type="absolute")
    with pytest.raises(ValueError):
        cr.calibrate(np.array([np.nan]), np.array([np.nan]))


def test_save_load_roundtrip(tmp_path):
    preds = np.array([0.0, 0.0, 0.0])
    targets = np.array([1.0, 2.0, 3.0])
    sigmas = np.array([0.5, 0.5, 0.5])

    cr = ConformalRegressor(score_type="normalized", epsilon=1e-6)
    cr.calibrate(preds, targets, uncertainties=sigmas)

    artifact = tmp_path / "conformal.json"
    cr.save(artifact)

    payload = json.loads(artifact.read_text())
    assert payload["score_type"] == "normalized"
    assert payload["epsilon"] == 1e-6
    assert payload["n_calibration"] == 3

    reloaded = ConformalRegressor.load(artifact)
    assert reloaded.score_type == "normalized"
    assert reloaded.is_calibrated
    assert np.allclose(reloaded.scores_, cr.scores_)
    assert reloaded.quantile(alpha=0.1) == pytest.approx(cr.quantile(alpha=0.1))


def test_save_before_calibrate_raises(tmp_path):
    cr = ConformalRegressor()
    with pytest.raises(RuntimeError):
        cr.save(tmp_path / "out.json")


def test_coverage_guarantee_synthetic():
    """Empirical coverage on a fresh test set should be >= 1 - alpha."""
    rng = np.random.default_rng(0)
    n_calib, n_test = 500, 2000
    true = rng.normal(0, 1, n_calib + n_test)
    sigma = np.full(n_calib + n_test, 1.0)
    noise = rng.normal(0, 1, n_calib + n_test)
    preds = true + noise * sigma

    cr = ConformalRegressor(score_type="normalized")
    cr.calibrate(preds[:n_calib], true[:n_calib], uncertainties=sigma[:n_calib])

    alpha = 0.1
    lower, upper = cr.predict_interval(
        preds[n_calib:], alpha=alpha, uncertainties=sigma[n_calib:]
    )
    test_targets = true[n_calib:]
    coverage = ((test_targets >= lower) & (test_targets <= upper)).mean()
    assert coverage >= 1 - alpha


# ---------------------------------------------------------------------------
# compute_conformal_metrics
# ---------------------------------------------------------------------------


def test_compute_conformal_metrics_basic():
    lower = np.array([0.0, 0.0, 0.0])
    upper = np.array([2.0, 2.0, 2.0])
    targets = np.array([1.0, 3.0, 0.5])  # 2 of 3 inside

    metrics = compute_conformal_metrics(lower, upper, targets)
    assert metrics["coverage"] == pytest.approx(2 / 3)
    assert metrics["mean_interval_width"] == pytest.approx(2.0)
    assert metrics["median_interval_width"] == pytest.approx(2.0)


def test_compute_conformal_metrics_nan_filtering():
    lower = np.array([0.0, np.nan, 0.0])
    upper = np.array([2.0, 2.0, 2.0])
    targets = np.array([1.0, 1.0, 3.0])  # after NaN filter: rows 0 and 2

    metrics = compute_conformal_metrics(lower, upper, targets)
    assert metrics["coverage"] == pytest.approx(0.5)


def test_compute_conformal_metrics_all_nan_returns_empty():
    lower = np.array([np.nan])
    upper = np.array([np.nan])
    metrics = compute_conformal_metrics(lower, upper, np.array([1.0]))
    assert metrics == {}


# ---------------------------------------------------------------------------
# evaluate_conformal_intervals
# ---------------------------------------------------------------------------


def test_evaluate_conformal_namespacing_absolute():
    preds = np.zeros(5)
    targets = np.ones(5)  # residuals all 1
    cr = ConformalRegressor(score_type="absolute")
    cr.calibrate(preds, targets)

    metrics = evaluate_conformal_intervals(cr, preds, targets, alpha=0.1)
    assert "coverage_90" in metrics
    assert "mean_interval_width_90" in metrics
    # All targets == 1, intervals [0 +/- 1] = [-1, 1] -> all covered
    assert metrics["coverage_90"] == pytest.approx(1.0)


def test_evaluate_conformal_normalized_uses_uncertainties():
    preds = np.zeros(4)
    targets = np.array([2.0, 4.0, 6.0, 8.0])
    sigmas = np.array([1.0, 2.0, 3.0, 4.0])  # normalized scores all 2

    cr = ConformalRegressor(score_type="normalized")
    cr.calibrate(preds, targets, uncertainties=sigmas)

    metrics = evaluate_conformal_intervals(
        cr, preds, targets, uncertainties=sigmas, alpha=0.1
    )
    assert "coverage_90" in metrics
    # intervals [0 +/- 2*sigma] contain targets exactly at the boundary
    assert metrics["coverage_90"] == pytest.approx(1.0)
    assert metrics["mean_interval_width_90"] == pytest.approx(np.mean(2 * 2 * sigmas))


def test_evaluate_conformal_alpha_namespacing():
    preds = np.zeros(5)
    targets = np.ones(5)
    cr = ConformalRegressor(score_type="absolute")
    cr.calibrate(preds, targets)

    m90 = evaluate_conformal_intervals(cr, preds, targets, alpha=0.1)
    m80 = evaluate_conformal_intervals(cr, preds, targets, alpha=0.2)
    assert "coverage_90" in m90
    assert "coverage_80" in m80
