import numpy as np
import pytest

from rl_portfolio.monitoring.drift import check_drift, population_stability_index, turnover


def test_psi_near_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    assert population_stability_index(rng.normal(size=5000), rng.normal(size=5000)) < 0.02


def test_psi_flags_shifted_distribution():
    rng = np.random.default_rng(0)
    assert population_stability_index(rng.normal(size=5000), rng.normal(1.0, size=500)) > 0.2


def test_turnover_counts_both_sides():
    assert turnover({"A": 1.0}, {"A": 0.5, "B": 0.5}) == pytest.approx(1.0)


def test_drift_report():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(500, 2))
    recent = np.column_stack([rng.normal(size=100), rng.normal(2.0, size=100)])
    report = check_drift(
        ref, recent, ["stable", "shifted"], previous_weights={"A": 1.0}, target_weights={"B": 1.0}
    )
    assert report.drifted_features == ["shifted"]
    assert report.turnover_alert
    assert not report.ok
