"""Drift checks run before each weekly allocation.

Two questions: has the input distribution moved away from what the model was
trained on (population stability index per feature), and is the model asking
for an unusually large reallocation (turnover)?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """PSI with quantile bins from the reference sample. Above 0.2 is usually read as drift."""
    reference = np.asarray(reference, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if reference.size < bins or current.size == 0:
        raise ValueError("not enough observations for PSI")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_share = np.histogram(reference, edges)[0] / reference.size
    cur_share = np.histogram(current, edges)[0] / current.size
    eps = 1e-6
    ref_share, cur_share = np.clip(ref_share, eps, None), np.clip(cur_share, eps, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def turnover(previous: dict[str, float], target: dict[str, float]) -> float:
    tickers = set(previous) | set(target)
    return float(sum(abs(target.get(t, 0.0) - previous.get(t, 0.0)) for t in tickers))


@dataclass(frozen=True)
class DriftReport:
    feature_psi: dict[str, float]
    drifted_features: list[str]
    turnover: float | None
    turnover_alert: bool

    @property
    def ok(self) -> bool:
        return not self.drifted_features and not self.turnover_alert


def check_drift(
    reference_features: np.ndarray,
    recent_features: np.ndarray,
    feature_names: list[str],
    psi_threshold: float = 0.2,
    previous_weights: dict[str, float] | None = None,
    target_weights: dict[str, float] | None = None,
    max_turnover: float = 0.5,
) -> DriftReport:
    if reference_features.shape[1] != len(feature_names):
        raise ValueError("feature_names must match the number of feature columns")
    psi = {
        name: population_stability_index(reference_features[:, i], recent_features[:, i])
        for i, name in enumerate(feature_names)
    }
    drifted = sorted(name for name, value in psi.items() if value > psi_threshold)
    to = None
    if previous_weights is not None and target_weights is not None:
        to = turnover(previous_weights, target_weights)
    return DriftReport(psi, drifted, to, bool(to is not None and to > max_turnover))
