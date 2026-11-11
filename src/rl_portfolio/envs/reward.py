"""Reward interface and a simple, documented default.

The reward used in the thesis and the deployed model is private. This module
keeps the same call signature so a different objective can be dropped in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rl_portfolio.config import RewardConfig


class RewardFunction(Protocol):
    def __call__(self, net_return: float, turnover: float) -> float: ...


@dataclass(frozen=True)
class MeanVarianceUtility:
    """Quadratic utility of the net weekly return: u(r) = r - (a / 2) * r^2.

    With a = 0 the agent maximises expected net return. Larger a penalises the
    second moment of returns, which is the classic mean-variance trade-off in
    per-step form. Transaction costs are already inside `net_return`.
    """

    risk_aversion: float = 1.0

    def __call__(self, net_return: float, turnover: float) -> float:
        return net_return - 0.5 * self.risk_aversion * net_return**2


def make_reward(config: RewardConfig) -> RewardFunction:
    if config.name == "mean_variance":
        return MeanVarianceUtility(risk_aversion=config.risk_aversion)
    raise ValueError(f"unknown reward: {config.name}")
