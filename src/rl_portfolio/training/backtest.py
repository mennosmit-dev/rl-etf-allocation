"""Roll a policy through a window of the environment and score it."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from rl_portfolio.envs.portfolio_env import PortfolioEnv
from rl_portfolio.metrics import summarize

Policy = Callable[[np.ndarray], np.ndarray]


@dataclass
class BacktestResult:
    net_returns: np.ndarray
    turnover: np.ndarray
    weights: np.ndarray
    final_weights: np.ndarray

    @property
    def metrics(self) -> dict[str, float]:
        return summarize(self.net_returns, self.turnover)


def run_policy(
    policy: Policy, env: PortfolioEnv, initial_weights: np.ndarray | None = None
) -> BacktestResult:
    options = None if initial_weights is None else {"initial_weights": initial_weights}
    obs, _ = env.reset(options=options)
    rets, turns, weights = [], [], []
    done = False
    while not done:
        obs, _, done, _, info = env.step(policy(obs))
        rets.append(info["net_return"])
        turns.append(info["turnover"])
        weights.append(info["weights"])
    return BacktestResult(np.array(rets), np.array(turns), np.array(weights), env.weights.copy())


def equal_weight_policy(n_assets: int) -> Policy:
    """Zero logits give equal weights after the softmax: weekly rebalanced 1/N."""
    zeros = np.zeros(n_assets, dtype=np.float32)
    return lambda _obs: zeros


def sb3_policy(model) -> Policy:
    return lambda obs: model.predict(obs, deterministic=True)[0]
