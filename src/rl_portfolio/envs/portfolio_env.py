"""Gymnasium environment for long-only weekly allocation."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_portfolio.envs.reward import MeanVarianceUtility, RewardFunction


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def drift_weights(weights: np.ndarray, asset_returns: np.ndarray) -> np.ndarray:
    """Weights after one period of price moves, before rebalancing."""
    grown = weights * (1.0 + asset_returns)
    total = grown.sum()
    return grown / total if total > 0 else np.full_like(weights, 1.0 / len(weights))


class PortfolioEnv(gym.Env):
    """One step is one week.

    Observation: features known at the end of week t-1, plus the current
    (drifted) portfolio weights. Action: one logit per asset, mapped to
    long-only weights with a softmax. Reward: a pluggable function of the net
    return after proportional transaction costs on turnover.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        returns: np.ndarray,
        features: np.ndarray,
        start: int,
        end: int,
        reward_fn: RewardFunction | None = None,
        transaction_cost: float = 0.001,
    ) -> None:
        super().__init__()
        returns = np.asarray(returns, dtype=np.float64)
        features = np.asarray(features, dtype=np.float64)
        if returns.ndim != 2 or features.ndim != 2 or len(returns) != len(features):
            raise ValueError("returns and features must be 2D with the same number of rows")
        if not 1 <= start < end <= len(returns):
            raise ValueError(f"invalid window start={start} end={end} for {len(returns)} rows")
        if np.isnan(features[start - 1 : end - 1]).any():
            raise ValueError("features contain NaN inside the window; start after the warm-up")

        self.returns = returns
        self.features = features
        self.start, self.end = start, end
        self.reward_fn = reward_fn or MeanVarianceUtility()
        self.transaction_cost = transaction_cost
        self.n_assets = returns.shape[1]

        obs_dim = features.shape[1] + self.n_assets
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-5.0, 5.0, shape=(self.n_assets,), dtype=np.float32)

        self.t = start
        self.weights = np.full(self.n_assets, 1.0 / self.n_assets)

    def _obs(self) -> np.ndarray:
        return np.concatenate([self.features[self.t - 1], self.weights]).astype(np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.t = self.start
        initial = (options or {}).get("initial_weights")
        if initial is None:
            self.weights = np.full(self.n_assets, 1.0 / self.n_assets)
        else:
            self.weights = np.asarray(initial, dtype=np.float64)
        return self._obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        target = softmax(action)
        turnover = float(np.abs(target - self.weights).sum())
        asset_returns = self.returns[self.t]
        gross = float(target @ asset_returns)
        net = gross - self.transaction_cost * turnover
        reward = float(self.reward_fn(net, turnover))

        self.weights = drift_weights(target, asset_returns)
        self.t += 1
        terminated = self.t >= self.end
        info = {"weights": target, "net_return": net, "gross_return": gross, "turnover": turnover}
        return self._obs(), reward, terminated, False, info
