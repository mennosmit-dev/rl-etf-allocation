"""Load a registered model and turn recent returns into target weights."""

from __future__ import annotations

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from rl_portfolio.config import FeatureSpec
from rl_portfolio.envs.portfolio_env import softmax
from rl_portfolio.features import build_feature_tensor, make_builders
from rl_portfolio.registry import ModelRecord


class AllocationPolicy:
    def __init__(self, record: ModelRecord) -> None:
        self.record = record
        self.tickers = list(record.tickers)
        specs = [FeatureSpec(**f) for f in record.features]
        self.builders = make_builders(specs)
        self.lookback = max(s.window for s in specs) + 1
        self.model = PPO.load(record.model_path, device="cpu")

    def target_weights(
        self,
        recent_returns: pd.DataFrame,
        current_weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Weights for next week, using returns up to and including the last completed week."""
        missing = [t for t in self.tickers if t not in recent_returns.columns]
        if missing:
            raise ValueError(f"returns missing tickers: {missing}")
        if len(recent_returns) < self.lookback:
            raise ValueError(f"need at least {self.lookback} weeks of returns")

        frame = recent_returns[self.tickers].astype("float64")
        latest = build_feature_tensor(frame, self.builders)[-1]
        if np.isnan(latest).any():
            raise ValueError("features incomplete for the latest week")

        if current_weights is None:
            held = np.full(len(self.tickers), 1.0 / len(self.tickers))
        else:
            held = np.array([current_weights.get(t, 0.0) for t in self.tickers], dtype=np.float64)
            if held.sum() <= 0:
                raise ValueError("current weights must sum to a positive number")
            held = held / held.sum()

        obs = np.concatenate([latest, held]).astype(np.float32)
        action, _ = self.model.predict(obs, deterministic=True)
        weights = softmax(action)
        return {t: float(w) for t, w in zip(self.tickers, weights, strict=True)}
