"""PPO training on one train window, scored on the following validation window."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from rl_portfolio.config import Config
from rl_portfolio.data.dataset import DatasetMetadata
from rl_portfolio.envs.portfolio_env import PortfolioEnv
from rl_portfolio.envs.reward import make_reward
from rl_portfolio.features import build_feature_tensor, make_builders
from rl_portfolio.tracking import tracking_run
from rl_portfolio.training.backtest import (
    BacktestResult,
    equal_weight_policy,
    run_policy,
    sb3_policy,
)

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")


@dataclass(frozen=True)
class Prepared:
    returns: np.ndarray
    features: np.ndarray
    dates: list[str]
    tickers: list[str]
    first_valid: int  # first week whose weights can be chosen from complete features


def prepare(config: Config, returns: pd.DataFrame) -> Prepared:
    builders = make_builders(config.features)
    features = build_feature_tensor(returns, builders)
    complete = ~np.isnan(features).any(axis=1)
    if not complete.any():
        raise ValueError("no week has complete features; dataset shorter than the lookback")
    first_complete = int(np.argmax(complete))
    return Prepared(
        returns=returns.to_numpy(dtype=np.float64),
        features=features,
        dates=[str(d) for d in returns.index],
        tickers=list(returns.columns),
        first_valid=first_complete + 1,
    )


def make_env(config: Config, prep: Prepared, start: int, end: int) -> PortfolioEnv:
    return PortfolioEnv(
        returns=prep.returns,
        features=prep.features,
        start=start,
        end=end,
        reward_fn=make_reward(config.reward),
        transaction_cost=config.reward.transaction_cost,
    )


def fit_ppo(config: Config, prep: Prepared, start: int, end: int) -> PPO:
    env = make_env(config, prep, start, end)
    p = config.ppo
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=p.learning_rate,
        n_steps=p.n_steps,
        batch_size=p.batch_size,
        gamma=p.gamma,
        policy_kwargs={"net_arch": list(p.net_arch)},
        seed=config.seed,
        device="cpu",
        verbose=0,
    )
    model.learn(total_timesteps=p.total_timesteps)
    return model


@dataclass
class WindowResult:
    model: PPO
    train_window: tuple[str, str]
    validation_window: tuple[str, str]
    rl: BacktestResult
    equal_weight: BacktestResult
    run_id: str | None = None
    model_path: Path | None = None
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def sharpe_edge(self) -> float:
        return self.rl.metrics["sharpe"] - self.equal_weight.metrics["sharpe"]


def evaluate_window(
    config: Config, prep: Prepared, model: PPO, train_start: int, train_end: int, val_end: int
) -> tuple[BacktestResult, BacktestResult]:
    """Score the policy out of sample. It starts from the weights it held at the end of training."""
    in_sample = run_policy(sb3_policy(model), make_env(config, prep, train_start, train_end))
    val_env = make_env(config, prep, train_end, val_end)
    rl = run_policy(sb3_policy(model), val_env, initial_weights=in_sample.final_weights)
    ew_env = make_env(config, prep, train_end, val_end)
    ew = run_policy(equal_weight_policy(len(prep.tickers)), ew_env)
    return rl, ew


def config_fingerprint(config: Config) -> str:
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def train_window(
    config: Config,
    prep: Prepared,
    meta: DatasetMetadata,
    train_start: int,
    train_end: int,
    val_end: int,
    run_name: str | None = None,
    nested: bool = False,
    models_dir: Path = MODELS_DIR,
) -> WindowResult:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    run_name = run_name or f"ppo-{prep.dates[val_end - 1]}-{stamp}"
    train_window = (prep.dates[train_start], prep.dates[train_end - 1])
    val_window = (prep.dates[train_end], prep.dates[val_end - 1])

    with tracking_run(config.tracking, run_name, nested=nested) as run:
        run.log_params(
            {
                "seed": config.seed,
                "tickers": ",".join(prep.tickers),
                "features": ",".join(f"{f.name}:{f.window}" for f in config.features),
                "reward": config.reward.name,
                "risk_aversion": config.reward.risk_aversion,
                "transaction_cost": config.reward.transaction_cost,
                **{f"ppo_{k}": v for k, v in config.ppo.model_dump().items()},
                "train_window": "/".join(train_window),
                "validation_window": "/".join(val_window),
            }
        )
        run.set_tags({"dataset_hash": meta.content_hash, "config_hash": config_fingerprint(config)})

        logger.info("Training PPO on %s to %s", *train_window)
        model = fit_ppo(config, prep, train_start, train_end)
        rl, ew = evaluate_window(config, prep, model, train_start, train_end, val_end)

        run.log_metrics({f"val_{k}": v for k, v in rl.metrics.items()})
        run.log_metrics({f"val_ew_{k}": v for k, v in ew.metrics.items()})

        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"{run_name}.zip"
        model.save(model_path)
        run.log_artifact(model_path)

        result = WindowResult(model, train_window, val_window, rl, ew, run.run_id, model_path)
        logger.info(
            "Validation %s to %s: Sharpe %.2f vs equal weight %.2f",
            *val_window,
            rl.metrics["sharpe"],
            ew.metrics["sharpe"],
        )
        return result


def latest_split(config: Config, prep: Prepared) -> tuple[int, int, int]:
    """Most recent train window followed by the most recent validation window."""
    val_end = len(prep.returns)
    train_end = val_end - config.rolling.validation_weeks
    train_start = max(prep.first_valid, train_end - config.rolling.train_weeks)
    if train_end - train_start < 20:
        raise ValueError("not enough data for a training window of at least 20 weeks")
    return train_start, train_end, val_end
