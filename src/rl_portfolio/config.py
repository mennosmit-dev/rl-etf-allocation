"""Typed configuration loaded from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    source: Literal["synthetic", "yfinance"] = "synthetic"
    tickers: list[str] = Field(min_length=2)
    start: str = "2010-01-01"
    frequency: Literal["weekly"] = "weekly"
    synthetic_weeks: int = Field(default=520, ge=60)
    data_dir: Path = Path("data")


class FeatureSpec(BaseModel):
    name: str
    window: int = Field(ge=1)


class RewardConfig(BaseModel):
    name: Literal["mean_variance"] = "mean_variance"
    risk_aversion: float = Field(default=1.0, ge=0.0)
    transaction_cost: float = Field(default=0.001, ge=0.0, lt=0.1)


class SizingConfig(BaseModel):
    name: Literal["fixed"] = "fixed"
    capital: float = Field(default=10_000.0, gt=0.0)


class PPOConfig(BaseModel):
    total_timesteps: int = Field(default=20_000, ge=1)
    learning_rate: float = Field(default=3e-4, gt=0.0)
    n_steps: int = Field(default=256, ge=2)
    batch_size: int = Field(default=64, ge=2)
    gamma: float = Field(default=0.99, gt=0.0, le=1.0)
    net_arch: list[int] = Field(default_factory=lambda: [64, 64])


class RollingConfig(BaseModel):
    train_weeks: int = Field(default=260, ge=20)
    validation_weeks: int = Field(default=52, ge=4)
    step_weeks: int = Field(default=52, ge=1)


class RegistryConfig(BaseModel):
    path: Path = Path("registry")
    min_sharpe_edge: float = 0.0


class TrackingConfig(BaseModel):
    enabled: bool = True
    uri: str = "sqlite:///mlruns.db"
    experiment: str = "rl-etf-allocation"


class MonitoringConfig(BaseModel):
    max_turnover: float = Field(default=0.5, gt=0.0)
    psi_threshold: float = Field(default=0.2, gt=0.0)


class Config(BaseModel):
    seed: int = 7
    data: DataConfig
    features: list[FeatureSpec] = Field(min_length=1)
    reward: RewardConfig = RewardConfig()
    sizing: SizingConfig = SizingConfig()
    ppo: PPOConfig = PPOConfig()
    rolling: RollingConfig = RollingConfig()
    registry: RegistryConfig = RegistryConfig()
    tracking: TrackingConfig = TrackingConfig()
    monitoring: MonitoringConfig = MonitoringConfig()

    @field_validator("features")
    @classmethod
    def _unique_feature_names(cls, specs: list[FeatureSpec]) -> list[FeatureSpec]:
        keys = [(s.name, s.window) for s in specs]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate feature specs")
        return specs

    @property
    def max_lookback(self) -> int:
        return max(spec.window for spec in self.features)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if overrides:
        raw = _deep_merge(raw, overrides)
    return Config.model_validate(raw)


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
