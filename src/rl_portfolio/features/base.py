"""Pluggable feature interface.

A feature builder maps a (weeks x assets) return frame to a frame of the same
shape. Row t may only use returns up to and including week t; the environment
then feeds row t-1 to the agent when it chooses weights for week t, so no
builder can leak future returns by construction.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np
import pandas as pd

from rl_portfolio.config import FeatureSpec


class FeatureBuilder(Protocol):
    name: str
    window: int

    def compute(self, returns: pd.DataFrame) -> pd.DataFrame: ...


_REGISTRY: dict[str, Callable[[int], FeatureBuilder]] = {}


def register(
    name: str,
) -> Callable[[Callable[[int], FeatureBuilder]], Callable[[int], FeatureBuilder]]:
    def decorator(factory: Callable[[int], FeatureBuilder]) -> Callable[[int], FeatureBuilder]:
        if name in _REGISTRY:
            raise ValueError(f"feature {name!r} already registered")
        _REGISTRY[name] = factory
        return factory

    return decorator


def available_features() -> list[str]:
    return sorted(_REGISTRY)


def make_builders(specs: list[FeatureSpec]) -> list[FeatureBuilder]:
    unknown = [s.name for s in specs if s.name not in _REGISTRY]
    if unknown:
        raise KeyError(f"unknown features {unknown}; available: {available_features()}")
    return [_REGISTRY[s.name](s.window) for s in specs]


def build_feature_tensor(returns: pd.DataFrame, builders: list[FeatureBuilder]) -> np.ndarray:
    """Stack features into shape (weeks, n_features * n_assets); warm-up rows are NaN."""
    blocks = []
    for builder in builders:
        frame = builder.compute(returns)
        if frame.shape != returns.shape:
            raise ValueError(
                f"{builder.name} returned shape {frame.shape}, expected {returns.shape}"
            )
        blocks.append(frame.to_numpy(dtype=np.float64))
    return np.concatenate(blocks, axis=1)
