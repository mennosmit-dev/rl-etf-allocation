"""Standard example features. The thesis feature set is private."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rl_portfolio.features.base import register


@register("trailing_return")
@dataclass(frozen=True)
class TrailingReturn:
    """Compounded return over the last `window` weeks."""

    window: int
    name: str = "trailing_return"

    def compute(self, returns: pd.DataFrame) -> pd.DataFrame:
        growth = np.log1p(returns).rolling(self.window).sum()
        return np.expm1(growth)


@register("rolling_volatility")
@dataclass(frozen=True)
class RollingVolatility:
    """Annualised standard deviation of weekly returns over `window` weeks."""

    window: int
    name: str = "rolling_volatility"

    def compute(self, returns: pd.DataFrame) -> pd.DataFrame:
        return returns.rolling(self.window).std() * np.sqrt(52)


@register("momentum")
@dataclass(frozen=True)
class Momentum:
    """Cross-sectional z-score of the `window`-week return."""

    window: int
    name: str = "momentum"

    def compute(self, returns: pd.DataFrame) -> pd.DataFrame:
        trailing = np.log1p(returns).rolling(self.window).sum()
        mean = trailing.mean(axis=1)
        std = trailing.std(axis=1).replace(0.0, np.nan)
        return trailing.sub(mean, axis=0).div(std, axis=0).fillna(0.0).where(trailing.notna())
