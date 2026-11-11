"""How much capital the strategy manages inside a broker account.

The deployed system sizes the portfolio dynamically; that logic is private.
The public default manages a fixed sleeve and never more than the account holds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rl_portfolio.config import SizingConfig


class PortfolioSizer(Protocol):
    def managed_capital(self, account_equity: float) -> float: ...


@dataclass(frozen=True)
class FixedSizer:
    capital: float

    def managed_capital(self, account_equity: float) -> float:
        if account_equity < 0:
            raise ValueError("account equity cannot be negative")
        return min(self.capital, account_equity)


def make_sizer(config: SizingConfig) -> PortfolioSizer:
    if config.name == "fixed":
        return FixedSizer(capital=config.capital)
    raise ValueError(f"unknown sizer: {config.name}")
