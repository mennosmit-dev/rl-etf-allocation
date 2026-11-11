"""Weekly paper-trading loop against a simulated broker.

Each week: read last week's closes, ask the policy for target weights, size the
sleeve, plan orders, run pre-trade checks, ask for approval when turnover is
unusually high, then fill. The deployed version runs the same steps as a
scheduled container against a broker's paper account.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rl_portfolio.envs.sizing import PortfolioSizer
from rl_portfolio.metrics import summarize
from rl_portfolio.paper.broker import SimulatedBroker
from rl_portfolio.paper.rebalance import RebalancePlan, check_orders, plan_rebalance

logger = logging.getLogger(__name__)

WeightsFn = Callable[[pd.DataFrame, dict[str, float] | None], dict[str, float]]
ApprovalFn = Callable[[RebalancePlan], bool]


@dataclass
class PaperResult:
    dates: list[str]
    equity: list[float]
    weights: list[dict[str, float]] = field(default_factory=list)
    held_for_approval: int = 0
    rejected_by_checks: int = 0
    costs: float = 0.0

    @property
    def returns(self) -> np.ndarray:
        eq = np.asarray(self.equity)
        return eq[1:] / eq[:-1] - 1.0

    @property
    def metrics(self) -> dict[str, float]:
        out = summarize(self.returns)
        out.update(held_for_approval=self.held_for_approval, costs=self.costs)
        return out


def prices_from_returns(returns: pd.DataFrame, base: float = 100.0) -> pd.DataFrame:
    return base * (1.0 + returns).cumprod()


def current_weights(
    holdings: dict[str, float], prices: dict[str, float]
) -> dict[str, float] | None:
    values = {t: q * prices[t] for t, q in holdings.items()}
    total = sum(values.values())
    return None if total <= 0 else {t: v / total for t, v in values.items()}


def run_paper_trading(
    weights_fn: WeightsFn,
    returns: pd.DataFrame,
    start: int,
    sizer: PortfolioSizer,
    initial_cash: float,
    cost_rate: float = 0.001,
    max_turnover: float = 0.5,
    approve: ApprovalFn | None = None,
) -> PaperResult:
    """Simulate weekly rebalancing from week `start` to the end of `returns`.

    `approve` stands in for the human approval step: it is only consulted when
    a plan's turnover exceeds `max_turnover`. Without it, such plans are held.
    """
    if not 1 <= start < len(returns):
        raise ValueError("start must leave at least one week to trade")
    prices = prices_from_returns(returns)
    broker = SimulatedBroker(cash=initial_cash, cost_rate=cost_rate)

    first_prices = prices.iloc[start - 1].to_dict()
    result = PaperResult(
        dates=[str(returns.index[start - 1])], equity=[broker.equity(first_prices)]
    )

    for t in range(start, len(returns)):
        last_close = prices.iloc[t - 1].to_dict()
        held = current_weights(broker.positions(), last_close)
        target = weights_fn(returns.iloc[:t], held)
        managed = sizer.managed_capital(broker.equity(last_close))
        plan = plan_rebalance(
            target, broker.positions(), last_close, managed, cost_rate, cash=broker.cash
        )

        if check_orders(plan):
            result.rejected_by_checks += 1
        elif held is not None and plan.turnover > max_turnover and not (approve and approve(plan)):
            result.held_for_approval += 1
        else:
            fills = broker.submit(plan.orders, last_close)
            result.costs += sum(f.cost for f in fills)

        close = prices.iloc[t].to_dict()
        result.dates.append(str(returns.index[t]))
        result.equity.append(broker.equity(close))
        result.weights.append(target)

    logger.info(
        "Paper run over %d weeks, %d plans held for approval",
        len(result.dates) - 1,
        result.held_for_approval,
    )
    return result


def equal_weight_fn(tickers: list[str]) -> WeightsFn:
    weights = {t: 1.0 / len(tickers) for t in tickers}
    return lambda _returns, _held: weights
