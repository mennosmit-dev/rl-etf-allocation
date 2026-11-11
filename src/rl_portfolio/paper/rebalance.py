"""Turn target weights into orders, with pre-trade checks."""

from __future__ import annotations

from dataclasses import dataclass

from rl_portfolio.paper.broker import Order


@dataclass(frozen=True)
class RebalancePlan:
    orders: list[Order]
    turnover: float
    managed_capital: float


def plan_rebalance(
    target_weights: dict[str, float],
    holdings: dict[str, float],
    prices: dict[str, float],
    managed_capital: float,
    cost_rate: float = 0.0,
    min_trade_notional: float = 10.0,
    cash: float | None = None,
) -> RebalancePlan:
    """Orders that move the managed sleeve to `target_weights`.

    The cost buffer scales targets down slightly so the buys stay fundable
    after the costs of the trades themselves. When `cash` is given, buys are
    scaled down if skipped small sells leave them underfunded.
    """
    total = sum(target_weights.values())
    if total <= 0 or any(w < 0 for w in target_weights.values()):
        raise ValueError("target weights must be non-negative with a positive sum")
    missing = [t for t in target_weights if t not in prices]
    if missing:
        raise ValueError(f"no price for {missing}")

    investable = managed_capital * (1.0 - 2 * cost_rate)
    orders: list[Order] = []
    traded = 0.0
    for ticker, weight in target_weights.items():
        price = prices[ticker]
        current_value = holdings.get(ticker, 0.0) * price
        delta = investable * weight / total - current_value
        if abs(delta) < min_trade_notional:
            continue
        side = "buy" if delta > 0 else "sell"
        orders.append(Order(ticker, side, abs(delta) / price, price))
        traded += abs(delta)
    if cash is not None:
        orders = _fund_buys(orders, cash, cost_rate)
    turnover = traded / managed_capital if managed_capital > 0 else 0.0
    return RebalancePlan(orders, turnover, managed_capital)


def _fund_buys(orders: list[Order], cash: float, cost_rate: float) -> list[Order]:
    proceeds = sum(o.notional for o in orders if o.side == "sell") * (1 - cost_rate)
    needed = sum(o.notional for o in orders if o.side == "buy") * (1 + cost_rate)
    available = cash + proceeds
    if needed <= available or needed == 0:
        return orders
    scale = max(available, 0.0) / needed * (1 - 1e-9)
    return [
        o if o.side == "sell" else Order(o.ticker, o.side, o.quantity * scale, o.reference_price)
        for o in orders
    ]


def check_orders(plan: RebalancePlan, max_order_fraction: float = 1.0) -> list[str]:
    """Pre-trade risk checks. Returns the list of violations (empty means OK)."""
    problems = []
    for order in plan.orders:
        if order.quantity <= 0:
            problems.append(f"{order.ticker}: non-positive quantity")
        if order.side == "buy" and order.notional > max_order_fraction * plan.managed_capital:
            problems.append(f"{order.ticker}: order is {order.notional:.0f}, above limit")
    return problems
