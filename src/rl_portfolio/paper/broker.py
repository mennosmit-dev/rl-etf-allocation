"""Broker interface and an in-memory simulated broker for paper trading.

A real broker adapter (the deployment uses a paper-trading API) implements the
same three methods; credentials come from the environment, never from files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol


@dataclass(frozen=True)
class Order:
    ticker: str
    side: Literal["buy", "sell"]
    quantity: float
    reference_price: float

    @property
    def notional(self) -> float:
        return self.quantity * self.reference_price


@dataclass(frozen=True)
class Fill:
    order: Order
    price: float
    cost: float


class Broker(Protocol):
    def positions(self) -> dict[str, float]: ...

    def equity(self, prices: dict[str, float]) -> float: ...

    def submit(self, orders: list[Order], prices: dict[str, float]) -> list[Fill]: ...


@dataclass
class SimulatedBroker:
    """Fills market orders at the given price plus a proportional cost."""

    cash: float
    cost_rate: float = 0.001
    holdings: dict[str, float] = field(default_factory=dict)

    def positions(self) -> dict[str, float]:
        return dict(self.holdings)

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + sum(q * prices[t] for t, q in self.holdings.items())

    def submit(self, orders: list[Order], prices: dict[str, float]) -> list[Fill]:
        fills = []
        # Sells first so buys are funded by the proceeds.
        for order in sorted(orders, key=lambda o: o.side != "sell"):
            price = prices[order.ticker]
            value = order.quantity * price
            cost = value * self.cost_rate
            held = self.holdings.get(order.ticker, 0.0)
            if order.side == "sell":
                if order.quantity > held + 1e-9:
                    raise ValueError(f"cannot sell {order.quantity} {order.ticker}, hold {held}")
                self.holdings[order.ticker] = held - order.quantity
                self.cash += value - cost
            else:
                if value + cost > self.cash + 1e-6:
                    raise ValueError(f"insufficient cash for {order.ticker}")
                self.holdings[order.ticker] = held + order.quantity
                self.cash -= value + cost
            fills.append(Fill(order, price, cost))
        return fills
