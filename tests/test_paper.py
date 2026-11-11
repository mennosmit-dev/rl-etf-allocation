import numpy as np
import pandas as pd
import pytest

from rl_portfolio.envs.sizing import FixedSizer
from rl_portfolio.paper.broker import Order, SimulatedBroker
from rl_portfolio.paper.rebalance import check_orders, plan_rebalance
from rl_portfolio.paper.simulate import equal_weight_fn, run_paper_trading


@pytest.fixture
def returns() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    data = rng.normal(0.001, 0.02, size=(60, 3))
    return pd.DataFrame(
        data,
        columns=["A", "B", "C"],
        index=pd.date_range("2020-01-03", periods=60, freq="W-FRI").strftime("%Y-%m-%d"),
    )


def test_plan_moves_cash_into_targets():
    prices = {"A": 50.0, "B": 25.0}
    plan = plan_rebalance({"A": 0.5, "B": 0.5}, {}, prices, managed_capital=1000)
    assert {o.ticker: o.quantity for o in plan.orders} == {"A": 10.0, "B": 20.0}
    assert plan.turnover == pytest.approx(1.0)
    assert check_orders(plan) == []


def test_plan_skips_tiny_trades():
    prices = {"A": 10.0, "B": 10.0}
    plan = plan_rebalance({"A": 0.5, "B": 0.5}, {"A": 50.5, "B": 49.5}, prices, 1000)
    assert plan.orders == []


def test_plan_rejects_bad_weights():
    with pytest.raises(ValueError):
        plan_rebalance({"A": -0.1, "B": 1.1}, {}, {"A": 1.0, "B": 1.0}, 100)


def test_broker_fills_and_charges_costs():
    broker = SimulatedBroker(cash=1000, cost_rate=0.01)
    broker.submit([Order("A", "buy", 5, 100.0)], {"A": 100.0})
    assert broker.cash == pytest.approx(1000 - 500 - 5)
    with pytest.raises(ValueError):
        broker.submit([Order("A", "sell", 6, 100.0)], {"A": 100.0})


def test_equal_weight_paper_run_tracks_benchmark(returns):
    result = run_paper_trading(
        equal_weight_fn(list(returns.columns)),
        returns,
        start=10,
        sizer=FixedSizer(10_000),
        initial_cash=10_000,
        cost_rate=0.0,
    )
    assert len(result.equity) == len(returns) - 10 + 1
    # with zero costs the sleeve should track a weekly-rebalanced 1/N portfolio closely
    ew = returns.iloc[10:].mean(axis=1).to_numpy()
    np.testing.assert_allclose(result.returns, ew, atol=2e-3)


def test_large_reallocations_wait_for_approval(returns):
    calls = {"n": 0}

    def switch_after_funding(_returns, _held):
        calls["n"] += 1
        return {"A": 1.0, "B": 0.0, "C": 0.0} if calls["n"] == 1 else {"A": 0.0, "B": 0.0, "C": 1.0}

    kwargs = dict(returns=returns, start=10, sizer=FixedSizer(10_000), initial_cash=10_000)
    held = run_paper_trading(switch_after_funding, max_turnover=0.5, **kwargs)
    assert held.held_for_approval == len(returns) - 11  # every week after the initial funding

    calls["n"] = 0
    approved = run_paper_trading(
        switch_after_funding, max_turnover=0.5, approve=lambda plan: True, **kwargs
    )
    assert approved.held_for_approval == 0
    assert approved.weights[-1]["C"] == 1.0


def test_buys_are_scaled_to_available_cash():
    prices = {"A": 10.0, "B": 10.0}
    # A should be sold by 5 (below the minimum trade), B bought by 15 with only 10 in cash
    plan = plan_rebalance({"A": 0.5, "B": 0.5}, {"A": 50.5, "B": 49.0}, prices, 1000.0, cash=10.0)
    buy = [o for o in plan.orders if o.side == "buy"]
    assert sum(o.notional for o in buy) <= 10.0
