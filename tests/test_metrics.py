import numpy as np
import pytest

from rl_portfolio.metrics import annualized_volatility, cagr, max_drawdown, sharpe_ratio


def test_cagr_of_constant_return():
    assert cagr(np.full(52, 0.01)) == pytest.approx(1.01**52 - 1)


def test_max_drawdown_simple_path():
    # wealth: 1.1, 0.88, 0.968 -> peak 1.1, trough 0.88 -> 20 percent
    assert max_drawdown(np.array([0.10, -0.20, 0.10])) == pytest.approx(0.20)
    assert max_drawdown(np.array([0.01, 0.02])) == 0.0


def test_sharpe_and_volatility_edge_cases():
    assert sharpe_ratio(np.full(10, 0.01)) == 0.0
    assert annualized_volatility(np.array([0.01])) == 0.0
    rng = np.random.default_rng(3)
    r = rng.normal(0.003, 0.02, 5000)
    assert sharpe_ratio(r) == pytest.approx(0.003 / 0.02 * np.sqrt(52), rel=0.15)
