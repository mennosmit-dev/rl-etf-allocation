import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rl_portfolio.envs.portfolio_env import PortfolioEnv, drift_weights, softmax
from rl_portfolio.envs.reward import MeanVarianceUtility
from rl_portfolio.envs.sizing import FixedSizer


def make_env(cost=0.001, reward=None, rows=40, assets=3):
    rng = np.random.default_rng(1)
    returns = rng.normal(0.001, 0.02, size=(rows, assets))
    features = rng.normal(size=(rows, 2 * assets))
    return PortfolioEnv(
        returns, features, start=1, end=rows, reward_fn=reward, transaction_cost=cost
    )


def test_passes_gymnasium_checker():
    check_env(make_env(), skip_render_check=True)


def test_softmax_gives_valid_weights():
    w = softmax(np.array([3.0, -1.0, 0.5]))
    assert w.sum() == pytest.approx(1.0)
    assert (w > 0).all()
    assert softmax(np.zeros(4)) == pytest.approx(np.full(4, 0.25))


def test_drift_weights_follow_prices():
    w = drift_weights(np.array([0.5, 0.5]), np.array([0.10, -0.10]))
    assert w == pytest.approx([0.55, 0.45])


def test_step_accounting_matches_formula():
    env = make_env(cost=0.01, reward=MeanVarianceUtility(risk_aversion=2.0))
    env.reset()
    action = np.array([2.0, 0.0, -2.0], dtype=np.float32)
    _, reward, _, _, info = env.step(action)
    w = softmax(action)
    turnover = np.abs(w - np.full(3, 1 / 3)).sum()
    net = w @ env.returns[1] - 0.01 * turnover
    assert info["turnover"] == pytest.approx(turnover)
    assert info["net_return"] == pytest.approx(net)
    assert reward == pytest.approx(net - 0.5 * 2.0 * net**2)


def test_observation_uses_previous_week_features():
    env = make_env()
    obs, _ = env.reset()
    np.testing.assert_allclose(obs[:6], env.features[0].astype(np.float32))


def test_episode_length():
    env = make_env(rows=30)
    env.reset()
    steps, done = 0, False
    while not done:
        _, _, done, _, _ = env.step(env.action_space.sample())
        steps += 1
    assert steps == 29


def test_rejects_nan_features_in_window():
    returns = np.zeros((10, 2))
    features = np.zeros((10, 2))
    features[3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        PortfolioEnv(returns, features, start=1, end=10)


def test_fixed_sizer_caps_at_account_equity():
    sizer = FixedSizer(capital=10_000)
    assert sizer.managed_capital(25_000) == 10_000
    assert sizer.managed_capital(4_000) == 4_000
    with pytest.raises(ValueError):
        sizer.managed_capital(-1)
