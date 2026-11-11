"""Training smoke tests: real PPO runs with very few timesteps."""

import numpy as np

from rl_portfolio.data.dataset import get_or_build_dataset
from rl_portfolio.training.backtest import equal_weight_policy, run_policy
from rl_portfolio.training.train import latest_split, make_env, prepare, train_window


def test_prepare_marks_first_valid_week(config):
    returns, _ = get_or_build_dataset(config)
    prep = prepare(config, returns)
    # features need `max_lookback` returns; the first decision uses the row before it
    assert prep.first_valid == config.max_lookback
    assert not np.isnan(prep.features[prep.first_valid - 1]).any()


def test_equal_weight_backtest_has_small_turnover(config):
    returns, _ = get_or_build_dataset(config)
    prep = prepare(config, returns)
    env = make_env(config, prep, prep.first_valid, len(prep.returns))
    result = run_policy(equal_weight_policy(len(prep.tickers)), env)
    assert len(result.net_returns) == len(prep.returns) - prep.first_valid
    assert result.turnover.max() < 0.2  # only drift back to 1/N
    np.testing.assert_allclose(result.weights.sum(axis=1), 1.0)


def test_latest_split_is_contiguous(config):
    returns, _ = get_or_build_dataset(config)
    prep = prepare(config, returns)
    start, train_end, val_end = latest_split(config, prep)
    assert val_end == len(prep.returns)
    assert val_end - train_end == config.rolling.validation_weeks
    assert start >= prep.first_valid


def test_smoke_train_saves_model_and_scores_validation(config, tmp_path):
    returns, meta = get_or_build_dataset(config)
    prep = prepare(config, returns)
    result = train_window(config, prep, meta, *latest_split(config, prep), models_dir=tmp_path)
    assert result.model_path.exists()
    assert len(result.rl.net_returns) == config.rolling.validation_weeks
    assert set(result.rl.metrics) >= {"sharpe", "cagr", "max_drawdown"}
    np.testing.assert_allclose(result.rl.weights.sum(axis=1), 1.0, rtol=1e-6)


def test_mlflow_tracking_logs_run(config, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # MLflow writes artifacts relative to the working directory
    import mlflow

    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    tracking = config.tracking.model_copy(update={"enabled": True, "uri": uri, "experiment": "t"})
    tracked = config.model_copy(update={"tracking": tracking})
    returns, meta = get_or_build_dataset(tracked)
    prep = prepare(tracked, returns)
    result = train_window(tracked, prep, meta, *latest_split(tracked, prep), models_dir=tmp_path)
    run = mlflow.MlflowClient(tracking_uri=uri).get_run(result.run_id)
    assert "val_sharpe" in run.data.metrics
    assert run.data.tags["dataset_hash"] == meta.content_hash
