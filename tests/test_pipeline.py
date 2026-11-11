"""Walk-forward retraining and candidate registration."""

import pytest

from rl_portfolio.data.dataset import get_or_build_dataset
from rl_portfolio.registry import ModelRegistry
from rl_portfolio.training.pipeline import rolling_retrain, train_and_register, walk_forward_windows
from rl_portfolio.training.train import prepare


def test_train_registers_candidate(config, tmp_path):
    record = train_and_register(config, models_dir=tmp_path / "models")
    assert record.status == "candidate"
    assert (tmp_path / "models" / f"{record.model_id}.zip").exists()
    assert ModelRegistry(config.registry.path).get(record.model_id).dataset_hash
    assert record.validation_metrics["weeks"] == config.rolling.validation_weeks


def test_walk_forward_windows_tile_the_timeline(config):
    returns, _ = get_or_build_dataset(config)
    prep = prepare(config, returns)
    windows = walk_forward_windows(config, prep)
    assert len(windows) >= 2
    for (_, _, prev_val_end), (_, next_train_end, _) in zip(windows, windows[1:], strict=False):
        assert next_train_end == prev_val_end
    assert all(end - start == config.rolling.train_weeks for start, end, _ in windows)


@pytest.mark.slow
def test_rolling_retrain_smoke(config, tmp_path):
    result = rolling_retrain(config, models_dir=tmp_path / "models")
    total = sum(len(w.rl.net_returns) for w in result.windows)
    assert result.rl_metrics["weeks"] == total
    assert result.candidate.status == "candidate"
