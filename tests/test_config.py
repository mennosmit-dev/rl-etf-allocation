from pathlib import Path

import pytest
from pydantic import ValidationError

from rl_portfolio.config import load_config

EXAMPLE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "example.yaml"


def test_example_config_loads():
    config = load_config(EXAMPLE_CONFIG)
    assert config.data.source == "synthetic"
    assert len(config.data.tickers) >= 2
    assert config.max_lookback == max(f.window for f in config.features)


def test_overrides_merge_nested_keys():
    config = load_config(EXAMPLE_CONFIG, overrides={"ppo": {"total_timesteps": 10}})
    assert config.ppo.total_timesteps == 10
    assert config.ppo.learning_rate > 0


def test_invalid_values_are_rejected():
    with pytest.raises(ValidationError):
        load_config(EXAMPLE_CONFIG, overrides={"reward": {"transaction_cost": -1}})
