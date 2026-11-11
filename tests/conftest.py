from pathlib import Path

import pytest

from rl_portfolio.config import Config, load_config

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = ROOT / "configs" / "example.yaml"


@pytest.fixture
def config(tmp_path: Path) -> Config:
    """The example config, shrunk so a full train/evaluate cycle takes seconds."""
    return load_config(
        EXAMPLE_CONFIG,
        overrides={
            "data": {"data_dir": str(tmp_path / "data"), "synthetic_weeks": 200},
            "ppo": {"total_timesteps": 64, "n_steps": 32, "batch_size": 16, "net_arch": [16]},
            "rolling": {"train_weeks": 60, "validation_weeks": 26, "step_weeks": 26},
            "registry": {"path": str(tmp_path / "registry"), "min_sharpe_edge": -10.0},
            "tracking": {"enabled": False},
        },
    )
