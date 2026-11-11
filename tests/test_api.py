import numpy as np
import pytest
from fastapi.testclient import TestClient

from rl_portfolio.registry import EvaluationGate, ModelRegistry
from rl_portfolio.serving.api import create_app
from rl_portfolio.training.pipeline import train_and_register


@pytest.fixture
def active_registry(config, tmp_path):
    record = train_and_register(config, models_dir=tmp_path / "models")
    registry = ModelRegistry(config.registry.path)
    registry.promote(record.model_id, EvaluationGate(min_sharpe_edge=-10.0, max_drawdown_slack=1.0))
    return config.registry.path, record


def test_health_and_no_model(tmp_path):
    client = TestClient(create_app(tmp_path / "empty"))
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/model").status_code == 503


def test_allocate_returns_valid_weights(active_registry):
    root, record = active_registry
    client = TestClient(create_app(root))
    assert client.get("/model").json()["model_id"] == record.model_id

    rng = np.random.default_rng(0)
    payload = {
        "tickers": record.tickers,
        "returns": rng.normal(0.001, 0.02, size=(40, len(record.tickers))).tolist(),
    }
    response = client.post("/allocate", json=payload)
    assert response.status_code == 200
    weights = response.json()["weights"]
    assert set(weights) == set(record.tickers)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert min(weights.values()) >= 0


def test_allocate_rejects_short_history(active_registry):
    root, record = active_registry
    client = TestClient(create_app(root))
    payload = {"tickers": record.tickers, "returns": [[0.0] * len(record.tickers)] * 3}
    assert client.post("/allocate", json=payload).status_code == 422
