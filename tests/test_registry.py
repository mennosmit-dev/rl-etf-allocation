import pytest

from rl_portfolio.registry import EvaluationGate, ModelRecord, ModelRegistry, PromotionError


def record(tmp_path, model_id: str, sharpe: float, ew_sharpe: float = 0.5, dd: float = 0.1):
    model_file = tmp_path / f"{model_id}.zip"
    model_file.write_bytes(b"stub")
    return ModelRecord(
        model_id=model_id,
        model_path=model_file.as_posix(),
        tickers=["A", "B"],
        features=[{"name": "trailing_return", "window": 4}],
        dataset_hash="abc",
        config_hash="def",
        train_window=["2020-01-03", "2023-01-06"],
        validation_window=["2023-01-13", "2024-01-05"],
        validation_metrics={"sharpe": sharpe, "max_drawdown": dd},
        equal_weight_metrics={"sharpe": ew_sharpe, "max_drawdown": 0.1},
    )


def test_promote_then_rollback(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    gate = EvaluationGate(min_sharpe_edge=0.0)
    registry.register_candidate(record(tmp_path, "m1", sharpe=0.8))
    registry.register_candidate(record(tmp_path, "m2", sharpe=0.9))

    registry.promote("m1", gate)
    registry.promote("m2", gate)
    assert registry.active().model_id == "m2"
    assert registry.get("m1").status == "retired"

    registry.rollback(reason="bad live behaviour")
    assert registry.active().model_id == "m1"
    assert [e["action"] for e in registry.history()] == ["promote", "promote", "rollback"]


def test_gate_rejects_candidate_that_loses_to_equal_weight(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    registry.register_candidate(record(tmp_path, "weak", sharpe=0.3, ew_sharpe=0.5))
    with pytest.raises(PromotionError, match="Sharpe edge"):
        registry.promote("weak", EvaluationGate(min_sharpe_edge=0.0))
    assert registry.get("weak").status == "rejected"
    assert registry.active() is None


def test_gate_rejects_large_drawdown_and_missing_file(tmp_path):
    rec = record(tmp_path, "risky", sharpe=1.0, dd=0.4)
    (tmp_path / "risky.zip").unlink()
    result = EvaluationGate().check(rec)
    assert not result.passed
    assert len(result.reasons) == 2


def test_cannot_promote_twice(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    registry.register_candidate(record(tmp_path, "m1", sharpe=0.8))
    registry.promote("m1", EvaluationGate())
    with pytest.raises(PromotionError, match="status active"):
        registry.promote("m1", EvaluationGate())


def test_rollback_without_history_fails(tmp_path):
    with pytest.raises(PromotionError):
        ModelRegistry(tmp_path / "registry").rollback()
