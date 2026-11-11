"""File-based model registry with an explicit promotion gate.

Every trained model becomes a candidate. Only a candidate that passes the gate
can become active, and every promotion is appended to a history file so a bad
model can be rolled back to the previous one. The serving layer and the paper
trader only ever load the active model.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

Status = Literal["candidate", "active", "rejected", "retired"]


@dataclass
class ModelRecord:
    model_id: str
    model_path: str
    tickers: list[str]
    features: list[dict]
    dataset_hash: str
    config_hash: str
    train_window: list[str]
    validation_window: list[str]
    validation_metrics: dict[str, float]
    equal_weight_metrics: dict[str, float]
    mlflow_run_id: str | None = None
    status: Status = "candidate"
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )
    gate_reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reasons: list[str]


@dataclass(frozen=True)
class EvaluationGate:
    """Out-of-sample checks a candidate must pass before it can trade, even on paper."""

    min_sharpe_edge: float = 0.0
    max_drawdown_slack: float = 0.10

    def check(self, record: ModelRecord) -> GateResult:
        reasons: list[str] = []
        if not Path(record.model_path).exists():
            reasons.append(f"model file missing: {record.model_path}")
        rl, ew = record.validation_metrics, record.equal_weight_metrics
        edge = rl.get("sharpe", float("-inf")) - ew.get("sharpe", 0.0)
        if edge < self.min_sharpe_edge:
            reasons.append(f"validation Sharpe edge {edge:.3f} below {self.min_sharpe_edge:.3f}")
        dd_gap = rl.get("max_drawdown", 1.0) - ew.get("max_drawdown", 0.0)
        if dd_gap > self.max_drawdown_slack:
            reasons.append(f"max drawdown {dd_gap:.1%} worse than equal weight")
        return GateResult(passed=not reasons, reasons=reasons)


class PromotionError(RuntimeError):
    pass


class ModelRegistry:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.models_dir = self.root / "models"
        self.active_file = self.root / "active.json"
        self.history_file = self.root / "history.jsonl"

    def _path(self, model_id: str) -> Path:
        return self.models_dir / f"{model_id}.json"

    def save(self, record: ModelRecord) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._path(record.model_id).write_text(json.dumps(asdict(record), indent=2), "utf-8")

    def get(self, model_id: str) -> ModelRecord:
        path = self._path(model_id)
        if not path.exists():
            raise KeyError(f"unknown model: {model_id}")
        return ModelRecord(**json.loads(path.read_text("utf-8")))

    def list(self) -> list[ModelRecord]:
        if not self.models_dir.exists():
            return []
        records = [
            ModelRecord(**json.loads(p.read_text("utf-8"))) for p in self.models_dir.glob("*.json")
        ]
        return sorted(records, key=lambda r: r.created_at_utc)

    def register_candidate(self, record: ModelRecord) -> ModelRecord:
        record.status = "candidate"
        self.save(record)
        logger.info("Registered candidate %s", record.model_id)
        return record

    def active(self) -> ModelRecord | None:
        if not self.active_file.exists():
            return None
        pointer = json.loads(self.active_file.read_text("utf-8"))
        return self.get(pointer["model_id"])

    def promote(self, model_id: str, gate: EvaluationGate, reason: str = "") -> ModelRecord:
        record = self.get(model_id)
        if record.status != "candidate":
            raise PromotionError(f"{model_id} has status {record.status}, expected candidate")
        result = gate.check(record)
        record.gate_reasons = result.reasons
        if not result.passed:
            record.status = "rejected"
            self.save(record)
            raise PromotionError(f"{model_id} rejected: {'; '.join(result.reasons)}")
        self._activate(record, action="promote", reason=reason)
        return record

    def rollback(self, reason: str = "") -> ModelRecord:
        """Re-activate the model that was active before the current one."""
        events = self.history()
        current = self.active()
        previous = [e for e in events if current is None or e["model_id"] != current.model_id]
        if not previous:
            raise PromotionError("no earlier active model to roll back to")
        record = self.get(previous[-1]["model_id"])
        self._activate(record, action="rollback", reason=reason)
        return record

    def history(self) -> list[dict]:
        if not self.history_file.exists():
            return []
        lines = self.history_file.read_text("utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def _activate(self, record: ModelRecord, action: str, reason: str) -> None:
        current = self.active()
        if current is not None and current.model_id != record.model_id:
            current.status = "retired"
            self.save(current)
        record.status = "active"
        self.save(record)
        self.root.mkdir(parents=True, exist_ok=True)
        now = datetime.now(UTC).isoformat(timespec="seconds")
        self.active_file.write_text(
            json.dumps({"model_id": record.model_id, "since_utc": now}, indent=2), "utf-8"
        )
        event = {"model_id": record.model_id, "action": action, "reason": reason, "at_utc": now}
        with self.history_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event) + "\n")
        logger.info("%s: %s is now active", action, record.model_id)
