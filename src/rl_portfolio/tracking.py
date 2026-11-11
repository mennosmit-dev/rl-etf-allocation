"""Thin MLflow wrapper so training code does not depend on MLflow being enabled."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rl_portfolio.config import TrackingConfig

logger = logging.getLogger(__name__)


class Run:
    def __init__(self, mlflow_module: Any | None) -> None:
        self._mlflow = mlflow_module
        self.run_id: str | None = None
        if mlflow_module is not None:
            self.run_id = mlflow_module.active_run().info.run_id

    def log_params(self, params: dict[str, Any]) -> None:
        if self._mlflow:
            self._mlflow.log_params({k: str(v)[:500] for k, v in params.items()})

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self._mlflow:
            self._mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def log_artifact(self, path: Path) -> None:
        if self._mlflow:
            self._mlflow.log_artifact(str(path))

    def set_tags(self, tags: dict[str, str]) -> None:
        if self._mlflow:
            self._mlflow.set_tags(tags)


@contextmanager
def tracking_run(config: TrackingConfig, run_name: str, nested: bool = False) -> Iterator[Run]:
    if not config.enabled:
        yield Run(None)
        return

    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    mlflow.set_tracking_uri(config.uri)
    mlflow.set_experiment(config.experiment)
    with mlflow.start_run(run_name=run_name, nested=nested):
        run = Run(mlflow)
        logger.info("MLflow run %s (%s)", run_name, run.run_id)
        yield run
