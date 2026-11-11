"""Training pipelines: a single latest-window run and a walk-forward retrain."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rl_portfolio.config import Config
from rl_portfolio.data.dataset import DatasetMetadata, get_or_build_dataset
from rl_portfolio.metrics import summarize
from rl_portfolio.registry import ModelRecord, ModelRegistry
from rl_portfolio.tracking import tracking_run
from rl_portfolio.training.train import (
    MODELS_DIR,
    Prepared,
    WindowResult,
    config_fingerprint,
    latest_split,
    prepare,
    train_window,
)

logger = logging.getLogger(__name__)


def to_record(
    config: Config, prep: Prepared, meta: DatasetMetadata, result: WindowResult
) -> ModelRecord:
    assert result.model_path is not None
    return ModelRecord(
        model_id=result.model_path.stem,
        model_path=result.model_path.as_posix(),
        tickers=prep.tickers,
        features=[spec.model_dump() for spec in config.features],
        dataset_hash=meta.content_hash,
        config_hash=config_fingerprint(config),
        train_window=list(result.train_window),
        validation_window=list(result.validation_window),
        validation_metrics=result.rl.metrics,
        equal_weight_metrics=result.equal_weight.metrics,
        mlflow_run_id=result.run_id,
    )


def train_and_register(config: Config, models_dir: Path = MODELS_DIR) -> ModelRecord:
    returns, meta = get_or_build_dataset(config)
    prep = prepare(config, returns)
    start, train_end, val_end = latest_split(config, prep)
    result = train_window(config, prep, meta, start, train_end, val_end, models_dir=models_dir)
    record = to_record(config, prep, meta, result)
    return ModelRegistry(config.registry.path).register_candidate(record)


@dataclass
class WalkForwardResult:
    windows: list[WindowResult]
    rl_metrics: dict[str, float]
    equal_weight_metrics: dict[str, float]
    candidate: ModelRecord


def walk_forward_windows(config: Config, prep: Prepared) -> list[tuple[int, int, int]]:
    """Fixed-length train windows that roll forward by `step_weeks`.

    Each window is followed by a validation window the model never saw. The
    validation windows tile the out-of-sample period without overlap when
    step_weeks equals validation_weeks.
    """
    r = config.rolling
    windows = []
    train_start = prep.first_valid
    while True:
        train_end = train_start + r.train_weeks
        val_end = train_end + r.validation_weeks
        if val_end > len(prep.returns):
            break
        windows.append((train_start, train_end, val_end))
        train_start += r.step_weeks
    if not windows:
        raise ValueError("dataset too short for one train plus validation window")
    return windows


def rolling_retrain(config: Config, models_dir: Path = MODELS_DIR) -> WalkForwardResult:
    """Retrain on each rolling window and stitch the validation periods together.

    The stitched series is the honest out-of-sample record of the procedure.
    The model from the last window becomes the new candidate.
    """
    returns, meta = get_or_build_dataset(config)
    prep = prepare(config, returns)
    windows = walk_forward_windows(config, prep)
    logger.info("Walk-forward retrain over %d windows", len(windows))

    results: list[WindowResult] = []
    with tracking_run(config.tracking, "walk-forward") as parent:
        parent.log_params({"windows": len(windows), **config.rolling.model_dump()})
        for i, (start, train_end, val_end) in enumerate(windows):
            res = train_window(
                config,
                prep,
                meta,
                start,
                train_end,
                val_end,
                run_name=f"window-{i:02d}-{prep.dates[val_end - 1]}",
                nested=config.tracking.enabled,
                models_dir=models_dir,
            )
            results.append(res)

        rl = summarize(
            np.concatenate([r.rl.net_returns for r in results]),
            np.concatenate([r.rl.turnover for r in results]),
        )
        ew = summarize(
            np.concatenate([r.equal_weight.net_returns for r in results]),
            np.concatenate([r.equal_weight.turnover for r in results]),
        )
        parent.log_metrics({f"oos_{k}": v for k, v in rl.items()})
        parent.log_metrics({f"oos_ew_{k}": v for k, v in ew.items()})

    logger.info("Out of sample: Sharpe %.2f vs equal weight %.2f", rl["sharpe"], ew["sharpe"])
    record = to_record(config, prep, meta, results[-1])
    ModelRegistry(config.registry.path).register_candidate(record)
    return WalkForwardResult(results, rl, ew, record)
