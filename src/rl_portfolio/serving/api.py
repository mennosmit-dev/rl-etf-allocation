"""FastAPI service that serves allocations from the active registered model."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rl_portfolio.registry import ModelRegistry
from rl_portfolio.serving.inference import AllocationPolicy


class AllocationRequest(BaseModel):
    tickers: list[str] = Field(min_length=2)
    returns: list[list[float]] = Field(
        description="Weekly returns, oldest first, one row per week, columns in `tickers` order"
    )
    current_weights: dict[str, float] | None = None


class AllocationResponse(BaseModel):
    model_id: str
    weights: dict[str, float]


def create_app(registry_root: Path | None = None) -> FastAPI:
    registry = ModelRegistry(registry_root or Path(os.getenv("REGISTRY_PATH", "registry")))
    app = FastAPI(title="RL ETF allocation", version="0.1.0")
    cache: dict[str, AllocationPolicy] = {}

    def active_policy() -> AllocationPolicy:
        record = registry.active()
        if record is None:
            raise HTTPException(
                status_code=503, detail="no active model; promote a candidate first"
            )
        if record.model_id not in cache:
            cache.clear()
            cache[record.model_id] = AllocationPolicy(record)
        return cache[record.model_id]

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/model")
    def model() -> dict:
        record = registry.active()
        if record is None:
            raise HTTPException(status_code=503, detail="no active model")
        return {
            "model_id": record.model_id,
            "tickers": record.tickers,
            "train_window": record.train_window,
            "validation_window": record.validation_window,
            "validation_metrics": record.validation_metrics,
            "equal_weight_metrics": record.equal_weight_metrics,
        }

    @app.post("/allocate", response_model=AllocationResponse)
    def allocate(req: AllocationRequest) -> AllocationResponse:
        policy = active_policy()
        if any(len(row) != len(req.tickers) for row in req.returns):
            raise HTTPException(status_code=422, detail="each returns row must match tickers")
        frame = pd.DataFrame(req.returns, columns=req.tickers)
        try:
            weights = policy.target_weights(frame, req.current_weights)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return AllocationResponse(model_id=policy.record.model_id, weights=weights)

    return app
