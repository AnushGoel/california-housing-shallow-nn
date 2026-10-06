"""REST API for the California housing shallow network.

Run locally:   uvicorn api.main:app --reload      then open http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from housing_app import __version__
from housing_app.artifacts import ArtifactError
from housing_app.config import APP_ROOT
from housing_app.service import PredictionService

Model = Literal["final", "tuned"]


class BlockGroup(BaseModel):
    """One census block group. Omitted features take the training median."""

    MedInc: Optional[float] = Field(None, gt=0, le=30, description="median household income, in $10,000s")
    HouseAge: Optional[float] = Field(None, ge=0, le=150, description="median house age, in years")
    AveRooms: Optional[float] = Field(None, gt=0, le=500, description="average rooms per household")
    AveBedrms: Optional[float] = Field(None, gt=0, le=200, description="average bedrooms per household")
    Population: Optional[float] = Field(None, ge=1, le=100_000, description="residents of the block group")
    AveOccup: Optional[float] = Field(None, gt=0, le=5_000, description="average people per household")
    Latitude: Optional[float] = Field(None, ge=30, le=45, description="degrees north")
    Longitude: Optional[float] = Field(None, ge=-130, le=-110, description="degrees east (negative = west)")


class PredictRequest(BaseModel):
    block_groups: List[BlockGroup] = Field(..., min_length=1, max_length=1000)
    model: Model = "final"
    level: float = Field(0.9, gt=0, lt=1, description="coverage of the conformal prediction interval")


@lru_cache(maxsize=1)
def service() -> PredictionService:
    return PredictionService.from_dir(Path(os.environ.get("ARTIFACTS_DIR", APP_ROOT / "artifacts")))


def ready() -> PredictionService:
    try:
        return service()
    except ArtifactError as err:
        raise HTTPException(status_code=503, detail=f"artifacts unavailable: {err}") from err


app = FastAPI(title="California housing shallow network", version=__version__,
              description="Predictions with conformal intervals, exact Shapley explanations and integrity checks.")


@app.get("/health")
def health() -> dict:
    b = ready().bundle
    return {"status": "ok", "version": __version__, "fingerprint": b.fingerprint, "checksums_verified": b.verified}


@app.get("/model")
def model_card() -> dict:
    b = ready().bundle
    fin = b.final
    return {"fingerprint": b.fingerprint, "features": b.features,
            "models": {label: {"description": net.describe(), "parameters": net.n_params, "best_epoch": net.best_epoch,
                               "conformal_intervals": net.has_intervals} for label, net in b.models.items()},
            "test_metrics": {"mae_usd": fin["test_mae"] * 100_000, "rmse_usd": fin["test_rmse"] * 100_000, "r2": fin["test_r2"]}}


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    svc = ready()
    try:
        results = [svc.predict_one(bg.model_dump(exclude_none=True), request.model, request.level) for bg in request.block_groups]
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err
    return {"fingerprint": svc.bundle.fingerprint, "predictions": results}


@app.post("/explain")
def explain(block_group: BlockGroup, model: Model = "final") -> dict:
    svc = ready()
    try:
        return svc.explain(block_group.model_dump(exclude_none=True), model)
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err
