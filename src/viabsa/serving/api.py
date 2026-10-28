"""FastAPI service: ``uvicorn viabsa.serving.api:app``.

Environment variables:
    VIABSA_MODEL_PATH   model directory or Hugging Face repo id (default models/baseline)
    VIABSA_DEVICE       torch device for transformer models, default cpu
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from viabsa import __version__
from viabsa.constants import ASPECTS, LABELS
from viabsa.inference import Predictor, load_predictor

logger = logging.getLogger("viabsa.api")

Review = Annotated[str, Field(min_length=1, max_length=2000)]


class PredictRequest(BaseModel):
    texts: list[Review] = Field(min_length=1, max_length=64, description="Raw review texts")


class AspectResult(BaseModel):
    label: str = Field(description=f"One of {LABELS}")
    confidence: float
    probs: dict[str, float]


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_type: str
    results: list[dict[str, AspectResult]]
    latency_ms: float


class ExplainRequest(BaseModel):
    text: Review
    aspect: str = Field(description=f"One of {ASPECTS}")


class WordScore(BaseModel):
    word: str
    score: float


class ExplainResponse(BaseModel):
    aspect: str
    target_label: str
    target_prob: float
    words: list[WordScore]


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    version: str
    model_type: str
    supports_explain: bool
    aspects: list[str]
    labels: list[str]
    meta: dict[str, Any]


def create_app(predictor: Predictor | None = None) -> FastAPI:
    """App factory; pass a ``predictor`` to skip loading from disk (used by the tests)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if predictor is not None:
            app.state.predictor = predictor
        else:
            path = os.environ.get("VIABSA_MODEL_PATH", "models/baseline")
            device = os.environ.get("VIABSA_DEVICE", "cpu")
            logger.info("Loading model from %s", path)
            app.state.predictor = load_predictor(path, device)
        yield

    app = FastAPI(
        title="ViABSA API",
        version=__version__,
        description="Aspect-based sentiment for Vietnamese cosmetics reviews.",
        lifespan=lifespan,
    )

    def get_predictor(request: Request) -> Predictor:
        pred: Predictor | None = getattr(request.app.state, "predictor", None)
        if pred is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        return pred

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        return {
            "status": "ok",
            "model_loaded": getattr(request.app.state, "predictor", None) is not None,
        }

    @app.get("/model-info", response_model=ModelInfo)
    def model_info(request: Request) -> ModelInfo:
        pred = get_predictor(request)
        return ModelInfo(
            version=__version__,
            model_type=pred.kind,
            supports_explain=pred.supports_explain,
            aspects=ASPECTS,
            labels=LABELS,
            meta=pred.meta,
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(body: PredictRequest, request: Request) -> PredictResponse:
        pred = get_predictor(request)
        start = time.perf_counter()
        results = pred.predict(body.texts)
        latency = (time.perf_counter() - start) * 1000
        logger.info("predict n=%d latency=%.1fms", len(body.texts), latency)
        return PredictResponse(
            model_type=pred.kind,
            results=[{a: AspectResult(**v) for a, v in r.items()} for r in results],
            latency_ms=round(latency, 2),
        )

    @app.post("/explain", response_model=ExplainResponse)
    def explain(body: ExplainRequest, request: Request) -> ExplainResponse:
        pred = get_predictor(request)
        if body.aspect not in ASPECTS:
            raise HTTPException(status_code=422, detail=f"aspect must be one of {ASPECTS}")
        if not pred.supports_explain:
            raise HTTPException(status_code=501, detail=f"'{pred.kind}' model has no explainer")
        out = pred.explain(body.text, body.aspect)
        return ExplainResponse(
            aspect=body.aspect,
            target_label=out["target_label"],
            target_prob=out["target_prob"],
            words=[WordScore(word=w, score=s) for w, s in out["words"]],
        )

    return app


app = create_app()
