"""FastAPI service for the sentiment classifier."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .model import SentimentModel
from .schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    StatsResponse,
)
from .stats import StatsTracker

MODEL_PATH = Path("models/baseline.joblib")
FRONTEND_DIR = Path("frontend")

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="ML-Web Sentiment API", version="1.0.0")
logger = logging.getLogger(__name__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="ui")

sentiment_model: SentimentModel | None = None
stats_tracker = StatsTracker(max_history=100)


@app.on_event("startup")
def load_model() -> None:
    global sentiment_model
    sentiment_model = SentimentModel(MODEL_PATH)
    if not MODEL_PATH.exists():
        logger.warning(
            "Model file %s is missing, using KeywordFallbackModel until training runs.",
            MODEL_PATH,
        )
    stats_tracker.reset()


@app.get("/health")
def healthcheck() -> dict:
    return {"status": "ok"}


def _require_model() -> SentimentModel:
    if sentiment_model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")
    return sentiment_model


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    model = _require_model()
    result = model.classify(request.text)
    stats_tracker.record(request.text, result["label"], result["scores"])
    return PredictResponse(**result)


@app.post("/predict_batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    model = _require_model()
    predictions = model.classify_batch(request.texts)
    for text, pred in zip(request.texts, predictions):
        stats_tracker.record(text, pred["label"], pred["scores"])
    return BatchPredictResponse(predictions=[PredictResponse(**pred) for pred in predictions])


@app.get("/")
def root() -> dict:
    return {
        "message": "Добро пожаловать в сервис анализа тональности",
        "endpoints": ["/predict", "/predict_batch", "/stats", "/model", "/health"],
    }


@app.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    summary = stats_tracker.snapshot()
    return StatsResponse(**summary)


@app.get("/model", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    model = _require_model()
    return ModelInfoResponse(**model.metadata)
