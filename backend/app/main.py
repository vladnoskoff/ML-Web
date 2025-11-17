"""FastAPI service for the sentiment classifier."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException

from .model import SentimentModel
from .schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
)

MODEL_PATH = Path("models/baseline.joblib")

app = FastAPI(title="ML-Web Sentiment API", version="0.1.0")
sentiment_model: SentimentModel | None = None


@app.on_event("startup")
def load_model() -> None:
    global sentiment_model
    try:
        sentiment_model = SentimentModel(MODEL_PATH)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Model file is missing. Run `python ml/train_baseline.py` first."
        ) from exc


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
    return PredictResponse(**result)


@app.post("/predict_batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    model = _require_model()
    predictions = model.classify_batch(request.texts)
    return BatchPredictResponse(predictions=[PredictResponse(**pred) for pred in predictions])


@app.get("/")
def root() -> dict:
    return {
        "message": "Добро пожаловать в сервис анализа тональности",
        "endpoints": ["/predict", "/predict_batch", "/health"],
    }
