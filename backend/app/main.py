"""FastAPI service for the sentiment classifier."""
from __future__ import annotations

import io
import logging
from collections import Counter

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import settings
from .feedback import FeedbackStore
from .model import SentimentModel
from .schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    FilePredictResponse,
    FeedbackListResponse,
    FeedbackRequest,
    FeedbackResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    StatsResponse,
)
from .stats import StatsTracker

MAX_FILE_RECORDS = settings.max_file_records

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="ML-Web Sentiment API", version="1.0.0")
logger = logging.getLogger(__name__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if settings.frontend_dir.exists():
    app.mount("/ui", StaticFiles(directory=settings.frontend_dir, html=True), name="ui")

sentiment_model: SentimentModel | None = None
stats_tracker = StatsTracker(
    max_history=settings.stats_max_history, history_path=settings.history_path
)
feedback_store = FeedbackStore(settings.feedback_path, cache_size=200)


@app.on_event("startup")
def load_model() -> None:
    global sentiment_model
    target_path = settings.model_path
    if not target_path.exists() and settings.transformer_dir.exists():
        target_path = settings.transformer_dir
        logger.info(
            "Primary model missing, falling back to transformer dir %s",
            settings.transformer_dir,
        )
    sentiment_model = SentimentModel(target_path)
    if not target_path.exists():
        logger.warning(
            "Model artifact %s is missing, using KeywordFallbackModel until training runs.",
            target_path,
        )


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


@app.post("/predict_file", response_model=FilePredictResponse)
async def predict_file(file: UploadFile = File(...)) -> FilePredictResponse:
    """Обработка CSV с колонкой text."""

    try:
        content = await file.read()
    except Exception as exc:  # pragma: no cover - FastAPI handles IO
        raise HTTPException(status_code=400, detail="Не удалось прочитать файл") from exc

    if not content:
        raise HTTPException(status_code=400, detail="Файл пустой")

    try:
        decoded = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="CSV должен быть в кодировке UTF-8") from exc

    try:
        dataframe = pd.read_csv(io.StringIO(decoded))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Не удалось распарсить CSV: {exc}",
        ) from exc

    if "text" not in dataframe.columns:
        raise HTTPException(status_code=400, detail="CSV должен содержать колонку 'text'")

    input_rows = len(dataframe)
    dataframe = dataframe.dropna(subset=["text"])
    if dataframe.empty:
        raise HTTPException(status_code=400, detail="В колонке 'text' нет строк для обработки")

    if len(dataframe) > MAX_FILE_RECORDS:
        dataframe = dataframe.head(MAX_FILE_RECORDS)

    model = _require_model()
    texts = dataframe["text"].astype(str).tolist()
    raw_predictions = model.classify_batch(texts)

    items = []
    for idx, text, pred in zip(dataframe.index.tolist(), texts, raw_predictions):
        stats_tracker.record(text, pred["label"], pred["scores"])
        items.append({
            "row": int(idx),
            "text": text,
            "label": pred["label"],
            "scores": pred["scores"],
        })

    class_counts = Counter(item["label"] for item in items)
    summary = {
        "input_rows": int(input_rows),
        "processed_rows": len(items),
        "skipped_rows": max(0, int(input_rows) - len(items)),
        "class_counts": dict(class_counts),
    }

    return FilePredictResponse(summary=summary, predictions=items)


@app.get("/")
def root() -> dict:
    return {
        "message": "Добро пожаловать в сервис анализа тональности",
        "endpoints": [
            "/predict",
            "/predict_batch",
            "/predict_file",
            "/stats",
            "/model",
            "/health",
        ],
    }


@app.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    summary = stats_tracker.snapshot()
    return StatsResponse(**summary)


@app.get("/model", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    model = _require_model()
    return ModelInfoResponse(**model.metadata)


@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    entry = feedback_store.append(
        text=request.text,
        predicted_label=request.predicted_label,
        user_label=request.user_label,
        scores=request.scores,
        notes=request.notes,
    )
    return FeedbackResponse(status="stored", entry=entry.to_dict())


@app.get("/feedback", response_model=FeedbackListResponse)
def list_feedback(limit: int = 50) -> FeedbackListResponse:
    limit = max(1, min(limit, 200))
    return FeedbackListResponse(
        total_items=feedback_store.count(), items=feedback_store.recent(limit)
    )
