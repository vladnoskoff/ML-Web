"""Pydantic schemas for the API."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, constr


class PredictRequest(BaseModel):
    text: constr(strip_whitespace=True, min_length=1, max_length=2000) = Field(
        ..., description="Русскоязычный текст обращения"
    )


class PredictResponse(BaseModel):
    label: str = Field(..., description="Предсказанный класс тональности")
    scores: Dict[str, float] = Field(..., description="Вероятности по каждому классу")


class BatchPredictRequest(BaseModel):
    texts: List[constr(strip_whitespace=True, min_length=1, max_length=2000)] = Field(
        ..., description="Список отзывов"
    )


class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]


class FeedbackRequest(BaseModel):
    text: constr(strip_whitespace=True, min_length=1, max_length=2000)
    predicted_label: constr(strip_whitespace=True, min_length=1)
    user_label: Optional[constr(strip_whitespace=True, min_length=1)] = None
    scores: Optional[Dict[str, float]] = None
    notes: Optional[constr(strip_whitespace=True, min_length=1, max_length=500)] = None


class PredictionHistoryItem(BaseModel):
    text: str
    label: str
    scores: Dict[str, float]
    timestamp: str


class StatsResponse(BaseModel):
    total_predictions: int
    label_distribution: Dict[str, float]
    recent_predictions: List[PredictionHistoryItem]


class FeedbackItem(BaseModel):
    text: str
    predicted_label: str
    user_label: Optional[str]
    scores: Optional[Dict[str, float]] = None
    notes: Optional[str] = None
    timestamp: str


class FeedbackResponse(BaseModel):
    status: str
    entry: FeedbackItem


class FeedbackListResponse(BaseModel):
    total_items: int
    items: List[FeedbackItem]


class ModelInfoResponse(BaseModel):
    model_path: str
    algorithm: Optional[str] = None
    vectorizer: Optional[str] = None
    classes: List[str]
    metrics: Optional[Dict[str, Any]] = None
    test_size: Optional[float] = None
    random_state: Optional[int] = None


class FilePrediction(BaseModel):
    row: int = Field(..., ge=0, description="Индекс строки в исходном файле")
    text: str = Field(..., description="Текст обращения")
    label: str = Field(..., description="Предсказанная тональность")
    scores: Dict[str, float] = Field(..., description="Вероятности по классам")


class FilePredictionSummary(BaseModel):
    input_rows: int = Field(..., description="Количество строк в CSV")
    processed_rows: int = Field(..., description="Сколько строк обработано")
    skipped_rows: int = Field(..., description="Сколько строк пропущено")
    class_counts: Dict[str, int] = Field(
        ..., description="Количество предсказаний по классам"
    )


class FilePredictResponse(BaseModel):
    summary: FilePredictionSummary
    predictions: List[FilePrediction]
