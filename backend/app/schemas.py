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


class PredictionHistoryItem(BaseModel):
    text: str
    label: str
    scores: Dict[str, float]
    timestamp: str


class StatsResponse(BaseModel):
    total_predictions: int
    label_distribution: Dict[str, float]
    recent_predictions: List[PredictionHistoryItem]


class ModelInfoResponse(BaseModel):
    model_path: str
    algorithm: Optional[str] = None
    vectorizer: Optional[str] = None
    classes: List[str]
    metrics: Optional[Dict[str, Any]] = None
    test_size: Optional[float] = None
    random_state: Optional[int] = None
