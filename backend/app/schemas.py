"""Pydantic schemas for the API."""
from __future__ import annotations

from typing import Dict, List

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
