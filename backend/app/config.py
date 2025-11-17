"""Application configuration powered by environment variables."""
from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic import BaseSettings, Field


class AppSettings(BaseSettings):
    """Centralized settings for the FastAPI service."""

    model_path: Path = Path("models/baseline.joblib")
    transformer_dir: Path = Path("models/transformer")
    frontend_dir: Path = Path("frontend")
    feedback_path: Path = Path("data/feedback.jsonl")
    history_path: Path = Path("data/prediction_history.jsonl")
    max_file_records: int = 1000
    stats_max_history: int = 100
    allow_origins: List[str] = Field(default_factory=lambda: ["*"])

    class Config:
        env_prefix = "APP_"
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = AppSettings()
