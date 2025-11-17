"""Utilities for tracking in-memory prediction statistics."""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from threading import Lock
from typing import Deque, Dict


@dataclass
class PredictionRecord:
    text: str
    label: str
    scores: Dict[str, float]
    timestamp: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class StatsTracker:
    """Thread-safe tracker that stores aggregate stats and recent history."""

    def __init__(self, max_history: int = 50) -> None:
        self.max_history = max_history
        self._history: Deque[PredictionRecord] = deque(maxlen=max_history)
        self._counts: Counter[str] = Counter()
        self._total: int = 0
        self._lock = Lock()
        self._default_labels = ["negative", "neutral", "positive"]

    def record(self, text: str, label: str, scores: Dict[str, float]) -> PredictionRecord:
        timestamp = datetime.now(timezone.utc).isoformat()
        record = PredictionRecord(
            text=_truncate_text(text),
            label=label,
            scores=scores,
            timestamp=timestamp,
        )
        with self._lock:
            self._history.appendleft(record)
            self._counts[label] += 1
            self._total += 1
        return record

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            labels = sorted(self._counts) or self._default_labels
            distribution = {
                label: (self._counts[label] / self._total if self._total else 0.0)
                for label in labels
            }
            history = [record.to_dict() for record in list(self._history)]
            total = self._total
        return {
            "total_predictions": total,
            "label_distribution": distribution,
            "recent_predictions": history,
        }

    def reset(self) -> None:
        with self._lock:
            self._history.clear()
            self._counts.clear()
            self._total = 0


def _truncate_text(text: str, max_length: int = 240) -> str:
    text = text.strip()
    if len(text) <= max_length:
        return text
    return text[: max_length - 1].rstrip() + "…"
