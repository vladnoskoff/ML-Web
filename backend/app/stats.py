"""Utilities for tracking prediction statistics with optional persistence."""
from __future__ import annotations

import json
from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Deque, Dict, Optional


@dataclass
class PredictionRecord:
    text: str
    label: str
    scores: Dict[str, float]
    timestamp: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class StatsTracker:
    """Thread-safe tracker that stores aggregate stats and optional JSONL history."""

    def __init__(self, max_history: int = 50, history_path: Optional[Path] = None) -> None:
        self.max_history = max_history
        self._history: Deque[PredictionRecord] = deque(maxlen=max_history)
        self._counts: Counter[str] = Counter()
        self._total: int = 0
        self._lock = Lock()
        self._file_lock = Lock()
        self._default_labels = ["negative", "neutral", "positive"]
        self.history_path = Path(history_path) if history_path else None
        if self.history_path:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            self._load_history_from_disk()

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
        self._append_record(record)
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

    def _append_record(self, record: PredictionRecord) -> None:
        if not self.history_path:
            return
        payload = json.dumps(record.to_dict(), ensure_ascii=False)
        with self._file_lock:
            try:
                with self.history_path.open("a", encoding="utf-8") as fh:
                    fh.write(payload + "\n")
            except OSError:
                pass

    def _load_history_from_disk(self) -> None:
        if not self.history_path or not self.history_path.exists():
            return
        records: list[PredictionRecord] = []
        counts: Counter[str] = Counter()
        total = 0
        try:
            with self.history_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    record = PredictionRecord(
                        text=str(payload.get("text", "")),
                        label=str(payload.get("label", "")),
                        scores=payload.get("scores") or {},
                        timestamp=str(payload.get("timestamp", "")),
                    )
                    records.append(record)
                    counts[record.label] += 1
                    total += 1
        except OSError:
            return
        with self._lock:
            self._counts = counts
            self._total = total
            self._history.clear()
            for record in reversed(records[-self.max_history :]):
                self._history.append(record)


def _truncate_text(text: str, max_length: int = 240) -> str:
    text = text.strip()
    if len(text) <= max_length:
        return text
    return text[: max_length - 1].rstrip() + "…"
