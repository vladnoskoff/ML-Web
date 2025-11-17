"""Feedback storage utilities for active learning."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Deque, Dict, List, Optional
from collections import deque


@dataclass
class FeedbackEntry:
    text: str
    predicted_label: str
    user_label: Optional[str]
    scores: Optional[Dict[str, float]]
    notes: Optional[str]
    timestamp: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class FeedbackStore:
    """Thread-safe append-only store backed by a JSONL file."""

    def __init__(self, path: Path, cache_size: int = 200) -> None:
        self.path = path
        self.cache_size = cache_size
        self._lock = Lock()
        self._recent: Deque[FeedbackEntry] = deque(maxlen=cache_size)
        self._total = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._bootstrap_cache()

    def _bootstrap_cache(self) -> None:
        if not self.path.exists():
            return
        lines = self.path.read_text(encoding="utf-8").splitlines()
        self._total = len(lines)
        for line in reversed(lines[-self.cache_size :]):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            entry = FeedbackEntry(
                text=payload.get("text", ""),
                predicted_label=payload.get("predicted_label", ""),
                user_label=payload.get("user_label"),
                scores=payload.get("scores"),
                notes=payload.get("notes"),
                timestamp=payload.get("timestamp", ""),
            )
            self._recent.appendleft(entry)

    def append(
        self,
        *,
        text: str,
        predicted_label: str,
        user_label: Optional[str] = None,
        scores: Optional[Dict[str, float]] = None,
        notes: Optional[str] = None,
    ) -> FeedbackEntry:
        entry = FeedbackEntry(
            text=text.strip(),
            predicted_label=predicted_label,
            user_label=user_label,
            scores=scores,
            notes=notes.strip() if notes else None,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        serialized = json.dumps(entry.to_dict(), ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(serialized + "\n")
            self._recent.appendleft(entry)
            self._total += 1
        return entry

    def recent(self, limit: Optional[int] = None) -> List[Dict[str, object]]:
        with self._lock:
            items = list(self._recent)
        if limit is not None:
            items = items[:limit]
        return [item.to_dict() for item in items]

    def count(self) -> int:
        return self._total
