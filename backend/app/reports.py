"""Helpers for loading evaluation and history reports for the dashboard."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


class ReportLoader:
    """Load JSON reports from disk with graceful fallbacks."""

    def __init__(self, eval_metrics_path: Path, history_summary_path: Path) -> None:
        self.eval_metrics_path = eval_metrics_path
        self.history_summary_path = history_summary_path
        self._eval_fallback = _fallback_path(eval_metrics_path)
        self._history_fallback = _fallback_path(history_summary_path)

    def load_eval_metrics(self) -> Dict[str, Any]:
        return self._load(self.eval_metrics_path) or self._load(self._eval_fallback) or {}

    def load_history_summary(self) -> Dict[str, Any]:
        return self._load(self.history_summary_path) or self._load(self._history_fallback) or {}

    @staticmethod
    def _load(path: Optional[Path]) -> Optional[Dict[str, Any]]:
        if not path or not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None


def _fallback_path(path: Path) -> Path:
    suffix = ".sample.json"
    if path.suffix:
        return path.with_suffix(f"{path.suffix}.sample.json" if not path.suffix.endswith(".json") else suffix)
    return Path(f"{path}.sample.json")
