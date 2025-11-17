"""Utilities to load and run the sentiment classifier."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

try:  # pragma: no cover - optional dependency
    import joblib
except ImportError:  # pragma: no cover
    joblib = None


class KeywordFallbackModel:
    """Simple keyword-based classifier used until a trained model is available."""

    classes_ = ["negative", "neutral", "positive"]

    positive_keywords = {
        "нравится",
        "спасибо",
        "удобно",
        "люблю",
        "хорошо",
        "стабильно",
        "отремонтирована",
    }
    negative_keywords = {
        "ужас",
        "плохо",
        "вылетает",
        "невозможно",
        "запутался",
        "молчит",
        "устаревшая",
        "ошибка",
        "проблема",
    }

    def predict(self, texts: Iterable[str]) -> List[str]:
        return [self._predict_text(text) for text in texts]

    def predict_proba(self, texts: Iterable[str]):
        return [self._scores(text) for text in texts]

    def _predict_text(self, text: str) -> str:
        probs = self._scores(text)
        max_index = max(range(len(probs)), key=lambda idx: probs[idx])
        return self.classes_[max_index]

    def _scores(self, text: str) -> List[float]:
        text_lower = text.lower()
        pos_hits = sum(word in text_lower for word in self.positive_keywords)
        neg_hits = sum(word in text_lower for word in self.negative_keywords)
        total = pos_hits + neg_hits
        if total == 0:
            return [0.2, 0.6, 0.2]
        neg_score = neg_hits / total
        pos_score = pos_hits / total
        neu_score = max(0.0, 1.0 - (neg_score + pos_score) / 2)
        scores = [neg_score, neu_score, pos_score]
        total_score = sum(scores)
        if total_score == 0:
            return [1 / 3, 1 / 3, 1 / 3]
        return [score / total_score for score in scores]


class SentimentModel:
    """Wrapper around a trained pipeline with a rule-based fallback."""

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.pipeline = self._load_pipeline(model_path)
        self.labels: List[str] = list(self.pipeline.classes_)

    def _load_pipeline(self, model_path: Path):
        if joblib is None or not model_path.exists():
            return KeywordFallbackModel()
        return joblib.load(model_path)

    def predict(self, text: str) -> Dict[str, float]:
        """Return class probabilities for a single text."""
        proba = self.pipeline.predict_proba([text])[0]
        return {label: float(score) for label, score in zip(self.labels, proba)}

    def predict_label(self, text: str) -> str:
        return self.pipeline.predict([text])[0]

    def classify(self, text: str) -> Dict[str, object]:
        probabilities = self.predict(text)
        label = max(probabilities.items(), key=lambda pair: pair[1])[0]
        return {"label": label, "scores": probabilities}

    def classify_batch(self, texts: List[str]) -> List[Dict[str, object]]:
        proba = self.pipeline.predict_proba(texts)
        labels = self.pipeline.predict(texts)
        return [
            {
                "label": label,
                "scores": {cls: float(score) for cls, score in zip(self.labels, row)},
            }
            for label, row in zip(labels, proba)
        ]
