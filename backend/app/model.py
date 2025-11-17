"""Utilities to load and run the sentiment classifier."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

try:  # pragma: no cover - optional dependency
    import joblib
except ImportError:  # pragma: no cover
    joblib = None

try:  # pragma: no cover - optional dependency
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers.pipelines import TextClassificationPipeline
except ImportError:  # pragma: no cover
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    TextClassificationPipeline = None


class BaseAdapter:
    classes_: List[str]

    def predict(self, texts: Iterable[str]) -> List[str]:  # pragma: no cover - interface
        raise NotImplementedError

    def predict_proba(self, texts: Iterable[str]):  # pragma: no cover - interface
        raise NotImplementedError


class JoblibAdapter(BaseAdapter):
    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline
        self.classes_ = list(getattr(pipeline, "classes_", ["negative", "neutral", "positive"]))

    def predict(self, texts: Iterable[str]) -> List[str]:
        return list(self.pipeline.predict(list(texts)))

    def predict_proba(self, texts: Iterable[str]):
        return self.pipeline.predict_proba(list(texts))


class KeywordFallbackAdapter(BaseAdapter):
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


class TransformerAdapter(BaseAdapter):
    def __init__(self, model_dir: Path) -> None:
        if AutoTokenizer is None or AutoModelForSequenceClassification is None:
            raise ImportError("transformers is required to load transformer models")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        if TextClassificationPipeline is None:  # pragma: no cover
            raise ImportError("transformers pipelines are required")
        self.pipeline = TextClassificationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            function_to_apply="softmax",
            top_k=None,
        )
        if hasattr(self.model.config, "id2label"):
            labels_map = self.model.config.id2label
            try:
                ordered_keys = sorted(labels_map.keys(), key=lambda key: int(key))
            except (TypeError, ValueError):
                ordered_keys = sorted(labels_map.keys())
            self.classes_ = [labels_map[key] for key in ordered_keys]
        else:
            self.classes_ = [f"LABEL_{idx}" for idx in range(self.model.config.num_labels)]

    def predict(self, texts: Iterable[str]) -> List[str]:
        proba = self.predict_proba(texts)
        predictions = []
        for row in proba:
            max_index = max(range(len(row)), key=lambda idx: row[idx])
            predictions.append(self.classes_[max_index])
        return predictions

    def predict_proba(self, texts: Iterable[str]):
        outputs = self.pipeline(
            list(texts),
            truncation=True,
            padding=True,
            return_all_scores=True,
        )
        proba_rows = []
        for sample_scores in outputs:
            mapping = {item["label"]: float(item["score"]) for item in sample_scores}
            row = [mapping.get(label, 0.0) for label in self.classes_]
            total = sum(row)
            if total:
                row = [val / total for val in row]
            proba_rows.append(row)
        return proba_rows


class SentimentModel:
    """Wrapper around a trained pipeline with a rule-based fallback."""

    def __init__(self, model_path: Path, metadata_path: Path | None = None):
        self.model_path = model_path
        if metadata_path is not None:
            self.metadata_path = metadata_path
        elif model_path.is_dir():
            self.metadata_path = model_path / "metadata.json"
        else:
            self.metadata_path = model_path.with_name("metadata.json")
        self.adapter = self._build_adapter(model_path)
        self.labels: List[str] = list(
            getattr(self.adapter, "classes_", ["negative", "neutral", "positive"])
        )
        self.metadata = self._load_metadata()

    def _build_adapter(self, model_path: Path) -> BaseAdapter:
        if model_path.is_dir() and (model_path / "config.json").exists():
            try:
                return TransformerAdapter(model_path)
            except Exception:  # pragma: no cover - fallback handled gracefully
                pass
        if joblib is not None and model_path.exists():
            try:
                pipeline = joblib.load(model_path)
                return JoblibAdapter(pipeline)
            except Exception:  # pragma: no cover - fallback handled below
                pass
        return KeywordFallbackAdapter()

    def _load_metadata(self) -> Dict[str, object]:
        if not self.metadata_path.exists():
            return {
                "model_path": str(self.model_path),
                "algorithm": getattr(self.adapter, "__class__", type(self.adapter)).__name__,
                "classes": self.labels,
            }
        try:
            return json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"model_path": str(self.model_path), "classes": self.labels}

    def predict(self, text: str) -> Dict[str, float]:
        """Return class probabilities for a single text."""
        proba = self.adapter.predict_proba([text])[0]
        return {label: float(score) for label, score in zip(self.labels, proba)}

    def predict_label(self, text: str) -> str:
        return self.adapter.predict([text])[0]

    def classify(self, text: str) -> Dict[str, object]:
        probabilities = self.predict(text)
        label = max(probabilities.items(), key=lambda pair: pair[1])[0]
        return {"label": label, "scores": probabilities}

    def classify_batch(self, texts: List[str]) -> List[Dict[str, object]]:
        proba = self.adapter.predict_proba(texts)
        labels = [
            self.labels[max(range(len(self.labels)), key=lambda idx: row[idx])]
            for row in proba
        ]
        return [
            {
                "label": label,
                "scores": {cls: float(score) for cls, score in zip(self.labels, row)},
            }
            for label, row in zip(labels, proba)
        ]
