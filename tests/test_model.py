import unittest
from pathlib import Path

from backend.app.model import KeywordFallbackAdapter, SentimentModel


class SentimentModelTests(unittest.TestCase):
    def test_keyword_fallback_used_when_model_missing(self) -> None:
        model = SentimentModel(Path("nonexistent_model.joblib"))
        self.assertIsInstance(model.adapter, KeywordFallbackAdapter)

        result = model.classify("Сервис работает ужасно")
        self.assertIn(result["label"], model.labels)
        self.assertGreater(result["scores"][result["label"]], 0)

    def test_batch_classification_structure(self) -> None:
        model = SentimentModel(Path("still_missing.joblib"))
        outputs = model.classify_batch(["Спасибо", "Проблемы с входом"])
        self.assertEqual(len(outputs), 2)
        for item in outputs:
            self.assertIn("label", item)
            self.assertIn("scores", item)
            self.assertEqual(set(item["scores"].keys()), set(model.labels))

    def test_toxic_guardrail_forces_negative_label(self) -> None:
        model = SentimentModel(Path("another_missing_model.joblib"))
        toxic_text = "ты грязная тварь"
        result = model.classify(toxic_text)

        self.assertEqual(result["label"], "negative")
        self.assertGreaterEqual(result["scores"]["negative"], 0.9)


if __name__ == "__main__":
    unittest.main()
