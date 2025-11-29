"""Quick CLI to classify comments as negative/neutral/positive."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, List

from backend.app.model import SentimentModel


def _load_texts(args: argparse.Namespace) -> List[str]:
    texts: List[str] = []
    if args.text:
        texts.extend(args.text)
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        if file_path.suffix.lower() == ".csv":
            with file_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if "text" not in (reader.fieldnames or []):
                    raise ValueError("CSV input must contain a 'text' column")
                texts.extend([row["text"].strip() for row in reader if row.get("text", "").strip()])
        else:
            file_text = file_path.read_text(encoding="utf-8").splitlines()
            texts.extend([line.strip() for line in file_text if line.strip()])
    if not texts:
        raise ValueError("Provide at least one comment via --text or --file")
    return texts


def classify_comments(texts: Iterable[str], model_path: Path | None = None) -> List[dict]:
    model_path = model_path or Path("models/baseline.joblib")
    model = SentimentModel(model_path=model_path)
    results = model.classify_batch(list(texts))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify comments into sentiment buckets")
    parser.add_argument(
        "--text",
        nargs="*",
        help="One or more comments to classify (can be combined with --file)",
    )
    parser.add_argument(
        "--file",
        help="UTF-8 text file with one comment per line",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/baseline.joblib"),
        help="Path to a trained joblib pipeline or transformer directory",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Optional path to metadata.json if it is not next to the model",
    )

    args = parser.parse_args()
    texts = _load_texts(args)
    model = SentimentModel(model_path=args.model, metadata_path=args.metadata)
    results = model.classify_batch(texts)

    annotated = []
    for text, result in zip(texts, results):
        annotated.append({"text": text, **result})

    print(json.dumps(annotated, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
