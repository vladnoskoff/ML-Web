from __future__ import annotations

"""Convert a JSONL sentiment dataset with `text` and `label` fields into CSV.

Usage:
    python ml/jsonl_to_csv.py --input data/training_sentiment_ru.jsonl --output data/training_sentiment_ru.csv
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, List, Tuple


def load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def extract_examples(entries: Iterable[dict]) -> List[Tuple[str, str]]:
    examples: List[Tuple[str, str]] = []
    for entry in entries:
        text = (entry.get("text") or "").strip()
        label = (entry.get("label") or "").strip()
        if text and label:
            examples.append((text, label))
    return examples


def save_csv(examples: List[Tuple[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows({"text": text, "label": label} for text, label in examples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/training_sentiment_ru.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/training_sentiment_ru.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = load_jsonl(args.input)
    examples = extract_examples(entries)
    if not examples:
        raise SystemExit("No valid records with text/label found in the JSONL file.")

    save_csv(examples, args.output)
    print(f"Saved {len(examples)} rows to {args.output}")


if __name__ == "__main__":
    main()
