"""Utility to fix label mapping in the Hack&Change sentiment dataset.

The supplied CSV files in `hahaton/ТОНАЛЬНОСТЬ` encode sentiment as
numeric classes (0=neutral, 1=positive, 2=negative). Earlier case
materials described a different order, so this script makes the
mapping explicit and converts labels to the string format expected by
the training pipelines in this repository.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

import pandas as pd

LabelValue = Literal[0, 1, 2]

MAPPING: dict[LabelValue, str] = {
    0: "neutral",
    1: "positive",
    2: "negative",
}
def convert_dataset(input_path: Path, output_path: Path, keep_meta: bool = False) -> None:
    """Convert numeric labels to strings and save a clean CSV.

    Parameters
    ----------
    input_path:
        Path to the original CSV with columns: id, text, source, label.
    output_path:
        Destination CSV path.
    keep_meta:
        Whether to retain the `id` and `source` columns in the output.
    """

    bad_lines: list[str] = []

    def _collect_bad_lines(line: list[str]) -> None:
        bad_lines.append(",".join(line) if isinstance(line, list) else str(line))

    df = pd.read_csv(
        input_path,
        header=None,
        names=["id", "text", "source", "label"],
        engine="python",
        on_bad_lines=_collect_bad_lines,
    )

    if bad_lines:
        print(
            f"Warning: skipped {len(bad_lines)} malformed line(s) while reading {input_path}",
            file=sys.stderr,
        )

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    if df["label"].isna().any():
        dropped = int(df["label"].isna().sum())
        print(
            f"Warning: dropped {dropped} row(s) with non-numeric labels during parsing",
            file=sys.stderr,
        )
        df = df.dropna(subset=["label"])

    unknown = set(df["label"].unique()) - set(MAPPING.keys())
    if unknown:
        raise ValueError(
            f"Unexpected label values {unknown}; expected only {tuple(MAPPING.keys())}"
        )

    df["label"] = df["label"].astype(int).map(MAPPING)

    if not keep_meta:
        df = df[["text", "label"]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Hack&Change CSV labels from numeric (0/1/2) to "
            "text labels (neutral/positive/negative)."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Path to source CSV file")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Where to save the converted CSV",
    )
    parser.add_argument(
        "--keep-meta",
        action="store_true",
        help="Keep id/source columns in the output (default: drop them)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_dataset(args.input, args.output, keep_meta=args.keep_meta)


if __name__ == "__main__":
    main()
