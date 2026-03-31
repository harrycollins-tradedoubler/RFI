#!/usr/bin/env python3
"""
Merge multiple Ready-style CSV datasets into one combined dataset.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REQUIRED_COLUMNS = [
    "question_id",
    "content",
    "canonical_question",
    "starter_answer",
    "category",
    "subcategory",
    "response_type",
    "item_type",
    "language",
    "evidence_reference",
    "source_files",
    "source_sections",
    "readiness_status",
    "metadata_json",
]


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    missing = [c for c in REQUIRED_COLUMNS if c not in rows[0]]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge Ready dataset CSV files.")
    parser.add_argument("--input", action="append", required=True, help="Input CSV path (repeatable)")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument(
        "--dedupe-key",
        default="question_id",
        help="Deduplicate on this key (last one wins). Default: question_id",
    )
    args = parser.parse_args()

    merged: dict[str, dict] = {}
    total_in = 0
    for src in args.input:
        p = Path(src).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Missing input: {p}")
        rows = read_rows(p)
        total_in += len(rows)
        for row in rows:
            key = (row.get(args.dedupe_key) or "").strip()
            if not key:
                continue
            merged[key] = {c: row.get(c, "") for c in REQUIRED_COLUMNS}

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(merged.values())

    print(f"Input rows: {total_in}")
    print(f"Output rows: {len(merged)}")
    print(f"Output file: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
