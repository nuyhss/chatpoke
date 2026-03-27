#!/usr/bin/env python3
"""
Validate benchmark JSONL structure and print a quick category summary.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "id",
    "category",
    "language",
    "document_source",
    "question",
    "should_answer_from_docs",
    "expected_answer_points",
}


def validate_line(obj: dict[str, Any], line_no: int) -> list[str]:
    errors: list[str] = []

    missing = sorted(REQUIRED_FIELDS - set(obj))
    if missing:
        errors.append(f"line {line_no}: missing required fields: {', '.join(missing)}")

    if "expected_answer_points" in obj and not isinstance(obj["expected_answer_points"], list):
        errors.append(f"line {line_no}: expected_answer_points must be a list")

    if "expected_sources" in obj and not isinstance(obj["expected_sources"], list):
        errors.append(f"line {line_no}: expected_sources must be a list")

    if "should_answer_from_docs" in obj and not isinstance(obj["should_answer_from_docs"], bool):
        errors.append(f"line {line_no}: should_answer_from_docs must be true/false")

    if "id" in obj and not str(obj["id"]).strip():
        errors.append(f"line {line_no}: id must be non-empty")

    if "question" in obj and not str(obj["question"]).strip():
        errors.append(f"line {line_no}: question must be non-empty")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate benchmark JSONL file.")
    parser.add_argument(
        "--path",
        default="finetuning/data/benchmark_template.jsonl",
        help="Path to benchmark JSONL",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"ERROR: file not found: {path}")
        return 1

    category_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    errors: list[str] = []
    seen_ids: set[str] = set()
    total = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            total += 1
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                errors.append(f"line {line_no}: invalid JSON ({e})")
                continue

            errors.extend(validate_line(obj, line_no))

            item_id = str(obj.get("id", "")).strip()
            if item_id:
                if item_id in seen_ids:
                    errors.append(f"line {line_no}: duplicate id '{item_id}'")
                seen_ids.add(item_id)

            category = str(obj.get("category", "unknown")).strip() or "unknown"
            language = str(obj.get("language", "unknown")).strip() or "unknown"
            category_counts[category] += 1
            language_counts[language] += 1

    if errors:
        print("VALIDATION FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    print("VALIDATION OK")
    print(f"rows: {total}")
    print("categories:")
    for key, value in sorted(category_counts.items()):
        print(f"  - {key}: {value}")
    print("languages:")
    for key, value in sorted(language_counts.items()):
        print(f"  - {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
