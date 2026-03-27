#!/usr/bin/env python3
"""
Run the benchmark dataset against the current RAG system.

Supports:
- retrieval-only evaluation
- answer generation evaluation
- combined runs

Outputs:
- JSONL file with per-item results
- JSON summary file with aggregate metrics
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.chat.handlers import handle_chat
from app.core.document_registry import list_documents
from app.retrieval.retriever import extract_sources, retrieve


DEFAULT_BENCHMARK = "finetuning/data/benchmark_template.jsonl"
DEFAULT_RESULTS_DIR = "finetuning/results"
SUMMARY_CATEGORIES_FULL_DOC = {"summary", "section_understanding"}


@dataclass
class BenchmarkScope:
    source: str | None
    doc_id: str | None
    source_type: str | None
    resolved: bool
    resolution_note: str


def load_benchmark(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            obj["_line_no"] = line_no
            rows.append(obj)
    return rows


def _pick_best_registry_entry(entries: list[dict[str, Any]]) -> dict[str, Any]:
    def sort_key(item: dict[str, Any]) -> tuple[int, str]:
        source_type_score = 0 if item.get("source_type") == "library" else 1
        updated = str(item.get("updated_at") or item.get("created_at") or "")
        return (source_type_score, updated)

    return sorted(entries, key=sort_key)[0]


def resolve_scope(document_source: str) -> BenchmarkScope:
    entries = [doc for doc in list_documents() if doc.get("source") == document_source]
    if not entries:
        return BenchmarkScope(
            source=document_source,
            doc_id=None,
            source_type=None,
            resolved=False,
            resolution_note="document source not found in registry",
        )

    best = _pick_best_registry_entry(entries)
    note = "matched exact source"
    if len(entries) > 1:
        note = f"matched {len(entries)} documents; selected preferred scope"

    return BenchmarkScope(
        source=best.get("source"),
        doc_id=best.get("doc_id"),
        source_type=best.get("source_type"),
        resolved=True,
        resolution_note=note,
    )


def should_use_full_document(item: dict[str, Any]) -> bool:
    if item.get("category") in SUMMARY_CATEGORIES_FULL_DOC:
        return True
    return bool(item.get("full_document", False))


def normalize_text(text: str) -> str:
    return " ".join((text or "").lower().split())


def score_answer_points(answer: str, expected_points: list[str]) -> dict[str, Any]:
    normalized_answer = normalize_text(answer)
    hits = []
    misses = []
    for point in expected_points:
        normalized_point = normalize_text(point)
        if normalized_point and normalized_point in normalized_answer:
            hits.append(point)
        else:
            misses.append(point)

    total = len(expected_points)
    recall = len(hits) / total if total else 1.0
    return {
        "expected_points_total": total,
        "expected_points_hit": len(hits),
        "expected_points_recall": round(recall, 3),
        "point_hits": hits,
        "point_misses": misses,
    }


def score_sources(actual_sources: list[dict[str, Any]], expected_sources: list[dict[str, Any]]) -> dict[str, Any]:
    if not expected_sources:
        return {
            "expected_source_total": 0,
            "expected_source_hit": 0,
            "expected_source_recall": 1.0,
            "source_hits": [],
            "source_misses": [],
        }

    hits = []
    misses = []
    for expected in expected_sources:
        source = expected.get("source")
        page = expected.get("page")
        matched = any(
            actual.get("source") == source and (page is None or actual.get("page") == page)
            for actual in actual_sources
        )
        if matched:
            hits.append(expected)
        else:
            misses.append(expected)

    total = len(expected_sources)
    recall = len(hits) / total if total else 1.0
    return {
        "expected_source_total": total,
        "expected_source_hit": len(hits),
        "expected_source_recall": round(recall, 3),
        "source_hits": hits,
        "source_misses": misses,
    }


def detect_refusal(answer: str) -> bool:
    lower = (answer or "").lower()
    indicators = [
        "couldn't find relevant information",
        "not found in the document",
        "제공된 문서에서 확인되지 않습니다",
        "질문과 관련된 정보를 찾지 못했습니다",
    ]
    return any(token in lower for token in indicators)


def run_retrieval(item: dict[str, Any], scope: BenchmarkScope) -> dict[str, Any]:
    if not scope.resolved:
        return {
            "resolved": False,
            "resolution_note": scope.resolution_note,
            "retrieved_sources": [],
            "confidence": 0.0,
            "strong_keyword_hit": False,
            "used_full_document": False,
        }

    result = retrieve(
        query=item["question"],
        source_filter=scope.source,
        doc_id_filter=scope.doc_id,
        full_document=should_use_full_document(item),
    )
    actual_sources = extract_sources(result.docs)
    return {
        "resolved": True,
        "resolution_note": scope.resolution_note,
        "retrieved_sources": actual_sources,
        "confidence": result.confidence,
        "strong_keyword_hit": result.strong_keyword_hit,
        "used_full_document": result.used_full_document,
    }


def run_answer(item: dict[str, Any], scope: BenchmarkScope, model: str | None) -> dict[str, Any]:
    if not scope.resolved:
        return {
            "resolved": False,
            "resolution_note": scope.resolution_note,
            "answer": "",
            "mode": "unresolved",
            "sources": [],
        }

    result = handle_chat(
        user_message=item["question"],
        model=model,
        active_source=scope.source,
        active_doc_id=scope.doc_id,
    )
    return {
        "resolved": True,
        "resolution_note": scope.resolution_note,
        "answer": result.get("answer", ""),
        "mode": result.get("mode", ""),
        "sources": result.get("sources", []),
    }


def summarize_results(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(results)
    category_counts = Counter(row["category"] for row in rows)
    unresolved = sum(1 for row in rows if not row["scope"]["resolved"])
    answer_recalls = [row["answer_scoring"]["expected_points_recall"] for row in rows if "answer_scoring" in row]
    source_recalls = [row["source_scoring"]["expected_source_recall"] for row in rows if "source_scoring" in row]
    refusal_cases = [row for row in rows if not row.get("should_answer_from_docs", True)]
    refusal_pass = sum(1 for row in refusal_cases if row.get("answer_refusal_detected"))

    return {
        "total_rows": len(rows),
        "unresolved_document_scope_rows": unresolved,
        "categories": dict(category_counts),
        "avg_answer_point_recall": round(sum(answer_recalls) / len(answer_recalls), 3) if answer_recalls else None,
        "avg_source_recall": round(sum(source_recalls) / len(source_recalls), 3) if source_recalls else None,
        "negative_case_count": len(refusal_cases),
        "negative_case_refusal_pass": refusal_pass,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run benchmark JSONL against the current RAG system.")
    parser.add_argument("--path", default=DEFAULT_BENCHMARK, help="Benchmark JSONL path")
    parser.add_argument(
        "--mode",
        choices=["retrieval", "answer", "both"],
        default="both",
        help="What to evaluate",
    )
    parser.add_argument("--model", default=None, help="Override model for answer generation")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of benchmark rows")
    parser.add_argument("--id", dest="ids", action="append", default=[], help="Run only these benchmark IDs")
    parser.add_argument("--output-dir", default=DEFAULT_RESULTS_DIR, help="Where to store benchmark results")
    args = parser.parse_args()

    benchmark_path = Path(args.path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_benchmark(benchmark_path)
    if args.ids:
        wanted = set(args.ids)
        rows = [row for row in rows if row.get("id") in wanted]
    if args.limit:
        rows = rows[: args.limit]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = output_dir / f"benchmark_results_{timestamp}.jsonl"
    summary_path = output_dir / f"benchmark_summary_{timestamp}.json"

    results: list[dict[str, Any]] = []
    with result_path.open("w", encoding="utf-8") as handle:
        for item in rows:
            scope = resolve_scope(item["document_source"])
            row_result: dict[str, Any] = {
                "id": item["id"],
                "category": item["category"],
                "language": item["language"],
                "document_source": item["document_source"],
                "question": item["question"],
                "should_answer_from_docs": item["should_answer_from_docs"],
                "scope": asdict(scope),
            }

            retrieval_payload = None
            answer_payload = None

            if args.mode in {"retrieval", "both"}:
                retrieval_payload = run_retrieval(item, scope)
                row_result["retrieval"] = retrieval_payload
                row_result["source_scoring"] = score_sources(
                    retrieval_payload.get("retrieved_sources", []),
                    item.get("expected_sources", []),
                )

            if args.mode in {"answer", "both"}:
                answer_payload = run_answer(item, scope, args.model)
                row_result["answer_result"] = answer_payload
                row_result["answer_scoring"] = score_answer_points(
                    answer_payload.get("answer", ""),
                    item.get("expected_answer_points", []),
                )
                row_result["answer_refusal_detected"] = detect_refusal(answer_payload.get("answer", ""))

            handle.write(json.dumps(row_result, ensure_ascii=False) + "\n")
            results.append(row_result)

    summary = summarize_results(results)
    summary["mode"] = args.mode
    summary["benchmark_path"] = str(benchmark_path)
    summary["result_path"] = str(result_path)

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("BENCHMARK COMPLETE")
    print(f"rows: {summary['total_rows']}")
    print(f"results: {result_path}")
    print(f"summary: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
