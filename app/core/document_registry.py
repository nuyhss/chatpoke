"""
Persistent document registry for stable document identity and lifecycle metadata.

This keeps the system document-aware instead of relying only on filenames/chunks.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from app.config import DOCUMENT_REGISTRY_PATH, LIBRARY_DIR, UPLOADS_DIR

logger = logging.getLogger("tilon.document_registry")

_registry_lock = Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_parent() -> None:
    DOCUMENT_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)


def _load_registry() -> Dict[str, Any]:
    _ensure_parent()
    if not DOCUMENT_REGISTRY_PATH.exists():
        return {"documents": []}

    try:
        return json.loads(DOCUMENT_REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Document registry was unreadable; resetting it.")
        return {"documents": []}


def _save_registry(data: Dict[str, Any]) -> None:
    _ensure_parent()
    DOCUMENT_REGISTRY_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def infer_source_type(file_path: Path) -> str:
    """Infer whether a document belongs to the persistent library or chat uploads."""
    resolved = file_path.resolve()
    try:
        resolved.relative_to(LIBRARY_DIR.resolve())
        return "library"
    except ValueError:
        pass

    try:
        resolved.relative_to(UPLOADS_DIR.resolve())
        return "upload"
    except ValueError:
        return "external"


def _page_kind_counts(docs: List[Document]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for doc in docs:
        kind = str(doc.metadata.get("page_kind") or "unknown")
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _summarize_extractors(docs: List[Document]) -> List[str]:
    extractors = {
        str(doc.metadata.get("extractors_used") or doc.metadata.get("extraction_method") or "")
        for doc in docs
    }
    return sorted(extractor for extractor in extractors if extractor)


def _summarize_languages(docs: List[Document]) -> List[str]:
    languages = {str(doc.metadata.get("language") or "") for doc in docs}
    return sorted(lang for lang in languages if lang and lang != "unknown")


def upsert_document(
    file_path: Path,
    page_docs: List[Document],
    chunk_count: int,
    owner_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Create or update a document registry entry from parsed page docs."""
    if not page_docs:
        return None

    first_meta = page_docs[0].metadata
    doc_id = first_meta.get("doc_id")
    if not doc_id:
        return None

    with _registry_lock:
        registry = _load_registry()
        documents = registry.setdefault("documents", [])

        existing = next((doc for doc in documents if doc.get("doc_id") == doc_id), None)
        now = _utc_now()
        source_type = first_meta.get("source_type") or infer_source_type(file_path)

        entry = {
            "doc_id": doc_id,
            "source": first_meta.get("source", file_path.name),
            "source_path": str(file_path),
            "source_type": source_type,
            "doc_scope": "persistent" if source_type == "library" else "chat_upload",
            "doc_checksum": first_meta.get("doc_checksum"),
            "input_type": first_meta.get("input_type"),
            "page_total": first_meta.get("page_total", len(page_docs)),
            "chunk_count": chunk_count,
            "extractors_used": _summarize_extractors(page_docs),
            "languages": _summarize_languages(page_docs),
            "page_kind_counts": _page_kind_counts(page_docs),
            "status": "ingested",
            "owner_id": owner_id if source_type == "upload" else None,
            "updated_at": now,
        }

        if existing:
            existing.update(entry)
            existing.setdefault("created_at", now)
            existing.setdefault("uploaded_at", now)
            saved = existing
        else:
            saved = {
                **entry,
                "created_at": now,
                "uploaded_at": now,
            }
            documents.append(saved)

        _save_registry(registry)
        logger.info(
            "Document registry upserted: %s (%s, %d chunks)",
            saved["doc_id"],
            saved["source_type"],
            chunk_count,
        )
        return saved


def get_document(doc_id: str) -> Optional[Dict[str, Any]]:
    with _registry_lock:
        registry = _load_registry()
    return next((doc for doc in registry.get("documents", []) if doc.get("doc_id") == doc_id), None)


def list_documents(owner_id: Optional[str] = None, source_type: Optional[str] = None) -> List[Dict[str, Any]]:
    with _registry_lock:
        registry = _load_registry()

    docs = list(registry.get("documents", []))
    if source_type:
        docs = [doc for doc in docs if doc.get("source_type") == source_type]

    if owner_id:
        docs = [
            doc
            for doc in docs
            if (doc.get("source_type") != "upload") or (doc.get("owner_id") == owner_id)
        ]

    return docs


def remove_documents(
    doc_id: Optional[str] = None,
    source: Optional[str] = None,
    source_type: Optional[str] = None,
    owner_id: Optional[str] = None,
) -> int:
    """Remove registry entries matching all provided filters."""
    if not any([doc_id, source, source_type, owner_id]):
        return 0

    with _registry_lock:
        registry = _load_registry()
        docs = registry.get("documents", [])
        kept = []
        removed = 0

        for doc in docs:
            matches = True
            if doc_id and doc.get("doc_id") != doc_id:
                matches = False
            if source and doc.get("source") != source:
                matches = False
            if source_type and doc.get("source_type") != source_type:
                matches = False
            if owner_id:
                if doc.get("source_type") != "upload":
                    matches = False
                elif doc.get("owner_id") != owner_id:
                    matches = False

            if matches:
                removed += 1
            else:
                kept.append(doc)

        if removed:
            registry["documents"] = kept
            _save_registry(registry)

    if removed:
        logger.info(
            "Removed %d documents from registry (source=%s, doc_id=%s, source_type=%s, owner_id=%s)",
            removed,
            source,
            doc_id,
            source_type,
            owner_id,
        )
    return removed


def clear_document_registry() -> None:
    with _registry_lock:
        _save_registry({"documents": []})
    logger.info("Document registry cleared.")
