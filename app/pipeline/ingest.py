"""
Document ingestion orchestrator: scan folder, parse, chunk, and store.

Improvements over the original version:
- Skips already-ingested files
- Returns detailed per-file status
- Separates orchestration from parsing/chunking
- Supports single-file ingestion for the upload endpoint
"""

import glob
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.documents import Document

from app.config import LIBRARY_DIR
from app.core.document_registry import infer_source_type, upsert_document
from app.core.vectorstore import add_documents, get_ingested_sources
from app.pipeline.chunker import chunk_documents
from app.pipeline.enricher import enrich_chunks
from app.pipeline.parser import parse_image, parse_pdf

logger = logging.getLogger("tilon.ingest")


def _annotate_source_identity(
    docs: List[Document],
    file_path: Path,
    owner_id: str = None,
) -> List[Document]:
    """Add source identity metadata before chunking so it survives downstream."""
    source_type = infer_source_type(file_path)
    doc_scope = "persistent" if source_type == "library" else "chat_upload"
    annotated = []
    for doc in docs:
        annotated.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "source_type": source_type,
                    "doc_scope": doc_scope,
                    "owner_id": owner_id if source_type == "upload" else None,
                },
            )
        )
    return annotated


def ingest_single_file(file_path: Path, owner_id: str = None) -> Dict[str, Any]:
    """
    Parse, chunk, and store a single file into the vectorstore.
    Used by the /upload endpoint when users attach files in chat.
    """
    if not file_path.exists():
        return {"message": f"File not found: {file_path.name}", "count": 0}

    ext = file_path.suffix.lower()
    name = file_path.name

    logger.info("Ingesting single file: %s", name)
    started_at = time.perf_counter()

    parse_started_at = time.perf_counter()
    if ext == ".pdf":
        docs = parse_pdf(str(file_path))
    elif ext in (".png", ".jpg", ".jpeg", ".webp"):
        docs = parse_image(str(file_path))
    else:
        return {
            "message": f"Unsupported file type: {ext}",
            "count": 0,
            "file": name,
        }
    parse_seconds = time.perf_counter() - parse_started_at

    docs = _annotate_source_identity(docs, file_path, owner_id=owner_id)

    if not docs:
        return {
            "message": (
                f"Could not extract text from {name}. "
                "The file may be a scanned image; check whether OCR is enabled."
            ),
            "count": 0,
            "file": name,
        }

    chunk_started_at = time.perf_counter()
    chunks = chunk_documents(docs)
    chunk_seconds = time.perf_counter() - chunk_started_at

    enrich_started_at = time.perf_counter()
    chunks = enrich_chunks(chunks)
    enrich_seconds = time.perf_counter() - enrich_started_at

    store_started_at = time.perf_counter()
    add_documents(chunks)
    registry_entry = upsert_document(file_path, docs, len(chunks), owner_id=owner_id)
    store_seconds = time.perf_counter() - store_started_at
    total_seconds = time.perf_counter() - started_at

    logger.info(
        "Ingested %s -> %d chunks (parse=%.2fs, chunk=%.2fs, enrich=%.2fs, store=%.2fs, total=%.2fs)",
        name,
        len(chunks),
        parse_seconds,
        chunk_seconds,
        enrich_seconds,
        store_seconds,
        total_seconds,
    )

    return {
        "message": f"Successfully ingested {name}: {len(chunks)} chunks stored.",
        "count": len(chunks),
        "file": name,
        "doc_id": registry_entry.get("doc_id") if registry_entry else None,
        "source_type": registry_entry.get("source_type") if registry_entry else None,
        "timings": {
            "parse_seconds": round(parse_seconds, 2),
            "chunk_seconds": round(chunk_seconds, 2),
            "enrich_seconds": round(enrich_seconds, 2),
            "store_seconds": round(store_seconds, 2),
            "total_seconds": round(total_seconds, 2),
        },
    }


def ingest_folder(folder_path: Path = None) -> Dict[str, Any]:
    """
    Scan a folder for PDFs and images, parse, chunk, and store.
    Skips files that are already in the vectorstore.
    """
    folder = folder_path or LIBRARY_DIR
    folder.mkdir(parents=True, exist_ok=True)

    already_ingested = get_ingested_sources()
    logger.info("Ingesting from %s -> %d files already in DB", folder, len(already_ingested))

    pdf_files = sorted(glob.glob(str(folder / "*.pdf")))
    image_files = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        image_files.extend(glob.glob(str(folder / ext)))
    image_files = sorted(image_files)

    processed_files = []
    skipped_files = []
    total_chunks = 0

    for file_path_str in pdf_files + image_files:
        path = Path(file_path_str)
        name = path.name
        if name in already_ingested:
            skipped_files.append(name)
            continue

        result = ingest_single_file(path)
        if result.get("count", 0) > 0:
            processed_files.append(name)
            total_chunks += int(result["count"])

    if skipped_files:
        logger.info("Skipped %d already-ingested files: %s", len(skipped_files), skipped_files)

    if total_chunks == 0:
        return {
            "message": "No new documents to ingest.",
            "count": 0,
            "files": [],
            "skipped": skipped_files,
        }

    return {
        "message": f"Ingested {total_chunks} chunks from {len(processed_files)} files.",
        "count": total_chunks,
        "files": processed_files,
        "skipped": skipped_files,
    }
