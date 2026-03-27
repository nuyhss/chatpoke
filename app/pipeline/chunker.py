"""
Semantic chunker — splits documents by meaning, not character count.

Strategy:
1. Detect markdown headings → split at section boundaries
2. Detect tables → keep as independent chunks
3. Large sections → split at paragraph boundaries
4. Oversized paragraphs → character-based fallback with Korean separators
5. Each chunk carries heading breadcrumb (e.g., "설치 가이드 > 요구사항")

Why: Old chunker split at 1200 chars blindly. A chunk could be half of
"Requirements" + half of "Installation". This chunker keeps sections whole.
"""

import re
import uuid
import time
import logging
from typing import List, Tuple
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    LARGE_FILE_FAST_MODE,
    LARGE_FILE_PAGE_THRESHOLD,
    LARGE_FILE_CHUNK_SIZE,
    LARGE_FILE_CHUNK_OVERLAP,
)

logger = logging.getLogger("tilon.chunker")

_HEADING_RE = re.compile(r'^(#{1,4})\s+(.+)$', re.MULTILINE)
_TABLE_RE = re.compile(r'(?:^[|].*[|]\s*\n?){2,}', re.MULTILINE)

_KOREAN_SEPARATORS = [
    "\n\n", "\n", "다. ", "요. ", "다.\n", "요.\n",
    ". ", "? ", "! ", " ", "",
]


@dataclass
class Section:
    heading: str
    level: int
    content: str
    breadcrumb: str
    is_table: bool = False


def _resolve_chunk_params(base_meta: dict) -> Tuple[int, int]:
    """Use larger chunks for very large documents to reduce embedding workload."""
    page_total = 0
    try:
        page_total = int(base_meta.get("page_total") or 0)
    except Exception:
        page_total = 0

    if LARGE_FILE_FAST_MODE and page_total >= LARGE_FILE_PAGE_THRESHOLD:
        chunk_size = max(CHUNK_SIZE, LARGE_FILE_CHUNK_SIZE)
        chunk_overlap = min(max(0, LARGE_FILE_CHUNK_OVERLAP), max(0, chunk_size - 1))
        return chunk_size, chunk_overlap

    return CHUNK_SIZE, CHUNK_OVERLAP


def _split_by_headings(text: str) -> List[Section]:
    """Split text into sections based on markdown headings with breadcrumb tracking."""
    sections = []
    heading_stack: List[Tuple[int, str]] = []
    matches = list(_HEADING_RE.finditer(text))

    if not matches:
        return [Section(heading="", level=0, content=text.strip(), breadcrumb="")]

    pre = text[:matches[0].start()].strip()
    if pre:
        sections.append(Section(heading="", level=0, content=pre, breadcrumb=""))

    for i, match in enumerate(matches):
        level = len(match.group(1))
        heading_text = match.group(2).strip()
        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[content_start:content_end].strip()

        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, heading_text))
        breadcrumb = " > ".join(h[1] for h in heading_stack)

        sections.append(Section(
            heading=heading_text, level=level,
            content=content, breadcrumb=breadcrumb,
        ))

    return sections


def _extract_tables(text: str) -> Tuple[str, List[str]]:
    """Extract markdown tables, return (remaining_text, [tables])."""
    tables = [m.group(0).strip() for m in _TABLE_RE.finditer(text) if len(m.group(0).strip()) > 50]
    remaining = _TABLE_RE.sub('\n\n', text).strip() if tables else text
    return remaining, tables


def _split_paragraphs(text: str, chunk_size: int) -> List[str]:
    """Split by paragraphs, group small ones together."""
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    grouped, current, current_len = [], [], 0

    for para in paragraphs:
        if current_len + len(para) > chunk_size and current:
            grouped.append("\n\n".join(current))
            current, current_len = [para], len(para)
        else:
            current.append(para)
            current_len += len(para)
    if current:
        grouped.append("\n\n".join(current))
    return grouped


def _char_split(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Fallback character-based split with Korean separators."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_KOREAN_SEPARATORS,
    ).split_text(text)


def _make_chunk(text, metadata, idx, section_title="", breadcrumb="", chunk_type="section"):
    return Document(
        page_content=text,
        metadata={
            **metadata,
            "chunk_index": idx,
            "chunk_id": str(uuid.uuid4()),
            "chunk_chars": len(text),
            "chunk_type": chunk_type,
            "section_title": section_title,
            "section_breadcrumb": breadcrumb,
            "timestamp": int(time.time()),
        },
    )


def _chunk_section(
    section: Section,
    base_meta: dict,
    start_idx: int,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """Chunk one section: tables → independent, text → whole or split."""
    chunks = []
    if not section.content.strip():
        return chunks

    remaining, tables = _extract_tables(section.content)

    # Table chunks
    for table in tables:
        pieces = [table] if len(table) <= chunk_size else _char_split(table, chunk_size, chunk_overlap)
        for piece in pieces:
            chunks.append(_make_chunk(
                piece, base_meta, start_idx + len(chunks),
                section.heading, section.breadcrumb, "table",
            ))

    # Text chunks
    if not remaining.strip():
        return chunks

    if len(remaining) <= chunk_size:
        chunks.append(_make_chunk(
            remaining, base_meta, start_idx + len(chunks),
            section.heading, section.breadcrumb, "section",
        ))
    else:
        for para in _split_paragraphs(remaining, chunk_size):
            pieces = [para] if len(para) <= chunk_size else _char_split(para, chunk_size, chunk_overlap)
            for piece in pieces:
                chunks.append(_make_chunk(
                    piece, base_meta, start_idx + len(chunks),
                    section.heading, section.breadcrumb, "paragraph",
                ))

    return chunks


def chunk_documents(docs: List[Document]) -> List[Document]:
    """
    Semantic chunking: split documents by meaning, not character count.

    For documents with headings: split at section boundaries.
    For documents without: split at paragraph boundaries.
    Tables always become independent chunks.
    """
    all_chunks: List[Document] = []
    total_sections = 0
    fast_chunk_docs = 0

    for doc in docs:
        text = doc.page_content
        # Copy metadata without chunk-specific fields
        base_meta = {
            k: v for k, v in doc.metadata.items()
            if k not in (
                "chunk_index", "chunk_id", "chunk_chars",
                "chunk_type", "section_breadcrumb", "timestamp",
            )
        }

        chunk_size, chunk_overlap = _resolve_chunk_params(base_meta)
        if chunk_size != CHUNK_SIZE:
            fast_chunk_docs += 1

        has_headings = bool(_HEADING_RE.search(text))

        if has_headings:
            sections = _split_by_headings(text)
            total_sections += len(sections)
            for section in sections:
                all_chunks.extend(
                    _chunk_section(
                        section,
                        base_meta,
                        len(all_chunks),
                        chunk_size,
                        chunk_overlap,
                    )
                )
        else:
            # No headings — paragraph-based
            remaining, tables = _extract_tables(text)
            section_title = base_meta.get("section_title", "")

            for table in tables:
                pieces = [table] if len(table) <= chunk_size else _char_split(table, chunk_size, chunk_overlap)
                for piece in pieces:
                    all_chunks.append(_make_chunk(
                        piece, base_meta, len(all_chunks), section_title, "", "table",
                    ))

            if remaining.strip():
                if len(remaining) <= chunk_size:
                    all_chunks.append(_make_chunk(
                        remaining, base_meta, len(all_chunks), section_title, "", "page",
                    ))
                else:
                    for para in _split_paragraphs(remaining, chunk_size):
                        pieces = [para] if len(para) <= chunk_size else _char_split(para, chunk_size, chunk_overlap)
                        for piece in pieces:
                            all_chunks.append(_make_chunk(
                                piece, base_meta, len(all_chunks), section_title, "", "paragraph",
                            ))

    types = {}
    for c in all_chunks:
        t = c.metadata.get("chunk_type", "?")
        types[t] = types.get(t, 0) + 1

    logger.info(
        "Semantic chunking: %d docs → %d sections → %d chunks | %s",
        len(docs), total_sections, len(all_chunks), types,
    )

    if fast_chunk_docs > 0:
        logger.info(
            "Large-file chunk profile applied to %d/%d docs (chunk_size=%d, overlap=%d)",
            fast_chunk_docs,
            len(docs),
            LARGE_FILE_CHUNK_SIZE,
            LARGE_FILE_CHUNK_OVERLAP,
        )

    return all_chunks
