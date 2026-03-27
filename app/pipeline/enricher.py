"""
Contextual enrichment — prepends document/section context to each chunk.

Before enrichment:
    chunk = "Click the button to confirm"
    → embedding captures: "clicking a button"

After enrichment:
    chunk = "[Dstation Admin Manual > User Management > Password Reset | p.12]
             Click the button to confirm"
    → embedding captures: "Dstation password reset procedure involving clicking a button"

This dramatically improves retrieval accuracy because the embedding now
encodes WHAT the chunk is about, not just the surface text.
"""

import logging
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger("tilon.enricher")


def enrich_chunks(chunks: List[Document]) -> List[Document]:
    """
    Prepend contextual header to each chunk's content before embedding.

    The header format is:
    [Document: {source} | Section: {breadcrumb} | Page: {page} | Lang: {lang}]

    This same format is used by retriever.py:format_context() when building
    the LLM prompt, so embeddings and prompts are aligned.
    """
    enriched = []

    for chunk in chunks:
        meta = chunk.metadata
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")
        lang = meta.get("language", "unknown")

        # Use breadcrumb if available, fall back to section_title
        breadcrumb = meta.get("section_breadcrumb", "")
        section = breadcrumb or meta.get("section_title", "")

        # Build context header
        parts = [f"Document: {source}"]
        if section:
            parts.append(f"Section: {section}")
        parts.append(f"Page: {page}")
        if lang and lang != "unknown":
            parts.append(f"Lang: {lang}")

        header = "[" + " | ".join(parts) + "]"

        # Prepend header to content
        enriched_content = f"{header}\n{chunk.page_content}"

        enriched.append(Document(
            page_content=enriched_content,
            metadata={
                **meta,
                # Store the header separately for debugging
                "context_header": header,
                # Update section_title with breadcrumb for retriever
                "section_title": section,
            },
        ))

    logger.info(
        "Enriched %d chunks with contextual headers (avg +%d chars)",
        len(enriched),
        sum(len(c.metadata.get("context_header", "")) for c in enriched) // max(len(enriched), 1),
    )

    return enriched