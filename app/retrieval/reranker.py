"""
Reranker using BGE-reranker-v2-m3.

NEW MODULE — not in the original code at all.

The original retrieved top-4 by vector similarity only.
A reranker re-scores results using cross-attention between the query
and each candidate, dramatically improving precision.

This is especially important for Korean+English mixed documents where
embedding similarity alone can miss semantic matches.
"""

import logging
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from app.config import (
    RERANKER_ENABLED,
    RERANKER_MODEL,
    RERANKER_TOP_N,
    RERANKER_DEVICE,
    RERANKER_USE_FP16,
)

logger = logging.getLogger("tilon.reranker")

_reranker = None
_reranker_load_failed = False
_reranker_device = RERANKER_DEVICE


def _load_reranker(force_device: Optional[str] = None):
    """Lazy-load the reranker model."""
    global _reranker
    global _reranker_load_failed
    global _reranker_device

    requested_device = force_device or _reranker_device

    if _reranker is not None and requested_device == _reranker_device:
        return _reranker

    if _reranker_load_failed:
        return None

    if not RERANKER_ENABLED:
        return None

    try:
        from FlagEmbedding import FlagReranker

        logger.info(
            "Loading reranker model '%s' (device=%s, fp16=%s)...",
            RERANKER_MODEL,
            requested_device,
            RERANKER_USE_FP16 if requested_device == "cuda" else False,
        )
        _reranker = FlagReranker(
            RERANKER_MODEL,
            use_fp16=RERANKER_USE_FP16 if requested_device == "cuda" else False,
            devices=requested_device,
        )
        _reranker_device = requested_device
        logger.info("Reranker loaded successfully.")
        return _reranker
    except ImportError:
        _reranker_load_failed = True
        logger.warning(
            "FlagEmbedding not installed — reranking disabled. "
            "Install with: pip install FlagEmbedding"
        )
        return None
    except Exception as e:
        _reranker_load_failed = True
        logger.error("Failed to load reranker: %s", e)
        return None


def rerank(
    query: str,
    documents: List[Document],
    top_n: Optional[int] = None,
) -> List[Document]:
    """
    Re-score documents against the query and return top_n best matches.

    If reranker is disabled or unavailable, returns documents unchanged.
    """
    if not RERANKER_ENABLED or not documents:
        return documents

    reranker = _load_reranker()
    if reranker is None:
        return documents

    top_n = top_n or RERANKER_TOP_N

    # Build query-document pairs
    pairs = [[query, doc.page_content] for doc in documents]

    try:
        scores = reranker.compute_score(pairs, normalize=True)

        # Handle single result (returns float instead of list)
        if isinstance(scores, (float, int)):
            scores = [scores]

        # Sort by score descending
        scored_docs: List[Tuple[float, Document]] = list(zip(scores, documents))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        results = [doc for _, doc in scored_docs[:top_n]]
        logger.debug(
            "Reranked %d → %d docs (scores: %s)",
            len(documents),
            len(results),
            [f"{s:.3f}" for s, _ in scored_docs[:top_n]],
        )
        return results

    except RuntimeError as e:
        if "out of memory" in str(e).lower() and _reranker_device != "cpu":
            logger.warning("Reranker hit CUDA OOM, retrying on CPU.")
            try:
                reranker = _load_reranker(force_device="cpu")
                if reranker is None:
                    return documents
                scores = reranker.compute_score(pairs, normalize=True)
                if isinstance(scores, (float, int)):
                    scores = [scores]
                scored_docs: List[Tuple[float, Document]] = list(zip(scores, documents))
                scored_docs.sort(key=lambda x: x[0], reverse=True)
                return [doc for _, doc in scored_docs[:top_n]]
            except Exception as cpu_error:
                logger.error("CPU reranking fallback failed, returning original order: %s", cpu_error)
                return documents
        logger.error("Reranking failed, returning original order: %s", e)
        return documents
    except Exception as e:
        logger.error("Reranking failed, returning original order: %s", e)
        return documents
