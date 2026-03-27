"""
Embedding model singleton.

IMPROVEMENTS over original:
- GPU auto-detection (original hardcoded CPU — very slow for bge-m3)
- Lazy loading with proper logging
- Isolated from other concerns
- NEW: runtime device switch (CUDA OOM -> CPU fallback)
"""

import logging
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import EMBEDDING_MODEL, EMBEDDING_DEVICE

logger = logging.getLogger("tilon.embeddings")

_embedding_model: Optional[HuggingFaceEmbeddings] = None
_embedding_device: str = EMBEDDING_DEVICE


def _load_embeddings(device: str) -> HuggingFaceEmbeddings:
    """Load embedding model on the requested device."""
    logger.info(
        "Loading embedding model '%s' on device '%s'...",
        EMBEDDING_MODEL,
        device,
    )
    model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        # Keep GPU memory usage predictable during ingestion.
        encode_kwargs={"normalize_embeddings": True, "batch_size": 8},
    )
    logger.info("Embedding model loaded successfully.")
    return model


def get_embeddings(force_device: Optional[str] = None) -> HuggingFaceEmbeddings:
    global _embedding_model
    global _embedding_device

    requested_device = force_device or _embedding_device

    if _embedding_model is not None and requested_device == _embedding_device:
        return _embedding_model

    _embedding_model = _load_embeddings(requested_device)
    _embedding_device = requested_device
    return _embedding_model


def get_embedding_device() -> str:
    """Return currently active embedding device."""
    return _embedding_device


def switch_embeddings_device(device: str) -> HuggingFaceEmbeddings:
    """Force-switch embedding backend device and reload model."""
    global _embedding_model
    global _embedding_device

    if device == _embedding_device and _embedding_model is not None:
        return _embedding_model

    _embedding_model = None
    _embedding_device = device
    return get_embeddings(force_device=device)
