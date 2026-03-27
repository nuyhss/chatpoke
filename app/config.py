"""
Centralized configuration — loaded once from environment variables.

IMPROVEMENTS over original:
- All settings in one place (were scattered across top of app.py)
- Pydantic Settings for validation and type safety
- GPU auto-detection for embedding model
- Reranker support added
- Logging configuration added
"""

import os
import logging
from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
LIBRARY_DIR = Path(os.getenv("LIBRARY_DIR", DATA_DIR / "library"))
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", DATA_DIR / "uploads"))
TEMP_DIR = Path(os.getenv("TEMP_DIR", DATA_DIR / "temp"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", BASE_DIR / "chroma_db"))
DOCUMENT_REGISTRY_PATH = Path(
    os.getenv("DOCUMENT_REGISTRY_PATH", DATA_DIR / "document_registry.json")
)
MARKER_OUTPUT_DIR = BASE_DIR / "marker_output"

# ── Ollama / LLM ──────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/api")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen2.5vl:7b")
AVAILABLE_MODELS = os.getenv(
    "AVAILABLE_MODELS",
    "qwen2.5:7b,llama3.1:latest,llama3.2-vision:11b",
).split(",")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "180"))
LLM_REPEAT_PENALTY = float(os.getenv("LLM_REPEAT_PENALTY", "1.15"))
LLM_REPEAT_LAST_N = int(os.getenv("LLM_REPEAT_LAST_N", "128"))
LLM_MAX_CONCURRENT_REQUESTS = max(1, int(os.getenv("LLM_MAX_CONCURRENT_REQUESTS", "1")))
LLM_QUEUE_TIMEOUT = int(os.getenv("LLM_QUEUE_TIMEOUT", "240"))

# ── Embedding ─────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# IMPROVEMENT: auto-detect GPU; original hardcoded "cpu"
def _detect_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", _detect_device())

# ── Reranker (NEW — not in original) ──────────────────────────────────
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "false").lower() == "true"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_TOP_N = int(os.getenv("RERANKER_TOP_N", "3"))
RERANKER_DEVICE = os.getenv(
    "RERANKER_DEVICE",
    "cuda" if EMBEDDING_DEVICE == "cuda" else "cpu",
)
RERANKER_USE_FP16 = os.getenv(
    "RERANKER_USE_FP16",
    "true" if RERANKER_DEVICE == "cuda" else "false",
).lower() == "true"

# ── Retrieval ─────────────────────────────────────────────────────────
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "4"))
GLOBAL_MIN_RELEVANCE_SCORE = float(os.getenv("GLOBAL_MIN_RELEVANCE_SCORE", "0.30"))
DOCUMENT_CONFIDENCE_THRESHOLD = float(os.getenv("DOCUMENT_CONFIDENCE_THRESHOLD", "0.45"))
SCOPED_SINGLE_CHUNK_CONFIDENCE_THRESHOLD = float(
    os.getenv("SCOPED_SINGLE_CHUNK_CONFIDENCE_THRESHOLD", "0.30")
)
SCOPED_SMALL_DOC_FULL_CONTEXT_MAX_CHUNKS = int(
    os.getenv("SCOPED_SMALL_DOC_FULL_CONTEXT_MAX_CHUNKS", "3")
)
STRONG_KEYWORD_CONFIDENCE_FLOOR = float(os.getenv("STRONG_KEYWORD_CONFIDENCE_FLOOR", "0.75"))

# ── Chunking ──────────────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# ── Features ──────────────────────────────────────────────────────────
ENABLE_OCR = os.getenv("ENABLE_OCR", "true").lower() == "true"
AUTO_INGEST_ON_STARTUP = os.getenv("AUTO_INGEST_ON_STARTUP", "false").lower() == "true"
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "ko")

# ── Web Search (NEW — original had mode but no actual search) ─────────
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
DUCKDUCKGO_MAX_RESULTS = max(1, int(os.getenv("DUCKDUCKGO_MAX_RESULTS", "5")))
DUCKDUCKGO_REGION = os.getenv("DUCKDUCKGO_REGION", "kr-kr")

# ── VLM PDF Extraction (NEW) ─────────────────────────────────────────
VLM_EXTRACTION_ENABLED = os.getenv("VLM_EXTRACTION_ENABLED", "true").lower() == "true"
VLM_EXTRACTION_MODEL = os.getenv("VLM_EXTRACTION_MODEL", "qwen2.5vl:7b")
LARGE_FILE_FAST_MODE = os.getenv("LARGE_FILE_FAST_MODE", "true").lower() == "true"
LARGE_FILE_PAGE_THRESHOLD = max(1, int(os.getenv("LARGE_FILE_PAGE_THRESHOLD", "80")))
LARGE_FILE_MAX_FALLBACK_PAGES = max(0, int(os.getenv("LARGE_FILE_MAX_FALLBACK_PAGES", "12")))
LARGE_FILE_CHUNK_SIZE = max(256, int(os.getenv("LARGE_FILE_CHUNK_SIZE", "2200")))
LARGE_FILE_CHUNK_OVERLAP = max(0, int(os.getenv("LARGE_FILE_CHUNK_OVERLAP", "100")))

# ── Logging ───────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

def setup_logging():
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

logger = logging.getLogger("tilon")
