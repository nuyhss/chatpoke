"""
Speech-to-text helpers.

Lazy-loads Whisper so the main API does not pay startup cost unless STT is used.
"""

from __future__ import annotations

import os
import tempfile
from threading import Lock
from typing import Optional

from fastapi import HTTPException

from app.config import WHISPER_LANGUAGE, WHISPER_MODEL

_WHISPER_MODEL = None
_WHISPER_LOCK = Lock()


def _get_whisper_model():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL

    with _WHISPER_LOCK:
        if _WHISPER_MODEL is None:
            try:
                import whisper
            except ImportError as exc:
                raise HTTPException(
                    status_code=500,
                    detail="Whisper is not installed. Install openai-whisper to enable STT.",
                ) from exc

            _WHISPER_MODEL = whisper.load_model(WHISPER_MODEL)

    return _WHISPER_MODEL


def transcribe_audio_bytes(
    content: bytes,
    filename: str,
    language: Optional[str] = None,
) -> str:
    """Transcribe uploaded audio bytes into text."""
    allowed_exts = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}
    ext = os.path.splitext((filename or "").lower())[1]

    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio type: {ext or 'unknown'}. Allowed: {sorted(allowed_exts)}",
        )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        model = _get_whisper_model()
        result = model.transcribe(tmp_path, language=language or WHISPER_LANGUAGE)
        text = (result.get("text") or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="No recognizable speech was found in the audio.")
        return text
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"STT failed: {exc}") from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
