"""
Vision helpers for direct image analysis with Ollama chat models.
"""

from __future__ import annotations

import base64
import os
from typing import Optional

import requests
from fastapi import HTTPException

from app.config import LLM_TIMEOUT, OLLAMA_BASE_URL, VISION_MODEL


def analyze_image_bytes(
    content: bytes,
    filename: str,
    prompt: str,
    model: Optional[str] = None,
) -> str:
    """Send one image + prompt to the configured Ollama vision model."""
    allowed_exts = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
    ext = os.path.splitext((filename or "").lower())[1]
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type: {ext or 'unknown'}. Allowed: {sorted(allowed_exts)}",
        )

    image_b64 = base64.b64encode(content).decode("utf-8")
    payload = {
        "model": model or VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_b64],
            }
        ],
        "stream": False,
    }

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/chat",
            json=payload,
            timeout=LLM_TIMEOUT,
        )
        if response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Vision model error: status={response.status_code}, body={response.text[:500]}",
            )

        data = response.json()
        return (data.get("message", {}) or {}).get("content", "").strip()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {exc}") from exc
