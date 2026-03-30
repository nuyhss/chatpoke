"""
OpenAI-compatible API endpoints (/v1/models, /v1/chat/completions).
Uses the unified chat handler — no mode routing.
"""

import uuid
import time
import logging
from typing import List

from fastapi import APIRouter, HTTPException

from app.config import OLLAMA_MODEL, AVAILABLE_MODELS
from app.models.schemas import Message, OpenAIMessage, OpenAIChatRequest
from app.chat.handlers import handle_chat

logger = logging.getLogger("tilon.openai_compat")

router = APIRouter(prefix="/v1", tags=["OpenAI Compatible"])


def _convert_openai_messages(messages: List[OpenAIMessage]):
    """Convert OpenAI-format messages to internal format."""
    system_prompt = None
    history: List[Message] = []
    user_message = ""

    for msg in messages:
        if msg.role == "system":
            system_prompt = msg.content
        elif msg.role == "user":
            user_message = msg.content
            history.append(Message(role="user", content=msg.content))
        elif msg.role == "assistant":
            history.append(Message(role="assistant", content=msg.content))

    if history and history[-1].role == "user":
        user_message = history[-1].content
        history = history[:-1]

    return system_prompt, history, user_message


@router.get("/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": model_id, "object": "model", "created": 0, "owned_by": "local"}
            for model_id in AVAILABLE_MODELS
        ],
    }


@router.post("/chat/completions")
def chat_completions(req: OpenAIChatRequest):
    try:
        if req.stream:
            raise HTTPException(
                status_code=400,
                detail="stream=true is not supported by this server yet. Use non-streaming requests.",
            )

        system_prompt, history, user_message = _convert_openai_messages(req.messages)
        selected_model = req.model or OLLAMA_MODEL

        result = handle_chat(
            user_message=user_message,
            history=history,
            model=selected_model,
            system_prompt=system_prompt,
        )
        response_metadata = result.get("response_metadata") or {}
        usage = response_metadata.get("usage") or {}
        request_id = response_metadata.get("request_id") or uuid.uuid4().hex

        return {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": selected_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result["answer"]},
                    "finish_reason": response_metadata.get("finish_reason") or "stop",
                }
            ],
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens") or 0,
                "completion_tokens": usage.get("completion_tokens") or 0,
                "total_tokens": usage.get("total_tokens") or 0,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("OpenAI-compatible chat failed")
        raise HTTPException(status_code=500, detail=f"OpenAI-compatible chat failed: {e}")
