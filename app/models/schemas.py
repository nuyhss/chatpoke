"""
Request / Response schemas for all API endpoints.

IMPROVEMENTS over original:
- Separated from business logic (were mixed into app.py)
- Added response models for type safety
- Added model validation
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


# ── Chat ───────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str
    system_prompt: Optional[str] = (
        "너는 한국어로 답하는 AI 챗봇이다. "
        "짧은 질문에는 짧고 자연스럽게 답한다. "
        "문서 질문은 문서 근거로만 답하고, 근거가 없으면 모른다고 말한다."
    )
    history: List[Message] = Field(default_factory=list)
    model: Optional[str] = None
    active_source: Optional[str] = None
    active_doc_id: Optional[str] = None
    active_source_type: Optional[str] = None
    web_search_enabled: bool = True
    user_id: Optional[str] = None


class SourceInfo(BaseModel):
    doc_id: Optional[str] = None
    source: Optional[str] = None
    source_type: Optional[str] = None
    source_path: Optional[str] = None
    page: Optional[int] = None
    section: Optional[str] = None
    chunk_index: Optional[int] = None
    chunk_type: Optional[str] = None
    extraction_method: Optional[str] = None


class ChatResponse(BaseModel):
    model: str
    answer: str
    sources: List[SourceInfo] = Field(default_factory=list)
    mode: str
    active_source: Optional[str] = None
    active_doc_id: Optional[str] = None
    done: bool = True


# ── Ingest ─────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    folder_path: Optional[str] = None


class IngestResponse(BaseModel):
    message: str
    count: int
    files: List[str] = []


# ── Keyword Count ──────────────────────────────────────────────────────

class CountKeywordRequest(BaseModel):
    filename: str
    keyword: str


class WebSearchRequest(BaseModel):
    query: str
    max_results: int = Field(default=5, ge=1, le=10)
    region: str = "kr-kr"


# ── OpenAI-Compatible ──────────────────────────────────────────────────

class OpenAIMessage(BaseModel):
    role: str
    content: str


class OpenAIChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[OpenAIMessage]
    temperature: Optional[float] = 0.2
    stream: Optional[bool] = False
