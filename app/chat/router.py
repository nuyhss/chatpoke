"""
Chat mode detection / routing.

Determines whether a query needs: document search, web search, OCR extraction,
or is a general conversation. Uses keyword matching + heuristics.

Full LLM-based intent classification is planned for Session 3.
"""

import logging
from typing import Literal

logger = logging.getLogger("tilon.router")

ModeType = Literal["general", "document_qa", "web_search", "ocr_extract"]

# ── OCR: user wants raw text extraction ──
OCR_KEYWORDS = [
    "ocr", "텍스트만", "텍스트 추출", "원문", "글자 추출",
    "읽어줘", "뽑아줘", "추출해줘", "문자 인식", "extract text",
    "read the text", "show the text", "only text", "글자만",
]

# ── Document: user is asking about uploaded/stored documents ──
DOCUMENT_KEYWORDS = [
    "pdf", "문서", "파일", "업로드", "페이지", "요약", "summarize",
    "첨부", "첨부파일", "이 문서", "이 파일", "이 pdf", "내용",
    "summarize this", "read the uploaded", "document", "what does",
    "이 자료", "보고서", "매뉴얼", "설명서", "안내서",
    "몇 페이지", "어디에", "어떤 내용", "무슨 내용",
    "section", "chapter", "table of contents", "index",
]

# ── Web search: user needs real-time or current information ──
WEB_SEARCH_KEYWORDS = [
    # Korean time/recency
    "오늘", "현재", "최신", "최근", "실시간", "지금", "올해", "이번",
    "어제", "내일", "이번 주", "이번 달", "금일",
    # Korean topics that need live data
    "주가", "날씨", "뉴스", "환율", "현재가", "시세", "경기", "선거",
    "코로나", "지진", "사고", "속보", "발표", "결과",
    # Korean question patterns for live info
    "누가 이겼", "몇 도", "얼마", "몇 시", "언제 열리",
    # English time/recency
    "today", "current", "latest", "recent", "now", "this week",
    "this month", "this year", "yesterday", "tomorrow", "live",
    "right now", "as of", "up to date", "breaking",
    # English topics
    "stock price", "weather", "news", "exchange rate", "score",
    "election", "trending", "update", "announcement", "released",
    # Search intent patterns
    "검색", "찾아줘", "알려줘", "search for", "look up", "find out",
    "what happened", "who won", "how much is", "what is the price",
]


def detect_mode(user_message: str, has_file: bool = False) -> ModeType:
    """
    Detect the appropriate response mode.

    Args:
        user_message: The user's text input
        has_file: Whether a file was uploaded with this message.
                  If True, defaults to document_qa instead of general.
    """
    text = user_message.lower().strip()

    # Priority 1: OCR extraction request
    if any(kw in text for kw in OCR_KEYWORDS):
        mode = "ocr_extract"

    # Priority 2: Web search (real-time info)
    elif any(kw in text for kw in WEB_SEARCH_KEYWORDS):
        mode = "web_search"

    # Priority 3: Document question
    elif any(kw in text for kw in DOCUMENT_KEYWORDS):
        mode = "document_qa"

    # Priority 4: File was uploaded → default to document_qa
    elif has_file:
        mode = "document_qa"

    # Default: general conversation
    else:
        mode = "general"

    logger.info("Mode: %s | has_file: %s | query: '%s'", mode, has_file, text[:60])
    return mode