"""
Unified chat handler — works like a normal chatbot.

NO hardcoded mode routing. Instead:
1. Always search vectorstore for relevant document context
2. If web search might help, search Tavily too
3. Build one prompt with all available context
4. LLM decides what to use

This is how ChatGPT/Claude work — retrieve first, let the model decide.
"""

import logging
from collections import Counter
from pathlib import Path
import re
from typing import List, Dict, Any, Optional

from app.models.schemas import Message
from app.core.llm import call_ollama, get_response_text
from app.core.web_search import format_search_results, search_web
from app.retrieval.retriever import retrieve, format_context, extract_sources
from app.core.vectorstore import get_documents_by_source, get_document_chunk_count
from app.core.document_registry import list_documents
from app.pipeline.parser import extract_full_text
from app.config import (
    OLLAMA_MODEL,
    DOCUMENT_CONFIDENCE_THRESHOLD,
    SCOPED_SINGLE_CHUNK_CONFIDENCE_THRESHOLD,
    SCOPED_SMALL_DOC_FULL_CONTEXT_MAX_CHUNKS,
    UPLOADS_DIR,
)

logger = logging.getLogger("tilon.chat")
_HANGUL_RE = re.compile(r"[가-힣]")
_CJK_HAN_RE = re.compile(r"[\u4e00-\u9fff]")
_CJK_PUNCT_RE = re.compile(r"[，。！？；：、﹐﹒﹔﹕「」『』【】《》〈〉（）〔〕］［]")


# ═══════════════════════════════════════════════════════════════════════
# Web Search (additive, not a separate mode)
# ═══════════════════════════════════════════════════════════════════════

def _search_web(query: str) -> str:
    """Search current information using the shared search helper."""
    try:
        return format_search_results(search_web(query, max_results=3))
    except Exception as e:
        logger.debug("Web search failed: %s", e)
        return ""


def _might_need_web_search(text: str) -> bool:
    """Light heuristic: does this query likely need real-time info?"""
    indicators = [
        "오늘", "현재", "최신", "최근", "실시간", "지금", "올해",
        "날씨", "뉴스", "환율", "주가", "시세", "속보",
        "today", "current", "latest", "recent", "now", "weather",
        "news", "stock", "price", "score", "live",
        "검색", "search", "look up", "find out",
    ]
    lower = text.lower()
    return any(kw in lower for kw in indicators)


def _needs_full_document_context(text: str) -> bool:
    """
    Detect requests that need the whole uploaded document, not just top-k chunks.

    This is used only for file-scoped chat after upload.
    """
    lower = text.lower().strip()
    indicators = [
        "요약", "요약해", "정리", "정리해", "전체", "전반", "구조", "목차",
        "분석", "분석해", "핵심", "주요 내용", "전체 내용", "섹션", "section",
        "structure", "outline", "overview", "summarize", "summary",
        "analyze", "analysis", "key points", "main points", "extract key",
        "extract data", "important information", "important info",
    ]
    return any(keyword in lower for keyword in indicators)


def _is_smalltalk_query(text: str) -> bool:
    """Detect greetings/acknowledgements that should bypass document scoping."""
    lower = text.lower().strip()
    indicators = [
        "안녕", "고마워", "감사", "오케이", "알겠", "응", "네",
        "hello", "hi", "thanks", "thank you", "okay", "ok", "got it",
    ]
    return any(keyword in lower for keyword in indicators)


def _is_document_intent_query(text: str) -> bool:
    """Heuristic: whether the user is likely asking about uploaded documents."""
    lower = (text or "").lower().strip()
    indicators = [
        "문서", "파일", "첨부", "업로드", "pdf", "페이지", "쪽", "본문", "원문",
        "가사", "근거", "출처", "해당 문서", "이 문서", "이 파일", "그 문서", "그 파일",
        "document", "file", "attachment", "uploaded", "upload", "pdf", "page",
        "section", "paragraph", "table", "figure", "source", "quote", "lyrics",
    ]
    followup_hints = [
        "그 부분", "위 내용", "방금 내용", "그거", "거기", "다시", "이어", "계속",
    ]

    if any(keyword in lower for keyword in indicators):
        return True

    # Very short follow-up utterances are often document continuation questions.
    if len(lower) <= 20 and any(hint in lower for hint in followup_hints):
        return True

    return False


def _is_scope_followup_query(text: str) -> bool:
    """Heuristic: follow-up utterance likely referring to previously scoped document chat."""
    lower = (text or "").lower().strip()
    if not lower:
        return False

    hints = [
        "그 부분", "그 내용", "위 내용", "방금", "앞에서", "이어서", "계속", "다시",
        "그거", "거기", "그 다음", "다음 조항", "다음 항목", "해당 내용", "그 문장",
        "이 부분", "이 내용", "이 조항", "몇 페이지", "어느 페이지",
    ]
    if any(hint in lower for hint in hints):
        return True

    # Short continuation-style queries are often follow-ups.
    if len(lower) <= 40 and any(token in lower for token in ["다시", "이어", "계속", "그거", "거기", "해당"]):
        return True

    return False


def _is_scope_reset_query(text: str) -> bool:
    """Detect explicit intent to stop document-scoped conversation."""
    lower = (text or "").lower().strip()
    indicators = [
        "문서 말고", "파일 말고", "업로드 말고", "일반 대화", "새 주제", "주제 바꿔",
        "이제 다른 질문", "그거 말고", "문서와 상관없이",
    ]
    return any(keyword in lower for keyword in indicators)


def _document_not_found_answer(
    user_message: str,
    active_source: Optional[str],
    active_doc_id: Optional[str] = None,
) -> str:
    """Return a grounded fallback when scoped retrieval confidence is too low."""
    source_name = active_source or active_doc_id or "the uploaded document"
    if re.search(r"[가-힣]", user_message):
        return (
            f"업로드된 문서 '{source_name}'에서 질문과 관련된 정보를 찾지 못했습니다. "
            "질문을 조금 더 구체적으로 해주세요."
        )
    return (
        f"I couldn't find relevant information in the uploaded document '{source_name}'. "
        "Please try a more specific question."
    )


def _is_direct_extraction_query(text: str) -> bool:
    """Detect requests that want raw OCR/text output from the uploaded file."""
    lower = text.lower().strip()
    indicators = [
        "텍스트 추출", "문자 추출", "글자 추출", "읽어줘", "텍스트만", "원문", "ocr",
        "모든 텍스트", "전체 텍스트", "텍스트 다", "다 뽑", "전문",
        "what does this image say", "what does the image say",
        "give me the text", "extract the text", "read the text",
        "read this image", "text in the image", "transcribe",
    ]
    return any(keyword in lower for keyword in indicators)


def _normalize_for_match(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^0-9a-z가-힣\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _list_uploaded_sources(owner_id: Optional[str] = None) -> List[str]:
    """Return unique uploaded source names from lightweight document registry."""
    try:
        seen = set()
        sources = []
        for doc in list_documents(owner_id=owner_id, source_type="upload"):
            if not doc or doc.get("source_type") != "upload":
                continue
            source = str(doc.get("source") or "").strip()
            if not source:
                continue
            key = source.lower()
            if key in seen:
                continue
            seen.add(key)
            sources.append(source)
        return sorted(sources)
    except Exception as e:
        logger.debug("Failed to list uploaded sources: %s", e)
        return []


def _infer_upload_source_from_query_or_history(user_message: str, history: List[Message], owner_id: Optional[str] = None) -> Optional[str]:
    """Best-effort: infer which uploaded file the user is referring to."""
    sources = _list_uploaded_sources(owner_id=owner_id)
    if not sources:
        return None

    msg_norm = _normalize_for_match(user_message)
    user_hist = [m.content for m in (history or []) if getattr(m, "role", "") == "user"]
    recent_hist = user_hist[-8:]

    best_source = None
    best_score = 0

    for source in sources:
        stem = Path(source).stem
        stem_norm = _normalize_for_match(stem)
        if not stem_norm:
            continue

        score = 0
        # Strong signal: full stem appears in current query.
        if stem_norm in msg_norm:
            score += 120

        # Token overlap signal.
        tokens = [tok for tok in stem_norm.split(" ") if len(tok) >= 2]
        token_hits = sum(1 for tok in tokens if tok in msg_norm)
        if token_hits >= 2:
            score += 50
        elif token_hits == 1 and any(len(tok) >= 4 and tok in msg_norm for tok in tokens):
            score += 20

        # Conversation carry-over: recently mentioned file in user turns.
        for idx, text in enumerate(reversed(recent_hist), start=1):
            hist_norm = _normalize_for_match(text)
            if stem_norm and stem_norm in hist_norm:
                score += max(0, 40 - (idx * 5))
                break

        if score > best_score:
            best_source = source
            best_score = score

    if best_score >= 35:
        logger.info("Inferred upload source '%s' (score=%d) from query/history", best_source, best_score)
        return best_source

    return None


def _direct_extraction_ambiguous_answer(upload_sources: List[str]) -> str:
    preview = ", ".join(upload_sources[:3])
    suffix = f" 외 {len(upload_sources) - 3}개" if len(upload_sources) > 3 else ""
    return (
        "업로드 파일이 여러 개라 어떤 파일에서 텍스트를 뽑아야 할지 애매합니다. "
        f"파일명을 같이 말해 주세요. (예: {preview}{suffix})"
    )


def _extract_text_from_upload_source(source: Optional[str], user_id: Optional[str] = None, source_path: Optional[str] = None) -> str:
    """Extract fresh text from uploaded source file to avoid stale/noisy chunk artifacts."""
    if not source:
        return ""
    try:
        candidate_paths: List[Path] = []

        if source_path:
            candidate_paths.append(Path(source_path))

        safe_name = Path(source).name
        if user_id:
            candidate_paths.append(UPLOADS_DIR / Path(user_id).name / safe_name)
        candidate_paths.append(UPLOADS_DIR / safe_name)

        for candidate in candidate_paths:
            if candidate.exists() and candidate.is_file():
                return (extract_full_text(str(candidate)) or "").strip()
        return ""
    except Exception as e:
        logger.debug("Direct source extraction failed for %s: %s", source, e)
        return ""


def _strip_enrichment_header(text: str) -> str:
    """Remove enrichment header prepended before embedding/retrieval."""
    return re.sub(r'^\[Document:.*?\]\n', '', text or '', flags=re.DOTALL).strip()


def _build_direct_extraction_answer(active_source: Optional[str], docs) -> str:
    """Return extracted document text directly for OCR/transcription requests."""
    extracted = "\n\n".join(
        _strip_enrichment_header(doc.page_content)
        for doc in docs
        if _strip_enrichment_header(doc.page_content)
    ).strip()

    if not extracted:
        fallback_source = active_source or (docs[0].metadata.get("source") if docs else None)
        fallback_doc_id = docs[0].metadata.get("doc_id") if docs else None
        return _document_not_found_answer("extract text", fallback_source, fallback_doc_id)

    if re.search(r"[가-힣]", extracted):
        return f"추출된 텍스트:\n\n{extracted}"
    return f"Extracted text:\n\n{extracted}"


def _scoped_confidence_threshold(docs) -> float:
    """
    Use a slightly lower threshold for tiny uploaded docs where one chunk is
    effectively the whole document (e.g. a screenshot or one-page upload).
    """
    if len(docs) != 1:
        return DOCUMENT_CONFIDENCE_THRESHOLD

    meta = docs[0].metadata
    if meta.get("source_type") == "upload":
        return SCOPED_SINGLE_CHUNK_CONFIDENCE_THRESHOLD

    return DOCUMENT_CONFIDENCE_THRESHOLD


def _should_force_small_doc_full_context(
    active_source: Optional[str],
    active_doc_id: Optional[str],
    owner_id: Optional[str] = None,
) -> bool:
    """Use full-document context for tiny scoped docs where top-k retrieval is brittle."""
    if not active_source and not active_doc_id:
        return False

    try:
        chunk_count = get_document_chunk_count(source=active_source, doc_id=active_doc_id, owner_id=owner_id)
    except Exception as e:
        logger.debug("Could not inspect scoped document chunk count: %s", e)
        return False

    if chunk_count <= 0:
        return False

    return chunk_count <= SCOPED_SMALL_DOC_FULL_CONTEXT_MAX_CHUNKS


# ═══════════════════════════════════════════════════════════════════════
# Prompt Builder
# ═══════════════════════════════════════════════════════════════════════

def _format_history(history: List[Message], max_turns: int = 8) -> str:
    if not history:
        return ""
    return "\n\n".join(
        f"[{msg.role}]\n{msg.content}" for msg in history[-max_turns:]
    )


_SYSTEM_PROMPT = """You are Tilon AI, a helpful document-based chatbot.

CRITICAL RULES:
1. Respond in the SAME language the user is using. Korean → Korean. English → English. NEVER output Chinese characters (中文/汉字).
2. If document context is provided and relevant to the question, answer based on that context and cite the source (document name, page number).
3. If document context is provided but NOT relevant to the question, ignore it and answer normally.
4. If no document context is available, answer from your general knowledge.
5. If web search results are provided, use them for current/real-time information.
6. If you don't know, say so honestly. Never make up information.
7. Be concise and direct. Answer the question first, then explain if needed.
8. When citing documents, mention the source naturally (e.g., "문서 3페이지에 따르면..." or "According to page 3...").
9. If retrieved text includes Chinese characters, translate/paraphrase them into Korean or the user's language instead of copying Chinese characters.
10. Do NOT use markdown emphasis symbols in the final answer (forbidden: **, __). Output plain text only."""


def _build_prompt(
    user_message: str,
    history: List[Message],
    doc_context: str = "",
    web_context: str = "",
    system_prompt: str = "",
) -> str:
    """Build a single unified prompt with all available context."""
    parts = [f"[System]\n{system_prompt or _SYSTEM_PROMPT}"]

    history_text = _format_history(history)
    if history_text:
        parts.append(f"[Conversation history]\n{history_text}")

    if doc_context:
        parts.append(f"[Retrieved document context]\n{doc_context}")

    if web_context:
        parts.append(f"[Web search results]\n{web_context}")

    parts.append(f"[User message]\n{user_message}")

    return "\n\n".join(parts)


def _contains_chinese_chars(text: str) -> bool:
    """Return True when any CJK Han character exists."""
    return bool(_CJK_HAN_RE.search(text or ""))


def _contains_cjk_punctuation(text: str) -> bool:
    """Return True when CJK full-width punctuation appears."""
    return bool(_CJK_PUNCT_RE.search(text or ""))


def _normalize_cjk_punctuation(text: str) -> str:
    """Normalize CJK punctuation into ASCII punctuation for clean Korean output."""
    if not text:
        return text
    table = str.maketrans({
        "，": ",", "。": ".", "！": "!", "？": "?", "；": ";", "：": ":",
        "﹐": ",", "﹒": ".", "﹔": ";", "﹕": ":",
        "（": "(", "）": ")", "［": "[", "］": "]",
    })
    normalized = text.translate(table)
    normalized = re.sub(r"[「」『』【】《》〈〉〔〕]", "", normalized)
    return normalized


def _looks_garbled_output(answer: str) -> bool:
    """Detect noisy/degenerated outputs (punctuation fragments, broken lines)."""
    text = (answer or "").strip()
    if not text:
        return False

    cjk_punct_count = len(_CJK_PUNCT_RE.findall(text))
    if cjk_punct_count >= 3:
        return True

    if re.search(r"[，。！？；：]\s*[，。！？；：]", text):
        return True

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    noisy_lines = 0
    for ln in lines:
        # Lines with mostly punctuation and almost no meaningful letters/numbers.
        meaningful = len(re.findall(r"[가-힣A-Za-z0-9]", ln))
        has_cjk_punct = bool(_CJK_PUNCT_RE.search(ln))
        if has_cjk_punct and meaningful <= 1:
            noisy_lines += 1

        # e.g., "- 。：，。" style broken list lines
        if re.fullmatch(r"[-•\s，。！？；：,.;:!?]+", ln) and len(ln) >= 2:
            noisy_lines += 1

    return noisy_lines >= 2


def _strip_markdown_emphasis(text: str) -> str:
    """Remove markdown emphasis/heading markers from model output."""
    if not text:
        return text
    cleaned = text.replace("**", "")
    cleaned = cleaned.replace("__", "")
    cleaned = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", cleaned)
    return cleaned


def _expects_korean_output(user_message: str) -> bool:
    """Treat messages containing Hangul as Korean-mode conversations."""
    return bool(_HANGUL_RE.search(user_message or ""))


def _needs_language_rewrite(user_message: str, answer: str) -> bool:
    """
    Rewrite when:
    1) Chinese chars/punctuation exist, or
    2) User wrote in Korean but answer is effectively non-Korean, or
    3) Output looks garbled/noisy.
    """
    if not answer:
        return False

    if _contains_chinese_chars(answer) or _contains_cjk_punctuation(answer):
        return True

    if _looks_garbled_output(answer):
        return True

    if _expects_korean_output(user_message):
        hangul_count = len(_HANGUL_RE.findall(answer))
        latin_count = len(re.findall(r"[A-Za-z]", answer))
        # If there is no Hangul and enough Latin text, force Korean rewrite.
        if hangul_count == 0 and latin_count >= 5:
            return True

    return False


def _rewrite_answer_to_korean(answer: str, user_message: str, model: str) -> str:
    """One-shot rewrite pass to remove Chinese and align language with the user."""
    rewrite_prompt = f"""[System]
You are a strict editor.
Respond in the SAME language as the user message.
If user message is Korean, output must include natural Korean sentences (Hangul).
ABSOLUTE RULE: Do not output Chinese characters (中文/汉字) at all.
If Chinese text exists in the draft, translate/paraphrase it into Korean or the user language.
Preserve facts, order, and citations from the draft answer.
Do not add new claims.

[User message]
{user_message}

[Draft answer]
{answer}

[Task]
Rewrite the draft to match the user language and remove Chinese characters.
""".strip()

    rewritten = call_ollama(
        rewrite_prompt,
        model=model,
        temperature=0.0,
    )
    return get_response_text(rewritten)


def _translate_to_korean_fallback(answer: str, model: str) -> str:
    """Last-resort translation to Korean when Korean mode is required."""
    prompt = f"""[System]
Translate the draft answer into natural Korean.
Do not output Chinese characters (中文/汉字).
Preserve meaning and important details.

[Draft answer]
{answer}
""".strip()
    rewritten = call_ollama(prompt, model=model, temperature=0.0)
    return get_response_text(rewritten)


def _apply_language_guard(user_message: str, answer: str, model: str) -> str:
    """
    Hard guard:
    - Never allow Chinese characters.
    - Korean user message -> enforce Korean output.
    """
    expect_korean = _expects_korean_output(user_message)

    if not _needs_language_rewrite(user_message, answer):
        return answer

    candidate = answer
    try:
        for _ in range(3):
            rewritten = _rewrite_answer_to_korean(candidate, user_message, model)
            if not rewritten:
                break
            candidate = rewritten

            no_chinese = not _contains_chinese_chars(candidate)
            no_cjk_punct = not _contains_cjk_punctuation(candidate)
            has_korean = bool(_HANGUL_RE.search(candidate))
            not_garbled = not _looks_garbled_output(candidate)
            if no_chinese and no_cjk_punct and not_garbled and (not expect_korean or has_korean):
                logger.info("Applied strict language guard.")
                return _strip_markdown_emphasis(candidate).strip()
    except Exception as e:
        logger.warning("Strict language guard rewrite failed: %s", e)

    # Remove any remaining Chinese chars first and normalize punctuation noise.
    stripped = _CJK_HAN_RE.sub("", candidate or "")
    stripped = _normalize_cjk_punctuation(stripped)
    stripped = re.sub(r"[ 	]{2,}", " ", stripped)
    stripped = re.sub(r"\n{3,}", "\n\n", stripped).strip()

    # If still garbled, ask for one more clean rewrite in user language.
    if _looks_garbled_output(stripped):
        try:
            cleaned = _rewrite_answer_to_korean(stripped, user_message, model)
            if cleaned:
                cleaned = _normalize_cjk_punctuation(cleaned)
                cleaned = _strip_markdown_emphasis(cleaned).strip()
                if cleaned and not _contains_chinese_chars(cleaned) and not _looks_garbled_output(cleaned):
                    logger.warning("Applied garbled-output cleanup rewrite in language guard.")
                    return cleaned
        except Exception as e:
            logger.warning("Garbled-output cleanup rewrite failed: %s", e)

    # If Korean is expected but still absent, force one translation pass.
    if expect_korean and not _HANGUL_RE.search(stripped):
        try:
            translated = _translate_to_korean_fallback(stripped or candidate, model)
            if translated:
                translated = _normalize_cjk_punctuation(translated)
            if translated and _HANGUL_RE.search(translated) and not _contains_chinese_chars(translated):
                translated = _strip_markdown_emphasis(translated).strip()
                if not _contains_cjk_punctuation(translated) and not _looks_garbled_output(translated):
                    logger.warning("Applied Korean fallback translation in language guard.")
                    return translated
        except Exception as e:
            logger.warning("Korean fallback translation failed: %s", e)

        return "한국어로 답변하도록 재시도했지만 변환에 실패했습니다. 같은 질문을 다시 입력해 주세요."

    if stripped:
        logger.warning("Applied hard-strip fallback in language guard.")
        return stripped

    return "언어 정책에 맞는 응답 생성에 실패했습니다. 같은 질문을 다시 시도해 주세요."


# ═══════════════════════════════════════════════════════════════════════

def _has_token_repetition_loop(answer: str) -> bool:
    """
    Detect degenerate repetitive answers like:
    "대학원생 대학원생 대학원생 ..."
    """
    tokens = [t for t in re.split(r"\s+", (answer or "").strip()) if t]
    if len(tokens) < 20:
        return False

    max_run = 1
    current_run = 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1

    if max_run >= 8:
        return True

    dominant_token, dominant_count = Counter(tokens).most_common(1)[0]
    if len(dominant_token) >= 2 and dominant_count >= max(12, int(len(tokens) * 0.35)):
        return True

    return False


def _rewrite_repetitive_answer(answer: str, user_message: str, model: str) -> str:
    """Rewrite one repetitive answer into a concise non-redundant form."""
    rewrite_prompt = f"""[System]
You are a careful editor.
Respond in the SAME language as the user message.
Remove repetitive loops and redundant phrases.
Do not add new facts.
Preserve important points and source mentions from the draft.

[User message]
{user_message}

[Draft answer]
{answer}

[Task]
Rewrite the draft so it is concise, readable, and non-repetitive.
""".strip()

    rewritten = call_ollama(
        rewrite_prompt,
        model=model,
        temperature=0.0,
    )
    return get_response_text(rewritten)


def _apply_repetition_guard(user_message: str, answer: str, model: str) -> str:
    """Post-process repetitive outputs with one rewrite pass."""
    if not _has_token_repetition_loop(answer):
        return answer

    try:
        rewritten = _rewrite_repetitive_answer(answer, user_message, model)
        rewritten = _apply_language_guard(user_message, rewritten, model)
        if rewritten:
            logger.info("Applied repetition guard (rewrote degenerate repetitive answer).")
            return rewritten
    except Exception as e:
        logger.warning("Repetition guard rewrite failed: %s", e)

    return answer
# Unified Chat Handler
# ═══════════════════════════════════════════════════════════════════════

def handle_chat(
    user_message: str,
    history: List[Message] = None,
    model: str = None,
    active_source: str = None,
    active_doc_id: str = None,
    active_source_type: str = None,
    system_prompt: str = None,
    web_search_enabled: bool = True,
    user_id: str = None,
) -> Dict[str, Any]:
    """
    Unified chat handler — works like a normal chatbot.

    Args:
        user_message: What the user said
        history: Conversation history
        model: Which Ollama model to use
        active_source: Active document filename scope for the current chat, if any
        active_doc_id: Stable document ID scope for the current chat, if any
        active_source_type: Source-type scope for the current chat (e.g., 'upload'), if any
        system_prompt: Override default system prompt
        web_search_enabled: Enable/disable Tavily web search
    """
    history = history or []
    selected_model = model or OLLAMA_MODEL
    scoped_source = None if _is_smalltalk_query(user_message) else active_source
    scoped_doc_id = None if _is_smalltalk_query(user_message) else active_doc_id
    scoped_source_type = None if _is_smalltalk_query(user_message) else active_source_type
    has_initial_scope = bool(scoped_source or scoped_doc_id or scoped_source_type)

    scope_reset_requested = _is_scope_reset_query(user_message)
    document_intent = (
        _is_document_intent_query(user_message)
        or _is_direct_extraction_query(user_message)
        or _needs_full_document_context(user_message)
        or (has_initial_scope and _is_scope_followup_query(user_message))
        or (has_initial_scope and not scope_reset_requested)
    )

    if scope_reset_requested:
        document_intent = False

    # Keep explicit chat-level document scope unless the user clearly resets it.
    if not document_intent:
        scoped_source = None
        scoped_doc_id = None
        scoped_source_type = None

    if not (scoped_source or scoped_doc_id) and scoped_source_type == "upload":
        inferred_source = _infer_upload_source_from_query_or_history(user_message, history, owner_id=user_id)
        if inferred_source:
            scoped_source = inferred_source

    if _is_direct_extraction_query(user_message) and not (scoped_source or scoped_doc_id) and scoped_source_type == "upload":
        upload_sources = _list_uploaded_sources(owner_id=user_id)
        return {
            "answer": _direct_extraction_ambiguous_answer(upload_sources),
            "sources": [],
            "mode": "ocr_extract",
            "active_source": active_source,
            "active_doc_id": active_doc_id,
        }

    if (scoped_source or scoped_doc_id) and _is_direct_extraction_query(user_message):
        docs = get_documents_by_source(source=scoped_source, doc_id=scoped_doc_id, owner_id=user_id)
        if docs:
            logger.info(
                "Direct extraction response for '%s'%s",
                scoped_source or "scoped document",
                f" ({scoped_doc_id})" if scoped_doc_id else "",
            )

            fresh_text = _extract_text_from_upload_source(scoped_source, user_id=user_id, source_path=(docs[0].metadata.get("source_path") if docs else None))
            if fresh_text:
                if re.search(r"[가-힣]", fresh_text):
                    extraction_answer = f"추출된 텍스트:\n\n{fresh_text}"
                else:
                    extraction_answer = f"Extracted text:\n\n{fresh_text}"
            else:
                extraction_answer = _build_direct_extraction_answer(scoped_source, docs)

            extraction_answer = _apply_language_guard(user_message, extraction_answer, selected_model)
            extraction_answer = _strip_markdown_emphasis(extraction_answer)
            return {
                "answer": extraction_answer,
                "sources": extract_sources(docs),
                "mode": "ocr_extract",
                "active_source": scoped_source,
                "active_doc_id": scoped_doc_id,
            }

    # ── Step 1: Always search for relevant document context ──
    doc_context = ""
    sources = []

    use_full_document = bool(
        (scoped_source or scoped_doc_id)
        and (
            _needs_full_document_context(user_message)
            or _should_force_small_doc_full_context(scoped_source, scoped_doc_id)
        )
    )
    retrieval = retrieve(
        user_message,
        source_filter=scoped_source,
        doc_id_filter=scoped_doc_id,
        source_type_filter=scoped_source_type,
        owner_id_filter=user_id,
        full_document=use_full_document,
    )
    docs = retrieval.docs

    has_explicit_scope = bool(scoped_source or scoped_doc_id)
    has_source_type_scope = bool(scoped_source_type)
    has_any_scope = has_explicit_scope or has_source_type_scope

    if has_any_scope and not use_full_document:
        threshold = _scoped_confidence_threshold(docs) if docs else DOCUMENT_CONFIDENCE_THRESHOLD
        low_confidence = (not docs) or (
            retrieval.confidence < threshold
            and not retrieval.strong_keyword_hit
        )
        if low_confidence:
            logger.info(
                "Low-confidence scoped retrieval for '%s'%s (confidence=%.2f, threshold=%.2f, keyword_hit=%s)",
                active_source or active_source_type or "scoped document",
                f" ({scoped_doc_id})" if scoped_doc_id else "",
                retrieval.confidence,
                threshold,
                retrieval.strong_keyword_hit,
            )

            # Explicit per-file scope should remain strict.
            if has_explicit_scope:
                return {
                    "answer": _document_not_found_answer(
                        user_message,
                        active_source or ("uploaded files" if active_source_type == "upload" else None),
                        active_doc_id,
                    ),
                    "sources": [],
                    "mode": "document_qa",
                    "active_source": active_source,
                    "active_doc_id": active_doc_id,
                }

            # Source-type-only scope (e.g. all uploads) should stay flexible.
            # Do not hard-stop here; allow history-aware/general continuation.
            logger.info(
                "Low-confidence upload-scoped retrieval; continuing with history-aware general response path."
            )

    if docs:
        doc_context = format_context(docs)
        sources = extract_sources(docs)
        if use_full_document:
            logger.info(
                "Loaded full document context: %d chunks from '%s'",
                len(docs),
                scoped_source,
            )
        else:
            logger.info(
                "Found %d relevant chunks%s (confidence=%.2f, keyword_hit=%s)",
                len(docs),
                (
                    " (scoped to "
                    + ", ".join(
                        part
                        for part in [
                            f"source='{scoped_source}'" if scoped_source else "",
                            f"doc_id='{scoped_doc_id}'" if scoped_doc_id else "",
                            f"source_type='{scoped_source_type}'" if scoped_source_type else "",
                        ]
                        if part
                    )
                    + ")"
                ) if (scoped_source or scoped_doc_id or scoped_source_type) else "",
                retrieval.confidence,
                retrieval.strong_keyword_hit,
            )
    else:
        logger.info("No relevant document chunks found")

    # ── Step 2: Optional web search (toggle-controlled) ──
    web_context = ""
    allow_web_search = web_search_enabled and not (scoped_source or scoped_doc_id)
    if scoped_source_type and _is_document_intent_query(user_message):
        allow_web_search = False

    if allow_web_search:
        if _might_need_web_search(user_message):
            web_results = _search_web(user_message)
            if web_results:
                web_context = web_results
                logger.info("Added web search results (toggle enabled)")
            else:
                logger.info("Web search requested but no results were returned.")
        else:
            logger.debug("Web search skipped by heuristic (query appears local/static).")
    # ── Step 3: Build prompt with all context and let LLM decide ──
    prompt = _build_prompt(
        user_message=user_message,
        history=history,
        doc_context=doc_context,
        web_context=web_context,
        system_prompt=system_prompt,
    )

    result = call_ollama(prompt, model=selected_model)
    answer = get_response_text(result)
    answer = _apply_language_guard(user_message, answer, selected_model)
    answer = _apply_repetition_guard(user_message, answer, selected_model)
    answer = _strip_markdown_emphasis(answer)

    # Determine what was used (for UI display)
    mode = "general"
    if doc_context and sources:
        mode = "document_qa"
    if web_context:
        mode = "web_search" if not doc_context else "document_qa+web"

    resolved_active_source = None
    resolved_active_doc_id = None
    if document_intent:
        resolved_active_source = scoped_source or active_source
        resolved_active_doc_id = scoped_doc_id or active_doc_id
        if (not resolved_active_source or not resolved_active_doc_id) and sources:
            first_source = sources[0] or {}
            resolved_active_source = resolved_active_source or first_source.get("source")
            resolved_active_doc_id = resolved_active_doc_id or first_source.get("doc_id")

    return {
        "answer": answer,
        "sources": sources,
        "mode": mode,
        "active_source": resolved_active_source,
        "active_doc_id": resolved_active_doc_id,
    }
