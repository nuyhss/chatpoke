"""
Prompt templates for each chat mode.

FIX: All prompts now match the user's language instead of hardcoding Korean.
The LLM (qwen2.5) is Chinese-origin and defaults to Chinese when confused.
Explicit language matching + "NEVER respond in Chinese" prevents this.
"""

from typing import List
from app.models.schemas import Message


def format_history(history: List[Message], max_turns: int = 8) -> str:
    if not history:
        return ""
    lines = []
    for msg in history[-max_turns:]:
        lines.append(f"[{msg.role}]\n{msg.content}")
    return "\n\n".join(lines)


# Language instruction that goes into every prompt
_LANG_RULE = (
    "CRITICAL: Respond in the SAME language the user is using. "
    "If the user writes in Korean, respond in Korean. "
    "If the user writes in English, respond in English. "
    "NEVER respond in Chinese (中文) under any circumstances."
)


# ═══════════════════════════════════════════════════════════════════════
# GENERAL CHAT
# ═══════════════════════════════════════════════════════════════════════

def build_general_prompt(
    system_prompt: str,
    history: List[Message],
    user_message: str,
) -> str:
    return f"""[시스템 지침]
{system_prompt}

답변 규칙:
1. {_LANG_RULE}
2. 사용자의 입력이 짧은 인사말이나 단순 표현이면 짧고 자연스럽게만 답한다.
3. 이전 문맥을 억지로 이어붙이지 않는다.
4. 모르면 모른다고 답한다.

[대화 이력]
{format_history(history)}

[사용자 질문]
{user_message}""".strip()


# ═══════════════════════════════════════════════════════════════════════
# DOCUMENT QA  ← This is what the fine-tuned model will receive
# ═══════════════════════════════════════════════════════════════════════

def build_document_prompt(
    system_prompt: str,
    history: List[Message],
    user_message: str,
    context: str,
) -> str:
    """
    Context format from retriever.py:format_context():
    [Doc: filename.pdf | Page: 3 | Section: title | Lang: ko]
    <chunk text here>
    """
    return f"""[시스템 지침]
{system_prompt}

답변 규칙:
1. {_LANG_RULE}
2. 제공된 문서 문맥만 근거로 답한다.
3. 문서에 없는 내용은 추측하지 않는다.
4. 문맥이 부족하면 "해당 내용은 제공된 문서에서 확인되지 않습니다."라고 답한다.
5. 핵심 답변을 먼저 말한다.
6. 가능하면 페이지와 문서를 근거로 설명한다.
7. 이미지에서 추출된 텍스트가 제공되면 해당 텍스트를 기반으로 답한다.

[대화 이력]
{format_history(history)}

[검색된 문서]
{context if context else "검색된 관련 문서 없음"}

[사용자 질문]
{user_message}

[답변 형식]
- 핵심 답변:
- 근거 요약:
- 참고 문서:""".strip()


# ═══════════════════════════════════════════════════════════════════════
# WEB SEARCH
# ═══════════════════════════════════════════════════════════════════════

def build_web_prompt(
    system_prompt: str,
    history: List[Message],
    user_message: str,
    search_results: str = "",
) -> str:
    search_section = ""
    if search_results:
        search_section = f"\n[웹 검색 결과]\n{search_results}\n"

    return f"""[시스템 지침]
{system_prompt}

답변 규칙:
1. {_LANG_RULE}
2. 최신 정보가 필요한 질문이다.
3. 웹 검색 결과가 제공되면 해당 정보를 우선 참고한다.
4. 최신 정보가 불확실하면 불확실하다고 말한다.
5. 핵심만 짧고 명확하게 정리한다.
{search_section}
[대화 이력]
{format_history(history)}

[사용자 질문]
{user_message}""".strip()