import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from app.api.openai_compat import chat_completions
from app.api.routes import chat
from app.chat.handlers import _format_history, _strip_question_echo, handle_chat
from app.models.schemas import ChatRequest, Message, OpenAIChatRequest, OpenAIMessage


class ChatRegressionTests(unittest.TestCase):
    @patch("app.api.routes.handle_chat")
    def test_chat_response_includes_response_metadata(self, mock_handle_chat):
        mock_handle_chat.return_value = {
            "answer": "테스트 응답",
            "sources": [],
            "mode": "general",
            "active_source": None,
            "active_doc_id": None,
            "conversation_state": {
                "current_topic": "제칠일안식일예수재림교회",
                "aliases": ["재림교회", "이 교회"],
                "last_user_intent": "역사 질문",
                "last_resolved_query": "제칠일안식일예수재림교회의 역사 알려줘",
                "topic_type": "organization",
            },
            "response_metadata": {
                "request_id": "ollama-test123",
                "finish_reason": "length",
                "usage": {
                    "prompt_tokens": 111,
                    "completion_tokens": 222,
                    "total_tokens": 333,
                },
                "prompt_capture": "[System]\n테스트 프롬프트",
                "response_capture": "원문 응답",
                "resolved_query": "제칠일안식일예수재림교회의 역사 알려줘",
            },
        }

        response = chat(
            ChatRequest(
                message="이 교회의 역사 알려줘",
                history=[Message(role="user", content="재림교회가 뭐야?")],
            )
        )

        payload = response.model_dump()
        self.assertEqual(payload["response_metadata"]["request_id"], "ollama-test123")
        self.assertEqual(payload["response_metadata"]["finish_reason"], "length")
        self.assertEqual(payload["response_metadata"]["usage"]["total_tokens"], 333)
        self.assertEqual(
            payload["conversation_state"]["last_resolved_query"],
            "제칠일안식일예수재림교회의 역사 알려줘",
        )

    def test_openai_stream_is_rejected_explicitly(self):
        with self.assertRaises(HTTPException) as ctx:
            chat_completions(
                OpenAIChatRequest(
                    model="qwen2.5:7b",
                    stream=True,
                    messages=[OpenAIMessage(role="user", content="안녕")],
                )
            )

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("stream=true is not supported", ctx.exception.detail)

    @patch("app.chat.handlers.call_ollama")
    @patch("app.chat.handlers.retrieve")
    def test_handle_chat_rewrites_followup_query_and_captures_metadata(self, mock_retrieve, mock_call_ollama):
        mock_retrieve.return_value = SimpleNamespace(
            docs=[],
            confidence=0.0,
            strong_keyword_hit=False,
        )
        mock_call_ollama.return_value = {
            "response": "재림교회의 역사는 19세기 미국에서 시작되었습니다.",
            "request_id": "ollama-req123",
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": 21,
                "completion_tokens": 34,
                "total_tokens": 55,
            },
        }

        result = handle_chat(
            user_message="이 교회의 역사 알려줘",
            history=[Message(role="user", content="재림교회가 뭐야?")],
            conversation_state={
                "current_topic": "제칠일안식일예수재림교회",
                "aliases": ["재림교회", "제칠일안식일예수재림교회", "이 교회", "그 교단"],
                "last_user_intent": "일반 질문",
            },
        )

        self.assertEqual(
            mock_retrieve.call_args.args[0],
            "제칠일안식일예수재림교회의 역사 알려줘",
        )
        self.assertEqual(result["response_metadata"]["request_id"], "ollama-req123")
        self.assertEqual(result["response_metadata"]["resolved_query"], "제칠일안식일예수재림교회의 역사 알려줘")
        self.assertIn("[Resolved user query]", result["response_metadata"]["prompt_capture"])
        self.assertEqual(
            result["response_metadata"]["response_capture"],
            "재림교회의 역사는 19세기 미국에서 시작되었습니다.",
        )

    def test_format_history_excludes_error_and_stopped_roles(self):
        history = [
            Message(role="user", content="첫 질문"),
            Message(role="error", content="서버 오류"),
            Message(role="assistant", content="첫 답변"),
            Message(role="stopped", content="응답 중지"),
        ]

        rendered = _format_history(history)

        self.assertIn("[user]\n첫 질문", rendered)
        self.assertIn("[assistant]\n첫 답변", rendered)
        self.assertNotIn("[error]", rendered)
        self.assertNotIn("[stopped]", rendered)

    def test_strip_question_echo_removes_leading_repeated_question(self):
        cleaned = _strip_question_echo(
            "이 교회는 이단이야?",
            "이 교회는 이단이야?\n\n재림교회를 이단으로 단정하는 것은 부정확합니다.",
        )

        self.assertEqual(
            cleaned,
            "재림교회를 이단으로 단정하는 것은 부정확합니다.",
        )


if __name__ == "__main__":
    unittest.main()
