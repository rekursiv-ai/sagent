"""Tests for sagent.providers.google."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

from PIL import Image

import pytest

from sagent.custom_types import (
    BytesMessage,
    JsonMessage,
    Message,
    ModelRequest,
    ModelResponse,
    MultipartMessage,
    Provider,
    TextMessage,
)
from sagent.lib.json import json_freeze
from sagent.lib.message import get_directive, get_tool_name
from sagent.providers.google import (
    Google,
    _build_request,
    _parse_response,
)
from sagent.providers.lib.cost import Pricing


def _user(text: str) -> Message:
    return TextMessage(text, "text/x-user-message")


def _text(resp: ModelResponse) -> str | None:
    if not isinstance(resp.content, MultipartMessage):
        return None
    for p in resp.content.content:
        if p.descriptor == "text/plain" and isinstance(p, TextMessage):
            return p.content
    return None


def _tool_calls(resp: ModelResponse) -> list[Message]:
    if not isinstance(resp.content, MultipartMessage):
        return []
    return [p for p in resp.content.content if p.descriptor == "multipart/x-tool-call"]


def _stream_client(lines: list[str]) -> AsyncMock:
    """AsyncClient mock whose `stream()` yields SSE lines."""

    async def _aiter() -> Any:
        for ln in lines:
            yield ln

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.aiter_lines = _aiter
    stream_cm = AsyncMock()
    stream_cm.__aenter__ = AsyncMock(return_value=mock_resp)
    stream_cm.__aexit__ = AsyncMock(return_value=False)
    mock_client = AsyncMock()
    mock_client.stream = MagicMock(return_value=stream_cm)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


# -- Provider ----------------------------------------------------------


class TestProvider:
    def test_from_key(self) -> None:
        p = Google.from_key("AIza-test")
        assert p.api_key == "AIza-test"

    def test_from_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("GOOGLE_API_KEY", "AIza-env")
        p = Google.from_env()
        assert p.api_key == "AIza-env"

    def test_from_env_reads_late_env_update(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="API key not configured"):
            Google.from_env()
        monkeypatch.setenv("GOOGLE_API_KEY", "AIza-late")
        assert Google.from_env().api_key == "AIza-late"

    def test_from_env_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="API key not configured"):
            Google.from_env()

    def test_model_creation(self) -> None:
        p: Provider = Google.from_key("AIza-test")
        m = p.model("gemini-2.0-flash")
        assert m.model_id == "gemini-2.0-flash"
        assert m.max_request_tokens == 1_000_000

    def test_model_custom_max_request_tokens(self) -> None:
        p: Provider = Google.from_key("AIza-test")
        m = p.model("gemini-2.0-flash", max_request_tokens=32_000)
        assert m.max_request_tokens == 32_000


# -- Request building --------------------------------------------------


class TestBuildRequest:
    def test_user_message(self) -> None:
        request = ModelRequest(
            messages=[_user("hello")],
        )
        body = cast(dict[str, Any], _build_request(request))
        assert body["contents"] == [
            {"role": "user", "parts": [{"text": "hello"}]},
        ]

    def test_system_prompt(self) -> None:
        request = ModelRequest(
            messages=[_user("hello")],
            system="Be concise.",
        )
        body = cast(dict[str, Any], _build_request(request))
        assert body["systemInstruction"] == {
            "parts": [{"text": "Be concise."}],
        }

    def test_assistant_with_tool_calls(self) -> None:
        request = ModelRequest(
            messages=[
                MultipartMessage(
                    (
                        TextMessage("Calling tool.", "text/plain"),
                        MultipartMessage(
                            (
                                TextMessage("t1", "text/x-queue-id"),
                                JsonMessage(
                                    json_freeze({"command": "ls"}),
                                    "application/x-tool-bash",
                                ),
                            ),
                            "multipart/x-tool-call",
                        ),
                    ),
                    "multipart/x-model-message",
                ),
            ],
        )
        body = cast(dict[str, Any], _build_request(request))
        parts = body["contents"][0]["parts"]
        assert parts[0] == {"text": "Calling tool."}
        assert parts[1] == {
            "functionCall": {
                "name": "bash",
                "args": {"command": "ls"},
            },
        }

    def test_tool_result(self) -> None:
        request = ModelRequest(
            messages=[
                MultipartMessage(
                    (
                        TextMessage("bash", "text/x-queue-id"),
                        TextMessage("file1 file2", "text/plain"),
                    ),
                    "multipart/x-tool-result",
                ),
            ],
        )
        body = cast(dict[str, Any], _build_request(request))
        content = body["contents"][0]
        assert content["role"] == "user"
        fr = content["parts"][0]["functionResponse"]
        assert fr["name"] == "bash"
        assert fr["response"]["content"] == "file1 file2"

    def test_generation_config(self) -> None:
        request = ModelRequest(
            messages=[_user("hello")],
            temperature=0.5,
        )
        body = cast(dict[str, Any], _build_request(request))
        gc = body["generationConfig"]
        assert "maxOutputTokens" not in gc
        assert gc["temperature"] == 0.5


# -- Response parsing -------------------------------------------------


class TestParseResponse:
    def test_text_response(self) -> None:
        data: dict[str, Any] = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Hello!"}],
                    },
                    "finishReason": "STOP",
                },
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
            },
        }
        resp = _parse_response(data, Pricing())
        assert _text(resp) == "Hello!"
        assert resp.tokens.input_tokens == 10
        assert resp.tokens.output_tokens == 5
        assert resp.stop_reason == "model_finished"

    def test_tool_call_response(self) -> None:
        data: dict[str, Any] = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "bash",
                                    "args": {"command": "ls"},
                                },
                            },
                        ],
                    },
                    # Real Gemini API uses ``STOP`` even for tool calls;
                    # the adapter's ``has_tool_use`` upgrade promotes it
                    # to canonical ``model_tool_use``.
                    "finishReason": "STOP",
                },
            ],
            "usageMetadata": {
                "promptTokenCount": 20,
                "candidatesTokenCount": 10,
            },
        }
        resp = _parse_response(data, Pricing())
        tcs = _tool_calls(resp)
        assert len(tcs) == 1
        assert tcs[0].descriptor == "multipart/x-tool-call"
        assert get_tool_name(tcs[0]) == "bash"
        assert get_directive(tcs[0]) == json_freeze({"command": "ls"})
        assert resp.stop_reason == "model_tool_use"

    def test_mixed_response(self) -> None:
        data: dict[str, Any] = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Let me check."},
                            {
                                "functionCall": {
                                    "name": "read",
                                    "args": {
                                        "file_path": "foo.py",
                                    },
                                },
                            },
                        ],
                    },
                    "finishReason": "STOP",
                },
            ],
            "usageMetadata": {},
        }
        resp = _parse_response(data, Pricing())
        assert _text(resp) == "Let me check."
        tcs = _tool_calls(resp)
        assert len(tcs) == 1
        assert tcs[0].descriptor == "multipart/x-tool-call"
        assert get_tool_name(tcs[0]) == "read"
        # ``STOP`` upgrades to ``model_tool_use`` because tool calls are present.
        assert resp.stop_reason == "model_tool_use"


class TestStopReasonNormalization:
    """End-to-end: Gemini wire format → canonical agent vocabulary."""

    def test_max_tokens_normalized(self) -> None:
        data: dict[str, Any] = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "trunc"}]},
                    "finishReason": "MAX_TOKENS",
                },
            ],
            "usageMetadata": {},
        }
        resp = _parse_response(data, Pricing())
        assert resp.stop_reason == "max_tokens"

    @pytest.mark.parametrize(
        "reason",
        ["SAFETY", "RECITATION", "PROHIBITED_CONTENT"],
    )
    def test_safety_categories_normalized_to_model_refusal(self, reason: str) -> None:
        data: dict[str, Any] = {
            "candidates": [
                {
                    "content": {"parts": [{"text": ""}]},
                    "finishReason": reason,
                },
            ],
            "usageMetadata": {},
        }
        resp = _parse_response(data, Pricing())
        assert resp.stop_reason == "model_refusal"

    def test_text_only_stop_to_model_finished(self) -> None:
        data: dict[str, Any] = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "done"}]},
                    "finishReason": "STOP",
                },
            ],
            "usageMetadata": {},
        }
        resp = _parse_response(data, Pricing())
        assert resp.stop_reason == "model_finished"


class TestToolResultMapping:
    def test_tool_result_uses_function_name(self) -> None:
        request = ModelRequest(
            messages=[
                MultipartMessage(
                    (
                        MultipartMessage(
                            (
                                TextMessage("call_abc", "text/x-queue-id"),
                                JsonMessage(
                                    json_freeze({"command": "ls"}),
                                    "application/x-tool-bash",
                                ),
                            ),
                            "multipart/x-tool-call",
                        ),
                    ),
                    "multipart/x-model-message",
                ),
                MultipartMessage(
                    (
                        TextMessage("call_abc", "text/x-queue-id"),
                        TextMessage("file1", "text/plain"),
                    ),
                    "multipart/x-tool-result",
                ),
            ],
        )
        body = cast(dict[str, Any], _build_request(request))
        fr = body["contents"][-1]["parts"][0]["functionResponse"]
        assert fr["name"] == "bash"


class TestSendMocked:
    @pytest.mark.anyio
    async def test_send(self) -> None:

        provider = Google.from_key("AIza-test")
        model = provider.model("gemini-2.0-flash")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "candidates": [
                {"content": {"parts": [{"text": "Hi!"}]}, "finishReason": "STOP"}
            ],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2},
        }
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        with patch(
            "sagent.providers.google.httpx.AsyncClient",
            return_value=mock_client,
        ):
            resp = await model.buffer(request=ModelRequest(messages=[_user("hello")]))
        assert _text(resp) == "Hi!"

    @pytest.mark.anyio
    async def test_stream_basic(self) -> None:
        """SSE streamGenerateContent assembles text + usage."""
        provider = Google.from_key("AIza-test")
        model = provider.model("gemini-2.0-flash")
        mock_client = _stream_client(
            [
                'data: {"candidates":[{"content":{"parts":[{"text":"Hi"}]}}]}',
                'data: {"candidates":[{"content":{"parts":[{"text":"!"}]},'
                '"finishReason":"STOP"}],'
                '"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2}}',
            ]
        )
        with patch(
            "sagent.providers.google.httpx.AsyncClient",
            return_value=mock_client,
        ):
            resp = await model.stream(request=ModelRequest(messages=[_user("hello")]))
        assert _text(resp) == "Hi!"
        assert resp.tokens.input_tokens == 5
        assert resp.tokens.output_tokens == 2


# -- stream with on_text callback (line 160) --------------------------


class TestStreamOnText:
    @pytest.mark.anyio
    async def test_stream_calls_on_text(self) -> None:
        provider = Google.from_key("AIza-test")
        model = provider.model("gemini-2.0-flash")
        mock_client = _stream_client(
            [
                'data: {"candidates":[{"content":{"parts":[{"text":"Hello!"}]}}]}',
            ]
        )
        chunks: list[str] = []
        with patch(
            "sagent.providers.google.httpx.AsyncClient",
            return_value=mock_client,
        ):
            resp = await model.stream(
                request=ModelRequest(messages=[_user("hi")]),
                on_text=chunks.append,
            )
        assert _text(resp) == "Hello!"
        assert "Hello!" in chunks


# -- _build_request with tools (line 242) ------------------------------


class TestBuildRequestWithTools:
    def test_tools_included(self) -> None:
        class _FakeTool:
            def __init__(self) -> None:
                self.name = "bash"
                self.tool_id = "application/x-tool-bash"
                self.description = "Run bash."
                self.supports_microcompaction = False
                self.directive_schema = json_freeze(
                    {"type": "object", "properties": {}}
                )

            def summary(self, msg: Message) -> str:
                del msg
                return self.name

            def summary_result(self, result: Message) -> str | None:
                del result
                return None

            def prompt(self) -> str | None:
                return None

            async def run(self, msg: Message) -> Message:
                del msg
                return TextMessage("", "text/plain")

        request = ModelRequest(
            messages=[_user("hello")],
            tools=[_FakeTool()],
        )
        body = cast(dict[str, Any], _build_request(request))
        assert "tools" in body
        decls = body["tools"][0]["functionDeclarations"]
        assert decls[0]["name"] == "bash"


class TestModelProperties:
    def test_all_properties(self) -> None:
        m = Google.from_key("k").model("gemini-2.0-flash")
        assert m.max_response_tokens > 0
        assert m.supports_streaming is True
        assert m.supports_thinking is False
        assert m.supports_effort is False
        assert m.supports_cache_control is False
        assert m.supports_context_management is False
        assert m.supports_persistent_retry is False
        assert m.supports_account_auth is False


class TestIsContextOverflow:
    def test_recognizes_keywords(self) -> None:
        m = Google.from_key("k").model("gemini-2.0-flash")
        assert m.is_context_overflow(RuntimeError("input is too large"))
        assert m.is_context_overflow(RuntimeError("prompt too long"))
        assert m.is_context_overflow(
            RuntimeError("exceeds the maximum number of tokens")
        )

    def test_unrelated_error(self) -> None:
        m = Google.from_key("k").model("gemini-2.0-flash")
        assert not m.is_context_overflow(RuntimeError("quota"))


class TestBuildRequestAttachments:
    def test_image_attachment_inline(self) -> None:
        buf = BytesIO()
        Image.new("RGB", (6, 6)).save(buf, format="PNG")
        request = ModelRequest(
            messages=[
                MultipartMessage(
                    (
                        TextMessage("describe", "text/plain"),
                        BytesMessage(buf.getvalue(), "image/png"),
                    ),
                    "multipart/x-user-message",
                ),
            ],
        )
        body = cast(dict[str, Any], _build_request(request))
        contents = body["contents"]
        parts = contents[0]["parts"]
        assert any("inlineData" in p for p in parts)
        assert any("text" in p for p in parts)

    def test_attachment_from_path(self, tmp_path: Path) -> None:
        buf = BytesIO()
        Image.new("RGB", (4, 4)).save(buf, format="PNG")
        p = tmp_path / "i.png"
        p.write_bytes(buf.getvalue())
        request = ModelRequest(
            messages=[
                MultipartMessage(
                    (BytesMessage(p.read_bytes(), "image/png"),),
                    "multipart/x-user-message",
                ),
            ],
        )
        body = cast(dict[str, Any], _build_request(request))
        parts = body["contents"][0]["parts"]
        assert any("inlineData" in p for p in parts)


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
