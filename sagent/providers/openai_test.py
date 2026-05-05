"""Tests for sagent.providers.openai."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

from PIL import Image

import pytest

from sagent.custom_exceptions import PromptTooLongError
from sagent.custom_types import (
    BytesMessage,
    JsonMessage,
    Message,
    ModelRequest,
    ModelResponse,
    MultipartMessage,
    TextMessage,
)
from sagent.lib.json import json_freeze
from sagent.lib.message import get_directive, get_tool_name
from sagent.providers.openai import OpenAI
from sagent.providers.openai_compat import (
    build_messages,
    parse_response,
)


_OPENAI_DEFAULT_PRICING = OpenAI.KNOWN_MODELS["gpt-4o"].pricing


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
    """Build an AsyncClient mock whose `stream()` yields SSE lines."""

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
        p = OpenAI.from_key("sk-test")
        assert p.api_key == "sk-test"

    def test_unknown_model_raises(self) -> None:
        p = OpenAI.from_key("sk-test")
        with pytest.raises(ValueError, match="Unknown model"):
            p.model("gpt-unknown")

    def test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        p = OpenAI.from_env()
        assert p.api_key == "sk-env"

    def test_from_env_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="API key not configured"):
            OpenAI.from_env()

    def test_model_creation(self) -> None:
        p = OpenAI.from_key("sk-test")
        m = p.model("gpt-4o")
        assert m.model_id == "gpt-4o"
        assert m.max_request_tokens == 128_000

    def test_model_custom_max_request_tokens(self) -> None:
        p = OpenAI.from_key("sk-test")
        m = p.model("gpt-4o", max_request_tokens=32_000)
        assert m.max_request_tokens == 32_000


# -- Media translation ----------------------------------------------


class TestBuildMessages:
    def test_user_message(self) -> None:
        request = ModelRequest(
            messages=[_user("hello")],
        )
        msgs = cast(list[Any], build_messages(request))
        assert msgs == [{"role": "user", "content": "hello"}]

    def test_system_prompt(self) -> None:
        request = ModelRequest(
            messages=[_user("hello")],
            system="Be concise.",
        )
        msgs = cast(list[Any], build_messages(request))
        assert msgs[0] == {
            "role": "system",
            "content": "Be concise.",
        }
        assert msgs[1] == {"role": "user", "content": "hello"}

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
        msgs = cast(list[Any], build_messages(request))
        assert msgs[0]["role"] == "assistant"
        assert msgs[0]["content"] == "Calling tool."
        tc = msgs[0]["tool_calls"][0]
        assert tc["id"] == "call_0"
        assert tc["function"]["name"] == "bash"

    def test_tool_result(self) -> None:
        request = ModelRequest(
            messages=[
                MultipartMessage(
                    (
                        TextMessage("t1", "text/x-queue-id"),
                        TextMessage("file1 file2", "text/plain"),
                    ),
                    "multipart/x-tool-result",
                ),
            ],
        )
        msgs = cast(list[Any], build_messages(request))
        assert msgs[0]["role"] == "tool"
        assert msgs[0]["tool_call_id"] == "call_0"
        assert msgs[0]["content"] == "file1 file2"


# -- Response parsing -------------------------------------------------


class TestParseResponse:
    def test_text_response(self) -> None:
        data: dict[str, Any] = {
            "choices": [
                {
                    "message": {"content": "Hello!"},
                    "finish_reason": "stop",
                },
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
            },
        }
        resp = parse_response(
            data, pricing=_OPENAI_DEFAULT_PRICING, reasoning_field=None
        )
        assert _text(resp) == "Hello!"
        assert resp.tokens.input_tokens == 10
        assert resp.tokens.output_tokens == 5
        assert resp.stop_reason == "model_finished"

    def test_tool_call_response(self) -> None:
        data: dict[str, Any] = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "bash",
                                    "arguments": '{"command": "ls"}',
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                },
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
            },
        }
        resp = parse_response(
            data, pricing=_OPENAI_DEFAULT_PRICING, reasoning_field=None
        )
        tcs = _tool_calls(resp)
        assert len(tcs) == 1
        assert tcs[0].descriptor == "multipart/x-tool-call"
        assert get_tool_name(tcs[0]) == "bash"
        assert get_directive(tcs[0]) == json_freeze({"command": "ls"})
        assert resp.stop_reason == "model_tool_use"


class TestSendMocked:
    @pytest.mark.anyio
    async def test_send(self) -> None:

        provider = OpenAI.from_key("sk-test")
        model = provider.model("gpt-4o")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Hi!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        with patch(
            "sagent.providers.openai_compat.httpx.AsyncClient",
            return_value=mock_client,
        ):
            resp = await model.buffer(request=ModelRequest(messages=[_user("hello")]))
        assert _text(resp) == "Hi!"

    @pytest.mark.anyio
    async def test_stream_basic(self) -> None:
        """Real SSE streaming returns assembled text + usage."""
        provider = OpenAI.from_key("sk-test")
        model = provider.model("gpt-4o")
        mock_client = _stream_client(
            [
                'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"Hi"}}]}',
                'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"!"}}]}',
                'data: {"id":"chatcmpl-1","choices":[{"delta":{},"finish_reason":"stop"}],'
                '"usage":{"prompt_tokens":5,"completion_tokens":2}}',
                "data: [DONE]",
            ]
        )
        with patch(
            "sagent.providers.openai_compat.httpx.AsyncClient",
            return_value=mock_client,
        ):
            resp = await model.stream(request=ModelRequest(messages=[_user("hello")]))
        assert _text(resp) == "Hi!"
        assert resp.tokens.input_tokens == 5
        assert resp.tokens.output_tokens == 2


# -- stream with on_text callback (line 176) --------------------------


class TestStreamOnText:
    @pytest.mark.anyio
    async def test_stream_calls_on_text(self) -> None:
        provider = OpenAI.from_key("sk-test")
        model = provider.model("gpt-4o")
        mock_client = _stream_client(
            [
                'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"Hello!"}}]}',
                'data: {"id":"chatcmpl-1","choices":[{"delta":{},"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
        )
        chunks: list[str] = []
        with patch(
            "sagent.providers.openai_compat.httpx.AsyncClient",
            return_value=mock_client,
        ):
            resp = await model.stream(
                request=ModelRequest(messages=[_user("hi")]),
                on_text=chunks.append,
            )
        assert _text(resp) == "Hello!"
        assert "Hello!" in chunks


# -- send with tools (line 137) ----------------------------------------


class TestSendWithTools:
    @pytest.mark.anyio
    async def test_tools_in_body(self) -> None:
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

            def prompt(self) -> str | None:
                return None

            async def run(self, msg: Message) -> Message:
                del msg
                return TextMessage("", "text/plain")

        provider = OpenAI.from_key("sk-test")
        model = provider.model("gpt-4o")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {},
        }
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        with patch(
            "sagent.providers.openai_compat.httpx.AsyncClient",
            return_value=mock_client,
        ):
            resp = await model.buffer(
                request=ModelRequest(
                    messages=[_user("hello")],
                    tools=[_FakeTool()],
                )
            )
        assert _text(resp) == "ok"
        call_args = mock_client.post.call_args
        body = call_args.kwargs.get("json") or call_args[1].get("json")
        assert "tools" in body
        assert body["tools"][0]["function"]["name"] == "bash"


class TestModelProperties:
    """Exercise the trivial property returns."""

    def test_properties_on_gpt4o(self) -> None:
        m = OpenAI.from_key("sk-x").model("gpt-4o")
        assert m.max_response_tokens > 0
        assert m.supports_streaming is True
        assert m.supports_thinking is False
        assert m.supports_effort is False
        assert m.supports_cache_control is False
        assert m.supports_context_management is False
        assert m.supports_persistent_retry is False
        assert m.supports_account_auth is False

    def test_supports_effort_for_o_series(self) -> None:
        m = OpenAI.from_key("sk-x").model("o1")
        assert m.supports_effort is True


class TestIsContextOverflow:
    def test_recognizes_keywords(self) -> None:
        m = OpenAI.from_key("sk-x").model("gpt-4o")
        assert m.is_context_overflow(RuntimeError("context_length_exceeded"))
        assert m.is_context_overflow(RuntimeError("maximum context length x"))

    def test_unrelated_error(self) -> None:
        m = OpenAI.from_key("sk-x").model("gpt-4o")
        assert not m.is_context_overflow(RuntimeError("rate limit"))


class TestBuildMessagesAttachments:
    def test_image_attachment(self) -> None:
        buf = BytesIO()
        Image.new("RGB", (8, 8), color="red").save(buf, format="PNG")
        request = ModelRequest(
            messages=[
                MultipartMessage(
                    (
                        TextMessage("what is this?", "text/plain"),
                        BytesMessage(buf.getvalue(), "image/png"),
                    ),
                    "multipart/x-user-message",
                ),
            ],
        )
        msgs = cast(list[Any], build_messages(request))
        assert msgs[0]["role"] == "user"
        blocks = msgs[0]["content"]
        assert any(b.get("type") == "image_url" for b in blocks)
        # Text part preserved when content is non-empty.
        assert any(b.get("type") == "text" for b in blocks)

    def test_attachment_empty_content(self, tmp_path: Path) -> None:
        buf = BytesIO()
        Image.new("RGB", (4, 4)).save(buf, format="PNG")

        p = tmp_path / "img.png"
        p.write_bytes(buf.getvalue())
        request = ModelRequest(
            messages=[
                MultipartMessage(
                    (BytesMessage(p.read_bytes(), "image/png"),),
                    "multipart/x-user-message",
                ),
            ],
        )
        msgs = cast(list[Any], build_messages(request))
        blocks = msgs[0]["content"]
        # No text block when content has no text part.
        assert not any(b.get("type") == "text" for b in blocks)


class TestStreamErrors:
    @pytest.mark.anyio
    async def test_400_context_overflow_raises_prompt_too_long(self) -> None:
        provider = OpenAI.from_key("sk-x")
        model = provider.model("gpt-4o")
        err_body = b'{"error":{"message":"context_length_exceeded: too long"}}'
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.aread = AsyncMock(return_value=err_body)
        mock_resp.raise_for_status = MagicMock()
        stream_cm = AsyncMock()
        stream_cm.__aenter__ = AsyncMock(return_value=mock_resp)
        stream_cm.__aexit__ = AsyncMock(return_value=False)
        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=stream_cm)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        with (
            patch(
                "sagent.providers.openai_compat.httpx.AsyncClient",
                return_value=mock_client,
            ),
            pytest.raises(PromptTooLongError),
        ):
            await model.stream(
                request=ModelRequest(messages=[_user("x")]),
                on_text=None,
            )


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
