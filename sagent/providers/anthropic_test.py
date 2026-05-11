"""Tests for sagent.providers.anthropic."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Self, cast
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import httpx
import pytest

from sagent.custom_exceptions import (
    PromptTooLongError,
    StreamInterruptedError,
)
from sagent.custom_types import (
    BytesMessage,
    JsonMessage,
    Message,
    ModelRequest,
    ModelResponse,
    MultipartMessage,
    Provider,
    TextMessage,
    TokenCount,
)
from sagent.lib.json import json_freeze
from sagent.lib.message import tool_call_message
from sagent.providers.anthropic import (
    Anthropic,
    _add_cache_breakpoint,
    _build_messages,
    _guard_stream_interrupt,
    _parse_response,
    _raise_if_prompt_too_long,
)
from sagent.providers.lib.cost import Pricing


def _user(text: str) -> Message:
    return TextMessage(text, "text/x-user-message")


def _assistant(
    text: str = "",
    *,
    tool_calls: list[Message] | None = None,
    thinking_blocks: list[dict[str, Any]] | None = None,
) -> Message:
    parts: list[Message] = [
        JsonMessage(json_freeze(tb), "application/x-thinking-structured")
        for tb in thinking_blocks or []
    ]
    if text:
        parts.append(TextMessage(text, "text/plain"))
    parts.extend(tool_calls or [])
    return MultipartMessage(tuple(parts), "multipart/x-model-message")


def _tool_result(queue_id: str, text: str, *, is_error: bool = False) -> Message:
    desc = "text/x-error" if is_error else "text/plain"
    return MultipartMessage(
        (
            TextMessage(queue_id, "text/x-queue-id"),
            TextMessage(text, desc),
        ),
        "multipart/x-tool-result",
    )


def _resp_text(resp: ModelResponse) -> str | None:
    if not isinstance(resp.content, MultipartMessage):
        return None
    for p in resp.content.content:
        if p.descriptor == "text/plain" and isinstance(p, TextMessage):
            return p.content
    return None


def _resp_thinking(resp: ModelResponse) -> str | None:
    if not isinstance(resp.content, MultipartMessage):
        return None
    for p in resp.content.content:
        if p.descriptor == "application/x-thinking-structured" and isinstance(
            p.content, MappingProxyType
        ):
            thinking = p.content.get("thinking")
            if isinstance(thinking, str):
                return thinking
    return None


def _resp_tool_calls(resp: ModelResponse) -> list[Message]:
    if not isinstance(resp.content, MultipartMessage):
        return []
    result: list[Message] = []
    for p in resp.content.content:
        if p.descriptor == "multipart/x-tool-call" and isinstance(p, MultipartMessage):
            result.extend(
                inner
                for inner in p.content
                if inner.descriptor.startswith("application/x-tool-")
            )
    return result


# ------------------------------------------------------------------
# Provider construction
# ------------------------------------------------------------------


class TestProviderConstruction:
    def test_from_key(self) -> None:
        provider = Anthropic.from_key("sk-ant-test")
        assert provider._api_key == "sk-ant-test"

    def test_model_creation(self) -> None:
        provider: Provider = Anthropic.from_key("sk-ant-test")
        m = provider.model("claude-sonnet-4-6")
        assert m.model_id == "claude-sonnet-4-6"
        assert m.max_request_tokens == 200_000

    def test_model_creation_1m_tag(self) -> None:
        provider = Anthropic.from_key("sk-ant-test")
        m = provider.model("claude-sonnet-4-6+1m")
        assert m.max_request_tokens == 1_000_000

    def test_model_creation_200k_tag(self) -> None:
        provider = Anthropic.from_key("sk-ant-test")
        m = provider.model("claude-sonnet-4-6+200k")
        assert m.max_request_tokens == 200_000


# ------------------------------------------------------------------
# System prompt injection
# ------------------------------------------------------------------


class TestSystemPrompt:
    def test_plain_mode_none_omits_system(self) -> None:
        provider = Anthropic.from_key("sk-ant-test")
        result = provider.build_system(None)
        assert result is anthropic.NOT_GIVEN

    def test_plain_mode_passes_through(self) -> None:
        provider = Anthropic.from_key("sk-ant-test")
        assert provider.build_system("Do X.") == "Do X."


# ------------------------------------------------------------------
# _build_messages
# ------------------------------------------------------------------


class TestBuildMessages:
    def test_user_message(self) -> None:
        request = ModelRequest(messages=[_user("hello")])
        msgs = _build_messages(request)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_empty_user_skipped(self) -> None:
        msgs = _build_messages(ModelRequest(messages=[_user("")]))
        assert len(msgs) == 0

    def test_assistant_with_tools(self) -> None:
        request = ModelRequest(
            messages=[
                _assistant(
                    "Calling.",
                    tool_calls=[
                        tool_call_message("t1", "bash", json_freeze({"command": "ls"}))
                    ],
                ),
            ],
        )
        msgs = _build_messages(request)
        content = msgs[0]["content"]
        assert isinstance(content, list)
        assert isinstance(content[0], dict)
        assert cast(dict[str, Any], content[0])["type"] == "text"
        assert isinstance(content[1], dict)
        assert cast(dict[str, Any], content[1])["type"] == "tool_use"

    def test_empty_assistant_skipped(self) -> None:
        msgs = _build_messages(ModelRequest(messages=[_assistant("")]))
        assert len(msgs) == 0

    def test_thinking_only_gets_placeholder(self) -> None:
        msgs = _build_messages(
            ModelRequest(
                messages=[
                    _user("hi"),
                    _assistant(
                        "",
                        thinking_blocks=[{"type": "thinking", "thinking": "hmm"}],
                    ),
                    _user("You produced no output."),
                ]
            )
        )
        assert len(msgs) == 3
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert msgs[2]["role"] == "user"
        content = msgs[1]["content"]
        assert isinstance(content, list)
        last_block = cast(dict[str, Any], content[-1])
        assert last_block.get("type") == "text"
        assert last_block.get("text") == "."

    def test_thinking_plus_text_no_placeholder(self) -> None:
        msgs = _build_messages(
            ModelRequest(
                messages=[
                    _assistant(
                        "I see.",
                        thinking_blocks=[{"type": "thinking", "thinking": "hmm"}],
                    ),
                ]
            )
        )
        content = msgs[0]["content"]
        assert isinstance(content, list)
        text_blocks = [
            b for b in content if isinstance(b, dict) and b.get("type") == "text"
        ]
        assert any(b.get("text") == "I see." for b in text_blocks)

    def test_thinking_plus_tool_use_no_placeholder(self) -> None:
        msgs = _build_messages(
            ModelRequest(
                messages=[
                    _assistant(
                        "",
                        thinking_blocks=[{"type": "thinking", "thinking": "hmm"}],
                        tool_calls=[
                            tool_call_message(
                                "t1", "Bash", json_freeze({"command": "ls"})
                            )
                        ],
                    ),
                ]
            )
        )
        content = msgs[0]["content"]
        assert isinstance(content, list)
        last_block = cast(dict[str, Any], content[-1])
        assert last_block.get("type") == "tool_use"

    def test_tool_results_batched(self) -> None:
        msgs = _build_messages(
            ModelRequest(
                messages=[
                    _tool_result("t1", "out1"),
                    _tool_result("t2", "out2"),
                ],
            ),
        )
        assert len(msgs) == 1
        content = msgs[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2

    def test_tool_result_is_error(self) -> None:
        msgs = _build_messages(
            ModelRequest(messages=[_tool_result("t1", "err", is_error=True)]),
        )
        content = msgs[0]["content"]
        assert isinstance(content, list)
        block = content[0]
        assert isinstance(block, dict)
        assert cast(dict[str, Any], block).get("is_error") is True

    def test_user_after_tool_results_coalesces_into_same_wire_msg(self) -> None:
        """Next-priority drain: a text-only user message placed after a
        run of tool results merges onto the same wire user message so
        the tool_results batch and the injected text share one API
        round-trip (no extra assistant bounce).
        """
        msgs = _build_messages(
            ModelRequest(
                messages=[
                    _tool_result("t1", "out1"),
                    _tool_result("t2", "out2"),
                    _user("by the way, also do Y"),
                ],
            ),
        )
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        content = msgs[0]["content"]
        assert isinstance(content, list)
        types = [cast(dict[str, Any], b).get("type") for b in content]
        assert types == ["tool_result", "tool_result", "text"]
        text_block = cast(dict[str, Any], content[-1])
        assert text_block.get("text") == "by the way, also do Y"

    def test_user_with_attachments_does_not_coalesce(self) -> None:
        """Attachment-bearing user messages (tuple content) keep their own
        wire message so the attachment layout (image / document blocks)
        stays intact.
        """
        msgs = _build_messages(
            ModelRequest(
                messages=[
                    _tool_result("t1", "out1"),
                    MultipartMessage(
                        (
                            BytesMessage(b"%PDF-1.4\n", "application/pdf"),
                            TextMessage("with pdf", "text/plain"),
                        ),
                        "multipart/x-user-message",
                    ),
                ],
            ),
        )
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "user"
        first_content = msgs[0]["content"]
        assert isinstance(first_content, list)
        assert cast(dict[str, Any], first_content[0]).get("type") == "tool_result"
        second_content = msgs[1]["content"]
        assert isinstance(second_content, list)
        types = [cast(dict[str, Any], b).get("type") for b in second_content]
        assert "document" in types
        assert "text" in types


# ------------------------------------------------------------------
# _parse_response
# ------------------------------------------------------------------


class TestParseResponse:
    def test_text_only(self) -> None:
        text_block = anthropic.types.TextBlock(type="text", text="Hello")
        raw = MagicMock()
        raw.model = "claude-sonnet-4-6"
        raw.content = [text_block]
        raw.usage.input_tokens = 10
        raw.usage.output_tokens = 5
        raw.usage.cache_creation_input_tokens = 0
        raw.usage.cache_read_input_tokens = 0
        raw.stop_reason = "end_turn"
        resp = _parse_response(raw, Pricing())
        assert _resp_text(resp) == "Hello"

    def test_tool_use(self) -> None:
        tool_block = anthropic.types.ToolUseBlock(
            type="tool_use", id="t1", name="bash", input={"command": "ls"}
        )
        raw = MagicMock()
        raw.model = "claude-sonnet-4-6"
        raw.content = [tool_block]
        raw.usage.input_tokens = 20
        raw.usage.output_tokens = 10
        raw.usage.cache_creation_input_tokens = 0
        raw.usage.cache_read_input_tokens = 0
        raw.stop_reason = "tool_use"
        resp = _parse_response(raw, Pricing())
        tcs = _resp_tool_calls(resp)
        assert len(tcs) == 1
        assert tcs[0].descriptor == "application/x-tool-bash"

    def test_thinking_block(self) -> None:
        text_block = anthropic.types.TextBlock(type="text", text="Answer")
        thinking = anthropic.types.ThinkingBlock(
            type="thinking",
            thinking="deep thoughts",
            signature="sig",
        )
        raw = MagicMock()
        raw.model = "claude-sonnet-4-6"
        raw.content = [text_block, thinking]
        raw.usage.input_tokens = 30
        raw.usage.output_tokens = 15
        raw.usage.cache_creation_input_tokens = 0
        raw.usage.cache_read_input_tokens = 0
        raw.stop_reason = "end_turn"
        resp = _parse_response(raw, Pricing())
        assert _resp_text(resp) == "Answer"
        assert _resp_thinking(resp) == "deep thoughts"


# ------------------------------------------------------------------
# _raise_if_prompt_too_long
# ------------------------------------------------------------------


class TestGuardStreamInterrupt:
    """``_guard_stream_interrupt`` encodes the "``stop_reason`` is
    unreliable" rule - gate on content (tool call parts), not on
    ``stop_reason``.
    """

    def _resp(
        self,
        *,
        stop_reason: str,
        tool_calls: list[Message] | None = None,
        text: str | None = None,
    ) -> ModelResponse:
        parts: list[Message] = []
        if text:
            parts.append(TextMessage(text, "text/plain"))
        parts.extend(tool_calls or [])
        return ModelResponse(
            content=MultipartMessage(
                tuple(parts),
                "multipart/x-model-message",
            ),
            stop_reason=stop_reason,
            tokens=TokenCount(),
        )

    def test_raises_when_model_tool_use_claimed_but_no_blocks(self) -> None:
        """The actual failure mode: API says ``model_tool_use`` but parse
        produced no tool call parts.
        """
        resp = self._resp(stop_reason="model_tool_use", tool_calls=[], text="partial")
        with pytest.raises(StreamInterruptedError) as exc:
            _guard_stream_interrupt(resp, kind="stream", model_id="mock")
        assert exc.value.response is resp

    def test_no_raise_when_model_tool_use_has_blocks(self) -> None:
        """Normal ``model_tool_use`` response - guard is a no-op."""
        resp = self._resp(
            stop_reason="model_tool_use",
            tool_calls=[tool_call_message("t1", "echo", json_freeze({}))],
        )
        _guard_stream_interrupt(resp, kind="stream", model_id="mock")

    def test_no_raise_when_model_finished_with_no_tools(self) -> None:
        """``model_finished`` with no tools is a legitimate final response,
        not a stream interrupt.
        """
        resp = self._resp(stop_reason="model_finished", tool_calls=[], text="done")
        _guard_stream_interrupt(resp, kind="stream", model_id="mock")

    def test_no_raise_when_model_finished_with_tools(self) -> None:
        """``model_finished`` with tools (unusual but legal) is not an interrupt."""
        resp = self._resp(
            stop_reason="model_finished",
            tool_calls=[tool_call_message("t1", "echo", json_freeze({}))],
        )
        _guard_stream_interrupt(resp, kind="stream", model_id="mock")

    def test_no_raise_on_other_stop_reasons(self) -> None:
        """``max_tokens`` / ``stop_sequence`` / etc. aren't interrupts
        even with empty tool_calls.
        """
        for reason in ("max_tokens", "stop_sequence", "model_continuing"):
            resp = self._resp(stop_reason=reason, tool_calls=[])
            _guard_stream_interrupt(resp, kind="stream", model_id="mock")


class TestRaiseIfPromptTooLong:
    def test_too_long(self) -> None:
        req = httpx.Request("POST", "https://api.anthropic.com")
        e = anthropic.BadRequestError(
            "prompt is too long", response=httpx.Response(400, request=req), body=None
        )
        with pytest.raises(PromptTooLongError):
            _raise_if_prompt_too_long(e)

    def test_extracts_token_counts(self) -> None:
        req = httpx.Request("POST", "https://api.anthropic.com")
        e = anthropic.BadRequestError(
            "prompt is too long: 150000 tokens > 100000 token maximum",
            response=httpx.Response(400, request=req),
            body=None,
        )
        with pytest.raises(PromptTooLongError) as exc_info:
            _raise_if_prompt_too_long(e)
        assert exc_info.value.actual_tokens == 150_000
        assert exc_info.value.limit_tokens == 100_000
        assert exc_info.value.token_gap == 50_000

    def test_no_token_counts_when_unparseable(self) -> None:
        req = httpx.Request("POST", "https://api.anthropic.com")
        e = anthropic.BadRequestError(
            "prompt is too long",
            response=httpx.Response(400, request=req),
            body=None,
        )
        with pytest.raises(PromptTooLongError) as exc_info:
            _raise_if_prompt_too_long(e)
        assert exc_info.value.actual_tokens is None
        assert exc_info.value.token_gap is None

    def test_unrelated_error(self) -> None:
        req = httpx.Request("POST", "https://api.anthropic.com")
        e = anthropic.BadRequestError(
            "invalid model", response=httpx.Response(400, request=req), body=None
        )
        _raise_if_prompt_too_long(e)


# ------------------------------------------------------------------
# _build_kwargs
# ------------------------------------------------------------------


class TestBuildKwargs:
    def test_basic(self) -> None:
        provider = Anthropic.from_key("sk-test")
        model = provider.model("claude-sonnet-4-6")
        request = ModelRequest(messages=[_user("hello")], system="Be concise.")
        kwargs = model._build_kwargs(request, _build_messages(request))
        assert kwargs["model"] == "claude-sonnet-4-6"
        assert "tools" not in kwargs

    def test_with_thinking(self) -> None:
        provider = Anthropic.from_key("sk-test")
        model = provider.model("claude-sonnet-4-6")
        request = ModelRequest(messages=[_user("hello")], thinking="adaptive")
        kwargs = model._build_kwargs(request, _build_messages(request))
        assert "thinking" in kwargs
        assert kwargs["temperature"] == 1.0


# ------------------------------------------------------------------
# get_sdk
# ------------------------------------------------------------------


class TestGetSdk:
    @pytest.mark.anyio
    async def test_api_key_creates_sdk(self) -> None:
        provider = Anthropic.from_key("sk-ant-test")
        sdk = await provider.get_sdk()
        assert sdk is not None
        sdk2 = await provider.get_sdk()
        assert sdk is sdk2
        await sdk.close()


# ------------------------------------------------------------------
# send with mock SDK
# ------------------------------------------------------------------


class TestAnthropicModelSend:
    @pytest.mark.anyio
    async def test_send(self) -> None:
        provider = Anthropic.from_key("sk-test")
        model = provider.model("claude-sonnet-4-6")
        text_block = anthropic.types.TextBlock(type="text", text="Hi!")
        mock_msg = MagicMock()
        mock_msg.model = "claude-sonnet-4-6"
        mock_msg.content = [text_block]
        mock_msg.usage.input_tokens = 10
        mock_msg.usage.output_tokens = 5
        mock_msg.usage.cache_creation_input_tokens = 0
        mock_msg.usage.cache_read_input_tokens = 0
        mock_msg.stop_reason = "end_turn"
        mock_sdk = AsyncMock()
        mock_sdk.messages.create = AsyncMock(return_value=mock_msg)
        with patch.object(provider, "get_sdk", AsyncMock(return_value=mock_sdk)):
            resp = await model.buffer(request=ModelRequest(messages=[_user("hello")]))
        assert _resp_text(resp) == "Hi!"
        assert resp.tokens.input_tokens == 10

    @pytest.mark.anyio
    async def test_buffer_prompt_too_long(self) -> None:
        provider = Anthropic.from_key("sk-test")
        model = provider.model("claude-sonnet-4-6")
        mock_sdk = AsyncMock()
        req = httpx.Request("POST", "https://api.anthropic.com")
        mock_sdk.messages.create = AsyncMock(
            side_effect=anthropic.BadRequestError(
                "prompt is too long",
                response=httpx.Response(400, request=req),
                body=None,
            ),
        )
        with (
            patch.object(provider, "get_sdk", AsyncMock(return_value=mock_sdk)),
            pytest.raises(PromptTooLongError),
        ):
            await model.buffer(request=ModelRequest(messages=[_user("hello")]))


# ------------------------------------------------------------------
# _build_kwargs: thinking=enabled + tools
# ------------------------------------------------------------------


class TestBuildKwargsThinkingAndTools:
    def test_haiku_does_not_support_thinking(self) -> None:
        provider = Anthropic.from_key("sk-test")
        assert not provider.model("claude-haiku-4-5").supports_thinking

    def test_sonnet_supports_thinking(self) -> None:
        provider = Anthropic.from_key("sk-test")
        assert provider.model("claude-sonnet-4-6").supports_thinking

    def test_haiku_omits_thinking_even_when_request_sets_it(self) -> None:
        provider = Anthropic.from_key("sk-test")
        m = provider.model("claude-haiku-4-5")
        request = ModelRequest(
            messages=[_user("think hard")],
            thinking="adaptive",
        )
        kwargs = m._build_kwargs(request, _build_messages(request))
        assert "thinking" not in kwargs

    def test_thinking_enabled(self) -> None:
        provider = Anthropic.from_key("sk-test")
        m = provider.model("claude-sonnet-4-6")
        request = ModelRequest(
            messages=[_user("think hard")],
            thinking="enabled",
        )
        kwargs = m._build_kwargs(request, _build_messages(request))
        thinking = cast(dict[str, object], kwargs["thinking"])
        assert thinking["type"] == "enabled"
        assert thinking["budget_tokens"] == m.max_response_tokens
        assert kwargs["max_tokens"] == m.max_response_tokens * 2

    def test_with_tools(self) -> None:
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

        provider = Anthropic.from_key("sk-test")
        m = provider.model("claude-sonnet-4-6")
        request = ModelRequest(
            messages=[_user("run it")],
            tools=[_FakeTool()],
        )
        kwargs = m._build_kwargs(request, _build_messages(request))
        assert "tools" in kwargs
        assert cast(list[dict[str, object]], kwargs["tools"])[0]["name"] == "bash"


# ------------------------------------------------------------------
# _AnthropicModel.stream
# ------------------------------------------------------------------


class TestAnthropicModelStream:
    @pytest.mark.anyio
    async def test_stream(self) -> None:
        provider = Anthropic.from_key("sk-test")
        m = provider.model("claude-sonnet-4-6")

        text_block = anthropic.types.TextBlock(type="text", text="Streamed!")
        mock_final = MagicMock()
        mock_final.model = "claude-sonnet-4-6"
        mock_final.content = [text_block]
        mock_final.usage.input_tokens = 5
        mock_final.usage.output_tokens = 3
        mock_final.usage.cache_creation_input_tokens = 0
        mock_final.usage.cache_read_input_tokens = 0
        mock_final.stop_reason = "end_turn"

        events: list[object] = [_text_delta_event("Stream"), _text_delta_event("ed!")]
        mock_stream = _AsyncIterator(events)
        mock_stream.get_final_message = AsyncMock(return_value=mock_final)

        mock_sdk = AsyncMock()
        mock_sdk.messages.stream = MagicMock(return_value=mock_stream)

        chunks: list[str] = []
        with patch.object(provider, "get_sdk", AsyncMock(return_value=mock_sdk)):
            resp = await m.stream(
                request=ModelRequest(messages=[_user("hi")]),
                on_text=chunks.append,
            )
        assert _resp_text(resp) == "Streamed!"
        assert "Stream" in chunks

    @pytest.mark.anyio
    async def test_stream_prompt_too_long(self) -> None:
        provider = Anthropic.from_key("sk-test")
        m = provider.model("claude-sonnet-4-6")

        req = httpx.Request("POST", "https://api.anthropic.com")
        exc = anthropic.BadRequestError(
            "prompt is too long",
            response=httpx.Response(400, request=req),
            body=None,
        )

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(side_effect=exc)
        mock_stream.__aexit__ = AsyncMock(return_value=False)

        mock_sdk = AsyncMock()
        mock_sdk.messages.stream = MagicMock(return_value=mock_stream)

        with (
            patch.object(provider, "get_sdk", AsyncMock(return_value=mock_sdk)),
            pytest.raises(PromptTooLongError),
        ):
            await m.stream(
                request=ModelRequest(messages=[_user("hi")]),
            )


# -- Helper for async iteration ----------------------------------------


class _AsyncIterator:
    def __init__(self, items: list[object]) -> None:
        self._items = items
        self._i = 0
        self.get_final_message: AsyncMock | None = None

    def __aiter__(self) -> _AsyncIterator:
        return self

    async def __anext__(self) -> object:
        if self._i >= len(self._items):
            raise StopAsyncIteration
        val = self._items[self._i]
        self._i += 1
        return val

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        pass


def _text_delta_event(text: str) -> MagicMock:
    """Create a mock stream event for a text delta."""
    event = MagicMock()
    event.type = "content_block_delta"
    event.delta.type = "text_delta"
    event.delta.text = text
    return event


# ------------------------------------------------------------------
# send: non-prompt-too-long BadRequestError re-raises
# ------------------------------------------------------------------


class TestSendNonPromptTooLongError:
    @pytest.mark.anyio
    async def test_non_prompt_too_long_bad_request_reraises(self) -> None:
        provider = Anthropic.from_key("sk-test")
        model = provider.model("claude-sonnet-4-6")
        mock_sdk = AsyncMock()
        req = httpx.Request("POST", "https://api.anthropic.com")
        mock_sdk.messages.create = AsyncMock(
            side_effect=anthropic.BadRequestError(
                "invalid model parameter",
                response=httpx.Response(400, request=req),
                body=None,
            ),
        )
        with (
            patch.object(provider, "get_sdk", AsyncMock(return_value=mock_sdk)),
            pytest.raises(anthropic.BadRequestError, match="invalid model"),
        ):
            await model.buffer(request=ModelRequest(messages=[_user("hello")]))


# ------------------------------------------------------------------
# stream: non-prompt-too-long BadRequestError re-raises
# ------------------------------------------------------------------


class TestStreamNonPromptTooLongError:
    @pytest.mark.anyio
    async def test_non_prompt_too_long_bad_request_reraises_stream(self) -> None:
        provider = Anthropic.from_key("sk-test")
        m = provider.model("claude-sonnet-4-6")
        req = httpx.Request("POST", "https://api.anthropic.com")
        exc = anthropic.BadRequestError(
            "invalid model parameter",
            response=httpx.Response(400, request=req),
            body=None,
        )
        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(side_effect=exc)
        mock_stream.__aexit__ = AsyncMock(return_value=False)
        mock_sdk = AsyncMock()
        mock_sdk.messages.stream = MagicMock(return_value=mock_stream)
        with (
            patch.object(provider, "get_sdk", AsyncMock(return_value=mock_sdk)),
            pytest.raises(anthropic.BadRequestError, match="invalid model"),
        ):
            await m.stream(
                request=ModelRequest(messages=[_user("hi")]),
            )


# ------------------------------------------------------------------
# _add_cache_breakpoint: list content with thinking blocks
# ------------------------------------------------------------------


class TestAddCacheBreakpoint:
    def test_skips_thinking_blocks(self) -> None:
        messages: list[anthropic.types.MessageParam] = [
            cast(
                anthropic.types.MessageParam,
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "hi"},
                        {"type": "thinking", "thinking": "deep"},
                    ],
                },
            )
        ]
        _add_cache_breakpoint(messages)
        content = messages[0]["content"]
        assert isinstance(content, list)
        b0 = cast(dict[str, Any], content[0])
        b1 = cast(dict[str, Any], content[1])
        assert "cache_control" in b0
        assert "cache_control" not in b1

    def test_default_ttl_is_5m_omits_field(self) -> None:
        messages: list[anthropic.types.MessageParam] = [
            cast(
                anthropic.types.MessageParam,
                {"role": "user", "content": "hi"},
            )
        ]
        _add_cache_breakpoint(messages)
        content = messages[0]["content"]
        assert isinstance(content, list)
        block = cast(dict[str, Any], content[0])
        cache = cast(dict[str, str], block["cache_control"])
        assert cache == {"type": "ephemeral"}

    def test_1h_ttl_sets_extended_marker(self) -> None:
        messages: list[anthropic.types.MessageParam] = [
            cast(
                anthropic.types.MessageParam,
                {"role": "user", "content": "hi"},
            )
        ]
        _add_cache_breakpoint(messages, "1h")
        content = messages[0]["content"]
        assert isinstance(content, list)
        block = cast(dict[str, Any], content[0])
        cache = cast(dict[str, str], block["cache_control"])
        assert cache == {"type": "ephemeral", "ttl": "1h"}

    def test_build_messages_propagates_request_cache_ttl(self) -> None:
        """Regression: ``ModelRequest.cache_ttl`` must reach
        ``_add_cache_breakpoint`` via ``_build_messages``. Removing the
        ``cache_ttl=request.cache_ttl`` argument silently drops the
        agent's setting.
        """
        request = ModelRequest(messages=[_user("hi")], cache_ttl="1h")
        msgs = _build_messages(request)
        content = msgs[0]["content"]
        assert isinstance(content, list)
        block = cast(dict[str, Any], content[0])
        assert block["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    def test_build_messages_default_cache_ttl_is_5m(self) -> None:
        request = ModelRequest(messages=[_user("hi")])
        msgs = _build_messages(request)
        content = msgs[0]["content"]
        assert isinstance(content, list)
        block = cast(dict[str, Any], content[0])
        assert block["cache_control"] == {"type": "ephemeral"}


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
