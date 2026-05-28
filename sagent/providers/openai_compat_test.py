"""Tests for ``providers.openai_compat``: wire-format conversion + stream parsing."""

from __future__ import annotations

from typing import ClassVar, cast

import json

import httpx
import pytest

from sagent.lib.json import MutableJSON
from sagent.providers.lib.cost import ModelProfile
from sagent.providers.openai_compat import (
    OpenAICompat,
    OpenAICompatModel,
    _is_context_overflow_text,
    build_messages,
    consume_stream,
    parse_response,
)
from sagent.types.model import (
    ModelRequest,
    Pricing,
    PromptTooLongError,
    StreamInterruptedError,
)
from sagent.types.runtime import (
    AssistantMessage,
    ModelContextEvent,
    ToolCall,
    ToolResult,
    UserMessage,
)


def _make_request(
    *, messages: list[ModelContextEvent], system: str | None = None
) -> ModelRequest:
    return ModelRequest(messages=messages, system=system)


def test_build_messages_system_prepended() -> None:
    req = _make_request(
        messages=[UserMessage(text="hi")],
        system="sys",
    )
    msgs = build_messages(req)
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "sys"


def test_build_messages_user_plain_text() -> None:
    req = _make_request(messages=[UserMessage(text="hello")])
    msgs = build_messages(req)
    assert msgs == [{"role": "user", "content": "hello"}]


def test_build_messages_assistant_with_tool_call_remaps_id() -> None:
    asst = AssistantMessage(
        text="thinking...",
        tool_calls=(ToolCall(id="orig-XYZ", name="Bash", args={"cmd": "ls"}),),
    )
    req = _make_request(messages=[asst])
    msgs = build_messages(req)
    assert msgs == [
        {
            "role": "assistant",
            "content": "thinking...",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": "Bash",
                        "arguments": json.dumps({"cmd": "ls"}),
                    },
                }
            ],
        }
    ]


def test_build_messages_assistant_empty_text_drops_content_to_none() -> None:
    asst = AssistantMessage(text="", tool_calls=(ToolCall(id="x", name="N", args={}),))
    msgs = build_messages(_make_request(messages=[asst]))
    assert msgs[0]["content"] is None


def test_build_messages_tool_result_pair_matches_call_id() -> None:
    asst = AssistantMessage(
        tool_calls=(ToolCall(id="orig-1", name="Bash", args={}),),
    )
    tool_res = ToolResult(call_id="orig-1", content="done")
    msgs = build_messages(_make_request(messages=[asst, tool_res]))
    # Last message is the tool response with the remapped id matching the call.
    tool_msg = msgs[-1]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "call_0"
    assert tool_msg["content"] == "done"


def test_build_messages_tool_result_error_prefixes_marker() -> None:
    res = ToolResult(call_id="cid", content="failed", is_error=True)
    msgs = build_messages(_make_request(messages=[res]))
    assert msgs[-1]["content"] == "[Error] failed"


def test_build_messages_empty_history_returns_empty_list() -> None:
    assert build_messages(_make_request(messages=[])) == []


def test_parse_response_text_only() -> None:
    data = cast(
        MutableJSON,
        {
            "id": "msg-1",
            "choices": [
                {
                    "message": {"content": "hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )
    resp = parse_response(data, pricing=Pricing(), reasoning_field=None)
    assert resp.message.text == "hello"
    assert resp.message.tool_calls == ()
    assert resp.tokens.input_tokens == 10
    assert resp.tokens.output_tokens == 5
    assert resp.stop_reason == "model_finished"
    assert resp.message_id == "msg-1"


def test_parse_response_extracts_tool_call() -> None:
    data = cast(
        MutableJSON,
        {
            "id": "msg-2",
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_xyz",
                                "type": "function",
                                "function": {
                                    "name": "Bash",
                                    "arguments": '{"cmd": "ls"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
    )
    resp = parse_response(data, pricing=Pricing(), reasoning_field=None)
    assert len(resp.message.tool_calls) == 1
    call = resp.message.tool_calls[0]
    assert call.id == "call_xyz"
    assert call.name == "Bash"
    assert dict(call.args) == {"cmd": "ls"}
    assert resp.stop_reason == "model_tool_use"


def test_parse_response_reasoning_field_round_trips() -> None:
    data = cast(
        MutableJSON,
        {
            "choices": [
                {
                    "message": {
                        "content": "out",
                        "reasoning_content": "thinking",
                    },
                    "finish_reason": "stop",
                }
            ],
        },
    )
    resp = parse_response(data, pricing=Pricing(), reasoning_field="reasoning_content")
    assert resp.message.text == "out"
    assert len(resp.message.thinking_blocks) == 1
    block = resp.message.thinking_blocks[0]
    assert block["type"] == "reasoning"
    assert block["text"] == "thinking"


def test_parse_response_cache_read_split_out_of_input() -> None:
    data = cast(
        MutableJSON,
        {
            "choices": [
                {"message": {"content": ""}, "finish_reason": "stop"},
            ],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 100,
                "prompt_tokens_details": {"cached_tokens": 400},
            },
        },
    )
    pricing = Pricing(request=1.0, response=2.0, cache_read=0.1)
    resp = parse_response(data, pricing=pricing, reasoning_field=None)
    assert resp.tokens.input_tokens == 1000
    assert resp.tokens.cache_read_tokens == 400
    # (1000-400)*1 + 400*0.1 = 600 + 40 = 640 / 1M = 0.00064.
    assert resp.input_cost == pytest.approx(0.00064)


def test_parse_response_empty_tool_arguments_becomes_empty_dict() -> None:
    data = cast(
        MutableJSON,
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "x",
                                "type": "function",
                                "function": {"name": "N", "arguments": ""},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
    )
    resp = parse_response(data, pricing=Pricing(), reasoning_field=None)
    assert dict(resp.message.tool_calls[0].args) == {}


def test_parse_response_invalid_json_tool_arguments_becomes_empty_dict() -> None:
    data = cast(
        MutableJSON,
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "x",
                                "type": "function",
                                "function": {"name": "N", "arguments": "not json"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
    )
    resp = parse_response(data, pricing=Pricing(), reasoning_field=None)
    assert dict(resp.message.tool_calls[0].args) == {}


def _sse_response(events: list[MutableJSON]) -> httpx.Response:
    """Build an in-memory httpx Response carrying SSE lines."""
    lines = [f"data: {json.dumps(e)}\n\n" for e in events]
    lines.append("data: [DONE]\n\n")
    return _sse_response_body("".join(lines).encode())


def _sse_response_body(body: bytes) -> httpx.Response:
    """Build an in-memory httpx Response carrying raw SSE bytes."""
    return httpx.Response(
        200,
        content=body,
        headers={"Content-Type": "text/event-stream"},
    )


@pytest.mark.asyncio
async def test_consume_stream_text_and_usage() -> None:
    events: list[MutableJSON] = [
        {
            "id": "stream-1",
            "choices": [{"delta": {"content": "he"}, "finish_reason": None}],
        },
        {
            "id": "stream-1",
            "choices": [{"delta": {"content": "llo"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        },
    ]
    text_acc: list[str] = []
    r = _sse_response(events)
    resp = await consume_stream(
        r,
        on_text=text_acc.append,
        on_thinking=None,
        pricing=Pricing(),
        reasoning_field=None,
    )
    assert resp.message.text == "hello"
    assert "".join(text_acc) == "hello"
    assert resp.tokens.input_tokens == 4
    assert resp.tokens.output_tokens == 2
    assert resp.stop_reason == "model_finished"
    assert resp.message_id == "stream-1"


@pytest.mark.asyncio
async def test_consume_stream_tool_call_accumulates() -> None:
    events: list[MutableJSON] = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_a",
                                "function": {"name": "Bash"},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '{"cmd"'}}
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": ': "ls"}'}}
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
    ]
    resp = await consume_stream(
        _sse_response(events),
        on_text=None,
        on_thinking=None,
        pricing=Pricing(),
        reasoning_field=None,
    )
    assert len(resp.message.tool_calls) == 1
    tc = resp.message.tool_calls[0]
    assert tc.id == "call_a"
    assert tc.name == "Bash"
    assert dict(tc.args) == {"cmd": "ls"}
    assert resp.stop_reason == "model_tool_use"


@pytest.mark.asyncio
async def test_consume_stream_reasoning_captured() -> None:
    events: list[MutableJSON] = [
        {
            "choices": [
                {
                    "delta": {"reasoning_content": "think "},
                    "finish_reason": None,
                }
            ],
        },
        {
            "choices": [
                {
                    "delta": {"reasoning_content": "more"},
                    "finish_reason": "stop",
                }
            ],
        },
    ]
    thinking_chunks: list[str] = []
    resp = await consume_stream(
        _sse_response(events),
        on_text=None,
        on_thinking=thinking_chunks.append,
        pricing=Pricing(),
        reasoning_field="reasoning_content",
    )
    assert thinking_chunks == ["think ", "more"]
    assert resp.message.thinking_blocks == (
        {"type": "reasoning", "text": "think more"},
    )


@pytest.mark.asyncio
async def test_consume_stream_skips_malformed_data() -> None:
    body = (
        b"data: not json\n\n"
        b'data: {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}\n\n'
        b"data: [DONE]\n\n"
    )
    resp = await consume_stream(
        _sse_response_body(body),
        on_text=None,
        on_thinking=None,
        pricing=Pricing(),
        reasoning_field=None,
    )
    assert resp.message.text == "ok"


@pytest.mark.asyncio
async def test_consume_stream_eof_without_done_raises_interrupted() -> None:
    body = (
        b'data: {"id": "stream-1", "choices": '
        b'[{"delta": {"content": "partial"}, "finish_reason": "stop"}], '
        b'"usage": {"prompt_tokens": 4, "completion_tokens": 2}}\n\n'
    )
    with pytest.raises(StreamInterruptedError) as exc_info:
        await consume_stream(
            _sse_response_body(body),
            on_text=None,
            on_thinking=None,
            pricing=Pricing(request=1.0, response=2.0),
            reasoning_field=None,
        )
    response = exc_info.value.response
    assert response.message.text == "partial"
    assert response.message_id == "stream-1"
    assert response.tokens.input_tokens == 4
    assert response.tokens.output_tokens == 2
    assert response.total_cost == pytest.approx(0.000008)


class _DummyProvider(OpenAICompat):
    DEFAULT_MODEL: ClassVar[str] = "stub-1"
    ENV_VAR: ClassVar[str] = "DUMMY_PROV_KEY"
    BASE_URL: ClassVar[str] = "https://stub.test/v1"
    KNOWN_MODELS: ClassVar[dict[str, ModelProfile]] = {
        "stub-1": ModelProfile(
            max_request_tokens=1000,
            max_response_tokens=200,
            pricing=Pricing(),
        )
    }


def test_provider_from_env_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DUMMY_PROV_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API key not configured"):
        _DummyProvider.from_env()


def test_provider_from_env_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DUMMY_PROV_KEY", "the-key")
    p = _DummyProvider.from_env()
    assert p.api_key == "the-key"
    assert p.base_url == "https://stub.test/v1"


def test_provider_from_env_requires_env_var_set() -> None:
    """A base class without ENV_VAR raises a clean error."""
    with pytest.raises(RuntimeError, match="has no ENV_VAR"):
        OpenAICompat.from_env()


def test_provider_model_unknown_id_raises() -> None:
    p = _DummyProvider.from_key("k")
    with pytest.raises(ValueError, match="Unknown model"):
        _ = p.model("does-not-exist")


def test_provider_model_default_picks_default_model() -> None:
    p = _DummyProvider.from_key("k")
    m = p.model()
    assert m.model_id == "stub-1"
    assert isinstance(m, OpenAICompatModel)


def test_provider_utility_model_uses_default_when_not_set() -> None:
    p = _DummyProvider.from_key("k")
    m = p.utility_model()
    assert m.model_id == "stub-1"


def test_model_properties_defaults() -> None:
    p = _DummyProvider.from_key("k")
    m = p.model()
    assert m.max_request_tokens == 1000
    assert m.max_response_tokens == 200
    assert m.supports_streaming is True
    assert m.supports_thinking is False  # no reasoning field on the base.
    assert m.supports_effort is False
    assert m.supports_cache_control is False
    assert m.supports_context_management is False
    assert m.supports_persistent_retry is False
    assert m.supports_account_auth is False
    assert m.approx_text_tokens("x" * 12) == 3
    assert m.max_image_dim == 2048
    assert m.max_image_bytes == 20 * 1024 * 1024


@pytest.mark.parametrize(
    "message",
    [
        "context_length_exceeded",
        "Maximum context length blah",
        "Request size exceeds model context window",
        "request size exceeds model context",
        "input too large",
    ],
)
def test_model_is_context_overflow_detection(message: str) -> None:
    p = _DummyProvider.from_key("k")
    m = p.model()
    assert m.is_context_overflow(RuntimeError(message)) is True
    assert m.is_retryable_provider_error(RuntimeError("transient")) is False


@pytest.mark.parametrize(
    "message",
    [
        "other failure",
        "internal error: too long traceback",
        "request took too long",
    ],
)
def test_model_is_context_overflow_rejects_unrelated_errors(message: str) -> None:
    p = _DummyProvider.from_key("k")
    m = p.model()
    assert m.is_context_overflow(RuntimeError(message)) is False


def test_is_context_overflow_text_false_positive_tools_schema_validation() -> None:
    """Tool-schema validation errors mention 'model context' benignly."""
    msg = "Provider rejected: 'model context' field missing in tools schema"
    assert _is_context_overflow_text(msg) is False


def test_is_context_overflow_text_structured_body_canonical_code() -> None:
    """``error.code == 'context_length_exceeded'`` is the canonical signal."""
    body = json.dumps(
        {
            "error": {
                "code": "context_length_exceeded",
                "message": "This model's maximum context length is 128000 tokens.",
            }
        }
    )
    assert _is_context_overflow_text(body) is True


def test_is_context_overflow_text_structured_body_unrelated_code() -> None:
    """Structured error with unrelated code must not classify as overflow."""
    body = json.dumps(
        {
            "error": {
                "code": "invalid_request_error",
                "message": "tools[0].function: 'model context' field missing",
            }
        }
    )
    assert _is_context_overflow_text(body) is False


def test_model_max_request_tokens_override() -> None:
    p = _DummyProvider.from_key("k")
    m = p.model("stub-1", max_request_tokens=500)
    assert m.max_request_tokens == 500


@pytest.mark.asyncio
async def test_actual_text_tokens_falls_back_to_approx_for_unknown_model() -> None:
    """Non-OpenAI model ids that tiktoken doesn't recognize use ``approx_text_tokens``."""
    p = _DummyProvider.from_key("k")
    m = p.model("stub-1")  # tiktoken has no encoding for ``stub-1``.
    assert await m.actual_text_tokens("x" * 12) == m.approx_text_tokens("x" * 12)


@pytest.mark.asyncio
async def test_actual_request_tokens_falls_back_to_approx_for_unknown_model() -> None:
    """Without a tiktoken encoding, ``actual_request_tokens`` returns ``approx``."""
    p = _DummyProvider.from_key("k")
    m = p.model("stub-1")
    req = ModelRequest(messages=[UserMessage(text="hello world")])
    assert await m.actual_request_tokens(req) == m.approx_request_tokens(req)


@pytest.mark.asyncio
async def test_model_close_closes_reusable_http_client() -> None:
    p = _DummyProvider.from_key("k")
    m = p.model("stub-1")
    client = httpx.AsyncClient()
    m._client = client
    await m.close()
    assert client.is_closed
    assert m._client is None


def _make_provider_with_mock(
    handler: httpx.MockTransport,
) -> tuple[_DummyProvider, OpenAICompatModel]:
    p = _DummyProvider.from_key("test-key")
    m = p.model()
    # Inject a pre-built client so ``_get_client`` returns it.
    m._client = httpx.AsyncClient(transport=handler)
    return p, m


@pytest.mark.asyncio
async def test_stream_unrelated_400_propagates_as_http_error() -> None:
    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="malformed body")

    transport = httpx.MockTransport(handle)
    _, model = _make_provider_with_mock(transport)
    req = ModelRequest(messages=[UserMessage(text="x")])
    with pytest.raises(httpx.HTTPStatusError):
        await model.stream(req)


@pytest.mark.asyncio
async def test_stream_parses_sse_via_mock_transport() -> None:
    sse_body = (
        b'data: {"id": "x", "choices": [{"delta": {"content": "hi"}, '
        b'"finish_reason": "stop"}]}\n\n'
        b"data: [DONE]\n\n"
    )

    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse_body,
            headers={"Content-Type": "text/event-stream"},
        )

    transport = httpx.MockTransport(handle)
    _, model = _make_provider_with_mock(transport)
    chunks: list[str] = []
    resp = await model.stream(
        ModelRequest(messages=[UserMessage(text="p")]),
        on_text=chunks.append,
    )
    assert resp.message.text == "hi"
    assert chunks == ["hi"]


@pytest.mark.parametrize(
    ("status_code", "message"),
    [
        (400, "context_length_exceeded"),
        (400, "input too large"),
        (413, "context_length_exceeded"),
        (413, "Request size exceeds model context window"),
    ],
)
@pytest.mark.asyncio
async def test_stream_4xx_context_overflow_raises_prompt_too_long(
    status_code: int,
    message: str,
) -> None:
    """Status code is not the signal: any 4xx with overflow body normalizes."""

    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, text=message)

    transport = httpx.MockTransport(handle)
    _, model = _make_provider_with_mock(transport)
    with pytest.raises(PromptTooLongError):
        await model.stream(ModelRequest(messages=[UserMessage(text="x")]))


@pytest.mark.asyncio
async def test_stream_500_with_overflow_keyword_is_http_error_not_overflow() -> None:
    """5xx server errors are infrastructure, never overflow (stream path)."""

    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal error: too long traceback")

    transport = httpx.MockTransport(handle)
    _, model = _make_provider_with_mock(transport)
    with pytest.raises(httpx.HTTPStatusError):
        await model.stream(ModelRequest(messages=[UserMessage(text="x")]))


def test_build_body_includes_max_tokens_when_set() -> None:
    p = _DummyProvider.from_key("k")
    m = p.model()
    body = m._build_body(
        ModelRequest(messages=[UserMessage(text="x")], max_response_tokens=42),
        stream=False,
    )
    assert body["max_tokens"] == 42


def test_build_body_stream_options_include_usage() -> None:
    p = _DummyProvider.from_key("k")
    m = p.model()
    body = m._build_body(
        ModelRequest(messages=[UserMessage(text="x")]),
        stream=True,
    )
    assert body["stream"] is True
    stream_options = cast(MutableJSON, body["stream_options"])
    assert stream_options["include_usage"] is True


def test_build_body_skips_effort_when_not_supported() -> None:
    p = _DummyProvider.from_key("k")
    m = p.model()
    # Base ``OpenAICompatModel`` has supports_effort=False.
    body = m._build_body(
        ModelRequest(messages=[UserMessage(text="x")], effort="low"),
        stream=False,
    )
    assert "reasoning_effort" not in body


def test_build_body_skips_service_tier_when_not_supported() -> None:
    p = _DummyProvider.from_key("k")
    m = p.model()
    # Base ``OpenAICompatModel`` has valid_service_tiers=().
    body = m._build_body(
        ModelRequest(messages=[UserMessage(text="x")], service_tier="priority"),
        stream=False,
    )
    assert "service_tier" not in body


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
