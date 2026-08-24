"""Tests for ``providers.openai_compat``: wire-format conversion + stream parsing."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import ClassVar, cast

import json

import httpx
import pytest

from sagent.lib.custom_json import MutableJSON
from sagent.providers.openai.compat import (
    OpenAICompat,
    OpenAICompatModel,
    _extract_usage,
    build_messages,
    consume_stream,
)
from sagent.types.cost import (
    PriceCatalog,
    PriceCatalogProduct,
    TokenPrice,
)
from sagent.types.model import (
    Limits,
    ModelCapability,
    ModelRequest,
    ModelSpec,
    PromptTooLongError,
    RequestTooLargeError,
    StreamInterruptedError,
)
from sagent.types.runtime import (
    AssistantMessage,
    ModelContextEvent,
    ModelResponsePartial,
    ModelResponseThinking,
    RuntimeEvent,
    ToolCall,
    ToolResult,
    UserMessage,
)


def _free_spec() -> ModelSpec:
    """A spec whose every rate is zero -- cost is not what these assert."""
    return ModelSpec(
        prices=PriceCatalog({PriceCatalogProduct(): TokenPrice()}),
    )


def _priced_spec() -> ModelSpec:
    """$1/$2 per Mtok, with the usual 1.25x cache-write multiplier."""
    return ModelSpec(
        prices=PriceCatalog(
            {
                PriceCatalogProduct(): TokenPrice(
                    request=1.0, response=2.0, cache_write=1.25
                )
            }
        )
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


def test_extract_usage_reports_full_input_and_cache_read_separately() -> None:
    """``_extract_usage`` returns the cache-INCLUSIVE input and cache_read split.

    ``_extract_usage`` does not subtract the cached portion: it returns the raw
    ``prompt_tokens`` plus ``cached_tokens`` as separate values. The disjoint
    split (input minus cache) happens later in ``consume_stream`` -- see
    :func:`test_consume_stream_input_tokens_exclude_cache_read`. Preserves the
    cache-accounting coverage formerly carried by the (now deleted) non-streaming
    ``parse_response`` path.
    """
    usage = cast(
        MutableJSON,
        {
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 400},
        },
    )
    input_tokens, output_tokens, cache_read, cache_write = _extract_usage(usage)
    assert input_tokens == 1000
    assert output_tokens == 100
    assert cache_read == 400
    assert cache_write == 0


@pytest.mark.asyncio
async def test_consume_stream_input_tokens_exclude_cache_read() -> None:
    """``TokenCount.input_tokens`` is the non-cached remainder of the prompt.

    Guards against double-counting the cached portion: OpenAI reports a
    cache-inclusive ``prompt_tokens``, so the stored ``input_tokens`` must drop
    ``cached_tokens`` to stay disjoint from ``cache_read_tokens``.
    """
    events: list[MutableJSON] = [
        {
            "id": "stream-1",
            "choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 10,
                "prompt_tokens_details": {"cached_tokens": 400},
            },
        },
    ]
    resp = await consume_stream(
        _sse_response(events),
        publish=None,
        spec=_free_spec(),
        reasoning_field=None,
    )
    assert resp.tokens.request == 600
    assert resp.tokens.cache_read == 400


@pytest.mark.asyncio
async def test_consume_stream_tracks_and_bills_cache_write_tokens() -> None:
    events: list[MutableJSON] = [
        {
            "id": "stream-cache-write",
            "choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 1309,
                "completion_tokens": 2,
                "prompt_tokens_details": {
                    "cached_tokens": 0,
                    "cache_write_tokens": 1306,
                },
            },
        },
    ]
    resp = await consume_stream(
        _sse_response(events),
        publish=None,
        spec=_priced_spec(),
        reasoning_field=None,
    )
    assert resp.tokens.request == 3
    assert resp.tokens.cache_write == 1306
    assert resp.tokens.cache_read == 0
    assert (
        resp.spend.request + resp.spend.cache_write + resp.spend.cache_read
    ) == pytest.approx((3 + 1306 * 1.25) / 1_000_000)


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

    def _sink(ev: RuntimeEvent) -> None:
        if isinstance(ev, ModelResponsePartial):
            text_acc.append(ev.text)

    r = _sse_response(events)
    resp = await consume_stream(
        r,
        publish=_sink,
        spec=_free_spec(),
        reasoning_field=None,
    )
    assert resp.message.text == "hello"
    assert "".join(text_acc) == "hello"
    assert resp.tokens.request == 4
    assert resp.tokens.response == 2
    assert resp.stop_reason == "model_finished"
    assert resp.message_id == "stream-1"


@pytest.mark.asyncio
async def test_consume_stream_preserves_chat_refusal_text() -> None:
    events: list[MutableJSON] = [
        {
            "id": "stream-refusal",
            "choices": [
                {
                    "delta": {"refusal": "I can’t help with that."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 6},
        },
    ]
    resp = await consume_stream(
        _sse_response(events),
        publish=None,
        spec=_free_spec(),
        reasoning_field=None,
    )
    assert resp.message.text == "I can’t help with that."
    assert resp.stop_reason == "model_refusal"


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
        publish=None,
        spec=_free_spec(),
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

    def _sink(ev: RuntimeEvent) -> None:
        if isinstance(ev, ModelResponseThinking):
            thinking_chunks.append(ev.text)

    resp = await consume_stream(
        _sse_response(events),
        publish=_sink,
        spec=_free_spec(),
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
        publish=None,
        spec=_free_spec(),
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
            publish=None,
            spec=_priced_spec(),
            reasoning_field=None,
        )
    response = exc_info.value.response
    assert response.message.text == "partial"
    assert response.message_id == "stream-1"
    assert response.tokens.request == 4
    assert response.tokens.response == 2
    assert response.total_cost == pytest.approx(0.000008)


def _stub_limits(request: int) -> Limits:
    return Limits(
        max_request_tokens=request,
        max_response_tokens=200,
        max_request_bytes=20 * 1024 * 1024,
        max_image_edge_px=2048,
        max_image_bytes=20 * 1024 * 1024,
    )


class _DummyProvider(OpenAICompat):
    DEFAULT_MODEL: ClassVar[str] = "stub-1"
    ENV_VAR: ClassVar[str] = "DUMMY_PROV_KEY"
    BASE_URL: ClassVar[str] = "https://stub.test/v1"
    CAPABILITIES: ClassVar[Mapping[str, ModelCapability]] = MappingProxyType(
        {
            "stub-1": ModelCapability(
                model_id="stub-1",
                context_limits=MappingProxyType(
                    {"": _stub_limits(1000), "+1m": _stub_limits(1_000_000)}
                ),
                prices=PriceCatalog({PriceCatalogProduct(): TokenPrice()}),
                # Plain chat-completions: no reasoning knob at all.
                supported_thinking_efforts=MappingProxyType({}),
                supported_thinking_budgets=frozenset(),
                supported_thinking_outputs=frozenset(),
            )
        }
    )


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


def test_provider_model_rejects_fast_tag_without_latency_mode() -> None:
    """Compat vendors expose no fast path; a ``+fast`` id fails fast."""
    p = _DummyProvider.from_key("k")
    with pytest.raises(ValueError, match="does not support fast mode"):
        _ = p.model("stub-1+fast")


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

    def _sink(ev: RuntimeEvent) -> None:
        if isinstance(ev, ModelResponsePartial):
            chunks.append(ev.text)

    resp = await model.stream(
        ModelRequest(messages=[UserMessage(text="p")]),
        publish=_sink,
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
    """A 4xx whose body names the token context window is token overflow.

    Includes a 413 with a context-window body: the body disambiguates it
    from the byte wire-limit, so it stays ``PromptTooLongError`` (a larger
    window helps) rather than routing to byte-overflow recovery.
    """

    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, text=message)

    transport = httpx.MockTransport(handle)
    _, model = _make_provider_with_mock(transport)
    with pytest.raises(PromptTooLongError):
        await model.stream(ModelRequest(messages=[UserMessage(text="x")]))


@pytest.mark.asyncio
async def test_stream_413_byte_body_raises_request_too_large() -> None:
    """A 413 with a byte-limit body raises ``RequestTooLargeError``.

    The byte ceiling is fixed across models, so it routes to byte-overflow
    recovery (shed attachment bytes), not token-overflow recovery.
    """

    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(413, text="Request entity too large")

    transport = httpx.MockTransport(handle)
    _, model = _make_provider_with_mock(transport)
    with pytest.raises(RequestTooLargeError):
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


def test_build_body_strips_window_tag_from_wire_model() -> None:
    """A ``+1m`` model sends the base id on the wire; the API rejects the tag."""
    p = _DummyProvider.from_key("k")
    m = p.model("stub-1+1m")
    assert m.model_id == "stub-1+1m"
    assert m.max_request_tokens == 1_000_000
    body = m._build_body(
        ModelRequest(messages=[UserMessage(text="x")]),
        stream=False,
    )
    assert body["model"] == "stub-1"


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
    from sagent.lib.testing.main import test_main

    test_main(__file__)
