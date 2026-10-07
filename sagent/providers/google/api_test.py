"""Tests for ``providers.google``: Gemini wire-format + parse."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import logging

import httpx2
import pytest

from sagent.lib.codec import MutablePlainTree, PlainTree, from_plain, loads
from sagent.providers.google.api import (
    Google,
    _build_request,
    _build_response,
    _strip_additional_properties,
)
from sagent.types.capability import (
    ModelCapability,
    ModelSettings,
    ThinkingBudget,
    ThinkingEffort,
    ThinkingOutput,
)
from sagent.types.cost import (
    PriceCatalog,
    PriceKey,
    TokenPrice,
)
from sagent.types.model import (
    ModelRequest,
    PromptTooLongError,
    RequestTooLargeError,
    StreamInterruptedError,
)
from sagent.types.runtime import (
    DETACHED_PLACEHOLDER,
    AssistantMessage,
    ModelContextEvent,
    ModelResponseThinking,
    RuntimeEvent,
    ToolCall,
    ToolResult,
    UserMessage,
)


if TYPE_CHECKING:
    from collections.abc import Mapping


def test_strip_additional_properties_removes_top_level_key() -> None:
    schema: MutablePlainTree = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"a": 1},
    }
    out = cast(dict[str, MutablePlainTree], _strip_additional_properties(schema))
    assert "additionalProperties" not in out
    assert out["type"] == "object"


def test_strip_additional_properties_recurses_into_lists_and_dicts() -> None:
    schema: MutablePlainTree = {
        "type": "object",
        "properties": {
            "nested": {
                "additionalProperties": False,
                "type": "object",
            },
        },
        "items": [{"additionalProperties": False}],
    }
    out = cast(dict[str, MutablePlainTree], _strip_additional_properties(schema))
    nested = cast(
        dict[str, MutablePlainTree],
        cast(dict[str, MutablePlainTree], out["properties"])["nested"],
    )
    assert "additionalProperties" not in nested
    items_list = from_plain(out["items"], list[dict[str, MutablePlainTree]])
    assert "additionalProperties" not in items_list[0]


def test_strip_additional_properties_scalar_passthrough() -> None:
    assert _strip_additional_properties(cast(MutablePlainTree, "x")) == "x"


def _make_request(messages: list[ModelContextEvent], **kw: object) -> ModelRequest:
    # Use explicit kwargs to keep the type checker happy.
    if "system" in kw:
        return ModelRequest(messages=messages, system=str(kw["system"]))
    return ModelRequest(messages=messages)


def _thinking_capability() -> ModelCapability:
    """Return the catalog row every wire test builds against."""
    return Google.from_key("k").model("gemini-2.5-pro").capability


# Bound rather than bare: every axis validates on assignment, so a default-capability
# object cannot hold the thinking selections these tests are about.
def _settings(
    *,
    thinking_effort: ThinkingEffort = "none",
    thinking_budget: ThinkingBudget = "none",
    thinking_output: ThinkingOutput = "none",
) -> ModelSettings:
    """Return settings bound to the thinking-capable row."""
    return ModelSettings(
        capability=_thinking_capability(),
        thinking_effort=thinking_effort,
        thinking_budget=thinking_budget,
        thinking_output=thinking_output,
    )


def _wire(
    request: ModelRequest,
    settings: ModelSettings | None = None,
) -> dict[str, object]:
    """Build the wire body against a thinking-capable catalog row."""
    capability = _thinking_capability()
    chosen = settings or ModelSettings.narrowest(capability)
    return dict(_build_request(request, capability, chosen, chosen.limits))


def test_build_request_user_message_text_part() -> None:
    body = _wire(_make_request([UserMessage(text="hi")]))
    contents = from_plain(body["contents"], list[dict[str, MutablePlainTree]])
    assert contents == [{"role": "user", "parts": [{"text": "hi"}]}]


def test_build_request_assistant_emits_function_call() -> None:
    asst = AssistantMessage(
        text="thinking",
        tool_calls=(ToolCall(id="ext-1", name="Bash", args={"cmd": "ls"}),),
    )
    body = _wire(_make_request([asst]))
    contents = from_plain(body["contents"], list[dict[str, MutablePlainTree]])
    assert contents[0]["role"] == "model"
    parts = from_plain(contents[0]["parts"], list[dict[str, MutablePlainTree]])
    assert parts[0] == {"text": "thinking"}
    fc_part = parts[1]
    assert "functionCall" in fc_part
    fc = cast(dict[str, MutablePlainTree], fc_part["functionCall"])
    assert fc["name"] == "Bash"
    assert fc["args"] == {"cmd": "ls"}


def test_build_request_tool_result_emits_function_response_with_name() -> None:
    asst = AssistantMessage(
        tool_calls=(ToolCall(id="ext-1", name="MyTool", args={}),),
    )
    res = ToolResult(call_id="ext-1", content="done")
    body = _wire(_make_request([asst, res]))
    contents = from_plain(body["contents"], list[dict[str, MutablePlainTree]])
    # Last content is the user message holding the functionResponse.
    user_msg = contents[-1]
    assert user_msg["role"] == "user"
    parts = from_plain(user_msg["parts"], list[dict[str, MutablePlainTree]])
    fr_part = parts[0]
    fr = cast(dict[str, MutablePlainTree], fr_part["functionResponse"])
    # Name comes from the prior tool_call binding, not the call_id.
    assert fr["name"] == "MyTool"
    response = cast(dict[str, MutablePlainTree], fr["response"])
    assert response["content"] == "done"


def test_user_after_tool_results_coalesces_into_same_wire_content() -> None:
    """C1: mid-cohort user text MUST merge with pending functionResponse parts.

    Emitting a separate ``role=user`` content after a ``role=user``
    holding functionResponse parts breaks Gemini's user/model
    alternation requirement. The fix coalesces both into the same wire
    content.
    """
    asst = AssistantMessage(
        tool_calls=(ToolCall(id="c1", name="MyTool", args={}),),
    )
    tool_result = ToolResult(call_id="c1", content=DETACHED_PLACEHOLDER)
    user_redirect = UserMessage(text="actually do something else")
    body = _wire(_make_request([asst, tool_result, user_redirect]))
    contents = from_plain(body["contents"], list[dict[str, MutablePlainTree]])
    # 2 contents: model(functionCall) + user(functionResponse + text).
    assert len(contents) == 2, [c["role"] for c in contents]
    assert contents[0]["role"] == "model"
    assert contents[1]["role"] == "user"
    parts = from_plain(contents[1]["parts"], list[dict[str, MutablePlainTree]])
    keys = {k for p in parts for k in p}
    assert "functionResponse" in keys
    assert "text" in keys


def test_build_request_tool_result_error_prefix() -> None:
    asst = AssistantMessage(tool_calls=(ToolCall(id="x", name="N", args={}),))
    res = ToolResult(call_id="x", content="boom", is_error=True)
    body = _wire(_make_request([asst, res]))
    contents = from_plain(body["contents"], list[dict[str, MutablePlainTree]])
    parts = from_plain(contents[-1]["parts"], list[dict[str, MutablePlainTree]])
    fr = cast(dict[str, MutablePlainTree], parts[0]["functionResponse"])
    assert (
        cast(dict[str, MutablePlainTree], fr["response"])["content"] == "[Error] boom"
    )


def test_build_request_system_instruction() -> None:
    body = _wire(_make_request([UserMessage(text="hi")], system="be terse"))
    sys_inst = cast(dict[str, MutablePlainTree], body["systemInstruction"])
    parts = from_plain(sys_inst["parts"], list[dict[str, MutablePlainTree]])
    assert parts == [{"text": "be terse"}]


def test_build_request_empty_user_emits_placeholder() -> None:
    body = _wire(_make_request([UserMessage(text="")]))
    contents = from_plain(body["contents"], list[dict[str, MutablePlainTree]])
    parts = from_plain(contents[0]["parts"], list[dict[str, MutablePlainTree]])
    assert parts == [{"text": ""}]


@pytest.mark.asyncio
async def test_google_stream_parses_text_tool_call_and_finish_reason() -> None:
    """Stream consumer extracts text, function calls, finish reason."""
    sse_body = (
        b'data: {"candidates":[{"content":{"parts":[{"text":"calling"},'
        b'{"functionCall":{"name":"Bash","args":{"cmd":"ls"}}}],'
        b'"role":"model"},"finishReason":"STOP"}],'
        b'"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5}}\n\n'
    )

    def handle(request: httpx2.Request) -> httpx2.Response:
        del request
        return httpx2.Response(
            200,
            content=sse_body,
            headers={"Content-Type": "text/event-stream"},
        )

    transport = httpx2.MockTransport(handle)
    p = Google.from_key("k")
    m = p.model("gemini-2.5-flash")
    m._client = httpx2.AsyncClient(transport=transport)
    resp = await m.stream(ModelRequest(messages=[UserMessage(text="x")]))
    assert resp.message.text == "calling"
    assert len(resp.message.tool_calls) == 1
    tc = resp.message.tool_calls[0]
    assert tc.name == "Bash"
    assert dict(tc.args) == {"cmd": "ls"}
    # STOP + tool calls → upgraded to model_tool_use.
    assert resp.stop_reason == "model_tool_use"
    assert resp.tokens.request == 10
    assert resp.tokens.response == 5
    assert resp.message_id.startswith("gemini_")
    assert resp.request_id == resp.message_id


@pytest.mark.asyncio
async def test_google_stream_routes_thought_parts_to_thinking() -> None:
    sse_body = (
        b'data: {"candidates":[{"content":{"parts":[{"text":"thinking",'
        b'"thought":true},{"text":"answer"}]},"finishReason":"STOP"}],'
        b'"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5}}\n\n'
    )
    thinking_chunks: list[str] = []

    def _sink(ev: RuntimeEvent) -> None:
        if isinstance(ev, ModelResponseThinking):
            thinking_chunks.append(ev.text)

    def handle(request: httpx2.Request) -> httpx2.Response:
        del request
        return httpx2.Response(
            200,
            content=sse_body,
            headers={"Content-Type": "text/event-stream"},
        )

    transport = httpx2.MockTransport(handle)
    p = Google.from_key("k")
    m = p.model("gemini-2.5-flash")
    m._client = httpx2.AsyncClient(transport=transport)
    resp = await m.stream(
        ModelRequest(messages=[UserMessage(text="x")]),
        publish=_sink,
    )
    assert thinking_chunks == ["thinking"]
    assert resp.message.text == "answer"
    assert resp.message.thinking_blocks == (
        {"type": "thinking", "thinking": "thinking"},
    )


@pytest.mark.asyncio
async def test_google_stream_logs_and_skips_malformed_json_chunk(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sse_body = (
        b"data: {not-json}\n\n"
        b'data: {"candidates":[{"content":{"parts":[{"text":"ok"}]}}]}\n\n'
    )

    def handle(request: httpx2.Request) -> httpx2.Response:
        del request
        return httpx2.Response(
            200,
            content=sse_body,
            headers={"Content-Type": "text/event-stream"},
        )

    transport = httpx2.MockTransport(handle)
    p = Google.from_key("k")
    m = p.model("gemini-2.5-flash")
    m._client = httpx2.AsyncClient(transport=transport)
    with (
        caplog.at_level(
            logging.WARNING,
            logger="sagent.providers.google.api",
        ),
        pytest.raises(StreamInterruptedError) as raised,
    ):
        await m.stream(ModelRequest(messages=[UserMessage(text="x")]))
    assert raised.value.response.message.text == "ok"
    assert any("malformed JSON chunk" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_google_stream_eof_without_finish_reason_raises_interrupted() -> None:
    sse_body = b'data: {"candidates":[{"content":{"parts":[{"text":"partial"}]}}]}\n\n'

    def handle(request: httpx2.Request) -> httpx2.Response:
        del request
        return httpx2.Response(
            200,
            content=sse_body,
            headers={"Content-Type": "text/event-stream"},
        )

    transport = httpx2.MockTransport(handle)
    p = Google.from_key("k")
    m = p.model("gemini-2.5-flash")
    m._client = httpx2.AsyncClient(transport=transport)
    with pytest.raises(StreamInterruptedError) as raised:
        await m.stream(ModelRequest(messages=[UserMessage(text="x")]))
    assert raised.value.response.message.text == "partial"


@pytest.mark.asyncio
async def test_google_stream_raises_when_all_json_chunks_are_malformed() -> None:
    sse_body = b"data: {not-json}\n\ndata: also-not-json\n\n"

    def handle(request: httpx2.Request) -> httpx2.Response:
        del request
        return httpx2.Response(
            200,
            content=sse_body,
            headers={"Content-Type": "text/event-stream"},
        )

    transport = httpx2.MockTransport(handle)
    p = Google.from_key("k")
    m = p.model("gemini-2.5-flash")
    m._client = httpx2.AsyncClient(transport=transport)
    with pytest.raises(ValueError, match="only malformed JSON chunks"):
        await m.stream(ModelRequest(messages=[UserMessage(text="x")]))


@pytest.mark.asyncio
async def test_google_stream_max_tokens_finish_reason() -> None:
    """``MAX_TOKENS`` finish reason normalizes to ``max_tokens``."""
    sse_body = (
        b'data: {"candidates":[{"content":{"parts":[{"text":"trunc"}]},'
        b'"finishReason":"MAX_TOKENS"}]}\n\n'
    )

    def handle(request: httpx2.Request) -> httpx2.Response:
        del request
        return httpx2.Response(
            200,
            content=sse_body,
            headers={"Content-Type": "text/event-stream"},
        )

    transport = httpx2.MockTransport(handle)
    p = Google.from_key("k")
    m = p.model("gemini-2.5-flash")
    m._client = httpx2.AsyncClient(transport=transport)
    resp = await m.stream(ModelRequest(messages=[UserMessage(text="x")]))
    assert resp.stop_reason == "max_tokens"


def test_google_build_response_cache_tokens_split_input_cost() -> None:
    """``cache_read`` is subtracted from input before the request-rate charge."""
    model = Google.from_key("k").model("gemini-2.5-pro")
    model._capability = replace(
        model.capability,
        prices=PriceCatalog(
            {
                PriceKey("auto"): TokenPrice(
                    request=1.0,
                    response=2.0,
                    cache_read=0.5,
                    cache_write=0.0,
                    cache_write_1h=0.0,
                ),
            },
        ),
    )
    usage: dict[str, MutablePlainTree] = {
        "promptTokenCount": 1000,
        "candidatesTokenCount": 100,
        "cachedContentTokenCount": 300,
    }
    resp = _build_response(
        text="",
        tool_calls=[],
        usage=usage,
        finish_reason="STOP",
        model=model,
    )
    assert resp.tokens.cache_read == 300
    # ``promptTokenCount`` is cache-inclusive; stored input drops the cached
    # portion so it stays disjoint from ``cache_read_tokens``.
    assert resp.tokens.request == 700
    # (1000-300)*1 + 300*0.5 = 850 / 1M = 0.00085.
    assert (
        resp.spend.request + resp.spend.cache_write + resp.spend.cache_read
    ) == pytest.approx(0.00085)


def test_google_from_key() -> None:
    p = Google.from_key("AIza-key")
    assert p.api_key == "AIza-key"


def test_google_from_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API key not configured"):
        Google.from_env()


def test_google_from_env_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "AIza-test")
    p = Google.from_env()
    assert p.api_key == "AIza-test"


def test_google_model_unknown_raises() -> None:
    p = Google.from_key("k")
    with pytest.raises(ValueError, match="Unknown model"):
        _ = p.model("not-a-gemini")


def test_google_model_default_uses_default_model() -> None:
    p = Google.from_key("k")
    m = p.model()
    assert m.capability.model_id == p.catalog.resolve("default")[0].model_id


def test_google_utility_model_uses_flash_lite() -> None:
    """Utility resolves to the Flash-Lite Google names for new projects."""
    p = Google.from_key("k")
    m = p.model("utility")
    assert m.capability.model_id == "gemini-flash-lite-3.5"


def test_google_model_properties() -> None:
    p = Google.from_key("k")
    m = p.model("gemini-2.5-pro")
    assert m.limits.max_request_tokens == 1_048_576
    assert m.capability.thinking.budget != frozenset({"none"})
    assert m.capability.thinking.effort != frozenset({"none"})
    assert m.capability.cache_ttl_sec == frozenset({0.0})
    # Gemini publishes no per-image pixel or byte cap (images are tiled
    # server-side); the only documented limit is the 100 MB total request size.
    assert m.limits.max_image_edge_px == 0
    assert m.limits.max_image_bytes == 0
    assert m.limits.max_request_bytes == 100 * 1024 * 1024


def _level_wire(model_id: str, effort: ThinkingEffort) -> object:
    """Return the ``thinkingConfig`` a fixed ``effort`` sends on ``model_id``."""
    capability = Google.from_key("k").model(model_id).capability
    settings = ModelSettings(
        capability=capability,
        thinking_effort=effort,
        thinking_budget="fixed",
        thinking_output="text",
    )
    body = _build_request(
        ModelRequest(messages=[UserMessage(text="x")]),
        capability,
        settings,
        settings.limits,
    )
    return cast(dict[str, MutablePlainTree], body["generationConfig"]).get(
        "thinkingConfig",
    )


@pytest.mark.parametrize(
    ("effort", "level"),
    [("min", "minimal"), ("low", "low"), ("medium", "medium"), ("high", "high")],
)
def test_gemini_3_sends_a_thinking_level_not_a_budget(
    effort: ThinkingEffort,
    level: str,
) -> None:
    """Gemini 3 takes ``thinkingLevel``; ``thinkingBudget`` is backward-compat only."""
    assert _level_wire("gemini-3.6-flash", effort) == {
        "includeThoughts": True,
        "thinkingLevel": level,
    }


def test_gemini_3_effort_none_sends_no_level() -> None:
    """``none`` lets the model's default level apply; it never asks for "off"."""
    assert _level_wire("gemini-3.1-pro-preview", "none") == {"includeThoughts": True}


def test_gemini_3_8_cannot_select_the_minimal_level() -> None:
    """3.8 Flash answers ``minimal`` with an error, so the row withholds it."""
    with pytest.raises(ValueError, match="thinking_effort='min'"):
        _level_wire("gemini-3.8-flash", "min")


def test_gemini_2_5_fixed_effort_none_sends_no_budget() -> None:
    """No effort, no cap: 2.5 then thinks at its own default."""
    assert _level_wire("gemini-2.5-flash", "none") == {"includeThoughts": True}


def test_build_request_adaptive_thinking_uses_dynamic_budget() -> None:
    body = _wire(
        ModelRequest(messages=[UserMessage(text="x")]),
        _settings(thinking_budget="auto", thinking_output="text"),
    )
    gen_config = cast(dict[str, MutablePlainTree], body["generationConfig"])
    assert gen_config["thinkingConfig"] == {
        "includeThoughts": True,
        "thinkingBudget": -1,
    }


def test_build_request_fixed_thinking_uses_the_effort_budget() -> None:
    body = _wire(
        ModelRequest(messages=[UserMessage(text="x")]),
        _settings(
            thinking_budget="fixed",
            thinking_effort="max",
            thinking_output="text",
        ),
    )
    gen_config = cast(dict[str, MutablePlainTree], body["generationConfig"])
    assert gen_config["thinkingConfig"] == {
        "includeThoughts": True,
        "thinkingBudget": 24_576,
    }


def test_build_request_effort_min_sets_small_budget() -> None:
    body = _wire(
        ModelRequest(messages=[UserMessage(text="x")]),
        _settings(
            thinking_budget="fixed",
            thinking_effort="min",
            thinking_output="text",
        ),
    )
    gen_config = cast(dict[str, MutablePlainTree], body["generationConfig"])
    assert gen_config["thinkingConfig"] == {
        "includeThoughts": True,
        "thinkingBudget": 1_024,
    }


def test_build_request_effort_max_sets_largest_budget() -> None:
    body = _wire(
        ModelRequest(messages=[UserMessage(text="x")]),
        _settings(
            thinking_budget="fixed",
            thinking_effort="max",
            thinking_output="text",
        ),
    )
    gen_config = cast(dict[str, MutablePlainTree], body["generationConfig"])
    assert gen_config["thinkingConfig"] == {
        "includeThoughts": True,
        "thinkingBudget": 24_576,
    }


def test_build_request_thinking_off_sends_no_config() -> None:
    """Gemini rejects ``thinkingConfig`` outright when thinking is off."""
    body = _wire(ModelRequest(messages=[UserMessage(text="x")]))
    assert "thinkingConfig" not in cast(
        dict[str, MutablePlainTree],
        body["generationConfig"],
    )


def test_build_request_thinking_omits_temperature() -> None:
    body = _wire(
        ModelRequest(messages=[UserMessage(text="x")]),
        _settings(thinking_budget="auto", thinking_output="text"),
    )
    gen_config = cast(dict[str, MutablePlainTree], body["generationConfig"])
    assert "thinkingConfig" in gen_config
    assert "temperature" not in gen_config


def test_build_request_without_thinking_keeps_temperature() -> None:
    body = _wire(ModelRequest(messages=[UserMessage(text="x")], temperature=0.3))
    gen_config = cast(dict[str, MutablePlainTree], body["generationConfig"])
    assert gen_config["temperature"] == 0.3


@pytest.mark.asyncio
async def test_google_model_close_closes_reusable_http_client() -> None:
    p = Google.from_key("k")
    m = p.model("gemini-2.5-flash")
    client = httpx2.AsyncClient()
    m._client = client
    await m.close()
    assert client.is_closed
    assert m._client is None


def test_google_model_context_overflow_detection() -> None:
    p = Google.from_key("k")
    m = p.model("gemini-2.5-pro")
    assert m.is_context_overflow(RuntimeError("Input too large for the model"))
    assert m.is_context_overflow(RuntimeError("exceeds the maximum context"))
    assert not m.is_context_overflow(RuntimeError("other error"))


def test_google_model_is_retryable_provider_error_false() -> None:
    p = Google.from_key("k")
    m = p.model("gemini-2.5-pro")
    assert not m.is_retryable_provider_error(RuntimeError("anything"))


def test_google_model_text_token_estimate_floor_division() -> None:
    p = Google.from_key("k")
    m = p.model("gemini-2.5-pro")
    assert m.approx_text_tokens("a" * 16) == 4


@pytest.mark.asyncio
async def test_google_stream_413_token_body_raises_prompt_too_long() -> None:
    """A 413 whose body names the context window is token overflow, not bytes.

    Gemini reuses 413 for token-context overflow; the body ("model
    context") disambiguates, so it stays ``PromptTooLongError`` (a larger
    window helps) rather than routing to byte-overflow recovery.
    """

    def handle(request: httpx2.Request) -> httpx2.Response:
        del request
        return httpx2.Response(413, text="Input too large for model context.")

    transport = httpx2.MockTransport(handle)
    p = Google.from_key("k")
    m = p.model("gemini-2.5-flash")
    m._client = httpx2.AsyncClient(transport=transport)
    with pytest.raises(PromptTooLongError):
        await m.stream(ModelRequest(messages=[UserMessage(text="x")]))


@pytest.mark.asyncio
async def test_google_stream_413_byte_body_raises_request_too_large() -> None:
    """A 413 with a byte-limit body raises ``RequestTooLargeError``.

    The byte ceiling is fixed across models, so it must route to
    byte-overflow recovery (shed attachment bytes), not token-overflow.
    """

    def handle(request: httpx2.Request) -> httpx2.Response:
        del request
        return httpx2.Response(413, text="Request entity too large.")

    transport = httpx2.MockTransport(handle)
    p = Google.from_key("k")
    m = p.model("gemini-2.5-flash")
    m._client = httpx2.AsyncClient(transport=transport)
    with pytest.raises(RequestTooLargeError):
        await m.stream(ModelRequest(messages=[UserMessage(text="x")]))


@pytest.mark.asyncio
async def test_google_stream_400_exceeds_maximum_normalizes() -> None:
    """The ``exceeds the maximum`` substring is the canonical Gemini overflow phrase."""

    def handle(request: httpx2.Request) -> httpx2.Response:
        del request
        return httpx2.Response(
            400,
            text="The input token count exceeds the maximum allowed.",
        )

    transport = httpx2.MockTransport(handle)
    p = Google.from_key("k")
    m = p.model("gemini-2.5-flash")
    m._client = httpx2.AsyncClient(transport=transport)
    with pytest.raises(PromptTooLongError):
        await m.stream(ModelRequest(messages=[UserMessage(text="x")]))


@pytest.mark.asyncio
async def test_google_stream_400_too_long_raises_prompt_too_long() -> None:
    """Stream path normalizes 4xx with overflow body to ``PromptTooLongError``."""
    sse_body = b'data: {"candidates":[{"content":{"parts":[{"text":"hi"}]}}]}\n\n'

    def handle(request: httpx2.Request) -> httpx2.Response:
        del request
        return httpx2.Response(400, text="Input too long for model context.")

    transport = httpx2.MockTransport(handle)
    p = Google.from_key("k")
    m = p.model("gemini-2.5-flash")
    m._client = httpx2.AsyncClient(transport=transport)
    del sse_body  # Only error path exercised.
    with pytest.raises(PromptTooLongError):
        await m.stream(ModelRequest(messages=[UserMessage(text="x")]))


@pytest.mark.asyncio
async def test_google_stream_500_with_overflow_keyword_is_http_error_not_overflow() -> (
    None
):
    """Stream 5xx with overflow keywords propagates as HTTPStatusError."""

    def handle(request: httpx2.Request) -> httpx2.Response:
        del request
        return httpx2.Response(500, text="internal error: too long traceback")

    transport = httpx2.MockTransport(handle)
    p = Google.from_key("k")
    m = p.model("gemini-2.5-flash")
    m._client = httpx2.AsyncClient(transport=transport)
    with pytest.raises(httpx2.HTTPStatusError):
        await m.stream(ModelRequest(messages=[UserMessage(text="x")]))


@pytest.mark.asyncio
async def test_google_actual_request_tokens_wraps_generate_request() -> None:
    """``:countTokens`` takes the generate body under ``generateContentRequest``."""
    seen: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json={"totalTokens": 314})

    m = Google.from_key("k").model("gemini-2.5-flash")
    m._client = httpx2.AsyncClient(transport=httpx2.MockTransport(handle))
    n = await m.actual_request_tokens(
        ModelRequest(messages=[UserMessage(text="ping")], system="sys"),
    )
    assert n == 314
    assert seen[0].url.path.endswith(":countTokens")
    body = from_plain(loads(seen[0].content), dict[str, object])
    assert set(body) == {"generateContentRequest"}
    inner = from_plain(body["generateContentRequest"], dict[str, object])
    model_id = m.capability.wire_model_id or m.capability.model_id
    assert inner["model"] == f"models/{model_id}"
    assert "systemInstruction" in inner


@pytest.mark.asyncio
async def test_google_stream_400_other_raises_http_status_error() -> None:
    def handle(request: httpx2.Request) -> httpx2.Response:
        del request
        return httpx2.Response(400, text="malformed request body")

    m = Google.from_key("k").model("gemini-2.5-flash")
    m._client = httpx2.AsyncClient(transport=httpx2.MockTransport(handle))
    with pytest.raises(httpx2.HTTPStatusError) as raised:
        await m.stream(ModelRequest(messages=[UserMessage(text="x")]))
    assert raised.value.response.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["stream", "count"])
async def test_google_tool_schema_too_long_is_not_overflow(path: str) -> None:
    """A schema validation 400 mentioning "too long" is not a context overflow."""

    def handle(request: httpx2.Request) -> httpx2.Response:
        del request
        return httpx2.Response(400, text="Function name is too long.")

    m = Google.from_key("k").model("gemini-2.5-flash")
    m._client = httpx2.AsyncClient(transport=httpx2.MockTransport(handle))
    request = ModelRequest(messages=[UserMessage(text="x")])
    call = m.stream if path == "stream" else m.actual_request_tokens
    with pytest.raises(httpx2.HTTPStatusError):
        await call(request)


class _StubTool:
    name: str = "Echo"
    tool_id: str = "application/x-tool-echo"
    description: str = "Echo"
    directive_schema: Mapping[str, PlainTree] = {
        "type": "object",
        "additionalProperties": False,
    }
    clearable_results: bool = False

    def summary(self, args: Mapping[str, object]) -> str:
        del args
        return ""

    def prompt(self) -> str | None:
        return None

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        del args
        return None

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        del args
        return ToolResult(call_id="", content="")


def test_build_request_tools_strip_additional_properties() -> None:
    """`additionalProperties` is recursively stripped from tool schemas."""
    tool = _StubTool()
    req = ModelRequest(
        messages=[UserMessage(text="hi")],
        tools=[tool],
    )
    body = _wire(req)
    tools_section = from_plain(body["tools"], list[dict[str, MutablePlainTree]])
    fns = from_plain(
        tools_section[0]["functionDeclarations"],
        list[dict[str, MutablePlainTree]],
    )
    schema = from_plain(fns[0]["parameters"], dict[str, MutablePlainTree])
    assert "additionalProperties" not in schema


def test_build_request_echoes_thought_signature() -> None:
    """Gemini 3.x requires the model's thought signature echoed back on its
    parts; the text part and each functionCall part carry their own.
    """
    asst = AssistantMessage(
        text="answer",
        thought_signature="sig-text",
        tool_calls=(
            ToolCall(
                id="ext-1",
                name="Bash",
                args={"cmd": "ls"},
                thought_signature="sig-fc",
            ),
        ),
    )
    body = _wire(_make_request([asst]))
    parts = from_plain(
        from_plain(body["contents"], list[dict[str, MutablePlainTree]])[0]["parts"],
        list[dict[str, MutablePlainTree]],
    )
    assert parts[0] == {"text": "answer", "thoughtSignature": "sig-text"}
    assert parts[1]["thoughtSignature"] == "sig-fc"


def test_build_request_omits_empty_thought_signature() -> None:
    """No signature (older models / thinking off) -> no thoughtSignature key."""
    asst = AssistantMessage(
        text="hi",
        tool_calls=(ToolCall(id="e", name="Bash", args={}),),
    )
    body = _wire(_make_request([asst]))
    parts = from_plain(
        from_plain(body["contents"], list[dict[str, MutablePlainTree]])[0]["parts"],
        list[dict[str, MutablePlainTree]],
    )
    assert parts[0] == {"text": "hi"}
    assert "thoughtSignature" not in parts[1]


def test_build_request_preserves_per_call_signature_order() -> None:
    """Each functionCall part keeps its own signature in call order; a future
    refactor that reorders or shares parts would break the signature chain.
    """
    asst = AssistantMessage(
        text="answer",
        thought_signature="sig-text",
        tool_calls=(
            ToolCall(id="a", name="Bash", args={"i": 1}, thought_signature="sig-a"),
            ToolCall(id="b", name="Bash", args={"i": 2}, thought_signature="sig-b"),
        ),
    )
    body = _wire(_make_request([asst]))
    parts = from_plain(
        from_plain(body["contents"], list[dict[str, MutablePlainTree]])[0]["parts"],
        list[dict[str, MutablePlainTree]],
    )
    assert cast(dict[str, MutablePlainTree], parts[1]["functionCall"])["args"] == {
        "i": 1,
    }
    assert parts[1]["thoughtSignature"] == "sig-a"
    assert cast(dict[str, MutablePlainTree], parts[2]["functionCall"])["args"] == {
        "i": 2,
    }
    assert parts[2]["thoughtSignature"] == "sig-b"


def test_build_request_tool_result_carries_no_signature() -> None:
    """Signatures attach to model-role parts only; a functionResponse (tool
    result) must never emit thoughtSignature.
    """
    asst = AssistantMessage(
        text="answer",
        thought_signature="sig-text",
        tool_calls=(
            ToolCall(id="c1", name="Bash", args={}, thought_signature="sig-fc"),
        ),
    )
    res = ToolResult(call_id="c1", content="done")
    body = _wire(_make_request([asst, res]))
    contents = from_plain(body["contents"], list[dict[str, MutablePlainTree]])
    user_parts = from_plain(contents[1]["parts"], list[dict[str, MutablePlainTree]])
    assert "functionResponse" in user_parts[0]
    assert "thoughtSignature" not in user_parts[0]


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
