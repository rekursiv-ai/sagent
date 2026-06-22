"""Tests for ``providers.google``: Gemini wire-format + parse."""

from __future__ import annotations

from typing import cast

import logging

import httpx
import pytest

from sagent.lib.custom_json import MutableJSON, MutableJSONValue
from sagent.providers.google import (
    Google,
    _build_request,
    _build_response,
    _strip_additional_properties,
)
from sagent.types.model import (
    ModelRequest,
    Pricing,
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
from sagent.types.tools import Tool


def test_strip_additional_properties_removes_top_level_key() -> None:
    schema = cast(
        MutableJSONValue,
        {"type": "object", "additionalProperties": False, "properties": {"a": 1}},
    )
    out = cast(MutableJSON, _strip_additional_properties(schema))
    assert "additionalProperties" not in out
    assert out["type"] == "object"


def test_strip_additional_properties_recurses_into_lists_and_dicts() -> None:
    schema = cast(
        MutableJSONValue,
        {
            "type": "object",
            "properties": {
                "nested": {
                    "additionalProperties": False,
                    "type": "object",
                }
            },
            "items": [{"additionalProperties": False}],
        },
    )
    out = cast(MutableJSON, _strip_additional_properties(schema))
    nested = cast(MutableJSON, cast(MutableJSON, out["properties"])["nested"])
    assert "additionalProperties" not in nested
    items_list = cast(list[MutableJSON], out["items"])
    assert "additionalProperties" not in items_list[0]


def test_strip_additional_properties_scalar_passthrough() -> None:
    assert _strip_additional_properties(cast(MutableJSONValue, "x")) == "x"


def _make_request(messages: list[ModelContextEvent], **kw: object) -> ModelRequest:
    # Use explicit kwargs to keep the type checker happy.
    if "system" in kw:
        return ModelRequest(messages=messages, system=cast(str, kw["system"]))
    return ModelRequest(messages=messages)


def test_build_request_user_message_text_part() -> None:
    body = _build_request(_make_request([UserMessage(text="hi")]))
    contents = cast(list[MutableJSON], body["contents"])
    assert contents == [{"role": "user", "parts": [{"text": "hi"}]}]


def test_build_request_assistant_emits_function_call() -> None:
    asst = AssistantMessage(
        text="thinking",
        tool_calls=(ToolCall(id="ext-1", name="Bash", args={"cmd": "ls"}),),
    )
    body = _build_request(_make_request([asst]))
    contents = cast(list[MutableJSON], body["contents"])
    assert contents[0]["role"] == "model"
    parts = cast(list[MutableJSON], contents[0]["parts"])
    assert parts[0] == {"text": "thinking"}
    fc_part = parts[1]
    assert "functionCall" in fc_part
    fc = cast(MutableJSON, fc_part["functionCall"])
    assert fc["name"] == "Bash"
    assert fc["args"] == {"cmd": "ls"}


def test_build_request_tool_result_emits_function_response_with_name() -> None:
    asst = AssistantMessage(
        tool_calls=(ToolCall(id="ext-1", name="MyTool", args={}),),
    )
    res = ToolResult(call_id="ext-1", content="done")
    body = _build_request(_make_request([asst, res]))
    contents = cast(list[MutableJSON], body["contents"])
    # Last content is the user message holding the functionResponse.
    user_msg = contents[-1]
    assert user_msg["role"] == "user"
    parts = cast(list[MutableJSON], user_msg["parts"])
    fr_part = parts[0]
    fr = cast(MutableJSON, fr_part["functionResponse"])
    # Name comes from the prior tool_call binding, not the call_id.
    assert fr["name"] == "MyTool"
    response = cast(MutableJSON, fr["response"])
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
    body = _build_request(_make_request([asst, tool_result, user_redirect]))
    contents = cast(list[MutableJSON], body["contents"])
    # 2 contents: model(functionCall) + user(functionResponse + text).
    assert len(contents) == 2, [c["role"] for c in contents]
    assert contents[0]["role"] == "model"
    assert contents[1]["role"] == "user"
    parts = cast(list[MutableJSON], contents[1]["parts"])
    keys = {k for p in parts for k in p}
    assert "functionResponse" in keys
    assert "text" in keys


def test_build_request_tool_result_error_prefix() -> None:
    asst = AssistantMessage(tool_calls=(ToolCall(id="x", name="N", args={}),))
    res = ToolResult(call_id="x", content="boom", is_error=True)
    body = _build_request(_make_request([asst, res]))
    contents = cast(list[MutableJSON], body["contents"])
    parts = cast(list[MutableJSON], contents[-1]["parts"])
    fr = cast(MutableJSON, parts[0]["functionResponse"])
    assert cast(MutableJSON, fr["response"])["content"] == "[Error] boom"


def test_build_request_system_instruction() -> None:
    body = _build_request(_make_request([UserMessage(text="hi")], system="be terse"))
    sys_inst = cast(MutableJSON, body["systemInstruction"])
    parts = cast(list[MutableJSON], sys_inst["parts"])
    assert parts == [{"text": "be terse"}]


def test_build_request_empty_user_emits_placeholder() -> None:
    body = _build_request(_make_request([UserMessage(text="")]))
    contents = cast(list[MutableJSON], body["contents"])
    parts = cast(list[MutableJSON], contents[0]["parts"])
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

    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse_body,
            headers={"Content-Type": "text/event-stream"},
        )

    transport = httpx.MockTransport(handle)
    p = Google.from_key("k")
    m = p.model("gemini-2.5-flash")
    m._client = httpx.AsyncClient(transport=transport)
    resp = await m.stream(ModelRequest(messages=[UserMessage(text="x")]))
    assert resp.message.text == "calling"
    assert len(resp.message.tool_calls) == 1
    tc = resp.message.tool_calls[0]
    assert tc.name == "Bash"
    assert dict(tc.args) == {"cmd": "ls"}
    # STOP + tool calls → upgraded to model_tool_use.
    assert resp.stop_reason == "model_tool_use"
    assert resp.tokens.input_tokens == 10
    assert resp.tokens.output_tokens == 5
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

    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse_body,
            headers={"Content-Type": "text/event-stream"},
        )

    transport = httpx.MockTransport(handle)
    p = Google.from_key("k")
    m = p.model("gemini-2.5-flash")
    m._client = httpx.AsyncClient(transport=transport)
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

    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse_body,
            headers={"Content-Type": "text/event-stream"},
        )

    transport = httpx.MockTransport(handle)
    p = Google.from_key("k")
    m = p.model("gemini-2.5-flash")
    m._client = httpx.AsyncClient(transport=transport)
    with (
        caplog.at_level(logging.WARNING, logger="sagent.providers.google"),
        pytest.raises(StreamInterruptedError) as raised,
    ):
        await m.stream(ModelRequest(messages=[UserMessage(text="x")]))
    assert raised.value.response.message.text == "ok"
    assert any("malformed JSON chunk" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_google_stream_eof_without_finish_reason_raises_interrupted() -> None:
    sse_body = b'data: {"candidates":[{"content":{"parts":[{"text":"partial"}]}}]}\n\n'

    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse_body,
            headers={"Content-Type": "text/event-stream"},
        )

    transport = httpx.MockTransport(handle)
    p = Google.from_key("k")
    m = p.model("gemini-2.5-flash")
    m._client = httpx.AsyncClient(transport=transport)
    with pytest.raises(StreamInterruptedError) as raised:
        await m.stream(ModelRequest(messages=[UserMessage(text="x")]))
    assert raised.value.response.message.text == "partial"


@pytest.mark.asyncio
async def test_google_stream_raises_when_all_json_chunks_are_malformed() -> None:
    sse_body = b"data: {not-json}\n\ndata: also-not-json\n\n"

    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse_body,
            headers={"Content-Type": "text/event-stream"},
        )

    transport = httpx.MockTransport(handle)
    p = Google.from_key("k")
    m = p.model("gemini-2.5-flash")
    m._client = httpx.AsyncClient(transport=transport)
    with pytest.raises(ValueError, match="only malformed JSON chunks"):
        await m.stream(ModelRequest(messages=[UserMessage(text="x")]))


@pytest.mark.asyncio
async def test_google_stream_max_tokens_finish_reason() -> None:
    """``MAX_TOKENS`` finish reason normalizes to ``max_tokens``."""
    sse_body = (
        b'data: {"candidates":[{"content":{"parts":[{"text":"trunc"}]},'
        b'"finishReason":"MAX_TOKENS"}]}\n\n'
    )

    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse_body,
            headers={"Content-Type": "text/event-stream"},
        )

    transport = httpx.MockTransport(handle)
    p = Google.from_key("k")
    m = p.model("gemini-2.5-flash")
    m._client = httpx.AsyncClient(transport=transport)
    resp = await m.stream(ModelRequest(messages=[UserMessage(text="x")]))
    assert resp.stop_reason == "max_tokens"


def test_google_build_response_cache_tokens_split_input_cost() -> None:
    """``cache_read`` is subtracted from input before the request-rate charge."""
    pricing = Pricing(request=1.0, response=2.0, cache_read=0.5)
    resp = _build_response(
        text="",
        tool_calls=[],
        usage=cast(
            MutableJSON,
            {
                "promptTokenCount": 1000,
                "candidatesTokenCount": 100,
                "cachedContentTokenCount": 300,
            },
        ),
        finish_reason="STOP",
        pricing=pricing,
    )
    assert resp.tokens.cache_read_tokens == 300
    # (1000-300)*1 + 300*0.5 = 850 / 1M = 0.00085.
    assert resp.input_cost == pytest.approx(0.00085)


def test_google_from_key() -> None:
    p = Google.from_key("AIza-key")
    assert isinstance(p, Google)
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
    assert m.model_id == Google.DEFAULT_MODEL


def test_google_utility_model_uses_flash_lite() -> None:
    """Utility model resolves to the cheapest current-gen Gemini.

    ``gemini-2.5-flash-lite`` is the cheapest non-deprecated entry in
    ``KNOWN_MODELS`` -- both transports (API key and CLI) inherit it.
    """
    p = Google.from_key("k")
    m = p.utility_model()
    assert m.model_id == Google.DEFAULT_UTILITY_MODEL


def test_google_model_properties() -> None:
    p = Google.from_key("k")
    m = p.model("gemini-2.5-pro")
    assert m.max_request_tokens == 1_000_000
    assert m.supports_streaming is True
    assert m.supports_thinking is True
    assert m.supports_effort is True
    assert m.supports_cache_control is False
    # Gemini publishes no per-image pixel or byte cap (images are tiled
    # server-side); the only documented limit is the 20 MB total request size.
    assert m.max_image_dim == 0
    assert m.max_image_bytes == 0
    assert m.max_request_bytes == 20 * 1024 * 1024


def test_legacy_gemini_models_do_not_support_thinking() -> None:
    """Legacy ``gemini-1.5-*`` models reject ``thinkingConfig``.

    The Google API answers HTTP 400 ``thinkingConfig is not supported`` for
    these snapshots; capability advertisement must match so the local layer
    short-circuits before the request flies.
    """
    p = Google.from_key("k")
    assert p.model("gemini-1.5-flash").supports_thinking is False
    assert p.model("gemini-1.5-pro").supports_thinking is False
    assert p.model("gemini-1.5-flash").supports_effort is False
    assert p.model("gemini-1.5-pro").supports_effort is False


def test_build_request_adaptive_thinking_uses_dynamic_budget() -> None:
    body = _build_request(
        ModelRequest(messages=[UserMessage(text="x")], thinking="adaptive")
    )
    gen_config = cast(MutableJSON, body["generationConfig"])
    assert gen_config["thinkingConfig"] == {
        "includeThoughts": True,
        "thinkingBudget": -1,
    }


def test_build_request_enabled_thinking_uses_dynamic_budget() -> None:
    body = _build_request(
        ModelRequest(messages=[UserMessage(text="x")], thinking="enabled")
    )
    gen_config = cast(MutableJSON, body["generationConfig"])
    assert gen_config["thinkingConfig"] == {
        "includeThoughts": True,
        "thinkingBudget": -1,
    }


def test_build_request_effort_min_sets_small_budget() -> None:
    body = _build_request(ModelRequest(messages=[UserMessage(text="x")], effort="min"))
    gen_config = cast(MutableJSON, body["generationConfig"])
    assert gen_config["thinkingConfig"] == {
        "includeThoughts": True,
        "thinkingBudget": 1_024,
    }


def test_build_request_effort_max_sets_largest_budget() -> None:
    body = _build_request(ModelRequest(messages=[UserMessage(text="x")], effort="max"))
    gen_config = cast(MutableJSON, body["generationConfig"])
    assert gen_config["thinkingConfig"] == {
        "includeThoughts": True,
        "thinkingBudget": 24_576,
    }


def test_build_request_invalid_effort_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Invalid Google effort"):
        _build_request(ModelRequest(messages=[UserMessage(text="x")], effort="turbo"))


def test_build_request_thinking_omits_temperature() -> None:
    body = _build_request(
        ModelRequest(messages=[UserMessage(text="x")], thinking="adaptive")
    )
    gen_config = cast(MutableJSON, body["generationConfig"])
    assert "thinkingConfig" in gen_config
    assert "temperature" not in gen_config


def test_build_request_without_thinking_keeps_temperature() -> None:
    body = _build_request(
        ModelRequest(messages=[UserMessage(text="x")], temperature=0.3)
    )
    gen_config = cast(MutableJSON, body["generationConfig"])
    assert gen_config["temperature"] == 0.3


@pytest.mark.asyncio
async def test_google_model_close_closes_reusable_http_client() -> None:
    p = Google.from_key("k")
    m = p.model("gemini-2.5-flash")
    client = httpx.AsyncClient()
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

    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(413, text="Input too large for model context.")

    transport = httpx.MockTransport(handle)
    p = Google.from_key("k")
    m = p.model("gemini-2.5-flash")
    m._client = httpx.AsyncClient(transport=transport)
    with pytest.raises(PromptTooLongError):
        await m.stream(ModelRequest(messages=[UserMessage(text="x")]))


@pytest.mark.asyncio
async def test_google_stream_413_byte_body_raises_request_too_large() -> None:
    """A 413 with a byte-limit body raises ``RequestTooLargeError``.

    The byte ceiling is fixed across models, so it must route to
    byte-overflow recovery (shed attachment bytes), not token-overflow.
    """

    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(413, text="Request entity too large.")

    transport = httpx.MockTransport(handle)
    p = Google.from_key("k")
    m = p.model("gemini-2.5-flash")
    m._client = httpx.AsyncClient(transport=transport)
    with pytest.raises(RequestTooLargeError):
        await m.stream(ModelRequest(messages=[UserMessage(text="x")]))


@pytest.mark.asyncio
async def test_google_stream_400_exceeds_maximum_normalizes() -> None:
    """The ``exceeds the maximum`` substring is the canonical Gemini overflow phrase."""

    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400, text="The input token count exceeds the maximum allowed."
        )

    transport = httpx.MockTransport(handle)
    p = Google.from_key("k")
    m = p.model("gemini-2.5-flash")
    m._client = httpx.AsyncClient(transport=transport)
    with pytest.raises(PromptTooLongError):
        await m.stream(ModelRequest(messages=[UserMessage(text="x")]))


@pytest.mark.asyncio
async def test_google_stream_400_too_long_raises_prompt_too_long() -> None:
    """Stream path normalizes 4xx with overflow body to ``PromptTooLongError``."""
    sse_body = b'data: {"candidates":[{"content":{"parts":[{"text":"hi"}]}}]}\n\n'

    def handle(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(400, text="Input too long for model context.")

    transport = httpx.MockTransport(handle)
    p = Google.from_key("k")
    m = p.model("gemini-2.5-flash")
    m._client = httpx.AsyncClient(transport=transport)
    del sse_body  # only error path exercised
    with pytest.raises(PromptTooLongError):
        await m.stream(ModelRequest(messages=[UserMessage(text="x")]))


@pytest.mark.asyncio
async def test_google_stream_500_with_overflow_keyword_is_http_error_not_overflow() -> (
    None
):
    """Stream 5xx with overflow keywords propagates as HTTPStatusError."""

    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal error: too long traceback")

    transport = httpx.MockTransport(handle)
    p = Google.from_key("k")
    m = p.model("gemini-2.5-flash")
    m._client = httpx.AsyncClient(transport=transport)
    with pytest.raises(httpx.HTTPStatusError):
        await m.stream(ModelRequest(messages=[UserMessage(text="x")]))


@pytest.mark.asyncio
async def test_google_actual_request_tokens_hits_count_tokens_endpoint() -> None:
    """``actual_request_tokens`` POSTs to ``:countTokens`` and reads ``totalTokens``."""
    seen_paths: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        return httpx.Response(200, json={"totalTokens": 314})

    transport = httpx.MockTransport(handle)
    p = Google.from_key("k")
    m = p.model("gemini-2.5-flash")
    m._client = httpx.AsyncClient(transport=transport)
    n = await m.actual_request_tokens(
        ModelRequest(messages=[UserMessage(text="ping")]),
    )
    assert n == 314
    assert any(path.endswith(":countTokens") for path in seen_paths)


@pytest.mark.asyncio
async def test_google_stream_400_other_raises_value_error() -> None:
    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="malformed request body")

    transport = httpx.MockTransport(handle)
    p = Google.from_key("k")
    m = p.model("gemini-2.5-flash")
    m._client = httpx.AsyncClient(transport=transport)
    with pytest.raises(ValueError, match="Google API 400"):
        await m.stream(ModelRequest(messages=[UserMessage(text="x")]))


class _StubTool:
    name: str = "Echo"
    tool_id: str = "application/x-tool-echo"
    description: str = "Echo"
    directive_schema: dict[str, object] = {  # noqa: RUF012 -- test stub
        "type": "object",
        "additionalProperties": False,
    }


def test_build_request_tools_strip_additional_properties() -> None:
    """`additionalProperties` is recursively stripped from tool schemas."""
    tool = _StubTool()
    req = ModelRequest(
        messages=[UserMessage(text="hi")],
        tools=cast("list[Tool]", [tool]),
    )
    body = _build_request(req)
    tools_section = cast(list[MutableJSON], body["tools"])
    fns = cast(list[MutableJSON], tools_section[0]["functionDeclarations"])
    schema = cast(MutableJSON, fns[0]["parameters"])
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
    body = _build_request(_make_request([asst]))
    parts = cast(
        list[MutableJSON], cast(list[MutableJSON], body["contents"])[0]["parts"]
    )
    assert parts[0] == {"text": "answer", "thoughtSignature": "sig-text"}
    assert parts[1]["thoughtSignature"] == "sig-fc"


def test_build_request_omits_empty_thought_signature() -> None:
    """No signature (older models / thinking off) -> no thoughtSignature key."""
    asst = AssistantMessage(
        text="hi",
        tool_calls=(ToolCall(id="e", name="Bash", args={}),),
    )
    body = _build_request(_make_request([asst]))
    parts = cast(
        list[MutableJSON], cast(list[MutableJSON], body["contents"])[0]["parts"]
    )
    assert parts[0] == {"text": "hi"}
    assert "thoughtSignature" not in parts[1]


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
