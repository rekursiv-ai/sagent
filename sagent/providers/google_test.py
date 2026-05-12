"""Tests for ``providers.google``: Gemini wire-format + parse."""

from __future__ import annotations

from typing import cast

import httpx
import pytest

from sagent.agent.runtime import (
    AssistantMessage,
    HistoryEntry,
    ToolCall,
    ToolResult,
    UserMessage,
)
from sagent.custom_exceptions import PromptTooLongError
from sagent.custom_types import ModelRequest, Pricing, Tool
from sagent.lib.json import MutableJSON, MutableJSONValue
from sagent.providers.google import (
    Google,
    _build_request,
    _parse_response,
    _strip_additional_properties,
)


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


def _make_request(messages: list[HistoryEntry], **kw: object) -> ModelRequest:
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


def test_parse_response_text_only() -> None:
    data = cast(
        MutableJSON,
        {
            "candidates": [
                {
                    "content": {"parts": [{"text": "hello"}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
            },
        },
    )

    resp = _parse_response(data, Pricing())
    assert resp.message.text == "hello"
    assert resp.stop_reason == "model_finished"
    assert resp.tokens.input_tokens == 10
    assert resp.tokens.output_tokens == 5


def test_parse_response_function_call_extracted() -> None:
    data = cast(
        MutableJSON,
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "calling"},
                            {
                                "functionCall": {
                                    "name": "Bash",
                                    "args": {"cmd": "ls"},
                                }
                            },
                        ]
                    },
                    "finishReason": "STOP",
                }
            ],
        },
    )

    resp = _parse_response(data, Pricing())
    assert len(resp.message.tool_calls) == 1
    tc = resp.message.tool_calls[0]
    assert tc.name == "Bash"
    assert dict(tc.args) == {"cmd": "ls"}
    # Stop reason is STOP but tool calls present → upgraded.
    assert resp.stop_reason == "model_tool_use"


def test_parse_response_max_tokens_finish_reason() -> None:
    data = cast(
        MutableJSON,
        {
            "candidates": [
                {
                    "content": {"parts": [{"text": "trunc"}]},
                    "finishReason": "MAX_TOKENS",
                }
            ],
        },
    )

    resp = _parse_response(data, Pricing())
    assert resp.stop_reason == "max_tokens"


def test_parse_response_no_candidates_raises() -> None:

    with pytest.raises(ValueError, match="no candidates"):
        _parse_response(cast(MutableJSON, {"candidates": []}), Pricing())


def test_parse_response_synthesized_message_id() -> None:
    data = cast(
        MutableJSON,
        {
            "candidates": [
                {"content": {"parts": [{"text": "x"}]}, "finishReason": "STOP"},
            ],
        },
    )

    resp = _parse_response(data, Pricing())
    assert resp.message_id.startswith("gemini_")
    # Same id used for request_id (Gemini has no separate header).
    assert resp.request_id == resp.message_id


def test_parse_response_cache_read_tokens_split() -> None:

    data = cast(
        MutableJSON,
        {
            "candidates": [
                {"content": {"parts": [{"text": ""}]}, "finishReason": "STOP"},
            ],
            "usageMetadata": {
                "promptTokenCount": 1000,
                "candidatesTokenCount": 100,
                "cachedContentTokenCount": 300,
            },
        },
    )
    pricing = Pricing(request=1.0, response=2.0, cache_read=0.5)
    resp = _parse_response(data, pricing)
    assert resp.tokens.cache_read_tokens == 300
    # (1000-300)*1 + 300*0.5 = 700 + 150 = 850 / 1M = 0.00085.
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


def test_google_utility_model_uses_flash() -> None:
    p = Google.from_key("k")
    m = p.utility_model()
    assert m.model_id == Google.DEFAULT_UTILITY_MODEL


def test_google_model_properties() -> None:
    p = Google.from_key("k")
    m = p.model("gemini-2.5-pro")
    assert m.max_request_tokens == 1_000_000
    assert m.supports_streaming is True
    assert m.supports_thinking is False
    assert m.supports_effort is False
    assert m.supports_cache_control is False
    assert m.max_image_dim == 3072
    assert m.max_image_bytes == 20 * 1024 * 1024


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
    assert m.estimate_text_token_count("a" * 16) == 4


@pytest.mark.asyncio
async def test_google_buffer_parses_via_mock_transport() -> None:
    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"parts": [{"text": "pong"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 1,
                    "candidatesTokenCount": 1,
                },
            },
        )

    transport = httpx.MockTransport(handle)
    p = Google.from_key("k")
    m = p.model("gemini-2.5-flash")
    m._client = httpx.AsyncClient(transport=transport)
    resp = await m.buffer(ModelRequest(messages=[UserMessage(text="ping")]))
    assert resp.message.text == "pong"


@pytest.mark.asyncio
async def test_google_buffer_400_too_large_raises_prompt_too_long() -> None:
    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="Input too large for model context.")

    transport = httpx.MockTransport(handle)
    p = Google.from_key("k")
    m = p.model("gemini-2.5-flash")
    m._client = httpx.AsyncClient(transport=transport)
    with pytest.raises(PromptTooLongError):
        await m.buffer(ModelRequest(messages=[UserMessage(text="x")]))


@pytest.mark.asyncio
async def test_google_buffer_400_other_raises_value_error() -> None:
    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="malformed request body")

    transport = httpx.MockTransport(handle)
    p = Google.from_key("k")
    m = p.model("gemini-2.5-flash")
    m._client = httpx.AsyncClient(transport=transport)
    with pytest.raises(ValueError, match="Google API 400"):
        await m.buffer(ModelRequest(messages=[UserMessage(text="x")]))


class _StubTool:
    name: str = "Echo"
    tool_id: str = "application/x-tool-echo"
    description: str = "Echo"
    directive_schema: dict[str, object] = {  # noqa: RUF012 -- test stub
        "type": "object",
        "additionalProperties": False,
    }
    supports_microcompaction: bool = False


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


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
