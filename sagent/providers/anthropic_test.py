"""Tests for ``providers.anthropic``: wire-format translation + response parse."""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import anthropic as anthropic_sdk
import httpx
import pytest

from sagent.providers.anthropic import (
    Anthropic,
    _assistant_blocks,
    _build_messages,
    _guard_stream_interrupt,
    _is_prompt_too_long_text,
    _parse_response,
    _strip_context_tag,
    _tool_result_block,
    _tool_use_block,
    context_betas,
)
from sagent.providers.lib.id_remap import IdRemapper
from sagent.types.exceptions import (
    PromptTooLongError,
    StreamInterruptedError,
)
from sagent.types.history import (
    AssistantMessage,
    HistoryEntry,
    ToolCall,
    ToolResult,
    UserMessage,
)
from sagent.types.model import (
    ModelRequest,
    ModelResponse,
    Pricing,
    TokenCount,
)


def _make_request(
    messages: list[HistoryEntry],
) -> ModelRequest:
    return ModelRequest(messages=messages)


@pytest.mark.parametrize(
    ("model_id", "stripped"),
    [
        ("claude-opus-4-7+1m", "claude-opus-4-7"),
        ("claude-opus-4-7+200k", "claude-opus-4-7"),
        ("claude-opus-4-7", "claude-opus-4-7"),
        ("Claude-Opus-4-7+1M", "Claude-Opus-4-7"),
    ],
)
def test_strip_context_tag(model_id: str, stripped: str) -> None:
    assert _strip_context_tag(model_id) == stripped


def test_context_betas_one_million_emits_beta() -> None:
    assert context_betas("claude-opus-4-7+1m") == ["context-1m-2025-08-07"]


def test_context_betas_no_tag_returns_empty() -> None:
    assert context_betas("claude-haiku-4-5") == []


@pytest.mark.parametrize(
    "msg",
    [
        "prompt is too long: 123",
        "Request exceeded context window",
        "PROMPT_TOO_LONG",
    ],
)
def test_is_prompt_too_long_text_positive(msg: str) -> None:
    assert _is_prompt_too_long_text(msg) is True


def test_is_prompt_too_long_text_negative() -> None:
    assert _is_prompt_too_long_text("rate limited") is False


def test_build_messages_user_text_only() -> None:
    msgs = _build_messages(_make_request([UserMessage(text="hi")]))
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    blocks = cast(list[dict[str, object]], msgs[0]["content"])
    # Cache breakpoint may have rewritten the last block to add cache_control;
    # ensure the underlying text payload is preserved.
    assert any(b.get("text") == "hi" for b in blocks)


def test_build_messages_assistant_with_tool_call_remaps_id() -> None:
    asst = AssistantMessage(
        text="thinking",
        tool_calls=(ToolCall(id="orig-9", name="Bash", args={"cmd": "ls"}),),
    )
    msgs = _build_messages(_make_request([asst]))
    assert msgs[0]["role"] == "assistant"
    blocks = cast(list[dict[str, object]], msgs[0]["content"])
    # First block is the text, second is tool_use with the remapped id.
    assert blocks[0]["type"] == "text"
    assert blocks[1]["type"] == "tool_use"
    assert blocks[1]["id"] == "toolu_0"
    assert blocks[1]["name"] == "Bash"


def test_build_messages_consecutive_tool_results_batched_into_single_user() -> None:
    asst = AssistantMessage(
        tool_calls=(
            ToolCall(id="c1", name="N", args={}),
            ToolCall(id="c2", name="N", args={}),
        ),
    )
    res1 = ToolResult(call_id="c1", content="r1")
    res2 = ToolResult(call_id="c2", content="r2")
    msgs = _build_messages(_make_request([asst, res1, res2]))
    # [assistant, user(with 2 tool_results)].
    assert len(msgs) == 2
    assert msgs[-1]["role"] == "user"
    content = cast(list[dict[str, object]], msgs[-1]["content"])
    # Both tool_results in one batch (last has cache_control sidecar).
    tool_results = [b for b in content if b.get("type") == "tool_result"]
    assert len(tool_results) == 2


def test_user_after_tool_results_coalesces_into_same_wire_msg() -> None:
    """C1: mid-cohort user text MUST merge with pending tool_results.

    Emitting a separate ``role=user`` message after a ``role=user``
    (tool_results) breaks Anthropic's strict alternation and triggers
    HTTP 400. The fix coalesces both into the same wire user message.
    """
    asst = AssistantMessage(tool_calls=(ToolCall(id="c1", name="N", args={}),))
    tool_result = ToolResult(call_id="c1", content="[detached]")
    user_redirect = UserMessage(text="actually do something else")
    msgs = _build_messages(_make_request([asst, tool_result, user_redirect]))
    # Expect 2 messages: assistant(tool_use) + user(tool_result + text).
    assert len(msgs) == 2, [m["role"] for m in msgs]
    assert msgs[0]["role"] == "assistant"
    assert msgs[1]["role"] == "user"
    content = cast(list[dict[str, object]], msgs[1]["content"])
    type_names = {b.get("type") for b in content}
    assert "tool_result" in type_names
    assert "text" in type_names


def test_build_messages_tool_result_id_pairs_with_call() -> None:
    asst = AssistantMessage(
        tool_calls=(ToolCall(id="orig-X", name="N", args={}),),
    )
    msgs = _build_messages(
        _make_request([asst, ToolResult(call_id="orig-X", content="ok")])
    )
    blocks = cast(list[dict[str, object]], msgs[0]["content"])
    tool_use = next(b for b in blocks if b.get("type") == "tool_use")
    tool_result_block_msg = cast(list[dict[str, object]], msgs[1]["content"])
    tool_result = next(
        b for b in tool_result_block_msg if b.get("type") == "tool_result"
    )
    assert tool_use["id"] == tool_result["tool_use_id"]


def test_build_messages_empty_history_no_messages() -> None:
    assert _build_messages(_make_request([])) == []


def test_assistant_blocks_thinking_only_appends_placeholder() -> None:
    """Thinking-only assistant gets a trailing ``.`` text block."""
    asst = AssistantMessage(
        thinking_blocks=({"type": "thinking", "thinking": "..."},),
    )
    blocks = _assistant_blocks(asst, IdRemapper("toolu_"))
    assert blocks[-1] == {"type": "text", "text": "."}


def test_assistant_blocks_text_only() -> None:
    asst = AssistantMessage(text="hello")
    blocks = _assistant_blocks(asst, IdRemapper("toolu_"))
    assert blocks == [{"type": "text", "text": "hello"}]


def test_assistant_blocks_thinking_text_tool() -> None:
    asst = AssistantMessage(
        text="answer",
        thinking_blocks=({"type": "thinking", "thinking": "..."},),
        tool_calls=(ToolCall(id="x", name="N", args={"a": 1}),),
    )
    blocks = _assistant_blocks(asst, IdRemapper("toolu_"))
    assert blocks[0]["type"] == "thinking"
    assert blocks[1] == {"type": "text", "text": "answer"}
    assert blocks[2]["type"] == "tool_use"


def test_tool_use_block_remaps_id_through_remapper() -> None:
    ids = IdRemapper("toolu_")
    out = _tool_use_block(ToolCall(id="ext", name="N", args={"k": "v"}), ids)
    assert out == {
        "type": "tool_use",
        "id": "toolu_0",
        "name": "N",
        "input": {"k": "v"},
    }


def test_tool_result_block_plain_text() -> None:
    ids = IdRemapper("toolu_")
    out = _tool_result_block(
        ToolResult(call_id="ext-1", content="done"),
        ids,
        max_image_dim=8000,
        max_image_bytes=5 * 1024 * 1024,
    )
    assert out == {
        "type": "tool_result",
        "tool_use_id": "toolu_0",
        "content": "done",
        "is_error": False,
    }


def test_tool_result_block_is_error_flag_propagates() -> None:
    ids = IdRemapper("toolu_")
    out = _tool_result_block(
        ToolResult(call_id="x", content="oops", is_error=True),
        ids,
        max_image_dim=8000,
        max_image_bytes=5 * 1024 * 1024,
    )
    assert out["is_error"] is True


def _build_anthropic_message(
    *,
    text: str = "",
    tool_calls: tuple[tuple[str, str, dict[str, object]], ...] = (),
    stop_reason: str = "end_turn",
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_creation: int = 0,
    cache_read: int = 0,
) -> object:
    """Return a duck-typed object mimicking ``anthropic.types.Message``."""
    # Use the real anthropic SDK classes via the lazy-imported module so
    # ``isinstance`` checks inside ``_parse_response`` succeed.
    text_blocks: list[object] = []
    if text:
        text_blocks.append(anthropic_sdk.types.TextBlock(type="text", text=text))
    for tc_id, tc_name, args in tool_calls:
        text_blocks.append(
            anthropic_sdk.types.ToolUseBlock(
                type="tool_use", id=tc_id, name=tc_name, input=args
            )
        )
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.cache_creation_input_tokens = cache_creation
    usage.cache_read_input_tokens = cache_read
    msg = MagicMock()
    msg.content = text_blocks
    msg.usage = usage
    msg.stop_reason = stop_reason
    msg.stop_sequence = None
    msg.id = "msg_xyz"
    msg._request_id = "req_xyz"
    return msg


def test_parse_response_text_only() -> None:
    raw = _build_anthropic_message(text="hi", input_tokens=5, output_tokens=2)
    resp = _parse_response(raw, Pricing())  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type] -- duck-typed SDK mock
    assert resp.message.text == "hi"
    assert resp.stop_reason == "model_finished"
    assert resp.tokens.input_tokens == 5
    assert resp.tokens.output_tokens == 2


def test_parse_response_tool_call_extracted() -> None:
    raw = _build_anthropic_message(
        tool_calls=(("toolu_xyz", "Bash", {"cmd": "ls"}),),
        stop_reason="tool_use",
    )
    resp = _parse_response(raw, Pricing())  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type] -- duck-typed SDK mock
    assert len(resp.message.tool_calls) == 1
    call = resp.message.tool_calls[0]
    assert call.id == "toolu_xyz"
    assert call.name == "Bash"
    assert dict(call.args) == {"cmd": "ls"}
    assert resp.stop_reason == "model_tool_use"


def test_parse_response_cache_tokens_split_correctly() -> None:
    raw = _build_anthropic_message(
        input_tokens=1000,
        output_tokens=100,
        cache_creation=200,
        cache_read=400,
    )
    pricing = Pricing(request=1.0, response=2.0, cache_write=4.0, cache_read=0.5)
    resp = _parse_response(raw, pricing)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type] -- duck-typed SDK mock
    assert resp.tokens.cache_creation_tokens == 200
    assert resp.tokens.cache_read_tokens == 400
    # input cost = 1000*1 + 200*4 + 400*0.5 = 2000 / 1M = 0.002.
    assert resp.input_cost == pytest.approx(0.002)


def test_parse_response_carries_message_and_request_ids() -> None:
    raw = _build_anthropic_message()
    resp = _parse_response(raw, Pricing())  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type] -- duck-typed SDK mock
    assert resp.message_id == "msg_xyz"
    assert resp.request_id == "req_xyz"


def test_guard_stream_interrupt_raises_when_tool_use_without_calls() -> None:
    resp = ModelResponse(
        message=AssistantMessage(text="partial"),
        stop_reason="model_tool_use",
    )
    with pytest.raises(StreamInterruptedError):
        _guard_stream_interrupt(resp, kind="stream", model_id="claude-x")


def test_guard_stream_interrupt_silent_when_calls_present() -> None:
    resp = ModelResponse(
        message=AssistantMessage(
            text="",
            tool_calls=(ToolCall(id="x", name="N", args={}),),
        ),
        stop_reason="model_tool_use",
    )
    _guard_stream_interrupt(resp, kind="stream", model_id="claude-x")


def test_guard_stream_interrupt_silent_when_not_tool_use() -> None:
    resp = ModelResponse(
        message=AssistantMessage(text="done"),
        stop_reason="model_finished",
    )
    _guard_stream_interrupt(resp, kind="stream", model_id="claude-x")


def test_anthropic_from_key() -> None:
    p = Anthropic.from_key("sk-ant-test")
    assert isinstance(p, Anthropic)


def test_anthropic_from_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API key not configured"):
        Anthropic.from_env()


def test_anthropic_from_env_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-key")
    p = Anthropic.from_env()
    assert isinstance(p, Anthropic)


def test_anthropic_model_known_id_returns_backend() -> None:
    p = Anthropic.from_key("k")
    m = p.model("claude-haiku-4-5")
    assert m.model_id == "claude-haiku-4-5"
    assert m.supports_thinking is False  # haiku profile.
    assert m.max_request_tokens == 200_000


def test_anthropic_model_unknown_id_raises() -> None:
    p = Anthropic.from_key("k")
    with pytest.raises(ValueError, match="Unknown model"):
        _ = p.model("unknown-claude")


def test_anthropic_model_strips_context_tag_for_profile_lookup() -> None:
    """``claude-sonnet-4-5+1m`` should resolve to the +1m profile entry."""
    p = Anthropic.from_key("k")
    m = p.model("claude-sonnet-4-5+1m")
    assert m.max_request_tokens == 1_000_000


def test_anthropic_default_model_resolves() -> None:
    p = Anthropic.from_key("k")
    m = p.model()
    assert m.model_id == Anthropic.DEFAULT_MODEL


def test_anthropic_utility_model_uses_haiku() -> None:
    p = Anthropic.from_key("k")
    m = p.utility_model()
    assert m.model_id == "claude-haiku-4-5"


def test_anthropic_subscription_property_false_on_api_key() -> None:
    assert Anthropic.from_key("k").subscription is False


def test_anthropic_model_is_context_overflow_via_prompt_too_long() -> None:
    p = Anthropic.from_key("k")
    m = p.model("claude-haiku-4-5")
    assert m.is_context_overflow(PromptTooLongError("too long")) is True


def test_anthropic_model_token_estimate_uses_profile_chars_per_token() -> None:
    p = Anthropic.from_key("k")
    m = p.model("claude-opus-4-7")
    # chars_per_token = 2.83; 28 chars / 2.83 ≈ 9.89 → int → 9.
    assert m.estimate_text_token_count("a" * 28) == 9


def test_anthropic_model_pricing_exposed() -> None:
    p = Anthropic.from_key("k")
    m = p.model("claude-haiku-4-5")
    assert m.pricing.request > 0


def test_anthropic_token_count_default_typing() -> None:
    # ``TokenCount`` from the model module must accept keyword construction.
    t = TokenCount(input_tokens=1, output_tokens=2)
    assert t.input_tokens == 1
    assert t.output_tokens == 2


def test_anthropic_model_supports_flags() -> None:
    p = Anthropic.from_key("k")
    m = p.model("claude-opus-4-7")
    assert m.supports_streaming is True
    assert m.supports_effort is True
    assert m.supports_cache_control is True
    assert m.supports_persistent_retry is True
    assert m.supports_context_management is False  # API-key, not sub.
    assert m.supports_account_auth is False
    assert m.valid_service_tiers == ("auto", "standard_only")


def test_anthropic_build_kwargs_emits_service_tier() -> None:
    p = Anthropic.from_key("k")
    m = p.model("claude-opus-4-7")
    req = ModelRequest(messages=[UserMessage(text="x")], service_tier="standard_only")
    kwargs = m._build_kwargs(req, [])
    assert kwargs["service_tier"] == "standard_only"


def test_anthropic_build_kwargs_omits_unknown_service_tier() -> None:
    p = Anthropic.from_key("k")
    m = p.model("claude-opus-4-7")
    req = ModelRequest(messages=[UserMessage(text="x")], service_tier="priority")
    kwargs = m._build_kwargs(req, [])
    assert "service_tier" not in kwargs


def test_anthropic_model_image_limits() -> None:
    p = Anthropic.from_key("k")
    m = p.model("claude-opus-4-7")
    assert m.max_image_dim == 8000
    assert m.max_image_bytes == 5 * 1024 * 1024


def test_anthropic_model_is_context_overflow_non_api_status_error() -> None:
    p = Anthropic.from_key("k")
    m = p.model("claude-opus-4-7")
    assert m.is_context_overflow(RuntimeError("anything")) is False


def _api_status_error(status_code: int, message: str) -> anthropic_sdk.APIStatusError:
    """Construct an APIStatusError as the SDK would, with an arbitrary status."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code, request=request)
    body = {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": message},
    }
    return anthropic_sdk.APIStatusError(message, response=response, body=body)


class _BodyError(Exception):
    """Exception with an SDK-like ``body`` attribute."""

    def __init__(self, body: object) -> None:
        self.body = body
        super().__init__("x")


def test_anthropic_model_is_context_overflow_unusual_status_with_overflow_text() -> (
    None
):
    """Generic APIStatusError (non-{400,413}) carrying overflow text is overflow.

    Reproduces the production failure where the SDK returned a status
    code outside the SDK's per-code subclass map -- yielding the bare
    ``APIStatusError`` -- with body text ``Request size exceeds model
    context window``. The body is the canonical signal; the status
    code adds nothing.
    """
    p = Anthropic.from_key("k")
    m = p.model("claude-opus-4-7")
    err = _api_status_error(414, "Request size exceeds model context window")
    assert m.is_context_overflow(err) is True


def test_anthropic_model_is_retryable_provider_error_rate_limit() -> None:
    p = Anthropic.from_key("k")
    m = p.model("claude-opus-4-7")
    err = _BodyError({"type": "rate_limit_error"})
    assert m.is_retryable_provider_error(err) is True


def test_anthropic_model_is_retryable_provider_error_nested_api_error() -> None:
    p = Anthropic.from_key("k")
    m = p.model("claude-opus-4-7")
    err = _BodyError({"type": "error", "error": {"type": "api_error"}})
    assert m.is_retryable_provider_error(err) is True


def test_anthropic_model_is_retryable_provider_error_nested_invalid_request() -> None:
    p = Anthropic.from_key("k")
    m = p.model("claude-opus-4-7")
    err = _BodyError({"type": "error", "error": {"type": "invalid_request_error"}})
    assert m.is_retryable_provider_error(err) is False


def test_anthropic_model_is_retryable_provider_error_unrelated_type() -> None:
    p = Anthropic.from_key("k")
    m = p.model("claude-opus-4-7")
    err = _BodyError({"type": "permanent_error"})
    assert m.is_retryable_provider_error(err) is False


def test_anthropic_model_is_retryable_provider_error_no_body_attr() -> None:
    p = Anthropic.from_key("k")
    m = p.model("claude-opus-4-7")
    assert m.is_retryable_provider_error(RuntimeError("x")) is False


def test_anthropic_provider_extra_headers_for_1m_includes_beta() -> None:
    p = Anthropic.from_key("k")
    headers = p.extra_headers("claude-opus-4-7+1m")
    assert headers.get("anthropic-beta", "").startswith("context-1m")


def test_anthropic_provider_extra_headers_no_tag_empty() -> None:
    p = Anthropic.from_key("k")
    assert p.extra_headers("claude-haiku-4-5") == {}


def test_anthropic_provider_extra_body_default_none() -> None:
    p = Anthropic.from_key("k")
    assert p.extra_body(has_thinking=False, cache_cold=False) is None


def test_anthropic_provider_build_system_passthrough() -> None:
    p = Anthropic.from_key("k")
    assert p.build_system("hello") == "hello"


def test_anthropic_provider_build_system_none_returns_not_given() -> None:
    p = Anthropic.from_key("k")
    out = p.build_system(None)
    # ``anthropic.NOT_GIVEN`` is a sentinel; type is anthropic.NotGiven.
    assert out is anthropic_sdk.NOT_GIVEN


def test_anthropic_model_max_request_tokens_override() -> None:
    p = Anthropic.from_key("k")
    m = p.model("claude-opus-4-7", max_request_tokens=10_000)
    assert m.max_request_tokens == 10_000


def test_anthropic_model_max_response_tokens_from_profile() -> None:
    p = Anthropic.from_key("k")
    m = p.model("claude-opus-4-7")
    assert m.max_response_tokens == 128_000


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
