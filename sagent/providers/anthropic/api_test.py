"""Tests for ``providers.anthropic``: wire-format translation + response parse."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast, override
from unittest.mock import AsyncMock, MagicMock, patch

from anthropic.types import MessageParam

import anthropic as anthropic_sdk
import httpx
import pytest

from sagent.agent.retry import error_status, is_retryable
from sagent.lib.custom_json import IntCodec, MutableJSON
from sagent.providers.anthropic.api import (
    Anthropic,
    _AnthropicModel,
    _assistant_blocks,
    _build_messages,
    _guard_stream_interrupt,
    _is_prompt_too_long_text,
    _parse_response,
    _raw_message_stream,
    _tool_result_block,
    _tool_use_block,
    build_context_management,
    context_betas,
    supports_native_context_management,
)
from sagent.providers.lib.errors import StreamingResponseNotReadError
from sagent.providers.lib.id_remap import IdRemapper
from sagent.types.cost import (
    PriceCatalog,
    PriceCatalogProduct,
    TokenPrice,
)
from sagent.types.model import (
    Limits,
    ModelRequest,
    ModelResponse,
    ModelSpec,
    PromptTooLongError,
    RequestTooLargeError,
    StreamInterruptedError,
    TokenCount,
)
from sagent.types.runtime import (
    DETACHED_PLACEHOLDER,
    AssistantMessage,
    BytesMessage,
    ModelContextEvent,
    ToolCall,
    ToolResult,
    UserMessage,
)
from sagent.types.tools import Tool


def _free_spec() -> ModelSpec:
    """A spec whose every rate is zero -- cost is not what these assert."""
    return ModelSpec(prices=PriceCatalog({PriceCatalogProduct(): TokenPrice()}))


def _fast_spec() -> ModelSpec:
    """Opus rates with a 2x fast tier, for the server-billed-speed tests."""
    return ModelSpec(
        prices=PriceCatalog(
            {
                PriceCatalogProduct(): TokenPrice(request=5.0, response=25.0),
                PriceCatalogProduct(fast=True): TokenPrice(request=10.0, response=50.0),
            }
        )
    )


def _make_request(
    messages: list[ModelContextEvent],
) -> ModelRequest:
    return ModelRequest(messages=messages)


def test_context_betas_one_million_emits_beta() -> None:
    assert "context-1m-2025-08-07" in context_betas("claude-opus-4-7+1m")


def test_context_betas_skip_one_million_beta_for_default_1m_model() -> None:
    assert "context-1m-2025-08-07" not in context_betas("claude-fable-5+1m")
    assert "context-1m-2025-08-07" not in context_betas("claude-sonnet-5+1m")


def test_context_betas_native_context_management_for_supported_models() -> None:
    assert "context-management-2025-06-27" in context_betas("claude-haiku-4-5")
    assert "context-management-2025-06-27" in context_betas("claude-opus-4-7+1m")
    assert "context-management-2025-06-27" in context_betas("claude-fable-5+1m")
    assert "context-management-2025-06-27" in context_betas("claude-sonnet-5+1m")


def test_context_betas_skip_native_context_management_for_unknown_models() -> None:
    # Older / unsupported models don't get the context-management beta.
    assert context_betas("claude-3-opus-20240229") == []


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


def test_is_prompt_too_long_text_false_positive_tool_schema_validation() -> None:
    """Tool-schema validation errors mention ``context window`` benignly."""
    msg = (
        "Invalid tool schema: parameter 'max_context_window' must be a positive integer"
    )
    assert _is_prompt_too_long_text(msg) is False


def test_is_prompt_too_long_text_structured_body_overflow() -> None:
    """Structured ``error.type`` + overflow ``message`` is the canonical signal."""
    body: Mapping[str, object] = {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "prompt is too long: 250000 tokens > 200000 maximum",
        },
    }
    assert _is_prompt_too_long_text("stringified blob", error_body=body) is True


def test_is_prompt_too_long_text_structured_body_unrelated_invalid_request() -> None:
    """Unrelated 400s under ``invalid_request_error`` must not classify as overflow."""
    body: Mapping[str, object] = {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": (
                "tools.0.input_schema: parameter 'context window' must be an object"
            ),
        },
    }
    assert _is_prompt_too_long_text(str(body), error_body=body) is False


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
    tool_result = ToolResult(call_id="c1", content=DETACHED_PLACEHOLDER)
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


def test_assistant_blocks_drops_signature_only_thinking() -> None:
    """Drop thinking blocks whose signed body was lost.

    A persisted ``thinking`` block with a non-empty ``signature`` but empty
    ``thinking`` text cannot re-validate against the server's signature, since
    the signed payload is gone. Replaying it triggers HTTP 400 ``thinking ...
    blocks in the latest assistant message cannot be modified``. The provider
    elides such blocks; intact thinking blocks and ``redacted_thinking`` (which
    has no client-visible body to lose) pass through.
    """
    asst = AssistantMessage(
        text="ok",
        thinking_blocks=(
            {"type": "thinking", "thinking": "", "signature": "sig0"},
            {"type": "thinking", "thinking": "kept", "signature": "sig1"},
            {"type": "redacted_thinking", "data": "opaque"},
        ),
    )
    blocks = _assistant_blocks(asst, IdRemapper("toolu_"))
    assert [b["type"] for b in blocks] == [
        "thinking",
        "redacted_thinking",
        "text",
    ]
    assert blocks[0].get("thinking") == "kept"


def test_assistant_blocks_drops_foreign_reasoning_blocks() -> None:
    """Non-native thinking-block types are dropped on cross-provider switch.

    After a session swaps from OpenAI / Moonshot / MiniMax / OpenAI-sub to
    Anthropic, ``AssistantMessage.thinking_blocks`` may carry
    ``{"type":"reasoning",...}`` entries. Anthropic's API rejects unknown
    block types with HTTP 400, and there's no faithful translation, so the
    serializer must silently elide them rather than forward verbatim.
    """
    asst = AssistantMessage(
        text="ok",
        thinking_blocks=(
            {"type": "reasoning", "text": "hidden"},
            {"type": "thinking", "thinking": "kept", "signature": "sig"},
        ),
    )
    blocks = _assistant_blocks(asst, IdRemapper("toolu_"))
    types = [b.get("type") for b in blocks]
    assert "reasoning" not in types
    assert types == ["thinking", "text"]


def test_assistant_blocks_drops_foreign_reasoning_only_appends_placeholder() -> None:
    """Foreign-only thinking + no text/tool_calls still emits a valid message.

    After elision the assistant message is empty; the existing rule that
    Anthropic rejects assistant messages with no content drives a fallback
    text block.
    """
    asst = AssistantMessage(
        thinking_blocks=({"type": "reasoning", "text": "hidden"},),
    )
    blocks = _assistant_blocks(asst, IdRemapper("toolu_"))
    # Empty blocks list after elision; no trailing placeholder is needed
    # because the message has no thinking block to mask. Anthropic will
    # reject an empty assistant turn upstream, but that's the caller's
    # contract -- we don't synthesize content out of nothing.
    assert blocks == []


def test_assistant_blocks_preserves_mixed_thinking_modes() -> None:
    asst = AssistantMessage(
        text="ok",
        thinking_blocks=(
            {"type": "thinking", "thinking": "visible", "signature": "sig1"},
            {"type": "redacted_thinking", "data": "opaque"},
            {"type": "thinking", "thinking": "visible again", "signature": "sig2"},
        ),
    )
    blocks = _assistant_blocks(asst, IdRemapper("toolu_"))
    assert blocks[:3] == [
        {"type": "thinking", "thinking": "visible", "signature": "sig1"},
        {"type": "redacted_thinking", "data": "opaque"},
        {"type": "thinking", "thinking": "visible again", "signature": "sig2"},
    ]
    assert blocks[3] == {"type": "text", "text": "ok"}


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


def test_tool_result_block_skips_svg_attachment() -> None:
    ids = IdRemapper("toolu_")
    out = _tool_result_block(
        ToolResult(
            call_id="x",
            content="[image: qr.svg]",
            attachments=(BytesMessage(data=b"<svg/>", descriptor="image/svg+xml"),),
        ),
        ids,
        max_image_dim=8000,
        max_image_bytes=5 * 1024 * 1024,
    )
    assert out["content"] == "[image: qr.svg]"


def _build_anthropic_message(
    *,
    text: str = "",
    tool_calls: tuple[tuple[str, str, dict[str, object]], ...] = (),
    stop_reason: str = "end_turn",
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_creation: int = 0,
    cache_read: int = 0,
    speed: str | None = None,
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
    usage.speed = speed
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
    resp = _parse_response(raw, _free_spec())  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type] -- duck-typed SDK mock
    assert resp.message.text == "hi"
    assert resp.stop_reason == "model_finished"
    assert resp.tokens.request == 5
    assert resp.tokens.response == 2


def test_parse_response_tool_call_extracted() -> None:
    raw = _build_anthropic_message(
        tool_calls=(("toolu_xyz", "Bash", {"cmd": "ls"}),),
        stop_reason="tool_use",
    )
    resp = _parse_response(raw, _free_spec())  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type] -- duck-typed SDK mock
    assert len(resp.message.tool_calls) == 1
    call = resp.message.tool_calls[0]
    assert call.id == "toolu_xyz"
    assert call.name == "Bash"
    assert dict(call.args) == {"cmd": "ls"}
    assert resp.stop_reason == "model_tool_use"


def test_parse_response_drops_placeholder_tool_name() -> None:
    """Tool blocks whose name is not a valid identifier are filtered out.

    Anthropic's API server injects its tool-use spec into the model's
    hidden context when tools are registered. When the user asks the
    model to dump its context, the model echoes the spec verbatim, and
    the API parses that echo as a structured ``tool_use`` block with
    ``name="$FUNCTION_NAME"`` (or other template placeholders). Without
    this guard, ``runtime._run_tool_and_post`` emits a visible
    "Unknown tool: $FUNCTION_NAME" and the unmatched tool_use poisons
    history for the next request.
    """
    raw = _build_anthropic_message(
        tool_calls=(
            ("toolu_real", "Bash", {"cmd": "ls"}),
            ("toolu_bad", "$FUNCTION_NAME", {}),
        ),
        stop_reason="tool_use",
    )
    resp = _parse_response(raw, _free_spec())  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type] -- duck-typed SDK mock
    assert [c.name for c in resp.message.tool_calls] == ["Bash"]


def test_parse_response_cache_tokens_split_correctly() -> None:
    raw = _build_anthropic_message(
        input_tokens=1000,
        output_tokens=100,
        cache_creation=200,
        cache_read=400,
    )
    spec = ModelSpec(
        prices=PriceCatalog(
            {
                PriceCatalogProduct(): TokenPrice(
                    request=1.0, response=2.0, cache_write=4.0, cache_read=0.5
                )
            }
        )
    )
    resp = _parse_response(raw, spec)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type] -- duck-typed SDK mock
    assert resp.tokens.cache_write == 200
    assert resp.tokens.cache_read == 400
    # input cost = 1000*1 + 200*4 + 400*0.5 = 2000 / 1M = 0.002.
    assert (
        resp.spend.request + resp.spend.cache_write + resp.spend.cache_read
    ) == pytest.approx(0.002)


def test_parse_response_carries_message_and_request_ids() -> None:
    raw = _build_anthropic_message()
    resp = _parse_response(raw, _free_spec())  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type] -- duck-typed SDK mock
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
    # haiku supports thinking via ``enabled`` only (measured: 249 readable
    # thinking chars; ``adaptive`` 400s 'not supported on this model').
    assert m.supports_thinking is True
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


def test_anthropic_model_accepts_fast_tag_on_supported_model() -> None:
    p = Anthropic.from_key("k")
    m = p.model("claude-opus-4-8+fast")
    assert m.model_id == "claude-opus-4-8+fast"
    assert m.spec.valid_latency_modes == ("fast",)


def test_anthropic_model_fast_tag_keeps_context_profile() -> None:
    """``+1m+fast`` resolves the ``+1m`` profile, not the base one."""
    p = Anthropic.from_key("k")
    m = p.model("claude-opus-4-8+1m+fast")
    assert m.max_request_tokens == 1_000_000


def test_anthropic_model_rejects_fast_tag_on_unsupported_model() -> None:
    p = Anthropic.from_key("k")
    with pytest.raises(ValueError, match="does not support fast mode"):
        _ = p.model("claude-haiku-4-5+fast")


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
    assert m.approx_text_tokens("a" * 28) == 11


def test_anthropic_model_pricing_exposed() -> None:
    p = Anthropic.from_key("k")
    m = p.model("claude-haiku-4-5")
    assert m.spec.prices[PriceCatalogProduct()].request > 0


def test_anthropic_fable_model_profile() -> None:
    p = Anthropic.from_key("k")
    assert Anthropic.DEFAULT_MODEL == "claude-opus-5"
    m = p.model("claude-fable-5")
    assert m.max_request_tokens == 1_000_000
    assert m.max_response_tokens == 128_000
    assert m.spec.prices[PriceCatalogProduct()].request == 10.0
    assert m.spec.prices[PriceCatalogProduct()].response == 50.0
    assert m.spec.prices[PriceCatalogProduct()].cache_write == 12.5
    assert m.spec.prices[PriceCatalogProduct()].cache_read == 1.0


def test_anthropic_fable_one_million_alias() -> None:
    p = Anthropic.from_key("k")
    m = p.model("claude-fable-5+1m")
    assert m.model_id == "claude-fable-5+1m"
    assert m.max_request_tokens == 1_000_000


def test_anthropic_sonnet_5_model_profile() -> None:
    p = Anthropic.from_key("k")
    m = p.model("claude-sonnet-5")
    assert m.max_request_tokens == 1_000_000
    assert m.max_response_tokens == 128_000
    assert m.spec.prices[PriceCatalogProduct()].request == 3.0
    assert m.spec.prices[PriceCatalogProduct()].response == 15.0
    assert m.valid_efforts == ("low", "medium", "high", "xhigh", "max")


def test_anthropic_sonnet_5_one_million_alias() -> None:
    p = Anthropic.from_key("k")
    m = p.model("claude-sonnet-5+1m")
    assert m.model_id == "claude-sonnet-5+1m"
    assert m.max_request_tokens == 1_000_000


@pytest.mark.asyncio
async def test_anthropic_actual_request_tokens_calls_count_tokens() -> None:
    """``actual_request_tokens`` routes through the SDK's ``count_tokens``."""
    p = Anthropic.from_key("k")
    m = p.model("claude-opus-4-7")
    fake_sdk = MagicMock()
    fake_result = MagicMock()
    fake_result.input_tokens = 42
    fake_sdk.messages.count_tokens = AsyncMock(return_value=fake_result)
    with patch.object(p, "get_sdk", AsyncMock(return_value=fake_sdk)):
        n = await m.actual_request_tokens(
            ModelRequest(messages=[UserMessage(text="ping")]),
        )
    assert n == 42
    # ``count_tokens`` rejects kwargs that ``create`` accepts -- verify
    # we stripped them before the call.
    await_args = fake_sdk.messages.count_tokens.await_args
    assert await_args is not None
    kwargs = await_args.kwargs
    assert "max_tokens" not in kwargs
    assert "temperature" not in kwargs
    assert kwargs["model"] == "claude-opus-4-7"


def test_anthropic_token_count_default_typing() -> None:
    # ``TokenCount`` from the model module must accept keyword construction.
    t = TokenCount(request=1, response=2)
    assert t.request == 1
    assert t.response == 2


def test_anthropic_model_supports_flags() -> None:
    p = Anthropic.from_key("k")
    m = p.model("claude-opus-4-7")
    assert m.supports_streaming is True
    assert m.supports_effort is True
    assert m.supports_cache_control is True
    assert m.supports_persistent_retry is True
    assert m.supports_context_management is False
    assert m.supports_account_auth is False
    assert m.spec.valid_service_tiers == ("auto", "standard_only")


def test_anthropic_model_supports_context_management_when_opted_in() -> None:
    p = Anthropic.from_key("k", server_side_context_management=True)
    m = p.model("claude-opus-4-7")
    assert m.supports_context_management is True


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


def test_anthropic_build_kwargs_enabled_thinking_respects_max_tokens_cap() -> None:
    # ``thinking="enabled"`` must never push ``max_tokens`` past the
    # model's output cap. opus-4-8's profile cap equals the API ceiling
    # (128k), so the old unconditional ``max_tok * 2`` emitted 256k and
    # the API rejected it. ``budget_tokens`` must stay strictly below
    # ``max_tokens`` per Anthropic's contract.
    p = Anthropic.from_key("k")
    m = p.model("claude-opus-4-8")
    req = ModelRequest(messages=[UserMessage(text="x")], thinking="enabled")
    kwargs = m._build_kwargs(req, [])
    max_tokens = IntCodec.coerce(kwargs["max_tokens"], 0)
    thinking = cast(dict[str, object], kwargs["thinking"])
    budget = IntCodec.coerce(thinking["budget_tokens"], 0)
    assert max_tokens <= m.max_response_tokens
    assert budget < max_tokens


def test_anthropic_valid_thinking_states_opus_4_6_all_six() -> None:
    """opus-4-6 streams readable thinking and accepts enabled: all six."""
    p = Anthropic.from_key("k")
    m = p.model("claude-opus-4-6")
    assert m.spec.valid_thinking_states == (
        "adaptive-show",
        "adaptive-hide",
        "on-show",
        "on-hide",
        "off-hide",
        "redact-hide",
    )


def test_anthropic_valid_thinking_states_opus_4_8_adaptive_only_no_text() -> None:
    """opus-4-8 returns empty thinking and rejects enabled: no -show, no on-*.

    Measured via API key: opus-4-8 streams zero ``thinking_delta`` chars
    (signed-but-empty block) and 400s on ``thinking.type=enabled``. So
    every ``-show`` state and every ``on-*`` state is unsatisfiable.
    """
    p = Anthropic.from_key("k")
    for model_id in ("claude-opus-4-8", "claude-opus-4-8+1m"):
        m = p.model(model_id)
        assert m.spec.valid_thinking_states == (
            "adaptive-hide",
            "off-hide",
            "redact-hide",
        ), model_id


def test_anthropic_valid_thinking_states_4_5_generation_enabled_only() -> None:
    """4-5 generation rejects ``adaptive`` (400); only ``on-*`` / ``off``.

    Measured via API key: opus-4-5, sonnet-4-5, haiku-4-5 all 400 on
    ``thinking.type=adaptive`` ('not supported on this model') and stream
    readable thinking under ``enabled``. So ``adaptive-*`` and
    ``redact-hide`` (which rides ``adaptive``) are excluded.
    """
    p = Anthropic.from_key("k")
    for model_id in ("claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5"):
        m = p.model(model_id)
        assert m.spec.valid_thinking_states == (
            "on-show",
            "on-hide",
            "off-hide",
        ), model_id


def test_anthropic_valid_latency_modes_fast_on_opus() -> None:
    p = Anthropic.from_key("k")
    assert p.model("claude-opus-4-8").spec.valid_latency_modes == ("fast",)
    assert p.model("claude-opus-4-8+1m").spec.valid_latency_modes == ("fast",)
    assert p.model("claude-fable-5").spec.valid_latency_modes == ()
    assert p.model("claude-sonnet-5").spec.valid_latency_modes == ()
    assert p.model("claude-haiku-4-5").spec.valid_latency_modes == ()


def test_anthropic_fast_latency_sets_speed_and_beta() -> None:
    p = Anthropic.from_key("k")
    m = p.model("claude-opus-4-8")
    req = ModelRequest(messages=[UserMessage(text="x")], latency="fast")
    kwargs = m._build_kwargs(req, [])
    body = cast(dict[str, object], kwargs["extra_body"])
    assert body["speed"] == "fast"
    headers = cast(dict[str, str], kwargs["extra_headers"])
    assert "fast-mode-2026-02-01" in headers["anthropic-beta"].split(",")


def test_anthropic_fast_latency_rejected_on_unsupported_model() -> None:
    p = Anthropic.from_key("k")
    m = p.model("claude-fable-5")
    req = ModelRequest(messages=[UserMessage(text="x")], latency="fast")
    with pytest.raises(ValueError, match="does not support fast mode"):
        m._build_kwargs(req, [])


def test_parse_response_bills_fast_when_server_reports_fast() -> None:
    raw = _build_anthropic_message(
        text="x", input_tokens=1_000_000, output_tokens=1_000_000, speed="fast"
    )
    resp = _parse_response(raw, _fast_spec())  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type] -- duck-typed SDK mock
    assert (resp.spend.request + resp.spend.cache_write + resp.spend.cache_read) == 10.0
    assert resp.spend.response == 50.0


def test_parse_response_bills_standard_when_server_falls_back() -> None:
    raw = _build_anthropic_message(
        text="x", input_tokens=1_000_000, output_tokens=1_000_000, speed="standard"
    )
    resp = _parse_response(raw, _fast_spec())  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type] -- duck-typed SDK mock
    assert (resp.spend.request + resp.spend.cache_write + resp.spend.cache_read) == 5.0
    assert resp.spend.response == 25.0


def test_anthropic_model_image_limits() -> None:
    p = Anthropic.from_key("k")
    m = p.model("claude-opus-4-7")
    # opus-4-7 is a high-resolution model: native long edge 2576 px (server
    # downscales above this), per the Anthropic Vision docs.
    assert m.max_image_dim == 2576
    assert m.max_image_bytes == 5 * 1024 * 1024


def test_anthropic_standard_model_image_dim_is_native_1568() -> None:
    """Pre-4.7 models cap the native long edge at 1568 px, not the high-res 2576."""
    m = Anthropic.from_key("k").model("claude-sonnet-4-5")
    assert m.max_image_dim == 1568


def test_anthropic_model_max_request_bytes() -> None:
    """Anthropic's request wire ceiling is ~32 MB (distinct from per-image)."""
    p = Anthropic.from_key("k")
    m = p.model("claude-opus-4-7")
    assert m.max_request_bytes == 32 * 1024 * 1024
    # The request ceiling is much larger than the per-image cap.
    assert m.max_request_bytes > m.max_image_bytes


def test_anthropic_image_byte_limits_read_from_spec_not_constant() -> None:
    """The three byte/image limits derive from the model SPEC, per-model.

    A constant ``return`` would report the same value for every model; reading
    the spec lets a future model with different vision/wire limits diverge
    without touching the provider class. Proven by a spec with distinct
    values flowing through to the model properties.
    """
    m = _AnthropicModel(
        provider=Anthropic.from_key("k"),
        model_id="claude-opus-4-7",
        spec=ModelSpec(
            context_limits=Limits(
                max_image_edge_px=4096,
                max_image_bytes=7 * 1024 * 1024,
                max_request_bytes=15 * 1024 * 1024,
            )
        ),
    )
    assert m.max_image_dim == 4096
    assert m.max_image_bytes == 7 * 1024 * 1024
    assert m.max_request_bytes == 15 * 1024 * 1024


def test_anthropic_model_is_context_overflow_non_api_status_error() -> None:
    p = Anthropic.from_key("k")
    m = p.model("claude-opus-4-7")
    assert m.is_context_overflow(RuntimeError("anything")) is False


def test_anthropic_byte_limit_not_classified_as_context_overflow() -> None:
    """A 413 byte-overflow must NOT classify as token-context overflow.

    The two limits are distinct: a larger-window model relieves token
    overflow but not the byte ceiling. ``is_context_overflow`` must
    return False so the byte case routes to byte-overflow recovery.
    """
    p = Anthropic.from_key("k")
    m = p.model("claude-opus-4-7")
    err = _request_too_large_error("Request exceeds the maximum size")
    assert m.is_context_overflow(err) is False


def test_anthropic_byte_413_without_structured_body_stays_byte() -> None:
    """A 413 byte error with no structured body still classifies as byte.

    The structured-body path already handles ``request_too_large``, but a
    413 whose body is not a parsed mapping (stringified error, proxy-mangled
    body) falls through to ``_matches_overflow_phrase`` on the raw text -- a
    text like "request too large: maximum exceeded by context window" would
    then mis-classify as token overflow. The shared ``is_request_too_large``
    guard keys on the 413 STATUS, closing that hole and matching every other
    provider's classifier.
    """
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(413, request=request)
    err = anthropic_sdk.APIStatusError(
        "request entity too large: maximum exceeded by context window",
        response=response,
        body=None,  # no structured body -> falls to substring path
    )
    p = Anthropic.from_key("k")
    m = p.model("claude-opus-4-7")
    assert m.is_context_overflow(err) is False


@pytest.mark.asyncio
async def test_anthropic_stream_request_too_large_raises_typed_error() -> None:
    """The stream path raises ``RequestTooLargeError`` chaining the original.

    The original ``APIStatusError`` must be preserved on ``__cause__`` so
    diagnostics and the retry classifier can inspect the provider error.
    """
    p = Anthropic.from_key("k")
    m = p.model("claude-opus-4-7")
    err = _request_too_large_error(
        "Request exceeds the maximum allowed number of bytes."
        " The maximum request size is 32 MB"
    )
    with (
        patch.object(p, "get_sdk", AsyncMock(return_value=MagicMock())),
        patch(
            "sagent.providers.anthropic.api._stream_impl",
            AsyncMock(side_effect=err),
        ),
        pytest.raises(RequestTooLargeError) as raised,
    ):
        await m.stream(ModelRequest(messages=[UserMessage(text="hi")]))
    assert raised.value.__cause__ is err, (
        "original APIStatusError must chain via __cause__"
    )


def _api_status_error(status_code: int, message: str) -> anthropic_sdk.APIStatusError:
    """Construct an APIStatusError as the SDK would, with an arbitrary status."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code, request=request)
    body = {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": message},
    }
    return anthropic_sdk.APIStatusError(message, response=response, body=body)


def _request_too_large_error(message: str) -> anthropic_sdk.APIStatusError:
    """Construct the 413 ``request_too_large`` APIStatusError the SDK yields."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(413, request=request)
    body = {
        "type": "error",
        "error": {"type": "request_too_large", "message": message},
    }
    return anthropic_sdk.APIStatusError(message, response=response, body=body)


class _BodyError(Exception):
    """Exception with an SDK-like ``body`` attribute."""

    def __init__(self, body: object) -> None:
        self.body = body
        super().__init__("x")


class _ResponseNotReadStatusError(Exception):
    """Exception mirroring SDK status metadata plus ResponseNotRead cause."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__("Anthropic streaming request failed")
        self.__cause__ = httpx.ResponseNotRead()


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


def test_anthropic_model_is_context_overflow_uses_structured_body() -> None:
    p = Anthropic.from_key("k")
    m = p.model("claude-opus-4-7")
    err = _api_status_error(400, "Request size exceeds model context window")
    err.args = ("unrelated invalid request",)

    assert m.is_context_overflow(err) is True


@pytest.mark.asyncio
async def test_anthropic_stream_uses_structured_overflow_body() -> None:
    p = Anthropic.from_key("k")
    m = p.model("claude-opus-4-7")
    err = _api_status_error(400, "Request size exceeds model context window")
    err.args = ("unrelated invalid request",)

    with (
        patch.object(p, "get_sdk", AsyncMock(return_value=MagicMock())),
        patch(
            "sagent.providers.anthropic.api._stream_impl",
            AsyncMock(side_effect=err),
        ),
        pytest.raises(PromptTooLongError),
    ):
        await m.stream(ModelRequest(messages=[UserMessage(text="hi")]))


@pytest.mark.asyncio
async def test_anthropic_raw_stream_reads_status_error_body() -> None:
    p = Anthropic.from_key("k")
    sdk = await p.get_sdk()
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(
        400,
        request=request,
        json={
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "Request size exceeds model context window",
            },
        },
    )
    with (
        patch.object(sdk._client, "send", AsyncMock(return_value=response)),
        pytest.raises(anthropic_sdk.APIStatusError) as raised,
    ):
        await _raw_message_stream(
            sdk,
            {
                "model": "claude-opus-4-7",
                "messages": [],
                "max_tokens": 1,
            },
        )

    assert raised.value.body == {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "Request size exceeds model context window",
        },
    }
    assert raised.value.status_code == 400


@pytest.mark.asyncio
async def test_anthropic_stream_status_body_overflow_triggers_prompt_too_long() -> None:
    p = Anthropic.from_key("k")
    sdk = await p.get_sdk()
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(
        400,
        request=request,
        json={
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "Request size exceeds model context window",
            },
        },
    )
    with (
        patch.object(sdk._client, "send", AsyncMock(return_value=response)),
        patch.object(p, "get_sdk", AsyncMock(return_value=sdk)),
    ):
        m = p.model("claude-opus-4-7")
        with pytest.raises(PromptTooLongError):
            await m.stream(ModelRequest(messages=[UserMessage(text="hi")]))


def test_response_not_read_provider_error_preserves_retry_status() -> None:
    p = Anthropic.from_key("k")
    m = p.model("claude-opus-4-7")
    err = _ResponseNotReadStatusError(529)

    assert is_retryable(err, m) is True
    assert error_status(err) == 529


@pytest.mark.asyncio
async def test_anthropic_stream_wraps_bare_response_not_read() -> None:
    # A raw ``httpx.ResponseNotRead`` escaping mid-stream is neither an
    # APIStatusError (so it dodges the overflow branch) nor a transport
    # error (so the retry classifier deems it fatal). It must be wrapped
    # in a user-facing error rather than surface as a bare exception.
    p = Anthropic.from_key("k")
    m = p.model("claude-opus-4-8")
    with (
        patch.object(p, "get_sdk", AsyncMock(return_value=MagicMock())),
        patch(
            "sagent.providers.anthropic.api._stream_impl",
            AsyncMock(side_effect=httpx.ResponseNotRead()),
        ),
        pytest.raises(StreamingResponseNotReadError),
    ):
        await m.stream(ModelRequest(messages=[UserMessage(text="hi")]))


@pytest.mark.asyncio
async def test_anthropic_stream_wraps_chained_response_not_read() -> None:
    # The SDK can chain ``ResponseNotRead`` under another exception while
    # formatting an error from an unread streaming body.
    p = Anthropic.from_key("k")
    m = p.model("claude-opus-4-8")
    wrapped = RuntimeError("stream formatting failed")
    wrapped.__cause__ = httpx.ResponseNotRead()
    with (
        patch.object(p, "get_sdk", AsyncMock(return_value=MagicMock())),
        patch(
            "sagent.providers.anthropic.api._stream_impl",
            AsyncMock(side_effect=wrapped),
        ),
        pytest.raises(StreamingResponseNotReadError),
    ):
        await m.stream(ModelRequest(messages=[UserMessage(text="hi")]))


@pytest.mark.asyncio
async def test_anthropic_stream_preserves_retryable_status_over_response_not_read() -> (
    None
):
    # A retryable 429 whose APIStatusError happens to chain a
    # ResponseNotRead must re-raise as the APIStatusError so the retry
    # classifier still sees the 429 and retries -- NOT be converted into a
    # fatal StreamingResponseNotReadError. The body-read crash is handled
    # downstream in retry.py::_response_body_excerpt.
    p = Anthropic.from_key("k")
    m = p.model("claude-opus-4-8")
    err = _api_status_error(429, "Rate limited")
    err.__cause__ = httpx.ResponseNotRead()
    with (
        patch.object(p, "get_sdk", AsyncMock(return_value=MagicMock())),
        patch(
            "sagent.providers.anthropic.api._stream_impl",
            AsyncMock(side_effect=err),
        ),
        pytest.raises(anthropic_sdk.APIStatusError) as raised,
    ):
        await m.stream(ModelRequest(messages=[UserMessage(text="hi")]))
    assert raised.value.status_code == 429
    assert not isinstance(raised.value, StreamingResponseNotReadError)


@pytest.mark.asyncio
async def test_anthropic_provider_close_sdk_closes_shared_sdk() -> None:
    """The provider owns the client, so the provider closes it."""
    p = Anthropic.from_key("k")
    fake_sdk = MagicMock()
    fake_sdk.close = AsyncMock()
    p._sdk = fake_sdk

    await p.close_sdk()

    fake_sdk.close.assert_awaited_once()
    assert p._sdk is None


@pytest.mark.asyncio
async def test_closing_one_model_leaves_a_sibling_model_usable() -> None:
    """The SDK belongs to the provider, so one model may not destroy it.

    ``sagent --advisor`` builds both models from one provider
    (``bin/cli.py``), and ``Agent.shutdown`` closes only its own model.
    Tearing down the shared client there strands the advisor mid-call.
    """
    p = Anthropic.from_key("k")
    model = p.model("claude-opus-4-7")
    _advisor = p.model("claude-sonnet-4-5")
    fake_sdk = MagicMock()
    fake_sdk.close = AsyncMock()
    p._sdk = fake_sdk

    await model.close()

    fake_sdk.close.assert_not_awaited()
    assert p._sdk is fake_sdk


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


def test_anthropic_provider_extra_headers_includes_context_management() -> None:
    """Modern models opt into the context-management beta unconditionally."""
    p = Anthropic.from_key("k")
    headers = p.extra_headers("claude-haiku-4-5")
    assert "context-management-2025-06-27" in headers.get("anthropic-beta", "")


def test_anthropic_provider_extra_headers_redact_thinking_opt_in() -> None:
    p = Anthropic.from_key("k", redact_thinking=True)
    headers = p.extra_headers("claude-opus-4-7")
    assert "redact-thinking-2026-02-12" in headers.get("anthropic-beta", "")


def test_anthropic_provider_extra_headers_redact_thinking_default_off() -> None:
    p = Anthropic.from_key("k")
    headers = p.extra_headers("claude-opus-4-7")
    assert "redact-thinking-2026-02-12" not in headers.get("anthropic-beta", "")


def test_anthropic_provider_extra_headers_unknown_model_empty() -> None:
    p = Anthropic.from_key("k")
    assert p.extra_headers("claude-3-opus-20240229") == {}


def test_anthropic_provider_extra_body_default_none() -> None:
    p = Anthropic.from_key("k")
    assert (
        p.extra_body(
            has_thinking=False,
            cache_cold=False,
            trigger_tokens=100_000,
            tools=(),
        )
        is None
    )


def test_build_kwargs_no_context_management_by_default() -> None:
    """Default is ``server_side_context_management=False`` -- no cm_config injected.

    Sagent's own client-side compactor handles budget. Server-side clearing
    must be explicitly opted in (see session bd952b0c audit for rationale).
    """
    p = Anthropic.from_key("k")
    model = p.model("claude-opus-4-7+1m")
    req = ModelRequest(messages=[UserMessage(text="hi")], system="s")
    msgs: list[MessageParam] = [
        cast(
            MessageParam, {"role": "user", "content": [{"type": "text", "text": "hi"}]}
        )
    ]
    kwargs = model._build_kwargs(req, msgs)
    body = kwargs.get("extra_body")
    assert body is None or "context_management" not in cast(dict[str, object], body)


def test_build_kwargs_includes_context_management_when_opted_in() -> None:
    """``server_side_context_management=True`` injects the ``clear_tool_uses`` config."""
    p = Anthropic.from_key("k", server_side_context_management=True)
    model = p.model("claude-opus-4-7+1m")
    req = ModelRequest(
        messages=[UserMessage(text="hi")],
        system="s",
        tools=None,
        max_response_tokens=128_000,
    )
    msgs: list[MessageParam] = [
        cast(
            MessageParam, {"role": "user", "content": [{"type": "text", "text": "hi"}]}
        )
    ]
    kwargs = model._build_kwargs(req, msgs)
    body = cast(dict[str, object], kwargs["extra_body"])
    cm = cast(dict[str, object], body["context_management"])
    edits = cast(list[dict[str, object]], cm["edits"])
    assert any(e["type"] == "clear_tool_uses_20250919" for e in edits)


def test_context_management_missing_clearable_results_defaults_unclearable() -> None:
    class LegacyTool:
        name = "Legacy"

    config = build_context_management(
        server_side_context_management=True,
        trigger_tokens=100_000,
        tools=cast(Sequence[Tool], [LegacyTool()]),
    )
    assert config is not None
    edit = cast(Mapping[str, object], config["edits"][0])
    assert edit["clear_tool_inputs"] == []
    assert edit["exclude_tools"] == ["Legacy"]


def test_build_kwargs_context_management_trigger_scales_with_context_window() -> None:
    """``trigger`` is half of the model's context window."""
    p = Anthropic.from_key("k", server_side_context_management=True)
    req = ModelRequest(messages=[UserMessage(text="hi")], system="s")
    msgs: list[MessageParam] = [
        cast(
            MessageParam, {"role": "user", "content": [{"type": "text", "text": "hi"}]}
        )
    ]

    m200 = p.model("claude-opus-4-7")
    kw200 = m200._build_kwargs(req, msgs)
    body200 = cast(dict[str, object], kw200["extra_body"])
    cm200 = cast(dict[str, object], body200["context_management"])
    edit200 = cast(list[dict[str, object]], cm200["edits"])[0]
    trig200 = cast(dict[str, object], edit200["trigger"])
    assert trig200["value"] == 100_000

    m1m = p.model("claude-opus-4-7+1m")
    kw1m = m1m._build_kwargs(req, msgs)
    body1m = cast(dict[str, object], kw1m["extra_body"])
    cm1m = cast(dict[str, object], body1m["context_management"])
    edit1m = cast(list[dict[str, object]], cm1m["edits"])[0]
    trig1m = cast(dict[str, object], edit1m["trigger"])
    assert trig1m["value"] == 500_000


def test_build_kwargs_preserves_provider_context_management() -> None:
    """Subclass-supplied ``context_management`` must NOT be clobbered.

    Regression test: ``Anthropic._build_kwargs`` previously always
    overwrote the ``context_management`` key with a hardcoded
    ``clear_tool_uses_20250919`` config, silently discarding whatever
    a subclass with its own env-gated, thinking-aware,
    whitelist-filtered policy had built.
    Bug: the subclass opted out of server clearing via env vars, but
    the base class injected aggressive clearing anyway.
    """
    # Custom config that mimics what a subclass builds when
    # ``USE_API_CLEAR_TOOL_RESULTS`` is set + thinking is on.
    custom_cm: MutableJSON = {
        "edits": [
            {"type": "clear_thinking_20251015", "keep": "all"},
            {
                "type": "clear_tool_uses_20250919",
                "trigger": {"type": "input_tokens", "value": 180_000},
                "clear_at_least": {"type": "input_tokens", "value": 140_000},
                "clear_tool_inputs": ["Bash", "Read", "Grep"],
                "exclude_tools": ["Edit", "Write"],
            },
        ],
    }

    class _CustomCmAnthropic(Anthropic):
        @override
        def extra_body(
            self,
            *,
            has_thinking: bool,
            cache_cold: bool,
            trigger_tokens: int,
            tools: Sequence[Tool],
        ) -> MutableJSON | None:
            del has_thinking, cache_cold, trigger_tokens, tools
            return {"context_management": custom_cm}

    # Opt the base into server-side clearing so the gate is open -- the
    # test then proves the subclass policy wins over the base default.
    p = _CustomCmAnthropic.from_key("k", server_side_context_management=True)
    model = p.model("claude-opus-4-7+1m")
    req = ModelRequest(messages=[UserMessage(text="hi")], system="s")
    msgs: list[MessageParam] = [
        cast(
            MessageParam, {"role": "user", "content": [{"type": "text", "text": "hi"}]}
        )
    ]
    kwargs = model._build_kwargs(req, msgs)
    body = cast(dict[str, object], kwargs["extra_body"])
    cm = cast(dict[str, object], body["context_management"])
    # The subclass config must round-trip untouched.
    assert cm is custom_cm
    edits = cast(list[dict[str, object]], cm["edits"])
    # Subscription's whitelist must survive (proves no merge/clobber).
    tool_edit = next(e for e in edits if e["type"] == "clear_tool_uses_20250919")
    assert tool_edit["clear_tool_inputs"] == ["Bash", "Read", "Grep"]
    assert tool_edit["exclude_tools"] == ["Edit", "Write"]
    trig = cast(dict[str, object], tool_edit["trigger"])
    assert trig["value"] == 180_000


def test_build_kwargs_no_context_management_for_unknown_model() -> None:
    """Unknown / older models don't get the context-management config."""
    assert not supports_native_context_management("claude-3-opus-20240229")


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
    from sagent.lib.testing.main import test_main

    test_main(__file__)
