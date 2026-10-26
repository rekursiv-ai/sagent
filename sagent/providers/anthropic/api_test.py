"""Tests for ``providers.anthropic``: wire-format translation + response parse."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from types import MappingProxyType
from typing import TYPE_CHECKING, cast, override
from unittest.mock import AsyncMock, MagicMock, patch

import json

from treekle import from_plain

import httpx2
import pytest

from sagent.agent.retry import error_status, is_retryable
from sagent.bin.cli import DEFAULT_TOOLS, resolve_tools
from sagent.catalog.anthropic import api
from sagent.catalog.table import ModelCatalog, ModelTable
from sagent.providers.anthropic.api import (
    Anthropic,
    _AnthropicModel,
    _assistant_blocks,
    _build_messages,
    _is_prompt_too_long_text,
    _parse_response,
    _raw_message_stream,
    _RawMessage,
    _tool_result_block,
    _tool_use_block,
    build_context_management,
)
from sagent.providers.lib.errors import (
    PolicyBlockedError,
    StreamingResponseNotReadError,
)
from sagent.providers.lib.id_remap import IdRemapper
from sagent.tools import tool
from sagent.types.capability import (
    ModelCapability,
    ModelLimits,
    ModelSettings,
    ThinkingCapability,
)
from sagent.types.cost import (
    PriceCatalog,
    PriceKey,
    TokenCount,
    TokenPrice,
)
from sagent.types.model import (
    ModelRequest,
    PromptTooLongError,
    RequestTooLargeError,
)
from sagent.types.runtime import (
    DETACHED_PLACEHOLDER,
    AssistantMessage,
    BytesMessage,
    ToolCall,
    ToolResult,
    UserMessage,
    WireMessage,
)


if TYPE_CHECKING:
    from anthropic.types import MessageParam
    from treekle import MutablePlainTree

    import anthropic as anthropic_sdk

    from sagent.types.capability import ThinkingBudget, ThinkingOutput
    from sagent.types.tools import Tool

else:
    from wrapt import lazy_import

    # The subject module defers the SDK (569ms cold); importing it eagerly
    # here would put that back onto every worker's collection.
    anthropic_sdk = lazy_import("anthropic")
    MessageParam = lazy_import("anthropic.types", "MessageParam")


@pytest.fixture(autouse=True)
def synthetic_catalog(monkeypatch: pytest.MonkeyPatch) -> ModelTable:
    row = ModelCapability(
        model_id="example-1.0",
        wire_model_id="wire-example",
        approx_chars_per_token=2.5,
        context={
            "": ModelLimits(max_request_tokens=600_000, max_response_tokens=96_000),
            "+200k": ModelLimits(
                max_request_tokens=200_000,
                max_response_tokens=48_000,
            ),
        },
        thinking=ThinkingCapability(
            effort=frozenset({"none", "low", "high"}),
            budget=frozenset({"none", "auto", "fixed"}),
            output=frozenset({"none", "text", "redacted"}),
        ),
        service_tier=frozenset({"auto", "default", "priority"}),
        prices=PriceCatalog(
            {
                PriceKey("auto"): TokenPrice(
                    request=3.0,
                    response=7.0,
                    cache_write=2.0,
                    cache_write_1h=4.0,
                    cache_read=0.5,
                ),
            },
        ),
    )
    table = ModelTable(
        rows=(
            row,
            replace(
                row,
                model_id="utility-example-1.0",
                service_tier=frozenset({"auto", "default"}),
            ),
        ),
        roles={"default": "example", "utility": "utility-example"},
    )
    monkeypatch.setattr(
        Anthropic,
        "catalog",
        ModelCatalog(models=table, transport=api()),
    )
    return table


def _free_model() -> _AnthropicModel:
    """Return a model whose every rate is zero -- cost is not what these assert."""
    m = Anthropic.from_key("k").model("wire-example")
    m._capability = replace(
        m.capability,
        prices=PriceCatalog(
            {
                PriceKey("auto"): TokenPrice(
                    request=0.0,
                    response=0.0,
                    cache_write=0.0,
                    cache_write_1h=0.0,
                    cache_read=0.0,
                ),
            },
        ),
    )
    return m


def _fast_model() -> _AnthropicModel:
    """Opus rates with a 2x priority tier, for the billed-speed tests."""
    m = Anthropic.from_key("k").model("wire-example")
    m._capability = replace(
        m.capability,
        prices=PriceCatalog(
            {
                PriceKey("auto"): TokenPrice(
                    request=5.0,
                    response=25.0,
                    cache_write=0.0,
                    cache_write_1h=0.0,
                    cache_read=0.0,
                ),
                PriceKey("priority"): TokenPrice(
                    request=10.0,
                    response=50.0,
                    cache_write=0.0,
                    cache_write_1h=0.0,
                    cache_read=0.0,
                ),
            },
        ),
    )
    m._settings = replace(m.settings, service_tier="priority")
    return m


def _make_request(
    messages: list[WireMessage],
) -> ModelRequest:
    return ModelRequest(messages=messages)


def test_normalized_anthropic_id_uses_vendor_id_on_the_wire() -> None:
    model = Anthropic.from_key("k").model("example-1.0")
    kwargs = model._build_kwargs(
        ModelRequest(messages=[UserMessage(text="hello")]),
        [],
    )
    assert kwargs["model"] == "wire-example"


def test_the_default_tools_serialize_as_request_json() -> None:
    # Tool schemas are frozen (nested mappingproxy); the SDK posts them with
    # ``json.dumps``, so a shallow copy fails at the first nested object.
    model = Anthropic.from_key("k").model("example-1.0")
    kwargs = model._build_kwargs(
        ModelRequest(
            messages=[UserMessage(text="hello")],
            tools=resolve_tools(list(DEFAULT_TOOLS)),
        ),
        [],
    )
    json.dumps(kwargs["tools"])


@pytest.mark.parametrize(
    "model_id",
    ["example", "example+200k", "utility"],
)
def test_no_model_sends_the_retired_1m_context_beta(model_id: str) -> None:
    """1M is the default on every 1M model; the beta header is no longer needed."""
    model = Anthropic.from_key("k").model(model_id)
    kwargs = model._build_kwargs(ModelRequest(messages=[]), [])
    headers = cast(dict[str, str], kwargs["extra_headers"])
    betas = headers["anthropic-beta"].split(",")
    assert "context-1m-2025-08-07" not in betas
    assert "context-management-2025-06-27" in betas


@pytest.mark.parametrize(
    ("model_id", "budget", "output", "display"),
    [
        ("example-1.0", "auto", "text", "summarized"),
        ("example", "auto", "text", "summarized"),
        ("example", "auto", "redacted", "omitted"),
        ("example", "fixed", "text", "summarized"),
        ("utility-example-1.0", "fixed", "redacted", "omitted"),
    ],
)
def test_thinking_output_selects_the_wire_display(
    model_id: str,
    budget: ThinkingBudget,
    output: ThinkingOutput,
    display: str,
) -> None:
    """4.7-and-later default ``display`` to omitted, so ``text`` must ask for it."""
    model = Anthropic.from_key("k").model(model_id)
    model.settings.thinking_budget = budget
    model.settings.thinking_output = output
    kwargs = model._build_kwargs(ModelRequest(messages=[]), [])
    thinking = cast(dict[str, object], kwargs["thinking"])
    assert thinking["display"] == display


def test_thinking_without_an_output_choice_sends_no_display() -> None:
    model = Anthropic.from_key("k").model("example-1.0")
    model.settings.thinking_budget = "auto"
    kwargs = model._build_kwargs(ModelRequest(messages=[]), [])
    assert kwargs["thinking"] == {"type": "adaptive"}


def test_no_thinking_sends_no_display() -> None:
    """``display`` is invalid with thinking off; nothing to display."""
    model = Anthropic.from_key("k").model("example")
    model.settings.thinking_output = "text"
    kwargs = model._build_kwargs(ModelRequest(messages=[]), [])
    assert "thinking" not in kwargs


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


def test_build_messages_marks_a_cache_point_ahead_of_the_last_message() -> None:
    document = UserMessage(text="long shared document", cache_point=True)
    question = UserMessage(text="the question")
    msgs = _build_messages(_make_request([document, question]), cache_ttl="1h")
    # Two user messages in a row coalesce into one wire message; both blocks
    # keep their own cache_control.
    blocks = [b for m in msgs for b in cast(list[dict[str, object]], m["content"])]
    marks = [(b.get("text"), b.get("cache_control")) for b in blocks]
    assert marks == [
        ("long shared document", {"type": "ephemeral", "ttl": "1h"}),
        ("the question", {"type": "ephemeral", "ttl": "1h"}),
    ]
    plain = _build_messages(_make_request([UserMessage(text="a"), question]))
    first = cast(list[dict[str, object]], plain[0]["content"])[0]
    assert "cache_control" not in first


def _look(topic: str) -> str:
    """Look a topic up."""
    return topic


def _peek(topic: str) -> str:
    """Peek at a topic."""
    return topic


def test_a_tool_gate_change_withdraws_and_restores_tools_where_it_happened() -> None:
    """Every schema stays in ``tools``; the gate changes as a system message.

    The change sits after the message that made it, so the history before it
    stays byte-identical and cached, and the last-message mark stays on the
    user's turn, not on the change.
    """
    request = ModelRequest(
        messages=[
            UserMessage(text="a", offered_tools=("Look",)),
            AssistantMessage(text="ok"),
            UserMessage(text="b", offered_tools=("Look", "Peek")),
        ],
        tools=[tool(_look, name="Look"), tool(_peek, name="Peek")],
    )
    msgs = _build_messages(request)
    assert [m["role"] for m in msgs] == [
        "user",
        "system",
        "assistant",
        "user",
        "system",
    ]
    peek = {"type": "tool_reference", "name": "Peek"}
    assert msgs[1]["content"] == [{"type": "tool_removal", "tool": peek}]
    assert msgs[4]["content"] == [{"type": "tool_addition", "tool": peek}]
    last_user = cast(list[dict[str, object]], msgs[3]["content"])
    assert "cache_control" in last_user[-1]
    model = Anthropic.from_key("k").model("example-1.0")
    headers = cast(dict[str, str], model._build_kwargs(request, msgs)["extra_headers"])
    assert "inline-tools-2026-09-15" in headers["anthropic-beta"]


def test_an_unchanged_offer_sends_no_tool_change() -> None:
    request = ModelRequest(
        messages=[UserMessage(text="a", offered_tools=("Look", "Peek"))],
        tools=[tool(_look, name="Look"), tool(_peek, name="Peek")],
    )
    msgs = _build_messages(request)
    assert [m["role"] for m in msgs] == ["user"]
    model = Anthropic.from_key("k").model("example-1.0")
    headers = cast(
        dict[str, str],
        model._build_kwargs(request, msgs).get("extra_headers") or {},
    )
    assert "inline-tools" not in headers.get("anthropic-beta", "")


def test_build_messages_keeps_cache_marks_within_anthropic_s_four() -> None:
    """The API rejects a request with more than 4 ``cache_control`` blocks.

    Each system block and the last message take one; the newest
    ``cache_point``s keep what is left, as the ones nearest the reusable end.
    """
    docs = [UserMessage(text=f"doc {i}", cache_point=True) for i in range(3)]
    cases: list[tuple[tuple[str, ...], list[str]]] = [
        (("a", "b"), ["doc 2"]),
        (("a", "", "c"), ["doc 2"]),
        (("a",), ["doc 1", "doc 2"]),
        (("a", "b", "c"), []),
    ]
    for system, kept in cases:
        msgs = _build_messages(
            ModelRequest(messages=[*docs, UserMessage(text="q")], system=system),
        )
        marked = [
            b.get("text")
            for m in msgs
            for b in cast(list[dict[str, object]], m["content"])
            if "cache_control" in b
        ]
        assert marked == [*kept, "q"]


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
        _make_request([asst, ToolResult(call_id="orig-X", content="ok")]),
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
        thinking_blocks=({"type": "thinking", "thinking": "...", "signature": "s"},),
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
        thinking_blocks=({"type": "thinking", "thinking": "...", "signature": "s"},),
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


def test_assistant_blocks_drops_unsigned_thinking() -> None:
    """A thinking block without a signature (Gemini, ACP) cannot validate."""
    asst = AssistantMessage(
        text="ok",
        thinking_blocks=(
            {"type": "thinking", "thinking": "gemini thought"},
            {"type": "thinking", "thinking": "kept", "signature": "sig"},
        ),
    )
    blocks = _assistant_blocks(asst, IdRemapper("toolu_"))
    assert [(b["type"], b.get("thinking")) for b in blocks] == [
        ("thinking", "kept"),
        ("text", None),
    ]


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


def test_tool_result_block_keeps_pdf_attachment() -> None:
    out = _tool_result_block(
        ToolResult(
            call_id="x",
            content="report",
            attachments=(BytesMessage(data=b"%PDF-1.4", descriptor="application/pdf"),),
        ),
        IdRemapper("toolu_"),
        max_image_dim=8000,
        max_image_bytes=5 * 1024 * 1024,
    )
    content = cast(list[dict[str, object]], out["content"])
    assert [block["type"] for block in content] == ["text", "document"]


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
) -> _RawMessage:
    """Return a duck-typed object mimicking ``anthropic.types.Message``."""
    # Use the real anthropic SDK classes via the lazy-imported module so
    # ``isinstance`` checks inside ``_parse_response`` succeed.
    text_blocks: list[object] = []
    if text:
        text_blocks.append(anthropic_sdk.types.TextBlock(type="text", text=text))
    for tc_id, tc_name, args in tool_calls:
        text_blocks.append(
            anthropic_sdk.types.ToolUseBlock(
                type="tool_use",
                id=tc_id,
                name=tc_name,
                input=args,
            ),
        )
    # ``speed`` is an extra the SDK model keeps but does not declare.
    usage = anthropic_sdk.types.Usage.model_validate(
        {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": cache_creation,
            "cache_read_input_tokens": cache_read,
            "speed": speed,
        },
    )
    msg = MagicMock()
    msg.content = text_blocks
    msg.usage = usage
    msg.stop_reason = stop_reason
    msg.stop_sequence = None
    msg.id = "msg_xyz"
    msg._request_id = "req_xyz"
    return cast(_RawMessage, msg)


def test_parse_response_text_only() -> None:
    raw = _build_anthropic_message(text="hi", input_tokens=5, output_tokens=2)
    resp = _parse_response(raw, _free_model())
    assert resp.message.text == "hi"
    assert resp.stop_reason == "model_finished"
    assert resp.tokens.request == 5
    assert resp.tokens.response == 2


def test_parse_response_tool_call_extracted() -> None:
    raw = _build_anthropic_message(
        tool_calls=(("toolu_xyz", "Bash", {"cmd": "ls"}),),
        stop_reason="tool_use",
    )
    resp = _parse_response(raw, _free_model())
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
    resp = _parse_response(raw, _free_model())
    assert [c.name for c in resp.message.tool_calls] == ["Bash"]


def test_parse_response_accepts_hyphenated_tool_name() -> None:
    raw = _build_anthropic_message(
        tool_calls=(("toolu_1", "mcp-server_tool", {}),),
        stop_reason="tool_use",
    )
    resp = _parse_response(raw, _free_model())
    assert [c.name for c in resp.message.tool_calls] == ["mcp-server_tool"]


def test_parse_response_all_placeholder_calls_is_finished() -> None:
    """Filtering every tool_use leaves prose, not an interrupted stream."""
    raw = _build_anthropic_message(
        text="here is my context",
        tool_calls=(("toolu_bad", "$FUNCTION_NAME", {}),),
        stop_reason="tool_use",
    )
    resp = _parse_response(raw, _free_model())
    assert resp.message.tool_calls == ()
    assert resp.stop_reason == "model_finished"


def test_parse_response_cache_tokens_split_correctly() -> None:
    raw = _build_anthropic_message(
        input_tokens=1000,
        output_tokens=100,
        cache_creation=200,
        cache_read=400,
    )
    model = Anthropic.from_key("k").model("wire-example")
    model._capability = replace(
        model.capability,
        prices=PriceCatalog(
            {
                PriceKey("auto"): TokenPrice(
                    request=1.0,
                    response=2.0,
                    cache_write=4.0,
                    cache_read=0.5,
                    cache_write_1h=0.0,
                ),
            },
        ),
    )
    resp = _parse_response(raw, model)
    assert resp.tokens.cache_write == 200
    assert resp.tokens.cache_read == 400
    # Input cost = 1000*1 + 200*4 + 400*0.5 = 2000 / 1M = 0.002.
    assert (
        resp.spend.request + resp.spend.cache_write + resp.spend.cache_read
    ) == pytest.approx(0.002)


def test_parse_response_carries_message_and_request_ids() -> None:
    raw = _build_anthropic_message()
    resp = _parse_response(raw, _free_model())
    assert resp.message_id == "msg_xyz"
    assert resp.request_id == "req_xyz"


def test_anthropic_from_key() -> None:
    p = Anthropic.from_key("sk-ant-test")
    assert p._api_key == "sk-ant-test"


def test_anthropic_from_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API key not configured"):
        Anthropic.from_env()


def test_anthropic_from_env_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-key")
    p = Anthropic.from_env()
    assert p._api_key == "sk-ant-key"


def test_anthropic_model_known_id_returns_backend(
    synthetic_catalog: ModelTable,
) -> None:
    p = Anthropic.from_key("k")
    m = p.model("example-1.0")
    expected = synthetic_catalog["example"] & api()
    assert m.capability == expected
    assert m.limits == expected.context[""]


def test_anthropic_model_unknown_id_raises() -> None:
    p = Anthropic.from_key("k")
    with pytest.raises(ValueError, match="Unknown model"):
        _ = p.model("unknown-claude")


def _call(fn: Callable[..., object], *args: object, **kwargs: object) -> object:
    """Call ``fn`` past its signature, as a caller with a typo would."""
    return fn(*args, **kwargs)


def test_anthropic_model_rejects_unknown_provider_options() -> None:
    p = Anthropic.from_key("k")
    with pytest.raises(TypeError, match="typoo"):
        _ = _call(p.model, "example", typoo=True)


def test_anthropic_model_strips_context_tag_for_profile_lookup() -> None:
    """``+200k`` selects the smaller context profile."""
    p = Anthropic.from_key("k")
    m = p.model("wire-example+200k")
    assert m.limits.max_request_tokens == 200_000


def test_anthropic_default_model_resolves() -> None:
    p = Anthropic.from_key("k")
    m = p.model()
    assert m.capability.model_id == p.catalog.resolve("default")[0].model_id


@pytest.mark.parametrize("role", ["best", "default", "everyday", "utility"])
def test_anthropic_roles_resolve(role: str, synthetic_catalog: ModelTable) -> None:
    p = Anthropic.from_key("k")
    assert p.model(role).capability.model_id == synthetic_catalog[role].model_id


def test_anthropic_context_selection_preserves_tag(
    synthetic_catalog: ModelTable,
) -> None:
    p = Anthropic.from_key("k")
    full = p.model("example")
    small = p.model("example+200k")
    assert full.limits == synthetic_catalog["example"].context[""]
    assert small.limits == synthetic_catalog["example"].context["+200k"]
    assert small.tagged_model_id == "example-1.0+200k"


def test_anthropic_subscription_property_false_on_api_key() -> None:
    assert Anthropic.from_key("k").subscription is False


def test_anthropic_model_is_context_overflow_via_prompt_too_long() -> None:
    m = Anthropic.from_key("k").model("example")
    assert m.is_context_overflow(PromptTooLongError("too long")) is True


def test_anthropic_model_token_estimate_uses_profile_approx_chars_per_token() -> None:
    m = Anthropic.from_key("k").model("example")
    assert m.approx_text_tokens("a" * 28) == 11


def test_anthropic_model_pricing_exposed(synthetic_catalog: ModelTable) -> None:
    m = Anthropic.from_key("k").model("example")
    assert m.capability.prices is synthetic_catalog["example"].prices


@pytest.mark.asyncio
async def test_anthropic_actual_request_tokens_calls_count_tokens() -> None:
    """``actual_request_tokens`` routes through the SDK's ``count_tokens``."""
    p = Anthropic.from_key("k")
    m = p.model("wire-example")
    fake_sdk = MagicMock()
    fake_result = MagicMock()
    fake_result.input_tokens = 42
    count_tokens = AsyncMock(return_value=fake_result)
    messages = MagicMock()
    messages.count_tokens = count_tokens
    fake_sdk.messages = messages
    with patch.object(p, "get_sdk", AsyncMock(return_value=fake_sdk)):
        n = await m.actual_request_tokens(
            ModelRequest(messages=[UserMessage(text="ping")]),
        )
    assert n == 42
    # ``count_tokens`` rejects kwargs that ``create`` accepts -- verify
    # we stripped them before the call.
    await_args = count_tokens.await_args
    assert await_args is not None
    kwargs = cast(dict[str, object], await_args.kwargs)
    assert "max_tokens" not in kwargs
    assert "temperature" not in kwargs
    assert kwargs["model"] == "wire-example"


def test_anthropic_token_count_default_typing() -> None:
    # ``TokenCount`` from the model module must accept keyword construction.
    t = TokenCount(request=1, response=2)
    assert t.request == 1
    assert t.response == 2


def test_anthropic_model_capability() -> None:
    p = Anthropic.from_key("k")
    m = p.model("wire-example")
    assert m.capability.thinking.effort != frozenset({"none"})
    assert m.capability.cache_ttl_sec == frozenset({300.0, 3600.0})
    assert m.capability.retries_internally is True
    assert m.capability.manage_context_server_side == frozenset({False, True})
    assert m.capability.account_auth is False
    assert m.capability.service_tier == frozenset({"auto", "default", "priority"})


def test_anthropic_model_supports_context_management_when_opted_in() -> None:
    p = Anthropic.from_key("k", server_side_context_management=True)
    m = p.model("wire-example")
    assert m.capability.manage_context_server_side == frozenset({False, True})


def test_anthropic_build_kwargs_sends_default_tier_as_standard_only() -> None:
    """The Messages API ``service_tier`` enum is ``auto`` | ``standard_only``."""
    p = Anthropic.from_key("k")
    m = p.model("wire-example")
    m._settings = replace(m.settings, service_tier="default")
    kwargs = m._build_kwargs(ModelRequest(messages=[UserMessage(text="x")]), [])
    assert kwargs["service_tier"] == "standard_only"


def test_anthropic_build_kwargs_omits_the_default_tier() -> None:
    """``auto`` is "let the vendor pick"; sending it pins nothing."""
    p = Anthropic.from_key("k")
    m = p.model("wire-example")
    kwargs = m._build_kwargs(ModelRequest(messages=[UserMessage(text="x")]), [])
    assert "service_tier" not in kwargs


def test_anthropic_build_kwargs_enabled_thinking_respects_max_tokens_cap() -> None:
    # Doubling the reply budget must not exceed the row's output ceiling.
    p = Anthropic.from_key("k")
    m = p.model("wire-example")
    m._settings = replace(m.settings, thinking_budget="fixed")
    kwargs = m._build_kwargs(ModelRequest(messages=[UserMessage(text="x")]), [])
    max_tokens = from_plain(kwargs["max_tokens"], int)
    thinking = cast(dict[str, object], kwargs["thinking"])
    budget = from_plain(thinking["budget_tokens"], int)
    assert max_tokens <= m.limits.max_response_tokens
    assert budget < max_tokens


def test_anthropic_build_kwargs_fixed_thinking_budget_meets_api_floor() -> None:
    """Anthropic rejects ``budget_tokens < 1024``, however small the reply."""
    m = Anthropic.from_key("k").model("wire-example")
    m._settings = replace(m.settings, thinking_budget="fixed")
    kwargs = m._build_kwargs(
        ModelRequest(messages=[UserMessage(text="x")], max_response_tokens=500),
        [],
    )
    thinking = cast(dict[str, object], kwargs["thinking"])
    budget = from_plain(thinking["budget_tokens"], int)
    assert budget >= 1024
    assert budget < from_plain(kwargs["max_tokens"], int)


def test_anthropic_priority_tier_sets_speed_and_beta() -> None:
    """``speed`` rides ``extra_body``: the stream path is a raw POST.

    A first-class ``speed=`` exists only on the SDK's beta endpoints, so
    dropping this injection silently bills standard. Fast mode is ``speed``
    alone: ``priority`` is not a value of the API's ``service_tier`` enum.
    """
    p = Anthropic.from_key("k")
    m = p.model("wire-example")
    m._settings = replace(m.settings, service_tier="priority")
    kwargs = m._build_kwargs(ModelRequest(messages=[UserMessage(text="x")]), [])
    body = cast(dict[str, object], kwargs["extra_body"])
    assert body["speed"] == "fast"
    headers = cast(dict[str, str], kwargs["extra_headers"])
    assert "fast-mode-2026-02-01" in headers["anthropic-beta"].split(",")
    assert "service_tier" not in kwargs


def test_anthropic_priority_tier_is_unselectable_on_an_unsupported_model() -> None:
    """The capability rejects it up front rather than billing standard."""
    m = Anthropic.from_key("k").model("utility-example")
    with pytest.raises(ValueError, match="service_tier"):
        m.settings.service_tier = "priority"


def test_parse_response_bills_fast_when_server_reports_fast() -> None:
    raw = _build_anthropic_message(
        text="x",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        speed="fast",
    )
    resp = _parse_response(raw, _fast_model())
    assert (resp.spend.request + resp.spend.cache_write + resp.spend.cache_read) == 10.0
    assert resp.spend.response == 50.0


def test_parse_response_bills_standard_when_server_falls_back() -> None:
    raw = _build_anthropic_message(
        text="x",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        speed="standard",
    )
    resp = _parse_response(raw, _fast_model())
    assert (resp.spend.request + resp.spend.cache_write + resp.spend.cache_read) == 5.0
    assert resp.spend.response == 25.0


def test_anthropic_image_byte_limits_read_from_the_row_not_a_constant() -> None:
    """The three byte/image limits come from the capability, per-model.

    A constant ``return`` would report one value for every model; reading
    the row lets a future model with different vision/wire limits diverge
    without touching the provider class.
    """
    row = ModelCapability(
        model_id="wire-example",
        context=MappingProxyType(
            {
                "": ModelLimits(
                    max_image_edge_px=4096,
                    max_image_bytes=7 * 1024 * 1024,
                    max_request_bytes=15 * 1024 * 1024,
                ),
            },
        ),
    )
    m = _AnthropicModel(
        provider=Anthropic.from_key("k"),
        capability=row,
        settings=ModelSettings.narrowest(row),
    )
    assert m.limits.max_image_edge_px == 4096
    assert m.limits.max_image_bytes == 7 * 1024 * 1024
    assert m.limits.max_request_bytes == 15 * 1024 * 1024


def test_anthropic_model_is_context_overflow_non_api_status_error() -> None:
    p = Anthropic.from_key("k")
    m = p.model("wire-example")
    assert m.is_context_overflow(RuntimeError("anything")) is False


def test_anthropic_byte_limit_not_classified_as_context_overflow() -> None:
    """A 413 byte-overflow must NOT classify as token-context overflow.

    The two limits are distinct: a larger-window model relieves token
    overflow but not the byte ceiling. ``is_context_overflow`` must
    return False so the byte case routes to byte-overflow recovery.
    """
    p = Anthropic.from_key("k")
    m = p.model("wire-example")
    err = _request_too_large_error("Request exceeds the maximum size")
    assert m.is_context_overflow(err) is False


def test_anthropic_byte_413_without_structured_body_stays_byte() -> None:
    """A 413 byte error with no structured body still classifies as byte.

    The structured-body path already handles ``request_too_large``, but a
    413 whose body is not a parsed mapping (stringified error, proxy-mangled
    body) falls through to ``is_context_overflow_text`` on the raw text -- a
    text like "request too large: maximum exceeded by context window" would
    then mis-classify as token overflow. The shared ``is_request_too_large``
    guard keys on the 413 STATUS, closing that hole and matching every other
    provider's classifier.
    """
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(413, request=request)
    err = anthropic_sdk.APIStatusError(
        "request entity too large: maximum exceeded by context window",
        response=response,
        body=None,  # No structured body -> falls to substring path.
    )
    p = Anthropic.from_key("k")
    m = p.model("wire-example")
    assert m.is_context_overflow(err) is False


@pytest.mark.asyncio
async def test_anthropic_stream_request_too_large_raises_typed_error() -> None:
    """The stream path raises ``RequestTooLargeError`` chaining the original.

    The original ``APIStatusError`` must be preserved on ``__cause__`` so
    diagnostics and the retry classifier can inspect the provider error.
    """
    p = Anthropic.from_key("k")
    m = p.model("wire-example")
    err = _request_too_large_error(
        "Request exceeds the maximum allowed number of bytes."
        " The maximum request size is 32 MB",
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


def _api_status_error(
    status_code: int,
    message: str,
    *,
    request_id: str = "",
) -> anthropic_sdk.APIStatusError:
    """Construct an APIStatusError as the SDK would, with an arbitrary status."""
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(
        status_code,
        request=request,
        headers={"request-id": request_id} if request_id else None,
    )
    body = {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": message},
    }
    return anthropic_sdk.APIStatusError(message, response=response, body=body)


def _request_too_large_error(message: str) -> anthropic_sdk.APIStatusError:
    """Construct the 413 ``request_too_large`` APIStatusError the SDK yields."""
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(413, request=request)
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
        self.__cause__ = httpx2.ResponseNotRead()


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
    m = p.model("wire-example")
    err = _api_status_error(414, "Request size exceeds model context window")
    assert m.is_context_overflow(err) is True


def test_anthropic_model_is_context_overflow_uses_structured_body() -> None:
    p = Anthropic.from_key("k")
    m = p.model("wire-example")
    err = _api_status_error(400, "Request size exceeds model context window")
    err.args = ("unrelated invalid request",)

    assert m.is_context_overflow(err) is True


@pytest.mark.asyncio
async def test_anthropic_stream_uses_structured_overflow_body() -> None:
    p = Anthropic.from_key("k")
    m = p.model("wire-example")
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
@pytest.mark.parametrize(
    "message",
    [
        "Output blocked by content filtering policy",
        "Fable 5's safeguards flagged this message under the Usage Policy.",
        "This request has been blocked by our content filters.",
        "Request rejected for safety reasons.",
    ],
)
async def test_anthropic_stream_raises_policy_blocked_error(message: str) -> None:
    """Safeguard 400s preserve Anthropic's reason and stop same-context retries."""
    p = Anthropic.from_key("k")
    m = p.model("wire-example")
    request_id = "req_policy_block"
    err = _api_status_error(400, message, request_id=request_id)

    with (
        patch.object(p, "get_sdk", AsyncMock(return_value=MagicMock())),
        patch(
            "sagent.providers.anthropic.api._stream_impl",
            AsyncMock(side_effect=err),
        ),
        pytest.raises(PolicyBlockedError) as raised,
    ):
        await m.stream(ModelRequest(messages=[UserMessage(text="hi")]))

    assert raised.value.provider_message == message
    assert raised.value.request_id == request_id
    assert message in str(raised.value)
    assert request_id in str(raised.value)
    assert "do not retry" in str(raised.value).lower()
    assert raised.value.__cause__ is err


@pytest.mark.asyncio
async def test_anthropic_auth_retry_normalizes_policy_error() -> None:
    """The second request must pass through the same error boundary as the first."""
    p = Anthropic.from_key("k")
    m = p.model("wire-example")
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(401, request=request)
    auth_error = anthropic_sdk.AuthenticationError(
        "Unauthorized",
        response=response,
        body=None,
    )
    policy_error = _api_status_error(
        400,
        "This request has been blocked by our content filters.",
    )

    with (
        patch.object(p, "get_sdk", AsyncMock(return_value=MagicMock())),
        patch.object(p, "handle_auth_error", AsyncMock()) as handle_auth,
        patch(
            "sagent.providers.anthropic.api._stream_impl",
            AsyncMock(side_effect=[auth_error, policy_error]),
        ),
        pytest.raises(PolicyBlockedError),
    ):
        await m.stream(ModelRequest(messages=[UserMessage(text="hi")]))

    handle_auth.assert_awaited_once()


@pytest.mark.asyncio
async def test_anthropic_stream_preserves_unrelated_invalid_request() -> None:
    """An ordinary invalid request must retain the SDK's diagnostic taxonomy."""
    p = Anthropic.from_key("k")
    m = p.model("wire-example")
    err = _api_status_error(400, "tools.0.name is invalid")

    with (
        patch.object(p, "get_sdk", AsyncMock(return_value=MagicMock())),
        patch(
            "sagent.providers.anthropic.api._stream_impl",
            AsyncMock(side_effect=err),
        ),
        pytest.raises(anthropic_sdk.APIStatusError) as raised,
    ):
        await m.stream(ModelRequest(messages=[UserMessage(text="hi")]))

    assert raised.value is err


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "Request blocked because this key lacks access",
        "cache_control rejected: at most 4 blocks are allowed by the caching policy",
        "Request rejected under your organization's data retention policy",
        "Unknown beta flag; see the API versioning policy",
    ],
)
async def test_anthropic_stream_does_not_misclassify_unrelated_block(
    message: str,
) -> None:
    """A rejection citing a non-safety policy is not a provider safety decision."""
    p = Anthropic.from_key("k")
    m = p.model("wire-example")
    err = _api_status_error(400, message)

    with (
        patch.object(p, "get_sdk", AsyncMock(return_value=MagicMock())),
        patch(
            "sagent.providers.anthropic.api._stream_impl",
            AsyncMock(side_effect=err),
        ),
        pytest.raises(anthropic_sdk.APIStatusError) as raised,
    ):
        await m.stream(ModelRequest(messages=[UserMessage(text="hi")]))

    assert raised.value is err


@pytest.mark.asyncio
async def test_anthropic_sdk_connect_timeout_outlasts_a_network_stall() -> None:
    """The SDK default connect timeout is 5s; a Wi-Fi stall outlasted it.

    Only ``send_with_retry`` retries: the SDK's own two retries made one stalled
    attempt three 30s connects, so five attempts blocked for ~450s.
    """
    sdk = await Anthropic.from_key("k").get_sdk()
    timeout = sdk.timeout
    assert isinstance(timeout, httpx2.Timeout)
    assert (timeout.connect, timeout.read) == (30.0, 600.0)
    assert sdk.max_retries == 0
    await sdk.close()


@pytest.mark.asyncio
async def test_anthropic_raw_stream_reads_status_error_body() -> None:
    p = Anthropic.from_key("k")
    sdk = await p.get_sdk()
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(
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
        patch.object(
            sdk._client,
            "send",
            AsyncMock(return_value=response),
        ) as send,
        pytest.raises(anthropic_sdk.APIStatusError) as raised,
    ):
        await _raw_message_stream(
            sdk,
            {
                "model": "wire-example",
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
    call = send.await_args
    assert call is not None
    sent_request = cast(httpx2.Request, call.args[0])
    assert sent_request.method == "POST"
    assert call.kwargs == {"stream": True}


@pytest.mark.asyncio
async def test_anthropic_stream_status_body_overflow_triggers_prompt_too_long() -> None:
    p = Anthropic.from_key("k")
    sdk = await p.get_sdk()
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(
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
        m = p.model("wire-example")
        with pytest.raises(PromptTooLongError):
            await m.stream(ModelRequest(messages=[UserMessage(text="hi")]))


def test_response_not_read_provider_error_preserves_retry_status() -> None:
    p = Anthropic.from_key("k")
    m = p.model("wire-example")
    err = _ResponseNotReadStatusError(529)

    assert is_retryable(err, m) is True
    assert error_status(err) == 529


@pytest.mark.asyncio
async def test_anthropic_stream_wraps_bare_response_not_read() -> None:
    # A raw ``httpx2.ResponseNotRead`` escaping mid-stream is neither an
    # APIStatusError (so it dodges the overflow branch) nor a transport
    # error (so the retry classifier deems it fatal). It must be wrapped
    # in a user-facing error rather than surface as a bare exception.
    p = Anthropic.from_key("k")
    m = p.model("wire-example")
    with (
        patch.object(p, "get_sdk", AsyncMock(return_value=MagicMock())),
        patch(
            "sagent.providers.anthropic.api._stream_impl",
            AsyncMock(side_effect=httpx2.ResponseNotRead()),
        ),
        pytest.raises(StreamingResponseNotReadError),
    ):
        await m.stream(ModelRequest(messages=[UserMessage(text="hi")]))


@pytest.mark.asyncio
async def test_anthropic_stream_wraps_chained_response_not_read() -> None:
    # The SDK can chain ``ResponseNotRead`` under another exception while
    # formatting an error from an unread streaming body.
    p = Anthropic.from_key("k")
    m = p.model("wire-example")
    wrapped = RuntimeError("stream formatting failed")
    wrapped.__cause__ = httpx2.ResponseNotRead()
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
    m = p.model("wire-example")
    err = _api_status_error(429, "Rate limited")
    err.__cause__ = httpx2.ResponseNotRead()
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
    close = AsyncMock()
    fake_sdk.close = close
    p._sdk = cast("anthropic_sdk.AsyncAnthropic", fake_sdk)

    await p.close_sdk()

    close.assert_awaited_once()
    assert p._sdk is None


@pytest.mark.asyncio
async def test_closing_one_model_leaves_a_sibling_model_usable() -> None:
    """The SDK belongs to the provider, so one model may not destroy it.

    ``sagent --advisor`` builds both models from one provider
    (``bin/cli.py``), and ``Agent.shutdown`` closes only its own model.
    Tearing down the shared client there strands the advisor mid-call.
    """
    p = Anthropic.from_key("k")
    model = p.model("wire-example")
    _advisor = p.model("utility-example")
    fake_sdk = MagicMock()
    close = AsyncMock()
    fake_sdk.close = close
    p._sdk = cast("anthropic_sdk.AsyncAnthropic", fake_sdk)

    await model.close()

    close.assert_not_awaited()
    assert p._sdk is fake_sdk


def test_anthropic_model_is_retryable_provider_error_rate_limit() -> None:
    p = Anthropic.from_key("k")
    m = p.model("wire-example")
    err = _BodyError({"type": "rate_limit_error"})
    assert m.is_retryable_provider_error(err) is True


def test_anthropic_model_is_retryable_provider_error_nested_api_error() -> None:
    p = Anthropic.from_key("k")
    m = p.model("wire-example")
    err = _BodyError({"type": "error", "error": {"type": "api_error"}})
    assert m.is_retryable_provider_error(err) is True


def test_anthropic_model_is_retryable_provider_error_nested_invalid_request() -> None:
    p = Anthropic.from_key("k")
    m = p.model("wire-example")
    err = _BodyError({"type": "error", "error": {"type": "invalid_request_error"}})
    assert m.is_retryable_provider_error(err) is False


def test_anthropic_model_is_retryable_provider_error_unrelated_type() -> None:
    p = Anthropic.from_key("k")
    m = p.model("wire-example")
    err = _BodyError({"type": "permanent_error"})
    assert m.is_retryable_provider_error(err) is False


def test_anthropic_model_is_retryable_provider_error_no_body_attr() -> None:
    p = Anthropic.from_key("k")
    m = p.model("wire-example")
    assert m.is_retryable_provider_error(RuntimeError("x")) is False


def test_anthropic_provider_extra_headers_for_1m_omit_the_context_beta() -> None:
    p = Anthropic.from_key("k")
    model = p.model("example")
    headers = p.extra_headers(model.capability, model.settings)
    assert "context-1m" not in headers.get("anthropic-beta", "")


def test_anthropic_provider_extra_headers_includes_context_management() -> None:
    """Modern models opt into the context-management beta unconditionally."""
    p = Anthropic.from_key("k")
    model = p.model("wire-example")
    headers = p.extra_headers(model.capability, model.settings)
    assert "context-management-2025-06-27" in headers.get("anthropic-beta", "")


def test_anthropic_provider_extra_headers_redact_thinking_opt_in() -> None:
    p = Anthropic.from_key("k", redact_thinking=True)
    model = p.model("wire-example")
    headers = p.extra_headers(model.capability, model.settings)
    assert "redact-thinking-2026-02-12" in headers.get("anthropic-beta", "")


def test_anthropic_provider_extra_headers_redact_thinking_default_off() -> None:
    p = Anthropic.from_key("k")
    model = p.model("wire-example")
    headers = p.extra_headers(model.capability, model.settings)
    assert "redact-thinking-2026-02-12" not in headers.get("anthropic-beta", "")


def test_anthropic_constructor_defaults_keep_optional_features_off() -> None:
    """Direct construction has the same conservative defaults as ``from_key``."""
    p = Anthropic(api_key="k")
    model = p.model("wire-example")

    assert p.server_side_context_management is False
    headers = p.extra_headers(model.capability, model.settings)
    assert "redact-thinking-2026-02-12" not in headers.get("anthropic-beta", "")


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
    model = p.model("wire-example")
    req = ModelRequest(messages=[UserMessage(text="hi")], system=("s",))
    msgs: list[MessageParam] = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
    ]
    kwargs = model._build_kwargs(req, msgs)
    body = kwargs.get("extra_body")
    assert body is None or "context_management" not in cast(dict[str, object], body)


def test_build_kwargs_includes_context_management_when_opted_in() -> None:
    """``server_side_context_management=True`` injects the ``clear_tool_uses`` config."""
    p = Anthropic.from_key("k", server_side_context_management=True)
    model = p.model("wire-example")
    req = ModelRequest(
        messages=[UserMessage(text="hi")],
        system=("s",),
        max_response_tokens=128_000,
    )
    msgs: list[MessageParam] = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
    ]
    kwargs = model._build_kwargs(req, msgs)
    body = cast(dict[str, object], kwargs["extra_body"])
    cm = cast(dict[str, object], body["context_management"])
    edits = cast(list[dict[str, object]], cm["edits"])
    assert any(e["type"] == "clear_tool_uses_20250919" for e in edits)


def test_context_management_unclearable_tool_is_excluded() -> None:
    class UnclearableTool:
        name = "Legacy"
        clearable_results = False

    config = build_context_management(
        server_side_context_management=True,
        trigger_tokens=100_000,
        tools=[UnclearableTool()],
    )
    assert config is not None
    edit = cast(Mapping[str, object], config["edits"][0])
    assert edit["clear_tool_inputs"] == []
    assert edit["exclude_tools"] == ["Legacy"]


def test_build_kwargs_context_management_trigger_scales_with_context_window() -> None:
    """``trigger`` is half of the model's context window."""
    p = Anthropic.from_key("k", server_side_context_management=True)
    req = ModelRequest(messages=[UserMessage(text="hi")], system=("s",))
    msgs: list[MessageParam] = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
    ]

    m200 = p.model("wire-example+200k")
    kw200 = m200._build_kwargs(req, msgs)
    body200 = cast(dict[str, object], kw200["extra_body"])
    cm200 = cast(dict[str, object], body200["context_management"])
    edit200 = cast(list[dict[str, object]], cm200["edits"])[0]
    trig200 = cast(dict[str, object], edit200["trigger"])
    assert trig200["value"] == 100_000

    m1m = p.model("wire-example")
    kw1m = m1m._build_kwargs(req, msgs)
    body1m = cast(dict[str, object], kw1m["extra_body"])
    cm1m = cast(dict[str, object], body1m["context_management"])
    edit1m = cast(list[dict[str, object]], cm1m["edits"])[0]
    trig1m = cast(dict[str, object], edit1m["trigger"])
    assert trig1m["value"] == m1m.limits.max_request_tokens // 2


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
    custom_cm: dict[str, MutablePlainTree] = {
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
        ) -> dict[str, MutablePlainTree] | None:
            del has_thinking, cache_cold, trigger_tokens, tools
            return {"context_management": custom_cm}

    # Opt the base into server-side clearing so the gate is open -- the
    # test then proves the subclass policy wins over the base default.
    p = _CustomCmAnthropic.from_key("k", server_side_context_management=True)
    model = p.model("wire-example")
    req = ModelRequest(messages=[UserMessage(text="hi")], system=("s",))
    msgs: list[MessageParam] = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
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


def test_anthropic_provider_build_system_marks_the_static_prefix() -> None:
    """Requests sharing a system prompt but diverging after it reuse its cache.

    Anthropic writes cache entries only at a breakpoint; a request whose only
    mark is its last message never matches a sibling's differing tail.
    """
    p = Anthropic.from_key("k")
    assert p.build_system(("hello",)) == [
        {"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}},
    ]
    assert p.build_system(("hello",), cache_ttl="1h") == [
        {
            "type": "text",
            "text": "hello",
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        },
    ]


def test_anthropic_provider_build_system_empty_returns_not_given() -> None:
    p = Anthropic.from_key("k")
    out = p.build_system(())
    # ``anthropic.NOT_GIVEN`` is a sentinel; type is anthropic.NotGiven.
    assert out is anthropic_sdk.NOT_GIVEN


def test_more_than_three_system_blocks_still_fit_the_four_cache_marks() -> None:
    """The last message takes one mark; the system blocks may take only three."""
    request = ModelRequest(
        messages=[UserMessage(text="q")],
        system=("a", "b", "c", "d"),
    )
    blocks = Anthropic.from_key("k").build_system(request.system)
    assert isinstance(blocks, list)
    assert ["cache_control" in b for b in blocks] == [False, True, True, True]
    marked = [
        b
        for m in _build_messages(request)
        for b in cast(list[dict[str, object]], m["content"])
        if "cache_control" in b
    ]
    assert len(marked) == 1


def test_anthropic_model_max_response_tokens_from_profile(
    synthetic_catalog: ModelTable,
) -> None:
    p = Anthropic.from_key("k")
    m = p.model("wire-example")
    assert (
        m.limits.max_response_tokens
        == synthetic_catalog["example"].context[""].max_response_tokens
    )


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
