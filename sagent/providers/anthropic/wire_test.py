"""Tests for the Anthropic Messages API translation: requests out, replies in."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from itertools import pairwise
from types import MappingProxyType
from typing import TYPE_CHECKING

import base64
import io

from PIL import Image
from treekle import PlainTree, from_plain

import pypdfium2 as pdfium
import pytest

from sagent.agent.agent import Agent
from sagent.agent.agents_md import AgentsMdConfig
from sagent.model.capability import ModelLimits, ModelSettings, PruningPolicy
from sagent.model.cost import TokenCount
from sagent.model.exceptions import (
    AuthError,
    BillingError,
    ContextOverflowError,
    OverloadedError,
    RateLimitedError,
    RefusedError,
    RequestTooLargeError,
)
from sagent.model.scripted import ScriptedProvider, call, reply
from sagent.model.types import Request, TextChunk
from sagent.providers.anthropic import wire
from sagent.providers.anthropic.api import models
from sagent.providers.types import UsageWindow
from sagent.session.types import (
    AgentToAgentMessage,
    AssistantMessage,
    Attachment,
    ContextClear,
    ContextState,
    IncompleteRecord,
    SystemMessage,
    Thinking,
    TokenUsage,
    ToolCall,
    TurnContext,
    UncategorizedRecord,
    UncategorizedToolResult,
    UserMessage,
)
from sagent.tools.display import ToolDisplay


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from pathlib import Path

    from sagent.model.capability import ModelCapability, ThinkingBudget
    from sagent.model.cost import ServiceTier
    from sagent.session.types import SessionRecord
    from sagent.tools.types import ToolContext


def _model(
    name: str = "opus-5.5",
    *,
    section: str = "anthropic",
) -> tuple[ModelCapability, ModelSettings]:
    return models(section).resolve(name)


def _encode(
    records: Sequence[SessionRecord] = (),
    *,
    capability: ModelCapability | None = None,
    settings: ModelSettings | None = None,
    system: Sequence[str] = (),
    tools: Sequence[_Echo] = (),
    cache_expired: bool = False,
) -> tuple[dict[str, object], list[str]]:
    opus, default = _model()
    chosen = default if settings is None else settings
    return wire.encode(
        Request(records=tuple(records), tools=tuple(tools)),
        capability=opus if capability is None else capability,
        settings=chosen,
        system=wire.system_blocks(system, settings=chosen),
        cache_expired=cache_expired,
    )


def _messages(
    records: Sequence[SessionRecord],
    *,
    settings: ModelSettings | None = None,
) -> list[dict[str, object]]:
    body, _ = _encode(records, settings=settings)
    return from_plain(body["messages"], list[dict[str, object]])


def _blocks(message: Mapping[str, object]) -> list[dict[str, object]]:
    return from_plain(message["content"], list[dict[str, object]])


def _png(width: int, height: int) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (width, height), "red").save(out, format="PNG")
    return out.getvalue()


@dataclass(frozen=True, slots=True, kw_only=True)
class _Echo:
    """A tool the model may call; these tests only offer it."""

    name: str = "Echo"
    description: str = "Say the text back."
    schema: Mapping[str, PlainTree] = field(
        default_factory=lambda: MappingProxyType(
            {"type": "object", "properties": {"text": {"type": "string"}}},
        ),
    )
    display: ToolDisplay = field(default_factory=ToolDisplay)

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        del arguments
        return self.name

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        del context, arguments
        return None

    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> UncategorizedToolResult:
        del context
        return UncategorizedToolResult(call_id="", content=str(arguments.get("text")))


# ---- encode: the request body ------------------------------------------------


def test_encode_names_the_vendor_id_and_streams() -> None:
    body, betas = _encode([UserMessage(content="hi")])
    assert body["model"] == "claude-opus-5-5"
    assert body["max_tokens"] == 128_000
    assert body["stream"] is True
    # A new model thinks as Claude Code has it: adaptively, at its default
    # effort, keeping its old thinking; and asks for no optional feature.
    assert betas == ["context-management-2025-06-27"]
    assert body.keys() == {
        "model",
        "max_tokens",
        "messages",
        "stream",
        "thinking",
        "context_management",
        "output_config",
    }
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "medium"}


@pytest.mark.parametrize("budget", ["none", "fixed"])
def test_encode_asks_no_more_than_the_room_the_request_leaves(
    budget: ThinkingBudget,
) -> None:
    capability, settings = _model("opus-4.5")
    settings = replace(settings, thinking_budget=budget)
    body, _ = wire.encode(
        Request(records=(UserMessage(content="hi"),), max_response_tokens=9_000),
        capability=capability,
        settings=settings,
        system=[],
    )
    assert body["max_tokens"] == 9_000
    if budget == "fixed":
        assert body["thinking"] == {"type": "enabled", "budget_tokens": 4_500}


def test_encode_sends_the_system_blocks_and_tools_it_is_given() -> None:
    body, _ = _encode(
        [UserMessage(content="hi")],
        system=["Be terse."],
        tools=[_Echo()],
    )
    assert body["system"] == [
        {"type": "text", "text": "Be terse.", "cache_control": {"type": "ephemeral"}},
    ]
    assert body["tools"] == [
        {
            "name": "Echo",
            "description": "Say the text back.",
            "input_schema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
            },
        },
    ]


def test_user_text_is_one_user_message_marked_for_the_cache() -> None:
    assert _messages([UserMessage(content="hi")]) == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}},
            ],
        },
    ]


def test_only_the_last_block_of_the_conversation_is_marked() -> None:
    messages = _messages(
        [
            UserMessage(content="one"),
            AssistantMessage(content="two"),
            UserMessage(content="three"),
        ],
    )
    assert messages == [
        {"role": "user", "content": [{"type": "text", "text": "one"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "two"}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "three",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
        },
    ]


def test_an_hour_long_cache_asks_for_an_hour_everywhere() -> None:
    _, settings = _model()
    settings = replace(settings, cache_ttl_sec=3600.0)
    body, _ = _encode([UserMessage(content="hi")], settings=settings, system=["S"])
    hour = {"type": "ephemeral", "ttl": "1h"}
    assert body["system"] == [{"type": "text", "text": "S", "cache_control": hour}]
    assert body["messages"] == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "hi", "cache_control": hour}],
        },
    ]


def test_records_of_one_role_share_one_message() -> None:
    messages = _messages(
        [
            UserMessage(content="go"),
            ToolCall(call_id="a", name="Echo", arguments={"text": "1"}),
            ToolCall(call_id="b", name="Echo", arguments={"text": "2"}),
            UncategorizedToolResult(call_id="a", content="1"),
            UncategorizedToolResult(call_id="b", content="2"),
            UserMessage(content="next"),
        ],
    )
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert [b["type"] for b in _blocks(messages[2])] == [
        "tool_result",
        "tool_result",
        "text",
    ]


def test_tool_ids_are_renamed_consistently_within_a_request() -> None:
    """Another vendor's ids (``fc_*``) fail Anthropic's id grammar."""
    messages = _messages(
        [
            UserMessage(content="go"),
            ToolCall(call_id="fc_1", name="Echo", arguments={"text": "x"}),
            ToolCall(call_id="fc_2", name="Echo"),
            UncategorizedToolResult(call_id="fc_2", content="b"),
            UncategorizedToolResult(call_id="fc_1", content="a"),
        ],
    )
    calls, results = _blocks(messages[1]), _blocks(messages[2])
    assert calls == [
        {"type": "tool_use", "id": "toolu_0", "name": "Echo", "input": {"text": "x"}},
        {"type": "tool_use", "id": "toolu_1", "name": "Echo", "input": {}},
    ]
    assert [r["tool_use_id"] for r in results] == ["toolu_1", "toolu_0"]


def test_a_tool_result_carries_its_error_flag_and_its_text() -> None:
    messages = _messages(
        [
            UserMessage(content="go"),
            ToolCall(call_id="a", name="Echo"),
            ToolCall(call_id="b", name="Echo"),
            UncategorizedToolResult(call_id="a", content="boom", is_error=True),
            UncategorizedToolResult(call_id="b"),
        ],
    )
    failed, empty = _blocks(messages[2])
    assert failed == {
        "type": "tool_result",
        "tool_use_id": "toolu_0",
        "is_error": True,
        "content": [{"type": "text", "text": "boom"}],
    }
    # A result with nothing to say sends no content at all.
    assert {key: value for key, value in empty.items() if key != "cache_control"} == {
        "type": "tool_result",
        "tool_use_id": "toolu_1",
        "is_error": False,
    }


def test_a_harness_note_rides_the_user_side_as_a_reminder() -> None:
    messages = _messages(
        [
            UserMessage(content="hi"),
            SystemMessage(content="Saved."),
            SystemMessage(content=None),
            SystemMessage(content=""),
            TurnContext(cwd="/work", model="opus"),
            TurnContext(),
            AssistantMessage(content="ok"),
        ],
    )
    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
                {
                    "type": "text",
                    "text": "<system-reminder>\nSaved.\n</system-reminder>",
                },
                {
                    "type": "text",
                    "text": "<system-reminder>\nWorking directory: /work\nModel: opus\n"
                    "</system-reminder>",
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "ok", "cache_control": {"type": "ephemeral"}},
            ],
        },
    ]


def test_an_agents_message_is_the_user_side_naming_its_sender() -> None:
    png = _png(2, 2)
    messages = _messages(
        [
            AgentToAgentMessage(sender="lead", content="status?"),
            AgentToAgentMessage(content="unsigned"),
            AgentToAgentMessage(),
            AgentToAgentMessage(
                sender="lead",
                attachments=(Attachment(mime_descriptor="image/png", data=png),),
            ),
            AssistantMessage(content="ok"),
        ],
    )
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == [
        {"type": "text", "text": "[from lead]: status?"},
        {"type": "text", "text": "unsigned"},
        {"type": "text", "text": "[from lead]: "},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(png).decode(),
            },
        },
    ]


def test_signed_thinking_returns_with_its_signature_and_sealed_stays_sealed() -> None:
    messages = _messages(
        [
            UserMessage(content="q"),
            Thinking(content="hmm", encrypted="sig"),
            Thinking(encrypted="opaque"),
            AssistantMessage(content="a"),
        ],
    )
    assert _blocks(messages[1])[:2] == [
        {"type": "thinking", "thinking": "hmm", "signature": "sig"},
        {"type": "redacted_thinking", "data": "opaque"},
    ]


@pytest.mark.parametrize(
    "thinking",
    [
        # Gemini's and OpenAI Chat's reasoning carries no seal at all.
        Thinking(content="unsigned"),
        Thinking(summary="summary only"),
        # OpenAI's Responses API seals its reasoning beside its reasoning id.
        Thinking(encrypted="gAAAA-openai", extra={"reasoning_id": "rs_1"}),
        Thinking(
            content="visible",
            encrypted="gAAAA-openai",
            extra={"reasoning_id": "rs_2"},
        ),
    ],
)
def test_reasoning_anthropic_did_not_sign_is_left_out(thinking: Thinking) -> None:
    """Anthropic rejects a thinking block whose signature it did not make."""
    messages = _messages(
        [UserMessage(content="q"), thinking, AssistantMessage(content="a")],
    )
    assert [b["type"] for b in _blocks(messages[1])] == ["text"]


@pytest.mark.parametrize(
    ("thinking", "block"),
    [
        (
            Thinking(content="hmm", encrypted="sig"),
            {"type": "thinking", "thinking": "hmm", "signature": "sig"},
        ),
        (Thinking(encrypted="opaque"), {"type": "redacted_thinking", "data": "opaque"}),
    ],
)
def test_an_assistant_turn_ending_on_thinking_gets_a_placeholder(
    thinking: Thinking,
    block: dict[str, object],
) -> None:
    """The API rejects an assistant turn whose last block is thinking."""
    messages = _messages(
        [UserMessage(content="q"), thinking, UserMessage(content="go on")],
    )
    assert messages[1]["content"] == [block, {"type": "text", "text": "."}]


def test_a_context_clear_sends_its_summary_then_its_history() -> None:
    messages = _messages(
        [
            ContextClear(
                summary="Earlier: we fixed the bug.",
                history=(
                    UserMessage(content="a"),
                    ToolCall(call_id="fc_9", name="Echo"),
                    UncategorizedToolResult(call_id="fc_9", content="b"),
                ),
            ),
            ToolCall(call_id="fc_9", name="Echo"),
        ],
    )
    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Earlier: we fixed the bug."},
                {"type": "text", "text": "a"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_0", "name": "Echo", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_0",
                    "is_error": False,
                    "content": [{"type": "text", "text": "b"}],
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_0",  # The same id after the clear as before it.
                    "name": "Echo",
                    "input": {},
                    "cache_control": {"type": "ephemeral"},
                },
            ],
        },
    ]
    assert _messages([ContextClear(history=(UserMessage(content="a"),))]) == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}},
            ],
        },
    ]


def test_attachments_inside_a_context_clear_are_sent() -> None:
    pdf = Attachment(mime_descriptor="application/pdf", data=b"%PDF-1.4")
    messages = _messages(
        [ContextClear(history=(UserMessage(content="a", attachments=(pdf,)),))],
    )
    assert [b["type"] for b in _blocks(messages[0])] == ["text", "document"]


def test_bookkeeping_records_send_nothing() -> None:
    assert (
        _messages(
            [
                TokenUsage(),
                ContextState(kind="x", content="y"),
                UncategorizedRecord(kind="x"),
                IncompleteRecord(text="{"),
                AssistantMessage(content=""),
                UserMessage(),
            ],
        )
        == []
    )


# ---- encode: optional features -----------------------------------------------


def test_adaptive_thinking_shows_the_summary_asked_for() -> None:
    _, settings = _model()
    settings = replace(settings, thinking_budget="auto")
    body, _ = _encode([UserMessage(content="q")], settings=settings)
    assert body["thinking"] == {"type": "adaptive"}
    settings = replace(settings, thinking_output="text")
    body, betas = _encode([UserMessage(content="q")], settings=settings)
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert "redact-thinking-2026-02-12" not in betas


def test_redacted_thinking_is_omitted_and_names_its_beta() -> None:
    _, settings = _model()
    settings = replace(settings, thinking_budget="auto", thinking_output="redacted")
    body, betas = _encode([UserMessage(content="q")], settings=settings)
    assert body["thinking"] == {"type": "adaptive", "display": "omitted"}
    assert "redact-thinking-2026-02-12" in betas


def test_a_fixed_budget_leaves_half_the_reply_for_the_answer() -> None:
    capability, settings = _model("opus-4.6")
    body, _ = _encode(
        [UserMessage(content="q")],
        capability=capability,
        settings=replace(settings, thinking_budget="fixed"),
    )
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 64_000}


def test_a_fixed_budget_never_falls_below_the_api_floor() -> None:
    capability, settings = _model("opus-4.6")
    small = replace(
        capability,
        limits=replace(capability.limits, max_response_tokens=1_500),
    )
    body, _ = _encode(
        [UserMessage(content="q")],
        capability=small,
        settings=replace(settings, thinking_budget="fixed"),
    )
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 1024}


def test_effort_is_sent_as_output_config() -> None:
    _, settings = _model()
    settings = replace(settings, thinking_budget="none", thinking_effort="high")
    body, _ = _encode([UserMessage(content="q")], settings=settings)
    assert body["output_config"] == {"effort": "high"}
    assert "thinking" not in body


@pytest.mark.parametrize(
    ("tier", "fields", "beta"),
    [
        ("default", {"service_tier": "standard_only"}, None),
        ("priority", {"speed": "fast"}, "fast-mode-2026-02-01"),
        ("auto", {}, None),
    ],
)
def test_a_service_tier_is_spelled_as_the_api_takes_it(
    tier: ServiceTier,
    fields: dict[str, object],
    beta: str | None,
) -> None:
    _, settings = _model()
    settings = replace(settings, service_tier=tier, thinking_budget="none")
    body, betas = _encode([UserMessage(content="q")], settings=settings)
    assert {k: body[k] for k in ("service_tier", "speed") if k in body} == fields
    assert betas == ([] if beta is None else [beta])


def test_old_thinking_is_kept_while_the_cache_holds_it() -> None:
    _, settings = _model()
    settings = replace(settings, thinking_budget="auto")
    body, betas = _encode([UserMessage(content="q")], settings=settings)
    assert body["context_management"] == {
        "edits": [{"type": "clear_thinking_20251015", "keep": "all"}],
    }
    assert "context-management-2025-06-27" in betas
    body, _ = _encode(
        [UserMessage(content="q")],
        settings=settings,
        cache_expired=True,
    )
    assert body["context_management"] == {
        "edits": [
            {
                "type": "clear_thinking_20251015",
                "keep": {"type": "thinking_turns", "value": 1},
            },
        ],
    }


def test_pruning_asks_the_server_to_clear_old_tool_uses() -> None:
    _, settings = _model()
    settings = replace(
        settings,
        thinking_budget="none",
        pruning=PruningPolicy(
            trigger_tokens=100_000,
            target_tokens=99_500,
            keep_recent=2,
            keep_tools=frozenset({"Read", "Bash"}),
        ),
    )
    body, _ = _encode([UserMessage(content="q")], settings=settings)
    assert "context_management" not in body  # The server was not asked to manage.
    settings = replace(settings, manage_context_server_side=True)
    body, betas = _encode([UserMessage(content="q")], settings=settings)
    assert body["context_management"] == {
        "edits": [
            {
                "type": "clear_tool_uses_20250919",
                "trigger": {"type": "input_tokens", "value": 100_000},
                # Clearing less than the minimum is not worth the cache it breaks.
                "clear_at_least": {"type": "input_tokens", "value": 1_000},
                "keep": {"type": "tool_uses", "value": 2},
                "exclude_tools": ["Bash", "Read"],
            },
        ],
    }
    assert betas == ["context-management-2025-06-27"]
    settings = replace(
        settings,
        pruning=PruningPolicy(trigger_tokens=100_000, target_tokens=40_000),
    )
    body, _ = _encode([UserMessage(content="q")], settings=settings)
    assert isinstance(body["context_management"], dict)
    assert body["context_management"]["edits"][0]["clear_at_least"] == {
        "type": "input_tokens",
        "value": 60_000,
    }


def test_a_context_windows_betas_ride_every_request() -> None:
    capability, settings = _model()
    limits = replace(capability.limits, request_betas=frozenset({"window-1"}))
    _, betas = _encode(
        [UserMessage(content="q")],
        capability=replace(capability, limits=limits),
        settings=replace(settings, thinking_budget="none"),
    )
    assert betas == ["window-1"]


# ---- system blocks and content -----------------------------------------------


def test_system_blocks_drop_empty_parts_and_mark_the_last_three() -> None:
    _, settings = _model()
    blocks = wire.system_blocks(["a", "", "b", "c", "d", "e"], settings=settings)
    assert [b["text"] for b in blocks] == ["a", "b", "c", "d", "e"]
    assert ["cache_control" in b for b in blocks] == [False, False, True, True, True]
    assert wire.system_blocks([], settings=settings) == []


@pytest.mark.parametrize(
    ("mime", "kind"),
    [
        ("image/png", "PNG"),
        ("image/jpeg", "JPEG"),
        ("image/gif", "GIF"),
        ("image/webp", "WEBP"),
    ],
)
def test_content_sends_images_and_pdfs_after_the_text(mime: str, kind: str) -> None:
    capability, _ = _model()
    out = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(out, format=kind)
    picture = out.getvalue()
    blocks = wire.content(
        "look",
        [
            # What Anthropic cannot read is left out, and the rest still sent.
            Attachment(mime_descriptor="text/plain", data=b"x"),
            Attachment(mime_descriptor=mime, data=picture),
            Attachment(mime_descriptor="image/svg+xml", data=b"<svg/>"),
            Attachment(mime_descriptor="application/pdf", data=b"%PDF-1.4"),
        ],
        limits=capability.limits,
    )
    assert blocks == [
        {"type": "text", "text": "look"},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime,
                "data": base64.b64encode(picture).decode(),
            },
        },
        {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": base64.b64encode(b"%PDF-1.4").decode(),
            },
        },
    ]


def _pdf(*sizes: tuple[int, int]) -> bytes:
    doc = pdfium.PdfDocument.new()
    for width, height in sizes:
        _ = doc.new_page(width, height)
    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()


@pytest.mark.parametrize(
    ("attachment", "tokens"),
    [
        (Attachment(mime_descriptor="image/png", data=_png(1500, 750)), 666),
        (Attachment(mime_descriptor="image/png", data=_png(600, 300)), 240),
        # A page is up to 3,000 tokens of text, and an image of its points.
        (
            Attachment(
                mime_descriptor="application/pdf",
                data=_pdf((612, 792), (1_584, 1_224)),
            ),
            3_000 + 612 * 792 // 750 + 3_000 + 1_000 * 772 // 750,
        ),
        (Attachment(mime_descriptor="text/plain", data=b"x" * 4_000), 0),
    ],
    ids=["shrunk-image", "small-image", "pdf", "unsent"],
)
def test_an_attachment_bills_by_the_vision_formula(
    attachment: Attachment,
    tokens: int,
) -> None:
    limits = ModelLimits(max_image_edge_px=1_000)
    assert wire.attachment_tokens(attachment, limits=limits) == tokens


def test_content_shrinks_an_image_to_the_models_limits() -> None:
    capability, _ = _model()
    limits = replace(capability.limits, max_image_edge_px=4)
    blocks = wire.content(
        "",
        [Attachment(mime_descriptor="image/png", data=_png(8, 2))],
        limits=limits,
    )
    assert len(blocks) == 1
    source = blocks[0]["source"]
    assert isinstance(source, dict)
    assert source["media_type"] == "image/webp"
    assert isinstance(source["data"], str)
    assert Image.open(io.BytesIO(base64.b64decode(source["data"]))).size == (4, 1)


def test_attachments_ride_user_messages_and_tool_results() -> None:
    png = _png(2, 2)
    image = Attachment(mime_descriptor="image/png", data=png)
    messages = _messages(
        [
            UserMessage(content="see", attachments=(image,)),
            ToolCall(call_id="a", name="Echo"),
            UncategorizedToolResult(call_id="a", attachments=(image,)),
        ],
    )
    assert [b["type"] for b in _blocks(messages[0])] == ["text", "image"]
    assert [b["type"] for b in _blocks(_blocks(messages[2])[0])] == ["image"]


# ---- the cached prefix -------------------------------------------------------


def _unmarked(tree: PlainTree) -> PlainTree:
    """Return ``tree`` without its cache marks, which move by design."""
    if isinstance(tree, Mapping):
        return {k: _unmarked(v) for k, v in tree.items() if k != "cache_control"}
    if isinstance(tree, list):
        return [_unmarked(v) for v in tree]
    return tree


def _sent(
    request: Request,
) -> tuple[PlainTree, PlainTree, list[tuple[PlainTree, PlainTree]]]:
    """Return what ``request`` sends, marks removed: tools, system, and each block."""
    capability, settings = _model()
    body, _ = wire.encode(
        request,
        capability=capability,
        settings=settings,
        system=wire.system_blocks(request.system, settings=settings),
    )
    messages = from_plain(body["messages"], list[dict[str, PlainTree]])
    blocks = [
        (message["role"], _unmarked(block))
        for message in messages
        for block in from_plain(message["content"], list[PlainTree])
    ]
    return (
        _unmarked(from_plain(body["tools"], list[PlainTree])),
        _unmarked(from_plain(body["system"], list[PlainTree])),
        blocks,
    )


@pytest.mark.asyncio
async def test_each_request_repeats_the_last_and_a_child_repeats_its_parent(
    tmp_path: Path,
) -> None:
    """Prefix caching hits only on identical bytes, up to each mark."""
    _ = (tmp_path / "AGENTS.md").write_text("be terse\n")
    tools = [_Echo(name="Other"), _Echo()]
    config = AgentsMdConfig(system_dir=tmp_path / "system", user_dir=tmp_path / "user")
    parent_model = ScriptedProvider(
        replies=[
            reply("1", calls=[call("Echo", "c1", text="x")]),
            reply("2"),
            reply("3"),
        ],
    )
    child_model = ScriptedProvider(replies=[reply("done")])
    parent = Agent(
        model=parent_model["scripted"],
        tools=tools,
        cwd=tmp_path,
        system_prompt="S",
        agents_md=config,
    )
    child = Agent(
        model=child_model["scripted"],
        tools=tools,
        allowed=["Echo"],
        cwd=tmp_path,
        system_prompt="S",
        agents_md=config,
    )
    async with parent:
        _ = await parent.send("one")
        _ = await parent.send("two")
    async with child:
        _ = await child.send("task")

    sent = [_sent(r) for r in parent_model.requests]
    assert len(sent) == 3
    for (tools_a, system_a, blocks_a), (tools_b, system_b, blocks_b) in pairwise(sent):
        assert (tools_a, system_a) == (tools_b, system_b)
        assert blocks_b[: len(blocks_a)] == blocks_a
    (child_tools, child_system, _), (parent_tools, parent_system, _) = (
        _sent(child_model.requests[0]),
        sent[0],
    )
    assert (child_tools, child_system) == (parent_tools, parent_system)
    assert "be terse" in str(parent_system)


# ---- events: server-sent events ----------------------------------------------


async def _lines(*lines: str) -> AsyncIterator[str]:
    for line in lines:
        yield line


@pytest.mark.parametrize("ending", [(), ("",)], ids=["unterminated", "terminated"])
@pytest.mark.asyncio
async def test_events_gather_data_lines_until_a_blank_line(
    ending: tuple[str, ...],
) -> None:
    lines = _lines(
        "event: message_start",
        'data: {"type": "message_start",',
        'data:  "n": 1}',
        "",
        "",
        ": a comment",
        'data: {"type": "ping"}',
        "",
        'data: {"type": "message_stop"}',
        *ending,
    )
    assert [event async for event in wire.events(lines)] == [
        {"type": "message_start", "n": 1},
        {"type": "ping"},
        {"type": "message_stop"},
    ]


# ---- StreamReader: the reply -------------------------------------------------


def _start(**usage: object) -> dict[str, object]:
    return {"type": "message_start", "message": {"usage": usage}}


def _block(index: int, **block: object) -> dict[str, object]:
    return {"type": "content_block_start", "index": index, "content_block": block}


def _delta(index: int, **delta: object) -> dict[str, object]:
    return {"type": "content_block_delta", "index": index, "delta": delta}


def _stop(reason: str, **usage: object) -> dict[str, object]:
    return {"type": "message_delta", "delta": {"stop_reason": reason}, "usage": usage}


def _read(
    *events: Mapping[str, object],
    settings: ModelSettings | None = None,
) -> tuple[list[TextChunk], wire.StreamReader]:
    capability, narrowest = _model()
    chosen = narrowest if settings is None else settings
    reader = wire.StreamReader(capability=capability, settings=chosen)
    chunks = [chunk for event in events if (chunk := reader.feed(event)) is not None]
    return chunks, reader


def test_text_streams_as_chunks_and_finishes_as_one_message() -> None:
    chunks, reader = _read(
        _start(input_tokens=10),
        _block(0, type="text", text=""),
        _delta(0, type="text_delta", text="Good"),
        _delta(0, type="text_delta", text="bye"),
        {"type": "content_block_stop", "index": 0},
        _stop("end_turn", output_tokens=5),
        {"type": "message_stop"},
    )
    assert chunks == [TextChunk(text="Good"), TextChunk(text="bye")]
    response = reader.response()
    assert response.records == (AssistantMessage(content="Goodbye"),)
    assert response.stop_reason == "end_turn"
    assert response.tokens == TokenCount(request=10, response=5)


def test_the_stop_reason_is_empty_until_the_model_says() -> None:
    _, reader = _read(_start(), _block(0, type="text"))
    assert reader.stop_reason == ""
    reader.feed(_stop("max_tokens"))
    assert reader.stop_reason == "max_tokens"


def test_thinking_streams_as_reasoning_and_keeps_its_signature() -> None:
    chunks, reader = _read(
        _start(),
        _block(0, type="thinking", thinking=""),
        _delta(0, type="thinking_delta", thinking="Let me "),
        _delta(0, type="thinking_delta", thinking="see."),
        _delta(0, type="signature_delta", signature="sig-"),
        _delta(0, type="signature_delta", signature="1"),
        _block(1, type="thinking", thinking=""),
        _delta(1, type="thinking_delta", thinking="unsigned"),
        _block(2, type="redacted_thinking", data="sealed"),
        _stop("end_turn"),
    )
    assert chunks == [
        TextChunk(text="Let me ", thinking=True),
        TextChunk(text="see.", thinking=True),
        TextChunk(text="unsigned", thinking=True, record=1),
    ]
    assert reader.response().records == (
        Thinking(content="Let me see.", encrypted="sig-1"),
        Thinking(content="unsigned"),
        Thinking(encrypted="sealed"),
    )


def test_a_tool_call_gathers_its_argument_json() -> None:
    chunks, reader = _read(
        _start(),
        _block(0, type="tool_use", id="toolu_1", name="Echo", input={}),
        _delta(0, type="input_json_delta", partial_json='{"te'),
        _delta(0, type="input_json_delta", partial_json='xt": "hi"}'),
        _block(1, type="tool_use", id="toolu_2", name="my-tool", input={}),
        _stop("tool_use"),
    )
    assert chunks == []
    assert reader.response().records == (
        ToolCall(call_id="toolu_1", name="Echo", arguments={"text": "hi"}),
        ToolCall(call_id="toolu_2", name="my-tool", arguments={}),
    )


def test_a_call_cut_off_by_max_tokens_is_dropped_and_the_text_kept() -> None:
    """A reply that ran out of tokens mid-call has said what it said."""
    _, reader = _read(
        _start(),
        _block(0, type="text"),
        _delta(0, type="text_delta", text="Let me check."),
        _block(1, type="tool_use", id="toolu_1", name="Echo"),
        _delta(1, type="input_json_delta", partial_json='{"a":'),
        _stop("max_tokens"),
    )
    response = reader.response()
    assert response.records == (AssistantMessage(content="Let me check."),)
    assert response.stop_reason == "max_tokens"


def test_a_tool_name_echoed_from_the_spec_is_not_a_call() -> None:
    """A model printing its context echoes ``$FUNCTION_NAME``; no result answers it."""
    _, reader = _read(
        _start(),
        _block(0, type="text"),
        _delta(0, type="text_delta", text="ok"),
        _block(1, type="tool_use", id="toolu_1", name="$FUNCTION_NAME"),
        _stop("tool_use"),
    )
    assert reader.response().records == (AssistantMessage(content="ok"),)


def test_records_follow_the_block_order_and_skip_unknown_blocks() -> None:
    _, reader = _read(
        _start(),
        _block(1, type="text"),
        _delta(1, type="text_delta", text="second"),
        _block(0, type="text"),
        _delta(0, type="text_delta", text="first"),
        _block(2, type="server_tool_use", id="srvtoolu_1", name="web_search"),
        _delta(7, type="text_delta", text="no such block"),
        _delta(0, type="citations_delta", citation={}),
        _stop("end_turn"),
    )
    assert reader.response().records == (
        AssistantMessage(content="first"),
        AssistantMessage(content="second"),
    )


def test_an_in_stream_error_raises_what_it_means() -> None:
    error = {"type": "error", "error": {"type": "overloaded_error", "message": "Busy"}}
    with pytest.raises(OverloadedError, match="Busy"):
        _read(_start(), error)


def test_a_split_cache_write_bills_each_lifetime() -> None:
    _, reader = _read(
        _start(
            input_tokens=11,
            cache_creation_input_tokens=50,
            cache_creation={"ephemeral_1h_input_tokens": 20},
            cache_read_input_tokens=3,
        ),
        _stop("end_turn", output_tokens=7),
    )
    assert reader.response().tokens == TokenCount(
        request=11,
        response=7,
        cache_write=30,
        cache_write_1h=20,
        cache_read=3,
    )


@pytest.mark.parametrize(("ttl_sec", "written_1h"), [(300.0, 0), (3600.0, 50)])
def test_an_unsplit_cache_write_bills_the_lifetime_asked_for(
    ttl_sec: float,
    written_1h: int,
) -> None:
    _, settings = _model()
    settings = replace(settings, cache_ttl_sec=ttl_sec)
    _, reader = _read(_start(cache_creation_input_tokens=50), settings=settings)
    tokens = reader.response().tokens
    assert (tokens.cache_write, tokens.cache_write_1h) == (50 - written_1h, written_1h)


def test_an_empty_usage_counts_nothing() -> None:
    _, reader = _read({"type": "message_start", "message": {}})
    assert reader.response().tokens == TokenCount()


@pytest.mark.parametrize(
    ("speed", "served"),
    [("fast", "priority"), ("standard", "auto")],
)
def test_a_reply_is_billed_at_the_speed_it_was_served(
    speed: str,
    served: ServiceTier,
) -> None:
    capability, _ = _model()
    _, reader = _read(
        _start(input_tokens=1_000_000),
        _stop("end_turn", output_tokens=1_000_000, speed=speed),
    )
    response = reader.response()
    tokens = response.tokens
    today = datetime.now(UTC).date()
    assert response.spend == capability.prices[served, tokens.prompt, today] * tokens
    assert response.spend.request == (8.0 if speed == "fast" else 4.0)


# ---- to_exception ------------------------------------------------------------


def _error(kind: str, message: str) -> dict[str, object]:
    return {"type": "error", "error": {"type": kind, "message": message}}


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (413, "Request Entity Too Large"),
        (400, _error("request_too_large", "Request exceeds the maximum size")),
    ],
)
def test_a_request_too_large_in_bytes(status: int, body: object) -> None:
    error = wire.to_exception(status, body=body)
    assert isinstance(error, RequestTooLargeError)
    assert str(error) in (
        "Request Entity Too Large",
        "Request exceeds the maximum size",
    )


@pytest.mark.parametrize(
    "message",
    ["prompt is too long: 250000 tokens > 200000 maximum", "Prompt is too long"],
)
def test_a_prompt_past_the_window_is_a_context_overflow(message: str) -> None:
    error = wire.to_exception(400, body=_error("invalid_request_error", message))
    assert isinstance(error, ContextOverflowError)
    assert str(error) == message


def test_a_tool_schema_naming_the_context_window_is_not_an_overflow() -> None:
    message = "tools.0.input_schema: parameter 'context window' must be an object"
    error = wire.to_exception(400, body=_error("invalid_request_error", message))
    assert type(error) is RuntimeError
    assert message in str(error)


@pytest.mark.parametrize(
    "message",
    [
        "Output blocked by content filtering policy",
        "Fable 5's safeguards flagged this message under the Usage Policy.",
        "This request has been blocked by our content filters.",
        "Request rejected for safety reasons.",
        "This conversation tripped our cyber-related safeguards.",
        "Output withheld under the content filtering policy.",
        "This request violates the Usage Policy.",
        "Request flagged by our safety systems.",
        "We refused this request for safety reasons.",
        "Blocked by a safeguard.",
    ],
)
def test_a_safeguard_rejection_is_a_refusal(message: str) -> None:
    error = wire.to_exception(400, body=_error("invalid_request_error", message))
    assert isinstance(error, RefusedError)
    assert str(error) == message


@pytest.mark.parametrize(
    "message",
    [
        "Request blocked because this key lacks access",
        "cache_control rejected: at most 4 blocks are allowed by the caching policy",
        "Request rejected under your organization's data retention policy",
        "Unknown beta flag; see the API versioning policy",
        "Safety settings are not supported by this model.",
    ],
)
def test_a_rejection_citing_another_policy_is_not_a_refusal(message: str) -> None:
    error = wire.to_exception(400, body=_error("invalid_request_error", message))
    assert type(error) is RuntimeError


@pytest.mark.parametrize(
    "message",
    [
        "Your credit balance is too low to access the Anthropic API.",
        "You are out of extra usage.",
        "Usage credits are required for this model.",
        "Insufficient credits.",
        "You have exceeded your usage credits.",
    ],
)
def test_an_empty_account_is_billing_even_as_a_429(message: str) -> None:
    error = wire.to_exception(429, body=_error("rate_limit_error", message))
    assert isinstance(error, BillingError)
    assert str(error) == message


@pytest.mark.parametrize(
    ("status", "kind"),
    [(401, ""), (403, ""), (400, "authentication_error"), (400, "permission_error")],
)
def test_rejected_credentials_are_an_auth_error(status: int, kind: str) -> None:
    error = wire.to_exception(status, body=_error(kind, "no"))
    assert isinstance(error, AuthError)
    assert str(error) == "no"


@pytest.mark.parametrize(
    ("retry_after", "wait_sec"),
    [
        ("7", 7.0),
        ("30", 30.0),
        ("-3", 0.0),
        ("999999", 691_200.0),
        ("soon", 0.0),
    ],
)
def test_a_rate_limit_waits_as_advised(retry_after: str, wait_sec: float) -> None:
    error = wire.to_exception(
        429,
        body=_error("rate_limit_error", "slow down"),
        headers={"Retry-After": retry_after},
    )
    assert isinstance(error, RateLimitedError)
    assert error.retry_after_sec == wait_sec
    assert str(error) == "slow down"


# The headers of a subscription's spent weekly window, as sent 2026-10-29.
_WEEKLY_SPENT = MappingProxyType(
    {
        "anthropic-ratelimit-unified-representative-claim": "seven_day",
        "anthropic-ratelimit-unified-7d-reset": "1793451600",
        "anthropic-ratelimit-unified-7d-status": "rejected",
        "retry-after": "166122",
        "anthropic-ratelimit-unified-5h-status": "allowed",
        "anthropic-ratelimit-unified-reset": "1793451600",
        "anthropic-ratelimit-unified-5h-reset": "1793303400",
        "anthropic-ratelimit-unified-status": "rejected",
    },
)


@pytest.mark.parametrize(
    ("claim", "limit"),
    [
        ("seven_day", "weekly"),
        ("five_hour", "5-hour"),
        ("seven_day_opus", "weekly Opus"),
        ("seven_day_sonnet", "weekly Sonnet"),
        ("overage", ""),
        (None, ""),
    ],
)
def test_a_spent_window_is_named(
    claim: str | None,
    limit: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(wire, "time", _Clock(now=1_793_285_478.0))
    headers = dict(_WEEKLY_SPENT)
    if claim is None:
        del headers["anthropic-ratelimit-unified-representative-claim"]
    else:
        headers["anthropic-ratelimit-unified-representative-claim"] = claim
    error = wire.to_exception(
        429,
        body=_error("rate_limit_error", "This request would exceed your rate limit."),
        headers=headers,
    )
    assert isinstance(error, RateLimitedError)
    assert error.retry_after_sec == 166_122.0
    assert error.limit == limit


def test_a_spent_window_waits_to_its_reset_not_the_rounded_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``retry-after`` is whole seconds short of the reset; it resumed at 05:59:59."""
    monkeypatch.setattr(wire, "time", _Clock(now=1_793_285_477.6))
    error = wire.to_exception(429, body="limit", headers=_WEEKLY_SPENT)
    assert isinstance(error, RateLimitedError)
    assert error.retry_after_sec == pytest.approx(166_122.4)


def test_a_throttle_while_the_window_allows_waits_as_advised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(wire, "time", _Clock(now=1_000.0))
    headers = {
        "retry-after": "5",
        "anthropic-ratelimit-unified-status": "allowed",
        "anthropic-ratelimit-unified-reset": "999999",
    }
    error = wire.to_exception(429, body="slow", headers=headers)
    assert isinstance(error, RateLimitedError)
    assert error.retry_after_sec == 5.0


def test_a_retry_after_date_is_a_wait_until_then(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(wire, "time", _Clock(now=1_000_000_000.0))
    error = wire.to_exception(
        503,
        body="unavailable",
        headers={"retry-after": "Sun, 09 Sep 2001 01:46:50 GMT"},
    )
    assert isinstance(error, OverloadedError)
    assert error.retry_after_sec == 10.0


def test_the_unified_reset_is_a_wait_only_for_a_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(wire, "time", _Clock(now=1_000.0))
    headers = {"anthropic-ratelimit-unified-reset": "1060"}
    limited = wire.to_exception(429, body="limit", headers=headers)
    assert isinstance(limited, RateLimitedError)
    assert limited.retry_after_sec == 60.0
    busy = wire.to_exception(529, body="busy", headers=headers)
    assert isinstance(busy, OverloadedError)
    assert busy.retry_after_sec == 0.0
    # A spent weekly window resets days away; the wait is the whole of it.
    for reset, wait_sec in (
        ("900", 0.0),
        ("260200", 259_200.0),
        ("999999", 691_200.0),
    ):
        clamped = wire.to_exception(
            429,
            body="limit",
            headers={"anthropic-ratelimit-unified-reset": reset},
        )
        assert isinstance(clamped, RateLimitedError)
        assert clamped.retry_after_sec == wait_sec


def test_only_an_invalid_request_can_be_a_refusal() -> None:
    """A failing safety service is an outage, not a verdict on the request."""
    message = "Safety service unavailable; request rejected."
    error = wire.to_exception(500, body=_error("api_error", message))
    assert isinstance(error, OverloadedError)


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (408, ""),
        (409, ""),
        (500, ""),
        (529, ""),
        (0, "overloaded_error"),
        (0, "api_error"),
    ],
)
def test_a_busy_or_failing_server_is_overloaded(status: int, kind: str) -> None:
    assert isinstance(
        wire.to_exception(status, body=_error(kind, "x")),
        OverloadedError,
    )


def test_an_in_stream_rate_limit_is_a_rate_limit() -> None:
    error = wire.to_exception(0, body=_error("rate_limit_error", "x"))
    assert isinstance(error, RateLimitedError)
    assert error.retry_after_sec == 0.0


def test_any_other_error_names_its_status_kind_and_text() -> None:
    error = wire.to_exception(404, body=_error("not_found_error", "no such model"))
    assert type(error) is RuntimeError
    assert str(error) == "Anthropic replied 404 not_found_error: no such model"
    error = wire.to_exception(418, body="<html>teapot</html>")
    assert str(error) == "Anthropic replied 418 : <html>teapot</html>"


# ---- rate_limits -------------------------------------------------------------


def test_rate_limits_read_both_windows() -> None:
    windows = wire.rate_limits(
        {
            "anthropic-ratelimit-unified-5h-utilization": "0.56",
            "anthropic-ratelimit-unified-5h-reset": "1780531800",
            "anthropic-ratelimit-unified-5h-status": "allowed",
            "anthropic-ratelimit-unified-7d-utilization": "0.89",
            "anthropic-ratelimit-unified-7d-reset": "1780750800",
            "anthropic-ratelimit-unified-7d-status": "allowed_warning",
        },
    )
    assert windows == (
        UsageWindow(label="5h", utilization=0.56, resets_at=1780531800.0),
        UsageWindow(label="7d", utilization=0.89, resets_at=1780750800.0),
    )


def test_a_rejected_window_is_blocked() -> None:
    windows = wire.rate_limits(
        {
            "anthropic-ratelimit-unified-7d-utilization": "1.0",
            "anthropic-ratelimit-unified-7d-status": "rejected",
        },
    )
    assert windows == (UsageWindow(label="7d", utilization=1.0, blocked=True),)


def test_an_overall_rejection_blocks_every_window() -> None:
    windows = wire.rate_limits(
        {
            "anthropic-ratelimit-unified-status": "rejected",
            "anthropic-ratelimit-unified-5h-utilization": "0.6",
            "anthropic-ratelimit-unified-7d-status": "allowed",
        },
    )
    assert [w.blocked for w in windows] == [True, True]


def test_a_window_may_report_only_its_status() -> None:
    windows = wire.rate_limits({"anthropic-ratelimit-unified-5h-status": "rejected"})
    assert windows == (UsageWindow(label="5h", blocked=True),)


@pytest.mark.parametrize(("raw", "utilization"), [("1.05", 1.0), ("-0.1", 0.0)])
def test_utilization_is_clamped_to_a_fraction(raw: str, utilization: float) -> None:
    windows = wire.rate_limits({"Anthropic-RateLimit-Unified-7d-Utilization": raw})
    assert windows == (UsageWindow(label="7d", utilization=utilization),)


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"x-other": "v"},
        {"anthropic-ratelimit-unified-7d-utilization": "inf"},
        {"anthropic-ratelimit-unified-5h-reset": "nan"},
        {"anthropic-ratelimit-unified-status": "rejected"},
    ],
)
def test_an_api_key_reply_reports_no_window(headers: dict[str, str]) -> None:
    assert wire.rate_limits(headers) == ()


# ---- edges of a reply --------------------------------------------------------


def test_fields_a_reply_leaves_out_read_as_empty() -> None:
    chunks, reader = _read(
        {"type": "message_start", "message": {"usage": {}}},
        {"type": "content_block_start", "content_block": {"type": "text"}},
        {"type": "content_block_delta", "delta": {"type": "text_delta"}},
        _block(1, type="tool_use", name="Echo"),
        _block(2, type="tool_use", id="toolu_2"),
        _block(3, type="redacted_thinking"),
        _block(4, type="thinking"),
        _delta(4, type="thinking_delta"),
        _delta(4, type="signature_delta"),
        _block(5, type="tool_use", id="toolu_5", name="Echo"),
        _delta(5, type="input_json_delta"),
        {"type": "message_delta", "delta": {}, "usage": {}},
    )
    # A block given no index is block 0.
    # Block 2 makes no record, so block 4 is the reply's fourth.
    assert chunks == [TextChunk(text=""), TextChunk(text="", thinking=True, record=3)]
    assert reader.response().records == (
        AssistantMessage(content=""),
        ToolCall(call_id="", name="Echo"),
        Thinking(encrypted=""),
        Thinking(content=""),
        ToolCall(call_id="toolu_5", name="Echo"),
    )
    assert reader.stop_reason == ""


def test_a_call_whose_arguments_are_not_an_object_is_dropped() -> None:
    _, reader = _read(
        _start(),
        _block(0, type="tool_use", id="toolu_1", name="Echo"),
        _delta(0, type="input_json_delta", partial_json="[1]"),
        _stop("tool_use"),
    )
    assert reader.response().records == ()


def test_a_cache_write_split_without_an_hour_part_is_all_short() -> None:
    _, settings = _model()
    settings = replace(settings, cache_ttl_sec=3600.0)
    _, reader = _read(
        _start(
            cache_creation_input_tokens=50,
            cache_creation={"ephemeral_5m_input_tokens": 50},
        ),
        settings=settings,
    )
    tokens = reader.response().tokens
    assert (tokens.cache_write, tokens.cache_write_1h) == (50, 0)


def test_an_unknown_in_stream_error_names_no_status() -> None:
    with pytest.raises(RuntimeError) as raised:
        _read(
            _start(),
            {"type": "error", "error": {"type": "odd_error", "message": "x"}},
        )
    assert str(raised.value) == "Anthropic replied 0 odd_error: x"


def test_a_fixed_budget_is_a_whole_number_of_tokens() -> None:
    capability, settings = _model("opus-4.6")
    small = replace(
        capability,
        limits=replace(capability.limits, max_response_tokens=4_097),
    )
    body, _ = _encode(
        [UserMessage(content="q")],
        capability=small,
        settings=replace(settings, thinking_budget="fixed"),
    )
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 2_048}


def test_the_cache_is_held_unexpired_unless_said_otherwise() -> None:
    capability, settings = _model()
    settings = replace(settings, thinking_budget="auto")
    body, _ = wire.encode(
        Request(records=(UserMessage(content="q"),)),
        capability=capability,
        settings=settings,
        system=[],
    )
    assert body["context_management"] == {
        "edits": [{"type": "clear_thinking_20251015", "keep": "all"}],
    }


@pytest.mark.asyncio
async def test_an_unterminated_event_of_several_lines_is_joined_by_newlines() -> None:
    lines = _lines('data: {"type":', 'data: "ping"}')
    assert [event async for event in wire.events(lines)] == [{"type": "ping"}]


@dataclass(frozen=True, slots=True, kw_only=True)
class _Clock:
    """Stands in for the ``time`` module, stopped at ``now``."""

    now: float

    def time(self) -> float:
        return self.now


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
