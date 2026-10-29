"""Tests for the Gemini wire: requests out, streamed replies and errors in."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, tzinfo
from typing import TYPE_CHECKING, Self, override

import base64
import hashlib
import io
import logging

from PIL import Image
from treekle import from_plain, json_dumps

import pypdfium2 as pdfium
import pytest

from sagent.model.capability import ModelLimits, ModelSettings
from sagent.model.cost import TokenCount
from sagent.model.exceptions import (
    AuthError,
    BillingError,
    ContextOverflowError,
    DisconnectedError,
    OverloadedError,
    RateLimitedError,
    RequestTooLargeError,
)
from sagent.model.types import Request, TextChunk
from sagent.providers.google import wire
from sagent.providers.google.api import models
from sagent.session.types import (
    AgentToAgentMessage,
    AssistantMessage,
    Attachment,
    ContextClear,
    ShellCommandResult,
    SystemMessage,
    Thinking,
    TokenUsage,
    ToolCall,
    TurnContext,
    UncategorizedToolResult,
    UserMessage,
)
from sagent.tools.display import ToolDisplay


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping, Sequence

    from treekle import PlainTree

    from sagent.model.capability import (
        ModelCapability,
        ThinkingBudget,
        ThinkingEffort,
        ThinkingOutput,
    )
    from sagent.session.types import AnyToolResult, SessionRecord
    from sagent.tools.types import ToolContext


# ---- encode: contents ----------------------------------------------------------


def test_a_user_message_is_one_user_content_and_nothing_else() -> None:
    assert _encode(UserMessage(content="hello")) == {
        "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
        "generationConfig": _LEVEL,
    }


def test_encode_asks_no_more_than_the_room_the_request_leaves() -> None:
    capability = _capability()
    body = wire.encode(
        Request(max_response_tokens=9_000),
        capability=capability,
        settings=capability.default,
    )
    assert body["generationConfig"] == {"maxOutputTokens": 9_000, **_LEVEL}


def test_the_system_prompt_joins_its_nonempty_parts() -> None:
    body = _encode(UserMessage(content="hi"), system=("Be terse.", "", "Rules."))
    assert body["systemInstruction"] == {"parts": [{"text": "Be terse.\n\nRules."}]}


def test_an_all_empty_system_prompt_is_left_out() -> None:
    assert "systemInstruction" not in _encode(UserMessage(content="hi"), system=("",))


@pytest.mark.parametrize("content", ["", None])
def test_an_empty_user_message_sends_one_empty_part(content: str | None) -> None:
    assert _contents(UserMessage(content=content)) == [
        {"role": "user", "parts": [{"text": ""}]},
    ]


def test_images_and_pdfs_ride_inline_and_other_attachments_are_left_out(
    caplog: pytest.LogCaptureFixture,
) -> None:
    png, pdf = _png(4, 4), b"%PDF-1.4 tiny"
    message = UserMessage(
        content="look",
        attachments=(
            Attachment(mime_descriptor="text/plain", data=b"x"),
            Attachment(mime_descriptor="image/png", data=png),
            Attachment(mime_descriptor="application/pdf", data=pdf),
        ),
    )
    with caplog.at_level(logging.WARNING):
        (content,) = _contents(message)
    assert content["parts"] == [
        {"text": "look"},
        {"inlineData": {"mimeType": "image/png", "data": _b64(png)}},
        {"inlineData": {"mimeType": "application/pdf", "data": _b64(pdf)}},
    ]
    assert caplog.messages == ["Gemini reads no text/plain attachment; leaving it out."]


def test_an_image_is_shrunk_to_the_model_limits() -> None:
    capability = _capability(limits=ModelLimits(max_image_edge_px=2))
    message = UserMessage(
        attachments=(Attachment(mime_descriptor="image/png", data=_png(4, 4)),),
    )
    inline = _first_part(_contents(message, capability=capability))["inlineData"]
    assert inline["mimeType"] == "image/webp"
    data = base64.b64decode(from_plain(inline["data"], str))
    assert Image.open(io.BytesIO(data)).size == (2, 2)


def test_a_byte_limit_alone_also_shrinks() -> None:
    data = _png(64, 64, noisy=True)
    capability = _capability(limits=ModelLimits(max_image_bytes=len(data) // 2))
    message = UserMessage(
        attachments=(Attachment(mime_descriptor="image/png", data=data),),
    )
    part = _first_part(_contents(message, capability=capability))
    assert part["inlineData"]["mimeType"] == "image/webp"


def test_a_conversation_with_calls_results_and_signatures() -> None:
    png = _png(4, 4)
    contents = _contents(
        UserMessage(content="run ls"),
        Thinking(content="hmm"),
        AssistantMessage(content="calling", extra={"thoughtSignature": "sig-text"}),
        ToolCall(
            call_id="c1",
            name="Bash",
            arguments={"cmd": "ls"},
            extra={"thoughtSignature": "sig-a"},
        ),
        ToolCall(call_id="c2", name="Bash", arguments={"cmd": "pwd"}),
        UncategorizedToolResult(
            call_id="c1",
            content="a.py",
            attachments=(Attachment(mime_descriptor="image/png", data=png),),
        ),
        ShellCommandResult(call_id="c2", stdout="denied", is_error=True),
        TokenUsage(),
        UserMessage(content="and?"),
    )
    # As the old builder sent it: the reasoning's text stays home, each part
    # keeps its own signature, and results and the next words share a turn.
    assert contents == [
        {"role": "user", "parts": [{"text": "run ls"}]},
        {
            "role": "model",
            "parts": [
                {"text": "calling", "thoughtSignature": "sig-text"},
                {
                    "functionCall": {"name": "Bash", "args": {"cmd": "ls"}},
                    "thoughtSignature": "sig-a",
                },
                {"functionCall": {"name": "Bash", "args": {"cmd": "pwd"}}},
            ],
        },
        {
            "role": "user",
            "parts": [
                {"functionResponse": {"name": "Bash", "response": {"content": "a.py"}}},
                {"inlineData": {"mimeType": "image/png", "data": _b64(png)}},
                {
                    "functionResponse": {
                        "name": "Bash",
                        "response": {"content": "[Error] denied"},
                    },
                },
                {"text": "and?"},
            ],
        },
    ]


def test_a_result_whose_call_is_gone_is_named_by_its_id() -> None:
    contents = _contents(UncategorizedToolResult(call_id="lost", content="x"))
    assert _first_part(contents)["functionResponse"]["name"] == "lost"


def test_an_empty_error_result_gets_no_error_mark() -> None:
    contents = _contents(UncategorizedToolResult(call_id="c", is_error=True))
    assert _first_part(contents)["functionResponse"]["response"] == {"content": ""}


def test_harness_notes_ride_the_user_side_as_reminders() -> None:
    contents = _contents(
        SystemMessage(content="mind the tests"),
        SystemMessage(content=""),
        TurnContext(cwd="/w", model="gemini"),
        TurnContext(),
        UserMessage(content="go"),
    )
    assert contents == [
        {
            "role": "user",
            "parts": [
                {"text": "<system-reminder>\nmind the tests\n</system-reminder>"},
                {
                    "text": "<system-reminder>\nWorking directory: /w\nModel: gemini"
                    "\n</system-reminder>",
                },
                {"text": "go"},
            ],
        },
    ]


def test_another_agents_words_are_attributed() -> None:
    contents = _contents(
        AgentToAgentMessage(sender="lead", content="status?"),
        AgentToAgentMessage(sender="lead"),
        AgentToAgentMessage(content=None),
    )
    assert contents == [
        {
            "role": "user",
            "parts": [
                {"text": "[from lead]: status?"},
                {"text": "[from lead]: "},
                {"text": ""},
            ],
        },
    ]


def test_another_agents_and_a_cleared_historys_images_are_shrunk_too() -> None:
    capability = _capability(limits=ModelLimits(max_image_edge_px=2))
    image = (Attachment(mime_descriptor="image/png", data=_png(4, 4)),)
    contents = _contents(
        AgentToAgentMessage(sender="lead", attachments=image),
        ContextClear(history=(UserMessage(attachments=image),)),
        capability=capability,
    )
    parts = from_plain(contents[0]["parts"], list[object])[1:]
    images = [from_plain(part, dict[str, dict[str, str]]) for part in parts]
    assert [image["inlineData"]["mimeType"] for image in images] == [
        "image/webp",
        "image/webp",
    ]


def test_a_context_clear_sends_its_summary_then_its_history() -> None:
    contents = _contents(
        ContextClear(
            summary="We fixed it.",
            history=(
                ToolCall(call_id="c", name="Read", arguments={}),
                UncategorizedToolResult(call_id="c", content="text"),
            ),
        ),
        ContextClear(history=(UserMessage(content="next"),)),
    )
    assert contents == [
        {"role": "user", "parts": [{"text": "We fixed it."}]},
        {"role": "model", "parts": [{"functionCall": {"name": "Read", "args": {}}}]},
        {
            "role": "user",
            "parts": [
                {"functionResponse": {"name": "Read", "response": {"content": "text"}}},
                {"text": "next"},
            ],
        },
    ]


def test_empty_words_from_the_model_are_left_out() -> None:
    assert _contents(AssistantMessage(content=""), AssistantMessage()) == []


@pytest.mark.parametrize("signature", ["", 7])
def test_an_empty_or_odd_signature_is_not_sent(signature: PlainTree) -> None:
    (content,) = _contents(
        AssistantMessage(content="a", extra={"thoughtSignature": signature}),
    )
    assert content["parts"] == [{"text": "a"}]


def test_tools_are_declared_without_additional_properties() -> None:
    schema: Mapping[str, PlainTree] = {
        "type": "object",
        "properties": {
            "cmd": {"type": "string"},
            "opts": {
                "anyOf": [
                    {"type": "object", "additionalProperties": False},
                    {"type": "null"},
                ],
            },
        },
        "additionalProperties": False,
    }
    body = _encode(UserMessage(content="x"), tools=(_Tool(schema=schema),))
    assert body["tools"] == [
        {
            "functionDeclarations": [
                {
                    "name": "Bash",
                    "description": "Run it.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "cmd": {"type": "string"},
                            "opts": {"anyOf": [{"type": "object"}, {"type": "null"}]},
                        },
                    },
                },
            ],
        },
    ]


# ---- encode: thinking ----------------------------------------------------------


def test_no_budget_sends_a_token_budget_row_no_thinking_config() -> None:
    assert _thinking("gemini-flash-2.5", budget="none", effort="high") is None


@pytest.mark.parametrize("budget", ["none", "auto", "fixed"])
def test_gemini_3_is_sent_its_level_whatever_the_budget(budget: ThinkingBudget) -> None:
    """Gemini 3 cannot stop thinking, so its level is the only say over how much."""
    config = _thinking("gemini-pro-3.1", budget=budget, effort="low")
    assert config == {"includeThoughts": False, "thinkingLevel": "low"}


@pytest.mark.parametrize(
    ("effort", "level"),
    [("low", "low"), ("medium", "medium"), ("high", "high")],
)
def test_gemini_3_sends_a_thinking_level(effort: ThinkingEffort, level: str) -> None:
    config = _thinking("gemini-pro-3.1", budget="fixed", effort=effort, output="text")
    assert config == {"includeThoughts": True, "thinkingLevel": level}


def test_gemini_3s_least_effort_is_minimal() -> None:
    config = _thinking("gemini-flash-lite-3.5", budget="auto", effort="min")
    assert config == {"includeThoughts": False, "thinkingLevel": "minimal"}


@pytest.mark.parametrize(
    ("effort", "budget"),
    [
        ("min", 1024),
        ("low", 4096),
        ("medium", 8192),
        ("high", 16_384),
        ("xhigh", 20_480),
        ("max", 24_576),
    ],
)
def test_gemini_2_5_sends_a_token_budget(effort: ThinkingEffort, budget: int) -> None:
    config = _thinking("gemini-pro-2.5", budget="fixed", effort=effort, output="text")
    assert config == {"includeThoughts": True, "thinkingBudget": budget}


def test_gemini_2_5_auto_sends_a_dynamic_budget() -> None:
    assert _thinking("gemini-flash-2.5", budget="auto", effort="high") == {
        "includeThoughts": False,
        "thinkingBudget": -1,
    }


@pytest.mark.parametrize("budget", ["auto", "fixed"])
def test_a_row_offering_no_effort_says_nothing_of_one(budget: ThinkingBudget) -> None:
    capability, settings = models("google").resolve("gemini-flash-2.5")
    capability = replace(capability, thinking_effort=frozenset({"none"}))
    settings = replace(
        settings,
        thinking_effort="none",
        thinking_budget=budget,
        thinking_output="text",
    )
    assert _thinking_of(capability, settings) == {"includeThoughts": True}


@pytest.mark.parametrize("budget", ["auto", "fixed"])
def test_gemini_2_5_with_no_effort_is_sent_a_zero_budget(
    budget: ThinkingBudget,
) -> None:
    assert _thinking("gemini-flash-2.5", budget=budget) == {
        "includeThoughts": False,
        "thinkingBudget": 0,
    }


@pytest.mark.parametrize(
    ("as_level", "effort"),
    [(True, "max"), (True, "xhigh"), (True, "none")],
)
def test_an_effort_the_wire_cannot_say_is_refused(
    as_level: bool,
    effort: ThinkingEffort,
) -> None:
    capability = replace(
        models("google")["gemini-pro-3.1"],
        effort_as_level=as_level,
        thinking_effort=frozenset({effort, "low"}),
        thinking_budget=frozenset({"fixed"}),
        thinking_output=frozenset({"none"}),
    )
    # A row never offers these with a level beside them: Gemini 3 has no level that
    # stops its thinking.
    settings = replace(
        capability.default,
        thinking_effort=effort,
        thinking_budget="fixed",
        thinking_output="none",
    )
    with pytest.raises(ValueError, match=effort):
        _thinking_of(capability, settings)


# ---- estimate ------------------------------------------------------------------


def test_the_estimate_is_text_over_characters_per_token_plus_image_tiles() -> None:
    capability = _capability()
    text = {"parts": [{"text": "x" * 400}]}
    assert wire.estimate(text, capability=capability) == int(
        len(json_dumps(text)) / 4.0,
    )
    wide = _inline("image/png", _png(800, 10))
    pdf = _inline("application/pdf", b"%PDF-1.4")
    body = {"parts": [{"text": "x" * 400}, {"nested": [wide, "s"]}, pdf]}
    # The inline bytes are no text; the image is two 768-pixel tiles wide.
    dropped = {
        "parts": [
            {"text": "x" * 400},
            {"nested": [dict[str, object](), "s"]},
            dict[str, object](),
        ],
    }
    assert wire.estimate(body, capability=capability) == (
        int(len(json_dumps(dropped)) / 4.0) + 2 * 258
    )


@pytest.mark.parametrize(("width", "tiles"), [(768, 1), (769, 2), (1537, 3)])
def test_tiles_are_768_pixels_wide(width: int, tiles: int) -> None:
    capability = _capability()
    wide = {"parts": [_inline("image/png", _png(width, 1))]}
    base = wire.estimate({"parts": [dict[str, object]()]}, capability=capability)
    assert wire.estimate(wide, capability=capability) == base + tiles * 258


def test_a_small_image_is_one_tile_and_a_tall_one_two() -> None:
    capability = _capability()
    small = {"parts": [_inline("image/png", _png(768, 4))]}
    tall = {"parts": [_inline("image/png", _png(10, 769))]}
    base = wire.estimate({"parts": [dict[str, object]()]}, capability=capability)
    assert wire.estimate(small, capability=capability) == base + 258
    assert wire.estimate(tall, capability=capability) == base + 2 * 258


def test_each_images_tiles_add_up() -> None:
    capability = _capability()
    two = {
        "parts": [_inline("image/png", _png(4, 4)), _inline("image/png", _png(4, 4))],
    }
    empty = {"parts": [dict[str, object](), dict[str, object]()]}
    assert wire.estimate(two, capability=capability) == (
        wire.estimate(empty, capability=capability) + 2 * 258
    )


def test_an_inline_pdf_is_258_tokens_a_page() -> None:
    capability = _capability()
    pdf = {"parts": [_inline("application/pdf", _pdf(3))]}
    base = wire.estimate({"parts": [dict[str, object]()]}, capability=capability)
    assert wire.estimate(pdf, capability=capability) == base + 3 * 258


def test_inline_data_without_an_image_is_no_tile() -> None:
    capability = _capability()
    odd = {"parts": [{"inlineData": {"mimeType": "image/png"}}]}
    base = wire.estimate({"parts": [dict[str, object]()]}, capability=capability)
    assert wire.estimate(odd, capability=capability) == base


# ---- events --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_events_join_data_lines_up_to_a_blank_line() -> None:
    events = await _events(
        'data: {"a":',
        "data: 1}",
        "",
        "event: ignored",
        ": comment",
        'data: {"b": 2}',
    )
    assert events == [{"a": 1}, {"b": 2}]


@pytest.mark.asyncio
async def test_other_lines_inside_an_event_do_not_end_it() -> None:
    events = await _events(
        'data: {"a":',
        "event: x",
        "data: 1}",
        "",
        "data: [",
        "data: 2]",
    )
    assert events == [{"a": 1}]


@pytest.mark.asyncio
async def test_the_last_event_needs_no_blank_line() -> None:
    assert await _events('data: {"a":', "data: 1}") == [{"a": 1}]


@pytest.mark.asyncio
async def test_a_blank_line_with_no_data_is_no_event(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        assert await _events("", "", 'data: {"a": 1}', "", "") == [{"a": 1}]
    assert caplog.messages == []


@pytest.mark.asyncio
async def test_a_malformed_event_is_logged_and_skipped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        events = await _events("data: {not-json}", "", 'data: {"ok": true}', "")
    assert events == [{"ok": True}]
    (message,) = caplog.messages
    assert message.startswith(
        "Gemini sent a malformed stream event: Expecting property name",
    )


@pytest.mark.asyncio
async def test_an_event_that_is_no_object_is_malformed() -> None:
    assert await _events("data: [1]", "", 'data: {"ok": 1}') == [{"ok": 1}]


@pytest.mark.asyncio
async def test_an_empty_stream_is_no_events_and_no_error() -> None:
    assert await _events() == []


@pytest.mark.asyncio
async def test_a_stream_of_only_malformed_events_raises() -> None:
    with pytest.raises(
        ValueError,
        match=r"^Gemini's stream held only malformed events\.$",
    ) as raised:
        await _events("data: {not-json}", "", "data: nope", "", "data: [1")
    # The last bad event is the cause, for whoever debugs the vendor.
    assert "delimiter" in str(raised.value.__cause__)


# ---- StreamReader --------------------------------------------------------------


def test_text_and_a_call_stop_as_a_tool_use() -> None:
    reader = wire.StreamReader(capability=models("google")["gemini-pro-3.1"])
    chunks = reader.feed(
        _event(
            [
                {"text": "calling"},
                {
                    "functionCall": {"name": "Bash", "args": {"cmd": "ls"}},
                    "thoughtSignature": "sig-fc",
                },
            ],
            finish="STOP",
            usage={"promptTokenCount": 10, "candidatesTokenCount": 5},
        ),
    )
    assert chunks == [TextChunk(text="calling")]
    response = reader.response()
    message, call = response.records
    assert message == AssistantMessage(content="calling")
    assert isinstance(call, ToolCall)
    assert (call.name, dict(call.arguments)) == ("Bash", {"cmd": "ls"})
    assert call.extra == {"thoughtSignature": "sig-fc"}
    assert call.call_id.startswith("call_")
    assert len(call.call_id) == len("call_") + 24
    assert response.stop_reason == "tool_use"
    assert response.tokens == TokenCount(request=10, response=5)


def test_each_call_gets_its_own_id_unless_gemini_names_one() -> None:
    reader = wire.StreamReader(capability=models("google")["gemini-pro-3.1"])
    reader.feed(
        _event(
            [
                {"functionCall": {"name": "A"}},
                {"functionCall": {"name": "B"}},
                {"functionCall": {"id": "given", "name": "C", "args": {"k": 1}}},
            ],
            finish="STOP",
        ),
    )
    a, b, c = reader.response().records
    assert isinstance(a, ToolCall)
    assert isinstance(b, ToolCall)
    assert isinstance(c, ToolCall)
    assert a.call_id != b.call_id
    assert (a.arguments, a.extra) == ({}, {})
    assert c.call_id == "given"


def test_a_call_with_no_name_is_dropped_and_an_empty_signature_ignored() -> None:
    reader = wire.StreamReader(capability=models("google")["gemini-pro-3.1"])
    reader.feed(
        _event(
            [
                {"functionCall": {"args": {}}},
                {"functionCall": {"name": ""}},
                {"functionCall": {"name": "A", "id": ""}, "thoughtSignature": ""},
                {"text": "ok", "thoughtSignature": ""},
                {"text": 5},
            ],
            finish="STOP",
        ),
    )
    message, call = reader.response().records
    assert message == AssistantMessage(content="ok")
    assert isinstance(call, ToolCall)
    assert (call.name, call.extra) == ("A", {})
    assert call.call_id.startswith("call_")


def test_reasoning_and_a_signed_answer_in_fragments() -> None:
    reader = wire.StreamReader(capability=models("google")["gemini-pro-3.1"])
    chunks = [
        *reader.feed(_event([{"text": "hmm", "thought": True}])),
        *reader.feed(_event([{"text": "He"}])),
        *reader.feed(_event([{"text": "", "thoughtSignature": "sig-0"}])),
        *reader.feed(
            _event(
                [{"text": "llo", "thoughtSignature": "sig-text"}],
                finish="STOP",
                usage={
                    "promptTokenCount": 1000,
                    "cachedContentTokenCount": 400,
                    "toolUsePromptTokenCount": 10,
                    "candidatesTokenCount": 5,
                    "thoughtsTokenCount": 7,
                },
            ),
        ),
    ]
    assert chunks == [
        TextChunk(text="hmm", thinking=True),
        TextChunk(text="He", record=1),
        TextChunk(text="llo", record=1),
    ]
    response = reader.response()
    assert response.records == (
        Thinking(content="hmm"),
        AssistantMessage(content="Hello", extra={"thoughtSignature": "sig-text"}),
    )
    assert response.stop_reason == "end_turn"
    # The prompt count holds the cache; reasoning bills as output.
    assert response.tokens == TokenCount(request=610, response=12, cache_read=400)
    capability = models("google")["gemini-pro-3.1"]
    assert (
        response.spend
        == capability.prices[
            "auto",
            response.tokens.prompt,
            datetime.now(UTC).date(),
        ]
        * response.tokens
    )
    assert response.spend.total > 0


def test_a_signature_on_reasoning_alone_stays_with_the_reasoning() -> None:
    reader = wire.StreamReader(capability=models("google")["gemini-pro-3.1"])
    reader.feed(_event([{"text": "hmm", "thought": True, "thoughtSignature": "s"}]))
    reader.feed(_event([], finish="STOP"))
    assert reader.response().records == (
        Thinking(content="hmm", extra={"thoughtSignature": "s"}),
    )


def test_the_last_usage_report_is_the_total() -> None:
    reader = wire.StreamReader(capability=models("google")["gemini-pro-3.1"])
    reader.feed(_event([{"text": "a"}], usage={"promptTokenCount": 3}))
    reader.feed(_event([], finish="STOP"))
    reader.feed({"usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 2}})
    assert reader.response().tokens == TokenCount(request=9, response=2)


def test_a_cached_count_above_the_prompt_bills_no_negative_request() -> None:
    reader = wire.StreamReader(capability=models("google")["gemini-pro-3.1"])
    reader.feed(
        _event(
            [],
            finish="STOP",
            usage={"promptTokenCount": 1, "cachedContentTokenCount": 5},
        ),
    )
    assert reader.response().tokens.request == 0


def test_an_event_without_candidates_adds_nothing() -> None:
    reader = wire.StreamReader(capability=models("google")["gemini-pro-3.1"])
    assert reader.feed({"usageMetadata": {"promptTokenCount": 1}}) == []
    assert reader.feed({"candidates": "garbage"}) == []
    assert reader.feed(_event(["not a part", {"inlineData": {}}], finish="STOP")) == []
    response = reader.response()
    assert response.records == ()
    assert response.tokens == TokenCount(request=1)


def test_no_usage_report_is_no_tokens() -> None:
    reader = wire.StreamReader(capability=models("google")["gemini-pro-3.1"])
    reader.feed(
        _event([{"text": "a", "thought": True}, {"text": "b", "thought": True}]),
    )
    reader.feed(_event([], finish="STOP"))
    response = reader.response()
    assert response.records == (Thinking(content="ab"),)
    assert response.tokens == TokenCount()


@pytest.mark.parametrize(
    ("finish", "stop"),
    [
        ("MAX_TOKENS", "max_tokens"),
        ("SAFETY", "refusal"),
        ("RECITATION", "refusal"),
        ("LANGUAGE", "refusal"),
        ("BLOCKLIST", "refusal"),
        ("PROHIBITED_CONTENT", "refusal"),
        ("SPII", "refusal"),
        ("IMAGE_SAFETY", "refusal"),
        ("MALFORMED_FUNCTION_CALL", "MALFORMED_FUNCTION_CALL"),
        ("OTHER", "OTHER"),
    ],
)
def test_finish_reasons_speak_the_shared_words(finish: str, stop: str) -> None:
    reader = wire.StreamReader(capability=models("google")["gemini-pro-3.1"])
    reader.feed(_event([{"text": "a"}], finish=finish))
    assert reader.response().stop_reason == stop


def test_the_finish_reason_survives_a_later_event_without_one() -> None:
    reader = wire.StreamReader(capability=models("google")["gemini-pro-3.1"])
    reader.feed(_event([{"text": "a"}], finish="MAX_TOKENS"))
    reader.feed(_event([{"text": "b"}]))
    assert reader.response().stop_reason == "max_tokens"


def test_a_blocked_prompt_is_a_refusal_with_no_records() -> None:
    reader = wire.StreamReader(capability=models("google")["gemini-pro-3.1"])
    reader.feed(
        {
            "promptFeedback": {"blockReason": "PROHIBITED_CONTENT"},
            "usageMetadata": {"promptTokenCount": 3},
        },
    )
    reader.feed({"promptFeedback": {}})
    response = reader.response()
    assert (response.records, response.stop_reason) == ((), "refusal")
    assert response.tokens == TokenCount(request=3)


@pytest.mark.parametrize("reason", ["", 3])
def test_an_empty_or_odd_block_reason_blocks_nothing(reason: object) -> None:
    reader = wire.StreamReader(capability=models("google")["gemini-pro-3.1"])
    reader.feed({"promptFeedback": {"blockReason": reason}})
    with pytest.raises(DisconnectedError):
        reader.response()


def test_a_stream_that_ends_unfinished_is_a_dropped_connection() -> None:
    reader = wire.StreamReader(capability=models("google")["gemini-pro-3.1"])
    reader.feed(
        _event(
            [{"text": "part"}],
            usage={"promptTokenCount": 4, "candidatesTokenCount": 1},
        ),
    )
    reader.feed({"candidates": [{"finishReason": ""}]})
    with pytest.raises(
        DisconnectedError,
        match=r"^Gemini's stream ended before its reply finished\.$",
    ):
        reader.response()


def test_a_reply_is_priced_at_the_card_in_force_on_its_utc_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(wire, "datetime", _Clock)
    # gemini-flash-3.8 doubles its rates on 2027-01-01.
    reader = wire.StreamReader(capability=models("google")["gemini-flash-3.8"])
    reader.feed(_event([], finish="STOP", usage={"promptTokenCount": 1_000_000}))
    assert reader.response().spend.request == 1.5


# ---- to_exception --------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "message", "kind"),
    [
        (400, "Request Entity Too Large", RequestTooLargeError),
        (413, "too big", RequestTooLargeError),
        (400, "request_too_large", RequestTooLargeError),
        (400, "exceeds the maximum request size of 20MB", RequestTooLargeError),
        (400, "The number of bytes exceeds the limit", RequestTooLargeError),
        (400, "Payload too large", RequestTooLargeError),
        # Ambiguous between bytes and tokens: a 413 decides it is bytes.
        (413, "Request exceeds the maximum size", RequestTooLargeError),
        (
            413,
            (
                "The input token count (1200000) exceeds the maximum number of tokens"
                " allowed (1048576)."
            ),
            ContextOverflowError,
        ),
        (400, "Input token count is too high", ContextOverflowError),
        (413, "It exceeds the maximum number of tokens allowed", ContextOverflowError),
        # Ambiguous, but not a 413: read as the token window.
        (400, "Your input exceeds the maximum allowed", ContextOverflowError),
        (400, "maximum context length is 8192", ContextOverflowError),
        (400, "context_length_exceeded", ContextOverflowError),
        (400, "Input too large", ContextOverflowError),
        (400, "Input too long", ContextOverflowError),
        (400, "Prompt is too long", ContextOverflowError),
        (400, "prompt too long", ContextOverflowError),
        (400, "contents: too_long", ContextOverflowError),
        (400, "string_above_max_length", ContextOverflowError),
        (400, "the context window was exceeded", ContextOverflowError),
        (400, "context window overflow", ContextOverflowError),
        (400, "context window maximum reached", ContextOverflowError),
        (400, "model context exceeded", ContextOverflowError),
        (400, "The billing account is disabled", BillingError),
        (429, "Your prepayment credits ran out", BillingError),
        (429, "credits are depleted", BillingError),
        (403, "This API requires billing to be enabled", BillingError),
        (401, "expired", AuthError),
        (403, "denied", AuthError),
        (400, "API key not valid. Please pass a valid API key.", AuthError),
        (429, "slow down", RateLimitedError),
        (500, "boom", OverloadedError),
        (503, "busy", OverloadedError),
        (408, "timeout", OverloadedError),
        (409, "aborted", OverloadedError),
        (400, "bad field", RuntimeError),
    ],
)
def test_an_error_reply_maps_to_the_most_specific_error(
    status: int,
    message: str,
    kind: type[Exception],
) -> None:
    error = wire.to_exception(
        status,
        body={"error": {"code": status, "message": message}},
    )
    assert type(error) is kind
    assert message in str(error)


@pytest.mark.parametrize(
    ("message", "kind"),
    [
        ("'model context' field missing in tools schema", RuntimeError),
        ("context window settings are invalid", RuntimeError),
    ],
)
def test_a_window_phrase_without_its_marker_is_no_overflow(
    message: str,
    kind: type[Exception],
) -> None:
    assert type(wire.to_exception(400, body={"error": {"message": message}})) is kind


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        ("UNAUTHENTICATED", AuthError),
        ("PERMISSION_DENIED", AuthError),
        ("RESOURCE_EXHAUSTED", RateLimitedError),
        ("UNAVAILABLE", OverloadedError),
        ("INTERNAL", OverloadedError),
        ("DEADLINE_EXCEEDED", OverloadedError),
    ],
)
def test_the_status_name_decides_without_an_http_status(
    status: str,
    kind: type[Exception],
) -> None:
    error = wire.to_exception(0, body={"error": {"status": status, "message": "m"}})
    assert type(error) is kind


def test_a_rate_limit_waits_as_retry_info_advises() -> None:
    error = wire.to_exception(
        429,
        body={
            "error": {
                "message": "quota",
                "details": [
                    {"@type": "type.googleapis.com/google.rpc.Help"},
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": "bad",
                    },
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": "17.5s",
                    },
                ],
            },
        },
        headers={"Retry-After": "3"},
    )
    assert isinstance(error, RateLimitedError)
    assert error.retry_after_sec == 17.5


@pytest.mark.parametrize(
    ("header", "wait_sec"),
    [("3", 3.0), ("-5", 0.0), ("999999", 691_200.0), ("soon", 0.0), ("inf", 0.0)],
)
def test_without_retry_info_the_header_advises(header: str, wait_sec: float) -> None:
    error = wire.to_exception(503, body="busy", headers={"Retry-After": header})
    assert isinstance(error, OverloadedError)
    assert error.retry_after_sec == wait_sec


@pytest.mark.parametrize(("delay", "wait_sec"), [("-4s", 0.0), ("999999s", 691_200.0)])
def test_retry_info_is_bounded_too(delay: str, wait_sec: float) -> None:
    error = wire.to_exception(
        429,
        body={"error": {"details": [{"@type": "x.RetryInfo", "retryDelay": delay}]}},
    )
    assert isinstance(error, RateLimitedError)
    assert error.retry_after_sec == wait_sec


def test_retry_info_with_an_odd_delay_defers_to_the_header() -> None:
    error = wire.to_exception(
        429,
        body={
            "error": {
                "details": [
                    {"@type": "x.RetryInfo", "retryDelay": 5},
                    {"retryDelay": "9s"},
                ],
            },
        },
        headers={"retry-after": "2"},
    )
    assert isinstance(error, RateLimitedError)
    assert error.retry_after_sec == 2.0


def test_no_advice_is_no_wait() -> None:
    error = wire.to_exception(429, body={"error": {"message": "m"}})
    assert isinstance(error, RateLimitedError)
    assert error.retry_after_sec == 0.0


def test_code_assist_wraps_its_error_in_a_list() -> None:
    error = wire.to_exception(
        429,
        body=[
            {"error": {"message": "capacity", "status": "RESOURCE_EXHAUSTED"}},
            {"error": {"message": "second", "status": "INTERNAL"}},
        ],
    )
    assert isinstance(error, RateLimitedError)
    assert str(error) == "capacity"


def test_a_text_body_is_the_message() -> None:
    error = wire.to_exception(418, body="I'm a teapot")
    assert isinstance(error, RuntimeError)
    assert str(error) == "Gemini replied 418 : I'm a teapot"


def test_an_unknown_error_names_its_status_and_kind() -> None:
    error = wire.to_exception(
        400,
        body={"error": {"status": "INVALID_ARGUMENT", "message": "m"}},
    )
    assert str(error) == "Gemini replied 400 INVALID_ARGUMENT: m"


# ---- helpers -------------------------------------------------------------------


# What Gemini 3.1 is sent at its defaults: its level, as it cannot stop thinking.
_LEVEL: Mapping[str, object] = {
    "thinkingConfig": {"includeThoughts": False, "thinkingLevel": "high"},
}


def _capability(*, limits: ModelLimits | None = None) -> ModelCapability:
    capability = models("google")["gemini-pro-3.1"]
    if limits is None:
        return capability
    return replace(capability, limits=limits)


def _encode(
    *records: SessionRecord,
    system: tuple[str, ...] = (),
    tools: Sequence[_Tool] = (),
    budget: ThinkingBudget = "none",
) -> dict[str, object]:
    capability = _capability()
    settings = replace(capability.default, thinking_budget=budget)
    return wire.encode(
        Request(system=system, records=records, tools=tools),
        capability=capability,
        settings=settings,
    )


def _contents(
    *records: SessionRecord,
    capability: ModelCapability | None = None,
) -> list[dict[str, object]]:
    capability = capability or _capability()
    body = wire.encode(
        Request(records=records),
        capability=capability,
        settings=capability.default,
    )
    return from_plain(body["contents"], list[dict[str, object]])


def _first_part(
    contents: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Return the first content's first part, which holds one object."""
    return from_plain(contents[0]["parts"], list[dict[str, dict[str, object]]])[0]


def _thinking(
    model: str,
    *,
    budget: ThinkingBudget,
    effort: ThinkingEffort = "none",
    output: ThinkingOutput = "none",
) -> object:
    capability, settings = models("google").resolve(model)
    settings = replace(
        settings,
        thinking_budget=budget,
        thinking_effort=effort,
        thinking_output=output,
    )
    return _thinking_of(capability, settings)


def _thinking_of(capability: ModelCapability, settings: ModelSettings) -> object:
    body = wire.encode(
        Request(records=(UserMessage(content="q"),)),
        capability=capability,
        settings=settings,
    )
    config = from_plain(body["generationConfig"], dict[str, object])
    return config.get("thinkingConfig")


def _event(
    parts: list[object],
    *,
    finish: str = "",
    usage: Mapping[str, int] | None = None,
) -> dict[str, object]:
    candidate: dict[str, object] = {"content": {"role": "model", "parts": parts}}
    if finish:
        candidate["finishReason"] = finish
    event: dict[str, object] = {"candidates": [candidate]}
    if usage is not None:
        event["usageMetadata"] = dict(usage)
    return event


async def _events(*lines: str) -> list[dict[str, object]]:
    return [event async for event in wire.events(_lines(lines))]


async def _lines(lines: Sequence[str]) -> AsyncIterator[str]:
    for line in lines:
        yield line


def _png(width: int, height: int, *, noisy: bool = False) -> bytes:
    if noisy:
        pixels = hashlib.shake_256(b"noise").digest(width * height * 3)
        image = Image.frombytes("RGB", (width, height), pixels)
    else:
        image = Image.new("RGB", (width, height), "red")
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def _pdf(pages: int) -> bytes:
    doc = pdfium.PdfDocument.new()
    for _ in range(pages):
        _ = doc.new_page(612, 792)
    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()


@pytest.mark.parametrize(
    ("attachment", "tokens"),
    [
        (Attachment(mime_descriptor="image/png", data=_png(1600, 800)), 258 * 3 * 2),
        (Attachment(mime_descriptor="image/png", data=_png(300, 200)), 258),
        (Attachment(mime_descriptor="application/pdf", data=_pdf(3)), 258 * 3),
        (Attachment(mime_descriptor="text/plain", data=b"x" * 4_000), 0),
    ],
    ids=["tiled-image", "small-image", "pdf", "unsent"],
)
def test_an_attachment_bills_258_per_tile_or_page(
    attachment: Attachment,
    tokens: int,
) -> None:
    assert wire.attachment_tokens(attachment, limits=ModelLimits()) == tokens


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _inline(mime: str, data: bytes) -> dict[str, object]:
    return {"inlineData": {"mimeType": mime, "data": _b64(data)}}


class _Clock(datetime):
    """A clock on 2027-06-01 in UTC, and still in 2026 in any other zone."""

    @classmethod
    @override
    def now(cls, tz: tzinfo | None = None) -> Self:
        if tz is UTC:
            return cls(2027, 6, 1, tzinfo=UTC)
        return cls(2026, 6, 1)


class _Tool:
    """A tool the model may call; only its name, description, and schema are sent."""

    def __init__(self, *, schema: Mapping[str, PlainTree]) -> None:
        self.name = "Bash"
        self.description = "Run it."
        self.schema = schema
        self.display = ToolDisplay()

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
    ) -> AnyToolResult:
        del context, arguments
        return UncategorizedToolResult(call_id="")


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
