"""Tests for the Responses API translation: request bodies and streamed replies."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import io

from PIL import Image
from treekle import mutable

import pytest

from sagent.model.capability import ModelLimits, ModelSettings
from sagent.model.cost import TokenCount
from sagent.model.exceptions import (
    DisconnectedError,
    RateLimitedError,
    RefusedError,
)
from sagent.model.types import Request, TextChunk
from sagent.providers.catalog import choose
from sagent.providers.openai import responses, wire
from sagent.providers.openai.compat import models
from sagent.session.types import (
    AssistantMessage,
    Attachment,
    ContextClear,
    Thinking,
    TokenUsage,
    ToolCall,
    UncategorizedToolResult,
    UserMessage,
)
from sagent.tools.read import Read


if TYPE_CHECKING:
    from collections.abc import Mapping

    from sagent.model.capability import ModelCapability
    from sagent.model.cost import ServiceTier


def _model(name: str = "luna-6.0") -> tuple[ModelCapability, ModelSettings]:
    table = models(
        "sagent.providers.openai",
        document="openai.yaml",
        section="openai",
    )
    return table.resolve(name)


def _png() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (2, 2), "blue").save(out, format="PNG")
    return out.getvalue()


def _encode(
    request: Request,
    name: str = "luna-6.0",
    **choices: object,
) -> dict[str, object]:
    capability, settings = _model(name)
    for field, value in choices.items():
        settings = choose(settings, field, value)
    capability.check(settings)
    return responses.encode(request, capability=capability, settings=settings)


# ---- the request -------------------------------------------------------------


def test_encode_sends_the_whole_conversation_statelessly() -> None:
    capability, settings = _model()
    settings = replace(settings, thinking_effort="none")
    request = Request(
        system=("Be terse.", "", "Mind the tests."),
        records=(UserMessage(content="hi"),),
    )
    assert responses.encode(request, capability=capability, settings=settings) == {
        "model": "gpt-6-luna",
        "input": [{"role": "user", "content": "hi"}],
        "instructions": "Be terse.\n\nMind the tests.",
        "stream": True,
        "store": False,
        "reasoning": {"effort": "none"},
        "max_output_tokens": 128_000,
        "prompt_cache_key": wire.cache_key(request),
    }


def test_encode_asks_no_more_than_the_room_the_request_leaves() -> None:
    body = _encode(Request(max_response_tokens=9_000))
    assert body["max_output_tokens"] == 9_000


def test_encode_renders_each_record_as_its_input_item() -> None:
    png = _png()
    url = wire.image_url(
        Attachment(mime_descriptor="image/png", data=png),
        limits=_model()[0].limits,
    )
    image = Attachment(mime_descriptor="image/png", data=png)
    pdf = Attachment(mime_descriptor="application/pdf", data=b"%PDF")
    records = (
        ContextClear(
            summary="Earlier: a bug.",
            history=(UserMessage(content="look", attachments=(image, pdf)),),
        ),
        UserMessage(attachments=(image,)),
        TokenUsage(),
        Thinking(summary="Plan.", encrypted="sealed-1", extra={"reasoning_id": "rs_1"}),
        Thinking(encrypted="sealed-2", extra={"reasoning_id": "rs_2"}),
        # Another vendor's thinking, and thinking that was never sealed.
        Thinking(content="Claude's", encrypted="signature"),
        Thinking(summary="unsealed", extra={"reasoning_id": "rs_3"}),
        Thinking(encrypted="sealed-3", extra={"reasoning_id": ""}),
        AssistantMessage(content="Reading."),
        AssistantMessage(content=""),
        ToolCall(call_id="toolu_a", name="Read", arguments={"file_path": "a.py"}),
        ToolCall(call_id="toolu_b", name="Read", arguments={"file_path": "b.py"}),
        UncategorizedToolResult(call_id="toolu_b", content="missing", is_error=True),
        UncategorizedToolResult(
            call_id="toolu_a",
            content="x = 1",
            attachments=(image, pdf),
        ),
        UncategorizedToolResult(call_id="toolu_c", attachments=(image,)),
        UncategorizedToolResult(call_id="toolu_d", is_error=True),
    )
    assert _encode(Request(records=records))["input"] == [
        {"role": "user", "content": "Earlier: a bug."},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "look"},
                {"type": "input_image", "detail": "auto", "image_url": url},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "input_image", "detail": "auto", "image_url": url}],
        },
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "Plan."}],
            "encrypted_content": "sealed-1",
        },
        {
            "type": "reasoning",
            "id": "rs_2",
            "summary": [],
            "encrypted_content": "sealed-2",
        },
        {"role": "assistant", "content": "Reading."},
        {
            "type": "function_call",
            "id": "fc_0",
            "call_id": "fc_0",
            "name": "Read",
            "arguments": '{"file_path": "a.py"}',
            "status": "completed",
        },
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "fc_1",
            "name": "Read",
            "arguments": '{"file_path": "b.py"}',
            "status": "completed",
        },
        {
            "type": "function_call_output",
            "call_id": "fc_1",
            "output": "[Error] missing",
            "status": "completed",
        },
        {
            "type": "function_call_output",
            "call_id": "fc_0",
            "output": [
                {"type": "input_text", "text": "x = 1"},
                {"type": "input_image", "detail": "auto", "image_url": url},
            ],
            "status": "completed",
        },
        {
            "type": "function_call_output",
            "call_id": "fc_2",
            "output": [{"type": "input_image", "detail": "auto", "image_url": url}],
            "status": "completed",
        },
        {
            "type": "function_call_output",
            "call_id": "fc_3",
            "output": "",
            "status": "completed",
        },
    ]


def test_encode_offers_the_tools_and_gates_which_may_be_called() -> None:
    read = Read()
    offered = {
        "type": "function",
        "name": "Read",
        "description": read.description,
        "parameters": mutable(read.schema),
        "strict": False,
    }
    body = _encode(Request(tools=(read,)))
    assert body["tools"] == [offered]
    assert body["parallel_tool_calls"] is True
    assert "tool_choice" not in body
    assert _encode(Request(tools=(read,), allowed=frozenset({"Read", "Bash"})))[
        "tool_choice"
    ] == {
        "type": "allowed_tools",
        "mode": "auto",
        "tools": [{"type": "function", "name": "Read"}],
    }
    assert _encode(Request(tools=(read,), allowed=frozenset()))["tool_choice"] == "none"
    assert "tools" not in _encode(Request(allowed=frozenset({"Read"})))


def test_encode_asks_for_reasoning_at_the_effort_chosen() -> None:
    # ``none`` is said, not left out: left out, the model reasons at its default.
    stopped = _encode(Request(), thinking_effort="none", thinking_output="text")
    assert stopped["reasoning"] == {"effort": "none"}
    assert "include" not in stopped
    # A model that cannot reason is sent no reasoning at all.
    assert "reasoning" not in _encode(Request(), "gpt-4.1")
    assert _encode(Request())["reasoning"] == {
        "effort": "medium",
        "context": "all_turns",
    }
    body = _encode(Request(), thinking_effort="high", thinking_output="text")
    assert body["reasoning"] == {
        "effort": "high",
        "summary": "auto",
        "context": "all_turns",
    }
    assert body["include"] == ["reasoning.encrypted_content"]
    # No summary unless one is to be shown, and only GPT-5.6 on reasons over every turn.
    assert _encode(Request(), "gpt-5.5", thinking_effort="low")["reasoning"] == {
        "effort": "low",
    }


def test_encode_sends_the_tier_and_a_temperature_only_where_they_apply() -> None:
    capability, settings = _model("gpt-4.1")
    body = responses.encode(
        Request(),
        capability=capability,
        settings=settings,
        temperature=1.0,
    )
    assert body["temperature"] == 1.0
    assert "service_tier" not in body
    capability, settings = _model()
    settings = replace(settings, service_tier="priority")
    body = responses.encode(
        Request(),
        capability=capability,
        settings=settings,
        temperature=1.0,
    )
    # A model that can reason rejects a temperature.
    assert "temperature" not in body
    assert body["service_tier"] == "priority"


# ---- the reply ---------------------------------------------------------------


def _reader(
    name: str = "luna-6.0",
    tier: ServiceTier = "auto",
) -> responses.StreamReader:
    capability, settings = _model(name)
    settings = replace(settings, service_tier=tier)
    return responses.StreamReader(capability=capability, settings=settings)


def _completed(**response: object) -> dict[str, object]:
    return {
        "type": "response.completed",
        "response": {"status": "completed", **response},
    }


def _done(index: int, item: Mapping[str, object]) -> dict[str, object]:
    return {"type": "response.output_item.done", "output_index": index, "item": item}


def test_feed_returns_the_text_and_reasoning_as_they_arrive() -> None:
    reader = _reader()
    assert reader.feed(
        {"type": "response.output_text.delta", "delta": "Hi"},
    ) == TextChunk(text="Hi")
    for kind in (
        "response.reasoning_summary_text.delta",
        "response.reasoning_text.delta",
    ):
        assert reader.feed({"type": kind, "delta": "hmm"}) == TextChunk(
            text="hmm",
            thinking=True,
        )
    assert reader.feed({"type": "response.created"}) is None
    assert reader.feed({"type": "response.output_text.done", "text": "Hi"}) is None


def test_a_refusal_is_shown_once_whether_streamed_or_sent_whole() -> None:
    reader = _reader()
    streamed = {"type": "response.refusal.delta", "item_id": "msg_1", "delta": "No."}
    assert reader.feed(streamed) == TextChunk(text="No.")
    assert (
        reader.feed(
            {"type": "response.refusal.done", "item_id": "msg_1", "refusal": "No."},
        )
        is None
    )
    whole = {"type": "response.refusal.done", "item_id": "msg_2", "refusal": "Never."}
    assert reader.feed(whole) == TextChunk(text="Never.")


def test_the_response_holds_each_finished_item_in_output_order() -> None:
    reader = _reader()
    reader.feed(
        _done(
            2,
            {
                "type": "function_call",
                "call_id": "call_9",
                "name": "Read",
                "arguments": '{"file_path": "a"}',
            },
        ),
    )
    reader.feed(
        _done(
            1,
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "Reading "},
                    {"type": "output_text", "text": "a."},
                ],
            },
        ),
    )
    reader.feed(
        _done(
            0,
            {
                "type": "reasoning",
                "id": "rs_7",
                "summary": [
                    {"type": "summary_text", "text": "First."},
                    {"type": "summary_text", "text": "Then."},
                ],
                "content": [{"type": "reasoning_text", "text": "raw"}],
                "encrypted_content": "sealed",
            },
        ),
    )
    reader.feed(
        _done(
            3,
            {
                "type": "function_call",
                "call_id": "call_8",
                "name": "Read",
                "arguments": "{not json",
            },
        ),
    )
    reader.feed(_done(7, {"type": "function_call", "call_id": "call_7", "name": "Ls"}))
    reader.feed(_done(4, {"type": "message", "content": []}))
    reader.feed(_done(5, {"type": "reasoning", "id": "rs_8", "summary": []}))
    reader.feed(_done(6, {"type": "web_search_call"}))
    reader.feed(_completed(usage={"input_tokens": 10, "output_tokens": 4}))
    response = reader.response()
    assert response.records == (
        Thinking(
            content="raw",
            summary="First.\n\nThen.",
            encrypted="sealed",
            extra={"reasoning_id": "rs_7"},
        ),
        AssistantMessage(content="Reading a."),
        ToolCall(call_id="call_9", name="Read", arguments={"file_path": "a"}),
        # Arguments that do not parse are none, so the tool reports the bad call.
        ToolCall(call_id="call_8", name="Read", arguments={}),
        ToolCall(call_id="call_7", name="Ls", arguments={}),
    )
    assert response.stop_reason == "tool_use"
    sealed_only = _reader()
    sealed_only.feed(
        _done(0, {"type": "reasoning", "id": "rs_9", "encrypted_content": "s"}),
    )
    sealed_only.feed(_completed())
    assert sealed_only.response().records == (
        Thinking(encrypted="s", extra={"reasoning_id": "rs_9"}),
    )


def test_the_tokens_split_the_cached_prompt_from_the_uncached() -> None:
    reader = _reader()
    usage = {
        "input_tokens": 1_000,
        "input_tokens_details": {"cached_tokens": 600, "cache_write_tokens": 300},
        "output_tokens": 50,
    }
    reader.feed(_completed(usage=usage))
    tokens = reader.response().tokens
    assert tokens == TokenCount(
        request=100,
        response=50,
        cache_write=300,
        cache_read=600,
    )


@pytest.mark.parametrize(
    ("events", "stop_reason"),
    [
        ([_completed()], "end_turn"),
        (
            [
                {
                    "type": "response.incomplete",
                    "response": {
                        "status": "incomplete",
                        "incomplete_details": {"reason": "max_output_tokens"},
                    },
                },
            ],
            "max_tokens",
        ),
        (
            [
                {
                    "type": "response.incomplete",
                    "response": {
                        "status": "incomplete",
                        "incomplete_details": {"reason": "content_filter"},
                    },
                },
            ],
            "refusal",
        ),
        (
            [
                _done(
                    0,
                    {
                        "type": "message",
                        "content": [{"type": "refusal", "refusal": "No."}],
                    },
                ),
                _completed(),
            ],
            "refusal",
        ),
    ],
)
def test_the_stop_reason_is_in_the_shared_vocabulary(
    events: list[dict[str, object]],
    stop_reason: str,
) -> None:
    reader = _reader()
    for event in events:
        reader.feed(event)
    assert reader.response().stop_reason == stop_reason


def test_a_refusal_is_kept_as_what_the_model_said() -> None:
    reader = _reader()
    reader.feed(
        _done(
            0,
            {
                "type": "message",
                "content": [{"type": "refusal", "refusal": "I can't."}],
            },
        ),
    )
    reader.feed(_completed())
    assert reader.response().records == (AssistantMessage(content="I can't."),)


def _cost(name: str, tokens: TokenCount, tier: ServiceTier) -> float:
    price = _model(name)[0].prices[tier, tokens.prompt, datetime.now(UTC).date()]
    return (price * tokens).total


@pytest.mark.parametrize(
    ("name", "asked", "served", "billed"),
    [
        ("luna-6.0", "auto", "priority", "priority"),
        ("luna-6.0", "priority", "default", "auto"),
        ("luna-6.0", "auto", "fast", "priority"),
        ("luna-6.0", "flex", "brand-new-tier", "flex"),
        # Served at a tier the model has no card for: billed as asked.
        ("gpt-pro-5.5", "flex", "priority", "flex"),
        ("luna-6.0", "flex", None, "auto"),
    ],
)
def test_the_reply_is_priced_at_the_tier_served(
    name: str,
    asked: ServiceTier,
    served: str | None,
    billed: ServiceTier,
) -> None:
    reader = _reader(name, tier=asked)
    usage = {"input_tokens": 1_000, "output_tokens": 1_000}
    reader.feed(_completed(usage=usage, service_tier=served))
    tokens = TokenCount(request=1_000, response=1_000)
    assert reader.response().spend.total == _cost(name, tokens, billed)
    assert _cost(name, tokens, billed) != _cost(
        name,
        tokens,
        "auto" if billed != "auto" else "flex",
    )


def test_a_reply_that_never_finished_is_a_disconnect() -> None:
    reader = _reader()
    reader.feed({"type": "response.output_text.delta", "delta": "Hal"})
    with pytest.raises(DisconnectedError):
        reader.response()


def test_a_failed_reply_or_error_event_raises_what_it_means() -> None:
    failed = {
        "type": "response.failed",
        "response": {"error": {"code": "rate_limit_exceeded", "message": "Slow down."}},
    }
    with pytest.raises(RateLimitedError, match="Slow down"):
        _reader().feed(failed)
    error = {"type": "error", "code": "content_filter", "message": "Blocked."}
    with pytest.raises(RefusedError, match="Blocked"):
        _reader().feed(error)
    with pytest.raises(RuntimeError, match="failed"):
        _reader().feed({"type": "response.failed", "response": {"status": "failed"}})


def test_a_model_of_unknown_reply_length_is_sent_no_cap() -> None:
    capability, _ = _model()
    capability = replace(capability, limits=ModelLimits())
    body = responses.encode(
        Request(),
        capability=capability,
        settings=capability.default,
    )
    assert "max_output_tokens" not in body


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
