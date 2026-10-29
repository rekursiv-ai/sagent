"""Tests for the Chat Completions translation: request bodies and streamed replies."""

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
from sagent.model.exceptions import DisconnectedError, RateLimitedError
from sagent.model.types import Request, TextChunk
from sagent.providers.catalog import choose
from sagent.providers.openai import chat, wire
from sagent.providers.openai.compat import models
from sagent.session.types import (
    AssistantMessage,
    Attachment,
    Thinking,
    TokenUsage,
    ToolCall,
    UncategorizedToolResult,
    UserMessage,
)
from sagent.tools.read import Read


if TYPE_CHECKING:
    from sagent.model.capability import ModelCapability


def _model(name: str) -> tuple[ModelCapability, ModelSettings]:
    """Return a DashScope row: a vendor whose rows span every thinking shape."""
    table = models(
        "sagent.providers.dashscope",
        document="dashscope.yaml",
        section="dashscope",
    )
    return table.resolve(name)


def _encode(
    request: Request,
    name: str = "qwen-plus-3.7",
    *,
    reasoning_field: str = "reasoning_content",
    temperature: float | None = None,
    **choices: object,
) -> dict[str, object]:
    capability, settings = _model(name)
    for field, value in choices.items():
        settings = choose(settings, field, value)
    capability.check(settings)
    return chat.encode(
        request,
        capability=capability,
        settings=settings,
        reasoning_field=reasoning_field,
        temperature=temperature,
    )


def _png() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (2, 2), "green").save(out, format="PNG")
    return out.getvalue()


# ---- the request -------------------------------------------------------------


def test_encode_streams_the_conversation_with_its_usage() -> None:
    request = Request(
        system=("Be terse.", "", "Mind the tests."),
        records=(UserMessage(content="hi"),),
    )
    assert _encode(request, temperature=1.0, thinking_effort="none") == {
        "model": "qwen3.7-plus",
        "messages": [
            {"role": "system", "content": "Be terse.\n\nMind the tests."},
            {"role": "user", "content": "hi"},
        ],
        "temperature": 1.0,
        "max_completion_tokens": 65_536,
        "stream": True,
        "stream_options": {"include_usage": True},
        "reasoning_effort": "none",
    }
    # A model that cannot reason takes ``max_tokens`` and no effort; no prompt
    # sends no system.
    body = _encode(Request(), "qwen-coder-480b-a35b-3.0")
    assert body["max_tokens"] == 65_536
    assert body["messages"] == []
    assert "max_completion_tokens" not in body
    assert "temperature" not in body
    assert "reasoning_effort" not in body


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("qwen-plus-3.7", "max_completion_tokens"),
        ("qwen-coder-480b-a35b-3.0", "max_tokens"),
    ],
)
def test_encode_asks_no_more_than_the_room_the_request_leaves(
    name: str,
    key: str,
) -> None:
    body = _encode(Request(max_response_tokens=9_000), name)
    assert body[key] == 9_000


def test_encode_merges_the_models_records_into_one_assistant_message() -> None:
    png = _png()
    image = Attachment(mime_descriptor="image/png", data=png)
    url = wire.image_url(image, limits=_model("qwen-plus-3.7")[0].limits)
    records = (
        UserMessage(
            content="look",
            attachments=(image, Attachment(mime_descriptor="text/plain", data=b"t")),
        ),
        Thinking(content="I should "),
        Thinking(summary="read it."),
        AssistantMessage(content="Reading"),
        AssistantMessage(content=" both."),
        ToolCall(call_id="toolu_a", name="Read", arguments={"file_path": "a.py"}),
        ToolCall(call_id="toolu_b", name="Read", arguments={"file_path": "b.py"}),
        TokenUsage(),
        UncategorizedToolResult(
            call_id="toolu_a",
            content="x = 1",
            attachments=(image,),
        ),
        UncategorizedToolResult(
            call_id="toolu_b",
            content="gone",
            is_error=True,
            attachments=(image,),
        ),
        Thinking(content="hmm"),
        UserMessage(content="next"),
        ToolCall(call_id="toolu_c", name="Read"),
    )
    image_block = {"type": "image_url", "image_url": {"url": url}}
    assert _encode(Request(records=records))["messages"] == [
        {"role": "user", "content": [image_block, {"type": "text", "text": "look"}]},
        {
            "role": "assistant",
            "content": "Reading both.",
            "reasoning_content": "I should read it.",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "Read", "arguments": '{"file_path": "a.py"}'},
                },
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "Read", "arguments": '{"file_path": "b.py"}'},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_0", "content": "x = 1"},
        {"role": "tool", "tool_call_id": "call_1", "content": "[Error] gone"},
        # A tool message cannot carry an image; the run's images follow it.
        {"role": "user", "content": [image_block, image_block]},
        {"role": "assistant", "content": None, "reasoning_content": "hmm"},
        {"role": "user", "content": "next"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "Read", "arguments": "{}"},
                },
            ],
        },
    ]


def test_encode_leaves_reasoning_out_for_a_server_that_reads_none_back() -> None:
    records = (
        Thinking(content="I should."),
        AssistantMessage(content="Done."),
        UserMessage(content="ok"),
        # Reasoning alone, unsent, leaves an empty message, which is dropped.
        Thinking(content="hmm"),
    )
    assert _encode(Request(records=records), reasoning_field="")["messages"] == [
        {"role": "assistant", "content": "Done."},
        {"role": "user", "content": "ok"},
    ]
    only_image = (
        UserMessage(
            attachments=(Attachment(mime_descriptor="image/png", data=_png()),),
        ),
    )
    messages = _encode(Request(records=only_image))["messages"]
    url = wire.image_url(
        only_image[0].attachments[0],
        limits=_model("qwen-plus-3.7")[0].limits,
    )
    # No text says nothing, so no text block rides with the image.
    assert messages == [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]},
    ]


@pytest.mark.parametrize(
    ("effort", "wire_effort"),
    [
        ("none", "none"),
        ("min", "minimal"),
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
        ("xhigh", "high"),
        ("max", "high"),
    ],
)
def test_encode_spells_the_effort_in_openais_words(
    effort: str,
    wire_effort: str,
) -> None:
    assert _encode(Request(), thinking_effort=effort)["reasoning_effort"] == wire_effort


def test_encode_offers_the_tools_and_names_a_tier_only_when_chosen() -> None:
    read = Read()
    body = _encode(Request(tools=(read,)), "qwen-coder-480b-a35b-3.0")
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "Read",
                "description": read.description,
                "parameters": mutable(read.schema),
            },
        },
    ]
    assert "reasoning_effort" not in body
    assert "service_tier" not in body
    capability, settings = models(
        "sagent.providers.minimax",
        document="minimax.yaml",
        section="minimax",
    ).resolve("minimax-3.0")
    settings = replace(settings, service_tier="priority")
    body = chat.encode(
        Request(),
        capability=capability,
        settings=settings,
        reasoning_field="",
    )
    assert body["service_tier"] == "priority"


# ---- the reply ---------------------------------------------------------------


def _reader(
    name: str = "qwen-plus-3.7",
    *,
    reasoning_field: str = "reasoning_content",
) -> chat.StreamReader:
    capability, settings = _model(name)
    return chat.StreamReader(
        capability=capability,
        settings=settings,
        reasoning_field=reasoning_field,
    )


def _chunk(finish: str | None = None, **delta: object) -> dict[str, object]:
    return {"choices": [{"delta": delta, "finish_reason": finish}], "usage": None}


def test_feed_returns_the_text_and_reasoning_as_they_arrive() -> None:
    reader = _reader()
    assert reader.feed(_chunk(role="assistant")) is None
    assert reader.feed(_chunk(reasoning_content="Hmm.")) == TextChunk(
        text="Hmm.",
        thinking=True,
    )
    assert reader.feed(_chunk(content="Hi")) == TextChunk(text="Hi", record=1)
    assert reader.feed(_chunk(content=" there")) == TextChunk(text=" there", record=1)
    assert reader.feed({"choices": [], "usage": None}) is None
    assert reader.feed(_chunk("stop")) is None
    response = reader.response()
    assert response.records == (
        Thinking(content="Hmm."),
        AssistantMessage(content="Hi there"),
    )
    assert response.stop_reason == "end_turn"


def test_a_server_that_streams_no_reasoning_has_its_field_ignored() -> None:
    reader = _reader(reasoning_field="")
    assert reader.feed(_chunk(reasoning_content="Hmm.")) is None
    reader.feed(_chunk("stop", content="Hi"))
    assert reader.response().records == (AssistantMessage(content="Hi"),)


def test_a_tool_calls_fragments_join_by_index() -> None:
    reader = _reader()
    first = {
        "index": 0,
        "id": "call_x",
        "function": {"name": "Read", "arguments": '{"file'},
    }
    second = {
        "index": 1,
        "id": "call_y",
        "function": {"name": "Read", "arguments": "{"},
    }
    reader.feed(_chunk(tool_calls=[first, second]))
    reader.feed(
        _chunk(tool_calls=[{"index": 0, "function": {"arguments": '_path": "a"}'}}]),
    )
    reader.feed(_chunk(tool_calls=[{"index": 1, "function": {"arguments": "bad"}}]))
    # A fragment that never names its call is no call.
    reader.feed(_chunk(tool_calls=[{"index": 2, "function": {"arguments": "{}"}}]))
    reader.feed(_chunk("tool_calls"))
    response = reader.response()
    assert response.records == (
        ToolCall(call_id="call_x", name="Read", arguments={"file_path": "a"}),
        ToolCall(call_id="call_y", name="Read", arguments={}),
    )
    assert response.stop_reason == "tool_use"


def test_arguments_that_are_json_but_no_object_are_none() -> None:
    reader = _reader()
    call = {
        "index": 0,
        "id": "call_x",
        "function": {"name": "Read", "arguments": "[1]"},
    }
    reader.feed(_chunk(tool_calls=[call]))
    reader.feed(_chunk("tool_calls"))
    assert reader.response().records == (
        ToolCall(call_id="call_x", name="Read", arguments={}),
    )


def test_the_tokens_split_the_cached_prompt_from_the_uncached_and_are_priced() -> None:
    reader = _reader()
    reader.feed(_chunk("stop", content="ok"))
    # A trailing chunk that names no finish leaves the one named standing.
    reader.feed(_chunk())
    usage = {
        "prompt_tokens": 1_000,
        "completion_tokens": 20,
        "prompt_tokens_details": {"cached_tokens": 700, "cache_write_tokens": 200},
    }
    reader.feed({"choices": [], "usage": usage})
    reader.feed({"choices": [], "usage": None})
    response = reader.response()
    tokens = TokenCount(request=100, response=20, cache_write=200, cache_read=700)
    assert response.tokens == tokens
    capability, _ = _model("qwen-plus-3.7")
    cost = capability.prices["auto", tokens.prompt, datetime.now(UTC).date()] * tokens
    assert response.spend == cost
    assert cost.total > 0


@pytest.mark.parametrize(
    ("finish", "refusal", "stop_reason"),
    [
        ("stop", "", "end_turn"),
        ("length", "", "max_tokens"),
        ("function_call", "", "tool_use"),
        ("content_filter", "", "refusal"),
        ("stop", "I can't.", "refusal"),
        ("eos", "", "eos"),
    ],
)
def test_the_stop_reason_is_in_the_shared_vocabulary(
    finish: str,
    refusal: str,
    stop_reason: str,
) -> None:
    reader = _reader()
    if refusal:
        assert reader.feed(_chunk(refusal=refusal)) == TextChunk(text=refusal)
    reader.feed(_chunk(finish))
    assert reader.response().stop_reason == stop_reason


def test_calls_under_a_plain_stop_are_tool_use() -> None:
    reader = _reader()
    reader.feed(
        _chunk(tool_calls=[{"index": 0, "id": "c", "function": {"name": "Read"}}]),
    )
    reader.feed(_chunk("stop"))
    assert reader.response().stop_reason == "tool_use"


def test_a_reply_that_never_finished_is_a_disconnect() -> None:
    reader = _reader()
    reader.feed(_chunk(content="Hal"))
    with pytest.raises(DisconnectedError):
        reader.response()


def test_an_in_stream_error_raises_what_it_means() -> None:
    with pytest.raises(RateLimitedError, match="Slow"):
        _reader().feed({"error": {"code": "rate_limit_exceeded", "message": "Slow"}})
    with pytest.raises(RuntimeError, match="broke"):
        _reader().feed({"object": "error", "message": "broke"})
    # A chunk that names no error is not one.
    assert _reader().feed({"error": None, **_chunk(content="ok")}) == TextChunk(
        text="ok",
    )


def test_a_model_of_unknown_reply_length_is_sent_no_cap() -> None:
    capability, _ = _model("qwen-plus-3.7")
    capability = replace(capability, limits=ModelLimits())
    settings = capability.default
    body = chat.encode(
        Request(),
        capability=capability,
        settings=settings,
        reasoning_field="",
    )
    assert not {"max_tokens", "max_completion_tokens"} & body.keys()


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
