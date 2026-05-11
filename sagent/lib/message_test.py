"""Tests for message accessors."""

from __future__ import annotations

from collections.abc import Mapping
from typing import NoReturn, cast

from sagent.custom_types import (
    JsonMessage,
    Message,
    MultipartMessage,
    TextMessage,
)
from sagent.lib.json import json_freeze
from sagent.lib.message import (
    build_error_message,
    get_directive,
    get_queue_id,
    get_tool_name,
    response_text,
    response_tool_calls,
    thinking_text,
)


def _plain(text: str) -> Message:
    return TextMessage(text, "text/plain")


def _model_msg(*parts: Message) -> Message:
    return MultipartMessage(tuple(parts), "multipart/x-model-message")


# -- get_queue_id ----------------------------------------------------------


def test_get_queue_id_non_multipart():
    assert get_queue_id(TextMessage("hi", "text/plain")) == ""


def test_get_queue_id_multipart():
    msg = _model_msg(TextMessage("q1", "text/x-queue-id"), _plain("hi"))
    assert get_queue_id(msg) == "q1"


# -- get_tool_name ---------------------------------------------------------


def test_get_tool_name_non_multipart():
    assert get_tool_name(TextMessage("hi", "text/plain")) == ""


def test_get_tool_name_multipart():
    msg = _model_msg(JsonMessage(json_freeze({}), "application/x-tool-bash"))
    assert get_tool_name(msg) == "bash"


# -- get_directive ---------------------------------------------------------


def test_get_directive_non_multipart():
    assert get_directive(TextMessage("hi", "text/plain")) == json_freeze({})


def test_get_directive_no_tool_part():
    msg = _model_msg(_plain("text only"))
    assert get_directive(msg) == json_freeze({})


def test_get_directive_with_tool():
    d = json_freeze({"command": "ls"})
    msg = _model_msg(JsonMessage(d, "application/x-tool-bash"))
    assert get_directive(msg) == d


# -- response_text ---------------------------------------------------------


def test_response_text_non_multipart():
    assert response_text(TextMessage("hi", "text/plain")) == ""


def test_response_text_multipart():
    msg = _model_msg(_plain("hello"), _plain("world"))
    assert response_text(msg) == "hello\nworld"


# -- thinking_text ---------------------------------------------------------


def test_thinking_text_plain():
    msg = TextMessage("deep thought", "text/x-thinking")
    assert thinking_text(msg) == "deep thought"


def test_thinking_text_structured():
    content = json_freeze({"thinking": "structured thought", "other": "x"})
    msg = JsonMessage(content, "application/x-thinking-structured")
    assert thinking_text(msg) == "structured thought"


def test_thinking_text_structured_missing_key():
    content = json_freeze({"other": "x"})
    msg = JsonMessage(content, "application/x-thinking-structured")
    assert thinking_text(msg) == ""


# -- response_tool_calls ---------------------------------------------------


def test_response_tool_calls_non_multipart():
    assert response_tool_calls(TextMessage("hi", "text/plain")) == []


def test_response_tool_calls_multipart():
    tc = MultipartMessage((_plain("tc"),), "multipart/x-tool-call")
    msg = _model_msg(_plain("text"), tc)
    assert response_tool_calls(msg) == [tc]


# -- build_error_message ---------------------------------------------------


def test_build_error_message_flat():
    msg = build_error_message("oops")
    assert isinstance(msg, TextMessage)
    assert msg.descriptor == "text/x-error"
    assert msg.content == "oops"


def _raise_value_error() -> NoReturn:
    raise ValueError("bad input")


def _raise_runtime_error() -> NoReturn:
    raise RuntimeError("boom")


def _raise_value_root() -> NoReturn:
    raise ValueError("root")


def _raise_chained_runtime() -> NoReturn:
    try:
        _raise_value_root()
    except ValueError as inner:
        raise RuntimeError("wrapper") from inner


def test_build_error_message_with_exception():
    try:
        _raise_value_error()
    except ValueError as e:
        msg = build_error_message("validation failed", e)
    assert isinstance(msg, MultipartMessage)
    assert msg.descriptor == "multipart/x-error"
    assert len(msg.content) == 2
    text_part, trace_part = msg.content
    assert isinstance(text_part, TextMessage)
    assert text_part.descriptor == "text/x-error"
    assert text_part.content == "validation failed"
    assert isinstance(trace_part, JsonMessage)
    assert trace_part.descriptor == "application/x-stack-trace"
    trace = cast(Mapping[str, object], trace_part.content)
    assert trace["type"] == "ValueError"
    assert "bad input" in cast(str, trace["message"])
    frames = cast(list[Mapping[str, object]], trace["frames"])
    assert len(frames) >= 1
    assert "test_build_error_message_with_exception" in {
        cast(str, f["function"]) for f in frames
    }
    assert trace["cause"] is None
    assert trace["context"] is None


def test_build_error_message_chained_cause():
    try:
        _raise_chained_runtime()
    except RuntimeError as e:
        msg = build_error_message("outer", e)
    assert isinstance(msg, MultipartMessage)
    trace_part = msg.content[1]
    assert isinstance(trace_part, JsonMessage)
    trace = cast(Mapping[str, object], trace_part.content)
    assert trace["type"] == "RuntimeError"
    cause = cast(Mapping[str, object], trace["cause"])
    assert cause["type"] == "ValueError"
    assert "root" in cast(str, cause["message"])


def test_build_error_message_round_trips_serialization():
    try:
        _raise_runtime_error()
    except RuntimeError as e:
        msg = build_error_message("oops", e)
    serialized = msg.serialize()
    restored = MultipartMessage.deserialize(serialized)
    assert isinstance(restored, MultipartMessage)
    assert restored.descriptor == "multipart/x-error"
    text_part, trace_part = restored.content
    assert text_part.content == "oops"
    assert isinstance(trace_part, JsonMessage)
    trace = cast(Mapping[str, object], trace_part.content)
    assert trace["type"] == "RuntimeError"


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
