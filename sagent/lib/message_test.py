"""Tests for message accessors."""

from __future__ import annotations

from sagent.custom_types import (
    JsonMessage,
    Message,
    MultipartMessage,
    TextMessage,
)
from sagent.lib.json import json_freeze
from sagent.lib.message import (
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


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
