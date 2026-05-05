from __future__ import annotations

from types import MappingProxyType

import pytest

from sagent.custom_types import (
    BytesMessage,
    JsonMessage,
    MessageBase,
    MultipartMessage,
    TextMessage,
)
from sagent.lib.json import json_freeze
from sagent.lib.message import tool_call_message


def _roundtrip(
    msg: TextMessage | MultipartMessage | BytesMessage | JsonMessage,
) -> TextMessage | MultipartMessage | BytesMessage | JsonMessage:
    return MessageBase.deserialize(msg.serialize())


def test_str_roundtrip() -> None:
    msg = TextMessage("hello", "text/plain")
    got = _roundtrip(msg)
    assert isinstance(got, TextMessage)
    assert got.content == "hello"


def test_bytes_roundtrip() -> None:
    payload = b"\x89PNG\r\n\x1a\n\x00\x00"
    msg = BytesMessage(payload, "image/png")
    got = _roundtrip(msg)
    assert isinstance(got, BytesMessage)
    assert got.content == payload


def test_json_roundtrip() -> None:
    frozen = json_freeze({"key": "value", "n": 42})
    msg = JsonMessage(frozen, "application/json")
    got = _roundtrip(msg)
    assert isinstance(got, JsonMessage)
    assert isinstance(got.content, MappingProxyType)
    assert got.content["key"] == "value"
    assert got.content["n"] == 42


def test_tuple_roundtrip() -> None:
    children: tuple[TextMessage | BytesMessage, ...] = (
        TextMessage("a", "text/plain"),
        BytesMessage(b"\xff", "application/pdf"),
    )
    msg = MultipartMessage(children, "multipart/x-tool-result")
    got = _roundtrip(msg)
    assert isinstance(got, MultipartMessage)
    assert len(got.content) == 2
    assert isinstance(got.content[0], TextMessage)
    assert got.content[0].content == "a"
    assert isinstance(got.content[1], BytesMessage)
    assert got.content[1].content == b"\xff"


def test_message_base_not_constructible() -> None:
    with pytest.raises(TypeError, match="Use TextMessage"):
        MessageBase()


def test_tool_call_message() -> None:
    msg = tool_call_message("tc1", "Bash", json_freeze({"command": "ls"}))
    assert msg.descriptor == "multipart/x-tool-call"
    assert len(msg.content) == 2
    assert msg.content[0].descriptor == "text/x-queue-id"
    assert msg.content[1].descriptor == "application/x-tool-bash"
