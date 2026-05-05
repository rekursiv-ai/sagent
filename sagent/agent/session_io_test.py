"""Tests for agent.session_io functions."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import json

from sagent.agent import session_io
from sagent.custom_types import (
    BytesDescriptor,
    BytesMessage,
    JsonMessage,
    Message,
    MessageBase,
    MessageContent,
    MultipartDescriptor,
    MultipartMessage,
    TextDescriptor,
    TextMessage,
)
from sagent.lib.json import json_freeze
from sagent.lib.message import (
    get_queue_id,
    get_tool_name,
    tool_call_message,
)


# -- Compatibility factories -------------------------------------------


def UserMessage(content: str) -> Message:  # noqa: N802 -- PascalCase factory mimics Message constructor
    return TextMessage(content, "text/x-user-message")


def AssistantMessage(  # noqa: N802 -- PascalCase factory mimics Message constructor
    content: str = "",
    tool_calls: list[Message] | None = None,
    message_id: str = "",
) -> Message:
    parts: list[Message] = []
    if message_id:
        parts.append(TextMessage(message_id, "text/x-queue-id"))
    if content:
        parts.append(TextMessage(content, "text/plain"))
    parts.extend(tool_calls or [])
    return MultipartMessage(tuple(parts), "multipart/x-model-message")


def _retag(p: Message, descriptor: str) -> Message:
    if isinstance(p, TextMessage):
        return TextMessage(p.content, cast(TextDescriptor, descriptor))
    if isinstance(p, MultipartMessage):
        return MultipartMessage(p.content, cast(MultipartDescriptor, descriptor))
    if isinstance(p, BytesMessage):
        return BytesMessage(p.content, cast(BytesDescriptor, descriptor))
    return JsonMessage(p.content, descriptor)


def ToolResult(  # noqa: N802 -- PascalCase factory mimics Message constructor
    *,
    queue_id: str,
    name: str,
    content: tuple[Message, ...],
    is_error: bool = False,
) -> Message:
    del name
    if is_error:
        content = tuple(
            _retag(p, "text/x-error" if p.descriptor == "text/plain" else p.descriptor)
            for p in content
        )
    return MultipartMessage(
        (TextMessage(queue_id, "text/x-queue-id"), *content),
        "multipart/x-tool-result",
    )


def Media(content: MessageContent, descriptor: str) -> Message:  # noqa: N802 -- PascalCase factory mimics Message constructor
    if isinstance(content, str):
        return TextMessage(content, cast(TextDescriptor, descriptor))
    if isinstance(content, tuple):
        return MultipartMessage(
            cast(tuple[Message, ...], content),  # pyright: ignore[reportUnnecessaryCast] -- ty needs the cast; pyright considers it redundant after isinstance
            cast(MultipartDescriptor, descriptor),
        )
    if isinstance(content, bytes):
        return BytesMessage(content, cast(BytesDescriptor, descriptor))
    return JsonMessage(content, descriptor)


def _is_error_result(m: Message) -> bool:
    return (
        m.descriptor == "multipart/x-tool-result"
        and isinstance(m, MultipartMessage)
        and any(p.descriptor == "text/x-error" for p in m.content)
    )


# -- Serialization round-trip -----------------------------------------


class TestSerialization:
    def test_user_message(self) -> None:
        msg = UserMessage(content="hello")
        data = msg.serialize()
        restored = MessageBase.deserialize(data)
        assert restored.descriptor == "text/x-user-message"
        assert restored.content == "hello"

    def test_assistant_message(self) -> None:
        msg = AssistantMessage(
            content="thinking...",
            tool_calls=[
                tool_call_message("t1", "bash", json_freeze({"command": "ls"})),
            ],
        )
        data = msg.serialize()
        restored = MessageBase.deserialize(data)
        assert restored.descriptor == "multipart/x-model-message"
        assert isinstance(restored, MultipartMessage)
        text = next(
            (str(p.content) for p in restored.content if p.descriptor == "text/plain"),
            "",
        )
        assert text == "thinking..."
        tcs = [p for p in restored.content if p.descriptor == "multipart/x-tool-call"]
        assert len(tcs) == 1
        assert get_tool_name(tcs[0]) == "bash"

    def test_tool_result_message(self) -> None:
        msg = ToolResult(
            queue_id="t1",
            name="",
            content=(Media("file1 file2", "text/plain"),),
        )
        data = msg.serialize()
        restored = MessageBase.deserialize(data)
        assert restored.descriptor == "multipart/x-tool-result"
        assert isinstance(restored, MultipartMessage)
        text = next(
            (str(p.content) for p in restored.content if p.descriptor == "text/plain"),
            "",
        )
        assert text == "file1 file2"
        assert get_queue_id(restored) == "t1"

    def test_timing_preserved_for_all_parts(self) -> None:
        msg = AssistantMessage(
            content="hi",
            tool_calls=[tool_call_message("q1", "bash", json_freeze({}))],
        )
        data = msg.serialize()
        restored = MessageBase.deserialize(data)
        assert restored.id == msg.id
        assert restored.timestamp == msg.timestamp
        assert isinstance(restored, MultipartMessage)
        assert isinstance(msg, MultipartMessage)
        for orig, got in zip(msg.content, restored.content, strict=True):
            assert got.id == orig.id
            assert got.timestamp == orig.timestamp

    def test_tool_call_directive_json_serializable(self) -> None:
        """Regression: nested MappingProxyType must not cause json.dumps to fail."""
        msg = AssistantMessage(
            content="",
            tool_calls=[
                tool_call_message(
                    "t1",
                    "read",
                    json_freeze(
                        {
                            "file_path": "/file",
                            "options": {"flag": True, "vals": [1, 2]},
                        }
                    ),
                ),
            ],
        )
        data = msg.serialize()
        result = json.dumps(data)
        assert '"file_path"' in result
        assert '"options"' in result


# -- Session loading ---------------------------------------------------


class TestLoadSession:
    def test_skips_corrupt_jsonl_line(self, tmp_path: Path) -> None:
        session_file = tmp_path / "session.jsonl"
        msg = TextMessage("keep me", "text/x-user-message")
        session_file.write_text(
            json.dumps({"kind": "meta", "session_id": "s1"})
            + "\n{not json}\n"
            + json.dumps({"kind": "message", **msg.serialize()})
            + "\n",
            encoding="utf-8",
        )

        loaded = session_io.load_session(tmp_path, {})

        assert loaded is not None
        meta, messages = loaded
        assert meta["session_id"] == "s1"
        assert messages == [msg]


# -- Serialization edge cases -----------------------------------------


class TestSerializationEdgeCases:
    def test_deserialize_tool_result_with_is_error(self) -> None:
        msg = ToolResult(
            queue_id="x",
            name="",
            content=(Media("oops", "text/plain"),),
            is_error=True,
        )
        restored = MessageBase.deserialize(msg.serialize())
        assert restored.descriptor == "multipart/x-tool-result"
        assert _is_error_result(restored)

    def test_serialize_assistant_no_tool_calls(self) -> None:
        msg = AssistantMessage(content="hi", tool_calls=[])
        data = msg.serialize()
        assert data["descriptor"] == "multipart/x-model-message"
        restored = MessageBase.deserialize(data)
        assert restored.descriptor == "multipart/x-model-message"
        assert isinstance(restored, MultipartMessage)
        text = next(
            (str(p.content) for p in restored.content if p.descriptor == "text/plain"),
            "",
        )
        assert text == "hi"


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
