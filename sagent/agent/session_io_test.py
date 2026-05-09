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

    def test_clear_marker_drops_prior_messages(self, tmp_path: Path) -> None:
        """``kind: clear`` resets the live messages list during load."""
        session_file = tmp_path / "session.jsonl"
        before = TextMessage("before clear", "text/x-user-message")
        after = TextMessage("after clear", "text/x-user-message")
        session_file.write_text(
            "\n".join(
                [
                    json.dumps({"kind": "meta", "session_id": "s1"}),
                    json.dumps({"kind": "message", **before.serialize()}),
                    json.dumps({"kind": "clear", "_timestamp": 1}),
                    json.dumps({"kind": "message", **after.serialize()}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        loaded = session_io.load_session(tmp_path, {})

        assert loaded is not None
        _, messages = loaded
        assert messages == [after]

    def test_latest_meta_wins(self, tmp_path: Path) -> None:
        """Multiple ``kind: meta`` lines: loader takes the latest."""
        session_file = tmp_path / "session.jsonl"
        session_file.write_text(
            "\n".join(
                [
                    json.dumps({"kind": "meta", "session_id": "old"}),
                    json.dumps({"kind": "meta", "session_id": "new"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        loaded = session_io.load_session(tmp_path, {})

        assert loaded is not None
        meta, _ = loaded
        assert meta["session_id"] == "new"

    def test_clear_barrier_drops_late_tool_result(self, tmp_path: Path) -> None:
        """Late tool results from pre-clear calls are not live history."""
        session_file = tmp_path / "session.jsonl"
        call = tool_call_message("late", "grep", json_freeze({}))
        assistant = AssistantMessage(tool_calls=[call])
        late_result = ToolResult(
            queue_id="late",
            name="",
            content=(Media("stale output", "text/plain"),),
        )
        after_clear = UserMessage("next")
        session_file.write_text(
            "\n".join(
                [
                    json.dumps({"kind": "meta", "session_id": "s1"}),
                    json.dumps({"kind": "message", **assistant.serialize()}),
                    json.dumps({"kind": "clear", "_timestamp": 1}),
                    json.dumps({"kind": "message", **after_clear.serialize()}),
                    json.dumps({"kind": "message", **late_result.serialize()}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        loaded = session_io.load_session(tmp_path, {})

        assert loaded is not None
        _, messages = loaded
        assert messages == [after_clear]
        assert "stale output" in session_file.read_text(encoding="utf-8")


class TestRepairDanglingToolCalls:
    def test_drops_unexpected_tool_result_after_assistant(self) -> None:
        """Tool results must match the immediately previous assistant calls."""
        msg = AssistantMessage(
            tool_calls=[tool_call_message("expected", "bash", json_freeze({}))],
        )
        repaired = session_io.repair_dangling_tool_calls(
            [
                UserMessage("u"),
                msg,
                ToolResult(
                    queue_id="expected",
                    name="",
                    content=(Media("ok", "text/plain"),),
                ),
                ToolResult(
                    queue_id="stale",
                    name="",
                    content=(Media("stale output", "text/plain"),),
                ),
            ]
        )

        assert [m.descriptor for m in repaired] == [
            "text/x-user-message",
            "multipart/x-model-message",
            "multipart/x-tool-result",
        ]
        assert get_queue_id(repaired[-1]) == "expected"

    def test_drops_orphan_without_clear_barrier(self) -> None:
        repaired = session_io.repair_dangling_tool_calls(
            [
                UserMessage("before"),
                ToolResult(
                    queue_id="stale",
                    name="",
                    content=(Media("stale output", "text/plain"),),
                ),
                UserMessage("after"),
            ]
        )

        assert [m.content for m in repaired] == ["before", "after"]

    def test_drops_leading_tool_result(self) -> None:
        repaired = session_io.repair_dangling_tool_calls(
            [
                ToolResult(
                    queue_id="stale",
                    name="",
                    content=(Media("old output", "text/plain"),),
                ),
                UserMessage("next"),
            ]
        )

        assert len(repaired) == 1
        assert repaired[0].descriptor == "text/x-user-message"
        assert repaired[0].content == "next"

    def test_synthesizes_missing_tool_result(self) -> None:
        repaired = session_io.repair_dangling_tool_calls(
            [
                UserMessage("u"),
                AssistantMessage(
                    tool_calls=[tool_call_message("missing", "bash", json_freeze({}))]
                ),
                UserMessage("next"),
            ]
        )

        assert [m.descriptor for m in repaired] == [
            "text/x-user-message",
            "multipart/x-model-message",
            "multipart/x-tool-result",
            "text/x-user-message",
        ]
        assert get_queue_id(repaired[2]) == "missing"
        assert _is_error_result(repaired[2])

    def test_repair_is_idempotent(self) -> None:
        messages = [
            UserMessage("u"),
            AssistantMessage(
                tool_calls=[tool_call_message("missing", "bash", json_freeze({}))]
            ),
            ToolResult(
                queue_id="stale",
                name="",
                content=(Media("stale output", "text/plain"),),
            ),
            UserMessage("next"),
        ]

        once = session_io.repair_dangling_tool_calls(messages)
        twice = session_io.repair_dangling_tool_calls(once)

        assert twice == once


class TestAppendSession:
    def test_append_creates_and_extends(self, tmp_path: Path) -> None:
        """Two append calls produce one file with both messages."""
        session_file = tmp_path / "session.jsonl"
        m1 = TextMessage("first", "text/x-user-message")
        m2 = TextMessage("second", "text/x-user-message")

        session_io.append_session(
            session_file,
            meta={"session_id": "s1"},
            messages_delta=[m1],
        )
        session_io.append_session(
            session_file,
            meta={"session_id": "s1"},
            messages_delta=[m2],
        )

        loaded = session_io.load_session(tmp_path, {})
        assert loaded is not None
        _, messages = loaded
        assert messages == [m1, m2]

    def test_append_with_clear_barrier(self, tmp_path: Path) -> None:
        """``clear=True`` writes a barrier; loader drops everything before."""
        session_file = tmp_path / "session.jsonl"
        before = TextMessage("doomed", "text/x-user-message")
        after = TextMessage("survives", "text/x-user-message")

        session_io.append_session(
            session_file,
            meta={"session_id": "s1"},
            messages_delta=[before],
        )
        session_io.append_session(
            session_file,
            meta={"session_id": "s1"},
            messages_delta=[after],
            clear=True,
        )

        # File still contains both messages -- not truncated.
        raw = session_file.read_text(encoding="utf-8")
        assert "doomed" in raw
        assert "survives" in raw
        assert '"kind": "clear"' in raw

        loaded = session_io.load_session(tmp_path, {})
        assert loaded is not None
        _, messages = loaded
        assert messages == [after]


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
