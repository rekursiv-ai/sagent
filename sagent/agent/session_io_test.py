"""Tests for agent.session_io functions."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import hashlib
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
from sagent.tools.core import ToolState


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

        original = session_file.read_text(encoding="utf-8")

        loaded = session_io.load_session(tmp_path, {})

        assert loaded is not None
        meta, messages = loaded
        assert meta["session_id"] == "s1"
        assert messages == [msg]
        corrupt_files = list(tmp_path.glob("session.jsonl.corrupt-*"))
        assert len(corrupt_files) == 1
        assert corrupt_files[0].read_text(encoding="utf-8") == original

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


class TestRebuildToolState:
    """Resume rebuild — file-stat path and back-compat path.

    Verifies that the resume bug ("change-detection mtime baseline gets
    reset to now at resume") is fixed when ``application/x-file-stat``
    parts are present, and that pre-file-stat sessions still load.
    """

    def _read_use(self, qid: str, file_path: str) -> Message:
        return tool_call_message(qid, "read", json_freeze({"file_path": file_path}))

    def _read_result(
        self,
        *,
        qid: str,
        text: str,
        stat_path: str | None,
        stat_mtime: float | None,
    ) -> Message:
        parts: list[Message] = [
            TextMessage(qid, "text/x-queue-id"),
            TextMessage(text, "text/plain"),
        ]
        if stat_path is not None:
            parts.append(
                JsonMessage(
                    json_freeze(
                        {
                            "path": stat_path,
                            "mtime": stat_mtime if stat_mtime is not None else 0.0,
                            "sha256": "deadbeef",
                        }
                    ),
                    "application/x-file-stat",
                )
            )
        return MultipartMessage(tuple(parts), "multipart/x-tool-result")

    def test_file_stat_used_when_present(self, tmp_path: Path) -> None:
        f = tmp_path / "x.txt"
        f.write_text("1\thello\n")
        original_mtime = 1_700_000_000.0  # arbitrary past timestamp
        messages: list[Message] = [
            AssistantMessage(tool_calls=[self._read_use("q1", str(f))]),
            self._read_result(
                qid="q1",
                text="1\thello\n",
                stat_path=str(f),
                stat_mtime=original_mtime,
            ),
        ]
        state = ToolState()
        session_io.rebuild_tool_state_from_messages(messages, state)
        cached = state.read_cache[str(f.resolve())]
        assert cached.mtime == original_mtime, (
            "file-stat mtime should override stat-now"
        )

    def test_no_file_stat_falls_back_to_stat_at_rebuild(self, tmp_path: Path) -> None:
        """Pre-file-stat sessions must use stat-at-rebuild, not synthetic mtime.

        Using ``msg.timestamp`` as a fallback would trigger noisy
        full-file diffs on first resume request for paths that cache no
        content (Edit, partial Read). The fallback is therefore ``None``
        so ``mark_read`` falls through to ``Path.stat()`` — same as the
        pre-commit behavior.
        """
        f = tmp_path / "y.txt"
        f.write_text("1\thi\n")
        disk_mtime = f.stat().st_mtime
        messages: list[Message] = [
            AssistantMessage(tool_calls=[self._read_use("q2", str(f))]),
            self._read_result(
                qid="q2", text="1\thi\n", stat_path=None, stat_mtime=None
            ),
        ]
        state = ToolState()
        session_io.rebuild_tool_state_from_messages(messages, state)
        cached = state.read_cache[str(f.resolve())]
        assert cached.mtime == disk_mtime, (
            "missing file-stat should use stat-at-rebuild, not msg.timestamp"
        )
        # And consume_changed_files should be quiet because mtime matches.
        assert state.consume_changed_files() == {}

    def test_bash_state_restores_cwd_from_history(self, tmp_path: Path) -> None:
        target_cwd = str(tmp_path)
        bash_use = tool_call_message(
            "b1", "bash", json_freeze({"command": "cd " + target_cwd})
        )
        bash_result = MultipartMessage(
            (
                TextMessage("b1", "text/x-queue-id"),
                TextMessage("", "text/plain"),
                JsonMessage(
                    json_freeze({"cwd": target_cwd}),
                    "application/x-bash-state",
                ),
            ),
            "multipart/x-tool-result",
        )
        messages: list[Message] = [
            AssistantMessage(tool_calls=[bash_use]),
            bash_result,
        ]
        state = ToolState()
        # Pre-rebuild cwd is process cwd; rebuild should override.
        original_cwd = state.bash_cwd
        session_io.rebuild_tool_state_from_messages(messages, state)
        assert state.bash_cwd == target_cwd
        assert state.bash_cwd != original_cwd or target_cwd == original_cwd

    def test_write_rebuild_uses_directive_content(self, tmp_path: Path) -> None:
        """Write rebuild caches the directive's content with file-stat mtime."""
        f = tmp_path / "w.txt"
        written_content = "hello from write\n"
        f.write_text(written_content)
        original_mtime = 1_700_000_000.0
        write_use = tool_call_message(
            "w1",
            "write",
            json_freeze({"file_path": str(f), "content": written_content}),
        )
        write_result = MultipartMessage(
            (
                TextMessage("w1", "text/x-queue-id"),
                TextMessage(f"Wrote {len(written_content)} bytes", "text/plain"),
                JsonMessage(
                    json_freeze(
                        {
                            "path": str(f.resolve()),
                            "mtime": original_mtime,
                            "sha256": "irrelevant-for-write",
                        }
                    ),
                    "application/x-file-stat",
                ),
            ),
            "multipart/x-tool-result",
        )
        messages: list[Message] = [
            AssistantMessage(tool_calls=[write_use]),
            write_result,
        ]
        state = ToolState()
        session_io.rebuild_tool_state_from_messages(messages, state)
        cached = state.read_cache[str(f.resolve())]
        assert cached.mtime == original_mtime

    def test_edit_rebuild_caches_disk_when_sha_matches(self, tmp_path: Path) -> None:
        """Edit rebuild trusts disk content when its sha matches file-stat."""
        f = tmp_path / "e.txt"
        post_edit_content = "post-edit content\n"
        f.write_text(post_edit_content)
        post_edit_sha = hashlib.sha256(post_edit_content.encode()).hexdigest()
        original_mtime = 1_700_000_000.0
        edit_use = tool_call_message(
            "e1",
            "edit",
            json_freeze(
                {
                    "file_path": str(f),
                    "old_string": "pre",
                    "new_string": "post",
                }
            ),
        )
        edit_result = MultipartMessage(
            (
                TextMessage("e1", "text/x-queue-id"),
                TextMessage("Replaced 1 occurrence(s)", "text/plain"),
                JsonMessage(
                    json_freeze(
                        {
                            "path": str(f.resolve()),
                            "mtime": original_mtime,
                            "sha256": post_edit_sha,
                        }
                    ),
                    "application/x-file-stat",
                ),
            ),
            "multipart/x-tool-result",
        )
        messages: list[Message] = [
            AssistantMessage(tool_calls=[edit_use]),
            edit_result,
        ]
        state = ToolState()
        session_io.rebuild_tool_state_from_messages(messages, state)
        cached = state.read_cache[str(f.resolve())]
        assert cached.mtime == original_mtime
        # consume_changed_files would diff cache vs disk; matching sha
        # means cache was populated from disk content → empty diff →
        # ``if diff:`` guard excludes the path from changes.
        changes = state.consume_changed_files()
        assert changes == {}

    def test_edit_rebuild_skips_disk_when_sha_mismatches(self, tmp_path: Path) -> None:
        """Edit rebuild leaves content empty when external mod broke sha integrity."""
        f = tmp_path / "ee.txt"
        # Disk content NOW differs from what Edit produced.
        f.write_text("externally modified content\n")
        original_mtime = 1_700_000_000.0
        edit_use = tool_call_message(
            "e2",
            "edit",
            json_freeze(
                {
                    "file_path": str(f),
                    "old_string": "x",
                    "new_string": "y",
                }
            ),
        )
        edit_result = MultipartMessage(
            (
                TextMessage("e2", "text/x-queue-id"),
                TextMessage("Replaced 1 occurrence(s)", "text/plain"),
                JsonMessage(
                    json_freeze(
                        {
                            "path": str(f.resolve()),
                            "mtime": original_mtime,
                            # sha of the post-edit content (NOT what's on disk).
                            "sha256": "0" * 64,
                        }
                    ),
                    "application/x-file-stat",
                ),
            ),
            "multipart/x-tool-result",
        )
        messages: list[Message] = [
            AssistantMessage(tool_calls=[edit_use]),
            edit_result,
        ]
        state = ToolState()
        session_io.rebuild_tool_state_from_messages(messages, state)
        # Cache mtime is from file-stat; cache content is empty due to
        # sha mismatch. consume_changed_files diffs empty vs disk →
        # full file appears as a diff so the model sees the drift.
        changes = state.consume_changed_files()
        assert str(f) in changes
        assert "externally modified content" in changes[str(f)]

    def test_legacy_session_without_either_part_still_loads(
        self, tmp_path: Path
    ) -> None:
        """Pre-bash-state, pre-file-stat sessions must still rebuild cleanly."""
        f = tmp_path / "z.txt"
        f.write_text("1\told\n")
        # Legacy result: queue-id + text/plain only, no JSON siblings.
        legacy_result = MultipartMessage(
            (
                TextMessage("q3", "text/x-queue-id"),
                TextMessage("1\told\n", "text/plain"),
            ),
            "multipart/x-tool-result",
        )
        messages: list[Message] = [
            AssistantMessage(tool_calls=[self._read_use("q3", str(f))]),
            legacy_result,
        ]
        state = ToolState()
        # Should not raise; should mark the path as read so enforce_read passes.
        session_io.rebuild_tool_state_from_messages(messages, state)
        assert state.has_been_read(str(f))


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
