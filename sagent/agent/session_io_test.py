"""Tests for ``agent.session_io``: v4 JSONL persistence."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast, override
from unittest.mock import patch

import dataclasses
import json
import os
import stat
import threading

import pytest

from sagent.agent import (
    runtime as agent_runtime,
    session_io,
)
from sagent.agent.agent import Agent
from sagent.agent.context import resolve_context, validate_context
from sagent.agent.session_io import (
    PersistentAgentRecord,
    SessionMeta,
    _apply_update_in_place,
    _att_from_json,
    _entry_from_json,
    _is_barrier_splice,
    _json_bool,
    _mask_from_json,
    _mask_runs,
    _optional_float,
    _optional_int,
    _persisted_refs,
    _ref_from_json,
    _tool_result_kind_from_json,
    append_context_repair,
    append_session,
    load_persistent_agents,
    load_session,
    repair_dangling_tool_calls,
    restore_model,
    restore_tool_state,
    serialize_tool_state,
    unpersisted_session_error,
)
from sagent.agent.state import ReadCacheEntry, ToolState
from sagent.sessions import new_session_dir
from sagent.testing import MockModelCaps
from sagent.types.model import ModelRequest, ModelResponse
from sagent.types.runtime import (
    CANCELLED_PLACEHOLDER,
    DETACHED_PLACEHOLDER,
    AgentSendMessage,
    AssistantMessage,
    BytesMessage,
    ModelContextEvent,
    ModelServiceSuspended,
    NoticeMessage,
    RuntimeEvent,
    SaveSession,
    ServiceErrorSnapshot,
    ToolCall,
    ToolResult,
    ToolResultKind,
    UserMessage,
)
from sagent.types.tape import (
    ContextSplice,
    MaskRange,
    ReferrableTapeEvent,
    TapeRecord,
    TapeRef,
)


class _RuntimeModel:
    async def stream(
        self,
        history: list[ModelContextEvent],
        publish: Callable[[RuntimeEvent], None],
    ) -> AssistantMessage:
        del history, publish
        return AssistantMessage(text="")


@dataclass(slots=True, kw_only=True)
class _NoopModel(MockModelCaps):
    model_id: str = "noop"
    max_request_tokens: int = 100_000
    max_response_tokens: int = 1_024

    @override
    def approx_request_tokens(self, request: ModelRequest) -> int:
        del request
        return 1

    async def buffer(self, request: ModelRequest) -> ModelResponse:
        return await self.stream(request)

    async def stream(
        self,
        request: ModelRequest,
        publish: Callable[[RuntimeEvent], None] | None = None,
    ) -> ModelResponse:
        del request, publish
        return ModelResponse(message=AssistantMessage(text=""))


def _records_from(entries: list[ModelContextEvent]) -> list[TapeRecord]:
    """Wrap each entry as a ``ReferrableTapeEvent`` with a synthetic ref."""
    return [
        ReferrableTapeEvent(ref=TapeRef(session_id="abc", ordinal=i), event=e)
        for i, e in enumerate(entries)
    ]


def _history_from_tape(tape: list[TapeRecord]) -> list[ModelContextEvent]:
    """Resolve a loaded tape to its provider-facing entries."""
    return resolve_context(tape).messages


def test_serialize_tool_state_empty_has_bash_cwd() -> None:
    blob = serialize_tool_state(ToolState())
    assert "bash_cwd" in blob


def test_serialize_tool_state_round_trip(tmp_path: Path) -> None:
    state = ToolState()
    state.bash_cwd = str(tmp_path)
    state.depth = 2
    state.additional_dirs = ["/tmp/a"]  # noqa: S108 -- placeholder
    state.invoked_rules.add("/tmp/a/.sagent/rules/python.md")  # noqa: S108
    state.read_cache["/tmp/x.txt"] = ReadCacheEntry(  # noqa: S108
        offset=0, limit=100, last_lines=10, mtime=1234.5
    )
    blob = serialize_tool_state(state)
    restored = ToolState()
    restore_tool_state(restored, blob)
    assert restored.bash_cwd == state.bash_cwd
    assert restored.depth == state.depth
    assert restored.additional_dirs == state.additional_dirs
    assert restored.invoked_rules == state.invoked_rules
    assert restored.read_cache["/tmp/x.txt"].mtime == 1234.5  # noqa: S108


def _round_trip(entry: ModelContextEvent, tmp_path: Path) -> ModelContextEvent:
    """Write ``entry`` to a fresh session and re-load the first record."""
    history = _round_trip_history([entry], tmp_path)
    assert len(history) == 1
    return history[0]


def _round_trip_history(
    entries: list[ModelContextEvent], tmp_path: Path
) -> list[ModelContextEvent]:
    """Write ``entries`` to a fresh session and return the reloaded history."""
    session_file = tmp_path / "session.jsonl"
    meta = SessionMeta(session_id="abc", model_id="m", provider="P", auth="env")
    append_session(
        session_file,
        meta=meta.serialize(),
        tape_delta=_records_from(entries),
    )
    loaded = load_session(tmp_path)
    assert loaded is not None
    _, tape, _ = loaded
    return _history_from_tape(tape)


def test_user_message_round_trip(tmp_path: Path) -> None:
    out = _round_trip(UserMessage(text="hello"), tmp_path)
    assert isinstance(out, UserMessage)
    assert out.text == "hello"
    assert out.hidden is False


def test_hidden_flag_round_trips(tmp_path: Path) -> None:
    """``hidden`` survives serialization (default ``False``, set ``True``)."""
    visible = _round_trip(UserMessage(text="plain"), tmp_path / "a")
    assert isinstance(visible, UserMessage)
    assert visible.hidden is False
    hidden = _round_trip(UserMessage(text="reminder", hidden=True), tmp_path / "b")
    assert isinstance(hidden, UserMessage)
    assert hidden.hidden is True


def test_agent_send_message_round_trip(tmp_path: Path) -> None:
    out = _round_trip(AgentSendMessage(source="reviewer", text="finding"), tmp_path)
    assert isinstance(out, AgentSendMessage)
    assert out.source == "reviewer"
    assert out.text == "finding"


def test_tool_result_splice_update_persists_through_reload(tmp_path: Path) -> None:
    """Splice updates of an existing ``ToolResult`` must survive session reload.

    Bug repro: when ``_run_tool_and_post`` posts a ``DetachedResult``
    that splices into a ``[detached]`` placeholder, the runtime mutates
    ``history[i]`` in memory. The persistence observer only writes new
    entries (``delta = history[persisted_len:]``), so the splice never
    reaches disk. On session resume the loader reconstructs history
    with the stale ``[detached]`` content, losing the real tool output.

    The fix: the persistence layer must support re-emitting an existing
    entry (same ``id``, updated content). The loader must dedupe by
    ``id`` keeping the LATEST occurrence so the spliced content wins.
    Append-only schema preserved.
    """
    session_file = tmp_path / "session.jsonl"
    meta = SessionMeta(session_id="abc", model_id="m", provider="P", auth="env")

    # Step 1: write a complete tool-use pair (assistant -> [detached]).
    assistant = AssistantMessage(
        text="",
        tool_calls=(ToolCall(id="toolu_bash_1", name="Bash", args={}),),
    )
    original = ToolResult(call_id="toolu_bash_1", content=DETACHED_PLACEHOLDER)
    append_session(
        session_file,
        meta=meta.serialize(),
        tape_delta=_records_from([assistant, original]),
    )

    # Step 2: write an override that suppresses the placeholder and
    # injects the real bash output (simulating the splice).
    spliced = ToolResult(
        id=original.id,
        call_id="toolu_bash_1",
        content="hello world\n",
    )
    placeholder_records = _records_from([assistant, original])
    placeholder_ref = placeholder_records[1].ref
    parent_ref = placeholder_records[0].ref
    splice = ContextSplice(
        ref=TapeRef(session_id="abc", ordinal=2),
        mask=(MaskRange.between(placeholder_ref, placeholder_ref),),
        insert_after=parent_ref,
        payload=(spliced,),
        strategy="detached_splice",
        paired_externally=frozenset({"toolu_bash_1"}),
    )
    append_session(session_file, tape_delta=[splice])

    # Step 3: reload + resolve and assert the spliced content wins.
    loaded = load_session(tmp_path)
    assert loaded is not None
    _, tape, _ = loaded
    messages = resolve_context(tape).messages
    matching = [
        e for e in messages if isinstance(e, ToolResult) and e.call_id == "toolu_bash_1"
    ]
    assert len(matching) == 1
    assert matching[0].content == "hello world\n"


def test_append_session_appends_in_place_without_rewrite(tmp_path: Path) -> None:
    """Each ``append_session`` extends the file in place, keeping its inode.

    The tape is append-only, so persistence must append -- not rewrite via
    tmp + rename. A stable inode across batches proves the write is
    O(bytes appended) (not O(file size)) and that ``tail -f`` / inotify
    watchers keep following the file instead of being orphaned on the
    renamed-away inode.
    """
    session_file = tmp_path / "session.jsonl"
    append_session(session_file, meta={"session_id": "s1"})
    ino_first = session_file.stat().st_ino
    append_session(session_file, meta={"session_id": "s1", "status": "two"})
    ino_second = session_file.stat().st_ino

    assert ino_first == ino_second
    lines = session_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["kind"] == "meta"
    assert json.loads(lines[1])["status"] == "two"


def test_user_message_with_attachment(tmp_path: Path) -> None:
    att = BytesMessage(data=b"\x89PNG", descriptor="image/png")
    out = _round_trip(UserMessage(text="see", attachments=(att,)), tmp_path)
    assert isinstance(out, UserMessage)
    assert out.text == "see"
    assert len(out.attachments) == 1
    assert out.attachments[0].data == b"\x89PNG"
    assert out.attachments[0].descriptor == "image/png"


def test_assistant_message_round_trip(tmp_path: Path) -> None:
    # Pair with a tool_result so the resume-time orphan repair doesn't
    # synthesize an ``[interrupted]`` entry.
    tc = ToolCall(id="c1", name="Echo", args={"msg": "hi"})
    asst = AssistantMessage(text="ok", tool_calls=(tc,))
    res = ToolResult(call_id="c1", content="done")
    history = _round_trip_history([asst, res], tmp_path)
    assert len(history) == 2
    out = history[0]
    assert isinstance(out, AssistantMessage)
    assert out.text == "ok"
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].id == "c1"
    assert out.tool_calls[0].name == "Echo"
    assert dict(out.tool_calls[0].args) == {"msg": "hi"}


def test_assistant_message_thinking_blocks_round_trip(tmp_path: Path) -> None:
    block = {"type": "thinking", "thinking": "step 1"}
    out = _round_trip(AssistantMessage(text="ok", thinking_blocks=(block,)), tmp_path)
    assert isinstance(out, AssistantMessage)
    assert out.thinking_blocks[0]["thinking"] == "step 1"


def test_tool_result_round_trip(tmp_path: Path) -> None:
    # Pair with the matching tool_use so the result isn't dropped as orphan.
    tc = ToolCall(id="c1", name="Echo", args={})
    asst = AssistantMessage(tool_calls=(tc,))
    res = ToolResult(
        call_id="c1", content="ran", diff="--- a\n+++ b\n", hint="hi", summary="1"
    )
    history = _round_trip_history([asst, res], tmp_path)
    assert len(history) == 2
    out = history[1]
    assert isinstance(out, ToolResult)
    assert out.call_id == "c1"
    assert out.content == "ran"
    assert out.diff == "--- a\n+++ b\n"
    # Default kind is FINAL and survives the round-trip.
    assert out.kind is ToolResultKind.FINAL


def test_tool_result_kind_round_trips(tmp_path: Path) -> None:
    """A non-default ``ToolResult.kind`` survives serialize/reload.

    ``kind`` is serialized under ``result_kind`` (the history-record wrapper
    owns the JSON ``kind`` key); a collision would silently drop the field.
    """
    tc = ToolCall(id="c1", name="Echo", args={})
    asst = AssistantMessage(tool_calls=(tc,))
    res = ToolResult(
        call_id="c1",
        content=DETACHED_PLACEHOLDER,
        kind=ToolResultKind.PENDING,
    )
    history = _round_trip_history([asst, res], tmp_path)
    out = history[1]
    assert isinstance(out, ToolResult)
    assert out.kind is ToolResultKind.PENDING


def test_legacy_tool_result_infers_kind_from_content() -> None:
    """A pre-discriminator record (no ``result_kind``) recovers kind by content.

    Sessions persisted before the ``kind`` field carry only ``content``; the
    loader infers ``PENDING`` / ``CANCELLED`` from the placeholder text one last
    time so a resumed old session does not mis-forward a stub.
    """
    pending = _entry_from_json(
        {"type": "tool_result", "call_id": "c1", "content": DETACHED_PLACEHOLDER}
    )
    assert isinstance(pending, ToolResult)
    assert pending.kind is ToolResultKind.PENDING

    cancelled = _entry_from_json(
        {"type": "tool_result", "call_id": "c1", "content": CANCELLED_PLACEHOLDER}
    )
    assert isinstance(cancelled, ToolResult)
    assert cancelled.kind is ToolResultKind.CANCELLED

    final = _entry_from_json(
        {"type": "tool_result", "call_id": "c1", "content": "real output"}
    )
    assert isinstance(final, ToolResult)
    assert final.kind is ToolResultKind.FINAL


def test_legacy_update_recomputes_kind_from_patched_content() -> None:
    """A legacy ``kind=update`` patch must not leave a stale ``PENDING`` kind.

    Regression for ``77bf1d67f`` review C4: a pre-discriminator session can
    back-patch a ``[detached]`` stub (inferred ``PENDING``) to the real result
    via ``kind=update``. The patch rewrote content but left ``kind=PENDING``,
    so the real result would be skipped by the forward path / history lookup.
    The kind is recomputed from the patched content.
    """
    stub = ToolResult(
        call_id="c1",
        content=DETACHED_PLACEHOLDER,
        kind=ToolResultKind.PENDING,
        id=7,
    )
    tape: list[TapeRecord] = [
        ReferrableTapeEvent(ref=TapeRef(session_id="s", ordinal=0), event=stub)
    ]
    _apply_update_in_place(
        tape,
        {"kind": "update", "id": 7, "content": "real output", "is_error": False},
    )
    patched = tape[0]
    assert isinstance(patched, ReferrableTapeEvent)
    assert isinstance(patched.event, ToolResult)
    assert patched.event.content == "real output"
    assert patched.event.kind is ToolResultKind.FINAL


def test_is_barrier_splice_is_session_scoped() -> None:
    """Barrier detection compares full TapeRef identity, not raw ordinal.

    Regression for ``77bf1d67f`` review C7: distinct sessions can share an
    ordinal, so an ordinal-only membership test judged a splice masking only
    ``A:0`` as also masking ``B:0`` -- wrongly classifying a non-barrier as a
    barrier and discarding a valid ``ToolState`` snapshot.
    """
    a0 = ReferrableTapeEvent(
        ref=TapeRef(session_id="A", ordinal=0), event=UserMessage(text="a")
    )
    b0 = ReferrableTapeEvent(
        ref=TapeRef(session_id="B", ordinal=0), event=UserMessage(text="b")
    )
    # A splice that masks only A:0 (not B:0), inserted at head.
    splice = ContextSplice.replay(
        ref=TapeRef(session_id="A", ordinal=1),
        mask=(MaskRange(session_id="A", lo=0, hi=0),),
        insert_after=None,
        payload=(),
        strategy="x",
    )
    # B:0 is an earlier record left unmasked -> not a full barrier.
    assert not _is_barrier_splice(splice, [a0, b0, splice])


def test_load_session_missing_returns_none(tmp_path: Path) -> None:
    assert load_session(tmp_path) is None


def test_append_context_repair_masks_current_view(tmp_path: Path) -> None:
    session_file = tmp_path / "session.jsonl"
    meta = SessionMeta(session_id="abc", model_id="m", provider="P", auth="env")
    old_records = _records_from(
        [
            UserMessage(text="x" * 1_000),
            UserMessage(text="y" * 1_000),
        ]
    )
    state = ToolState()
    state.invoked_skills.add("debug")
    append_session(
        session_file,
        meta=meta.serialize(),
        tape_delta=old_records,
        tool_state_snapshot=serialize_tool_state(state),
    )

    loaded = load_session(tmp_path)
    assert loaded is not None
    _, tape, _ = loaded
    assert (
        sum(
            len(m.text)
            for m in resolve_context(tape).messages
            if isinstance(m, UserMessage)
        )
        == 2_000
    )

    repair = append_context_repair(
        session_file,
        tape,
        payload=[UserMessage(text="slim repair")],
        strategy="manual_repair",
    )

    loaded_after = load_session(tmp_path)
    assert loaded_after is not None
    _, repaired_tape, repaired_state = loaded_after
    messages = resolve_context(repaired_tape).messages
    assert repair.strategy == "manual_repair"
    assert [m.text for m in messages if isinstance(m, UserMessage)] == ["slim repair"]
    assert repaired_state.invoked_skills == set()
    validate_context(messages)


def test_clear_barrier_drops_prior_history(tmp_path: Path) -> None:
    session_file = tmp_path / "session.jsonl"
    meta = SessionMeta(session_id="abc", model_id="m", provider="P", auth="env")
    # Explicit refs to avoid collision between two ``_records_from`` calls.
    old_ref = TapeRef(session_id="abc", ordinal=0)
    splice_ref = TapeRef(session_id="abc", ordinal=1)
    new_ref = TapeRef(session_id="abc", ordinal=2)
    append_session(
        session_file,
        meta=meta.serialize(),
        tape_delta=[ReferrableTapeEvent(ref=old_ref, event=UserMessage(text="old"))],
    )
    append_session(
        session_file,
        tape_delta=[
            ContextSplice(
                ref=splice_ref,
                mask=(MaskRange.between(old_ref, old_ref),),
                insert_after=None,
                payload=(),
                strategy="clear",
            )
        ],
    )
    append_session(
        session_file,
        tape_delta=[ReferrableTapeEvent(ref=new_ref, event=UserMessage(text="new"))],
    )
    loaded = load_session(tmp_path)
    assert loaded is not None
    _, tape, _ = loaded
    history = _history_from_tape(tape)
    texts = [e.text for e in history if isinstance(e, UserMessage)]
    assert texts == ["new"]


def test_meta_latest_wins(tmp_path: Path) -> None:
    session_file = tmp_path / "session.jsonl"
    append_session(
        session_file,
        meta=SessionMeta(session_id="old", model_id="m").serialize(),
    )
    append_session(
        session_file,
        meta=SessionMeta(session_id="new", model_id="m").serialize(),
    )
    loaded = load_session(tmp_path)
    assert loaded is not None
    meta, _, _ = loaded
    assert meta.session_id == "new"


def test_tool_state_post_clear_wins(tmp_path: Path) -> None:
    session_file = tmp_path / "session.jsonl"
    s1 = ToolState()
    s1.bash_cwd = "/old"
    s2 = ToolState()
    s2.bash_cwd = "/new"
    append_session(session_file, tool_state_snapshot=serialize_tool_state(s1))
    append_session(
        session_file,
        tape_delta=[
            ContextSplice(
                ref=TapeRef(session_id="abc", ordinal=999),
                mask=(),
                insert_after=None,
                payload=(),
                strategy="clear",
            )
        ],
    )
    append_session(session_file, tool_state_snapshot=serialize_tool_state(s2))
    loaded = load_session(tmp_path)
    assert loaded is not None
    _, _, state = loaded
    assert state.bash_cwd == "/new"


def test_append_session_no_ops_on_empty_batch(tmp_path: Path) -> None:
    session_file = tmp_path / "session.jsonl"
    append_session(session_file)
    assert not session_file.exists()


def test_append_session_crash_writing_oversized_splice_preserves_prior_state(
    tmp_path: Path,
) -> None:
    """Splice records routinely exceed Linux PIPE_BUF (4096); a crash mid-write
    over such a record must not corrupt the prior committed state.

    The retired comment in ``append_session`` claimed line-level kernel
    atomicity via ``O_APPEND`` for "all current record types", but a
    ``ContextSplice`` with a 16 KiB payload entry already serializes
    well past the 4096 ceiling. The atomic-replace path makes the
    failure mode batch-level, regardless of line length.
    """
    session_file = tmp_path / "session.jsonl"
    prior = SessionMeta(session_id="abc", model_id="m", provider="P", auth="env")
    append_session(session_file, meta=prior.serialize())
    prior_bytes = session_file.read_bytes()

    splice = ContextSplice(
        ref=TapeRef(session_id="abc", ordinal=0),
        mask=(),
        insert_after=None,
        payload=(UserMessage(text="x" * 16_384),),
        strategy="summary",
    )
    sizing_path = tmp_path / "size.jsonl"
    append_session(sizing_path, tape_delta=[splice])
    splice_line_bytes = len(sizing_path.read_bytes().splitlines()[0])
    assert splice_line_bytes > 4_096

    def _explode(fd: int, data: bytes) -> int:
        del fd, data
        raise OSError("simulated crash on oversized record")

    with (
        patch("sagent.agent.session_io.os.write", _explode),
        pytest.raises(OSError, match="simulated crash"),
    ):
        append_session(session_file, tape_delta=[splice])

    assert session_file.read_bytes() == prior_bytes


def test_append_session_crash_during_write_preserves_prior_state(
    tmp_path: Path,
) -> None:
    """A crash mid-append leaves prior records intact plus a torn tail.

    Append-in-place never rewrites or loses prior records, so a failure
    partway through the new batch leaves the prior bytes verbatim with a
    truncated (non-newline-terminated) trailing line. The loader skips
    that torn tail and recovers the prior committed state -- the new
    batch is simply absent, as if the save had never happened. (The
    stronger byte-identical-on-crash guarantee belonged to the rejected
    rewrite-to-tmp + rename approach; append trades it for O(delta)
    writes and a stable inode that ``tail -f`` can follow.)
    """
    session_file = tmp_path / "session.jsonl"
    prior_meta = SessionMeta(session_id="abc", model_id="m", provider="P", auth="env")
    append_session(session_file, meta=prior_meta.serialize())
    prior_bytes = session_file.read_bytes()

    real_write = os.write
    fail_after = [1]

    def _explode(fd: int, data: bytes) -> int:
        if fail_after[0] <= 0:
            raise OSError("simulated crash")
        fail_after[0] -= 1
        return real_write(fd, data[:1])

    new_meta = SessionMeta(
        session_id="abc", model_id="m2", provider="P", auth="env", status="failed"
    )
    with (
        patch("sagent.agent.session_io.os.write", _explode),
        pytest.raises(OSError, match="simulated crash"),
    ):
        append_session(
            session_file,
            meta=new_meta.serialize(),
            tape_delta=_records_from([UserMessage(text="y" * 2_000)]),
        )

    # Prior records are byte-for-byte intact; only an unterminated tail
    # was appended, and the loader recovers the prior committed state.
    assert session_file.read_bytes().startswith(prior_bytes)
    loaded = load_session(tmp_path)
    assert loaded is not None
    meta, tape, _ = loaded
    assert meta.model_id == "m"
    assert meta.status != "failed"
    assert tape == []


def test_load_session_orders_loaded_tape_by_ordinal(tmp_path: Path) -> None:
    """Persisted and synthetic refs load in canonical ordinal order."""
    session_file = tmp_path / "session.jsonl"
    _write_jsonl(
        session_file,
        {"kind": "meta", "session_id": "abc"},
        {
            "kind": "history",
            "ref": {"session_id": "abc", "ordinal": 1},
            "type": "user",
            "text": "one",
        },
        {"kind": "history", "type": "user", "text": "synthetic"},
        {
            "kind": "history",
            "ref": {"session_id": "abc", "ordinal": 0},
            "type": "user",
            "text": "zero",
        },
    )

    loaded = load_session(tmp_path)
    assert loaded is not None
    _, tape, _ = loaded
    assert [record.ref.ordinal for record in tape] == [0, 1, 2]
    user_messages: list[UserMessage] = []
    for entry in _history_from_tape(tape):
        assert isinstance(entry, UserMessage)
        user_messages.append(entry)
    assert [entry.text for entry in user_messages] == ["zero", "one", "synthetic"]


def test_legacy_override_with_gap_preserves_unmasked_ref(tmp_path: Path) -> None:
    session_file = tmp_path / "session.jsonl"
    refs = [TapeRef(session_id="abc", ordinal=i) for i in range(4)]
    _write_jsonl(
        session_file,
        {"kind": "meta", "session_id": "abc"},
        {
            "kind": "history",
            "ref": {"session_id": "abc", "ordinal": 0},
            "type": "user",
            "text": "first",
        },
        {
            "kind": "history",
            "ref": {"session_id": "abc", "ordinal": 1},
            "type": "user",
            "text": "middle",
        },
        {
            "kind": "history",
            "ref": {"session_id": "abc", "ordinal": 2},
            "type": "user",
            "text": "last",
        },
        {
            "kind": "context_override",
            "ref": {"session_id": "abc", "ordinal": 3},
            "suppresses": [
                {"session_id": "abc", "ordinal": 0},
                {"session_id": "abc", "ordinal": 2},
            ],
            "inject_after": None,
            "payload": [{"type": "user", "text": "replacement"}],
        },
    )

    loaded = load_session(tmp_path)

    assert loaded is not None
    _, tape, _ = loaded
    messages = resolve_context(tape).messages
    assert [entry.text for entry in messages if isinstance(entry, UserMessage)] == [
        "replacement",
        "middle",
    ]
    splice = tape[-1]
    assert isinstance(splice, ContextSplice)
    assert splice.mask == (
        MaskRange.between(refs[0], refs[0]),
        MaskRange.between(refs[2], refs[2]),
    )


@pytest.mark.asyncio
async def test_persistence_skips_externally_replayed_records(tmp_path: Path) -> None:
    session_file = tmp_path / "session.jsonl"
    persisted = ReferrableTapeEvent(
        ref=TapeRef(session_id="abc", ordinal=0),
        event=UserMessage(text="persisted"),
    )
    append_session(session_file, tape_delta=[persisted])
    agent = Agent(model=_NoopModel(), session_dir=tmp_path)
    agent.runtime.replay_tape([persisted])

    agent.runtime.publish(SaveSession())

    lines = session_file.read_text(encoding="utf-8").splitlines()
    texts = [json.loads(line)["text"] for line in lines if '"kind": "history"' in line]
    assert texts.count("persisted") == 1


@pytest.mark.asyncio
async def test_save_session_skips_tool_state_when_unchanged(tmp_path: Path) -> None:
    agent = Agent(model=_NoopModel(), session_dir=tmp_path)
    agent.runtime.append_history(UserMessage(text="x"))
    agent.runtime.publish(SaveSession())
    session_file = tmp_path / "session.jsonl"
    size = session_file.stat().st_size

    for _ in range(5):
        agent.runtime.publish(SaveSession())

    assert session_file.stat().st_size == size


@pytest.mark.asyncio
async def test_save_session_writes_tool_state_when_changed(tmp_path: Path) -> None:
    agent = Agent(model=_NoopModel(), session_dir=tmp_path)
    agent.runtime.publish(SaveSession())
    lines_before = (tmp_path / "session.jsonl").read_text(encoding="utf-8").splitlines()

    agent.tool_state.bash_cwd = "/changed"
    agent.runtime.publish(SaveSession())

    lines_after = (tmp_path / "session.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines_after) > len(lines_before)
    assert json.loads(lines_after[-1])["bash_cwd"] == "/changed"


@pytest.mark.asyncio
async def test_load_session_with_repair_preserves_meta_bash_cwd(
    tmp_path: Path,
) -> None:
    session_file = tmp_path / "session.jsonl"
    meta = SessionMeta(
        session_id="dangling",
        model_id="m",
        provider="P",
        auth="env",
        bash_cwd="/project",
    )
    append_session(
        session_file,
        meta=meta.serialize(),
        tape_delta=[
            ReferrableTapeEvent(
                ref=TapeRef(session_id="dangling", ordinal=0),
                event=AssistantMessage(
                    tool_calls=(ToolCall(id="call_1", name="Bash", args={}),)
                ),
            ),
        ],
    )

    loaded = load_session(tmp_path)

    assert loaded is not None
    _, tape, state = loaded
    validate_context(resolve_context(tape).messages)
    assert state.bash_cwd == "/project"


def test_out_of_order_barrier_resets_prior_tool_state(tmp_path: Path) -> None:
    session_file = tmp_path / "session.jsonl"
    old_state = ToolState()
    old_state.invoked_skills.add("debug")
    _write_jsonl(
        session_file,
        {"kind": "meta", "session_id": "abc"},
        {"kind": "tool_state", **serialize_tool_state(old_state)},
        {
            "kind": "context_splice",
            "ref": {"session_id": "abc", "ordinal": 2},
            "mask": [
                [
                    {"session_id": "abc", "ordinal": 0},
                    {"session_id": "abc", "ordinal": 1},
                ]
            ],
            "insert_after": None,
            "payload": [{"type": "user", "text": "after"}],
            "strategy": "summary",
            "paired_externally": [],
        },
        {
            "kind": "history",
            "ref": {"session_id": "abc", "ordinal": 0},
            "type": "user",
            "text": "before",
        },
    )

    loaded = load_session(tmp_path)
    assert loaded is not None
    _, _, state = loaded
    assert state.invoked_skills == set()


def test_load_session_dangling_repair_resets_prior_tool_state(tmp_path: Path) -> None:
    session_file = tmp_path / "session.jsonl"
    meta = SessionMeta(session_id="dangling", model_id="m", provider="P", auth="env")
    state = ToolState()
    state.invoked_skills.add("debug")
    state.bash_cwd = "/stale"
    append_session(
        session_file,
        meta=meta.serialize(),
        tape_delta=[
            ReferrableTapeEvent(
                ref=TapeRef(session_id="dangling", ordinal=0),
                event=AssistantMessage(
                    tool_calls=(ToolCall(id="call_1", name="Bash", args={}),)
                ),
            ),
        ],
        tool_state_snapshot=serialize_tool_state(state),
    )

    loaded = load_session(tmp_path)
    assert loaded is not None
    _, tape, repaired_state = loaded
    validate_context(resolve_context(tape).messages)
    assert repaired_state.invoked_skills == set()
    assert repaired_state.bash_cwd == ToolState().bash_cwd


def test_load_session_repairs_orphan_tool_result(tmp_path: Path) -> None:
    """Disk-loaded ``ToolResult`` with no matching ``tool_use`` is hidden on resume.

    Regression: a session that crashed mid-tool or that imported a
    foreign history could persist a ``ToolResult`` whose ``call_id``
    has no preceding ``AssistantMessage.tool_calls`` match.
    ``repair_dangling_tool_calls`` drops the orphan from its returned
    list, but the loaded tape still contained the orphan as a
    ``ReferrableTapeEvent``; the resolved context surfaced it and the next
    provider call rejected the request with 400. The load-time tape
    repair must append a suppression override so the resolved view
    is wire-format-valid out of the box.
    """
    session_file = tmp_path / "session.jsonl"
    meta = SessionMeta(session_id="orphan", model_id="m", provider="P", auth="env")
    # Persist: user -> orphan ToolResult -> user.
    user1 = UserMessage(text="kick off")
    orphan = ToolResult(call_id="ghost_1", content="dangling")
    user2 = UserMessage(text="continue")
    append_session(
        session_file,
        meta=meta.serialize(),
        tape_delta=[
            ReferrableTapeEvent(
                ref=TapeRef(session_id="orphan", ordinal=0), event=user1
            ),
            ReferrableTapeEvent(
                ref=TapeRef(session_id="orphan", ordinal=1), event=orphan
            ),
            ReferrableTapeEvent(
                ref=TapeRef(session_id="orphan", ordinal=2), event=user2
            ),
        ],
    )
    loaded = load_session(tmp_path)
    assert loaded is not None
    _, tape, _ = loaded
    resolved = resolve_context(tape).messages
    # The orphan must not surface in the resolved view; user messages survive.
    assert not any(
        isinstance(m, ToolResult) and m.call_id == "ghost_1" for m in resolved
    ), f"orphan ToolResult should be suppressed on load; resolved: {resolved}"
    users = [m for m in resolved if isinstance(m, UserMessage)]
    assert len(users) == 1
    assert users[0].text == f"{user1.text}\n\n{user2.text}"


def test_appending_after_a_torn_tail_does_not_destroy_the_next_record(
    tmp_path: Path,
) -> None:
    """A crash truncates one record; it must not also eat the recovery.

    ``_append_lines`` opens ``O_APPEND`` and writes, so a file whose last line
    lost its newline gets the next record concatenated onto it. Both lines then
    fail to parse and BOTH are skipped -- the crash costs the record it
    interrupted plus the first one written after recovery, which is the one
    saying what the user did next.
    """
    session_file = tmp_path / "session.jsonl"
    append_session(
        session_file,
        meta=SessionMeta(session_id="torn", model_id="m").serialize(),
    )
    with session_file.open("a", encoding="utf-8") as handle:
        _ = handle.write('{"kind": "history", "ref": {"session_id": "torn", "ord')

    append_session(
        session_file,
        tape_delta=[
            ReferrableTapeEvent(
                ref=TapeRef(session_id="torn", ordinal=1),
                event=UserMessage(text="survivor"),
            ),
        ],
    )

    loaded = load_session(tmp_path)
    assert loaded is not None
    texts = [getattr(m, "text", "") for m in resolve_context(loaded[1]).messages]
    assert "survivor" in texts, f"record after a torn tail was lost; got {texts}"


def test_load_preserves_append_order_across_sessions(tmp_path: Path) -> None:
    """Tape order is append order; a session-id sort is not a tie-break.

    The resolver anchors a splice against the records emitted BEFORE it, so
    ordering by ``session_id`` first hoists one session's whole run ahead of
    another's and an anchor that has not been emitted yet falls into HEAD --
    silently reversing the conversation. Resumed and forked tapes carry two
    session ids by design, and one real session on disk reorders this way.
    """
    session_file = tmp_path / "session.jsonl"
    b0 = TapeRef(session_id="B", ordinal=0)
    append_session(
        session_file,
        meta=SessionMeta(session_id="A", model_id="m").serialize(),
        tape_delta=[
            ReferrableTapeEvent(ref=b0, event=UserMessage(text="asked")),
            ContextSplice(
                ref=TapeRef(session_id="A", ordinal=1),
                mask=(),
                insert_after=b0,
                payload=(AssistantMessage(text="answered"),),
                strategy="probe",
            ),
        ],
    )

    loaded = load_session(tmp_path)
    assert loaded is not None
    texts = [getattr(m, "text", "") for m in resolve_context(loaded[1]).messages]
    assert texts == ["asked", "answered"], f"load reordered the tape; got {texts}"


def test_a_json_string_is_not_a_persisted_barrier_flag(tmp_path: Path) -> None:
    """``bool("false")`` is True, so a typed-wrong record inverted its flag.

    A legacy override carrying ``"barrier": "false"`` was read as a barrier
    and masked every message ahead of it -- the conversation disappeared
    because some writer stringified a boolean. Only a real JSON boolean sets a
    persisted flag; anything else is malformed and takes the default.
    """
    session_file = tmp_path / "session.jsonl"
    records: tuple[Mapping[str, object], ...] = (
        {"kind": "meta", "session_id": "flag", "model_id": "m"},
        {
            "kind": "history",
            "ref": {"session_id": "flag", "ordinal": 0},
            "type": "user",
            "text": "keep me",
        },
        {
            "kind": "context_override",
            "ref": {"session_id": "flag", "ordinal": 1},
            "suppresses": [],
            "inject_after": None,
            "barrier": "false",
            "payload": [],
        },
    )
    session_file.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    loaded = load_session(tmp_path)
    assert loaded is not None
    texts = [getattr(m, "text", "") for m in resolve_context(loaded[1]).messages]
    assert texts == ["keep me"], f"a stringified false masked history; got {texts}"


def test_a_boolean_is_not_a_number_at_the_disk_boundary() -> None:
    """``isinstance(True, int)`` holds, so a JSON ``true`` became a real cap.

    A persisted ``max_tool_call_rounds: true`` decoded to ``True``, which is
    ``1`` everywhere downstream -- a one-round budget the operator never set;
    ``max_budget_usd: true`` became a $1 ceiling. ``_ref_from_json`` already
    rejects this exact trap, so the sibling decoders disagreed about the same
    wire value at the same boundary.
    """
    assert _optional_int(True) is None
    assert _optional_float(True) is None
    assert _optional_int(3) == 3
    assert _optional_float(2.5) == 2.5


def test_an_attachment_descriptor_is_matched_exactly_not_by_prefix() -> None:
    """``application/pdf`` is a complete type, not the head of a family.

    Prefix matching admitted ``application/pdf-malware`` and
    ``application/jsonevil`` -- descriptors the allowlist exists to exclude --
    while the genuine families (``image/``, ``text/``) do end in a separator
    and remain prefixes.
    """
    assert _att_from_json({"mime": "application/pdf-malware", "data": "aA=="}) is None
    assert _att_from_json({"mime": "application/jsonevil", "data": "aA=="}) is None
    assert _att_from_json({"mime": "application/pdf", "data": "aA=="}) is not None
    assert _att_from_json({"mime": "image/png", "data": "aA=="}) is not None


def test_an_unknown_result_kind_infers_from_content_not_final() -> None:
    """An unreadable lifecycle must not default to "this is the real answer".

    ``FINAL`` is the forward-deliverable, terminal state, so a typo in the
    persisted enum promoted a still-pending stub to a real result -- the
    ``[detached]`` placeholder would be forwarded to the model as output. The
    content-inference path that already exists for pre-discriminator records
    is the safe reading; use it whenever the field cannot be honoured.
    """
    assert (
        _tool_result_kind_from_json("typo", DETACHED_PLACEHOLDER)
        is ToolResultKind.PENDING
    )
    assert (
        _tool_result_kind_from_json("typo", CANCELLED_PLACEHOLDER)
        is ToolResultKind.CANCELLED
    )
    assert _tool_result_kind_from_json("pending", "x") is ToolResultKind.PENDING


def test_a_tool_call_without_an_id_or_name_is_dropped() -> None:
    """A call the runtime cannot key or dispatch is not a call.

    Missing fields decoded to ``""``, and the runtime keys ``running_tools``,
    the cohort, and every pairing map by ``call_id`` -- so a single empty id
    is dispatchable under ``""`` and collides with the next one. Drop the
    malformed call at the boundary; the surrounding message still loads.
    """
    record: Mapping[str, object] = {
        "type": "assistant",
        "text": "hi",
        "tool_calls": [
            {"id": "", "name": "Bash", "args": cast("Mapping[str, object]", {})},
            {"id": "t1", "name": "", "args": cast("Mapping[str, object]", {})},
            {"id": "t2", "name": "Bash", "args": cast("Mapping[str, object]", {})},
        ],
    }
    entry = _entry_from_json(record)
    assert isinstance(entry, AssistantMessage)
    assert [tc.id for tc in entry.tool_calls] == ["t2"]


def test_a_malformed_base64_attachment_is_dropped() -> None:
    """The drop path must actually fire for garbage, not silently empty it.

    ``b64decode`` without ``validate=True`` discards non-alphabet bytes rather
    than raising, so ``"!!!!"`` decodes to ``b""`` and the ``except`` below it
    never runs. An attachment that survives as zero bytes is worse than one
    that is dropped: it reaches the provider as a real, empty image.
    """
    assert _att_from_json({"mime": "image/png", "data": "!!!!"}) is None, (
        "malformed base64 must be dropped, not decoded to empty bytes"
    )


def test_a_tape_ref_rejects_a_non_position_ordinal() -> None:
    """``ordinal`` is a 0-based position, and the disk is a trust boundary.

    ``isinstance(True, int)`` holds, so a JSON ``true`` became ordinal 1 and
    collided with a real record; a negative ordinal is unmaskable, since
    ``MaskRange`` rejects ``lo < 0``. The sibling types disagreed about the
    same field at the same boundary.
    """
    assert _ref_from_json({"session_id": "s", "ordinal": True}) is None
    assert _ref_from_json({"session_id": "s", "ordinal": -1}) is None
    assert _ref_from_json({"session_id": "s", "ordinal": 3}) == TapeRef(
        session_id="s", ordinal=3
    )


def test_a_relocated_record_is_not_written_back_to_disk(tmp_path: Path) -> None:
    """Resume must not re-append records the load merely relocated.

    The persistence observer decides what is new by comparing tape refs
    against the refs it read off disk, and a record relocated by
    ``_renumber_duplicate_refs`` carries a ref that appears in neither -- so
    it would look new and be appended again, recreating on every save the
    duplicate the load had just resolved. ``resume``'s rebaseline is what
    prevents that, by seeding the cursor from the POST-load tape rather than
    from the file. Pinned here because the relocation and the rebaseline are
    in different modules and nothing else ties them together.
    """
    session_file = tmp_path / "session.jsonl"
    collision = TapeRef(session_id="respawn", ordinal=0)
    append_session(
        session_file,
        meta=SessionMeta(session_id="respawn", model_id="m").serialize(),
        tape_delta=[
            ReferrableTapeEvent(ref=collision, event=UserMessage(text="first")),
            ReferrableTapeEvent(ref=collision, event=UserMessage(text="second")),
        ],
    )
    loaded = load_session(tmp_path)
    assert loaded is not None
    meta, tape, _state = loaded

    agent = Agent(model=_NoopModel(), tools=[], session_dir=tmp_path)
    agent.resume(meta, tape, ToolState())
    agent.runtime.publish(SaveSession())

    reloaded = load_session(tmp_path)
    assert reloaded is not None
    assert len(reloaded[1]) == len(tape), (
        f"resume re-appended {len(reloaded[1]) - len(tape)} record(s)"
    )


def test_one_malformed_record_does_not_abort_the_resume(tmp_path: Path) -> None:
    """A single bad record costs that record, not the whole conversation.

    ``_entry_from_json`` builds an ``AssistantMessage`` directly, and its
    ``__post_init__`` raises on a duplicate ``tool_calls`` id. ``load_session``
    catches only ``JSONDecodeError`` and ``OSError``, so a semantically
    malformed line propagated out and the session became unresumable -- every
    other record in the file lost with it. The loader's whole posture is
    legacy repair; it must degrade per record.
    """
    session_file = tmp_path / "session.jsonl"
    records: tuple[Mapping[str, object], ...] = (
        {"kind": "meta", "session_id": "dup", "model_id": "m"},
        {
            "kind": "history",
            "ref": {"session_id": "dup", "ordinal": 0},
            "type": "user",
            "text": "keep me",
        },
        {
            "kind": "history",
            "ref": {"session_id": "dup", "ordinal": 1},
            "type": "assistant",
            "text": "",
            "tool_calls": [
                {"id": "t1", "name": "Bash", "args": cast("Mapping[str, object]", {})},
                {"id": "t1", "name": "Bash", "args": cast("Mapping[str, object]", {})},
            ],
        },
        {
            "kind": "history",
            "ref": {"session_id": "dup", "ordinal": 2},
            "type": "user",
            "text": "and me",
        },
    )
    session_file.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    loaded = load_session(tmp_path)
    assert loaded is not None, "one malformed record made the session unloadable"
    texts = [getattr(m, "text", "") for m in resolve_context(loaded[1]).messages]
    assert "keep me" in " ".join(texts)
    assert "and me" in " ".join(texts)


def test_a_legacy_clear_masks_the_highest_ordinal_not_the_last_appended(
    tmp_path: Path,
) -> None:
    """A clear wipes the whole prefix, so it must mask every earlier ordinal.

    The mask ended at ``tape[-1].ref.ordinal`` -- the last record READ, taken
    before the load sorts by ordinal. A file whose records are out of order
    therefore left the highest-ordinal record visible past a clear.
    """
    session_file = tmp_path / "session.jsonl"
    records: tuple[Mapping[str, object], ...] = (
        {"kind": "meta", "session_id": "s", "model_id": "m"},
        {
            "kind": "history",
            "ref": {"session_id": "s", "ordinal": 5},
            "type": "user",
            "text": "high-ordinal",
        },
        {
            "kind": "history",
            "ref": {"session_id": "s", "ordinal": 1},
            "type": "user",
            "text": "low-ordinal",
        },
        {"kind": "clear"},
    )
    session_file.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    loaded = load_session(tmp_path)
    assert loaded is not None
    texts = [getattr(m, "text", "") for m in resolve_context(loaded[1]).messages]
    assert texts == [], f"records survived a legacy clear; got {texts}"


def test_a_legacy_clear_masks_every_session_on_the_tape(tmp_path: Path) -> None:
    """A clear wipes the view, and the view spans every session on the tape.

    The mask was built in the clear's own ``session_id`` only, so on a resumed
    or forked tape -- the shape ``_sort_tape_by_ordinal`` exists for -- records
    from the other session stayed visible after the user asked for a wipe.
    """
    session_file = tmp_path / "session.jsonl"
    records: tuple[Mapping[str, object], ...] = (
        {"kind": "meta", "session_id": "B", "model_id": "m"},
        {
            "kind": "history",
            "ref": {"session_id": "A", "ordinal": 0},
            "type": "user",
            "text": "from-session-A",
        },
        {
            "kind": "history",
            "ref": {"session_id": "B", "ordinal": 1},
            "type": "user",
            "text": "from-session-B",
        },
        {"kind": "context_clear", "ref": {"session_id": "B", "ordinal": 2}},
    )
    session_file.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    loaded = load_session(tmp_path)
    assert loaded is not None
    texts = [getattr(m, "text", "") for m in resolve_context(loaded[1]).messages]
    assert texts == [], f"a foreign session survived a legacy clear; got {texts}"


def test_a_relocated_record_does_not_land_inside_an_existing_mask(
    tmp_path: Path,
) -> None:
    """Relocation must not move a record into a mask that never covered it.

    A duplicate moves past the tape's high-water mark, but a mask can already
    claim ordinals ABOVE every record -- a barrier written when the tape was
    longer, or one whose range was widened. The relocated record lands inside
    it and disappears, even though it was written AFTER that barrier and so
    was never something that barrier meant to hide.
    """
    session_file = tmp_path / "session.jsonl"
    sid = "wide"
    records: tuple[Mapping[str, object], ...] = (
        {"kind": "meta", "session_id": sid, "model_id": "m"},
        {
            "kind": "history",
            "ref": {"session_id": sid, "ordinal": 0},
            "type": "user",
            "text": "a",
        },
        {
            "kind": "context_splice",
            "ref": {"session_id": sid, "ordinal": 1},
            "mask": [
                [
                    {"session_id": sid, "ordinal": 0},
                    {"session_id": sid, "ordinal": 99},
                ]
            ],
            "insert_after": None,
            "payload": [{"type": "user", "text": "barrier"}],
            "strategy": "barrier",
        },
        {
            "kind": "history",
            "ref": {"session_id": sid, "ordinal": 0},
            "type": "user",
            "text": "later-real",
        },
    )
    session_file.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    loaded = load_session(tmp_path)
    assert loaded is not None
    texts = [getattr(m, "text", "") for m in resolve_context(loaded[1]).messages]
    assert "later-real" in texts, f"relocation hid a post-barrier record; {texts}"


def test_a_carried_mask_does_not_overlap_the_range_it_extends(
    tmp_path: Path,
) -> None:
    """Carrying a mask onto a moved record must not duplicate coverage.

    The carried singleton is appended to the splice's existing ranges without
    merging, and ``ContextSplice`` rejects overlapping ranges. The raise comes
    out of ``dataclasses.replace`` AFTER the read loop, outside the per-record
    catch -- so one duplicated ref made the whole session unloadable.
    """
    session_file = tmp_path / "session.jsonl"
    sid = "overlap"
    records: tuple[Mapping[str, object], ...] = (
        {"kind": "meta", "session_id": sid, "model_id": "m"},
        {
            "kind": "history",
            "ref": {"session_id": sid, "ordinal": 0},
            "type": "user",
            "text": "a",
        },
        {
            "kind": "history",
            "ref": {"session_id": sid, "ordinal": 0},
            "type": "user",
            "text": "a-dup",
        },
        {
            "kind": "context_splice",
            "ref": {"session_id": sid, "ordinal": 1},
            "mask": [
                [
                    {"session_id": sid, "ordinal": 0},
                    {"session_id": sid, "ordinal": 10},
                ]
            ],
            "insert_after": None,
            "payload": [],
            "strategy": "wide",
        },
    )
    session_file.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    loaded = load_session(tmp_path)
    assert loaded is not None, "an overlapping carried mask made the session unloadable"


def test_a_duplicated_dead_splice_is_not_resurrected(tmp_path: Path) -> None:
    """Re-numbering must not move a splice out from under what killed it.

    A splice is dead iff its ref falls inside an alive splice's mask, so
    moving that ref past the high-water mark revives the edit and the content
    it hid disappears again. A re-appended edit is not a second edit: drop it.
    """
    session_file = tmp_path / "session.jsonl"
    sid = "resurrect"
    poison = TapeRef(session_id=sid, ordinal=1)
    append_session(
        session_file,
        meta=SessionMeta(session_id=sid, model_id="m").serialize(),
        tape_delta=[
            ReferrableTapeEvent(
                ref=TapeRef(session_id=sid, ordinal=0), event=UserMessage(text="real")
            ),
            ContextSplice(
                ref=poison,
                mask=(MaskRange(session_id=sid, lo=0, hi=0),),
                insert_after=None,
                payload=(UserMessage(text="poison"),),
                strategy="poison",
            ),
            ContextSplice(
                ref=TapeRef(session_id=sid, ordinal=2),
                mask=(MaskRange(session_id=sid, lo=1, hi=1),),
                insert_after=None,
                payload=(),
                strategy="killer",
            ),
            ContextSplice.replay(
                ref=poison,
                mask=(MaskRange(session_id=sid, lo=0, hi=0),),
                insert_after=None,
                payload=(UserMessage(text="poison-dup"),),
                strategy="poison",
            ),
        ],
    )

    loaded = load_session(tmp_path)
    assert loaded is not None
    texts = [getattr(m, "text", "") for m in resolve_context(loaded[1]).messages]
    assert texts == ["real"], f"dead splice resurrected; got {texts}"


def test_a_second_writers_splice_is_relocated_not_dropped(tmp_path: Path) -> None:
    """A live duplicate splice is new content, not a re-appended edit.

    Two agents resumed from one session directory both mint from an in-memory
    ordinal cursor seeded at load, so each claims the same next position. When
    the second one's record is a ``ContextSplice`` -- which it is for every
    user-message coalesce -- dropping it as a re-append discards the payload
    outright, and the user's message is gone with nothing on disk to recover
    it.

    The discriminator is the POSITION's fate, matching the rule this function
    already applies to a plain record: a duplicate landing on a ref some alive
    splice masks is dead, and relocating it would revive an edit (see
    ``test_a_duplicated_dead_splice_is_not_resurrected``). A duplicate landing
    on a live ref is a second writer's real work and must survive.
    """
    session_file = tmp_path / "session.jsonl"
    sid = "concurrent"
    collision = TapeRef(session_id=sid, ordinal=1)
    append_session(
        session_file,
        meta=SessionMeta(session_id=sid, model_id="m").serialize(),
        tape_delta=[
            ReferrableTapeEvent(
                ref=TapeRef(session_id=sid, ordinal=0),
                event=UserMessage(text="shared-history"),
            ),
            ReferrableTapeEvent(ref=collision, event=UserMessage(text="said-to-A")),
            ContextSplice.replay(
                ref=collision,
                mask=(),
                insert_after=None,
                payload=(UserMessage(text="said-to-B"),),
                strategy="user_coalesce",
            ),
        ],
    )

    loaded = load_session(tmp_path)
    assert loaded is not None
    texts = [getattr(m, "text", "") for m in resolve_context(loaded[1]).messages]
    assert "said-to-B" in texts, f"second writer's message was dropped; got {texts}"
    assert "said-to-A" in texts, f"first writer's message was dropped; got {texts}"


def test_a_duplicate_written_before_a_barrier_stays_masked(tmp_path: Path) -> None:
    """A moved record keeps the fate of the position it was written at.

    The barrier masked that ordinal, and this claimant already existed when
    the barrier landed -- so the barrier meant to hide it. Re-numbering it to
    the tail without carrying that mask makes it reappear alongside the
    summary that replaced it.
    """
    session_file = tmp_path / "session.jsonl"
    sid = "premask"
    collision = TapeRef(session_id=sid, ordinal=0)
    append_session(
        session_file,
        meta=SessionMeta(session_id=sid, model_id="m").serialize(),
        tape_delta=[
            ReferrableTapeEvent(ref=collision, event=UserMessage(text="a")),
            ReferrableTapeEvent(ref=collision, event=UserMessage(text="a-dup")),
            ContextSplice(
                ref=TapeRef(session_id=sid, ordinal=2),
                mask=(MaskRange(session_id=sid, lo=0, hi=1),),
                insert_after=None,
                payload=(UserMessage(text="barrier"),),
                strategy="barrier",
            ),
        ],
    )

    loaded = load_session(tmp_path)
    assert loaded is not None
    texts = [getattr(m, "text", "") for m in resolve_context(loaded[1]).messages]
    assert texts == ["barrier"], f"pre-barrier duplicate escaped its mask; {texts}"


def test_a_duplicate_written_after_a_barrier_survives_it(tmp_path: Path) -> None:
    """A barrier cannot mask a record that did not exist when it was written.

    The mirror of the pre-barrier case, and why one policy cannot serve both:
    this claimant was appended by a second writer AFTER the barrier, so
    carrying the barrier's mask onto it would delete a message the user
    actually received.
    """
    session_file = tmp_path / "session.jsonl"
    sid = "postmask"
    collision = TapeRef(session_id=sid, ordinal=0)
    append_session(
        session_file,
        meta=SessionMeta(session_id=sid, model_id="m").serialize(),
        tape_delta=[
            ReferrableTapeEvent(ref=collision, event=UserMessage(text="a")),
            ContextSplice(
                ref=TapeRef(session_id=sid, ordinal=1),
                mask=(MaskRange(session_id=sid, lo=0, hi=0),),
                insert_after=None,
                payload=(UserMessage(text="barrier"),),
                strategy="barrier",
            ),
            ReferrableTapeEvent(ref=collision, event=UserMessage(text="later-real")),
        ],
    )

    loaded = load_session(tmp_path)
    assert loaded is not None
    texts = [getattr(m, "text", "") for m in resolve_context(loaded[1]).messages]
    assert texts == ["barrier", "later-real"], f"post-barrier record lost; {texts}"


def test_a_corrupt_session_backup_is_not_world_readable(tmp_path: Path) -> None:
    """REV6 PS2-005: the forensic copy holds everything the original does.

    ``write_bytes`` creates under the umask, so on a default ``022`` host the
    ``.corrupt-*`` sibling landed at ``0644`` -- the backup publishing the
    prompts, tool output, and secrets that the live transcript protects.
    """
    original = os.umask(0o022)
    try:
        session_file = tmp_path / "session.jsonl"
        _ = session_file.write_text("{not json\n", encoding="utf-8")
        session_file.chmod(0o600)

        session_io._preserve_corrupt_session(session_file)

        backup = next(tmp_path.glob("session.jsonl.corrupt-*"))
        mode = stat.S_IMODE(backup.stat().st_mode)
    finally:
        _ = os.umask(original)

    assert not mode & 0o077, f"corrupt-session backup is 0o{mode:o}"


def test_a_caller_cannot_retype_a_record_via_its_payload(tmp_path: Path) -> None:
    """REV6 PS2-007: the ``kind`` discriminator belongs to the API, not the data.

    ``{"kind": "meta", **meta}`` let a ``kind`` key inside ``meta`` win, so a
    metadata blob could be written tagged ``history`` -- the loader would then
    treat it as a message and the session's metadata would vanish silently.
    """
    session_file = tmp_path / "session.jsonl"

    append_session(
        session_file,
        meta={"kind": "history", "session_id": "s", "type": "user", "text": "hijack"},
        tool_state_snapshot={"kind": "meta", "bash_cwd": "/x"},
    )

    kinds = [
        json.loads(line)["kind"]
        for line in session_file.read_text(encoding="utf-8").splitlines()
    ]
    assert kinds == ["meta", "tool_state"], f"caller data retyped a record: {kinds}"


def test_a_persisted_true_is_not_a_tool_depth(tmp_path: Path) -> None:
    """REV6 PS2-009: ``isinstance(True, int)`` holds, so ``true`` became depth 1.

    Depth caps ``AgentSpawn`` nesting, so a bool decoding to 1 is a spawn
    budget the operator never set. The sibling decoders already reject this
    exact trap at the same boundary.
    """
    del tmp_path
    state = ToolState()
    default = state.depth

    restore_tool_state(state, {"depth": True})

    assert state.depth == default, f"a JSON bool became depth {state.depth!r}"
    restore_tool_state(state, {"depth": 3})
    assert state.depth == 3, "a real integer depth must still restore"


def test_a_transcript_is_not_readable_by_other_users(tmp_path: Path) -> None:
    """A transcript is private: prompts, tool output, file contents, secrets.

    ``_append_lines`` created it ``0o644`` and the parent directory took the
    umask default, so on any shared or multi-account host every local user
    could read whole conversations. Measured on one developer's machine: 400
    of 400 sampled transcripts were group- and world-readable.
    """
    original = os.umask(0)
    try:
        session_dir = tmp_path / "s"
        session_file = session_dir / "session.jsonl"
        append_session(
            session_file,
            meta=SessionMeta(session_id="private", model_id="m").serialize(),
        )
        file_mode = stat.S_IMODE(session_file.stat().st_mode)
        dir_mode = stat.S_IMODE(session_dir.stat().st_mode)
    finally:
        _ = os.umask(original)

    assert not file_mode & 0o077, f"transcript is 0o{file_mode:o}"
    assert not dir_mode & 0o077, f"session dir is 0o{dir_mode:o}"


def test_an_already_permissive_transcript_is_tightened_on_append(
    tmp_path: Path,
) -> None:
    """Every transcript on disk predates the mode arguments, so they miss it.

    ``mkdir(mode=...)`` is ignored when the directory exists and
    ``os.open(..., 0o600)`` applies only when the file is created, so the modes
    added for new sessions never reach the sessions that already exist -- which
    is all of them. Measured: 400 of 400 sampled transcripts are group- and
    other-readable, under ``0o777`` directories.
    """
    original = os.umask(0)
    try:
        session_dir = tmp_path / "s"
        session_dir.mkdir(mode=0o777)
        session_file = session_dir / "session.jsonl"
        _ = session_file.write_text("", encoding="utf-8")
        session_file.chmod(0o644)

        append_session(
            session_file,
            meta=SessionMeta(session_id="legacy", model_id="m").serialize(),
        )

        file_mode = stat.S_IMODE(session_file.stat().st_mode)
        dir_mode = stat.S_IMODE(session_dir.stat().st_mode)
    finally:
        _ = os.umask(original)

    assert not file_mode & 0o077, f"pre-existing transcript is 0o{file_mode:o}"
    assert not dir_mode & 0o077, f"pre-existing session dir is 0o{dir_mode:o}"


def test_a_short_write_cannot_interleave_another_append(tmp_path: Path) -> None:
    """``O_APPEND`` makes one ``os.write`` atomic -- not a loop of them.

    The batch is written by a retry loop, so a short write leaves half a JSON
    line on disk with the file offset released; another writer appending in
    that gap lands its records between the halves and the spliced line never
    parses again. The tape is the only copy of the conversation, so an
    unparseable record is lost history.
    """
    session_file = tmp_path / "session.jsonl"
    real_write = os.write
    halved: list[bool] = []
    intruder_done = threading.Event()

    def intruder() -> None:
        with patch.object(os, "write", real_write):
            append_session(
                session_file,
                tape_delta=[
                    ReferrableTapeEvent(
                        ref=TapeRef(session_id="other", ordinal=0),
                        event=UserMessage(text="from the other writer"),
                    ),
                ],
            )
        intruder_done.set()

    threads: list[threading.Thread] = []

    def short_write(fd: int, data: object) -> int:
        buffer = cast(memoryview, data)
        if halved or len(buffer) < 2:
            return real_write(fd, buffer)
        halved.append(True)
        written = real_write(fd, buffer[: len(buffer) // 2])
        thread = threading.Thread(target=intruder)
        thread.start()
        threads.append(thread)
        # Long enough for an unguarded intruder to land its records in the gap;
        # a guarded one is still blocked when this returns.
        _ = intruder_done.wait(0.05)
        return written

    with patch.object(os, "write", short_write):
        append_session(
            session_file,
            tape_delta=[
                ReferrableTapeEvent(
                    ref=TapeRef(session_id="main", ordinal=0),
                    event=UserMessage(text="y" * 4096),
                ),
            ],
        )
    for thread in threads:
        thread.join(5.0)

    assert halved, "the short write never fired; the test proves nothing"
    for line in session_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        _ = json.loads(line)


def test_new_session_dir_is_not_readable_by_other_users(tmp_path: Path) -> None:
    """The directory is created before any append, so it needs its own mode.

    ``new_session_dir`` is what mints a session directory; ``append_session``
    only ever sees it as pre-existing. Under a default umask the listing --
    which leaks nothing but the session ids -- was world-readable, and the
    directory stayed ``0o777`` for every file written into it afterwards.
    """
    original = os.umask(0)
    try:
        created = new_session_dir(tmp_path / "proj", projects_dir=tmp_path / "root")
        mode = stat.S_IMODE(created.stat().st_mode)
        parent_mode = stat.S_IMODE(created.parent.stat().st_mode)
    finally:
        _ = os.umask(original)

    assert not mode & 0o077, f"session dir is 0o{mode:o}"
    assert not parent_mode & 0o077, f"project dir is 0o{parent_mode:o}"


def test_load_session_keeps_both_records_when_a_ref_collides(
    tmp_path: Path,
) -> None:
    """A duplicated ref must be re-numbered, not resolved as one record twice.

    Two writers minting the same ordinal is a real shape -- three sessions on
    disk carry it. The resolver now refuses a duplicate rather than silently
    rendering the later record twice, so the loader has to reconcile it: the
    file is the only copy of the conversation, and dropping either record (or
    raising) loses history the user cannot get back.
    """
    session_file = tmp_path / "session.jsonl"
    meta = SessionMeta(session_id="collide", model_id="m", provider="P", auth="env")
    collision = TapeRef(session_id="collide", ordinal=0)
    append_session(
        session_file,
        meta=meta.serialize(),
        tape_delta=[
            ReferrableTapeEvent(ref=collision, event=UserMessage(text="first")),
            ReferrableTapeEvent(ref=collision, event=UserMessage(text="second")),
        ],
    )

    loaded = load_session(tmp_path)
    assert loaded is not None
    _, tape, _ = loaded

    assert len({record.ref for record in tape}) == len(tape), (
        f"duplicate refs survived load; got {[r.ref for r in tape]}"
    )
    texts = [getattr(m, "text", "") for m in resolve_context(tape).messages]
    assert "first" in " ".join(texts), f"earlier record dropped; got {texts}"
    assert "second" in " ".join(texts), f"later record dropped; got {texts}"


def test_load_session_repairs_orphan_tool_result_from_splice_payload(
    tmp_path: Path,
) -> None:
    session_file = tmp_path / "session.jsonl"
    meta = SessionMeta(
        session_id="orphan-splice", model_id="m", provider="P", auth="env"
    )
    user = UserMessage(text="go")
    assistant = AssistantMessage(
        tool_calls=(ToolCall(id="kept", name="Echo", args={}),)
    )
    result = ToolResult(call_id="kept", content="done")
    summary = UserMessage(text="[summary]")
    orphan = ToolResult(call_id="ghost", content="late")
    refs = [TapeRef(session_id="orphan-splice", ordinal=i) for i in range(5)]
    append_session(
        session_file,
        meta=meta.serialize(),
        tape_delta=[
            ReferrableTapeEvent(ref=refs[0], event=user),
            ReferrableTapeEvent(ref=refs[1], event=assistant),
            ReferrableTapeEvent(ref=refs[2], event=result),
            ContextSplice(
                ref=refs[3],
                mask=(MaskRange.between(refs[0], refs[2]),),
                insert_after=None,
                payload=(summary,),
                strategy="summary",
            ),
            ContextSplice(
                ref=refs[4],
                mask=(MaskRange.between(refs[2], refs[2]),),
                insert_after=refs[1],
                payload=(orphan,),
                strategy="detached_splice",
                paired_externally=frozenset({"ghost"}),
            ),
        ],
    )

    loaded = load_session(tmp_path)
    assert loaded is not None
    _, tape, _ = loaded
    resolved = resolve_context(tape).messages
    validate_context(resolved)
    assert resolved == [summary]
    assert any(
        isinstance(record, ContextSplice)
        and record.strategy == "orphan_tool_result_repair"
        and record.mask == (MaskRange.between(refs[0], refs[4]),)
        for record in tape
    )
    runtime = agent_runtime.AgentRuntime(model=_RuntimeModel())
    runtime.replay_tape(tape)
    runtime.append_splice(
        mask=(MaskRange.between(runtime.tape[0].ref, runtime.tape[-1].ref),),
        insert_after=None,
        payload=(UserMessage(text="[next summary]"),),
        strategy="summary",
        # A summary replaces what it absorbs; the carry guard compares text,
        # so the producer states the intent instead of it being inferred.
        discards_content=True,
    )


def test_resumed_dangling_session_survives_the_next_user_message(
    tmp_path: Path,
) -> None:
    """A resumed mid-tool session must not lose its history to one message.

    The incident: ``load_session`` repairs an interrupted tool turn by
    appending a barrier splice that masks the whole tape and carries the
    conversation as its payload. The user's next message coalesces onto that
    payload's user-side tail, absorbing the barrier's mask -- which kills the
    barrier and, unless its payload is carried forward, deletes every message
    the session had. Presents as a resume that "worked" and then remembered
    nothing on the first reply.
    """
    session_file = tmp_path / "session.jsonl"
    meta = SessionMeta(session_id="resumed", model_id="m", provider="P", auth="env")
    refs = [TapeRef(session_id="resumed", ordinal=i) for i in range(4)]
    append_session(
        session_file,
        meta=meta.serialize(),
        tape_delta=[
            ReferrableTapeEvent(ref=refs[0], event=UserMessage(text="do the thing")),
            ReferrableTapeEvent(ref=refs[1], event=AssistantMessage(text="working")),
            # Interrupted mid-tool: the tool_use persisted, its result did not.
            ReferrableTapeEvent(
                ref=refs[2],
                event=AssistantMessage(
                    tool_calls=(ToolCall(id="call_1", name="Bash", args={}),)
                ),
            ),
            ReferrableTapeEvent(ref=refs[3], event=UserMessage(text="still there?")),
        ],
    )

    loaded = load_session(tmp_path)
    assert loaded is not None
    _, tape, _ = loaded
    runtime = agent_runtime.AgentRuntime(model=_RuntimeModel(), session_id="resumed")
    runtime.replay_tape(tape)
    before = len(runtime.context().messages)
    assert before > 1, "fixture did not resume a multi-message conversation"

    runtime._append_or_coalesce_user(UserMessage(text="next"))

    messages = runtime.context().messages
    assert len(messages) == before, (
        f"resume lost history: {before} messages before the reply, "
        f"{len(messages)} after -- {[getattr(m, 'text', '') for m in messages]}"
    )
    assert any(
        isinstance(m, UserMessage) and "do the thing" in m.text for m in messages
    ), "the session's first user message is gone"
    validate_context(messages)


def test_append_session_writes_meta_then_tape_delta(tmp_path: Path) -> None:
    """Within one batch: ``meta`` precedes any tape records."""
    session_file = tmp_path / "session.jsonl"
    meta = SessionMeta(session_id="abc")
    append_session(
        session_file,
        meta=meta.serialize(),
        tape_delta=_records_from([UserMessage(text="x")]),
    )
    lines = session_file.read_text(encoding="utf-8").splitlines()
    parsed = [json.loads(line) for line in lines]
    kinds = [r["kind"] for r in parsed]
    assert kinds == ["meta", "history"]


def _service_suspended(account: str | None = "default") -> ModelServiceSuspended:
    """Build a persisted service-suspension fixture."""
    return ModelServiceSuspended(
        provider="OpenAISubscription",
        auth="credentials",
        account=account,
        model_id="gpt-5.5",
        retry_at=1_800_000_000.0,
        delay_sec=14_868.0,
        server_supplied=True,
        error=ServiceErrorSnapshot(
            type_name="RateLimitError",
            message="limited",
            status=429,
            headers={"retry-after": "14868"},
            body='{"error":"rate_limit"}',
        ),
    )


def test_append_session_writes_model_service_suspended_event(tmp_path: Path) -> None:
    session_file = tmp_path / "session.jsonl"
    append_session(session_file, runtime_events=[_service_suspended()])

    record = json.loads(session_file.read_text(encoding="utf-8"))
    assert record == {
        "kind": "runtime_event",
        "type": "model_service_suspended",
        "timestamp": record["timestamp"],
        "provider": "OpenAISubscription",
        "auth": "credentials",
        "account": "default",
        "model_id": "gpt-5.5",
        "retry_at": 1_800_000_000.0,
        "delay_sec": 14_868.0,
        "server_supplied": True,
        "error": {
            "type_name": "RateLimitError",
            "message": "limited",
            "status": 429,
            "headers": {"retry-after": "14868"},
            "body": '{"error":"rate_limit"}',
        },
    }


def test_load_session_decodes_model_service_suspended_event(tmp_path: Path) -> None:
    append_session(tmp_path / "session.jsonl", runtime_events=[_service_suspended()])

    loaded = load_session(tmp_path)

    assert loaded is not None
    meta, _, _ = loaded
    assert meta.runtime_events == (_service_suspended(),)


def test_model_service_suspended_account_none_round_trips(tmp_path: Path) -> None:
    append_session(
        tmp_path / "session.jsonl", runtime_events=[_service_suspended(None)]
    )

    loaded = load_session(tmp_path)

    assert loaded is not None
    meta, _, _ = loaded
    assert meta.runtime_events == (_service_suspended(None),)


def _notice() -> NoticeMessage:
    """Build a persisted notice fixture with an error snapshot."""
    return NoticeMessage(
        text="[rate-limited: allowed_warning at 89% weekly]",
        tier="advisory",
        error=ServiceErrorSnapshot(
            type_name="RateLimitError",
            message="Rate limited",
            status=200,
            headers={"anthropic-ratelimit-unified-status": "allowed_warning"},
        ),
    )


def test_append_session_writes_notice_message_event(tmp_path: Path) -> None:
    session_file = tmp_path / "session.jsonl"
    append_session(session_file, runtime_events=[_notice()])

    record = json.loads(session_file.read_text(encoding="utf-8"))
    assert record == {
        "kind": "runtime_event",
        "type": "notice_message",
        "timestamp": record["timestamp"],
        "text": "[rate-limited: allowed_warning at 89% weekly]",
        "tier": "advisory",
        "error": {
            "type_name": "RateLimitError",
            "message": "Rate limited",
            "status": 200,
            "headers": {"anthropic-ratelimit-unified-status": "allowed_warning"},
            "body": "",
        },
    }


def test_load_session_decodes_notice_message_event(tmp_path: Path) -> None:
    append_session(tmp_path / "session.jsonl", runtime_events=[_notice()])

    loaded = load_session(tmp_path)

    assert loaded is not None
    meta, _, _ = loaded
    assert meta.runtime_events == (_notice(),)


def test_notice_message_without_error_round_trips(tmp_path: Path) -> None:
    notice = NoticeMessage(text="[heads up]", tier="advisory")
    append_session(tmp_path / "session.jsonl", runtime_events=[notice])

    loaded = load_session(tmp_path)

    assert loaded is not None
    meta, _, _ = loaded
    assert meta.runtime_events == (notice,)


def test_append_session_writes_persistent_agent_lifecycle(tmp_path: Path) -> None:
    session_file = tmp_path / "session.jsonl"
    append_session(
        session_file,
        persistent_agents=[
            PersistentAgentRecord(
                label="fix-tools",
                run_id="run-1",
                session_dir=str(tmp_path / "session" / "fix-tools"),
                state="running",
                provider="OpenAISubscription",
                auth="credentials",
                account="default",
                model_id="gpt-5.5",
                tools=("Read", "Edit"),
                system="system text",
                notify_on_asleep=True,
            )
        ],
    )

    record = json.loads(session_file.read_text(encoding="utf-8"))
    assert record == {
        "kind": "persistent_agent",
        "label": "fix-tools",
        "run_id": "run-1",
        "session_dir": str(tmp_path / "session" / "fix-tools"),
        "state": "running",
        "provider": "OpenAISubscription",
        "auth": "credentials",
        "account": "default",
        "model_id": "gpt-5.5",
        "tools": ["Read", "Edit"],
        "system": "system text",
        "notify_on_asleep": True,
        "max_tool_call_rounds": None,
        "max_request_tokens": None,
        "max_response_tokens": None,
        "thinking_budget": "none",
        "thinking_output": "none",
        "show_thinking": True,
        "effort": "none",
        "cache_ttl_sec": 300.0,
        "service_tier": "auto",
        "max_budget_usd": None,
        "persistent_retry": False,
        "timestamp": record["timestamp"],
    }


def test_load_persistent_agents_returns_latest_running_records(tmp_path: Path) -> None:
    session_file = tmp_path / "session.jsonl"
    running = PersistentAgentRecord(
        label="fix-tools",
        run_id="run-1",
        session_dir=str(tmp_path / "children" / "run-1"),
        state="running",
        provider="OpenAISubscription",
        auth="credentials",
        account="default",
        model_id="gpt-5.5",
        tools=("Read",),
        system="system text",
        notify_on_asleep=True,
    )
    append_session(session_file, persistent_agents=[running])
    append_session(
        session_file,
        persistent_agents=[dataclasses.replace(running, state="cancelled")],
    )
    still_running = dataclasses.replace(running, run_id="run-2", label="fix-compact")
    append_session(session_file, persistent_agents=[still_running])

    assert load_persistent_agents(tmp_path) == [still_running]


def test_load_persistent_agents_ignores_all_terminal_states(tmp_path: Path) -> None:
    session_file = tmp_path / "session.jsonl"
    running = PersistentAgentRecord(
        label="fix-tools",
        run_id="run-1",
        session_dir=str(tmp_path / "children" / "run-1"),
        state="running",
        provider="OpenAISubscription",
        auth="credentials",
        account="default",
        model_id="gpt-5.5",
        tools=("Read",),
        system="system text",
        notify_on_asleep=True,
    )
    append_session(session_file, persistent_agents=[running])
    for state in ("completed", "failed", "cancelled"):
        append_session(
            session_file,
            persistent_agents=[dataclasses.replace(running, state=state)],
        )

    assert load_persistent_agents(tmp_path) == []


def test_load_persistent_agents_omits_completed_oneshot(tmp_path: Path) -> None:
    """T6: a finished oneshot child (state='completed') is not resurrected.

    A oneshot child writes a terminal ``completed`` lifecycle record when
    it self-stops. ``load_persistent_agents`` keeps only ``running``
    records, so the completed oneshot is omitted and resume never
    re-hosts it.
    """
    session_file = tmp_path / "session.jsonl"
    completed = PersistentAgentRecord(
        label="oneshot-child",
        run_id="run-oneshot",
        session_dir=str(tmp_path / "children" / "run-oneshot"),
        state="completed",
        provider="OpenAISubscription",
        auth="credentials",
        account="default",
        model_id="gpt-5.5",
        tools=("Read",),
        system="system text",
        notify_on_asleep=False,
    )
    append_session(session_file, persistent_agents=[completed])

    assert load_persistent_agents(tmp_path) == []


def test_load_persistent_agents_rejects_empty_session_dir(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "session.jsonl",
        {
            "kind": "persistent_agent",
            "label": "fix-tools",
            "run_id": "run-1",
            "session_dir": "",
            "state": "running",
            "provider": "OpenAISubscription",
            "auth": "credentials",
            "account": None,
            "model_id": "gpt-5.5",
            "tools": ["Read"],
            "system": "system text",
            "notify_on_asleep": True,
        },
    )

    assert load_persistent_agents(tmp_path) == []


def test_persistent_agent_legacy_notify_on_asleep_defaults_true(tmp_path: Path) -> None:
    session_dir = tmp_path / "children" / "run-1"
    _write_jsonl(
        tmp_path / "session.jsonl",
        {
            "kind": "persistent_agent",
            "label": "fix-tools",
            "run_id": "run-1",
            "session_dir": str(session_dir),
            "state": "running",
            "provider": "OpenAISubscription",
            "auth": "credentials",
            "account": None,
            "model_id": "gpt-5.5",
            "tools": ["Read"],
            "system": "system text",
        },
    )

    records = load_persistent_agents(tmp_path)

    assert len(records) == 1
    assert records[0].notify_on_asleep is True
    assert records[0].account is None


@pytest.mark.parametrize(
    ("legacy", "budget", "output", "show"),
    [
        ({"thinking_state": "adaptive-show"}, "auto", "text", True),
        ({"thinking_state": "adaptive-hide"}, "auto", "text", False),
        ({"thinking_state": "on-show"}, "fixed", "text", True),
        ({"thinking_state": "off-hide"}, "none", "none", False),
        ({"thinking_state": "redact-hide"}, "auto", "redacted", False),
        # No state at all: the wire mode was the only other record of it.
        ({"thinking": "adaptive"}, "auto", "text", True),
        ({"thinking": "enabled"}, "fixed", "text", True),
        ({}, "none", "none", True),
    ],
)
def test_persistent_agent_upgrades_a_pre_split_thinking_record(
    tmp_path: Path,
    legacy: dict[str, object],
    budget: str,
    output: str,
    show: bool,
) -> None:
    """A session written before the split still resumes.

    The fused ``thinking_state`` spelled the same two axes plus the display
    bit, so an old record is decoded rather than silently read as "off".
    """
    _write_jsonl(
        tmp_path / "session.jsonl",
        {
            "kind": "persistent_agent",
            "label": "fix-tools",
            "run_id": "run-1",
            "session_dir": str(tmp_path / "children" / "run-1"),
            "state": "running",
            "provider": "Anthropic",
            "auth": "env",
            "account": None,
            "model_id": "claude-opus-4-8",
            "tools": ["Read"],
            "system": "system text",
            **legacy,
        },
    )

    record = load_persistent_agents(tmp_path)[0]

    assert record.thinking_budget == budget
    assert record.thinking_output == output
    assert record.show_thinking is show


def test_json_bool_default_is_reachable_from_every_caller() -> None:
    """A decoder with a ``default=`` no caller passes has two shapes on disk.

    ``notify_on_asleep`` hand-rolled the same isinstance-else-default decode
    while ``_json_bool``'s own ``default=`` went unused across six call sites.
    Two spellings of one decode is how they drift: the wire values must agree.
    """
    assert _json_bool("false", default=True) is True
    assert _json_bool(False, default=True) is False
    assert _json_bool(None, default=True) is True


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(True, True), (False, False), ("false", True), (1, True), (None, True)],
)
def test_persistent_agent_notify_on_asleep_decodes_through_json_bool(
    tmp_path: Path, raw: object, expected: bool
) -> None:
    """Non-bool ``notify_on_asleep`` takes the default; a JSON ``false`` does not.

    The field is the parent's idle-ping switch: a truthy-string decode would
    turn a persisted ``"false"`` into notifications the operator disabled.
    """
    _write_jsonl(
        tmp_path / "session.jsonl",
        {
            "kind": "persistent_agent",
            "label": "fix-tools",
            "run_id": "run-1",
            "session_dir": str(tmp_path / "children" / "run-1"),
            "state": "running",
            "provider": "OpenAISubscription",
            "auth": "credentials",
            "account": None,
            "model_id": "gpt-5.5",
            "tools": ["Read"],
            "system": "system text",
            "notify_on_asleep": raw,
        },
    )

    records = load_persistent_agents(tmp_path)

    assert len(records) == 1
    assert records[0].notify_on_asleep is expected


def test_session_meta_round_trip() -> None:
    src = SessionMeta(
        session_id="x", model_id="y", provider="P", auth="env", status="busy"
    )
    blob = src.serialize()
    back = SessionMeta.deserialize(blob)
    assert back.session_id == "x"
    assert back.model_id == "y"
    assert back.provider == "P"
    assert back.status == "busy"


def _write_jsonl(path: Path, *records: object) -> None:
    """Write each record as a JSON line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            _ = f.write(json.dumps(r) + "\n")


def test_load_session_skips_unknown_history_type(tmp_path: Path) -> None:
    """Unknown history ``type`` falls through ``_entry_from_json`` → None."""
    session_file = tmp_path / "session.jsonl"
    _write_jsonl(
        session_file,
        {"kind": "meta", "session_id": "abc"},
        {"kind": "history", "type": "mystery", "text": "x"},
    )

    loaded = load_session(tmp_path)
    assert loaded is not None
    _, tape, _ = loaded
    history = _history_from_tape(tape)
    # Unknown ``type`` was silently dropped.
    assert history == []


def test_load_session_drops_attachment_with_bad_mime_or_data(tmp_path: Path) -> None:
    """``_att_from_json`` drops attachments missing or malformed fields."""
    session_file = tmp_path / "session.jsonl"
    _write_jsonl(
        session_file,
        {"kind": "meta", "session_id": "x"},
        {
            "kind": "history",
            "type": "user",
            "text": "with bad atts",
            "attachments": [
                "not a dict",
                {"mime": 5, "data": "abc"},
                {"mime": "image/png", "data": "!!! not base64 !!!"},
                {"mime": "image/png", "data": "aGVsbG8="},  # valid
            ],
        },
    )
    loaded = load_session(tmp_path)
    assert loaded is not None
    _, tape, _ = loaded
    history = _history_from_tape(tape)
    assert len(history) == 1
    entry = history[0]
    assert isinstance(entry, UserMessage)
    # Only the valid base64 attachment survives.
    assert len(entry.attachments) == 1
    assert entry.attachments[0].data == b"hello"


def test_load_session_drops_attachment_with_unknown_mime_prefix(
    tmp_path: Path,
) -> None:
    """``_att_from_json`` rejects descriptors outside the known-prefix allow-list."""
    session_file = tmp_path / "session.jsonl"
    _write_jsonl(
        session_file,
        {"kind": "meta", "session_id": "x"},
        {
            "kind": "history",
            "type": "user",
            "text": "mixed",
            "attachments": [
                {"mime": "rogue/descriptor", "data": "aGVsbG8="},
                {"mime": "application/x-malware", "data": "aGVsbG8="},
                {"mime": "image/png", "data": "aGVsbG8="},
            ],
        },
    )
    loaded = load_session(tmp_path)
    assert loaded is not None
    _, tape, _ = loaded
    entry = _history_from_tape(tape)[0]
    assert isinstance(entry, UserMessage)
    assert [a.descriptor for a in entry.attachments] == ["image/png"]


def test_repair_dangling_tape_handles_legacy_consecutive_assistants(
    tmp_path: Path,
) -> None:
    """Tapes with two adjacent assistants + orphan tool_use load without raising.

    Without ``ContextSplice.replay`` in ``_repair_dangling_tape`` the
    validating constructor rejects the role-alternation violation when
    the repair barrier is materialized.
    """
    session_file = tmp_path / "session.jsonl"
    _write_jsonl(
        session_file,
        {"kind": "meta", "session_id": "abc"},
        {
            "kind": "history",
            "ref": {"session_id": "abc", "ordinal": 0},
            "type": "user",
            "text": "go",
        },
        {
            "kind": "history",
            "ref": {"session_id": "abc", "ordinal": 1},
            "type": "assistant",
            "text": "first",
        },
        cast(
            dict[str, object],
            {
                "kind": "history",
                "ref": {"session_id": "abc", "ordinal": 2},
                "type": "assistant",
                "text": "second with orphan call",
                "tool_calls": [
                    {
                        "id": "call_x",
                        "name": "echo",
                        "args": cast("Mapping[str, object]", {}),
                    }
                ],
            },
        ),
    )
    loaded = load_session(tmp_path)
    assert loaded is not None
    _, tape, _ = loaded
    history = _history_from_tape(tape)
    # Repair pairs the orphan tool_use with a synthetic [interrupted] result.
    tool_results = [e for e in history if isinstance(e, ToolResult)]
    assert any("[interrupted]" in tr.content for tr in tool_results)


def test_repair_dangling_tape_handles_legacy_duplicate_tool_call_id(
    tmp_path: Path,
) -> None:
    """Legacy tape with two AMs sharing a tool_call id must load, not wedge.

    The canonical repair drops the duplicate id from the later AM and, when
    that leaves the AM hollow (no text / thinking), drops the AM entirely. No
    two AMs then share an id, so the validating ``ContextSplice`` constructor
    accepts the repair splice instead of raising ``duplicate tool_call id``
    and wedging the resume (F1, sibling of the compaction-path H2).
    """
    session_file = tmp_path / "session.jsonl"
    _write_jsonl(
        session_file,
        {"kind": "meta", "session_id": "abc"},
        cast(
            dict[str, object],
            {
                "kind": "history",
                "ref": {"session_id": "abc", "ordinal": 0},
                "type": "assistant",
                "tool_calls": [
                    {
                        "id": "t1",
                        "name": "echo",
                        "args": cast("Mapping[str, object]", {}),
                    }
                ],
            },
        ),
        cast(
            dict[str, object],
            {
                "kind": "history",
                "ref": {"session_id": "abc", "ordinal": 1},
                "type": "assistant",
                "tool_calls": [
                    {
                        "id": "t1",
                        "name": "echo",
                        "args": cast("Mapping[str, object]", {}),
                    }
                ],
            },
        ),
    )
    loaded = load_session(tmp_path)
    assert loaded is not None


def test_sort_tape_by_ordinal_keeps_file_order_for_a_tie() -> None:
    """Ordinal orders the tape; a same-ordinal tie keeps the recorded order.

    Sorting by ``session_id`` first grouped each session's whole run together,
    which is not what the tape means: the resolver reads it as append order and
    anchors splices against what precedes them, so the regrouping reversed
    conversations on resumed and forked tapes.
    """
    rec_a = ReferrableTapeEvent(
        ref=TapeRef(session_id="b", ordinal=1), event=UserMessage(text="b-1")
    )
    rec_b = ReferrableTapeEvent(
        ref=TapeRef(session_id="a", ordinal=1), event=UserMessage(text="a-1")
    )
    rec_c = ReferrableTapeEvent(
        ref=TapeRef(session_id="a", ordinal=0), event=UserMessage(text="a-0")
    )
    sorted_tape = session_io._sort_tape_by_ordinal([rec_a, rec_b, rec_c])
    assert [(r.ref.session_id, r.ref.ordinal) for r in sorted_tape] == [
        ("a", 0),
        ("b", 1),
        ("a", 1),
    ]


def test_load_session_drops_non_list_attachments_and_thinking(
    tmp_path: Path,
) -> None:
    """Non-list ``attachments`` / ``thinking_blocks`` parse to empty."""
    session_file = tmp_path / "session.jsonl"
    _write_jsonl(
        session_file,
        {
            "kind": "history",
            "type": "user",
            "text": "u",
            "attachments": "not-a-list",
        },
        {
            "kind": "history",
            "type": "assistant",
            "text": "a",
            "thinking_blocks": "not-a-list",
        },
    )
    loaded = load_session(tmp_path)
    assert loaded is not None
    _, tape, _ = loaded
    history = _history_from_tape(tape)
    user = history[0]
    asst = history[1]
    assert isinstance(user, UserMessage)
    assert user.attachments == ()
    assert isinstance(asst, AssistantMessage)
    assert asst.thinking_blocks == ()


def test_load_session_drops_non_dict_tool_calls(tmp_path: Path) -> None:
    """``tool_calls`` items that aren't dicts are skipped."""
    session_file = tmp_path / "session.jsonl"
    bad_tc: object = "not a dict"
    good_tc: dict[str, object] = {
        "id": "c1",
        "name": "echo",
        "args": cast("Mapping[str, object]", {}),
    }
    record: dict[str, object] = {
        "kind": "history",
        "type": "assistant",
        "text": "x",
        "tool_calls": [bad_tc, good_tc],
    }
    _write_jsonl(session_file, record)
    loaded = load_session(tmp_path)
    assert loaded is not None
    _, tape, _ = loaded
    history = _history_from_tape(tape)
    asst = history[0]
    assert isinstance(asst, AssistantMessage)
    assert len(asst.tool_calls) == 1
    assert asst.tool_calls[0].id == "c1"


def test_load_session_skips_blank_lines_and_non_dict_records(tmp_path: Path) -> None:
    """Blank lines and non-dict JSON records are skipped without error."""
    session_file = tmp_path / "session.jsonl"
    session_file.parent.mkdir(parents=True, exist_ok=True)
    with session_file.open("w", encoding="utf-8") as f:
        _ = f.write("\n")  # blank line → ``continue``
        _ = f.write("   \n")  # whitespace-only line
        _ = f.write("42\n")  # not a dict
        _ = f.write(
            json.dumps({"kind": "history", "type": "user", "text": "hi"}) + "\n"
        )

    loaded = load_session(tmp_path)
    assert loaded is not None
    _, tape, _ = loaded
    history = _history_from_tape(tape)
    assert len(history) == 1
    entry = history[0]
    assert isinstance(entry, UserMessage)
    assert entry.text == "hi"


def test_load_session_preserves_and_skips_corrupt_lines(tmp_path: Path) -> None:
    """A corrupt JSON line triggers backup and skip; valid lines load."""
    session_file = tmp_path / "session.jsonl"
    session_file.parent.mkdir(parents=True, exist_ok=True)
    with session_file.open("w", encoding="utf-8") as f:
        _ = f.write("{not valid json\n")
        _ = f.write(
            json.dumps({"kind": "history", "type": "user", "text": "after"}) + "\n"
        )

    loaded = load_session(tmp_path)
    assert loaded is not None
    _, tape, _ = loaded
    history = _history_from_tape(tape)
    assert len(history) == 1
    entry = history[0]
    assert isinstance(entry, UserMessage)
    assert entry.text == "after"
    # A `*.corrupt-*` sibling was created to preserve the original bytes.
    backups = list(tmp_path.glob("session.jsonl.corrupt-*"))
    assert len(backups) == 1


def test_load_session_returns_none_when_file_unreadable(tmp_path: Path) -> None:
    """OSError while reading the session file returns None."""
    session_file = tmp_path / "session.jsonl"
    session_file.touch()

    def _boom(self: Path, *args: object, **kwargs: object) -> object:
        del self, args, kwargs
        raise OSError("permission denied")

    with patch.object(Path, "open", _boom):
        assert load_session(tmp_path) is None


def test_load_session_uses_meta_bash_cwd_when_no_snapshot(tmp_path: Path) -> None:
    """``meta.bash_cwd`` seeds ``ToolState`` when no tool_state record exists."""
    session_file = tmp_path / "session.jsonl"
    append_session(
        session_file,
        meta=SessionMeta(session_id="x", bash_cwd="/from/meta").serialize(),
    )
    loaded = load_session(tmp_path)
    assert loaded is not None
    _, _, state = loaded
    assert state.bash_cwd == "/from/meta"


def test_preserve_corrupt_session_swallows_oserror(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """``_preserve_corrupt_session`` logs and returns if write fails."""
    session_file = tmp_path / "session.jsonl"
    session_file.write_text("garbage", encoding="utf-8")

    def _boom(self: Path, data: object) -> int:
        del self, data
        raise OSError("disk full")

    with caplog.at_level("ERROR"), patch.object(Path, "write_bytes", _boom):
        session_io._preserve_corrupt_session(session_file)

    assert "Could not preserve corrupt session file" in caplog.text


def test_restore_tool_state_drops_bad_read_cache_entries() -> None:
    """``restore_tool_state`` ignores non-dict and missing-path read_cache rows."""
    state = ToolState()
    snapshot: dict[str, object] = {
        "bash_cwd": "/x",
        "depth": 0,
        "additional_dirs": [],
        "recent_files": ["", 42, "/real/path"],  # blank, non-str, valid
        "read_cache": [
            "not-a-dict",  # skipped
            {"path": "", "offset": 0},  # skipped (empty path)
            {
                "path": "/p/x.txt",
                "offset": 0,
                "limit": 100,
                "last_lines": 10,
                "mtime": 1.0,
            },
        ],
    }
    restore_tool_state(state, snapshot)
    assert "/p/x.txt" in state.read_cache
    # Only "/real/path" survives the recent_files filter.
    assert any("/real/path" in p for p in state.recent_files)


def test_restore_model_returns_none_when_missing_provider_or_model() -> None:
    """``restore_model`` short-circuits to None for missing fields."""
    assert restore_model(SessionMeta()) is None
    assert restore_model(SessionMeta(provider="P")) is None
    assert restore_model(SessionMeta(model_id="m")) is None


def test_restore_model_returns_none_on_attribute_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Provider lookup failure (AttributeError) returns None."""
    meta = SessionMeta(provider="DoesNotExist", model_id="m", auth="env")
    with caplog.at_level("WARNING"):
        assert restore_model(meta) is None
    assert "Failed to restore model" in caplog.text


def test_restore_model_success_path() -> None:
    """A working provider builds a model and spec."""

    class _FakeModel:
        tagged_model_id: str = "fake-m"

    class _FakeProvider:
        def model(self, model_id: str) -> _FakeModel:
            del model_id
            return _FakeModel()

    class _FakeBuilder:
        def build_provider(
            self, provider: str, auth: str, *, account: str | None = None
        ) -> _FakeProvider:
            del provider, auth, account
            return _FakeProvider()

    meta = SessionMeta(provider="Fake", model_id="fake-m", auth="env", account="me")
    with patch.object(session_io, "providers_lib", _FakeBuilder()):
        result = restore_model(meta)
    assert result is not None
    _, spec = result
    assert spec.provider == "Fake"
    assert spec.model_id == "fake-m"
    assert spec.account == "me"


def test_repair_synthesizes_missing_tool_result() -> None:
    """C2: orphan tool_use gets a synthetic ``[interrupted]`` placeholder."""
    asst = AssistantMessage(tool_calls=(ToolCall(id="c1", name="N", args={}),))
    history: list[ModelContextEvent] = [
        UserMessage(text="do X"),
        asst,
    ]
    repaired = repair_dangling_tool_calls(history)
    assert len(repaired) == 3
    last = repaired[-1]
    assert isinstance(last, ToolResult)
    assert last.call_id == "c1"
    assert last.content == "[interrupted]"
    assert last.is_error is True


def test_repair_is_idempotent() -> None:
    """C2: re-running the repair pass over its own output is a no-op."""
    asst = AssistantMessage(tool_calls=(ToolCall(id="c1", name="N", args={}),))
    history: list[ModelContextEvent] = [
        UserMessage(text="do X"),
        asst,
    ]
    repaired = repair_dangling_tool_calls(history)
    again = repair_dangling_tool_calls(repaired)
    assert [type(x) for x in again] == [type(x) for x in repaired]
    assert len(again) == len(repaired)


def test_repair_drops_orphan_tool_result_with_no_call() -> None:
    """C2: dangling ToolResult lacking a parent AssistantMessage is dropped."""
    orphan = ToolResult(call_id="ghost", content="leftover")
    history: list[ModelContextEvent] = [
        UserMessage(text="hi"),
        orphan,
    ]
    repaired = repair_dangling_tool_calls(history)
    assert len(repaired) == 1
    assert isinstance(repaired[0], UserMessage)


def test_repair_preserves_matching_tool_result_pair() -> None:
    """C2: existing tool_use + tool_result pair stays intact."""
    asst = AssistantMessage(tool_calls=(ToolCall(id="c1", name="N", args={}),))
    res = ToolResult(call_id="c1", content="OK")
    history: list[ModelContextEvent] = [
        UserMessage(text="hi"),
        asst,
        res,
    ]
    repaired = repair_dangling_tool_calls(history)
    assert repaired == history


def test_resume_does_not_eagerly_reread_touched_paths(tmp_path: Path) -> None:
    """Resume must not block on a touched path that migrated to a hung mount.

    Regression: resume used to eagerly re-read every Read/Edit/Write
    ``file_path`` from the tape to warm the staleness cache. When one of
    those paths now resolves onto a hung fuse.sshfs mount, the bare
    ``read_text`` blocks the resume thread forever -- a hard freeze before
    the REPL is interactive (session 82eb595a). A FIFO with no writer
    reproduces the same indefinite block without a network mount. Resume
    leaves ``_content_cache`` empty; content reloads lazily on first
    ``check_stale`` instead.
    """
    fifo = tmp_path / "hung.md"
    os.mkfifo(fifo)
    tape: list[TapeRecord] = [
        ReferrableTapeEvent(
            ref=TapeRef(session_id="s", ordinal=0),
            event=AssistantMessage(
                tool_calls=(
                    ToolCall(id="c1", name="Read", args={"file_path": str(fifo)}),
                ),
            ),
        ),
        ReferrableTapeEvent(
            ref=TapeRef(session_id="s", ordinal=1),
            event=ToolResult(call_id="c1", content="body"),
        ),
    ]
    agent = Agent(model=_NoopModel(), session_dir=tmp_path)

    done = threading.Event()

    def _run() -> None:
        agent.resume(SessionMeta(), tape, ToolState())
        done.set()

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    assert done.wait(timeout=5.0), (
        "resume blocked on a FIFO-touched path; a hung sshfs mount would "
        "hard-freeze resume before the REPL is interactive"
    )
    # Lazy by design: no eager content read of the touched path.
    resolved = str(Path(fifo).resolve())
    assert resolved not in agent.tool_state._content_cache


def test_mask_runs_empty_returns_empty() -> None:
    # SESSION-6: an empty ref list must not IndexError on ``ordered[0]``.
    assert _mask_runs([]) == ()


def test_mask_runs_groups_contiguous() -> None:
    refs = [
        TapeRef(session_id="s", ordinal=0),
        TapeRef(session_id="s", ordinal=1),
        TapeRef(session_id="s", ordinal=3),
    ]
    runs = _mask_runs(refs)
    assert runs == (
        MaskRange(session_id="s", lo=0, hi=1),
        MaskRange(session_id="s", lo=3, hi=3),
    )


def test_mask_from_json_drops_malformed_legacy_ranges(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Cross-session and negative-ordinal legacy ranges drop; valid ones survive.

    The old wire shape was two independent ``TapeRef`` endpoints, so a
    cross-session ``(s:0, legacy:1)`` or a negative-ordinal range was
    representable on disk. ``MaskRange`` makes both unconstructable, so
    ``_mask_from_json`` is the single boundary that normalizes them: each
    malformed range is dropped with a warning rather than crashing the load,
    and well-formed siblings in the same mask are kept.
    """
    raw = [
        # Cross-session: dropped.
        [
            {"session_id": "s", "ordinal": 0},
            {"session_id": "legacy", "ordinal": 1},
        ],
        # Negative lower ordinal: dropped.
        [
            {"session_id": "s", "ordinal": -1},
            {"session_id": "s", "ordinal": 3},
        ],
        # Inverted (also negative hi): dropped.
        [
            {"session_id": "s", "ordinal": 0},
            {"session_id": "s", "ordinal": -1},
        ],
        # Well-formed: survives.
        [
            {"session_id": "s", "ordinal": 4},
            {"session_id": "s", "ordinal": 6},
        ],
    ]
    with caplog.at_level("WARNING"):
        ranges = _mask_from_json(raw)
    assert ranges == (MaskRange(session_id="s", lo=4, hi=6),)
    # Each of the three malformed ranges logged a drop warning.
    assert (
        sum("dropping malformed legacy mask" in r.message for r in caplog.records) == 3
    )


def test_persisted_refs_warns_on_unreadable_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # SESSION-3: an existing-but-unreadable session file must warn, not
    # silently return empty (which would re-append the whole tape next save).
    session_file = tmp_path / "session.jsonl"
    session_file.write_text("{}\n", encoding="utf-8")
    with (
        patch.object(Path, "open", side_effect=OSError("boom")),
        caplog.at_level("WARNING"),
    ):
        refs = _persisted_refs(session_file)
    assert refs == set()
    assert any("persistence may duplicate" in r.message for r in caplog.records)


def test_thought_signature_round_trips(tmp_path: Path) -> None:
    """Gemini 3.x thought signatures must survive a session save/reload, else a
    resumed tape breaks the signature chain (400 on the next turn).
    """
    assistant = AssistantMessage(
        text="answer",
        thought_signature="sig-text-abc",
        tool_calls=(
            ToolCall(
                id="toolu_1",
                name="Bash",
                args={"cmd": "ls"},
                thought_signature="sig-fc-xyz",
            ),
        ),
    )
    result = ToolResult(call_id="toolu_1", content="ok")
    reloaded = _round_trip_history([assistant, result], tmp_path)[0]
    assert isinstance(reloaded, AssistantMessage)
    assert reloaded.thought_signature == "sig-text-abc"
    assert reloaded.tool_calls[0].thought_signature == "sig-fc-xyz"


def test_unpersisted_session_error_none_without_session_dir() -> None:
    # Persistence disabled (no session_dir): never an error.
    agent = Agent(model=_NoopModel(), session_dir=None)
    agent.runtime.append_history(UserMessage(text="x"))
    assert unpersisted_session_error(agent) is None


def test_unpersisted_session_error_none_for_empty_tape(tmp_path: Path) -> None:
    # Opened and quit without a turn: legitimately empty, nothing to save.
    agent = Agent(model=_NoopModel(), session_dir=tmp_path)
    assert not agent.runtime.tape
    assert not (tmp_path / "session.jsonl").exists()
    assert unpersisted_session_error(agent) is None


def test_unpersisted_session_error_none_when_persisted(tmp_path: Path) -> None:
    # Normal path: a turn happened and the observer wrote the transcript.
    agent = Agent(model=_NoopModel(), session_dir=tmp_path)
    agent.runtime.append_history(UserMessage(text="x"))
    agent.runtime.publish(SaveSession())
    assert (tmp_path / "session.jsonl").exists()
    assert unpersisted_session_error(agent) is None


def test_unpersisted_session_error_flags_silent_data_loss(tmp_path: Path) -> None:
    # The failure mode: tape has conversation but no transcript reached disk
    # (every persistence write was dropped). Must surface as an error.
    agent = Agent(model=_NoopModel(), session_dir=tmp_path)
    agent.runtime.append_history(UserMessage(text="lost work"))
    # No SaveSession published -> no session.jsonl, simulating a swallowed write.
    assert not (tmp_path / "session.jsonl").exists()
    msg = unpersisted_session_error(agent)
    assert msg is not None
    assert "cannot be resumed" in msg


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
