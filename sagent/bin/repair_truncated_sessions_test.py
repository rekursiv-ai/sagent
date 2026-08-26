"""Tests for the truncated-session repair tool."""

from __future__ import annotations

from collections.abc import Generator, Sequence
from pathlib import Path
from unittest.mock import patch

import contextlib
import json

from sagent.agent import session_io
from sagent.agent.context import resolve_context, validate_context
from sagent.agent.session_io import (
    SessionMeta,
    append_session,
    load_session,
)
from sagent.agent.state import ToolState
from sagent.bin import repair_truncated_sessions
from sagent.bin.repair_truncated_sessions import (
    main,
    poison_splices,
    repair_session,
)
from sagent.types.runtime import (
    AssistantMessage,
    ToolCall,
    UserMessage,
)
from sagent.types.tape import (
    ContextSplice,
    MaskRange,
    ReferrableTapeEvent,
    TapeRecord,
    TapeRef,
)


def _write_truncated_session(
    session_dir: Path, *, session_id: str = "poisoned"
) -> None:
    """Persist a session whose tape carries the truncating coalesce splice.

    Reproduces the on-disk shape the coalesce bug left behind: a barrier
    carrying the conversation, then a ``user_coalesce`` that absorbed the
    barrier's whole-tape mask while injecting only the merged user message.
    """
    refs = [TapeRef(session_id=session_id, ordinal=i) for i in range(5)]
    conversation = (
        UserMessage(text="the original question"),
        AssistantMessage(text="the original answer"),
        UserMessage(text="a follow-up"),
    )
    append_session(
        session_dir / "session.jsonl",
        meta=SessionMeta(session_id=session_id, model_id="m").serialize(),
        tape_delta=[
            ReferrableTapeEvent(ref=refs[0], event=conversation[0]),
            ReferrableTapeEvent(ref=refs[1], event=conversation[1]),
            ReferrableTapeEvent(ref=refs[2], event=conversation[2]),
            ContextSplice(
                ref=refs[3],
                mask=(MaskRange(session_id=session_id, lo=0, hi=2),),
                insert_after=None,
                payload=conversation,
                strategy="orphan_tool_result_repair",
            ),
            ContextSplice.replay(
                ref=refs[4],
                mask=(MaskRange(session_id=session_id, lo=0, hi=3),),
                insert_after=None,
                payload=(UserMessage(text="a follow-up\n\nnext"),),
                strategy="user_coalesce",
            ),
        ],
    )


def test_poison_splices_finds_the_truncating_coalesce(tmp_path: Path) -> None:
    _write_truncated_session(tmp_path)
    loaded = load_session(tmp_path)
    assert loaded is not None
    _, tape, _ = loaded

    found = poison_splices(tape)

    assert [s.ref.ordinal for s in found] == [4]


def _write_incident_shape(session_dir: Path, *, session_id: str = "incident") -> int:
    """Persist the REAL on-disk shape of the truncation, returning its size.

    The barrier the coalesce absorbed was synthesized inside ``load_session``
    and never written, so on disk the coalesce masks plain history records and
    the absorbed splice is absent entirely. A detector that compares payload
    lengths against the absorbed splice therefore sees nothing -- which is what
    happened on the first pass over the real session file.
    """
    count = 12
    refs = [TapeRef(session_id=session_id, ordinal=i) for i in range(count)]
    history = [
        ReferrableTapeEvent(
            ref=refs[i],
            event=(
                UserMessage(text=f"u{i}")
                if i % 2 == 0
                else AssistantMessage(text=f"a{i}")
            ),
        )
        for i in range(count)
    ]
    append_session(
        session_dir / "session.jsonl",
        meta=SessionMeta(session_id=session_id, model_id="m").serialize(),
        tape_delta=[
            *history,
            ContextSplice.replay(
                ref=TapeRef(session_id=session_id, ordinal=count),
                mask=(MaskRange(session_id=session_id, lo=0, hi=count - 1),),
                insert_after=None,
                payload=(UserMessage(text="the only survivor"),),
                strategy="user_coalesce",
            ),
        ],
    )
    return count


def test_poison_splices_finds_the_unpersisted_barrier_shape(tmp_path: Path) -> None:
    written = _write_incident_shape(tmp_path)
    loaded = load_session(tmp_path)
    assert loaded is not None
    _, tape, _ = loaded
    assert len(resolve_context(tape).messages) == 1, "fixture is not truncated"

    found = poison_splices(tape)

    assert [s.ref.ordinal for s in found] == [written]


def test_repair_restores_the_unpersisted_barrier_shape(tmp_path: Path) -> None:
    written = _write_incident_shape(tmp_path)

    assert repair_session(tmp_path)

    after = load_session(tmp_path)
    assert after is not None
    messages = resolve_context(after[1]).messages
    assert len(messages) == written
    validate_context(messages)


def test_poison_splices_ignores_a_lossless_barrier(tmp_path: Path) -> None:
    """A barrier that re-injects everything it masks is not poison.

    The load-time repair splice masks the whole tape by design and carries
    every message forward; flagging it would make the tool rewrite healthy
    sessions.
    """
    session_id = "healthy"
    refs = [TapeRef(session_id=session_id, ordinal=i) for i in range(2)]
    append_session(
        tmp_path / "session.jsonl",
        meta=SessionMeta(session_id=session_id, model_id="m").serialize(),
        tape_delta=[
            ReferrableTapeEvent(ref=refs[0], event=UserMessage(text="hi")),
            ReferrableTapeEvent(
                ref=refs[1],
                event=AssistantMessage(
                    tool_calls=(ToolCall(id="c1", name="Bash", args={}),)
                ),
            ),
        ],
    )
    loaded = load_session(tmp_path)
    assert loaded is not None
    _, tape, _ = loaded

    assert poison_splices(tape) == []


def test_repair_restores_the_masked_conversation(tmp_path: Path) -> None:
    _write_truncated_session(tmp_path)
    before = load_session(tmp_path)
    assert before is not None
    assert len(resolve_context(before[1]).messages) == 1, (
        "fixture is not truncated; the repair would prove nothing"
    )

    repaired = repair_session(tmp_path)

    assert repaired
    after = load_session(tmp_path)
    assert after is not None
    messages = resolve_context(after[1]).messages
    texts = [getattr(m, "text", "") for m in messages]
    assert "the original question" in texts
    assert "the original answer" in texts
    validate_context(messages)


def test_repair_writes_a_backup_before_modifying(tmp_path: Path) -> None:
    """The original bytes survive the repair, byte-for-byte."""
    _write_truncated_session(tmp_path)
    original = (tmp_path / "session.jsonl").read_bytes()

    _ = repair_session(tmp_path)

    backups = list(tmp_path.glob("session.jsonl.bak-*"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == original


def test_repair_is_idempotent(tmp_path: Path) -> None:
    """A second run finds nothing to do and leaves the file alone."""
    _write_truncated_session(tmp_path)
    assert repair_session(tmp_path)
    after_first = (tmp_path / "session.jsonl").read_bytes()

    assert not repair_session(tmp_path)
    assert (tmp_path / "session.jsonl").read_bytes() == after_first


def test_repair_does_not_reuse_an_ordinal_written_since_it_read(
    tmp_path: Path,
) -> None:
    """The tape may grow between the read and the write; minting must notice.

    The ordinal is computed from a snapshot, so an agent appending to the same
    live session in between makes both records claim one position. That is not
    theoretical -- running this tool against live session directories put
    duplicate refs into three of them.
    """
    _write_truncated_session(tmp_path)

    def racing_load(session_dir: Path) -> object:
        """Land a concurrent append between the tool's read and its write."""
        loaded = real_load(session_dir)
        if not raced and loaded is not None:
            raced.append(True)
            append_session(
                session_dir / "session.jsonl",
                tape_delta=[
                    ReferrableTapeEvent(
                        ref=TapeRef(
                            session_id="poisoned",
                            ordinal=max(r.ref.ordinal for r in loaded[1]) + 1,
                        ),
                        event=UserMessage(text="written concurrently"),
                    ),
                ],
            )
        return loaded

    raced: list[bool] = []
    real_load = session_io.load_session
    with patch.object(repair_truncated_sessions, "load_session", racing_load):
        assert repair_session(tmp_path)

    # Read the file, not the loader: the loader repairs a collision on the way
    # in, so asking it would grade the reader instead of the writer.
    refs = [
        (rec["ref"]["session_id"], rec["ref"]["ordinal"])
        for line in (tmp_path / "session.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if isinstance((rec := json.loads(line)).get("ref"), dict)
    ]
    assert len(set(refs)) == len(refs), f"repair collided with a live write; {refs}"


def test_the_mint_and_the_append_happen_under_one_lock(tmp_path: Path) -> None:
    """Choosing an ordinal and writing it must be one indivisible step.

    Re-reading immediately before the mint narrowed the window but left it
    open, and duplicate refs reached three real sessions through it. Asserted
    structurally rather than by racing threads: the collision needs the two
    writers to interleave on a sub-millisecond window, so a timing test
    reports "no bug" most runs and pins nothing.
    """
    _write_truncated_session(tmp_path)
    held: list[str] = []

    @contextlib.contextmanager
    def tracking_lock(path: Path) -> Generator[None]:
        del path
        held.append("acquire")
        try:
            yield
        finally:
            held.append("release")

    def tracking_load(
        session_dir: Path, *, preserve_corrupt: bool = True
    ) -> tuple[SessionMeta, list[TapeRecord], ToolState] | None:
        held.append("load")
        return load_session(session_dir, preserve_corrupt=preserve_corrupt)

    def tracking_append(
        path: Path, *, tape_delta: Sequence[TapeRecord] | None = None
    ) -> None:
        held.append("append")
        append_session(path, tape_delta=tape_delta)

    with (
        patch.object(repair_truncated_sessions, "session_file_lock", tracking_lock),
        patch.object(repair_truncated_sessions, "load_session", tracking_load),
        patch.object(repair_truncated_sessions, "append_session", tracking_append),
    ):
        assert repair_session(tmp_path)

    assert "acquire" in held, f"the repair never took the session lock; {held}"
    tail = held[held.index("acquire") :]
    assert tail[: tail.index("release")].count("append") == 1, (
        f"the append escaped the lock that guards its ordinal; {held}"
    )
    assert "load" in tail[: tail.index("release")], (
        f"the ordinal was read outside the lock that guards it; {held}"
    )


def test_dry_run_leaves_a_corrupt_session_byte_identical(tmp_path: Path) -> None:
    """``--dry-run`` must not write, including through the loader it calls.

    Loading is the tool's inspection step, but ``load_session`` copies the file
    aside on the first unparseable line -- so the flag that promises "without
    modifying them" wrote a backup for exactly the damaged sessions an operator
    runs a dry run against.
    """
    _write_truncated_session(tmp_path)
    with (tmp_path / "session.jsonl").open("a", encoding="utf-8") as handle:
        _ = handle.write("not json at all\n")
    before = sorted(p.name for p in tmp_path.iterdir())
    original = (tmp_path / "session.jsonl").read_bytes()

    _ = main(["--dry-run", str(tmp_path)])

    assert sorted(p.name for p in tmp_path.iterdir()) == before
    assert (tmp_path / "session.jsonl").read_bytes() == original


def test_repair_leaves_a_healthy_session_untouched(tmp_path: Path) -> None:
    session_id = "healthy"
    append_session(
        tmp_path / "session.jsonl",
        meta=SessionMeta(session_id=session_id, model_id="m").serialize(),
        tape_delta=[
            ReferrableTapeEvent(
                ref=TapeRef(session_id=session_id, ordinal=0),
                event=UserMessage(text="hi"),
            ),
        ],
    )
    original = (tmp_path / "session.jsonl").read_bytes()

    assert not repair_session(tmp_path)

    assert (tmp_path / "session.jsonl").read_bytes() == original
    assert list(tmp_path.glob("session.jsonl.bak-*")) == []


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
