"""Tests for the tape resolver and the provider-context validator."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, override

import itertools
import logging

import pytest

from sagent.model.scripted import call, reply
from sagent.model.types import Response
from sagent.runtime import context
from sagent.runtime.context import (
    InvalidContextError,
    Tape,
    alive_splices,
    masked_refs_by_alive,
    resolve_context,
    validate_context,
    widen_barrier_mask,
)
from sagent.runtime.tape import (
    ContextSplice,
    InvalidSpliceError,
    MaskRange,
    ReferrableTapeEvent,
    TapeRecord,
    TapeRef,
)
from sagent.runtime.types import (
    ModelContextEvent,
    SessionCompactCompleted,
    SessionCompacted,
    SessionCompactStarted,
    Stubs,
)
from sagent.session.types import (
    AgentMessage,
    AssistantMessage,
    ContextCompaction,
    Failed,
    SessionContext,
    Succeeded,
    SystemMessage,
    ToolMessage,
    UncategorizedToolResult,
    UserMessage,
    as_sent,
)


if TYPE_CHECKING:
    from collections.abc import Sequence


def _ref(ordinal: int, session: str = "s") -> TapeRef:
    return TapeRef(session_id=session, ordinal=ordinal)


def _one(ordinal: int, session: str = "s") -> MaskRange:
    return MaskRange(session_id=session, lo=ordinal, hi=ordinal)


def _event(
    ordinal: int,
    entry: ModelContextEvent,
    session: str = "s",
) -> ReferrableTapeEvent:
    return ReferrableTapeEvent(ref=_ref(ordinal, session), event=entry)


def _splice(
    ordinal: int,
    *,
    mask: tuple[MaskRange, ...] = (),
    insert_after: TapeRef | None = None,
    replacement: tuple[ModelContextEvent, ...] = (),
    paired_externally: frozenset[str] = frozenset(),
    session: str = "s",
) -> ContextSplice:
    return ContextSplice(
        ref=_ref(ordinal, session),
        mask=mask,
        insert_after=insert_after,
        replacement=replacement,
        strategy="test",
        paired_externally=paired_externally,
    )


def _user(text: str) -> UserMessage:
    return UserMessage(content=text)


def _calling(*ids: str) -> Response:
    return reply(calls=[call("echo", cid) for cid in ids])


def _texts(tape: Sequence[TapeRecord]) -> list[str]:
    out: list[str] = []
    for m in resolve_context(tape).messages:
        if isinstance(m, Response):
            out.append(
                "".join(
                    r.content or ""
                    for r in m.records
                    if isinstance(r, AssistantMessage)
                ),
            )
        elif isinstance(m, ToolMessage):
            out.append(m.text())
        elif isinstance(m, (UserMessage, AgentMessage, SystemMessage)):
            out.append(m.content or "")
    return out


def test_empty_tape_resolves_to_nothing() -> None:
    resolved = resolve_context([])
    assert (resolved.messages, resolved.origins, resolved.version) == ([], [], 0)


def test_events_render_in_tape_order_with_their_origins() -> None:
    tape = [_event(0, _user("a")), _event(1, reply("b")), _event(2, _user("c"))]
    resolved = resolve_context(tape)
    assert _texts(tape) == ["a", "b", "c"]
    assert resolved.origins == [_ref(0), _ref(1), _ref(2)]
    assert resolved.version == 3
    assert resolved.messages[0] is tape[0].event


def test_compaction_markers_render_nothing() -> None:
    tape: list[TapeRecord] = [
        _event(0, _user("a")),
        ReferrableTapeEvent(ref=_ref(1), event=SessionCompactStarted()),
        ReferrableTapeEvent(
            ref=_ref(2),
            event=SessionCompactCompleted(outcome=Succeeded(value=SessionCompacted())),
        ),
        ReferrableTapeEvent(
            ref=_ref(3),
            event=SessionCompactCompleted(
                outcome=Failed(exception=RuntimeError()),
                tape_len_before=1,
            ),
        ),
    ]
    assert _texts(tape) == ["a"]
    assert resolve_context(tape).version == 4


def test_a_splice_masks_and_inserts_after_its_anchor() -> None:
    tape = [
        _event(0, _user("a")),
        _event(1, _user("b")),
        _event(2, _user("c")),
        _splice(3, mask=(_one(1),), insert_after=_ref(0), replacement=(_user("B"),)),
    ]
    assert _texts(tape) == ["a", "B", "c"]
    assert resolve_context(tape).origins == [_ref(0), _ref(3), _ref(2)]


def test_a_mask_names_one_session() -> None:
    tape = [
        _event(0, _user("a0"), "a"),
        _event(0, _user("b0"), "b"),
        _splice(
            1,
            mask=(_one(0, "a"),),
            replacement=(_user("summary a"),),
            session="a",
        ),
    ]
    assert _texts(tape) == ["summary a", "b0"]


def test_a_splice_without_anchor_inserts_at_the_head() -> None:
    tape = [_event(0, _user("a")), _splice(1, replacement=(_user("HEAD"),))]
    assert _texts(tape) == ["HEAD", "a"]


def test_a_splice_anchored_off_tape_falls_to_the_head(
    caplog: pytest.LogCaptureFixture,
) -> None:
    tape = [
        _event(0, _user("a")),
        _splice(1, insert_after=_ref(9), replacement=(_user("X"),)),
    ]
    with caplog.at_level(logging.WARNING, logger=context.__name__):
        assert _texts(tape) == ["X", "a"]
    (record,) = caplog.records
    assert record.getMessage() == (
        f"resolver: splice {_ref(1)} insert_after={_ref(9)} not on tape;"
        " falling into HEAD"
    )


def test_an_anchor_appended_later_is_not_on_tape_yet() -> None:
    tape = [
        _event(0, _user("a")),
        _splice(1, insert_after=_ref(2), replacement=(_user("X"),)),
        _event(2, _user("b")),
    ]
    assert _texts(tape) == ["X", "a", "b"]


def test_an_empty_payload_deletes_and_an_empty_mask_inserts() -> None:
    deleted = [
        _event(0, _user("a")),
        _event(1, _user("b")),
        _splice(2, mask=(_one(1),)),
    ]
    assert _texts(deleted) == ["a"]
    inserted = [
        _event(0, _user("a")),
        _splice(1, insert_after=_ref(0), replacement=(_user("X"),)),
    ]
    assert _texts(inserted) == ["a", "X"]


def test_a_sparse_mask_removes_only_its_ranges() -> None:
    tape: list[TapeRecord] = [_event(i, _user(t)) for i, t in enumerate("abcde")]
    tape.append(
        _splice(
            5,
            mask=(_one(1), _one(3)),
            insert_after=_ref(0),
            replacement=(_user("S"),),
        ),
    )
    assert _texts(tape) == ["a", "S", "c", "e"]


def test_records_after_a_barrier_render_after_its_payload() -> None:
    tape = [
        _event(0, _user("a")),
        _event(1, _user("b")),
        _splice(
            2,
            mask=(MaskRange(session_id="s", lo=0, hi=1),),
            replacement=(_user("S"),),
        ),
        _event(3, _user("c")),
    ]
    assert _texts(tape) == ["S", "c"]


def test_masking_a_splice_undoes_its_mask() -> None:
    """F masks D, so D's mask of b lapses and b resurfaces."""
    tape = [
        _event(0, _user("a")),
        _event(1, _user("b")),
        _event(2, _user("c")),
        _splice(3, mask=(_one(1),), insert_after=_ref(0), replacement=(_user("D"),)),
        _event(4, _user("e")),
        _splice(5, mask=(_one(3),), insert_after=_ref(0), replacement=(_user("F"),)),
    ]
    assert _texts(tape) == ["a", "F", "b", "c", "e"]


def test_a_dead_splice_contributes_no_payload() -> None:
    tape = [
        _event(0, _user("a")),
        _splice(1, insert_after=_ref(0), replacement=(_user("X"),)),
        _splice(2, mask=(_one(1),), insert_after=_ref(0), replacement=(_user("Y"),)),
    ]
    assert _texts(tape) == ["a", "Y"]


def test_two_independent_splices_both_apply() -> None:
    tape: list[TapeRecord] = [_event(i, _user(t)) for i, t in enumerate("abcde")]
    tape.append(
        _splice(5, mask=(_one(3),), insert_after=_ref(2), replacement=(_user("F"),)),
    )
    tape.append(
        _splice(6, mask=(_one(1),), insert_after=_ref(0), replacement=(_user("G"),)),
    )
    assert _texts(tape) == ["a", "G", "c", "F", "e"]


def test_alive_splices_excludes_a_masked_splice() -> None:
    tape = [
        _event(0, _user("a")),
        _splice(1, insert_after=_ref(0), replacement=(_user("X"),)),
        _splice(2, mask=(_one(1),)),
    ]
    assert alive_splices(tape) == {_ref(2)}


def test_a_splice_killed_by_a_dead_splice_stays_alive() -> None:
    tape = [
        _event(0, _user("a")),
        _splice(1, mask=(_one(0),)),
        _splice(2, mask=(_one(1),)),
        _splice(3, mask=(_one(2),)),
    ]
    assert alive_splices(tape) == {_ref(1), _ref(3)}
    assert _texts(tape) == []


def test_masked_refs_by_alive_ignores_a_dead_splices_mask() -> None:
    tape = [
        _event(0, _user("a")),
        _event(1, _user("b")),
        _splice(2, mask=(_one(1),)),
        _splice(3, mask=(_one(2),)),
    ]
    masked = masked_refs_by_alive(tape, alive_splices(tape))
    assert masked == {_ref(2)}


@pytest.mark.parametrize(
    ("repeat", "stride", "first"),
    [
        (repeat, stride, first)
        for repeat, stride in ((2, 1), (3, 3))
        for first in [(lo, hi) for lo in range(7) for hi in range(lo, 7)][::stride]
    ],
)
def test_masked_refs_are_the_union_of_every_alive_mask(
    repeat: int,
    stride: int,
    first: tuple[int, int],
) -> None:
    """Merging intervals as they arrive, in any order, must lose no position."""
    spans = [(lo, hi) for lo in range(7) for hi in range(lo, 7)]
    for rest in itertools.product(spans[::stride], repeat=repeat - 1):
        masks = (first, *rest)
        tape: list[TapeRecord] = [_event(i, _user(str(i))) for i in range(8)]
        tape.extend(
            _splice(100 + n, mask=(MaskRange(session_id="s", lo=lo, hi=hi),))
            for n, (lo, hi) in enumerate(masks)
        )
        expected = {_ref(i) for lo, hi in masks for i in range(lo, hi + 1)}
        assert masked_refs_by_alive(tape, alive_splices(tape)) == expected


def test_mask_index_merges_touching_intervals() -> None:
    tape: list[TapeRecord] = [_event(i, _user(str(i))) for i in range(8)]
    tape.append(
        _splice(
            8,
            mask=(
                MaskRange(session_id="s", lo=4, hi=5),
                MaskRange(session_id="s", lo=0, hi=1),
                MaskRange(session_id="s", lo=2, hi=2),
            ),
        ),
    )
    assert _texts(tape) == ["3", "6", "7"]


def test_many_alive_splices_resolve() -> None:
    tape: list[TapeRecord] = []
    for i in range(500):
        tape.append(_event(2 * i, _user(str(i))))
        tape.append(_splice(2 * i + 1, mask=(_one(2 * i),)))
    assert len(alive_splices(tape)) == 500
    assert _texts(tape) == []


def test_the_message_list_is_fresh_each_resolve() -> None:
    tape = [_event(0, _user("a"))]
    resolve_context(tape).messages.append(_user("BOGUS"))
    assert len(resolve_context(tape).messages) == 1


def test_resolve_probes_each_anchor_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """A per-splice scan of the emitted order made the resolver quadratic."""
    reads = 0

    class CountingLinks(dict[TapeRef | None, TapeRef | None]):
        @override
        def __getitem__(self, key: TapeRef | None) -> TapeRef | None:
            nonlocal reads
            reads += 1
            return super().__getitem__(key)

    real_order = context._OrderedRefs

    def order() -> context._OrderedRefs:
        linked = real_order()
        linked._next = CountingLinks(linked._next)
        return linked

    monkeypatch.setattr(context, "_OrderedRefs", order)
    tape: list[TapeRecord] = []
    replies = [reply(f"a{i}") for i in range(3)]
    for i, response in enumerate(replies):
        tape.append(_event(2 * i, _user(f"u{i}")))
        tape.append(
            _splice(
                2 * i + 1,
                mask=(_one(2 * i),),
                insert_after=_ref(2 * i - 1) if i else None,
                replacement=(response,),
            ),
        )
    resolved = resolve_context(tape)
    assert resolved.messages == replies
    assert resolved.origins == [_ref(1), _ref(3), _ref(5)]
    assert reads <= 2 * len(tape) + 1


def test_a_duplicate_ref_is_refused() -> None:
    tape = [_event(0, _user("a")), _event(0, _user("b"))]
    with pytest.raises(InvalidContextError, match="duplicate tape ref"):
        _ = resolve_context(tape)


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [_user("hi")],
        [_user("hi"), reply("hello")],
        # A window may open on the model's turn, as after a summary.
        [reply("hello"), _user("hi")],
        [
            _user("go"),
            _calling("t1"),
            ToolMessage(
                call_id="t1",
                state="done",
                stdout=("ok",),
                result=UncategorizedToolResult(),
            ),
            reply("done"),
        ],
        [
            _user("go"),
            _calling("t1", "t2"),
            ToolMessage(
                call_id="t2",
                state="done",
                stdout=("ok",),
                result=UncategorizedToolResult(),
            ),
            ToolMessage(
                call_id="t1",
                state="done",
                stdout=("ok",),
                result=UncategorizedToolResult(),
            ),
        ],
        [_user("a"), AgentMessage(sender="A", content="b"), _user("c")],
        [_user("a"), SystemMessage(content="n"), _user("b"), reply("both")],
        [
            _user("q"),
            SystemMessage(content="cwd: /x"),
            _calling("c1"),
            SessionContext(cwd=Path("/x")),
            ToolMessage(
                call_id="c1",
                state="done",
                stdout=("ok",),
                result=UncategorizedToolResult(),
            ),
            SystemMessage(content="later"),
            reply("done"),
        ],
        # Every provider sends a note on the user side, so one parts two replies.
        [reply("a"), SystemMessage(content="n", subtype="nudge"), reply("b")],
        # A call run apart: its stub, its news, then its result, each sent as
        # the user's side; the news parts two replies.
        [
            _user("go"),
            _calling("t1"),
            ToolMessage(call_id="t1", state="pending", shown="[detached]"),
            reply("waiting"),
            ToolMessage(call_id="t1", state="pending", stdout=("half",)),
            reply("still waiting"),
            ToolMessage(call_id="t1", state="done", stdout=("all",)),
            reply("done"),
        ],
        # News of a call whose turn a compaction summarized away.
        [_user("[summary]"), ToolMessage(call_id="t1", state="done"), reply("ok")],
        [reply("a"), SessionContext(cwd=Path("/x")), reply("b")],
    ],
)
def test_valid_contexts_pass(messages: list[ModelContextEvent]) -> None:
    validate_context(messages)


@pytest.mark.parametrize(
    ("messages", "match"),
    [
        ([reply("a"), reply("b")], "^provider context violates role alternation$"),
        (
            [reply("a"), SystemMessage(), SessionContext(), reply("b")],
            "^provider context violates role alternation$",
        ),
        # News between a turn's results is a user turn before they are all in.
        (
            [
                _user("go"),
                _calling("t1", "t2"),
                ToolMessage(call_id="t1", state="pending", shown="[detached]"),
                ToolMessage(call_id="t1", state="done", stdout=("all",)),
                ToolMessage(call_id="t2", state="done", stdout=("ok",)),
            ],
            r"^user message before tool results: pending \['t2'\]$",
        ),
        # Once a call's result is in, nothing more of it is news.
        (
            [
                _user("go"),
                _calling("t1"),
                ToolMessage(call_id="t1", state="pending", shown="[detached]"),
                reply("waiting"),
                ToolMessage(call_id="t1", state="done", stdout=("all",)),
                reply("ok"),
                ToolMessage(call_id="t1", state="pending", stdout=("more",)),
            ],
            "^duplicate ToolMessage for call_id 't1'$",
        ),
        ([_user("go"), _calling("t1")], "without results at end"),
        ([_user("go"), _calling("t1"), reply("oops")], "pending tool calls"),
        (
            [
                _user("go"),
                _calling("t1"),
                _user("wait"),
                ToolMessage(
                    call_id="t1",
                    state="done",
                    stdout=("ok",),
                    result=UncategorizedToolResult(),
                ),
            ],
            "user message before",
        ),
        (
            [
                _user("go"),
                _calling("t1"),
                ToolMessage(
                    call_id="t1",
                    state="done",
                    stdout=("a",),
                    result=UncategorizedToolResult(),
                ),
                ToolMessage(
                    call_id="t1",
                    state="done",
                    stdout=("b",),
                    result=UncategorizedToolResult(),
                ),
            ],
            "^duplicate ToolMessage for call_id 't1'$",
        ),
    ],
)
def test_invalid_contexts_are_named(
    messages: list[ModelContextEvent],
    match: str,
) -> None:
    with pytest.raises(InvalidContextError, match=match):
        validate_context(messages)


@pytest.mark.parametrize(
    ("added", "runs"),
    [
        # Adjacent runs merge, before and after.
        ([(5, 6), (3, 4)], [(3, 6)]),
        ([(3, 4), (5, 6)], [(3, 6)]),
        # A one-ordinal gap keeps two runs.
        ([(3, 4), (6, 7)], [(3, 4), (6, 7)]),
        ([(6, 7), (3, 4)], [(3, 4), (6, 7)]),
        # A run spanning several swallows them.
        ([(1, 2), (4, 5), (7, 8), (2, 7)], [(1, 8)]),
        ([(1, 10), (3, 4)], [(1, 10)]),
    ],
)
def test_masked_ranges_are_kept_as_the_fewest_runs(
    added: list[tuple[int, int]],
    runs: list[tuple[int, int]],
) -> None:
    index = context._MaskIndex()
    for lo, hi in added:
        index.add_mask((MaskRange(session_id="s", lo=lo, hi=hi),))

    assert index.intervals_by_session == {"s": runs}


def test_a_call_s_later_messages_are_sent_as_the_user_s_side() -> None:
    stub = ToolMessage(call_id="t1", state="pending", shown="[detached]")
    entries: list[ModelContextEvent] = [
        _user("go"),
        _calling("t1"),
        stub,
        reply("waiting"),
        ToolMessage(
            call_id="t1",
            state="pending",
            stdout=("[Bash t1 still running after 9s]\nhalf",),
        ),
        reply("still waiting"),
        ToolMessage(call_id="t1", state="failed", stderr=("boom",)),
    ]

    sent = as_sent(entries)

    assert sent[:4] == entries[:4]
    assert sent[4:] == [
        UserMessage(content="[Bash t1 still running after 9s]\nhalf"),
        entries[5],
        UserMessage(content="[call t1 failed]\nboom"),
    ]
    validate_context(sent)


def test_a_compacted_tool_round_validates() -> None:
    tape = [
        _event(0, _user("go")),
        _event(1, _calling("t1")),
        _event(
            2,
            ToolMessage(
                call_id="t1",
                state="done",
                stdout=("ok",),
                result=UncategorizedToolResult(),
            ),
        ),
        _splice(
            3,
            mask=(MaskRange(session_id="s", lo=0, hi=2),),
            replacement=(_user("[summary]"),),
        ),
    ]
    validate_context(resolve_context(tape).messages)
    assert _texts(tape) == ["[summary]"]


def test_a_result_spliced_over_its_stub_validates() -> None:
    tape = [
        _event(0, _user("go")),
        _event(1, _calling("t1")),
        _event(
            2,
            ToolMessage(
                call_id="t1",
                state="done",
                stdout=(Stubs().detached,),
                result=UncategorizedToolResult(),
            ),
        ),
        _splice(
            3,
            mask=(_one(2),),
            insert_after=_ref(1),
            replacement=(
                ToolMessage(
                    call_id="t1",
                    state="done",
                    stdout=("real",),
                    result=UncategorizedToolResult(),
                ),
            ),
            paired_externally=frozenset({"t1"}),
        ),
    ]
    validate_context(resolve_context(tape).messages)
    assert _texts(tape) == ["go", "", "real"]


def _output_texts(entries: Sequence[ModelContextEvent]) -> list[str]:
    return [entry.text() for entry in entries if isinstance(entry, ToolMessage)]


def test_only_a_tool_s_output_may_be_stubbed() -> None:
    entries = [
        UserMessage(content="go"),
        reply(calls=[call("Read", "c0")]),
        ToolMessage(call_id="c0", state="done", stdout=("x" * 400,)),
        reply("done"),
    ]
    with pytest.raises(TypeError, match=r"^Entry 0 is no tool output\.$"):
        _ = context.stubbed(entries, [0], stub="gone")

    assert context.stubbed(entries, [], stub="gone") == entries
    assert _output_texts(context.stubbed(entries, [2], stub="gone")) == ["gone"]
    # The entries given are left as they were.
    assert _output_texts(entries) == ["x" * 400]


def test_a_stubbed_output_keeps_the_line_that_says_which_call_and_what_became_of_it() -> (
    None
):
    entries = [
        reply(calls=[call("Bash", c) for c in ("c0", "c1", "c2")]),
        ToolMessage(call_id="c0", state="pending", shown=f"[detached]\n{'x' * 400}"),
        ToolMessage(call_id="c1", state="cancelled", shown=f"[cancelled]\n{'x' * 400}"),
        ToolMessage(call_id="c2", state="done", stdout=("x" * 400,)),
        reply("waiting"),
        ToolMessage(
            call_id="c0",
            state="pending",
            stdout=("[talk c0 still running after 15s]\none\ntwo",),
        ),
        ToolMessage(call_id="c0", state="pending"),
    ]

    after = context.stubbed(entries, [1, 2, 3, 5, 6], stub="[gone]")

    assert _output_texts(after) == [
        "[detached]\n[gone]",
        "[cancelled]\n[gone]",
        "[gone]",
        "[talk c0 still running after 15s]\n[gone]",
        "\n[gone]",
    ]
    states = [e.state for e in after if isinstance(e, ToolMessage)]
    assert states == ["pending", "cancelled", "done", "pending", "pending"]


def test_a_splice_may_reclaim_what_a_dead_splice_masked() -> None:
    tape = Tape("s")
    tape.append_history(_user("u0"))
    first = tape.append_splice(
        mask=(MaskRange(session_id="s", lo=0, hi=0),),
        strategy="x",
    )
    tape.append_splice(
        mask=(MaskRange(session_id="s", lo=first.ordinal, hi=first.ordinal),),
        strategy="undo",
    )
    assert _texts(tape) == ["u0"]
    tape.append_splice(
        mask=(MaskRange(session_id="s", lo=0, hi=0),),
        strategy="again",
    )
    assert _texts(tape) == []


def test_appends_mint_consecutive_refs_in_the_session() -> None:
    tape = Tape("s")
    first = tape.append_history(_user("a"))
    second = tape.append_splice(
        insert_after=first,
        replacement=(reply("b"),),
        strategy="x",
    )
    assert (first, second) == (
        TapeRef(session_id="s", ordinal=0),
        TapeRef(session_id="s", ordinal=1),
    )
    assert tape._by_ref.get(first) == ReferrableTapeEvent(
        ref=first,
        event=_user("a"),
    )
    assert tape._by_ref.get(TapeRef(session_id="s", ordinal=9)) is None


def test_context_is_memoized_until_the_next_append() -> None:
    tape = Tape("s")
    tape.append_history(_user("a"))
    first = tape.context()
    assert tape.context() is first
    tape.append_history(reply("b"))
    second = tape.context()
    assert second is not first
    assert second.version == 2
    tape.append_clear()
    assert tape.context().messages == []


def test_a_splice_may_not_claim_a_position_another_holds() -> None:
    tape = Tape("s")
    tape.append_history(_user("a"))
    tape.append_history(_user("b"))
    tape.append_splice(
        mask=(MaskRange(session_id="s", lo=1, hi=1),),
        strategy="x",
    )
    with pytest.raises(InvalidSpliceError, match="overlaps alive splice"):
        _ = tape.append_splice(
            mask=(MaskRange(session_id="s", lo=0, hi=1),),
            strategy="y",
        )


def test_a_splice_may_claim_a_position_after_absorbing_its_holder() -> None:
    tape = Tape("s")
    tape.append_history(_user("a"))
    tape.append_history(_user("b"))
    holder = tape.append_splice(
        mask=(MaskRange(session_id="s", lo=1, hi=1),),
        strategy="x",
    )
    tape.append_splice(
        mask=(MaskRange(session_id="s", lo=1, hi=holder.ordinal),),
        strategy="y",
    )
    assert _texts(tape) == ["a"]


def test_a_splice_may_not_anchor_inside_its_own_mask() -> None:
    tape = Tape("s")
    tape.append_splice(
        mask=(MaskRange(session_id="A", lo=0, hi=10),),
        insert_after=TapeRef(session_id="B", ordinal=5),
        strategy="other session",
    )
    with pytest.raises(InvalidSpliceError, match="inside this splice's own mask"):
        _ = tape.append_splice(
            mask=(MaskRange(session_id="A", lo=20, hi=30),),
            insert_after=TapeRef(session_id="A", ordinal=25),
            strategy="same session",
        )


def _barrier(tape: Tape, *replacement: ModelContextEvent) -> TapeRef:
    tape.append_history(_user("u0"))
    return tape.append_splice(
        mask=(MaskRange(session_id="s", lo=0, hi=0),),
        replacement=replacement,
        strategy="summary",
    )


def test_absorbing_a_splice_must_carry_its_text() -> None:
    tape = Tape("s")
    barrier = _barrier(tape, _user("must survive"), reply("b"))
    mask = (MaskRange(session_id="s", lo=0, hi=barrier.ordinal),)
    with pytest.raises(InvalidSpliceError, match="replacement drops 2 entry"):
        _ = tape.append_splice(
            mask=mask,
            replacement=(_user("unrelated"), reply("other")),
            strategy="careless",
        )
    with pytest.raises(InvalidSpliceError, match="replacement drops"):
        tape.adopt_record(
            ContextSplice(
                ref=tape.mint_ref(),
                mask=mask,
                insert_after=None,
                replacement=(_user("x"),),
                strategy="c",
            ),
        )
    assert _texts(tape) == ["must survive", "b"]


def test_absorbing_a_vendor_s_summary_must_carry_its_text() -> None:
    tape = Tape("s")
    summary = ContextCompaction(summary="the gist", provider="anthropic")
    barrier = _barrier(tape, summary, _user("next"))
    mask = (MaskRange(session_id="s", lo=0, hi=barrier.ordinal),)
    with pytest.raises(InvalidSpliceError, match="first: 'the gist'"):
        _ = tape.append_splice(mask=mask, replacement=(_user("next"),), strategy="c")


def test_absorbing_a_splice_while_carrying_it_is_allowed() -> None:
    tape = Tape("s")
    barrier = _barrier(tape, _user("a"), reply("b"))
    tape.append_splice(
        mask=(MaskRange(session_id="s", lo=0, hi=barrier.ordinal),),
        replacement=(_user("a"), reply("b"), _user("more")),
        strategy="careful",
    )
    assert _texts(tape) == ["a", "b", "more"]


def test_a_declared_discard_may_replace_what_it_absorbs() -> None:
    tape = Tape("s")
    barrier = _barrier(tape, _user("a"), reply("b"))
    mask = (MaskRange(session_id="s", lo=0, hi=barrier.ordinal),)
    tape.append_splice(
        mask=mask,
        replacement=(_user("[summary]"),),
        strategy="summary",
        discards_content=True,
    )
    assert _texts(tape) == ["[summary]"]


def test_an_empty_absorber_needs_no_carry() -> None:
    tape = Tape("s")
    _barrier(tape, _user("a"), reply("b"))
    tape.append_clear()
    assert tape.context().messages == []


def test_adopt_record_refuses_an_overlapping_mask() -> None:
    tape = Tape("s")
    tape.append_history(_user("a"))
    tape.append_splice(
        mask=(MaskRange(session_id="s", lo=0, hi=0),),
        strategy="x",
    )
    with pytest.raises(InvalidSpliceError, match="overlaps alive splice"):
        tape.adopt_record(
            ContextSplice(
                ref=tape.mint_ref(),
                mask=(MaskRange(session_id="s", lo=0, hi=0),),
                insert_after=None,
                replacement=(),
                strategy="y",
            ),
        )


def test_clearing_an_empty_tape_marks_it() -> None:
    tape = Tape("s")
    ref = tape.append_clear()
    record = tape._by_ref.get(ref)
    assert isinstance(record, ContextSplice)
    assert (record.mask, record.strategy) == ((), "clear")


def test_adopting_a_ref_already_on_the_tape_is_refused() -> None:
    tape = Tape("s")
    taken = tape.append_history(_user("a"))
    with pytest.raises(InvalidContextError, match="duplicate tape ref"):
        tape.adopt_record(ReferrableTapeEvent(ref=taken, event=_user("b")))
    assert _texts(tape) == ["a"]


def test_append_history_keeps_the_entry_itself() -> None:
    tape = Tape("s")
    entry = _user("hi")
    tape.append_history(entry)
    (seen,) = tape.context().messages
    assert seen is entry


def test_clearing_a_taped_session_returns_its_barrier() -> None:
    tape = Tape("s")
    tape.append_history(_user("a"))
    tape.append_history(reply("b"))
    ref = tape.append_clear()
    assert ref == TapeRef(session_id="s", ordinal=2)
    record = tape._by_ref.get(ref)
    assert isinstance(record, ContextSplice)
    assert (record.strategy, record.mask, record.replacement) == (
        "clear",
        (MaskRange(session_id="s", lo=0, hi=1),),
        (),
    )
    assert tape.context().messages == []


def _tape(*refs: TapeRef) -> tuple[TapeRecord, ...]:
    return tuple(ReferrableTapeEvent(ref=r, event=_user(str(r.ordinal))) for r in refs)


def _summary_splice(
    *mask: MaskRange,
    session: str = "s",
    ordinal: int = 9,
) -> ContextSplice:
    return ContextSplice(
        ref=TapeRef(session_id=session, ordinal=ordinal),
        mask=mask,
        insert_after=None,
        replacement=(_user("summary"),),
        strategy="summary",
    )


def test_widening_preserves_gaps_in_the_mask() -> None:
    tape = _tape(*(TapeRef(session_id="s", ordinal=i) for i in range(4)))
    override = _summary_splice(
        MaskRange(session_id="s", lo=0, hi=0),
        MaskRange(session_id="s", lo=2, hi=2),
    )
    widened = widen_barrier_mask(override, tape)
    assert widened.mask == (
        MaskRange(session_id="s", lo=0, hi=0),
        MaskRange(session_id="s", lo=2, hi=3),
    )
    assert widened.replacement == override.replacement


def test_widening_partitions_by_session() -> None:
    tape = _tape(
        TapeRef(session_id="", ordinal=0),
        TapeRef(session_id="", ordinal=1),
        TapeRef(session_id="sess-2", ordinal=2),
    )
    widened = widen_barrier_mask(
        _summary_splice(MaskRange(session_id="", lo=0, hi=0)),
        tape,
    )
    assert set(widened.mask) == {
        MaskRange(session_id="", lo=0, hi=1),
        MaskRange(session_id="sess-2", lo=2, hi=2),
    }


def test_widening_an_empty_mask_covers_the_tape() -> None:
    tape = _tape(TapeRef(session_id="s", ordinal=0), TapeRef(session_id="s", ordinal=1))
    assert widen_barrier_mask(_summary_splice(), tape).mask == (
        MaskRange(session_id="s", lo=0, hi=1),
    )


def test_widening_a_covering_mask_returns_the_override() -> None:
    tape = _tape(TapeRef(session_id="s", ordinal=0), TapeRef(session_id="s", ordinal=1))
    override = _summary_splice(MaskRange(session_id="s", lo=0, hi=1))
    assert widen_barrier_mask(override, tape) is override
    empty = _summary_splice()
    assert widen_barrier_mask(empty, ()) is empty


def test_adopting_a_taped_event_appends_it() -> None:
    tape = Tape("s")
    ref = tape.mint_ref()
    tape.adopt_record(ReferrableTapeEvent(ref=ref, event=_user("a")))
    assert _texts(tape) == ["a"]
    assert tape._by_ref.get(ref) == ReferrableTapeEvent(
        ref=ref,
        event=_user("a"),
    )


def test_absorbing_a_splice_must_carry_its_results_and_settings() -> None:
    tape = Tape("s")
    tape.append_history(_user("u0"))
    barrier = tape.append_splice(
        mask=(MaskRange(session_id="s", lo=0, hi=0),),
        replacement=(
            _user("a"),
            reply(calls=[call("echo", "c1")]),
            ToolMessage(
                call_id="c1",
                state="done",
                stdout=("tool says",),
                result=UncategorizedToolResult(),
            ),
            SessionContext(cwd=Path("/x")),
        ),
        strategy="summary",
    )
    mask = (MaskRange(session_id="s", lo=0, hi=barrier.ordinal),)
    with pytest.raises(InvalidSpliceError, match="tool says"):
        _ = tape.append_splice(
            mask=mask,
            replacement=(
                _user("a"),
                reply(calls=[call("echo", "c1")]),
                ToolMessage(
                    call_id="c1",
                    state="done",
                    stdout=("other",),
                    result=UncategorizedToolResult(),
                ),
                SessionContext(cwd=Path("/x")),
            ),
            strategy="careless",
        )
    with pytest.raises(InvalidSpliceError, match="Working directory: /x"):
        _ = tape.append_splice(
            mask=mask,
            replacement=(
                _user("a"),
                reply(calls=[call("echo", "c1")]),
                ToolMessage(
                    call_id="c1",
                    state="done",
                    stdout=("tool says",),
                    result=UncategorizedToolResult(),
                ),
            ),
            strategy="careless",
        )


def test_widening_skips_runs_of_gap_positions() -> None:
    tape = _tape(*(TapeRef(session_id="s", ordinal=i) for i in range(4)))
    override = _summary_splice(
        MaskRange(session_id="s", lo=0, hi=0),
        MaskRange(session_id="s", lo=3, hi=3),
    )
    assert widen_barrier_mask(override, tape).mask == override.mask
    shorter = _tape(*(TapeRef(session_id="s", ordinal=i) for i in range(3)))
    wide_gap = _summary_splice(
        MaskRange(session_id="s", lo=0, hi=0),
        MaskRange(session_id="s", lo=9, hi=9),
    )
    assert widen_barrier_mask(wide_gap, shorter).mask == (
        MaskRange(session_id="s", lo=0, hi=0),
    )


def test_a_splice_keeps_what_its_producer_says_of_it() -> None:
    tape = Tape("s")
    tape.append_history(_user("u0"))
    ref = tape.append_splice(
        mask=(MaskRange(session_id="s", lo=0, hi=0),),
        replacement=(_user("summary"),),
        strategy="summary",
        masked_tokens=90,
        replacement_tokens=9,
        summary_fallback_reason="model refused",
        entries_kept_verbatim=2,
    )
    record = tape._by_ref.get(ref)
    assert isinstance(record, ContextSplice)
    assert (
        record.strategy,
        record.masked_tokens,
        record.replacement_tokens,
        record.summary_fallback_reason,
        record.entries_kept_verbatim,
    ) == ("summary", 90, 9, "model refused", 2)
    bare = tape._by_ref.get(tape.append_splice(insert_after=ref))
    assert isinstance(bare, ContextSplice)
    assert (
        bare.strategy,
        bare.masked_tokens,
        bare.replacement_tokens,
        bare.entries_kept_verbatim,
    ) == ("", 0, 0, 0)


def test_absorbing_checks_every_absorbed_splice() -> None:
    tape = Tape("s")
    tape.append_history(_user("u0"))
    tape.append_splice(
        insert_after=TapeRef(session_id="s", ordinal=0),
        replacement=(reply("kept"),),
    )
    barrier = tape.append_splice(
        mask=(MaskRange(session_id="s", lo=0, hi=0),),
        replacement=(_user("must survive"),),
        strategy="summary",
    )
    with pytest.raises(InvalidSpliceError, match="must survive"):
        _ = tape.append_splice(
            mask=(MaskRange(session_id="s", lo=barrier.ordinal, hi=barrier.ordinal),),
            replacement=(_user("other"),),
        )


def test_an_absorbed_splice_does_not_excuse_overlapping_another() -> None:
    tape = Tape("s")
    for text in ("a", "b"):
        tape.append_history(_user(text))
    first = tape.append_splice(mask=(MaskRange(session_id="s", lo=0, hi=0),))
    tape.append_splice(mask=(MaskRange(session_id="s", lo=1, hi=1),))
    with pytest.raises(InvalidSpliceError, match="overlaps alive splice"):
        _ = tape.append_splice(
            mask=(
                MaskRange(session_id="s", lo=1, hi=1),
                MaskRange(session_id="s", lo=first.ordinal, hi=first.ordinal),
            ),
        )


def _strategies(tape: Tape) -> list[str]:
    return [r.strategy for r in tape if isinstance(r, ContextSplice)]


def test_a_valid_tape_needs_no_repair() -> None:
    tape = Tape("s")
    tape.append_history(_user("go"))
    tape.append_history(_calling("c1"))
    tape.append_history(ToolMessage(call_id="c1", state="done", stdout=("ok",)))

    assert tape.repair() is None
    assert _strategies(tape) == []


def test_a_recorded_call_left_open_is_answered_interrupted_beside_its_turn() -> None:
    tape = Tape("s")
    tape.append_history(_user("go"))
    turn = tape.append_history(_calling("c1", "c2"))
    tape.append_history(ToolMessage(call_id="c1", state="done", stdout=("ok",)))
    tape.append_history(_user("next"))

    assert tape.repair() is None

    (repair,) = (r for r in tape if isinstance(r, ContextSplice))
    assert (repair.strategy, repair.insert_after) == ("orphan_tool_use_repair", turn)
    assert repair.paired_externally == frozenset({"c2"})
    assert _texts(tape) == ["go", "", "[interrupted]", "ok", "next"]
    validate_context(tape.context().messages)


def test_a_recorded_call_open_at_the_end_is_answered_interrupted() -> None:
    tape = Tape("s")
    tape.append_history(_user("go"))
    tape.append_history(_calling("c1"))

    assert tape.repair() is None

    assert _strategies(tape) == ["orphan_tool_use_repair"]
    assert _texts(tape) == ["go", "", "[interrupted]"]


def test_a_recorded_repeat_of_a_result_is_hidden() -> None:
    tape = Tape("s")
    tape.append_history(_user("go"))
    tape.append_history(_calling("c1"))
    tape.append_history(ToolMessage(call_id="c1", state="done", stdout=("ok",)))
    tape.append_history(ToolMessage(call_id="c1", state="done", stdout=("again",)))

    assert tape.repair() is None

    assert _strategies(tape) == ["orphan_tool_result_repair"]
    assert _texts(tape) == ["go", "", "ok"]


def test_splices_that_leave_partners_unpaired_are_rebuilt_whole() -> None:
    """Each splice is valid; together they leave a call and a result unpaired."""
    tape = Tape("s")
    tape.append_history(_user("dropped"))
    tape.append_splice(
        replacement=(_user("[summary]"), _calling("toolu_X")),
        strategy="summary",
        paired_externally=frozenset({"toolu_X"}),
    )
    tape.append_splice(
        replacement=(ToolMessage(call_id="ghost", state="done", stdout=("dangling",)),),
        strategy="lazy_pairing",
        paired_externally=frozenset({"ghost"}),
    )

    assert tape.repair() is None

    # No repair anchors at a splice: what a splice shows is the rebuild's.
    assert _strategies(tape) == ["summary", "lazy_pairing", "context_rescue"]
    validate_context(tape.context().messages)
    assert "[interrupted]" in _texts(tape)


def test_a_tape_where_two_records_claim_one_ref_is_left_unrepaired() -> None:
    """A ref adopted ahead of the count is minted again by the next append."""
    tape = Tape("s")
    tape.adopt_record(_event(0, _user("adopted")))
    tape.append_history(_user("minted"))

    error = tape.repair()

    assert isinstance(error, InvalidContextError)
    assert "duplicate tape ref" in str(error)


def test_a_declared_discard_may_be_adopted_over_what_it_absorbs() -> None:
    tape = Tape("s")
    barrier = _barrier(tape, _user("a"), reply("b"))
    tape.adopt_record(
        ContextSplice(
            ref=tape.mint_ref(),
            mask=(MaskRange(session_id="s", lo=0, hi=barrier.ordinal),),
            insert_after=None,
            replacement=(_user("[summary]"),),
            strategy="summary",
        ),
        discards_content=True,
    )
    assert _texts(tape) == ["[summary]"]


def test_a_tape_reads_as_the_sequence_of_its_records() -> None:
    tape = Tape("s")
    first = tape.append_history(_user("a"))
    tape.append_history(_user("b"))

    assert len(tape) == 2
    assert tape[0] == ReferrableTapeEvent(ref=first, event=_user("a"))
    assert list(tape[1:]) == [tape[1]]
    assert list(tape) == [tape[0], tape[1]]


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
