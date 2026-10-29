"""Tests for the tape resolver and the provider-context validator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import itertools
import logging

import pytest

from sagent.model.scripted import call, reply
from sagent.model.types import Response
from sagent.runtime import context
from sagent.runtime.context import (
    InvalidContextError,
    alive_splices,
    masked_refs_by_alive,
    resolve_context,
    validate_context,
)
from sagent.runtime.tape import (
    ContextSplice,
    MaskRange,
    ReferrableTapeEvent,
    TapeRecord,
    TapeRef,
)
from sagent.runtime.types import (
    CompactComplete,
    CompactFailed,
    CompactStarted,
    Markers,
    ModelContextEvent,
)
from sagent.session.types import (
    AgentToAgentMessage,
    AssistantMessage,
    SystemMessage,
    TurnContext,
    UncategorizedToolResult,
    UserMessage,
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
    payload: tuple[ModelContextEvent, ...] = (),
    paired_externally: frozenset[str] = frozenset(),
    session: str = "s",
) -> ContextSplice:
    return ContextSplice(
        ref=_ref(ordinal, session),
        mask=mask,
        insert_after=insert_after,
        payload=payload,
        strategy="test",
        paired_externally=paired_externally,
    )


def _user(text: str) -> UserMessage:
    return UserMessage(content=text)


def _result(call_id: str, text: str = "ok") -> UncategorizedToolResult:
    return UncategorizedToolResult(call_id=call_id, content=text)


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
        elif isinstance(m, UncategorizedToolResult):
            out.append(m.text())
        elif isinstance(m, (UserMessage, AgentToAgentMessage, SystemMessage)):
            out.append(m.content or "")
    return out


# --- Resolver ------------------------------------------------------------


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
        ReferrableTapeEvent(ref=_ref(1), event=CompactStarted()),
        ReferrableTapeEvent(ref=_ref(2), event=CompactComplete()),
        ReferrableTapeEvent(
            ref=_ref(3),
            event=CompactFailed(exception=RuntimeError(), tape_len=1),
        ),
    ]
    assert _texts(tape) == ["a"]
    assert resolve_context(tape).version == 4


def test_a_splice_masks_and_inserts_after_its_anchor() -> None:
    tape = [
        _event(0, _user("a")),
        _event(1, _user("b")),
        _event(2, _user("c")),
        _splice(3, mask=(_one(1),), insert_after=_ref(0), payload=(_user("B"),)),
    ]
    assert _texts(tape) == ["a", "B", "c"]
    assert resolve_context(tape).origins == [_ref(0), _ref(3), _ref(2)]


def test_a_mask_names_one_session() -> None:
    tape = [
        _event(0, _user("a0"), "a"),
        _event(0, _user("b0"), "b"),
        _splice(1, mask=(_one(0, "a"),), payload=(_user("summary a"),), session="a"),
    ]
    assert _texts(tape) == ["summary a", "b0"]


def test_a_splice_without_anchor_inserts_at_the_head() -> None:
    tape = [_event(0, _user("a")), _splice(1, payload=(_user("HEAD"),))]
    assert _texts(tape) == ["HEAD", "a"]


def test_a_splice_anchored_off_tape_falls_to_the_head(
    caplog: pytest.LogCaptureFixture,
) -> None:
    tape = [
        _event(0, _user("a")),
        _splice(1, insert_after=_ref(9), payload=(_user("X"),)),
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
        _splice(1, insert_after=_ref(2), payload=(_user("X"),)),
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
        _splice(1, insert_after=_ref(0), payload=(_user("X"),)),
    ]
    assert _texts(inserted) == ["a", "X"]


def test_a_sparse_mask_removes_only_its_ranges() -> None:
    tape: list[TapeRecord] = [_event(i, _user(t)) for i, t in enumerate("abcde")]
    tape.append(
        _splice(
            5,
            mask=(_one(1), _one(3)),
            insert_after=_ref(0),
            payload=(_user("S"),),
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
            payload=(_user("S"),),
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
        _splice(3, mask=(_one(1),), insert_after=_ref(0), payload=(_user("D"),)),
        _event(4, _user("e")),
        _splice(5, mask=(_one(3),), insert_after=_ref(0), payload=(_user("F"),)),
    ]
    assert _texts(tape) == ["a", "F", "b", "c", "e"]


def test_a_dead_splice_contributes_no_payload() -> None:
    tape = [
        _event(0, _user("a")),
        _splice(1, insert_after=_ref(0), payload=(_user("X"),)),
        _splice(2, mask=(_one(1),), insert_after=_ref(0), payload=(_user("Y"),)),
    ]
    assert _texts(tape) == ["a", "Y"]


def test_two_independent_splices_both_apply() -> None:
    tape: list[TapeRecord] = [_event(i, _user(t)) for i, t in enumerate("abcde")]
    tape.append(
        _splice(5, mask=(_one(3),), insert_after=_ref(2), payload=(_user("F"),)),
    )
    tape.append(
        _splice(6, mask=(_one(1),), insert_after=_ref(0), payload=(_user("G"),)),
    )
    assert _texts(tape) == ["a", "G", "c", "F", "e"]


def test_alive_splices_excludes_a_masked_splice() -> None:
    tape = [
        _event(0, _user("a")),
        _splice(1, insert_after=_ref(0), payload=(_user("X"),)),
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


def test_masked_refs_are_the_union_of_every_alive_mask() -> None:
    """Merging intervals as they arrive, in any order, must lose no position."""
    spans = [(lo, hi) for lo in range(7) for hi in range(lo, 7)]
    for masks in itertools.chain(
        itertools.product(spans, repeat=2),
        itertools.product(spans[::3], repeat=3),
    ):
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
    probes = 0
    real_contains = context._OrderedRefs.contains

    def counting(order: context._OrderedRefs, ref: TapeRef) -> bool:
        nonlocal probes
        probes += 1
        return real_contains(order, ref)

    monkeypatch.setattr(context._OrderedRefs, "contains", counting)
    tape: list[TapeRecord] = []
    for i in range(100):
        tape.append(_event(2 * i, _user(f"u{i}")))
        anchor = _ref(2 * i - 1) if i else None
        tape.append(
            _splice(
                2 * i + 1,
                mask=(_one(2 * i),),
                insert_after=anchor,
                payload=(reply(f"a{i}"),),
            ),
        )
    _ = resolve_context(tape)
    assert probes <= len(tape)


def test_a_duplicate_ref_is_refused() -> None:
    tape = [_event(0, _user("a")), _event(0, _user("b"))]
    with pytest.raises(InvalidContextError, match="duplicate tape ref"):
        _ = resolve_context(tape)


# --- Validator -------------------------------------------------------------


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [_user("hi")],
        [_user("hi"), reply("hello")],
        [_user("go"), _calling("t1"), _result("t1"), reply("done")],
        [_user("go"), _calling("t1", "t2"), _result("t2"), _result("t1")],
        [_user("a"), AgentToAgentMessage(sender="A", content="b"), _user("c")],
        [_user("a"), SystemMessage(content="n"), _user("b"), reply("both")],
        [
            _user("q"),
            SystemMessage(content="cwd: /x"),
            _calling("c1"),
            TurnContext(cwd="/x"),
            _result("c1"),
            SystemMessage(content="later"),
            reply("done"),
        ],
    ],
)
def test_valid_contexts_pass(messages: list[ModelContextEvent]) -> None:
    validate_context(messages)


@pytest.mark.parametrize(
    ("messages", "match"),
    [
        ([reply("a"), reply("b")], "^provider context violates role alternation$"),
        (
            [reply("a"), SystemMessage(content="n"), reply("b")],
            "^provider context violates role alternation$",
        ),
        ([_result("t1")], "orphan ToolResult"),
        (
            [_user("go"), _calling("t1"), _result("t1"), _result("t2")],
            "orphan ToolResult",
        ),
        ([_user("go"), _calling("t1")], "without results at end"),
        ([_user("go"), _calling("t1"), reply("oops")], "pending tool calls"),
        (
            [_user("go"), _calling("t1"), _user("wait"), _result("t1")],
            "user message before",
        ),
        (
            [_user("go"), _calling("t1"), _result("t1", "a"), _result("t1", "b")],
            "duplicate ToolResult",
        ),
    ],
)
def test_invalid_contexts_are_named(
    messages: list[ModelContextEvent],
    match: str,
) -> None:
    with pytest.raises(InvalidContextError, match=match):
        validate_context(messages)


def test_a_compacted_tool_round_validates() -> None:
    tape = [
        _event(0, _user("go")),
        _event(1, _calling("t1")),
        _event(2, _result("t1")),
        _splice(
            3,
            mask=(MaskRange(session_id="s", lo=0, hi=2),),
            payload=(_user("[summary]"),),
        ),
    ]
    validate_context(resolve_context(tape).messages)
    assert _texts(tape) == ["[summary]"]


def test_a_result_spliced_over_its_stub_validates() -> None:
    tape = [
        _event(0, _user("go")),
        _event(1, _calling("t1")),
        _event(2, _result("t1", Markers().detached)),
        _splice(
            3,
            mask=(_one(2),),
            insert_after=_ref(1),
            payload=(_result("t1", "real"),),
            paired_externally=frozenset({"t1"}),
        ),
    ]
    validate_context(resolve_context(tape).messages)
    assert _texts(tape) == ["go", "", "real"]


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
