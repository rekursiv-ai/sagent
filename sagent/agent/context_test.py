"""Tests for ``agent.context``: tape resolver and validator.

Verifies the new ``ContextSplice`` model: mask + insert_after + payload,
with "undelete via cover-the-cover" semantics (a splice that masks
another splice's ref nullifies the masked splice's effects, restoring
the originally-masked content).
"""

from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import patch

import pytest

from sagent.agent import context as context_module
from sagent.agent.context import (
    InvalidContextError,
    ResolvedContext,
    alive_splices,
    masked_refs_by_alive,
    resolve_context,
    validate_context,
)
from sagent.types.runtime import (
    DETACHED_PLACEHOLDER,
    AgentSendMessage,
    AssistantMessage,
    ModelContextEvent,
    ToolCall,
    ToolResult,
    UserMessage,
)
from sagent.types.tape import (
    ContextSplice,
    MaskRange,
    ReferrableTapeEvent,
    TapeEvent,
    TapeRecord,
    TapeRef,
)


def _ref(ordinal: int, session: str = "s") -> TapeRef:
    return TapeRef(session_id=session, ordinal=ordinal)


def _hr(ordinal: int, entry: TapeEvent, session: str = "s") -> ReferrableTapeEvent:
    return ReferrableTapeEvent(ref=_ref(ordinal, session), event=entry)


def _splice(
    ordinal: int,
    *,
    mask: tuple[MaskRange, ...] = (),
    insert_after: TapeRef | None = None,
    payload: tuple[ModelContextEvent, ...] = (),
    strategy: str = "test",
    paired_externally: frozenset[str] = frozenset(),
    session: str = "s",
) -> ContextSplice:
    return ContextSplice(
        ref=_ref(ordinal, session),
        mask=mask,
        insert_after=insert_after,
        payload=payload,
        strategy=strategy,
        paired_externally=paired_externally,
    )


def _user(text: str) -> UserMessage:
    return UserMessage(text=text)


def _assistant(
    text: str = "",
    tool_calls: tuple[ToolCall, ...] = (),
) -> AssistantMessage:
    return AssistantMessage(text=text, tool_calls=tool_calls)


def _tool_call(call_id: str, name: str = "echo") -> ToolCall:
    return ToolCall(id=call_id, name=name, args={})


def _tool_result(call_id: str, content: str = "ok") -> ToolResult:
    return ToolResult(call_id=call_id, content=content)


def _resolve_messages(tape: Sequence[TapeRecord]) -> list[ModelContextEvent]:
    return resolve_context(tape).messages


def test_validate_context_rejects_consecutive_users() -> None:
    with pytest.raises(InvalidContextError, match="alternation"):
        validate_context([_user("a"), _user("b")])


def test_validate_context_rejects_consecutive_assistants() -> None:
    with pytest.raises(InvalidContextError, match="alternation"):
        validate_context([_assistant("a"), _assistant("b")])


def test_validate_context_rejects_user_message_then_agent_send() -> None:
    """UserMessage + AgentSendMessage both serialize as user role on wire."""
    with pytest.raises(InvalidContextError, match="alternation"):
        validate_context(
            [UserMessage(text="u1"), AgentSendMessage(source="A", text="u2")],
        )


def test_validate_context_rejects_agent_send_then_user_message() -> None:
    """Reverse cross-type adjacency also violates user-role alternation."""
    with pytest.raises(InvalidContextError, match="alternation"):
        validate_context(
            [AgentSendMessage(source="A", text="u1"), UserMessage(text="u2")],
        )


# --- Resolver: visibility and masking ----------------------------------


def test_empty_tape_resolves_to_empty_messages() -> None:
    """An empty tape renders no messages and version 0."""
    resolved = resolve_context([])
    assert resolved.messages == []
    assert resolved.version == 0


def test_history_only_tape_renders_in_tape_order() -> None:
    """Pure history records emit their entries in tape order."""
    tape = [_hr(0, _user("a")), _hr(1, _assistant("b")), _hr(2, _user("c"))]
    msgs = _resolve_messages(tape)
    assert [getattr(m, "text", None) for m in msgs] == ["a", "b", "c"]


def test_history_entries_are_object_identical_across_calls() -> None:
    """The resolver returns the same ``TapeEvent`` instances."""
    hr = _hr(0, _user("x"))
    tape = [hr]
    first = resolve_context(tape)
    second = resolve_context(tape)
    assert first.messages[0] is second.messages[0]


# --- Splice: masking ----------------------------------------------------


def test_splice_masks_earlier_history_record() -> None:
    """A splice whose mask covers a HR position removes its entry.

    With ``insert_after=_ref(0)``, the payload renders after ``a`` —
    which is where ``b`` used to be in tape order.
    """
    tape = [
        _hr(0, _user("a")),
        _hr(1, _user("b")),
        _splice(
            2,
            mask=(MaskRange.between(_ref(1), _ref(1)),),
            insert_after=_ref(0),
            payload=(_user("B"),),
        ),
    ]
    msgs = _resolve_messages(tape)
    assert [getattr(m, "text", None) for m in msgs] == ["a", "B"]


def test_splice_mask_respects_session_id() -> None:
    tape = [
        _hr(0, _user("a0"), session="a"),
        _hr(0, _user("b0"), session="b"),
        _splice(
            1,
            mask=(MaskRange.between(_ref(0, "a"), _ref(0, "a")),),
            insert_after=None,
            payload=(_user("summary a"),),
            session="a",
        ),
    ]
    msgs = _resolve_messages(tape)
    assert [getattr(m, "text", None) for m in msgs] == ["summary a", "b0"]


def test_splice_with_no_anchor_inserts_at_head() -> None:
    """``insert_after=None`` injects payload at the start of the view."""
    tape = [
        _hr(0, _user("a")),
        _splice(1, mask=(), insert_after=None, payload=(_user("HEAD"),)),
    ]
    msgs = _resolve_messages(tape)
    assert [getattr(m, "text", None) for m in msgs] == ["HEAD", "a"]


def test_splice_payload_renders_after_anchor() -> None:
    """``insert_after=ref(N)`` places payload immediately after ref(N)."""
    tape = [
        _hr(0, _user("a")),
        _hr(1, _user("b")),
        _splice(2, insert_after=_ref(0), payload=(_user("X"),)),
    ]
    # Payload at insert_after=ref(0) lands between a and b in tape order.
    msgs = _resolve_messages(tape)
    assert [getattr(m, "text", None) for m in msgs] == ["a", "X", "b"]


def test_pure_deletion_with_empty_payload() -> None:
    """A splice with empty payload is pure deletion: masked ref vanishes."""
    tape = [
        _hr(0, _user("a")),
        _hr(1, _user("b")),
        _hr(2, _user("c")),
        _splice(3, mask=(MaskRange.between(_ref(1), _ref(1)),), payload=()),
    ]
    msgs = _resolve_messages(tape)
    assert [getattr(m, "text", None) for m in msgs] == ["a", "c"]


def test_pure_insertion_with_empty_mask() -> None:
    """A splice with empty mask is pure insertion at the anchor."""
    tape = [
        _hr(0, _user("a")),
        _splice(1, mask=(), insert_after=_ref(0), payload=(_user("X"),)),
    ]
    msgs = _resolve_messages(tape)
    assert [getattr(m, "text", None) for m in msgs] == ["a", "X"]


def test_multi_range_mask_removes_non_contiguous_positions() -> None:
    """A single splice with two mask ranges removes both ranges."""
    tape = [
        _hr(0, _user("a")),
        _hr(1, _user("b")),
        _hr(2, _user("c")),
        _hr(3, _user("d")),
        _hr(4, _user("e")),
        _splice(
            5,
            mask=(
                MaskRange.between(_ref(1), _ref(1)),
                MaskRange.between(_ref(3), _ref(3)),
            ),
            insert_after=_ref(0),
            payload=(_user("SUMMARY"),),
        ),
    ]
    msgs = _resolve_messages(tape)
    # b and d masked, summary at head-after-a, c and e survive.
    assert [getattr(m, "text", None) for m in msgs] == ["a", "SUMMARY", "c", "e"]


def test_barrier_splice_masks_entire_prefix() -> None:
    """A splice whose mask covers from head to last ref masks everything."""
    tape = [
        _hr(0, _user("a")),
        _hr(1, _user("b")),
        _hr(2, _user("c")),
        _splice(
            3,
            mask=(MaskRange.between(_ref(0), _ref(2)),),
            insert_after=None,
            payload=(_user("SUMMARY"),),
        ),
    ]
    msgs = _resolve_messages(tape)
    assert [getattr(m, "text", None) for m in msgs] == ["SUMMARY"]


def test_history_records_after_barrier_apply_normally() -> None:
    """Records appended after a barrier-style splice render after its payload."""
    tape = [
        _hr(0, _user("a")),
        _hr(1, _user("b")),
        _splice(
            2,
            mask=(MaskRange.between(_ref(0), _ref(1)),),
            insert_after=None,
            payload=(_user("SUMMARY"),),
        ),
        _hr(3, _user("c")),
    ]
    msgs = _resolve_messages(tape)
    assert [getattr(m, "text", None) for m in msgs] == ["SUMMARY", "c"]


# --- Undelete: cover-the-cover semantic --------------------------------


def test_cover_the_cover_resurrects_originally_masked_content() -> None:
    """A splice that masks another splice's ref undoes its masking effect.

    Tape walk: d masks b (resolved view loses b). f masks d (d becomes
    dead → its mask of b lapses → b resurfaces). f's payload renders at
    its insert_after.
    """
    tape = [
        _hr(0, _user("a")),
        _hr(1, _user("b")),
        _hr(2, _user("c")),
        _splice(
            3,
            mask=(MaskRange.between(_ref(1), _ref(1)),),
            insert_after=_ref(0),
            payload=(_user("D"),),
        ),
        _hr(4, _user("e")),
        _splice(
            5,
            mask=(MaskRange.between(_ref(3), _ref(3)),),
            insert_after=_ref(0),
            payload=(_user("F"),),
        ),
    ]
    msgs = _resolve_messages(tape)
    # f killed d → d's mask of b lapses → b resurfaces.
    # f's payload renders after a. Order: a, F, b, c, e.
    assert [getattr(m, "text", None) for m in msgs] == ["a", "F", "b", "c", "e"]


def test_dead_splice_does_not_contribute_payload() -> None:
    """A splice whose record ref is masked contributes nothing."""
    tape = [
        _hr(0, _user("a")),
        _splice(
            1,
            mask=(),
            insert_after=_ref(0),
            payload=(_user("X"),),
        ),
        _splice(
            2,
            mask=(MaskRange.between(_ref(1), _ref(1)),),
            insert_after=_ref(0),
            payload=(_user("Y"),),
        ),
    ]
    msgs = _resolve_messages(tape)
    # Splice 1 is dead (masked by splice 2). Splice 2 is alive, its
    # payload renders. Splice 1's payload is dropped.
    assert [getattr(m, "text", None) for m in msgs] == ["a", "Y"]


def test_user_example_walkthrough() -> None:
    """The user's example: g covers b, f covers d.

    Tape: [a, b, c, d, e, f, g] where f at pos 5 masks d at pos 3,
    g at pos 6 masks b at pos 1. Each adds its own payload.
    """
    tape = [
        _hr(0, _user("a")),
        _hr(1, _user("b")),
        _hr(2, _user("c")),
        _hr(3, _user("d")),
        _hr(4, _user("e")),
        _splice(
            5,
            mask=(MaskRange.between(_ref(3), _ref(3)),),
            insert_after=_ref(2),
            payload=(_user("F"),),
        ),
        _splice(
            6,
            mask=(MaskRange.between(_ref(1), _ref(1)),),
            insert_after=_ref(0),
            payload=(_user("G"),),
        ),
    ]
    msgs = _resolve_messages(tape)
    assert [getattr(m, "text", None) for m in msgs] == ["a", "G", "c", "F", "e"]


# --- Aliveness helpers -------------------------------------------------


def test_alive_splices_excludes_masked_splice_ref() -> None:
    """A splice masked by a later splice is not in the alive set."""
    tape = [
        _hr(0, _user("a")),
        _splice(1, mask=(), insert_after=_ref(0), payload=(_user("X"),)),
        _splice(2, mask=(MaskRange.between(_ref(1), _ref(1)),), payload=()),
    ]
    alive = alive_splices(tape)
    assert _ref(2) in alive
    assert _ref(1) not in alive


def test_masked_refs_by_alive_excludes_dead_splice_masking() -> None:
    """A dead splice's mask does not contribute to the alive-masked set."""
    tape = [
        _hr(0, _user("a")),
        _hr(1, _user("b")),
        _splice(2, mask=(MaskRange.between(_ref(1), _ref(1)),), payload=()),
        _splice(3, mask=(MaskRange.between(_ref(2), _ref(2)),), payload=()),
    ]
    alive = alive_splices(tape)
    masked = masked_refs_by_alive(tape, alive)
    # Splice 2 is dead (masked by 3); its mask of b lapses.
    assert _ref(1) not in masked
    # Splice 3 is alive; its mask of splice 2's ref applies.
    assert _ref(2) in masked


def test_alive_splices_handles_large_alive_splice_sets_quickly() -> None:
    tape: list[TapeRecord] = []
    for idx in range(2_000):
        tape.append(_hr(idx * 2, _user(str(idx))))
        tape.append(
            _splice(
                idx * 2 + 1,
                mask=(MaskRange.between(_ref(idx * 2), _ref(idx * 2)),),
                payload=(),
            )
        )
    alive = alive_splices(tape)
    assert len(alive) == 2_000


# --- Version -----------------------------------------------------------


def test_version_equals_tape_length() -> None:
    """``version`` reports the number of tape records."""
    tape = [_hr(0, _user("a")), _hr(1, _user("b"))]
    assert resolve_context(tape).version == 2


def test_resolved_context_returns_independent_message_list() -> None:
    """Mutating the returned list does not affect a fresh resolve."""
    tape = [_hr(0, _user("a"))]
    first = resolve_context(tape)
    first.messages.append(_user("BOGUS"))
    second = resolve_context(tape)
    assert len(second.messages) == 1


def test_resolve_scales_linearly_in_tape_length() -> None:
    """Anchor lookup must not rescan the emitted order for every splice.

    Anchor lookup walked the emitted list per alive splice, and a coalesce splice per
    user turn is the ordinary shape -- so the resolver grew quadratically in a
    long session, on a path the runtime re-runs after every single append.
    Counted rather than timed: a wall-clock bound would be flaky under load,
    while the scan count is the property that actually decides the growth.
    """
    scans = 0
    real_contains = context_module._OrderedRefs.contains

    def counting_contains(order: context_module._OrderedRefs, ref: TapeRef) -> bool:
        nonlocal scans
        scans += 1
        return real_contains(order, ref)

    tape: list[TapeRecord] = []
    ordinal = 0
    for i in range(200):
        tape.append(_hr(ordinal, _user(f"u{i}")))
        ordinal += 1
        tape.append(
            _splice(
                ordinal,
                mask=(MaskRange(session_id="s", lo=ordinal - 1, hi=ordinal - 1),),
                insert_after=_ref(ordinal - 2) if ordinal >= 2 else None,
                payload=(_assistant(f"a{i}"),),
            )
        )
        ordinal += 1

    with patch.object(context_module._OrderedRefs, "contains", counting_contains):
        _ = resolve_context(tape)

    assert scans <= len(tape), (
        f"anchor lookup cost {scans} probes over a {len(tape)}-record tape"
    )


def test_a_duplicate_ref_is_refused_rather_than_silently_doubled() -> None:
    """One ref names one record; two claimants is corruption, not a choice.

    ``segments`` is keyed by ref while ``order`` keeps every occurrence, so a
    tape carrying a ref twice rendered the LAST record twice and dropped the
    first -- no error, no log. That is how a concurrent writer colliding on an
    ordinal turns into a conversation that reads back wrong, and three real
    sessions on disk carry the shape.
    """
    ref = _ref(0)
    tape = [
        ReferrableTapeEvent(ref=ref, event=_user("a")),
        ReferrableTapeEvent(ref=ref, event=_user("b")),
    ]
    with pytest.raises(InvalidContextError, match="duplicate"):
        _ = resolve_context(tape)


# --- validate_context: unchanged from prior model ----------------------


def test_validate_empty_context_passes() -> None:
    validate_context([])


def test_validate_user_only_context_passes() -> None:
    validate_context([_user("hi")])


def test_validate_assistant_text_only_passes() -> None:
    validate_context([_user("hi"), _assistant("hello")])


def test_validate_full_tool_round_trip_passes() -> None:
    validate_context(
        [
            _user("go"),
            _assistant(tool_calls=(_tool_call("t1"),)),
            _tool_result("t1"),
            _assistant("done"),
        ],
    )


def test_validate_multi_tool_call_batch_all_results_present_passes() -> None:
    validate_context(
        [
            _user("go"),
            _assistant(tool_calls=(_tool_call("t1"), _tool_call("t2"))),
            _tool_result("t1"),
            _tool_result("t2"),
        ],
    )


def test_validate_tool_results_in_any_order_within_batch_passes() -> None:
    validate_context(
        [
            _user("go"),
            _assistant(tool_calls=(_tool_call("t1"), _tool_call("t2"))),
            _tool_result("t2"),
            _tool_result("t1"),
        ],
    )


def test_validate_orphan_tool_result_raises() -> None:
    with pytest.raises(InvalidContextError, match="orphan ToolResult"):
        validate_context([_tool_result("t1")])


def test_validate_missing_tool_result_at_end_raises() -> None:
    with pytest.raises(InvalidContextError, match="without results at end"):
        validate_context(
            [_user("go"), _assistant(tool_calls=(_tool_call("t1"),))],
        )


def test_validate_missing_tool_result_before_next_assistant_raises() -> None:
    with pytest.raises(InvalidContextError, match="pending tool calls"):
        validate_context(
            [
                _user("go"),
                _assistant(tool_calls=(_tool_call("t1"),)),
                _assistant("oops"),
            ],
        )


def test_validate_user_between_call_and_result_raises() -> None:
    with pytest.raises(InvalidContextError, match="user message before tool results"):
        validate_context(
            [
                _user("go"),
                _assistant(tool_calls=(_tool_call("t1"),)),
                _user("wait"),
                _tool_result("t1"),
            ],
        )


def test_validate_duplicate_tool_result_raises() -> None:
    with pytest.raises(InvalidContextError, match="duplicate ToolResult"):
        validate_context(
            [
                _user("go"),
                _assistant(tool_calls=(_tool_call("t1"),)),
                _tool_result("t1", content="a"),
                _tool_result("t1", content="b"),
            ],
        )


def test_validate_unknown_tool_result_after_batch_raises() -> None:
    with pytest.raises(InvalidContextError, match="orphan ToolResult"):
        validate_context(
            [
                _user("go"),
                _assistant(tool_calls=(_tool_call("t1"),)),
                _tool_result("t1"),
                _tool_result("t2"),
            ],
        )


# --- Integration: resolve + validate -----------------------------------


def test_resolved_context_after_compaction_validates_cleanly() -> None:
    """A summary-style splice yields a wire-format-valid resolved view."""
    tape = [
        _hr(0, _user("go")),
        _hr(1, _assistant(tool_calls=(_tool_call("t1"),))),
        _hr(2, _tool_result("t1")),
        _splice(
            3,
            mask=(MaskRange.between(_ref(0), _ref(2)),),
            insert_after=None,
            payload=(_user("[summary]"),),
        ),
    ]
    msgs = _resolve_messages(tape)
    validate_context(msgs)
    assert [getattr(m, "text", None) for m in msgs] == ["[summary]"]


def test_detached_splice_keeps_tool_pairing_valid() -> None:
    """Real TR splice into placeholder slot yields a valid context."""
    tape = [
        _hr(0, _user("go")),
        _hr(1, _assistant(tool_calls=(_tool_call("t1"),))),
        _hr(2, _tool_result("t1", content=DETACHED_PLACEHOLDER)),
        _splice(
            3,
            mask=(MaskRange.between(_ref(2), _ref(2)),),
            insert_after=_ref(1),
            payload=(_tool_result("t1", content="real"),),
            paired_externally=frozenset({"t1"}),
        ),
    ]
    msgs = _resolve_messages(tape)
    validate_context(msgs)
    trs = [m for m in msgs if isinstance(m, ToolResult)]
    assert len(trs) == 1
    assert trs[0].content == "real"


def test_resolved_context_is_a_resolved_context_instance() -> None:
    """Sanity: ``resolve_context`` returns a ``ResolvedContext``."""
    result = resolve_context([])
    assert isinstance(result, ResolvedContext)


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
