"""Tests for ``agent.context``: tape resolver and validator.

Per ``docs/private/better_compaction.md``. These tests define the
contract for ``resolve_context`` and ``validate_context`` before the
implementation lands, so the C2/C3/C4/C5 conversions can be checked
against a fixed surface.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from sagent.agent.context import (
    InvalidContextError,
    ResolvedContext,
    resolve_context,
    validate_context,
)
from sagent.types.history import (
    AssistantMessage,
    HistoryEntry,
    ToolCall,
    ToolResult,
    UserMessage,
)
from sagent.types.tape import (
    ContextClear,
    ContextOverride,
    HistoryRecord,
    TapeRecord,
    TapeRef,
)


def _ref(ordinal: int, session: str = "s") -> TapeRef:
    return TapeRef(session_id=session, ordinal=ordinal)


def _hr(ordinal: int, entry: HistoryEntry, session: str = "s") -> HistoryRecord:
    return HistoryRecord(ref=_ref(ordinal, session), entry=entry)


def _override(
    ordinal: int,
    *,
    suppresses: tuple[TapeRef, ...] = (),
    inject_after: TapeRef | None = None,
    payload: tuple[HistoryEntry, ...] = (),
    strategy: str = "test",
    barrier: bool = False,
    paired_externally: frozenset[str] = frozenset(),
    session: str = "s",
) -> ContextOverride:
    return ContextOverride(
        ref=_ref(ordinal, session),
        suppresses=suppresses,
        inject_after=inject_after,
        payload=payload,
        strategy=strategy,
        barrier=barrier,
        paired_externally=paired_externally,
    )


def _clear(ordinal: int, session: str = "s") -> ContextClear:
    return ContextClear(ref=_ref(ordinal, session))


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


def _resolve_messages(tape: Sequence[TapeRecord]) -> list[HistoryEntry]:
    return resolve_context(tape).messages


# --- Resolver: visibility and barriers ----------------------------------


def test_empty_tape_resolves_to_empty_messages() -> None:
    """An empty tape renders no messages, version 0, no discontinuity."""
    resolved = resolve_context([])
    assert resolved.messages == []
    assert resolved.version == 0
    assert resolved.discontinuity is False


def test_history_only_tape_renders_in_tape_order() -> None:
    """Legacy history-only tape renders entries in tape order."""
    u, a = _user("hi"), _assistant("hello")
    tape = [_hr(0, u), _hr(1, a)]
    assert _resolve_messages(tape) == [u, a]


def test_history_entries_are_object_identical_across_calls() -> None:
    """Identical resolves return the same ``HistoryEntry`` objects."""
    u = _user("hi")
    tape = [_hr(0, u)]
    first = resolve_context(tape).messages
    second = resolve_context(tape).messages
    assert first[0] is u
    assert second[0] is u


def test_override_suppresses_earlier_history_record() -> None:
    """A visible override hides the records named in ``suppresses``."""
    u, a = _user("hi"), _assistant("hello")
    tape: list[TapeRecord] = [
        _hr(0, u),
        _hr(1, a),
        _override(2, suppresses=(_ref(0),), payload=(), inject_after=_ref(1)),
    ]
    assert _resolve_messages(tape) == [a]


def test_override_with_no_anchor_inserts_at_head() -> None:
    """``inject_after=None`` puts the payload at the start of the slice."""
    u, a = _user("hi"), _assistant("hello")
    sysmsg = _user("[summary]")
    tape: list[TapeRecord] = [
        _hr(0, u),
        _hr(1, a),
        _override(2, inject_after=None, payload=(sysmsg,)),
    ]
    assert _resolve_messages(tape) == [sysmsg, u, a]


def test_override_payload_renders_after_anchor() -> None:
    """Payload appears immediately after the anchor's visible record."""
    u, a = _user("hi"), _assistant("hello")
    inj = _user("[note]")
    tape: list[TapeRecord] = [
        _hr(0, u),
        _hr(1, a),
        _override(2, inject_after=_ref(0), payload=(inj,)),
    ]
    assert _resolve_messages(tape) == [u, inj, a]


def test_missing_anchor_falls_back_to_head() -> None:
    """Anchor pointing at a non-visible ref injects at the head."""
    u, a = _user("hi"), _assistant("hello")
    inj = _user("[ghost]")
    missing = _ref(999)
    tape: list[TapeRecord] = [
        _hr(0, u),
        _hr(1, a),
        _override(2, inject_after=missing, payload=(inj,)),
    ]
    assert _resolve_messages(tape) == [inj, u, a]


def test_anchor_suppressed_falls_back_to_head() -> None:
    """Anchor that the same override suppresses degrades to head insert."""
    u, a = _user("hi"), _assistant("hello")
    inj = _user("[replacement]")
    tape: list[TapeRecord] = [
        _hr(0, u),
        _hr(1, a),
        _override(
            2,
            suppresses=(_ref(0),),
            inject_after=_ref(0),
            payload=(inj,),
        ),
    ]
    assert _resolve_messages(tape) == [inj, a]


def test_same_anchor_overrides_render_in_tape_order() -> None:
    """Multiple overrides with the same anchor render by tape order."""
    u, a = _user("hi"), _assistant("hello")
    p1, p2 = _user("[first]"), _user("[second]")
    tape: list[TapeRecord] = [
        _hr(0, u),
        _hr(1, a),
        _override(2, inject_after=_ref(0), payload=(p1,)),
        _override(3, inject_after=_ref(0), payload=(p2,)),
    ]
    assert _resolve_messages(tape) == [u, p1, p2, a]


def test_suppressed_override_has_no_effect() -> None:
    """An override that is itself suppressed contributes nothing."""
    u, a = _user("hi"), _assistant("hello")
    dead = _user("[suppressed]")
    tape: list[TapeRecord] = [
        _hr(0, u),
        _hr(1, a),
        _override(2, inject_after=_ref(0), payload=(dead,)),
        _override(3, suppresses=(_ref(2),)),
    ]
    assert _resolve_messages(tape) == [u, a]


def test_barrier_override_stops_reverse_walk() -> None:
    """A visible barrier hides all earlier records, suppressed or not."""
    u, a = _user("hi"), _assistant("hello")
    summary = _user("[barrier summary]")
    after = _user("after")
    tape: list[TapeRecord] = [
        _hr(0, u),
        _hr(1, a),
        _override(2, payload=(summary,), barrier=True),
        _hr(3, after),
    ]
    assert _resolve_messages(tape) == [summary, after]


def test_context_clear_stops_reverse_walk_with_no_payload() -> None:
    """A visible ``ContextClear`` drops everything earlier and emits nothing."""
    u, a = _user("hi"), _assistant("hello")
    after = _user("after")
    tape: list[TapeRecord] = [
        _hr(0, u),
        _hr(1, a),
        _clear(2),
        _hr(3, after),
    ]
    assert _resolve_messages(tape) == [after]


def test_clear_with_nothing_after_renders_empty() -> None:
    """A trailing ``ContextClear`` renders an empty context."""
    u = _user("hi")
    tape: list[TapeRecord] = [_hr(0, u), _clear(1)]
    assert _resolve_messages(tape) == []


def test_overrides_after_barrier_apply_normally() -> None:
    """Records after a barrier follow the usual rules within the slice."""
    summary = _user("[barrier]")
    u, a = _user("post"), _assistant("ack")
    inj = _user("[note]")
    tape: list[TapeRecord] = [
        _hr(0, _user("dropped")),
        _override(1, payload=(summary,), barrier=True),
        _hr(2, u),
        _hr(3, a),
        _override(4, inject_after=_ref(2), payload=(inj,)),
    ]
    assert _resolve_messages(tape) == [summary, u, inj, a]


def test_resolver_preserves_history_entry_identity_through_override() -> None:
    """Suppression keeps surviving ``HistoryEntry`` objects identical."""
    u, a = _user("hi"), _assistant("hello")
    keep = _user("keep")
    tape: list[TapeRecord] = [
        _hr(0, u),
        _hr(1, a),
        _hr(2, keep),
        _override(3, suppresses=(_ref(0),)),
    ]
    messages = _resolve_messages(tape)
    assert messages[0] is a
    assert messages[1] is keep


# --- Resolver: version and discontinuity --------------------------------


def test_version_equals_tape_length() -> None:
    """``ResolvedContext.version`` records ``len(tape)`` at resolve time."""
    assert resolve_context([]).version == 0
    tape = [_hr(0, _user("hi"))]
    assert resolve_context(tape).version == 1
    tape.append(_hr(1, _assistant("hello")))
    assert resolve_context(tape).version == 2


def test_discontinuity_is_false_when_prior_is_none() -> None:
    """First resolve has no prior; discontinuity is reported as ``False``."""
    tape = [_hr(0, _user("hi"))]
    assert resolve_context(tape, prior=None).discontinuity is False


def test_discontinuity_is_false_on_pure_history_append() -> None:
    """Appending a ``HistoryRecord`` keeps the prior prefix object-identical."""
    u = _user("hi")
    tape: list[TapeRecord] = [_hr(0, u)]
    first = resolve_context(tape)
    tape.append(_hr(1, _assistant("hello")))
    second = resolve_context(tape, prior=first)
    assert second.discontinuity is False


def test_discontinuity_is_true_when_override_changes_visible_prefix() -> None:
    """A new override that hides an earlier visible record is a discontinuity."""
    u, a = _user("hi"), _assistant("hello")
    tape: list[TapeRecord] = [_hr(0, u), _hr(1, a)]
    first = resolve_context(tape)
    tape.append(_override(2, suppresses=(_ref(0),)))
    second = resolve_context(tape, prior=first)
    assert second.discontinuity is True


def test_discontinuity_is_true_after_context_clear() -> None:
    """A new ``ContextClear`` drops the prior prefix entirely."""
    u = _user("hi")
    tape: list[TapeRecord] = [_hr(0, u)]
    first = resolve_context(tape)
    tape.append(_clear(1))
    second = resolve_context(tape, prior=first)
    assert second.discontinuity is True


def test_discontinuity_is_true_when_override_injects_mid_prefix() -> None:
    """Inserting a payload before the tail breaks pure-append identity."""
    u, a = _user("hi"), _assistant("hello")
    tape: list[TapeRecord] = [_hr(0, u), _hr(1, a)]
    first = resolve_context(tape)
    tape.append(_override(2, inject_after=_ref(0), payload=(_user("[note]"),)))
    second = resolve_context(tape, prior=first)
    assert second.discontinuity is True


def test_discontinuity_is_false_when_prior_equals_current() -> None:
    """Re-resolving an unchanged tape is not a discontinuity."""
    tape = [_hr(0, _user("hi")), _hr(1, _assistant("hello"))]
    first = resolve_context(tape)
    second = resolve_context(tape, prior=first)
    assert second.discontinuity is False


def test_resolved_context_returns_independent_message_list() -> None:
    """Mutating the returned list does not leak into subsequent resolves."""
    tape = [_hr(0, _user("hi"))]
    resolved = resolve_context(tape)
    resolved.messages.append(_assistant("oops"))
    assert len(resolve_context(tape).messages) == 1


# --- Validator -----------------------------------------------------------


def test_validate_empty_context_passes() -> None:
    """Empty messages are valid."""
    validate_context([])


def test_validate_user_only_context_passes() -> None:
    """A single user message is valid."""
    validate_context([_user("hi")])


def test_validate_assistant_text_only_passes() -> None:
    """User + assistant text-only round-trip is valid."""
    validate_context([_user("hi"), _assistant("hello")])


def test_validate_full_tool_round_trip_passes() -> None:
    """User, assistant-with-call, tool-result, assistant-text is valid."""
    validate_context(
        [
            _user("hi"),
            _assistant(tool_calls=(_tool_call("c1"),)),
            _tool_result("c1"),
            _assistant("done"),
        ],
    )


def test_validate_multi_tool_call_batch_all_results_present_passes() -> None:
    """Multiple tool calls with all results before the next turn pass."""
    validate_context(
        [
            _user("hi"),
            _assistant(tool_calls=(_tool_call("c1"), _tool_call("c2"))),
            _tool_result("c1"),
            _tool_result("c2"),
            _assistant("done"),
        ],
    )


def test_validate_tool_results_in_any_order_within_batch_passes() -> None:
    """Tool results can come in any order within the batch."""
    validate_context(
        [
            _user("hi"),
            _assistant(tool_calls=(_tool_call("c1"), _tool_call("c2"))),
            _tool_result("c2"),
            _tool_result("c1"),
        ],
    )


def test_validate_orphan_tool_result_raises() -> None:
    """A ``ToolResult`` with no preceding ``ToolCall`` is invalid."""
    with pytest.raises(InvalidContextError, match="orphan"):
        validate_context([_user("hi"), _tool_result("nope")])


def test_validate_missing_tool_result_at_end_raises() -> None:
    """A tool call without its result before context end is invalid."""
    with pytest.raises(InvalidContextError, match="without results"):
        validate_context(
            [_user("hi"), _assistant(tool_calls=(_tool_call("c1"),))],
        )


def test_validate_missing_tool_result_before_next_assistant_raises() -> None:
    """A tool call without its result before the next assistant turn fails."""
    with pytest.raises(InvalidContextError, match="pending tool calls"):
        validate_context(
            [
                _user("hi"),
                _assistant(tool_calls=(_tool_call("c1"),)),
                _assistant("oops"),
            ],
        )


def test_validate_user_between_call_and_result_raises() -> None:
    """A user message interleaved between call and result is invalid."""
    with pytest.raises(InvalidContextError, match="user message"):
        validate_context(
            [
                _user("hi"),
                _assistant(tool_calls=(_tool_call("c1"),)),
                _user("interrupting"),
                _tool_result("c1"),
            ],
        )


def test_validate_duplicate_tool_result_raises() -> None:
    """Two ``ToolResult``s with the same call_id are invalid."""
    with pytest.raises(InvalidContextError, match="duplicate"):
        validate_context(
            [
                _user("hi"),
                _assistant(tool_calls=(_tool_call("c1"),)),
                _tool_result("c1"),
                _tool_result("c1"),
            ],
        )


def test_validate_unknown_tool_result_after_batch_raises() -> None:
    """A ``ToolResult`` whose call_id is not pending is an orphan."""
    with pytest.raises(InvalidContextError, match="orphan"):
        validate_context(
            [
                _user("hi"),
                _assistant(tool_calls=(_tool_call("c1"),)),
                _tool_result("c1"),
                _tool_result("c2"),
            ],
        )


# --- Resolver + validator: integration ----------------------------------


def test_resolved_context_after_compaction_validates_cleanly() -> None:
    """A barrier-summary override produces validation-clean context."""
    summary = _user("[summary of prior session]")
    tape: list[TapeRecord] = [
        _hr(0, _user("user-1")),
        _hr(1, _assistant(tool_calls=(_tool_call("c1"),))),
        _hr(2, _tool_result("c1")),
        _hr(3, _assistant("done")),
        _override(4, payload=(summary,), barrier=True),
        _hr(5, _user("follow up")),
    ]
    validate_context(resolve_context(tape).messages)


def test_detached_splice_override_keeps_tool_pairing_valid() -> None:
    """Override-based detached splice preserves provider tool ordering."""
    real = _tool_result("c1", "real result")
    placeholder = _tool_result("c1", "[pending]")
    tape: list[TapeRecord] = [
        _hr(0, _user("hi")),
        _hr(1, _assistant(tool_calls=(_tool_call("c1"),))),
        _hr(2, placeholder),
        _override(
            3,
            suppresses=(_ref(2),),
            inject_after=_ref(1),
            payload=(real,),
            strategy="detached_splice",
            paired_externally=frozenset({"c1"}),
        ),
    ]
    messages = resolve_context(tape).messages
    assert messages[-1] is real
    validate_context(messages)


def test_resolved_context_is_a_resolved_context_instance() -> None:
    """``resolve_context`` returns a ``ResolvedContext`` value object."""
    resolved = resolve_context([])
    assert isinstance(resolved, ResolvedContext)


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
