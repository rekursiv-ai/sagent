"""Tests for ``AgentRuntime``'s tape-native append API.

These tests pin the C2 contract: ``append_history``, ``append_override``,
``append_clear``, ``replay_tape``, ``context()``, ``tape``, ``session_id``,
and the ``history`` readonly compatibility view. Mutation-site
conversions in C2c are checked by the pre-existing ``runtime_test.py``.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from sagent.agent.context import (
    InvalidContextError,
    ResolvedContext,
    resolve_context,
    validate_context,
)
from sagent.agent.runtime import AgentRuntime, Model, Tool
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
    TapeRef,
)


class _NoopModel:
    """Model that never streams; satisfies the ``Model`` protocol shape."""

    async def stream(
        self,
        history: list[HistoryEntry],
        system: str,
        tools: list[Tool],
        on_text: Callable[[str], None],
        on_thinking: Callable[[str], None],
    ) -> AssistantMessage:
        del history, system, tools, on_text, on_thinking
        return AssistantMessage(text="")


def _runtime(session_id: str = "s") -> AgentRuntime:
    model: Model = _NoopModel()
    return AgentRuntime(model=model, session_id=session_id)


# --- Construction and surface -----------------------------------------------


def test_new_runtime_has_empty_tape() -> None:
    """A fresh runtime starts with an empty tape."""
    runtime = _runtime()
    assert runtime.tape == []
    assert runtime.session_id == "s"


def test_session_id_defaults_to_empty_string() -> None:
    """Omitting ``session_id`` yields an empty default."""
    model: Model = _NoopModel()
    runtime = AgentRuntime(model=model)
    assert runtime.session_id == ""


def test_context_returns_resolved_context_value() -> None:
    """``runtime.context()`` returns a ``ResolvedContext`` instance."""
    runtime = _runtime()
    resolved = runtime.context()
    assert isinstance(resolved, ResolvedContext)
    assert resolved.messages == []
    assert resolved.version == 0


# --- append_history ---------------------------------------------------------


def test_append_history_returns_taperef_with_runtime_session_id() -> None:
    """``append_history`` mints a ``TapeRef`` tagged with the runtime's session."""
    runtime = _runtime(session_id="abc")
    ref = runtime.append_history(UserMessage(text="hi"))
    assert ref.session_id == "abc"
    assert ref.ordinal == 0


def test_append_history_ordinals_monotonically_increase() -> None:
    """Successive appends mint strictly increasing ordinals."""
    runtime = _runtime()
    r0 = runtime.append_history(UserMessage(text="a"))
    r1 = runtime.append_history(AssistantMessage(text="b"))
    r2 = runtime.append_history(UserMessage(text="c"))
    assert (r0.ordinal, r1.ordinal, r2.ordinal) == (0, 1, 2)


def test_append_history_records_history_record_on_tape() -> None:
    """Appended entries land as ``HistoryRecord`` tape records."""
    runtime = _runtime()
    entry = UserMessage(text="hi")
    ref = runtime.append_history(entry)
    assert len(runtime.tape) == 1
    record = runtime.tape[0]
    assert isinstance(record, HistoryRecord)
    assert record.ref == ref
    assert record.entry is entry


def test_append_history_updates_context_messages() -> None:
    """``context().messages`` reflects appended history in tape order."""
    runtime = _runtime()
    u = UserMessage(text="hi")
    a = AssistantMessage(text="hello")
    runtime.append_history(u)
    runtime.append_history(a)
    assert runtime.context().messages == [u, a]


def test_append_history_preserves_entry_object_identity() -> None:
    """Resolved messages reuse the exact ``HistoryEntry`` instances appended."""
    runtime = _runtime()
    u = UserMessage(text="hi")
    runtime.append_history(u)
    assert runtime.context().messages[0] is u


# --- append_override --------------------------------------------------------


def test_append_override_returns_taperef() -> None:
    """``append_override`` mints a ``TapeRef`` and appends the record."""
    runtime = _runtime()
    u = UserMessage(text="hi")
    hist_ref = runtime.append_history(u)
    over_ref = runtime.append_override(
        suppresses=(hist_ref,),
        payload=(UserMessage(text="[summary]"),),
        barrier=True,
        strategy="summary",
    )
    assert over_ref.ordinal == 1
    assert over_ref.session_id == "s"


def test_append_override_records_context_override() -> None:
    """``append_override`` stores a ``ContextOverride`` with the supplied fields."""
    runtime = _runtime()
    hist_ref = runtime.append_history(UserMessage(text="hi"))
    payload = (UserMessage(text="[summary]"),)
    over_ref = runtime.append_override(
        suppresses=(hist_ref,),
        inject_after=None,
        payload=payload,
        strategy="summary",
        barrier=True,
        token_before=100,
        token_after=20,
    )
    record = runtime.tape[1]
    assert isinstance(record, ContextOverride)
    assert record.ref == over_ref
    assert record.suppresses == (hist_ref,)
    assert record.inject_after is None
    assert record.payload == payload
    assert record.strategy == "summary"
    assert record.barrier is True
    assert record.token_before == 100
    assert record.token_after == 20


def test_append_override_changes_resolved_context() -> None:
    """A visible barrier override replaces the prior context."""
    runtime = _runtime()
    runtime.append_history(UserMessage(text="hi"))
    runtime.append_history(AssistantMessage(text="hello"))
    summary = UserMessage(text="[summary]")
    runtime.append_override(payload=(summary,), barrier=True, strategy="summary")
    assert runtime.context().messages == [summary]


def test_append_override_supports_user_coalesce_pattern() -> None:
    """User coalesce shape: suppress prior user, inject combined at prior anchor."""
    runtime = _runtime()
    r0 = runtime.append_history(UserMessage(text="first"))
    r1 = runtime.append_history(UserMessage(text="second"))
    combined = UserMessage(text="first\n\nsecond")
    runtime.append_override(
        suppresses=(r1,),
        inject_after=r0,
        payload=(combined,),
        strategy="user_coalesce",
    )
    messages = runtime.context().messages
    assert all(isinstance(m, UserMessage) for m in messages)
    assert [m.text for m in messages if isinstance(m, UserMessage)] == [
        "first",
        "first\n\nsecond",
    ]


def test_append_override_supports_detached_splice_pattern() -> None:
    """Detached splice: anchor on parent assistant, suppress placeholder."""
    runtime = _runtime()
    runtime.append_history(UserMessage(text="hi"))
    parent_ref = runtime.append_history(
        AssistantMessage(tool_calls=(ToolCall(id="c1", name="t", args={}),)),
    )
    placeholder_ref = runtime.append_history(
        ToolResult(call_id="c1", content="[detached]"),
    )
    real_result = ToolResult(call_id="c1", content="real")
    runtime.append_override(
        suppresses=(placeholder_ref,),
        inject_after=parent_ref,
        payload=(real_result,),
        strategy="detached_splice",
        paired_externally=frozenset({"c1"}),
    )
    messages = runtime.context().messages
    assert messages[-1] is real_result
    validate_context(messages)


# --- append_clear -----------------------------------------------------------


def test_append_clear_returns_taperef() -> None:
    """``append_clear`` mints a ``TapeRef`` for the clear record."""
    runtime = _runtime()
    runtime.append_history(UserMessage(text="hi"))
    ref = runtime.append_clear()
    assert ref.ordinal == 1
    assert ref.session_id == "s"


def test_append_clear_records_context_clear() -> None:
    """``append_clear`` appends a ``ContextClear`` with ``barrier=True``."""
    runtime = _runtime()
    runtime.append_clear()
    record = runtime.tape[0]
    assert isinstance(record, ContextClear)
    assert record.barrier is True


def test_append_clear_empties_resolved_context() -> None:
    """``append_clear`` removes prior visible history from resolved context."""
    runtime = _runtime()
    runtime.append_history(UserMessage(text="hi"))
    runtime.append_history(AssistantMessage(text="hello"))
    runtime.append_clear()
    assert runtime.context().messages == []


def test_append_history_after_clear_is_only_visible_entry() -> None:
    """History appended after a clear renders alone."""
    runtime = _runtime()
    runtime.append_history(UserMessage(text="dropped"))
    runtime.append_clear()
    u = UserMessage(text="kept")
    runtime.append_history(u)
    assert runtime.context().messages == [u]


# --- replay_tape ------------------------------------------------------------


def test_replay_tape_loads_records_verbatim() -> None:
    """Replay stores the supplied records on the tape."""
    runtime = _runtime()
    records = [
        HistoryRecord(
            ref=TapeRef(session_id="other", ordinal=0),
            entry=UserMessage(text="hi"),
        ),
        HistoryRecord(
            ref=TapeRef(session_id="other", ordinal=1),
            entry=AssistantMessage(text="hello"),
        ),
    ]
    runtime.replay_tape(records)
    assert runtime.tape == records


def test_replay_tape_advances_ordinal_cursor() -> None:
    """Subsequent appends continue from ``max(replayed ordinal) + 1``."""
    runtime = _runtime(session_id="new")
    runtime.replay_tape(
        [
            HistoryRecord(
                ref=TapeRef(session_id="old", ordinal=0),
                entry=UserMessage(text="a"),
            ),
            HistoryRecord(
                ref=TapeRef(session_id="old", ordinal=5),
                entry=AssistantMessage(text="b"),
            ),
        ],
    )
    new_ref = runtime.append_history(UserMessage(text="c"))
    assert new_ref.session_id == "new"
    assert new_ref.ordinal == 6


def test_replay_tape_resolves_loaded_history_in_order() -> None:
    """Replayed history records render through the resolver."""
    runtime = _runtime()
    u = UserMessage(text="loaded user")
    a = AssistantMessage(text="loaded assistant")
    runtime.replay_tape(
        [
            HistoryRecord(ref=TapeRef(session_id="x", ordinal=0), entry=u),
            HistoryRecord(ref=TapeRef(session_id="x", ordinal=1), entry=a),
        ],
    )
    assert runtime.context().messages == [u, a]


def test_replay_tape_accepts_mixed_record_types() -> None:
    """Replay handles ``HistoryRecord`` / ``ContextOverride`` / ``ContextClear``."""
    runtime = _runtime()
    u_ref = TapeRef(session_id="x", ordinal=0)
    runtime.replay_tape(
        [
            HistoryRecord(ref=u_ref, entry=UserMessage(text="hi")),
            ContextOverride(
                ref=TapeRef(session_id="x", ordinal=1),
                suppresses=(u_ref,),
                inject_after=None,
                payload=(UserMessage(text="[summary]"),),
                strategy="summary",
                barrier=True,
            ),
            HistoryRecord(
                ref=TapeRef(session_id="x", ordinal=2),
                entry=UserMessage(text="post"),
            ),
        ],
    )
    messages = runtime.context().messages
    assert all(isinstance(m, UserMessage) for m in messages)
    assert [m.text for m in messages if isinstance(m, UserMessage)] == [
        "[summary]",
        "post",
    ]


def test_replay_tape_with_empty_records_is_noop() -> None:
    """Replaying ``[]`` leaves the runtime unchanged."""
    runtime = _runtime()
    runtime.replay_tape([])
    assert runtime.tape == []
    new_ref = runtime.append_history(UserMessage(text="hi"))
    assert new_ref.ordinal == 0


# --- context() memoization and version --------------------------------------


def test_context_version_equals_tape_length() -> None:
    """``ResolvedContext.version`` mirrors ``len(runtime.tape)``."""
    runtime = _runtime()
    assert runtime.context().version == 0
    runtime.append_history(UserMessage(text="hi"))
    assert runtime.context().version == 1
    runtime.append_clear()
    assert runtime.context().version == 2


def test_context_caches_until_next_append() -> None:
    """Two ``context()`` calls without an append return the same messages list."""
    runtime = _runtime()
    runtime.append_history(UserMessage(text="hi"))
    first = runtime.context().messages
    second = runtime.context().messages
    assert first is second


def test_context_invalidates_on_append() -> None:
    """After an append, the cached messages list is rebuilt."""
    runtime = _runtime()
    runtime.append_history(UserMessage(text="hi"))
    before = runtime.context().messages
    runtime.append_history(AssistantMessage(text="hello"))
    after = runtime.context().messages
    assert before is not after
    assert len(after) == 2


# --- invariants the runtime keeps -------------------------------------------


def test_full_tool_round_trip_through_append_history_validates() -> None:
    """Appending a valid tool round-trip yields a validation-clean context."""
    runtime = _runtime()
    runtime.append_history(UserMessage(text="hi"))
    runtime.append_history(
        AssistantMessage(tool_calls=(ToolCall(id="c1", name="t", args={}),)),
    )
    runtime.append_history(ToolResult(call_id="c1", content="ok"))
    runtime.append_history(AssistantMessage(text="done"))
    validate_context(runtime.context().messages)


def test_compaction_override_via_append_validates() -> None:
    """A barrier-summary override yields a validation-clean resolved context."""
    runtime = _runtime()
    refs: list[TapeRef] = []
    refs.append(runtime.append_history(UserMessage(text="u1")))
    refs.append(
        runtime.append_history(
            AssistantMessage(tool_calls=(ToolCall(id="c1", name="t", args={}),)),
        ),
    )
    refs.append(runtime.append_history(ToolResult(call_id="c1", content="ok")))
    refs.append(runtime.append_history(AssistantMessage(text="done")))
    runtime.append_override(
        suppresses=tuple(refs),
        payload=(UserMessage(text="[summary]"),),
        strategy="summary",
        barrier=True,
    )
    runtime.append_history(UserMessage(text="follow up"))
    messages = runtime.context().messages
    validate_context(messages)
    assert [type(m).__name__ for m in messages] == ["UserMessage", "UserMessage"]


def test_resolve_context_called_directly_matches_runtime_context() -> None:
    """``resolve_context(runtime.tape)`` matches ``runtime.context()`` messages."""
    runtime = _runtime()
    runtime.append_history(UserMessage(text="hi"))
    runtime.append_history(AssistantMessage(text="hello"))
    assert resolve_context(runtime.tape).messages == runtime.context().messages


def test_orphan_tool_result_via_append_is_detected_by_validator() -> None:
    """Validation catches a malformed sequence pushed through the new API."""
    runtime = _runtime()
    runtime.append_history(UserMessage(text="hi"))
    runtime.append_history(ToolResult(call_id="c1", content="ok"))
    with pytest.raises(InvalidContextError):
        validate_context(runtime.context().messages)


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
