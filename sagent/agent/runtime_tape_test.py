"""Tests for ``AgentRuntime``'s tape-native append API.

Covers ``append_history``, ``append_splice``, ``append_clear``,
``replay_tape``, ``context()``, ``tape``, ``session_id``, and the
append-time mask-overlap validator (once-overwrite rule).
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
from sagent.agent.runtime import AgentRuntime, Model
from sagent.types.runtime import (
    DETACHED_ARRIVED_MIMIC_PREFIX,
    DETACHED_PLACEHOLDER,
    AssistantMessage,
    ModelContextEvent,
    ToolCall,
    ToolResult,
    UserMessage,
)
from sagent.types.tape import (
    ContextSplice,
    InvalidSpliceError,
    MaskRange,
    ReferrableTapeEvent,
    TapeRef,
)


class _NoopModel:
    """Model that never streams; satisfies the ``Model`` protocol shape."""

    async def stream(
        self,
        history: list[ModelContextEvent],
        on_text: Callable[[str], None],
        on_thinking: Callable[[str], None],
    ) -> AssistantMessage:
        del history, on_text, on_thinking
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
    """Appended entries land as ``ReferrableTapeEvent`` tape records."""
    runtime = _runtime()
    entry = UserMessage(text="hi")
    ref = runtime.append_history(entry)
    assert len(runtime.tape) == 1
    record = runtime.tape[0]
    assert isinstance(record, ReferrableTapeEvent)
    assert record.ref == ref
    assert record.event is entry


def test_append_history_updates_context_messages() -> None:
    """``context().messages`` reflects appended history in tape order."""
    runtime = _runtime()
    u = UserMessage(text="hi")
    a = AssistantMessage(text="hello")
    runtime.append_history(u)
    runtime.append_history(a)
    assert runtime.context().messages == [u, a]


def test_append_history_preserves_entry_object_identity() -> None:
    """Resolved messages reuse the exact ``TapeEvent`` instances appended."""
    runtime = _runtime()
    u = UserMessage(text="hi")
    runtime.append_history(u)
    assert runtime.context().messages[0] is u


# --- append_splice ----------------------------------------------------------


def test_append_splice_returns_taperef() -> None:
    """``append_splice`` mints a ``TapeRef`` and appends the record."""
    runtime = _runtime()
    hist_ref = runtime.append_history(UserMessage(text="hi"))
    splice_ref = runtime.append_splice(
        mask=(MaskRange.between(hist_ref, hist_ref),),
        insert_after=None,
        payload=(UserMessage(text="[summary]"),),
        strategy="summary",
    )
    assert splice_ref.ordinal == 1
    assert splice_ref.session_id == "s"


def test_append_splice_records_context_splice() -> None:
    """``append_splice`` stores a ``ContextSplice`` with the supplied fields."""
    runtime = _runtime()
    hist_ref = runtime.append_history(UserMessage(text="hi"))
    payload = (UserMessage(text="[summary]"),)
    splice_ref = runtime.append_splice(
        mask=(MaskRange.between(hist_ref, hist_ref),),
        insert_after=None,
        payload=payload,
        strategy="summary",
        token_before=100,
        token_after=20,
    )
    record = runtime.tape[1]
    assert isinstance(record, ContextSplice)
    assert record.ref == splice_ref
    assert record.mask == (MaskRange.between(hist_ref, hist_ref),)
    assert record.insert_after is None
    assert record.payload == payload
    assert record.strategy == "summary"
    assert record.token_before == 100
    assert record.token_after == 20


def test_append_splice_changes_resolved_context() -> None:
    """A barrier-style splice replaces the prior context."""
    runtime = _runtime()
    r0 = runtime.append_history(UserMessage(text="hi"))
    r1 = runtime.append_history(AssistantMessage(text="hello"))
    summary = UserMessage(text="[summary]")
    runtime.append_splice(
        mask=(MaskRange.between(r0, r1),),
        insert_after=None,
        payload=(summary,),
        strategy="summary",
    )
    assert runtime.context().messages == [summary]


def test_append_splice_user_coalesce_pattern() -> None:
    """Coalesce shape: mask prior user tail, insert combined after preceding ref."""
    runtime = _runtime()
    r0 = runtime.append_history(UserMessage(text="first"))
    r1 = runtime.append_history(UserMessage(text="second"))
    combined = UserMessage(text="first\n\nsecond")
    runtime.append_splice(
        mask=(MaskRange.between(r1, r1),),
        insert_after=r0,
        payload=(combined,),
        strategy="user_coalesce",
    )
    messages = runtime.context().messages
    assert [m.text for m in messages if isinstance(m, UserMessage)] == [
        "first",
        "first\n\nsecond",
    ]


def test_append_splice_detached_splice_pattern() -> None:
    """Detached splice: mask placeholder, anchor on parent assistant."""
    runtime = _runtime()
    runtime.append_history(UserMessage(text="hi"))
    parent_ref = runtime.append_history(
        AssistantMessage(tool_calls=(ToolCall(id="c1", name="t", args={}),)),
    )
    placeholder_ref = runtime.append_history(
        ToolResult(call_id="c1", content=DETACHED_PLACEHOLDER),
    )
    real_result = ToolResult(call_id="c1", content="real")
    runtime.append_splice(
        mask=(MaskRange.between(placeholder_ref, placeholder_ref),),
        insert_after=parent_ref,
        payload=(real_result,),
        strategy="detached_splice",
        paired_externally=frozenset({"c1"}),
    )
    messages = runtime.context().messages
    assert messages[-1] is real_result
    validate_context(messages)


# --- append_splice: validation ---------------------------------------------


def test_append_splice_rejects_double_mask_of_same_position() -> None:
    """Two splices trying to mask the same alive HR ref are rejected."""
    runtime = _runtime()
    hr = runtime.append_history(UserMessage(text="x"))
    runtime.append_splice(
        mask=(MaskRange.between(hr, hr),),
        insert_after=None,
        payload=(UserMessage(text="first"),),
        strategy="test",
    )
    with pytest.raises(InvalidSpliceError, match="overlaps alive splice"):
        runtime.append_splice(
            mask=(MaskRange.between(hr, hr),),
            insert_after=None,
            payload=(UserMessage(text="second"),),
            strategy="test",
        )


def test_append_splice_allows_mask_after_absorbing_prior() -> None:
    """A splice that masks both the prior splice's ref AND its target passes."""
    runtime = _runtime()
    hr = runtime.append_history(UserMessage(text="x"))
    prior = runtime.append_splice(
        mask=(MaskRange.between(hr, hr),),
        insert_after=None,
        payload=(UserMessage(text="first"),),
        strategy="test",
    )
    # Re-edit by absorbing the prior splice. Mask range covers both hr
    # and prior, so prior is absorbed and its masking lapses.
    runtime.append_splice(
        mask=(MaskRange.between(hr, prior),),
        insert_after=None,
        payload=(UserMessage(text="second"),),
        strategy="test",
    )
    assert [
        m.text for m in runtime.context().messages if isinstance(m, UserMessage)
    ] == ["second"]


def test_append_splice_rejects_insert_after_inside_own_mask() -> None:
    """``insert_after`` pointing into this splice's own mask is rejected."""
    runtime = _runtime()
    r0 = runtime.append_history(UserMessage(text="a"))
    r1 = runtime.append_history(UserMessage(text="b"))
    with pytest.raises(InvalidSpliceError, match="insert_after"):
        runtime.append_splice(
            mask=(MaskRange.between(r0, r1),),
            insert_after=r0,
            payload=(),
            strategy="test",
        )


def test_adopt_record_rejects_mask_overlap_like_append_splice() -> None:
    """``adopt_record`` enforces the same splice invariants as append."""
    runtime = _runtime()
    hr = runtime.append_history(UserMessage(text="x"))
    prior = runtime.append_splice(
        mask=(MaskRange.between(hr, hr),),
        insert_after=None,
        payload=(UserMessage(text="first"),),
        strategy="test",
    )
    bad_mask = (MaskRange.between(hr, hr),)
    with pytest.raises(InvalidSpliceError):
        runtime.append_splice(
            mask=bad_mask,
            insert_after=None,
            payload=(UserMessage(text="summary"),),
            strategy="summary",
        )
    legacy_override = ContextSplice(
        ref=runtime.mint_ref(),
        mask=bad_mask,
        insert_after=None,
        payload=(UserMessage(text="summary"),),
        strategy="summary",
    )
    with pytest.raises(InvalidSpliceError):
        runtime.adopt_record(legacy_override)

    absorbing_override = ContextSplice(
        ref=runtime.mint_ref(),
        mask=(MaskRange.between(hr, prior),),
        insert_after=None,
        payload=(UserMessage(text="summary"),),
        strategy="summary",
    )
    runtime.adopt_record(absorbing_override)


# --- append_clear -----------------------------------------------------------


def test_append_clear_returns_taperef() -> None:
    """``append_clear`` mints a ``TapeRef`` for the barrier splice."""
    runtime = _runtime()
    runtime.append_history(UserMessage(text="hi"))
    ref = runtime.append_clear()
    assert ref.ordinal == 1
    assert ref.session_id == "s"


def test_append_clear_empties_resolved_context() -> None:
    """``append_clear`` masks all prior records, yielding empty view."""
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
        ReferrableTapeEvent(
            ref=TapeRef(session_id="other", ordinal=0),
            event=UserMessage(text="hi"),
        ),
        ReferrableTapeEvent(
            ref=TapeRef(session_id="other", ordinal=1),
            event=AssistantMessage(text="hello"),
        ),
    ]
    runtime.replay_tape(records)
    assert runtime.tape == records


def test_replay_tape_advances_ordinal_cursor() -> None:
    """Subsequent appends continue from ``max(replayed ordinal) + 1``."""
    runtime = _runtime(session_id="new")
    runtime.replay_tape(
        [
            ReferrableTapeEvent(
                ref=TapeRef(session_id="old", ordinal=0),
                event=UserMessage(text="a"),
            ),
            ReferrableTapeEvent(
                ref=TapeRef(session_id="old", ordinal=5),
                event=AssistantMessage(text="b"),
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
            ReferrableTapeEvent(ref=TapeRef(session_id="x", ordinal=0), event=u),
            ReferrableTapeEvent(ref=TapeRef(session_id="x", ordinal=1), event=a),
        ],
    )
    assert runtime.context().messages == [u, a]


def test_replay_tape_accepts_mixed_record_types() -> None:
    """Replay handles ``ReferrableTapeEvent`` + ``ContextSplice`` together."""
    runtime = _runtime()
    u_ref = TapeRef(session_id="x", ordinal=0)
    runtime.replay_tape(
        [
            ReferrableTapeEvent(ref=u_ref, event=UserMessage(text="hi")),
            ContextSplice(
                ref=TapeRef(session_id="x", ordinal=1),
                mask=(MaskRange.between(u_ref, u_ref),),
                insert_after=None,
                payload=(UserMessage(text="[summary]"),),
                strategy="summary",
            ),
            ReferrableTapeEvent(
                ref=TapeRef(session_id="x", ordinal=2),
                event=UserMessage(text="post"),
            ),
        ],
    )
    messages = runtime.context().messages
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


def test_replay_tape_seeds_mimic_counter_past_loaded_ids() -> None:
    """A resumed forged ``DetachedArrived`` id must not collide with the tape.

    ``_sanitize_forged_arrivals`` mints ``DetachedArrived:mimic:N`` from an
    instance-local counter. On resume that counter restarts at 0 unless
    ``replay_tape`` seeds it past the ids already on the loaded tape -- a
    second ``mimic:3`` then duplicates the first, producing a duplicate
    ``ToolResult`` call_id that wedges the model-call gate. Replay must seed
    the counter so the next forged id is globally unique.
    """
    runtime = _runtime(session_id="new")
    existing = f"{DETACHED_ARRIVED_MIMIC_PREFIX}3"
    runtime.replay_tape(
        [
            ReferrableTapeEvent(
                ref=TapeRef(session_id="old", ordinal=0),
                event=AssistantMessage(
                    tool_calls=(ToolCall(id=existing, name="DetachedArrived", args={}),)
                ),
            ),
            ReferrableTapeEvent(
                ref=TapeRef(session_id="old", ordinal=1),
                event=ToolResult(call_id=existing, content="r"),
            ),
        ],
    )
    # Forge enough arrivals that an unseeded counter (starting at 0) would
    # climb through ``mimic:0..mimic:3`` and re-mint the loaded ``mimic:3``.
    minted = [
        runtime._sanitize_forged_arrivals(
            AssistantMessage(
                tool_calls=(ToolCall(id="forged", name="DetachedArrived", args={}),)
            ),
        )
        .tool_calls[0]
        .id
        for _ in range(5)
    ]
    assert existing not in minted, (
        f"a resumed forged id re-minted the loaded {existing!r}: {minted}"
    )


def test_replay_tape_seeds_mimic_counter_from_lone_tool_result() -> None:
    """Seeding must cover a mimic id surviving only as a ``ToolResult``.

    ``_commit_pairing`` splices a lone mimic ``ToolResult`` (its parent
    ``AssistantMessage`` paired externally and possibly compacted away). The
    seed scan must read ``ToolResult.call_id`` too -- not just
    ``AssistantMessage.tool_calls`` -- or the counter under-seeds and a resumed
    forge re-mints the surviving id. Same partial-namespace-coverage bug the
    counter seed exists to prevent, on the consumer side of the id.
    """
    runtime = _runtime(session_id="new")
    existing = f"{DETACHED_ARRIVED_MIMIC_PREFIX}7"
    runtime.replay_tape(
        [
            ContextSplice(
                ref=TapeRef(session_id="old", ordinal=0),
                mask=(),
                insert_after=None,
                payload=(ToolResult(call_id=existing, content="err", is_error=True),),
                strategy="lazy_pairing",
                paired_externally=frozenset({existing}),
            ),
        ],
    )
    assert runtime._mimic_counter == 8, (
        f"lone mimic ToolResult must seed the counter; got {runtime._mimic_counter}"
    )


def test_replay_tape_seeds_mimic_counter_from_paired_externally_only() -> None:
    """A legacy ``replay()`` record can carry a mimic id only in ``paired_externally``.

    ``ContextSplice.replay`` (on-disk load) bypasses ``_validate_payload``, so a
    persisted record can declare a ``paired_externally`` mimic id whose local
    pair is absent from ``payload``. Scanning ``payload`` alone then under-seeds
    the counter and a resumed forge re-mints the id. The seed must also read
    ``paired_externally`` -- the load path's trust boundary, not the producer's.
    """
    runtime = _runtime(session_id="new")
    existing = f"{DETACHED_ARRIVED_MIMIC_PREFIX}11"
    runtime.replay_tape(
        [
            ContextSplice.replay(
                ref=TapeRef(session_id="old", ordinal=0),
                mask=(),
                insert_after=None,
                payload=(),
                strategy="legacy",
                paired_externally=frozenset({existing}),
            ),
        ],
    )
    assert runtime._mimic_counter == 12, (
        f"paired_externally mimic id must seed the counter; got "
        f"{runtime._mimic_counter}"
    )


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


def test_compaction_splice_via_append_validates() -> None:
    """A barrier-summary splice yields a validation-clean resolved context."""
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
    runtime.append_splice(
        mask=(MaskRange.between(refs[0], refs[-1]),),
        insert_after=None,
        payload=(UserMessage(text="[summary]"),),
        strategy="summary",
    )
    runtime._append_or_coalesce_user(UserMessage(text="follow up"))
    messages = runtime.context().messages
    validate_context(messages)
    assert [type(m).__name__ for m in messages] == ["UserMessage"]
    summary = messages[0]
    assert isinstance(summary, UserMessage)
    assert summary.text == "[summary]\n\nfollow up"


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
