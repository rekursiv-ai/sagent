"""Tests for ``ContextSplice.__post_init__`` payload and mask validation.

The validators enforce structural correctness at construct time so
producer bugs fail loudly at construction rather than silently at
resolve. Mask-disjointness is checked per-splice; cross-splice
mask-overlap (the once-overwrite rule) is enforced separately in
``AgentRuntime.append_splice``.
"""

from __future__ import annotations

import pytest

from sagent.types.runtime import (
    AgentSendMessage,
    AssistantMessage,
    ModelContextEvent,
    ToolCall,
    ToolResult,
    UserMessage,
)
from sagent.types.tape import (
    ContextSplice,
    InvalidPayloadError,
    MaskRange,
    TapeRef,
    coalesce_roles,
    merge_mask_ranges,
    pair_and_dedup_tool_calls,
    splice_safe_repair,
)


_REF = TapeRef(session_id="s", ordinal=0)


def _ref(ordinal: int) -> TapeRef:
    """Test helper: build a TapeRef with the standard session id."""
    return TapeRef(session_id="s", ordinal=ordinal)


def _mr(lo: int, hi: int, session_id: str = "s") -> MaskRange:
    """Test helper: build a single-session MaskRange."""
    return MaskRange(session_id=session_id, lo=lo, hi=hi)


def test_coalesce_roles_is_total_on_duplicate_tool_call_ids() -> None:
    """Merging adjacent AMs that share a tool_call id must not raise.

    ``coalesce_roles`` concatenates merged ``tool_calls``; if two adjacent
    AMs carry the same id, ``AssistantMessage.__post_init__`` rejects the
    duplicate and the coalescer -- the single canonical splice repair --
    raises ``ValueError`` instead of returning a valid payload. It must be
    total over its declared input domain: dedupe ids on merge (keep first).
    """
    tc = ToolCall(id="t1", name="x", args={})
    out = coalesce_roles(
        (
            AssistantMessage(text="a", tool_calls=(tc,)),
            AssistantMessage(text="b", tool_calls=(tc,)),
        ),
    )
    assert len(out) == 1
    merged = out[0]
    assert isinstance(merged, AssistantMessage)
    assert [c.id for c in merged.tool_calls] == ["t1"]
    # The whole point: the result must construct as a splice payload.
    ContextSplice(
        ref=_REF,
        mask=(),
        insert_after=None,
        strategy="x",
        payload=(*out, ToolResult(call_id="t1", content="r")),
    )


def test_coalesce_roles_drops_signed_thinking_on_assistant_merge() -> None:
    """A merged AM must not carry signed thinking blocks from two turns.

    Anthropic signed ``thinking`` blocks bind to their originating turn; a
    merged AM that concatenates signed blocks from two turns serializes an
    invalid signature set and the provider 400s. On merge, signed thinking
    cannot be re-signed, so it must be dropped (unsigned / redacted blocks
    are safe to keep).
    """
    signed_a = {"type": "thinking", "thinking": "A", "signature": "sigA"}
    signed_b = {"type": "thinking", "thinking": "B", "signature": "sigB"}
    out = coalesce_roles(
        (
            AssistantMessage(text="a", thinking_blocks=(signed_a,)),
            AssistantMessage(text="b", thinking_blocks=(signed_b,)),
        ),
    )
    assert len(out) == 1
    merged = out[0]
    assert isinstance(merged, AssistantMessage)
    signed = [
        b
        for b in merged.thinking_blocks
        if b.get("type") == "thinking" and b.get("signature")
    ]
    assert signed == [], f"merged AM kept cross-turn signed thinking: {signed}"


def test_coalesce_roles_merges_across_hidden_boundary_visible_wins() -> None:
    """A hidden/visible adjacency merges losslessly; visible wins the bit.

    ``hidden`` is render-only -- the model receives the text either way, the
    bit only suppresses REPL display -- so merging across the boundary loses
    nothing on the wire. The merged ``hidden`` is the AND of both parts: a
    block stays suppressed only if every part was, so any visible part keeps
    the merged block visible. Critically, ``coalesce_roles`` must NOT raise
    here: the rescue and load paths feed it arbitrary legacy adjacencies and
    rely on its totality (C-001 / rescue regression).
    """
    out = coalesce_roles(
        (
            UserMessage(text="visible"),
            UserMessage(text="hidden-part", hidden=True),
        ),
    )
    assert len(out) == 1
    merged = out[0]
    assert isinstance(merged, UserMessage)
    assert merged.hidden is False  # a visible part keeps the block visible
    assert merged.text == "visible\n\nhidden-part"
    # Both-hidden stays hidden.
    both = coalesce_roles(
        (
            UserMessage(text="a", hidden=True),
            UserMessage(text="b", hidden=True),
        ),
    )
    assert len(both) == 1
    assert isinstance(both[0], UserMessage)
    assert both[0].hidden is True


def test_payload_rejects_adjacent_agent_sends() -> None:
    """Two adjacent ``AgentSendMessage`` are both wire-``user``: rejected raw.

    The constructor must reject the un-coalesced shape (the contract
    ``coalesce_roles`` exists to satisfy), not just the ``UserMessage`` pair.
    """
    with pytest.raises(InvalidPayloadError, match="alternation"):
        ContextSplice(
            ref=_REF,
            mask=(),
            insert_after=None,
            strategy="x",
            payload=(
                AgentSendMessage(source="A", text="a"),
                AgentSendMessage(source="B", text="b"),
            ),
        )


def test_coalesced_consecutive_assistants_pass_validating_constructor() -> None:
    """A legacy two-AM shape, once coalesced, constructs without ``replay``.

    Locks the H24 guarantee: ``coalesce_roles`` makes the on-disk-repair
    payload splice-valid, so ``load_session``'s repair can use the validating
    constructor instead of the validation-bypassing ``replay``.
    """
    ContextSplice(
        ref=_REF,
        mask=(),
        insert_after=None,
        strategy="orphan_tool_result_repair",
        payload=coalesce_roles(
            (AssistantMessage(text="one"), AssistantMessage(text="two")),
        ),
    )


def test_pair_and_dedup_defers_agent_send_between_tool_pair() -> None:
    """An ``AgentSendMessage`` mid-tool-turn must not flush pending calls.

    A peer agent's send interleaves with an in-flight tool turn; it is not an
    interrupt (only a ``UserMessage`` is). Treating it as a flush point
    synthesizes a bogus ``[interrupted]`` result and then drops the *real*
    ``ToolResult`` as an orphan -- losing a real tool result. Defer the send
    until the tool pair closes, matching ``summary._drop_orphan_tool_results``.
    """
    am = AssistantMessage(tool_calls=(ToolCall(id="t1", name="N", args={}),))
    peer = AgentSendMessage(source="rev", text="finding")
    tr = ToolResult(call_id="t1", content="real")
    out = pair_and_dedup_tool_calls([am, peer, tr])
    assert out == [am, tr, peer]
    assert not any(
        isinstance(e, ToolResult) and e.content == "[interrupted]" for e in out
    )


def test_pair_and_dedup_user_message_is_a_hard_interrupt() -> None:
    """A ``UserMessage`` mid-tool-turn flushes pending calls (real interrupt).

    The counterpart to the agent-send defer: a human turn between an AM and
    its result models the Ctrl+C / mid-tool injection, so the open call is
    closed with a synthetic ``[interrupted]`` and a later result is an orphan.
    """
    am = AssistantMessage(tool_calls=(ToolCall(id="t1", name="N", args={}),))
    user = UserMessage(text="stop")
    tr = ToolResult(call_id="t1", content="late")
    out = pair_and_dedup_tool_calls([am, user, tr])
    assert [type(e).__name__ for e in out] == [
        "AssistantMessage",
        "ToolResult",
        "UserMessage",
    ]
    synth = out[1]
    assert isinstance(synth, ToolResult)
    assert synth.call_id == "t1"
    assert synth.content == "[interrupted]"
    assert synth.is_error


def test_splice_safe_repair_output_is_splice_valid() -> None:
    """``splice_safe_repair`` output constructs as a splice with no extra step.

    Dropping a hollow duplicate AM between two real AMs would strand them
    adjacent (role-alternation violation) if repair did not coalesce. The
    splice-safe composition pairs *and* coalesces, so callers pass its output
    straight to the validating ``ContextSplice`` constructor (F2).
    """
    am_setup = AssistantMessage(
        text="setup", tool_calls=(ToolCall(id="t1", name="x", args={}),)
    )
    tr_setup = ToolResult(call_id="t1", content="ok")
    am_a = AssistantMessage(text="a")
    am_hollow_dup = AssistantMessage(tool_calls=(ToolCall(id="t1", name="x", args={}),))
    am_b = AssistantMessage(text="b")
    out = splice_safe_repair([am_setup, tr_setup, am_a, am_hollow_dup, am_b])
    # No two adjacent assistant turns survive; constructs without raising.
    ContextSplice(
        ref=_REF,
        mask=(),
        insert_after=None,
        strategy="x",
        payload=tuple(out),
    )


def test_coalesce_roles_is_idempotent() -> None:
    """Re-coalescing an already-coalesced payload is a no-op."""
    payload = (
        UserMessage(text="u1"),
        AgentSendMessage(source="a", text="u2"),
        AssistantMessage(text="a1"),
        AssistantMessage(text="a2"),
    )
    once = coalesce_roles(payload)
    twice = coalesce_roles(once)
    assert once == twice


def test_coalesce_roles_idempotent_multi_source_labels_once() -> None:
    """A 3-way cross-source chain labels each segment exactly once.

    The 2-element idempotency case is idempotent "by accident" (the demoted
    result is a lone ``UserMessage`` with nothing left to merge). A 3-element
    chain forces real re-merge territory and pins that ``_labeled_text``'s
    ``startswith`` guard prevents double-labeling on a second pass.
    """
    payload = (
        UserMessage(text="u1"),
        AgentSendMessage(source="a", text="u2"),
        AgentSendMessage(source="b", text="u3"),
    )
    once = coalesce_roles(payload)
    twice = coalesce_roles(once)
    assert once == twice
    assert len(once) == 1
    merged = once[0]
    assert isinstance(merged, UserMessage)
    assert merged.text == "u1\n\n[from a]: u2\n\n[from b]: u3"
    assert "[from a]: [from a]:" not in merged.text


def test_merge_mask_ranges_preserves_gaps() -> None:
    """Sparse ranges stay sparse; a real gap is not filled."""
    merged = merge_mask_ranges((_mr(1, 1), _mr(10, 10)))
    assert merged == (_mr(1, 1), _mr(10, 10))


def test_merge_mask_ranges_coalesces_overlapping_and_touching() -> None:
    """Overlapping or adjacent (abutting) ranges merge into one."""
    # Touching: 1..3 and 4..5 abut (gap 0) -> one range.
    assert merge_mask_ranges((_mr(1, 3), _mr(4, 5))) == (_mr(1, 5),)
    # Overlapping: 1..5 and 3..8 -> one range.
    assert merge_mask_ranges((_mr(1, 5), _mr(3, 8))) == (_mr(1, 8),)


def test_merge_mask_ranges_partitions_by_session() -> None:
    """Ranges in different sessions never merge across the boundary."""
    a = _mr(1, 1, session_id="a")
    b = _mr(1, 1, session_id="b")
    merged = merge_mask_ranges((a, b))
    assert set(merged) == {a, b}


def test_mask_range_cross_session_unconstructable() -> None:
    """Issue#313 / review C5: a cross-session range cannot be built.

    The former runtime guard inside ``merge_mask_ranges`` is deleted; the
    illegal shape now fails at ``MaskRange.between`` construction instead.
    """
    a = TapeRef(session_id="a", ordinal=0)
    b = TapeRef(session_id="b", ordinal=1)
    with pytest.raises(InvalidPayloadError, match="crosses session"):
        MaskRange.between(a, b)


def _splice(
    *,
    mask: tuple[MaskRange, ...] = (),
    payload: tuple[ModelContextEvent, ...] = (),
    paired_externally: frozenset[str] = frozenset(),
) -> ContextSplice:
    """Build a splice with sensible defaults for terse tests."""
    return ContextSplice(
        ref=_REF,
        mask=mask,
        insert_after=None,
        payload=payload,
        strategy="test",
        paired_externally=paired_externally,
    )


def test_empty_payload_is_valid() -> None:
    """Pure-mask splices (no payload) construct cleanly."""
    splice = _splice()
    assert splice.payload == ()


def test_text_only_payload_is_valid() -> None:
    """Coalesce / summary payloads with alternating roles pass trivially."""
    splice = _splice(payload=(UserMessage(text="a"), AssistantMessage(text="b")))
    assert len(splice.payload) == 2


def test_payload_rejects_consecutive_same_role_messages() -> None:
    with pytest.raises(InvalidPayloadError, match="alternation"):
        _splice(payload=(UserMessage(text="a"), UserMessage(text="b")))


def test_paired_am_tr_block_is_valid() -> None:
    """An AM followed by its matching TR within payload passes."""
    am = AssistantMessage(
        text="",
        tool_calls=(ToolCall(id="t1", name="Bash", args={}),),
    )
    tr = ToolResult(call_id="t1", content="ok")
    splice = _splice(payload=(am, tr))
    assert len(splice.payload) == 2


def test_multi_call_am_block_is_valid() -> None:
    """AM with multiple tool_calls + all TRs in payload passes."""
    am = AssistantMessage(
        text="",
        tool_calls=(
            ToolCall(id="t1", name="Bash", args={}),
            ToolCall(id="t2", name="Bash", args={}),
        ),
    )
    splice = _splice(
        payload=(
            am,
            ToolResult(call_id="t1", content="a"),
            ToolResult(call_id="t2", content="b"),
        ),
    )
    assert len(splice.payload) == 3


def test_unpaired_am_in_payload_rejected() -> None:
    """An AM with tool_calls but no following TR fails validation."""
    am = AssistantMessage(
        text="",
        tool_calls=(ToolCall(id="t1", name="Bash", args={}),),
    )
    with pytest.raises(InvalidPayloadError, match="unpaired tool_call id"):
        _splice(payload=(am,))


def test_orphan_tr_in_payload_rejected() -> None:
    """A TR with no preceding AM in payload fails validation."""
    with pytest.raises(InvalidPayloadError, match="orphan ToolResult"):
        _splice(payload=(ToolResult(call_id="t1", content="ok"),))


def test_duplicate_tr_in_payload_rejected() -> None:
    """Two TRs sharing a call_id within one payload fail validation."""
    am = AssistantMessage(
        text="",
        tool_calls=(ToolCall(id="t1", name="Bash", args={}),),
    )
    with pytest.raises(InvalidPayloadError, match="duplicate ToolResult"):
        _splice(
            payload=(
                am,
                ToolResult(call_id="t1", content="ok"),
                ToolResult(call_id="t1", content="again"),
            ),
        )


def test_paired_externally_allows_orphan_tr() -> None:
    """Splice-style payload (single TR) is valid when declared external."""
    splice = _splice(
        payload=(ToolResult(call_id="t1", content="late"),),
        paired_externally=frozenset({"t1"}),
    )
    assert len(splice.payload) == 1


def test_paired_externally_allows_orphan_am() -> None:
    """Payload with AM whose tool_calls live elsewhere passes when declared."""
    am = AssistantMessage(
        text="",
        tool_calls=(ToolCall(id="t1", name="Bash", args={}),),
    )
    splice = _splice(payload=(am,), paired_externally=frozenset({"t1"}))
    assert len(splice.payload) == 1


def test_paired_externally_partial_still_rejects_unmatched() -> None:
    """Declaring one of two ids externally still rejects the other."""
    am = AssistantMessage(
        text="",
        tool_calls=(
            ToolCall(id="t1", name="Bash", args={}),
            ToolCall(id="t2", name="Bash", args={}),
        ),
    )
    with pytest.raises(InvalidPayloadError, match=r"unpaired tool_call id.*'t2'"):
        _splice(payload=(am,), paired_externally=frozenset({"t1"}))


def test_payload_rejects_user_external_tr_user_alternation() -> None:
    """An external-paired TR must not reset role tracking for a following user.

    A ``ToolResult`` declared ``paired_externally`` has its ``AssistantMessage``
    *outside* this payload, so locally no assistant turn opened. It must not
    license a following user-side turn: ``[user, external-TR, user]`` is two
    user-role turns separated only by a TR whose AM is elsewhere -- still a
    role-alternation violation (F7).
    """
    with pytest.raises(InvalidPayloadError, match="alternation"):
        _splice(
            payload=(
                UserMessage(text="a"),
                ToolResult(call_id="t1", content="late"),
                UserMessage(text="b"),
            ),
            paired_externally=frozenset({"t1"}),
        )


def test_coalesce_roles_assistant_merge_keeps_max_id() -> None:
    """Merging two AMs keeps the larger ``id`` so the id-counter never under-seeds.

    ``_seed_id_counter`` reseeds the global counter past the max ``id`` it scans
    in tape payloads. If an assistant merge kept only ``prior``'s (smaller) id
    and the originals are absent from the tape, a later mint could collide with
    the dropped ``entry.id``. Keep ``max(prior.id, entry.id)`` (F5).
    """
    a = AssistantMessage(text="one")
    b = AssistantMessage(text="two")
    lo, hi = (a, b) if a.id < b.id else (b, a)
    out = coalesce_roles((lo, hi))
    assert len(out) == 1
    merged = out[0]
    assert isinstance(merged, AssistantMessage)
    assert merged.id == hi.id


def test_coalesce_roles_user_merge_keeps_max_id() -> None:
    """User-side merge likewise keeps the larger ``id`` (F5)."""
    a = UserMessage(text="one")
    b = UserMessage(text="two")
    lo, hi = (a, b) if a.id < b.id else (b, a)
    out = coalesce_roles((lo, hi))
    assert len(out) == 1
    assert out[0].id == hi.id


def test_two_back_to_back_unpaired_ams_rejected() -> None:
    """Two consecutive AMs without TRs fails (the ceedc0fa pattern)."""
    am1 = AssistantMessage(
        text="",
        tool_calls=(ToolCall(id="t1", name="Bash", args={}),),
    )
    am2 = AssistantMessage(text="continuing")
    with pytest.raises(InvalidPayloadError, match=r"unpaired tool_call id.*'t1'"):
        _splice(payload=(am1, am2))


def test_mask_disjoint_within_splice_passes() -> None:
    """Two disjoint mask ranges in the same splice pass validation."""
    splice = _splice(mask=(_mr(1, 2), _mr(5, 7)))
    assert len(splice.mask) == 2


def test_mask_adjacent_ranges_within_splice_pass() -> None:
    """Adjacent (touching) ranges do not share a position; allowed."""
    splice = _splice(mask=(_mr(1, 3), _mr(4, 6)))
    assert len(splice.mask) == 2


def test_mask_overlap_within_splice_rejected() -> None:
    """Two mask ranges in the same splice that share a position fail."""
    with pytest.raises(InvalidPayloadError, match="mask ranges overlap"):
        _splice(mask=(_mr(1, 5), _mr(4, 8)))


def test_mask_inverted_range_unconstructable() -> None:
    """Issue#313: an inverted range cannot be built.

    The former ``_validate_mask_disjoint`` inverted check is deleted; the
    illegal shape now fails in ``MaskRange.__post_init__`` instead.
    """
    with pytest.raises(InvalidPayloadError, match="inverted"):
        MaskRange(session_id="s", lo=5, hi=3)


def test_mask_negative_ordinal_unconstructable() -> None:
    """A negative lower ordinal is rejected at the MaskRange trust boundary.

    Tape ordinals are minted monotonically from 0, so a negative endpoint is
    malformed wire/legacy data. Rejecting it in ``__post_init__`` keeps
    downstream ``contains`` / ``overlaps`` from silently honoring an
    out-of-range mask.
    """
    with pytest.raises(InvalidPayloadError, match="negative"):
        MaskRange(session_id="s", lo=-1, hi=3)


def test_replay_bypasses_validation() -> None:
    """``replay()`` accepts payloads and masks the validators would reject."""
    am = AssistantMessage(
        text="",
        tool_calls=(ToolCall(id="t1", name="Bash", args={}),),
    )
    splice = ContextSplice.replay(
        ref=_REF,
        mask=(_mr(1, 5), _mr(4, 8)),
        insert_after=None,
        payload=(am,),
        strategy="legacy",
    )
    assert splice.payload == (am,)
    assert splice.paired_externally == frozenset()
    assert len(splice.mask) == 2


def test_payload_rejects_duplicate_tool_call_id_across_messages() -> None:
    """D9: duplicate ``tool_call.id`` across two AMs in one payload fails.

    Pairing logic within a single payload was checked, but two AMs both
    declaring tool_call id ``t1`` (each properly paired with their own
    ToolResult) used to slip past validation. Providers reject the
    resulting payload at send time with confusing tool-pairing errors;
    catch at construct time instead.
    """
    am1 = AssistantMessage(
        text="",
        tool_calls=(ToolCall(id="t1", name="Bash", args={}),),
    )
    am2 = AssistantMessage(
        text="",
        tool_calls=(ToolCall(id="t1", name="Bash", args={}),),
    )
    with pytest.raises(InvalidPayloadError, match="duplicate tool_call"):
        _splice(
            payload=(
                am1,
                ToolResult(call_id="t1", content="a"),
                am2,
                ToolResult(call_id="t1", content="b"),
            ),
        )


def test_replay_tolerates_new_default_field_additions() -> None:
    """B16: ``replay`` survives a new dataclass field with a default.

    The hand-built ``values`` dict used to ``KeyError`` for any field
    not enumerated. The default-fallback path lets ``replay`` continue
    to construct legacy splices even when the dataclass grows new
    fields (so long as the new fields carry defaults).
    """
    splice = ContextSplice.replay(
        ref=_REF,
        mask=(),
        insert_after=None,
        payload=(UserMessage(text="x"),),
        strategy="legacy",
    )
    # Default-valued fields land at their dataclass defaults.
    assert splice.token_before == 0
    assert splice.token_after == 0
    assert splice.fallback_reason == ""
    assert splice.preserved_tail_count == 0
    assert splice.paired_externally == frozenset()


def test_replay_preserves_all_fields() -> None:
    """``replay()`` populates every dataclass field, including defaults."""
    splice = ContextSplice.replay(
        ref=_REF,
        mask=(_mr(1, 2),),
        insert_after=_ref(3),
        payload=(UserMessage(text="x"),),
        strategy="summary",
        token_before=10,
        token_after=20,
        fallback_reason="ran out",
        preserved_tail_count=3,
        paired_externally=frozenset({"q"}),
    )
    assert splice.token_before == 10
    assert splice.token_after == 20
    assert splice.fallback_reason == "ran out"
    assert splice.preserved_tail_count == 3
    assert splice.paired_externally == frozenset({"q"})


def test_paired_externally_does_not_hide_local_invalid_pair_order() -> None:
    """SAGENT-REV-003: ``paired_externally`` must not absolve local mis-ordering.

    A payload of ``AM(t1), User, TR(t1)`` is wire-invalid (the user
    turn between an AM with pending tool_calls and its matching TR
    breaks provider alternation). The validator used to silently accept
    it whenever ``t1`` appeared in ``paired_externally``, because the
    user-side branch drained ``pending`` via ``-= paired_externally``
    and the later TR was admitted through the orphan branch's external
    escape hatch. The field contract says the pair lives *outside*
    this payload, not "skip pairing checks for this id locally".
    """
    with pytest.raises(
        InvalidPayloadError,
        match=r"tool_call|alternation|orphan|paired_externally",
    ):
        _splice(
            payload=(
                AssistantMessage(tool_calls=(ToolCall(id="c1", name="t", args={}),)),
                UserMessage(text="interrupt"),
                ToolResult(call_id="c1", content="late"),
            ),
            paired_externally=frozenset({"c1"}),
        )


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
