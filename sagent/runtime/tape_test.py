"""Tests for tape records: refs, masks, splice validation, and payload repair."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import re

from hypothesis import (
    given,
    settings,
    strategies as st,
)

import pytest

from sagent.model.cost import TokenCost, TokenCount
from sagent.model.scripted import call, reply
from sagent.model.types import Response
from sagent.runtime import tape
from sagent.runtime.context import validate_context
from sagent.runtime.tape import (
    ContextSplice,
    InvalidPayloadError,
    MaskRange,
    ReferrableTapeEvent,
    TapeRecord,
    TapeRef,
    calls,
    coalesce_roles,
    full_tape_mask,
    mask_contains_ref,
    mask_ranges_overlap,
    merge_mask_ranges,
    splice_safe_repair,
    unpaired_call_ids,
)
from sagent.session.types import (
    AgentToAgentMessage,
    AssistantMessage,
    Attachment,
    SystemMessage,
    Thinking,
    ToolCall,
    ToolResult,
    TurnContext,
    UncategorizedToolResult,
    UserMessage,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from sagent.runtime.types import ModelContextEvent


_REF = TapeRef(session_id="s", ordinal=0)


def _exactly(message: str) -> str:
    return f"^{re.escape(message)}$"


def _unpaired(call_id: str) -> str:
    return _exactly(
        f"unpaired tool_call id(s) in payload: [{call_id!r}]"
        " (no matching ToolResult; not in paired_externally)",
    )


_ALTERNATION = _exactly("payload violates role alternation")
_BOTH = _exactly(
    "call_id 't1' appears as both ToolCall and ToolResult in payload yet is"
    " declared paired_externally; the partner cannot be both local and external",
)


def _user(text: str) -> UserMessage:
    return UserMessage(content=text)


def _peer(sender: str, text: str) -> AgentToAgentMessage:
    return AgentToAgentMessage(sender=sender, content=text)


def _result(call_id: str, text: str = "ok") -> UncategorizedToolResult:
    return UncategorizedToolResult(call_id=call_id, content=text)


def _calling(*ids: str) -> Response:
    return reply(calls=[call("Bash", cid) for cid in ids])


def _splice(
    *,
    mask: tuple[MaskRange, ...] = (),
    payload: tuple[ModelContextEvent, ...] = (),
    paired_externally: frozenset[str] = frozenset(),
) -> ContextSplice:
    return ContextSplice(
        ref=_REF,
        mask=mask,
        insert_after=None,
        payload=payload,
        strategy="test",
        paired_externally=paired_externally,
    )


def _texts(
    entries: tuple[ModelContextEvent, ...] | list[ModelContextEvent],
) -> list[str]:
    out: list[str] = []
    for entry in entries:
        if isinstance(entry, Response):
            out.append(
                "".join(
                    r.content or ""
                    for r in entry.records
                    if isinstance(r, AssistantMessage)
                ),
            )
        elif isinstance(entry, (UserMessage, AgentToAgentMessage, SystemMessage)):
            out.append(entry.content or "")
        elif isinstance(entry, UncategorizedToolResult):
            out.append(entry.text())
    return out


# --- TapeRef and MaskRange ---------------------------------------------


def test_tape_ref_rejects_a_negative_ordinal() -> None:
    with pytest.raises(
        InvalidPayloadError,
        match=_exactly("TapeRef ordinal must be a non-negative int, got -1"),
    ):
        _ = TapeRef(session_id="s", ordinal=-1)


def test_tape_ref_rejects_a_bool_ordinal() -> None:
    """``True`` is an ``int``; as ordinal 1 it collided with the record there."""
    with pytest.raises(
        InvalidPayloadError,
        match=_exactly("TapeRef ordinal must be a non-negative int, got True"),
    ):
        _ = TapeRef(session_id="s", ordinal=True)


def test_tape_ref_accepts_zero() -> None:
    assert TapeRef(session_id="s", ordinal=0).ordinal == 0


def test_mask_range_rejects_an_inverted_range() -> None:
    with pytest.raises(
        InvalidPayloadError,
        match=_exactly("MaskRange hi=3 < lo=5 is inverted"),
    ):
        _ = MaskRange(session_id="s", lo=5, hi=3)


def test_mask_range_rejects_a_negative_low() -> None:
    with pytest.raises(
        InvalidPayloadError,
        match=_exactly("MaskRange lo=-1 is negative"),
    ):
        _ = MaskRange(session_id="s", lo=-1, hi=3)


def test_mask_range_accepts_a_single_position() -> None:
    assert MaskRange(session_id="s", lo=0, hi=0).contains(_REF)


def test_mask_range_contains_only_its_own_session() -> None:
    r = MaskRange(session_id="a", lo=2, hi=4)
    assert r.contains(TapeRef(session_id="a", ordinal=2))
    assert r.contains(TapeRef(session_id="a", ordinal=4))
    assert not r.contains(TapeRef(session_id="a", ordinal=1))
    assert not r.contains(TapeRef(session_id="a", ordinal=5))
    assert not r.contains(TapeRef(session_id="b", ordinal=3))


def test_mask_range_overlap_is_session_scoped_and_inclusive() -> None:
    r = MaskRange(session_id="a", lo=2, hi=4)
    assert r.overlaps(MaskRange(session_id="a", lo=4, hi=9))
    assert r.overlaps(MaskRange(session_id="a", lo=0, hi=2))
    assert not r.overlaps(MaskRange(session_id="a", lo=5, hi=9))
    assert not r.overlaps(MaskRange(session_id="a", lo=0, hi=1))
    assert not r.overlaps(MaskRange(session_id="b", lo=2, hi=4))


def test_mask_contains_ref_checks_every_range() -> None:
    mask = (
        MaskRange(session_id="s", lo=0, hi=0),
        MaskRange(session_id="s", lo=5, hi=6),
    )
    assert mask_contains_ref(mask, TapeRef(session_id="s", ordinal=6))
    assert not mask_contains_ref(mask, TapeRef(session_id="s", ordinal=3))
    assert not mask_contains_ref((), _REF)


def test_mask_ranges_overlap_checks_every_pair() -> None:
    left = (
        MaskRange(session_id="s", lo=0, hi=0),
        MaskRange(session_id="s", lo=9, hi=9),
    )
    assert mask_ranges_overlap(left, (MaskRange(session_id="s", lo=8, hi=10),))
    assert not mask_ranges_overlap(left, (MaskRange(session_id="s", lo=1, hi=8),))
    assert not mask_ranges_overlap(left, ())


def test_full_tape_mask_has_one_range_per_session_sorted() -> None:
    records: list[TapeRecord] = [
        ReferrableTapeEvent(ref=TapeRef(session_id="z", ordinal=3), event=_user("a")),
        ReferrableTapeEvent(ref=TapeRef(session_id="", ordinal=0), event=_user("b")),
        ReferrableTapeEvent(ref=TapeRef(session_id="z", ordinal=7), event=_user("c")),
        ReferrableTapeEvent(ref=TapeRef(session_id="", ordinal=1), event=_user("d")),
    ]
    assert full_tape_mask(records) == (
        MaskRange(session_id="", lo=0, hi=1),
        MaskRange(session_id="z", lo=3, hi=7),
    )
    assert full_tape_mask([]) == ()


def test_merge_mask_ranges_preserves_gaps() -> None:
    sparse = (
        MaskRange(session_id="s", lo=3, hi=3),
        MaskRange(session_id="s", lo=1, hi=1),
    )
    assert merge_mask_ranges(sparse) == (
        MaskRange(session_id="s", lo=1, hi=1),
        MaskRange(session_id="s", lo=3, hi=3),
    )


def test_merge_mask_ranges_joins_touching_and_overlapping() -> None:
    assert merge_mask_ranges(
        (MaskRange(session_id="s", lo=1, hi=3), MaskRange(session_id="s", lo=4, hi=5)),
    ) == (MaskRange(session_id="s", lo=1, hi=5),)
    assert merge_mask_ranges(
        (MaskRange(session_id="s", lo=1, hi=8), MaskRange(session_id="s", lo=3, hi=5)),
    ) == (MaskRange(session_id="s", lo=1, hi=8),)


def test_merge_mask_ranges_never_joins_across_sessions() -> None:
    a = MaskRange(session_id="a", lo=1, hi=1)
    b = MaskRange(session_id="b", lo=1, hi=1)
    assert merge_mask_ranges((b, a)) == (a, b)


# --- ContextSplice validation --------------------------------------------


def test_payloads_that_alternate_construct() -> None:
    assert _splice().payload == ()
    assert len(_splice(payload=(_user("a"), reply("b"))).payload) == 2


def test_adjacent_user_side_turns_construct_for_the_request_to_merge() -> None:
    for payload in ((_user("a"), _user("b")), (_peer("A", "a"), _peer("B", "b"))):
        assert _splice(payload=payload).payload == payload


def test_two_adjacent_model_turns_are_rejected() -> None:
    with pytest.raises(InvalidPayloadError, match=_ALTERNATION):
        _ = _splice(payload=(reply("a"), reply("b")))


def test_a_paired_tool_round_constructs() -> None:
    splice = _splice(payload=(_calling("t1", "t2"), _result("t2"), _result("t1")))
    assert len(splice.payload) == 3


def test_an_unanswered_call_is_rejected() -> None:
    with pytest.raises(InvalidPayloadError, match=_unpaired("t1")):
        _ = _splice(payload=(_calling("t1"),))


def test_a_call_answered_after_a_user_turn_is_rejected() -> None:
    with pytest.raises(InvalidPayloadError, match=_unpaired("t1")):
        _ = _splice(payload=(_calling("t1"), _user("wait"), _result("t1")))


def test_a_call_unanswered_before_the_next_model_turn_is_rejected() -> None:
    with pytest.raises(InvalidPayloadError, match=_unpaired("t1")):
        _ = _splice(payload=(_calling("t1"), reply("continuing")))


def test_an_orphan_result_is_rejected() -> None:
    with pytest.raises(
        InvalidPayloadError,
        match=_exactly(
            "orphan ToolResult for call_id 't1' in payload (no matching ToolCall; not in paired_externally)",
        ),
    ):
        _ = _splice(payload=(_result("t1"),))


def test_a_duplicate_result_is_rejected() -> None:
    with pytest.raises(
        InvalidPayloadError,
        match=_exactly("duplicate ToolResult for call_id 't1' in payload"),
    ):
        _ = _splice(payload=(_calling("t1"), _result("t1"), _result("t1", "again")))


def test_one_call_id_in_two_model_turns_is_rejected() -> None:
    with pytest.raises(
        InvalidPayloadError,
        match=_exactly("duplicate tool_call id 't1' across model turns in payload"),
    ):
        _ = _splice(
            payload=(_calling("t1"), _result("t1"), _calling("t1"), _result("t1")),
        )


def test_externally_paired_result_and_call_construct() -> None:
    external = frozenset({"t1"})
    assert _splice(payload=(_result("t1"),), paired_externally=external).payload
    assert _splice(payload=(_calling("t1"),), paired_externally=external).payload


def test_an_external_declaration_covers_only_the_ids_it_names() -> None:
    with pytest.raises(InvalidPayloadError, match=_unpaired("t2")):
        _ = _splice(
            payload=(_calling("t1", "t2"),),
            paired_externally=frozenset({"t1"}),
        )


def test_an_id_paired_locally_cannot_also_be_external() -> None:
    external = frozenset({"t1"})
    with pytest.raises(InvalidPayloadError, match=_BOTH):
        _ = _splice(payload=(_calling("t1"), _result("t1")), paired_externally=external)
    with pytest.raises(InvalidPayloadError, match=_BOTH):
        _ = _splice(
            payload=(_user("u"), _result("t1"), _calling("t1")),
            paired_externally=external,
        )


def test_an_external_result_may_stand_between_two_user_turns() -> None:
    payload = (_user("a"), _result("t1"), _user("b"))
    splice = _splice(payload=payload, paired_externally=frozenset({"t1"}))
    assert splice.payload == payload


def test_an_external_result_closes_a_local_model_turn() -> None:
    splice = _splice(
        payload=(reply("a"), _result("t1"), reply("b")),
        paired_externally=frozenset({"t1"}),
    )
    assert len(splice.payload) == 3


def test_notes_are_neutral_to_alternation_and_pairing() -> None:
    payload = (
        _user("q"),
        SystemMessage(content="cwd: /x"),
        _calling("c1"),
        TurnContext(cwd="/x"),
        _result("c1"),
        SystemMessage(content="later"),
        reply("done"),
    )
    assert _splice(payload=payload).payload == payload
    with pytest.raises(InvalidPayloadError, match=_ALTERNATION):
        _ = _splice(payload=(reply("a"), SystemMessage(content="n"), reply("b")))


def test_disjoint_or_touching_mask_ranges_construct() -> None:
    splice = _splice(
        mask=(
            MaskRange(session_id="s", lo=1, hi=3),
            MaskRange(session_id="s", lo=4, hi=6),
            MaskRange(session_id="t", lo=4, hi=6),
        ),
    )
    assert len(splice.mask) == 3


def test_overlapping_mask_ranges_are_rejected() -> None:
    with pytest.raises(
        InvalidPayloadError,
        match=_exactly("mask ranges overlap in session 's'"),
    ):
        _ = _splice(
            mask=(
                MaskRange(session_id="s", lo=1, hi=5),
                MaskRange(session_id="s", lo=4, hi=8),
            ),
        )


# --- pair_and_dedup_tool_calls / splice_safe_repair ----------------------


def test_repair_answers_unanswered_calls_in_sorted_order() -> None:
    out = tape._pair_and_dedup_tool_calls(
        [_calling("c2", "c1", "c3"), _result("c3"), _user("done")],
    )
    assert _texts(out) == ["", "ok", "[interrupted]", "[interrupted]", "done"]
    synthesized = [
        e for e in out if isinstance(e, UncategorizedToolResult) and e.is_error is True
    ]
    assert [e.call_id for e in synthesized] == ["c1", "c2"]


def test_repair_answers_calls_left_open_at_the_end() -> None:
    out = tape._pair_and_dedup_tool_calls([_calling("c1")])
    assert [e.call_id for e in out if isinstance(e, UncategorizedToolResult)] == ["c1"]


def test_repair_drops_orphan_and_duplicate_results() -> None:
    out = tape._pair_and_dedup_tool_calls(
        [
            _result("ghost"),
            _calling("c1"),
            _result("c1", "first"),
            _result("c1", "second"),
        ],
    )
    assert _texts(out) == ["", "first"]


def test_repair_drops_a_call_id_an_earlier_turn_declared() -> None:
    first = reply("one", calls=[call("Bash", "c1")])
    again = reply("two", calls=[call("Bash", "c1"), call("Bash", "c2")])
    out = tape._pair_and_dedup_tool_calls([first, _result("c1"), again, _result("c2")])
    second = out[2]
    assert isinstance(second, Response)
    assert [c.call_id for c in calls(second.records)] == ["c2"]
    assert out[0] is first


def test_repair_drops_a_turn_left_empty() -> None:
    out = tape._pair_and_dedup_tool_calls(
        [_calling("c1"), _result("c1"), _calling("c1"), _result("c1")],
    )
    assert len(out) == 2


def test_a_user_turn_interrupts_an_open_call() -> None:
    out = tape._pair_and_dedup_tool_calls(
        [_calling("t1"), _user("stop"), _result("t1", "late")],
    )
    assert _texts(out) == ["", "[interrupted]", "stop"]


def test_an_agent_message_waits_for_the_open_call() -> None:
    """A peer's send is not an interrupt: the real result still lands."""
    am, peer, tr = _calling("t1"), _peer("rev", "finding"), _result("t1", "real")
    assert tape._pair_and_dedup_tool_calls([am, peer, tr]) == [am, tr, peer]


def test_held_agent_messages_keep_their_arrival_order() -> None:
    out = tape._pair_and_dedup_tool_calls(
        [
            _calling("c1"),
            _peer("A", "first"),
            _result("c1", "done"),
            _peer("A", "second"),
        ],
    )
    assert _texts(out) == ["", "done", "first", "second"]


def test_held_agent_messages_follow_an_interrupt() -> None:
    out = tape._pair_and_dedup_tool_calls(
        [_calling("c1"), _peer("A", "peer"), reply("next")],
    )
    assert _texts(out) == ["", "[interrupted]", "peer", "next"]


def test_repair_leaves_an_externally_paired_call_open() -> None:
    """Answering a still-running call ``[interrupted]`` tells the model it failed."""
    payload = (_user("[summary]"), _calling("live-1"))
    repaired = splice_safe_repair(payload, paired_externally=frozenset({"live-1"}))
    assert repaired == payload
    assert unpaired_call_ids(repaired) == frozenset({"live-1"})


def test_repair_is_idempotent() -> None:
    payload: list[ModelContextEvent] = [
        _calling("c1", "c2"),
        _peer("A", "x"),
        _result("c1"),
        _user("u"),
        _result("ghost"),
    ]
    once = tape._pair_and_dedup_tool_calls(payload)
    assert tape._pair_and_dedup_tool_calls(once) == once


def test_splice_safe_repair_of_a_turn_merged_across_a_note_is_a_fixed_point() -> None:
    """The merge carries the call to before the note; its answer follows the call."""
    history = [
        reply("a"),
        SystemMessage(content="n"),
        reply(calls=[call("Bash", "u0")]),
    ]
    once = splice_safe_repair(history)
    assert splice_safe_repair(once) == once
    assert [type(e).__name__ for e in once] == [
        "Response",
        "UncategorizedToolResult",
        "SystemMessage",
    ]


def test_splice_safe_repair_never_strands_two_model_turns() -> None:
    """Dropping a hollow duplicate between two turns leaves them adjacent."""
    out = splice_safe_repair(
        [_calling("t1"), _result("t1"), reply("a"), _calling("t1"), reply("b")],
    )
    assert _texts(out)[-1] == "ab"
    assert _splice(payload=out).payload == out


# --- coalesce_roles -------------------------------------------------------


def test_same_sender_user_turns_merge_with_attachments() -> None:
    a1 = Attachment(mime_descriptor="image/png", data=b"a")
    a2 = Attachment(mime_descriptor="image/png", data=b"b")
    out = coalesce_roles(
        (
            UserMessage(content="a", attachments=(a1,)),
            UserMessage(content="b", attachments=(a2,)),
        ),
    )
    assert out == (UserMessage(content="a\n\nb", attachments=(a1, a2)),)


def test_merging_tolerates_turns_without_text() -> None:
    picture = Attachment(mime_descriptor="image/png", data=b"p")
    assert coalesce_roles((UserMessage(attachments=(picture,)), _user("b"))) == (
        UserMessage(content="\n\nb", attachments=(picture,)),
    )
    assert coalesce_roles((_user("a"), UserMessage())) == (
        UserMessage(content="a\n\n"),
    )
    assert coalesce_roles((UserMessage(), _peer("A", "a"))) == (
        UserMessage(content="\n\n[from A]: a"),
    )


def test_a_merge_across_senders_keeps_the_attachments() -> None:
    picture = Attachment(mime_descriptor="image/png", data=b"p")
    merged = coalesce_roles(
        (
            _user("u"),
            AgentToAgentMessage(sender="A", content="a", attachments=(picture,)),
        ),
    )
    assert merged == (UserMessage(content="u\n\n[from A]: a", attachments=(picture,)),)


def test_one_peers_messages_merge_and_keep_their_type() -> None:
    out = coalesce_roles((_peer("A", "a"), _peer("A", "b")))
    assert out == (AgentToAgentMessage(sender="A", content="a\n\nb"),)


def test_mixed_senders_demote_to_a_labelled_user_turn() -> None:
    out = coalesce_roles((_user("u1"), _peer("a", "u2"), _peer("b", "u3")))
    assert out == (UserMessage(content="u1\n\n[from a]: u2\n\n[from b]: u3"),)
    assert coalesce_roles(out) == out


def test_user_turns_merge_across_a_note_and_the_note_stays_after() -> None:
    note = SystemMessage(content="cwd: /x")
    merged = coalesce_roles((_user("a"), note, _user("b")))
    assert merged == (UserMessage(content="a\n\nb"), note)
    assert merged[1] is note


def test_a_leading_note_does_not_merge() -> None:
    note = TurnContext(cwd="/x")
    assert coalesce_roles((note, _user("a"))) == (note, _user("a"))


def test_a_tool_result_blocks_a_model_turn_merge() -> None:
    payload = (reply("a"), _result("t"), reply("b"))
    assert coalesce_roles(payload) == payload


def test_model_turn_merge_sums_usage_and_keeps_the_last_stop() -> None:
    first = reply(
        "a",
        tokens=TokenCount(request=1, response=2),
        spend=TokenCost(request=0.5),
    )
    second = reply(
        "b",
        tokens=TokenCount(request=10),
        spend=TokenCost(response=0.25),
        stop_reason="max_tokens",
    )
    (merged,) = coalesce_roles((first, second))
    assert isinstance(merged, Response)
    assert merged.tokens == TokenCount(request=11, response=2)
    assert merged.spend == TokenCost(request=0.5, response=0.25)
    assert merged.stop_reason == "max_tokens"
    assert _texts([merged]) == ["ab"]


def test_model_turn_merge_dedupes_call_ids() -> None:
    tc = call("Bash", "t1")
    out = coalesce_roles(
        (reply("a", calls=[tc]), reply("b", calls=[tc, call("Bash", "t2")])),
    )
    (merged,) = out
    assert isinstance(merged, Response)
    assert [c.call_id for c in calls(merged.records)] == ["t1", "t2"]
    assert _splice(payload=(*out, _result("t1"), _result("t2"))).payload


def test_model_turn_merge_drops_signed_thinking_and_keeps_sealed() -> None:
    signed = Thinking(content="A", encrypted="sigA")
    sealed = Thinking(encrypted="blob")
    readable = Thinking(content="plain")
    first = Response(
        records=(signed, AssistantMessage(content="a")),
        tokens=TokenCount(),
        spend=TokenCost(),
        stop_reason="end_turn",
    )
    second = Response(
        records=(sealed, readable, AssistantMessage(content="b")),
        tokens=TokenCount(),
        spend=TokenCost(),
        stop_reason="end_turn",
    )
    (merged,) = coalesce_roles((first, second))
    assert isinstance(merged, Response)
    thinking = [r for r in merged.records if isinstance(r, Thinking)]
    assert thinking == [sealed, readable]


def test_coalesce_roles_is_idempotent() -> None:
    once = coalesce_roles((_user("u1"), _peer("a", "u2"), reply("a1"), reply("a2")))
    assert coalesce_roles(once) == once


# --- unpaired_call_ids, same_sender, calls --------------------------------


def test_unpaired_call_ids_names_both_missing_sides() -> None:
    payload = (
        _result("tr-only"),
        _calling("open", "closed"),
        SystemMessage(content="n"),
        _result("closed"),
        _user("closes the turn"),
        _calling("last"),
    )
    assert unpaired_call_ids(payload) == frozenset({"tr-only", "open", "last"})


def test_unpaired_call_ids_closes_pending_at_the_next_model_turn() -> None:
    assert unpaired_call_ids((_calling("a"), _calling("b"), _result("b"))) == frozenset(
        {"a"},
    )


def test_unpaired_call_ids_of_a_paired_payload_is_empty() -> None:
    assert unpaired_call_ids((_calling("a"), _result("a"))) == frozenset()


def test_same_sender() -> None:
    assert tape._same_sender(_user("a"), _user("b"))
    assert tape._same_sender(_peer("A", "a"), _peer("A", "b"))
    assert not tape._same_sender(_peer("A", "a"), _peer("B", "b"))
    assert not tape._same_sender(_user("a"), _peer("A", "b"))
    assert not tape._same_sender(_peer("A", "a"), _user("b"))


def test_calls_returns_tool_calls_in_order() -> None:
    a, b = call("X", "a"), call("Y", "b")
    records: tuple[AssistantMessage | ToolCall, ...] = (
        a,
        AssistantMessage(content="t"),
        b,
    )
    assert calls(records) == (a, b)


# --- Properties -------------------------------------------------------------
#
# Every input shape the repairs and the coalescer may meet: either side's turns,
# tool results, notes, and call ids drawn from a pool small enough that turns and
# results collide. A model turn never repeats an id within itself (``Runtime._reply``
# refuses such a reply), so the strategy draws each turn's ids unique.


def _call_ids() -> st.SearchStrategy[str]:
    return st.sampled_from(("c0", "c1", "c2", "c3"))


def _histories() -> st.SearchStrategy[list[ModelContextEvent]]:
    text = st.text(alphabet="ab ", max_size=6)
    entry: st.SearchStrategy[ModelContextEvent] = st.one_of(
        st.builds(_user, text),
        st.builds(_peer, st.sampled_from(("A", "B")), text),
        st.builds(
            _turn,
            text,
            st.lists(_call_ids(), max_size=3, unique=True),
            signed=st.booleans(),
        ),
        st.builds(_result, _call_ids(), text),
        st.one_of(
            st.builds(SystemMessage, content=text),
            st.just(TurnContext(cwd="/x")),
        ),
    )
    return st.lists(entry, max_size=10)


def _turn(text: str, ids: list[str], *, signed: bool) -> Response:
    """Return a model turn saying ``text`` and calling ``ids``, its thinking signed."""
    turn = reply(text, calls=[call("Bash", cid) for cid in ids])
    if not signed:
        return turn
    return replace(turn, records=(Thinking(content="t", encrypted="s"), *turn.records))


# A second witness for the pairing half of ``validate_context``, written apart from
# the code under test so a property asserts against something other than itself.
def _pairing_violation(entries: Sequence[ModelContextEvent]) -> str | None:
    """Return the first tool-pairing violation in ``entries``, or ``None``."""
    pending: set[str] = set()
    answered: set[str] = set()
    for entry in entries:
        if isinstance(entry, (SystemMessage, TurnContext)):
            continue
        if isinstance(entry, Response):
            if pending:
                return f"a model turn while {sorted(pending)} are open"
            pending = {c.call_id for c in calls(entry.records)}
        elif isinstance(entry, ToolResult):
            if entry.call_id in answered or entry.call_id not in pending:
                return f"a result for {entry.call_id!r} that answers no open call"
            pending.discard(entry.call_id)
            answered.add(entry.call_id)
        elif pending:
            return f"a user-side turn while {sorted(pending)} are open"
    return f"{sorted(pending)} open at the end" if pending else None


def _others(entries: Sequence[ModelContextEvent]) -> list[ModelContextEvent]:
    """Return the entries that are neither side's turn: results and notes."""
    return [
        e
        for e in entries
        if not isinstance(e, (UserMessage, AgentToAgentMessage, Response))
    ]


def _model_said(entries: Sequence[ModelContextEvent]) -> list[AssistantMessage]:
    return [
        r
        for e in entries
        if isinstance(e, Response)
        for r in e.records
        if isinstance(r, AssistantMessage)
    ]


def _user_side(
    entries: Sequence[ModelContextEvent],
) -> list[UserMessage | AgentToAgentMessage]:
    return [e for e in entries if isinstance(e, (UserMessage, AgentToAgentMessage))]


@settings(max_examples=60, deadline=None)
@given(_histories())
def test_coalescing_still_names_every_peer(history: list[ModelContextEvent]) -> None:
    """A merge across senders demotes to a user turn; each part keeps its label."""
    out = coalesce_roles(history)
    senders = {e.sender for e in out if isinstance(e, AgentToAgentMessage)}
    labeled = "".join(e.content or "" for e in out if isinstance(e, UserMessage))
    for entry in history:
        if isinstance(entry, AgentToAgentMessage):
            assert entry.sender in senders or f"[from {entry.sender}]: " in labeled


@settings(max_examples=60, deadline=None)
@given(_histories())
def test_coalescing_loses_no_user_side_text(history: list[ModelContextEvent]) -> None:
    out = [e.content or "" for e in _user_side(coalesce_roles(history))]
    given_texts = [e.content or "" for e in _user_side(history)]
    assert all(any(text in merged for merged in out) for text in given_texts)
    assert sum(map(len, out)) >= sum(map(len, given_texts))


@settings(max_examples=60, deadline=None)
@given(_histories())
def test_coalescing_keeps_the_order_of_results_notes_and_model_text(
    history: list[ModelContextEvent],
) -> None:
    out = coalesce_roles(history)
    assert _others(out) == _others(history)
    assert _model_said(out) == _model_said(history)


@settings(max_examples=60, deadline=None)
@given(_histories())
def test_coalescing_is_idempotent(history: list[ModelContextEvent]) -> None:
    once = coalesce_roles(history)
    assert coalesce_roles(once) == once


@settings(max_examples=60, deadline=None)
@given(_histories())
def test_repair_answers_every_call_exactly_once(
    history: list[ModelContextEvent],
) -> None:
    """Whatever the input, no call is left open, answered twice, or interleaved."""
    assert _pairing_violation(tape._pair_and_dedup_tool_calls(history)) is None


@settings(max_examples=60, deadline=None)
@given(_histories())
def test_splice_safe_repair_always_builds_a_splice(
    history: list[ModelContextEvent],
) -> None:
    """The rescue's payload must construct for ANY input, or the session wedges."""
    payload = splice_safe_repair(history)
    assert unpaired_call_ids(payload) == frozenset()
    assert _splice(payload=payload).payload == payload


@settings(max_examples=60, deadline=None)
@given(_histories())
def test_splice_safe_repair_is_idempotent(history: list[ModelContextEvent]) -> None:
    """A repair that is not a fixed point rewrites a valid context on every pass."""
    once = splice_safe_repair(history)
    assert splice_safe_repair(once) == once


def test_turns_left_empty_by_a_merge_bring_their_neighbors_together() -> None:
    """Two signed turns with no text merge to nothing; the user turns either side meet."""
    history = [
        _user("a"),
        _turn("", [], signed=True),
        _turn("", [], signed=True),
        _user("b"),
    ]
    once = splice_safe_repair(history)
    assert once == (_user("a\n\nb"),)
    assert splice_safe_repair(once) == once


def test_a_repair_that_never_settles_stops_after_a_pass_per_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A merge that kept changing its output would otherwise loop forever."""
    passes: list[int] = []

    def restless(payload: Sequence[ModelContextEvent]) -> tuple[ModelContextEvent, ...]:
        passes.append(len(passes))
        return (_user(str(len(passes))), *payload[1:])

    monkeypatch.setattr(tape, "coalesce_roles", restless)
    history = [_user("a"), _user("b"), _user("c")]
    assert splice_safe_repair(history)[1:] == tuple(history[1:])
    assert len(passes) == len(history)


@settings(max_examples=60, deadline=None)
@given(_histories())
def test_splice_safe_repair_answers_only_calls_that_were_made(
    history: list[ModelContextEvent],
) -> None:
    made = {
        c.call_id for e in history if isinstance(e, Response) for c in calls(e.records)
    }
    answered = {
        e.call_id for e in splice_safe_repair(history) if isinstance(e, ToolResult)
    }
    assert answered <= made


@settings(max_examples=60, deadline=None)
@given(_histories())
def test_splice_safe_repair_output_is_a_valid_context(
    history: list[ModelContextEvent],
) -> None:
    validate_context(splice_safe_repair(history))


@settings(max_examples=60, deadline=None)
@given(_histories(), st.frozensets(_call_ids()))
def test_repair_leaves_external_calls_open_and_declares_only_them(
    history: list[ModelContextEvent],
    external: frozenset[str],
) -> None:
    """An external call's result lands later; ``[interrupted]`` would call it dead."""
    payload = splice_safe_repair(history, paired_externally=external)
    interrupted = {
        e.call_id
        for e in payload
        if isinstance(e, UncategorizedToolResult)
        and e.is_error is True
        and e.content == "[interrupted]"
    }
    assert not interrupted & external
    still_open = unpaired_call_ids(payload)
    assert still_open <= external
    assert _splice(payload=payload, paired_externally=still_open).payload == payload


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
