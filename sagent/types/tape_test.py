"""Tests for ``ContextSplice.__post_init__`` payload and mask validation.

The validators enforce structural correctness at construct time so
producer bugs fail loudly at construction rather than silently at
resolve. Mask-disjointness is checked per-splice; cross-splice
mask-overlap (the once-overwrite rule) is enforced separately in
``AgentRuntime.append_splice``.
"""

from __future__ import annotations

import pytest

from sagent.types.history import (
    AssistantMessage,
    HistoryEntry,
    ToolCall,
    ToolResult,
    UserMessage,
)
from sagent.types.tape import (
    ContextSplice,
    InvalidPayloadError,
    TapeRef,
)


_REF = TapeRef(session_id="s", ordinal=0)


def _ref(ordinal: int) -> TapeRef:
    """Test helper: build a TapeRef with the standard session id."""
    return TapeRef(session_id="s", ordinal=ordinal)


def _splice(
    *,
    mask: tuple[tuple[TapeRef, TapeRef], ...] = (),
    payload: tuple[HistoryEntry, ...] = (),
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
    """Coalesce / summary payloads with no AM/TR pass trivially."""
    splice = _splice(payload=(UserMessage(text="a"), UserMessage(text="b")))
    assert len(splice.payload) == 2


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
    splice = _splice(mask=((_ref(1), _ref(2)), (_ref(5), _ref(7))))
    assert len(splice.mask) == 2


def test_mask_adjacent_ranges_within_splice_pass() -> None:
    """Adjacent (touching) ranges do not share a position; allowed."""
    splice = _splice(mask=((_ref(1), _ref(3)), (_ref(4), _ref(6))))
    assert len(splice.mask) == 2


def test_mask_overlap_within_splice_rejected() -> None:
    """Two mask ranges in the same splice that share a position fail."""
    with pytest.raises(InvalidPayloadError, match="mask ranges overlap"):
        _splice(mask=((_ref(1), _ref(5)), (_ref(4), _ref(8))))


def test_mask_inverted_range_rejected() -> None:
    """A mask range whose end ordinal is below the start fails."""
    with pytest.raises(InvalidPayloadError, match="inverted"):
        _splice(mask=((_ref(5), _ref(3)),))


def test_replay_bypasses_validation() -> None:
    """``replay()`` accepts payloads and masks the validators would reject."""
    am = AssistantMessage(
        text="",
        tool_calls=(ToolCall(id="t1", name="Bash", args={}),),
    )
    splice = ContextSplice.replay(
        ref=_REF,
        mask=((_ref(1), _ref(5)), (_ref(4), _ref(8))),
        insert_after=None,
        payload=(am,),
        strategy="legacy",
    )
    assert splice.payload == (am,)
    assert splice.paired_externally == frozenset()
    assert len(splice.mask) == 2


def test_replay_preserves_all_fields() -> None:
    """``replay()`` populates every dataclass field, including defaults."""
    splice = ContextSplice.replay(
        ref=_REF,
        mask=((_ref(1), _ref(2)),),
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


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
