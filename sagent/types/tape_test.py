"""Tests for ``ContextOverride.__post_init__`` payload validation.

The validator is the structural enforcement that replaces the prior
runtime-side phase 1 repair. Producer correctness fails loudly at
construct rather than silently at resolve.
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
    ContextOverride,
    InvalidPayloadError,
    TapeRef,
)


_REF = TapeRef(session_id="s", ordinal=0)


def _ov(
    *,
    payload: tuple[HistoryEntry, ...] = (),
    paired_externally: frozenset[str] = frozenset(),
) -> ContextOverride:
    """Build an override with sensible defaults for terse tests."""
    return ContextOverride(
        ref=_REF,
        suppresses=(),
        inject_after=None,
        payload=payload,
        strategy="test",
        paired_externally=paired_externally,
    )


def test_empty_payload_is_valid() -> None:
    """Suppress-only overrides (no payload) construct cleanly."""
    ov = _ov()
    assert ov.payload == ()


def test_text_only_payload_is_valid() -> None:
    """Coalesce / summary payloads with no AM/TR pass trivially."""
    ov = _ov(payload=(UserMessage(text="a"), UserMessage(text="b")))
    assert len(ov.payload) == 2


def test_paired_am_tr_block_is_valid() -> None:
    """An AM followed by its matching TR within payload passes."""
    am = AssistantMessage(
        text="",
        tool_calls=(ToolCall(id="t1", name="Bash", args={}),),
    )
    tr = ToolResult(call_id="t1", content="ok")
    ov = _ov(payload=(am, tr))
    assert len(ov.payload) == 2


def test_multi_call_am_block_is_valid() -> None:
    """AM with multiple tool_calls + all TRs in payload passes."""
    am = AssistantMessage(
        text="",
        tool_calls=(
            ToolCall(id="t1", name="Bash", args={}),
            ToolCall(id="t2", name="Bash", args={}),
        ),
    )
    ov = _ov(
        payload=(
            am,
            ToolResult(call_id="t1", content="a"),
            ToolResult(call_id="t2", content="b"),
        ),
    )
    assert len(ov.payload) == 3


def test_unpaired_am_in_payload_rejected() -> None:
    """An AM with tool_calls but no following TR fails validation."""
    am = AssistantMessage(
        text="",
        tool_calls=(ToolCall(id="t1", name="Bash", args={}),),
    )
    with pytest.raises(InvalidPayloadError, match="unpaired tool_call id"):
        _ov(payload=(am,))


def test_orphan_tr_in_payload_rejected() -> None:
    """A TR with no preceding AM in payload fails validation."""
    with pytest.raises(InvalidPayloadError, match="orphan ToolResult"):
        _ov(payload=(ToolResult(call_id="t1", content="ok"),))


def test_duplicate_tr_in_payload_rejected() -> None:
    """Two TRs sharing a call_id within one payload fail validation."""
    am = AssistantMessage(
        text="",
        tool_calls=(ToolCall(id="t1", name="Bash", args={}),),
    )
    with pytest.raises(InvalidPayloadError, match="duplicate ToolResult"):
        _ov(
            payload=(
                am,
                ToolResult(call_id="t1", content="ok"),
                ToolResult(call_id="t1", content="again"),
            ),
        )


def test_paired_externally_allows_orphan_tr() -> None:
    """Splice-style payload (single TR) is valid when declared external."""
    ov = _ov(
        payload=(ToolResult(call_id="t1", content="late"),),
        paired_externally=frozenset({"t1"}),
    )
    assert len(ov.payload) == 1


def test_paired_externally_allows_orphan_am() -> None:
    """Microcompact-style AM payload (no TR) is valid when declared external."""
    am = AssistantMessage(
        text="",
        tool_calls=(ToolCall(id="t1", name="Bash", args={}),),
    )
    ov = _ov(payload=(am,), paired_externally=frozenset({"t1"}))
    assert len(ov.payload) == 1


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
        _ov(payload=(am,), paired_externally=frozenset({"t1"}))


def test_two_back_to_back_unpaired_ams_rejected() -> None:
    """Two consecutive AMs without TRs (the ceedc0fa pattern) fails."""
    am1 = AssistantMessage(
        text="",
        tool_calls=(ToolCall(id="t1", name="Bash", args={}),),
    )
    am2 = AssistantMessage(text="continuing")
    with pytest.raises(InvalidPayloadError, match=r"unpaired tool_call id.*'t1'"):
        _ov(payload=(am1, am2))


def test_replay_bypasses_validation() -> None:
    """``replay()`` accepts payloads the validator would reject."""
    am = AssistantMessage(
        text="",
        tool_calls=(ToolCall(id="t1", name="Bash", args={}),),
    )
    ov = ContextOverride.replay(
        ref=_REF,
        suppresses=(),
        inject_after=None,
        payload=(am,),
        strategy="legacy",
    )
    assert ov.payload == (am,)
    assert ov.paired_externally == frozenset()


def test_replay_preserves_all_fields() -> None:
    """``replay()`` populates every dataclass field, including defaults."""
    ov = ContextOverride.replay(
        ref=_REF,
        suppresses=(TapeRef(session_id="s", ordinal=1),),
        inject_after=TapeRef(session_id="s", ordinal=2),
        payload=(UserMessage(text="x"),),
        strategy="summary",
        barrier=True,
        token_before=10,
        token_after=20,
        fallback_reason="ran out",
        preserved_tail_count=3,
        paired_externally=frozenset({"q"}),
    )
    assert ov.barrier
    assert ov.token_before == 10
    assert ov.token_after == 20
    assert ov.fallback_reason == "ran out"
    assert ov.preserved_tail_count == 3
    assert ov.paired_externally == frozenset({"q"})


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
