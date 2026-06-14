"""Tests for ``compaction.scrunch``: post-hoc oldest-to-newest scrunch maneuver.

The scrunch maneuver rescues a tape whose resolved view exceeds the
active model's input window. It runs ``compactor.compact`` over the
oldest partition, then the next-oldest, and so on, stopping as soon as
the resolved view fits. The recompact stage folds prior summaries
together when the uncompacted tail is empty.

Strategy: pre-plan the partition layout (token-bounded, pair-safe),
then execute each partition as a separate compaction call. Stop early
once the running estimate dips under target.

This test file pins the contract of both the planner (pure function)
and the executor (calls the compactor once per partition).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import cast, override

import pytest

from sagent.compaction.scrunch import (
    ScrunchTooLargeError,
    plan_scrunch,
    scrunch_to_fit,
)
from sagent.testing import MockModelCaps
from sagent.types.model import Model
from sagent.types.runtime import (
    AssistantMessage,
    ModelContextEvent,
    ToolCall,
    ToolResult,
    UserMessage,
)
from sagent.types.tape import (
    ContextSplice,
    ReferrableTapeEvent,
    TapeRecord,
    TapeRef,
    full_tape_mask,
)


def _stub_model() -> Model:
    return cast(Model, MockModelCaps())


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _user(text: str) -> UserMessage:
    return UserMessage(text=text)


def _assistant(
    text: str = "", tool_calls: tuple[ToolCall, ...] = ()
) -> AssistantMessage:
    return AssistantMessage(text=text, tool_calls=tool_calls)


def _tool_result(call_id: str, content: str) -> ToolResult:
    return ToolResult(call_id=call_id, content=content)


def _tape_from(history: list[ModelContextEvent]) -> list[TapeRecord]:
    return [
        ReferrableTapeEvent(ref=TapeRef(session_id="t", ordinal=i), event=e)
        for i, e in enumerate(history)
    ]


def _ref_factory(start: int) -> Callable[[], TapeRef]:
    n = [start]

    def mint() -> TapeRef:
        ref = TapeRef(session_id="t", ordinal=n[0])
        n[0] += 1
        return ref

    return mint


@dataclass(slots=True, kw_only=True)
class _SizedCompactor:
    """Deterministic compactor for scrunch tests.

    Each ``compact`` call produces a single :class:`UserMessage` whose
    text length is exactly ``summary_chars``. Records every call so
    tests can assert partition layout and ordering.
    """

    summary_chars: int = 40
    calls: list[Sequence[ModelContextEvent]] = field(default_factory=list)

    async def compact(
        self,
        tape: Sequence[TapeRecord],
        context: Sequence[ModelContextEvent],
        model: object,
        mint_ref: Callable[[], TapeRef],
        custom_instructions: str | None = None,
    ) -> ContextSplice:
        del model, custom_instructions
        self.calls.append(list(context))
        return ContextSplice(
            ref=mint_ref(),
            mask=full_tape_mask(tape),
            insert_after=None,
            payload=(UserMessage(text="S" * self.summary_chars),),
            strategy="summary",
        )


# ---------------------------------------------------------------------------
# Planner: pure function, no model calls.
# ---------------------------------------------------------------------------


def test_plan_scrunch_already_fits_returns_empty_plan() -> None:
    """No partitions when estimated tokens already <= target."""
    context: list[ModelContextEvent] = [_user("x" * 40)]  # ~10 tokens
    plan = plan_scrunch(
        context=context,
        model=_stub_model(),
        target_input_tokens=1000,
        max_partition_tokens=500,
    )
    assert plan.partitions == ()


def test_plan_scrunch_one_oversized_partition_returns_single_pass() -> None:
    """When the deficit fits in one partition, plan one pass."""
    # 4 user messages, each ~25 tokens; total ~100 tokens; target 50.
    context: list[ModelContextEvent] = [_user("x" * 100) for _ in range(4)]
    plan = plan_scrunch(
        context=context,
        model=_stub_model(),
        target_input_tokens=50,
        max_partition_tokens=200,
        summary_size_estimate_tokens=2,
    )
    # Must produce at least one partition; first partition covers oldest.
    assert len(plan.partitions) >= 1
    first = plan.partitions[0]
    assert first.start == 0
    assert first.stop >= 1


def test_plan_scrunch_multiple_partitions_when_budget_too_small() -> None:
    """Many small partitions when each partition can carry little."""
    # 10 messages of ~25 tokens each = ~250 tokens total. Target 50.
    # Each partition must fit ~50 tokens, so 2 messages per partition.
    context: list[ModelContextEvent] = [_user("x" * 100) for _ in range(10)]
    plan = plan_scrunch(
        context=context,
        model=_stub_model(),
        target_input_tokens=50,
        max_partition_tokens=50,
        summary_size_estimate_tokens=2,
    )
    assert len(plan.partitions) >= 3
    # Partitions are non-overlapping and ordered from oldest.
    for left, right in zip(plan.partitions, plan.partitions[1:], strict=False):
        assert left.stop <= right.start


def test_plan_scrunch_partition_is_pair_safe_grows_to_include_tool_result() -> None:
    """A partition ending mid-AM/TR pair grows to include the TR.

    The partition's stop index must land on a position where every AM
    in the partition has its matching ToolResult also in the partition.
    """
    context: list[ModelContextEvent] = [
        _user("u0"),
        _assistant(tool_calls=(ToolCall(id="c1", name="t", args={}),)),
        _tool_result("c1", "r1" * 200),  # large; partition would want to stop here
        _user("u3"),
        _user("u4"),
    ]
    plan = plan_scrunch(
        context=context,
        model=_stub_model(),
        target_input_tokens=20,
        max_partition_tokens=200,
        summary_size_estimate_tokens=2,
    )
    assert len(plan.partitions) >= 1
    # First partition must include the AM/TR pair together (or neither).
    first = plan.partitions[0]
    indices = set(range(first.start, first.stop))
    if 1 in indices:
        assert 2 in indices, (
            f"partition {first!r} splits AM(c1) from its matching TR(c1)"
        )


def test_plan_scrunch_stops_allocating_once_deficit_covered() -> None:
    """Don't plan more passes than needed.

    With a small deficit, one partition should suffice even if there
    are many candidate messages.
    """
    # 8 messages, ~25 tokens each = 200 tokens total. Target 175 (~deficit 25).
    # One partition of ~50 tokens covers it.
    context: list[ModelContextEvent] = [_user("x" * 100) for _ in range(8)]
    plan = plan_scrunch(
        context=context,
        model=_stub_model(),
        target_input_tokens=175,
        max_partition_tokens=500,
        summary_size_estimate_tokens=2,
    )
    total_covered = sum(p.stop - p.start for p in plan.partitions)
    assert total_covered < len(context), (
        f"planner allocated {total_covered} of {len(context)} entries; "
        f"deficit was small"
    )


def test_plan_scrunch_rejects_zero_partition_cap() -> None:
    with pytest.raises(ValueError, match=r"max_partition_tokens"):
        _ = plan_scrunch(
            context=[],
            model=_stub_model(),
            target_input_tokens=0,
            max_partition_tokens=0,
        )


def test_plan_scrunch_rejects_negative_summary_estimate() -> None:
    with pytest.raises(ValueError, match=r"summary_size_estimate_tokens"):
        _ = plan_scrunch(
            context=[],
            model=_stub_model(),
            target_input_tokens=0,
            max_partition_tokens=10,
            summary_size_estimate_tokens=-1,
        )


def test_plan_scrunch_rejects_summary_estimate_at_or_above_cap() -> None:
    # A summary estimate >= the per-call cap forces every partition past
    # the cap via the forward-progress floor -- a degenerate config.
    with pytest.raises(ValueError, match=r"< max_partition_tokens"):
        _ = plan_scrunch(
            context=[_user("x" * 100)],
            model=_stub_model(),
            target_input_tokens=1,
            max_partition_tokens=10,
            summary_size_estimate_tokens=10,
        )


def test_plan_scrunch_measures_tokens_via_model_not_chars() -> None:
    """The planner must size partitions by ``model.approx_text_tokens``,
    not by a char count over a fixed ratio.

    A model that tokenizes at 1 token per 2 chars sees a 200-char message
    as 100 tokens; under the old ``chars // 4`` it would have been 50.
    With ``target=60`` the 100-token message is over budget and must be
    partitioned; a chars//4 planner would have called it already-fitting.
    """

    @dataclass(slots=True, kw_only=True)
    class _HalfModel(MockModelCaps):
        model_id: str = "half"
        max_request_tokens: int = 200_000

        @override
        def approx_text_tokens(self, text: str) -> int:  # 2 chars/token
            return len(text) // 2

    context: list[ModelContextEvent] = [_user("x" * 200)]  # 100 tokens at 2 c/tok
    plan = plan_scrunch(
        context=context,
        model=cast(Model, _HalfModel()),
        target_input_tokens=60,
        max_partition_tokens=500,
        summary_size_estimate_tokens=2,
    )
    # 100 tokens > 60 target -> must plan a partition. chars//4 = 50 would not.
    assert len(plan.partitions) >= 1


# ---------------------------------------------------------------------------
# Executor: runs the producer compactor per partition.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scrunch_to_fit_already_fits_returns_no_splices() -> None:
    """Resolved view already under target -> no work."""
    context: list[ModelContextEvent] = [_user("ok")]
    tape = _tape_from(context)
    compactor = _SizedCompactor()
    result = await scrunch_to_fit(
        context=context,
        tape=tape,
        model=_stub_model(),
        compactor=compactor,
        mint_ref=_ref_factory(len(tape)),
        target_input_tokens=1000,
        max_partition_tokens=500,
    )
    assert result.splices == ()
    # Already-fits: the view is the input verbatim.
    assert result.view == tuple(context)
    assert compactor.calls == []


@pytest.mark.asyncio
async def test_scrunch_to_fit_one_pass_when_one_partition_suffices() -> None:
    """One scrunch pass produces one splice when the deficit fits in one."""
    # 4 messages, ~25 tokens each = ~100 tokens total. Target 50.
    context: list[ModelContextEvent] = [_user("x" * 100) for _ in range(4)]
    tape = _tape_from(context)
    compactor = _SizedCompactor(summary_chars=20)
    result = await scrunch_to_fit(
        context=context,
        tape=tape,
        model=_stub_model(),
        compactor=compactor,
        mint_ref=_ref_factory(len(tape)),
        target_input_tokens=50,
        max_partition_tokens=500,
        summary_size_estimate_tokens=5,
    )
    assert len(result.splices) >= 1
    assert len(compactor.calls) == len(result.splices)
    # First call's first entry must be the oldest message in context.
    first_call = compactor.calls[0]
    assert first_call[0] is context[0]


@pytest.mark.asyncio
async def test_scrunch_to_fit_returns_splices_in_order() -> None:
    """Splices come back in the order they were produced (oldest first)."""
    context: list[ModelContextEvent] = [_user("x" * 100) for _ in range(8)]
    tape = _tape_from(context)
    compactor = _SizedCompactor(summary_chars=20)
    result = await scrunch_to_fit(
        context=context,
        tape=tape,
        model=_stub_model(),
        compactor=compactor,
        mint_ref=_ref_factory(len(tape)),
        target_input_tokens=40,
        max_partition_tokens=400,
        summary_size_estimate_tokens=5,
    )
    refs = [s.ref.ordinal for s in result.splices]
    assert refs == sorted(refs)


@pytest.mark.asyncio
async def test_scrunch_to_fit_stops_early_once_fits() -> None:
    """Executor stops the moment the running estimate dips below target."""
    # 6 big messages, small summary chars -> few passes needed.
    context: list[ModelContextEvent] = [_user("x" * 1000) for _ in range(6)]
    tape = _tape_from(context)
    compactor = _SizedCompactor(summary_chars=10)
    result = await scrunch_to_fit(
        context=context,
        tape=tape,
        model=_stub_model(),
        compactor=compactor,
        mint_ref=_ref_factory(len(tape)),
        target_input_tokens=500,
        max_partition_tokens=2_000,
        summary_size_estimate_tokens=2,
    )
    # We don't pin an exact count; we pin "fewer than the worst case".
    assert len(result.splices) < len(context)


# ---------------------------------------------------------------------------
# Pathological: a producer compactor that overflows on every partition.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scrunch_to_fit_view_matches_flat_fold_across_passes() -> None:
    """Multi-pass ``view`` folds away earlier summaries, never resurrects them.

    Regression for the bridge's hand-rolled mask replay (REV-MR-001). A
    second scrunch pass partitions over a region that includes the
    first pass's summary splice; the producer's ``full_tape_mask`` then
    covers that splice's ref. Re-deriving the resolved view from the
    produced splices would trigger the resolver's cover-the-cover
    undelete -- masking pass 1's splice resurrects the content pass 1
    folded away -- yielding a view *larger* than the input. The
    executor's flat splice-by-replacement is the ground truth, and
    ``ScrunchResult.view`` must return exactly it.

    The fold replaces each oldest partition with one ``"S"*chars``
    summary, so the final view is a short prefix of summaries plus the
    untouched recent tail. It must contain no original folded message
    and must be strictly smaller than the input.
    """
    context: list[ModelContextEvent] = [_user("x" * 600) for _ in range(8)]
    tape = _tape_from(context)
    compactor = _SizedCompactor(summary_chars=20)
    result = await scrunch_to_fit(
        context=context,
        tape=tape,
        model=_stub_model(),
        compactor=compactor,
        mint_ref=_ref_factory(len(tape)),
        target_input_tokens=100,
        max_partition_tokens=400,
        summary_size_estimate_tokens=5,
    )
    # Multiple passes ran (the fold needed more than one partition).
    assert len(result.splices) >= 2, (
        f"expected a multi-pass fold; got {len(result.splices)} splice(s)"
    )
    view_texts = [m.text for m in result.view if isinstance(m, UserMessage)]
    summary_text = "S" * 20
    # Every folded original ("x"*600) is gone; only summaries (+ any
    # untouched recent tail, which here is also folded) remain. No
    # resurrected original content.
    assert all(t == summary_text for t in view_texts), (
        f"resolved view leaked or resurrected non-summary content: {view_texts!r}"
    )
    # The view is strictly smaller than the input (the whole point of the
    # fold); a resurrecting re-derivation would make it larger.
    assert len(result.view) < len(context)


@pytest.mark.asyncio
async def test_scrunch_to_fit_raises_when_producer_overflows() -> None:
    """Producer compactor itself raises -> wrap in ScrunchTooLargeError.

    Scrunch cannot make progress when the producer compactor cannot
    summarize any partition we hand it. Wrap the producer's error in
    :class:`ScrunchTooLargeError` so the agent layer can surface a
    useful diagnostic instead of a bare ``PromptTooLongError`` from
    the compactor's internal ``stream`` call.
    """

    @dataclass(slots=True, kw_only=True)
    class _AlwaysOverflowsCompactor:
        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[ModelContextEvent],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del tape, context, model, mint_ref, custom_instructions
            raise RuntimeError("partition itself too large")

    context: list[ModelContextEvent] = [_user("x" * 100_000), _user("ok")]
    tape = _tape_from(context)

    with pytest.raises(ScrunchTooLargeError):
        _ = await scrunch_to_fit(
            context=context,
            tape=tape,
            model=_stub_model(),
            compactor=_AlwaysOverflowsCompactor(),
            mint_ref=_ref_factory(len(tape)),
            target_input_tokens=200,
            max_partition_tokens=200,
        )


@pytest.mark.asyncio
async def test_scrunch_to_fit_raises_when_summary_does_not_shrink() -> None:
    """Producer that returns a summary equal-or-larger than input -> raise.

    Without this guard the executor would loop summarizing every pass
    without making progress (the resolved view never gets smaller).
    Raise instead of spinning.
    """
    # Producer "summary" is larger than the partition input.
    context: list[ModelContextEvent] = [_user("x" * 100) for _ in range(4)]
    tape = _tape_from(context)
    compactor = _SizedCompactor(summary_chars=10_000)
    with pytest.raises(ScrunchTooLargeError, match=r"shrink"):
        _ = await scrunch_to_fit(
            context=context,
            tape=tape,
            model=_stub_model(),
            compactor=compactor,
            mint_ref=_ref_factory(len(tape)),
            target_input_tokens=50,
            max_partition_tokens=500,
            summary_size_estimate_tokens=5,
        )


@pytest.mark.asyncio
async def test_scrunch_to_fit_rejects_zero_max_passes() -> None:
    with pytest.raises(ValueError, match=r"max_passes"):
        _ = await scrunch_to_fit(
            context=[_user("x" * 100)],
            tape=_tape_from([_user("x" * 100)]),
            model=_stub_model(),
            compactor=_SizedCompactor(),
            mint_ref=_ref_factory(1),
            target_input_tokens=1,
            max_partition_tokens=500,
            max_passes=0,
        )


@pytest.mark.asyncio
async def test_scrunch_to_fit_clamps_summary_estimate_below_cap() -> None:
    # Default summary estimate (1500) exceeds a tiny per-call cap; the
    # executor clamps it rather than raising the planner precondition.
    context: list[ModelContextEvent] = [_user("x" * 100) for _ in range(4)]
    tape = _tape_from(context)
    result = await scrunch_to_fit(
        context=context,
        tape=tape,
        model=_stub_model(),
        compactor=_SizedCompactor(summary_chars=4),
        mint_ref=_ref_factory(len(tape)),
        target_input_tokens=10,
        max_partition_tokens=30,  # < default summary_size_estimate_tokens (1500)
    )
    # No ValueError; scrunch ran and produced at least one splice.
    assert len(result.splices) >= 1


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
