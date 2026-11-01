"""Tests for scrunch: summarizing the oldest history in passes until it fits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import logging

import pytest

from sagent.agent.scrunch import safe_boundaries, scrunch
from sagent.model.scripted import call, reply
from sagent.model.types import Response
from sagent.runtime.types import ModelContextEvent
from sagent.session.types import (
    AssistantMessage,
    ToolResult,
    UncategorizedToolResult,
    UserMessage,
)


if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(slots=True, kw_only=True)
class _Summarizer:
    """Summarize any partition to one message of ``size`` tokens."""

    size: int = 1
    dropped: list[str] = field(default_factory=list)
    """Why each of the first calls dropped input; later calls drop nothing."""

    error: Exception | None = None
    calls: list[list[ModelContextEvent]] = field(default_factory=list)

    async def __call__(
        self,
        part: list[ModelContextEvent],
    ) -> tuple[tuple[ModelContextEvent, ...], str]:
        self.calls.append(part)
        if self.error is not None:
            raise self.error
        why = self.dropped.pop(0) if self.dropped else ""
        return (UserMessage(content="s" * self.size),), why


def _tokens(entries: Sequence[ModelContextEvent]) -> int:
    """Count one token per character the entries hold."""
    return sum(len(_text(e)) for e in entries)


def _text(entry: ModelContextEvent) -> str:
    if isinstance(entry, Response):
        return "".join(
            r.content or "" for r in entry.records if isinstance(r, AssistantMessage)
        )
    if isinstance(entry, ToolResult):
        return entry.text()
    assert isinstance(entry, UserMessage)
    return entry.content or ""


def _messages(count: int, size: int) -> list[ModelContextEvent]:
    return [UserMessage(content=chr(ord("a") + i) * size) for i in range(count)]


@pytest.mark.asyncio
async def test_a_history_that_fits_is_returned_without_a_summary() -> None:
    summarize = _Summarizer()
    context = _messages(3, 10)

    view, stopped = await scrunch(
        context,
        summarize=summarize,
        tokens=_tokens,
        target_tokens=30,
        max_partition_tokens=100,
    )

    assert (view, stopped) == (context, "")
    assert summarize.calls == []


@pytest.mark.asyncio
async def test_one_pass_summarizes_the_oldest_entries_and_keeps_the_rest() -> None:
    summarize = _Summarizer(size=2)
    context = _messages(9, 10)

    view, stopped = await scrunch(
        context,
        summarize=summarize,
        tokens=_tokens,
        target_tokens=72,
        max_partition_tokens=100,
        summary_tokens=2,
    )

    assert stopped == ""
    assert summarize.calls == [context[:2]]
    assert view == [UserMessage(content="ss"), *context[2:]]


@pytest.mark.asyncio
async def test_passes_continue_oldest_first_and_stop_once_the_whole_fits() -> None:
    summarize = _Summarizer(size=1)
    context = _messages(10, 10)

    view, stopped = await scrunch(
        context,
        summarize=summarize,
        tokens=_tokens,
        target_tokens=60,
        max_partition_tokens=20,
        summary_tokens=1,
    )

    assert stopped == ""
    assert _tokens(view) <= 60
    assert summarize.calls[0] == context[:2]
    assert summarize.calls[1][0] == UserMessage(content="s")
    assert view[-1] == context[-1]
    assert len(summarize.calls) == 4


@pytest.mark.asyncio
async def test_a_partition_never_cuts_a_call_from_its_result() -> None:
    turn = reply("r" * 10, calls=[call("Read", "c1")])
    result = UncategorizedToolResult(call_id="c1", content="t" * 10)
    context: list[ModelContextEvent] = [
        UserMessage(content="u" * 10),
        turn,
        result,
        *_messages(4, 10),
    ]
    summarize = _Summarizer(size=1)

    _, stopped = await scrunch(
        context,
        summarize=summarize,
        tokens=_tokens,
        target_tokens=45,
        max_partition_tokens=100,
        summary_tokens=1,
    )

    assert stopped == ""
    assert summarize.calls[0] == context[:3]


@pytest.mark.asyncio
async def test_a_summary_larger_than_its_estimate_grows_the_next_partition() -> None:
    summarize = _Summarizer(size=15)

    view, stopped = await scrunch(
        _messages(10, 10),
        summarize=summarize,
        tokens=_tokens,
        target_tokens=40,
        max_partition_tokens=20,
        summary_tokens=1,
    )

    assert stopped == ""
    assert _tokens(view) <= 40
    assert all(_tokens(part) > 15 for part in summarize.calls)


@pytest.mark.asyncio
async def test_the_summary_estimate_is_held_below_the_partition_cap() -> None:
    summarize = _Summarizer(size=1)

    view, stopped = await scrunch(
        _messages(20, 1),
        summarize=summarize,
        tokens=_tokens,
        target_tokens=10,
        max_partition_tokens=5,
    )

    assert stopped == ""
    assert _tokens(view) <= 10
    assert all(_tokens(part) <= 5 for part in summarize.calls)


@pytest.mark.asyncio
async def test_a_pass_that_dropped_input_is_retried_and_its_view_not_kept() -> None:
    summarize = _Summarizer(size=2, dropped=["lost a turn"])
    context = _messages(9, 10)

    view, stopped = await scrunch(
        context,
        summarize=summarize,
        tokens=_tokens,
        target_tokens=72,
        max_partition_tokens=100,
        summary_tokens=2,
    )

    assert stopped == ""
    assert summarize.calls == [context[:2], context[:2]]
    assert view == [UserMessage(content="ss"), *context[2:]]


@pytest.mark.asyncio
async def test_consecutive_rejections_stop_with_the_best_view_so_far() -> None:
    summarize = _Summarizer(size=1, dropped=["", "gone", "gone", "gone"])
    context = _messages(10, 10)

    view, stopped = await scrunch(
        context,
        summarize=summarize,
        tokens=_tokens,
        target_tokens=20,
        max_partition_tokens=20,
        summary_tokens=1,
    )

    assert stopped == "pass 3: the summary dropped input (gone)"
    assert view == [UserMessage(content="s"), *context[2:]]


@pytest.mark.asyncio
async def test_an_accepted_pass_resets_the_rejection_count() -> None:
    summarize = _Summarizer(size=1, dropped=["gone", "", "gone", "gone"])

    view, stopped = await scrunch(
        _messages(10, 10),
        summarize=summarize,
        tokens=_tokens,
        target_tokens=60,
        max_partition_tokens=20,
        summary_tokens=1,
    )

    assert stopped == ""
    assert _tokens(view) <= 60


@pytest.mark.asyncio
async def test_a_summary_that_does_not_shrink_the_whole_is_rejected() -> None:
    summarize = _Summarizer(size=50)
    context = _messages(5, 10)

    view, stopped = await scrunch(
        context,
        summarize=summarize,
        tokens=_tokens,
        target_tokens=20,
        max_partition_tokens=100,
        max_rejected=2,
    )

    assert stopped == "pass 1: the summary did not shrink the whole (50 -> 50 tokens)"
    assert view == context
    assert len(summarize.calls) == 2


@pytest.mark.asyncio
async def test_a_summary_that_raises_ends_with_the_best_view() -> None:
    summarize = _Summarizer(error=RuntimeError("boom"))
    context = _messages(5, 10)

    view, stopped = await scrunch(
        context,
        summarize=summarize,
        tokens=_tokens,
        target_tokens=20,
        max_partition_tokens=100,
    )

    assert view == context
    assert stopped.startswith("pass 0: a ")
    assert stopped.endswith("-entry partition failed: boom")


@pytest.mark.asyncio
async def test_running_out_of_passes_says_so() -> None:
    summarize = _Summarizer(size=1)

    view, stopped = await scrunch(
        _messages(10, 10),
        summarize=summarize,
        tokens=_tokens,
        target_tokens=5,
        max_partition_tokens=20,
        summary_tokens=1,
        max_passes=2,
    )

    assert stopped == "no fit after 2 passes"
    assert len(summarize.calls) == 2
    assert _tokens(view) == 71


@pytest.mark.asyncio
async def test_the_last_pass_that_fits_reports_no_shortfall() -> None:
    summarize = _Summarizer(size=1)

    _, stopped = await scrunch(
        _messages(4, 10),
        summarize=summarize,
        tokens=_tokens,
        target_tokens=31,
        max_partition_tokens=100,
        summary_tokens=1,
        max_passes=1,
    )

    assert stopped == ""


@pytest.mark.asyncio
async def test_a_partition_stops_growing_once_past_one_summary() -> None:
    summarize = _Summarizer(size=1)
    context = _messages(5, 10)

    _, stopped = await scrunch(
        context,
        summarize=summarize,
        tokens=_tokens,
        target_tokens=49,
        max_partition_tokens=100,
        summary_tokens=15,
    )

    assert stopped == ""
    assert summarize.calls == [context[:2]]


@pytest.mark.asyncio
async def test_a_second_entry_over_the_limit_is_left_out() -> None:
    summarize = _Summarizer(size=1)
    context = [UserMessage(content="a" * 3), UserMessage(content="b" * 10)]

    _, stopped = await scrunch(
        context,
        summarize=summarize,
        tokens=_tokens,
        target_tokens=12,
        max_partition_tokens=100,
        summary_tokens=1,
    )

    assert stopped == ""
    assert summarize.calls == [context[:1]]


@pytest.mark.asyncio
async def test_one_summary_is_expected_to_take_1500_tokens() -> None:
    summarize = _Summarizer(size=1)
    context = _messages(3, 1_501)

    _, stopped = await scrunch(
        context,
        summarize=summarize,
        tokens=_tokens,
        target_tokens=3 * 1_501 - 1,
        max_partition_tokens=1_600,
    )

    assert stopped == ""
    # 1,501 tokens is already past one summary, so nothing more is taken.
    assert summarize.calls == [context[:1]]


@pytest.mark.asyncio
async def test_sixteen_passes_are_made_at_most() -> None:
    summarize = _Summarizer(size=1_000)

    _, stopped = await scrunch(
        _messages(300, 100),
        summarize=summarize,
        tokens=_tokens,
        target_tokens=1,
        max_partition_tokens=1_600,
    )

    assert stopped == "no fit after 16 passes"
    assert len(summarize.calls) == 16


@pytest.mark.asyncio
async def test_the_first_summary_estimate_is_one_below_the_cap() -> None:
    summarize = _Summarizer(size=1)
    context = _messages(11, 1)

    _, stopped = await scrunch(
        context,
        summarize=summarize,
        tokens=_tokens,
        target_tokens=10,
        max_partition_tokens=5,
    )

    assert stopped == ""
    # Four tokens expected of a summary, plus the one to remove.
    assert summarize.calls == [context[:5]]


@pytest.mark.asyncio
async def test_a_learned_summary_estimate_is_one_below_the_cap() -> None:
    summaries: list[tuple[ModelContextEvent, ...]] = [
        tuple(UserMessage(content="s") for _ in range(10)),
        (UserMessage(content="s"),),
    ]
    calls: list[list[ModelContextEvent]] = []

    async def summarize(
        part: list[ModelContextEvent],
    ) -> tuple[tuple[ModelContextEvent, ...], str]:
        calls.append(part)
        return summaries.pop(0), ""

    _, stopped = await scrunch(
        [UserMessage(content="b" * 20), *_messages(2, 1)],
        summarize=summarize,
        tokens=_tokens,
        target_tokens=11,
        max_partition_tokens=5,
        summary_tokens=1,
    )

    assert stopped == ""
    # A ten-token summary teaches an estimate of four, the cap less one.
    assert [len(part) for part in calls] == [1, 5]


@pytest.mark.asyncio
async def test_a_partition_over_the_cap_and_a_rejection_are_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    summarize = _Summarizer(size=1, dropped=["lost"])

    with caplog.at_level(logging.WARNING, logger="sagent.agent.scrunch"):
        _, stopped = await scrunch(
            [*_messages(2, 5), UserMessage(content="c" * 20)],
            summarize=summarize,
            tokens=_tokens,
            target_tokens=1,
            max_partition_tokens=5,
            summary_tokens=1,
        )

    assert stopped == ""
    # Passes 0 and 1 fill the cap exactly, which is no excess.
    assert caplog.messages == [
        "scrunch pass 0 rejected: the summary dropped input (lost)",
        "scrunch pass 2: the partition exceeds the cap (6 > 5 tokens)",
        "scrunch pass 3: the partition exceeds the cap (21 > 5 tokens)",
    ]


@pytest.mark.asyncio
async def test_a_cap_of_one_token_is_allowed() -> None:
    _, stopped = await scrunch(
        _messages(2, 1),
        summarize=_Summarizer(size=0),
        tokens=_tokens,
        target_tokens=1,
        max_partition_tokens=1,
    )

    assert stopped == ""


def test_the_boundary_before_the_first_entry_is_safe() -> None:
    assert safe_boundaries([]) == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("max_passes", "max_partition_tokens", "match"),
    [(0, 10, "max_passes"), (1, 0, "max_partition_tokens")],
)
async def test_degenerate_limits_are_refused(
    max_passes: int,
    max_partition_tokens: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _ = await scrunch(
            _messages(1, 1),
            summarize=_Summarizer(),
            tokens=_tokens,
            target_tokens=0,
            max_partition_tokens=max_partition_tokens,
            max_passes=max_passes,
        )


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
