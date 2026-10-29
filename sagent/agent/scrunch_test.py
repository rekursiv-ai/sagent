"""Tests for scrunch: summarizing the oldest history in passes until it fits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from sagent.agent.scrunch import scrunch
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
