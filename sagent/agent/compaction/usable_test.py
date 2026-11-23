"""Tests for usable: asking a model until a reply can be used, by one retry policy."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import asyncio
import logging

import pytest

from sagent.agent.compaction.usable import UnusableReplyError, usable
from sagent.model.exceptions import (
    ContextOverflowError,
    OverloadedError,
    RefusedError,
)
from sagent.model.scripted import reply
from sagent.model.types import Response


_GOOD = reply("good")
_BAD = reply("bad")
_REFUSED = reply("No.", stop_reason="refusal", stop_details="cyber: Blocked.")


@dataclass(slots=True, kw_only=True)
class _Ask:
    """Answer each call with the next of ``outcomes``: a reply, or an error raised."""

    outcomes: Sequence[Response | Exception]
    calls: int = 0

    async def __call__(self) -> Response:
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _good(response: Response) -> str | None:
    return "read" if response is _GOOD else None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "first",
    [_REFUSED, _BAD, OverloadedError("busy")],
    ids=["refused", "unusable", "transient"],
)
async def test_a_failed_attempt_is_asked_again_and_a_later_answer_used(
    monkeypatch: pytest.MonkeyPatch,
    first: Response | Exception,
) -> None:
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    ask = _Ask(outcomes=(first, _GOOD))

    assert await usable(ask, read=_good, attempts=3, what="summary") == "read"
    assert ask.calls == 2


@pytest.mark.asyncio
async def test_every_refusal_says_how_many_of_the_attempts_were_refused(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ask = _Ask(outcomes=(_REFUSED, _REFUSED, _REFUSED))

    with (
        caplog.at_level(logging.WARNING, logger="sagent.agent.compaction.usable"),
        pytest.raises(
            RefusedError,
            match=r"^The model refused 3 of 3 attempts at the summary: cyber: Blocked\.$",
        ),
    ):
        _ = await usable(ask, read=_good, attempts=3, what="summary")
    assert caplog.messages == ["summary refused: cyber: Blocked."] * 3


@pytest.mark.asyncio
async def test_a_refusal_after_other_failures_counts_only_the_refusals() -> None:
    shortened: list[int] = []
    ask = _Ask(outcomes=(ContextOverflowError("long"), _BAD, _REFUSED))

    with pytest.raises(RefusedError, match=r"refused 1 of 3 attempts"):
        _ = await usable(
            ask,
            read=_good,
            attempts=3,
            what="summary",
            shorten=lambda: shortened.append(1),
        )
    assert shortened == [1]


@pytest.mark.asyncio
async def test_a_refusal_with_no_reason_says_so() -> None:
    ask = _Ask(outcomes=(reply(stop_reason="refusal"),))

    with pytest.raises(RefusedError, match=r"summary: no reason given$"):
        _ = await usable(ask, read=_good, attempts=1, what="summary")


@pytest.mark.asyncio
async def test_the_last_failure_is_raised_as_what_it_was() -> None:
    ask = _Ask(outcomes=(_REFUSED, _BAD))

    with pytest.raises(
        UnusableReplyError,
        match=r"^The last of 2 attempts returned no usable session audit\.$",
    ):
        _ = await usable(ask, read=_good, attempts=2, what="session audit")


@pytest.mark.asyncio
async def test_an_overflow_with_nothing_to_shorten_is_raised_at_once() -> None:
    ask = _Ask(outcomes=(ContextOverflowError("long"), _GOOD))

    with pytest.raises(ContextOverflowError, match=r"^long$"):
        _ = await usable(ask, read=_good, attempts=3, what="summary")
    assert ask.calls == 1


@pytest.mark.asyncio
async def test_what_shortening_raises_ends_the_attempts() -> None:
    ask = _Ask(outcomes=(ContextOverflowError("long"), _GOOD))

    with pytest.raises(ContextOverflowError, match=r"^nothing shorter fits$"):
        _ = await usable(
            ask,
            read=_good,
            attempts=3,
            what="summary",
            shorten=_nothing_shorter,
        )
    assert ask.calls == 1


@pytest.mark.asyncio
async def test_the_last_overflow_is_raised_without_shortening_for_no_next_attempt() -> (
    None
):
    shortened: list[int] = []
    ask = _Ask(outcomes=[ContextOverflowError("long")] * 2)

    with pytest.raises(ContextOverflowError, match=r"^long$"):
        _ = await usable(
            ask,
            read=_good,
            attempts=2,
            what="summary",
            shorten=lambda: shortened.append(1),
        )
    assert shortened == [1]


@pytest.mark.asyncio
async def test_a_transient_failure_waits_as_advised_except_after_the_last(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waited: list[float] = []

    async def sleep(delay: float) -> None:
        waited.append(delay)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    ask = _Ask(outcomes=[OverloadedError("busy", retry_after_sec=0.25)] * 2)

    with pytest.raises(OverloadedError, match=r"^busy$"):
        _ = await usable(ask, read=_good, attempts=2, what="summary")
    assert waited == [0.25]


def _nothing_shorter() -> None:
    raise ContextOverflowError("nothing shorter fits")


async def _no_sleep(delay: float) -> None:
    del delay


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
