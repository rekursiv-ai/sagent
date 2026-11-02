"""Tests for retrying a stream that fails before its first event."""

from __future__ import annotations

from typing import TYPE_CHECKING

import asyncio

import pytest

from sagent.model.exceptions import OverloadedError, RetryableError
from sagent.providers.retry import retried


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable


def _script(
    *attempts: list[str] | RetryableError | tuple[list[str], RetryableError],
) -> tuple[Callable[[], AsyncIterator[str]], list[int]]:
    """Return a stream whose each attempt yields, fails, or yields then fails."""
    made: list[int] = []

    async def stream() -> AsyncIterator[str]:
        attempt = attempts[len(made)]
        made.append(len(made))
        if isinstance(attempt, RetryableError):
            raise attempt
        events, failure = attempt if isinstance(attempt, tuple) else (attempt, None)
        for event in events:
            yield event
        if failure is not None:
            raise failure

    return stream, made


@pytest.fixture
def slept(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record each backoff instead of sleeping it."""
    waits: list[float] = []

    async def sleep(delay: float) -> None:
        waits.append(delay)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    return waits


async def _all(stream: AsyncIterator[str]) -> list[str]:
    return [event async for event in stream]


@pytest.mark.asyncio
async def test_a_stream_that_works_is_streamed_once(slept: list[float]) -> None:
    stream, made = _script(["a", "b"])

    assert await _all(retried(stream)) == ["a", "b"]
    assert made == [0]
    assert slept == []


@pytest.mark.asyncio
async def test_a_stream_of_no_events_is_done(slept: list[float]) -> None:
    stream, made = _script([], ["never"])

    assert await _all(retried(stream)) == []
    assert made == [0]
    assert slept == []


@pytest.mark.asyncio
async def test_a_failure_before_any_event_is_retried_with_backoff(
    slept: list[float],
) -> None:
    stream, made = _script(
        OverloadedError("busy"),
        OverloadedError("busy"),
        OverloadedError("busy", retry_after_sec=3.0),
        ["ok"],
    )

    assert await _all(retried(stream)) == ["ok"]
    assert made == [0, 1, 2, 3]
    # Doubling from half a second, unless the server advised longer.
    assert slept == [0.5, 1.0, 3.0]


@pytest.mark.asyncio
async def test_the_fifth_failure_is_raised(slept: list[float]) -> None:
    stream, made = _script(*(OverloadedError(f"busy {i}") for i in range(5)))

    with pytest.raises(OverloadedError, match=r"^busy 4$"):
        await _all(retried(stream))
    assert made == [0, 1, 2, 3, 4]
    assert slept == [0.5, 1.0, 2.0, 4.0]


@pytest.mark.asyncio
async def test_a_failure_after_an_event_is_raised_at_once(slept: list[float]) -> None:
    stream, made = _script((["shown"], OverloadedError("dropped")), ["never"])
    events = retried(stream)

    assert await anext(events) == "shown"
    with pytest.raises(OverloadedError, match="dropped"):
        await anext(events)
    assert made == [0]
    assert slept == []


@pytest.mark.parametrize(("advised", "retried_again"), [(30.0, True), (30.5, False)])
@pytest.mark.asyncio
async def test_an_advised_wait_over_thirty_seconds_is_the_callers(
    slept: list[float],
    advised: float,
    retried_again: bool,
) -> None:
    stream, _ = _script(OverloadedError("later", retry_after_sec=advised), ["ok"])

    if retried_again:
        assert await _all(retried(stream)) == ["ok"]
        assert slept == [advised]
    else:
        with pytest.raises(OverloadedError, match="later"):
            await _all(retried(stream))
        assert slept == []


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
