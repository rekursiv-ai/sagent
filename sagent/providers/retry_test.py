"""Tests for retrying a stream that fails before its first event, paced per account."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, override

import asyncio

from wesearch.fetch.gate import FileGate

import pytest

from sagent.model.exceptions import (
    OverloadedError,
    RateLimitedError,
    RetryableError,
)
from sagent.providers import retry
from sagent.providers.retry import account_key, observed, retried
from sagent.providers.types import UsageWindow


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable


_KEY = "openai-sub@work/sol-6.1"


class _Clock:
    """The time every gate and retry reads, moved only by the sleeps recorded."""

    def __init__(self) -> None:
        self.now = 1_000.0
        self.slept: list[float] = []

    def time(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.slept.append(delay)
        self.now += delay


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    made = _Clock()
    monkeypatch.setattr("wesearch.fetch.gate.time", made)
    monkeypatch.setattr("sagent.providers.retry.time", made)
    monkeypatch.setattr(asyncio, "sleep", made.sleep)
    return made


@pytest.fixture
def slept(clock: _Clock) -> list[float]:
    """Record each wait instead of sleeping it."""
    return clock.slept


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


async def _all(stream: AsyncIterator[str]) -> list[str]:
    return [event async for event in stream]


def _retried(
    stream: Callable[[], AsyncIterator[str]],
    *,
    key: str = _KEY,
) -> AsyncIterator[str]:
    return retried(stream, gate=FileGate(), key=key)


@pytest.mark.asyncio
async def test_a_stream_that_works_is_streamed_once(slept: list[float]) -> None:
    stream, made = _script(["a", "b"])

    assert await _all(_retried(stream)) == ["a", "b"]
    assert made == [0]
    assert slept == []


@pytest.mark.asyncio
async def test_a_stream_of_no_events_is_done(slept: list[float]) -> None:
    stream, made = _script([], ["never"])

    assert await _all(_retried(stream)) == []
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

    assert await _all(_retried(stream)) == ["ok"]
    assert made == [0, 1, 2, 3]
    # Doubling from half a second, unless the server advised longer.
    assert slept == [0.5, 1.0, 3.0]


@pytest.mark.asyncio
async def test_the_fifth_failure_is_raised(slept: list[float]) -> None:
    stream, made = _script(*(OverloadedError(f"busy {i}") for i in range(5)))

    with pytest.raises(OverloadedError, match=r"^busy 4$"):
        await _all(_retried(stream))
    assert made == [0, 1, 2, 3, 4]
    assert slept == [0.5, 1.0, 2.0, 4.0]


@pytest.mark.asyncio
async def test_a_failure_after_an_event_is_raised_at_once(slept: list[float]) -> None:
    stream, made = _script((["shown"], OverloadedError("dropped")), ["never"])
    events = _retried(stream)

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
        assert await _all(_retried(stream)) == ["ok"]
        assert slept == [advised]
    else:
        with pytest.raises(OverloadedError, match="later"):
            await _all(_retried(stream))
        assert slept == []


class _Releasing(FileGate):
    """A gate that records each key whose probe it frees."""

    def __init__(self) -> None:
        super().__init__()
        self.released: list[str] = []

    @override
    def release_probe(self, key: str) -> None:
        self.released.append(key)
        super().release_probe(key)


@pytest.mark.asyncio
async def test_any_other_failure_frees_the_probe_and_is_raised_untried(
    slept: list[float],
) -> None:
    """A caller's own bug says nothing about the account, so no one waits on it."""
    made: list[int] = []

    nothing: tuple[str, ...] = ()

    async def broken() -> AsyncIterator[str]:
        made.append(len(made))
        for event in nothing:
            yield event
        raise ValueError("bad request body")

    gate = _Releasing()
    with pytest.raises(ValueError, match="bad request body"):
        await _all(retried(broken, gate=gate, key=_KEY))
    assert made == [0]
    assert gate.released == [_KEY]
    assert slept == []


# -- the account's gate: one caller's rate limit holds back every other ----------


@pytest.mark.asyncio
async def test_a_rate_limit_is_sat_out_then_retried(clock: _Clock) -> None:
    stream, made = _script(RateLimitedError("slow", retry_after_sec=10.0), ["ok"])

    assert await _all(_retried(stream)) == ["ok"]
    assert made == [0, 1]
    [wait] = clock.slept
    assert 10.0 <= wait <= 12.0


@pytest.mark.asyncio
async def test_one_callers_rate_limit_holds_back_every_other(clock: _Clock) -> None:
    # The fan-out of four children on one account: one meets the 429, and the
    # other three send nothing until it resets.
    first, _ = _script(RateLimitedError("slow", retry_after_sec=39.0))
    with pytest.raises(RateLimitedError):
        await _all(_retried(first))
    sent_at: list[float] = []
    for _ in range(3):

        async def stream() -> AsyncIterator[str]:
            sent_at.append(clock.now)
            yield "ok"

        assert await _all(_retried(stream)) == ["ok"]
    assert all(at >= 1_039.0 for at in sent_at)


@pytest.mark.asyncio
async def test_another_account_is_not_held(clock: _Clock) -> None:
    first, _ = _script(RateLimitedError("slow", retry_after_sec=39.0))
    with pytest.raises(RateLimitedError):
        await _all(_retried(first))
    other, made = _script(["ok"])

    assert await _all(_retried(other, key="openai-sub@home/sol-6.1")) == ["ok"]
    assert made == [0]
    assert clock.slept == []


@pytest.mark.asyncio
async def test_a_hold_seen_by_another_process_is_sat_out_too(clock: _Clock) -> None:
    # Another process's gate is another ``FileGate`` on the same files.
    FileGate().hold(_KEY, until=clock.now + 20.0)
    stream, made = _script(["ok"])

    assert await _all(_retried(stream)) == ["ok"]
    assert made == [0]
    assert clock.now >= 1_020.0


@pytest.mark.asyncio
async def test_a_hold_too_long_to_wait_refuses_without_sending(clock: _Clock) -> None:
    FileGate().hold(_KEY, until=clock.now + 3_600.0)
    stream, made = _script(["never"])

    with pytest.raises(RateLimitedError, match=r"^The account is cooling down") as no:
        await _all(_retried(stream))
    assert made == []
    assert no.value.retry_after_sec >= 3_600.0
    assert clock.slept == []


@pytest.mark.asyncio
async def test_a_hold_as_long_as_the_wait_allows_is_sat_out(
    clock: _Clock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        retry,
        "ACCOUNT_POLICY",
        replace(retry.ACCOUNT_POLICY, jitter_sec=0.0),
    )
    FileGate().hold(_KEY, until=clock.now + retry.ACCOUNT_POLICY.max_wait_sec)
    stream, made = _script(["ok"])

    assert await _all(_retried(stream)) == ["ok"]
    assert made == [0]
    assert clock.slept == [retry.ACCOUNT_POLICY.max_wait_sec]


@pytest.mark.asyncio
async def test_a_rate_limit_naming_no_wait_cools_down(clock: _Clock) -> None:
    stream, made = _script(RateLimitedError("slow"), ["ok"])

    assert await _all(_retried(stream)) == ["ok"]
    assert made == [0, 1]
    [wait] = clock.slept
    assert 5.0 <= wait <= 7.0


@pytest.mark.asyncio
async def test_a_success_lifts_the_cooldown(clock: _Clock) -> None:
    stream, _ = _script(RateLimitedError("slow"), ["ok"])
    await _all(_retried(stream))
    clock.slept.clear()
    again, _ = _script(["ok"])

    assert await _all(_retried(again)) == ["ok"]
    assert clock.slept == []


@pytest.mark.usefixtures("clock")
@pytest.mark.asyncio
async def test_a_spent_window_in_a_reply_holds_the_account() -> None:
    gate = FileGate()
    await observed(
        gate,
        _KEY,
        (
            UsageWindow(
                label="requests",
                utilization=1.0,
                resets_at=1_050.0,
                blocked=True,
            ),
            UsageWindow(label="tokens", utilization=0.2, resets_at=9_999.0),
        ),
    )
    assert gate.states()[_KEY].held_until == 1_050.0


@pytest.mark.usefixtures("clock")
@pytest.mark.asyncio
async def test_a_reply_with_nothing_spent_holds_nothing() -> None:
    gate = FileGate()
    await observed(
        gate,
        _KEY,
        (UsageWindow(label="5h", utilization=1.0, resets_at=900.0, blocked=True),),
    )
    assert gate.states() == {}


def test_an_api_key_is_kept_only_as_a_digest() -> None:
    key = account_key("anthropic", model="opus-5.5", credential="sk-ant-secret")
    who, _, model = key.removeprefix("anthropic@").partition("/")
    assert (len(who), model) == (16, "opus-5.5")
    assert "secret" not in key
    assert account_key("anthropic", model="m") == account_key(
        "anthropic",
        model="m",
        credential="",
    )
    assert key != account_key("anthropic", model="opus-5.5", credential="sk-other")


def test_each_model_of_an_account_is_paced_apart() -> None:
    """A window named for one family leaves the account's other models free."""
    opus = account_key("claude", model="opus-5.5", account="work")
    assert opus == "claude@work/opus-5.5"
    assert opus != account_key("claude", model="sonnet-5", account="work")


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
