"""Tests for what sign-in tests share: a free port, and a visit to one."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, override
from unittest.mock import AsyncMock, Mock, call

import asyncio
import socket

from wesearch.fetch.gate import FileGate

import pytest

from sagent.model.exceptions import OverloadedError
from sagent.providers import login, retry
from sagent.providers.login import Logins, Spent
from sagent.providers.openai import wire
from sagent.providers.retry import retried
from sagent.providers.testing import (
    bound_port,
    first_live,
    free_port,
    paused_backoff,
    refuse_browsers,
    send,
    visit,
    window,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def test_a_free_port_can_be_bound() -> None:
    port = free_port()
    with socket.socket() as bound:
        bound.bind(("127.0.0.1", port))
        assert bound_port(bound) == port


@pytest.mark.asyncio
async def test_a_visit_returns_the_status_of_the_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readline = AsyncMock(return_value=b"HTTP/1.1 418 Teapot\r\n")
    reader = Mock(spec=asyncio.StreamReader, readline=readline)
    write, close = Mock(), Mock()
    drain, wait_closed = AsyncMock(), AsyncMock()
    writer = Mock(
        spec=asyncio.StreamWriter,
        write=write,
        drain=drain,
        close=close,
        wait_closed=wait_closed,
    )
    connect = AsyncMock(return_value=(reader, writer))
    monkeypatch.setattr(asyncio, "open_connection", connect)

    assert await visit(9, "/a?b=c") == 418
    assert await send(9, b"garbage\r\n") == 418

    assert connect.await_args_list == [call("127.0.0.1", 9)] * 2
    assert write.call_args_list == [
        call(b"GET /a?b=c HTTP/1.1\r\nhost: localhost\r\n\r\n"),
        call(b"garbage\r\n"),
    ]
    assert drain.await_count == readline.await_count == 2
    assert close.call_count == wait_closed.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["drain", "read", "status"])
async def test_a_failed_visit_still_closes_its_connection(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    error = OSError("connection lost")
    drain = AsyncMock(side_effect=error if failure == "drain" else None)
    readline = AsyncMock(
        return_value=b"HTTP/1.1 invalid Status\r\n",
        side_effect=error if failure == "read" else None,
    )
    close, wait_closed = Mock(), AsyncMock()
    writer = Mock(
        spec=asyncio.StreamWriter,
        drain=drain,
        close=close,
        wait_closed=wait_closed,
    )
    connect = AsyncMock(
        return_value=(Mock(spec=asyncio.StreamReader, readline=readline), writer),
    )
    monkeypatch.setattr(asyncio, "open_connection", connect)

    with pytest.raises(ValueError if failure == "status" else OSError) as caught:
        await visit(9, "/callback")

    if failure != "status":
        assert caught.value is error
    close.assert_called_once_with()
    wait_closed.assert_awaited_once_with()


def test_a_browser_opened_through_a_refused_module_fails_the_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refuse_browsers(monkeypatch, login)
    with pytest.raises(AssertionError, match=r"^A test opened a real browser at u\.$"):
        login.open_browser("u")


_WEEKLY = Spent(window="weekly", model="opus-5.5", until=2_000.0)


class _Ranked(Logins):
    """Accounts ranked as given, the first one asked for put first."""

    def __init__(self, ranked: tuple[tuple[str, Spent | None], ...]) -> None:
        super().__init__(Path("unused"))
        self._ranked = ranked

    @override
    def ranked(
        self,
        model: str,
        *,
        now: float,
        first: str | None = None,
    ) -> tuple[tuple[str, Spent | None], ...]:
        del model, now
        chosen = [row for row in self._ranked if row[0] == first]
        return (*chosen, *(row for row in self._ranked if row[0] != first))


def test_the_first_live_account_is_the_first_nothing_keeps_out() -> None:
    logins = _Ranked((("spent", _WEEKLY), ("work", None), ("home", None)))

    assert first_live(logins, "opus-5.5", now=1_000.0) == "work"
    assert first_live(_Ranked((("spent", _WEEKLY),)), "opus-5.5", now=1.0) is None


def test_the_window_is_what_keeps_the_account_out() -> None:
    logins = _Ranked((("default", _WEEKLY), ("work", None)))

    assert window(logins, None, "opus-5.5", now=1_000.0) == _WEEKLY
    assert window(logins, "work", "opus-5.5", now=1_000.0) is None


@pytest.mark.asyncio
async def test_paused_backoff_records_each_wait_on_the_gates_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retry's wait is recorded, not slept, and the gate reads the same clock."""
    waits = paused_backoff(monkeypatch, wire)
    attempts: list[int] = []

    async def stream() -> AsyncIterator[str]:
        attempts.append(len(attempts))
        if len(attempts) == 1:
            raise OverloadedError("busy")
        yield "ok"

    assert [event async for event in retried(stream, gate=FileGate(), key="k")] == [
        "ok",
    ]
    assert waits == [0.5]
    FileGate().hold("held", until=1_000_000.5 + 7.0)
    admission = FileGate().admit("held", retry.ACCOUNT_POLICY)
    assert admission.wait_sec == pytest.approx(7.0)
    assert retry.ACCOUNT_POLICY.jitter_sec == 0.0


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
