"""What provider tests share: a free loopback port, a browser's visit, quick retries.

Typical usage example:

  port = free_port()
  assert await visit(port, "/callback?code=c&state=s") == 200
  waits = paused_backoff(monkeypatch)            # each retry's wait, unslept
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Protocol

import asyncio
import socket

from wesearch.fetch import gate

from sagent.providers import retry


if TYPE_CHECKING:
    from types import ModuleType

    import pytest

    from sagent.providers.login import Logins, Spent


__all__ = [
    "bound_port",
    "first_live",
    "free_port",
    "paused_backoff",
    "refuse_browsers",
    "send",
    "visit",
    "window",
]


def first_live(logins: Logins, model: str, *, now: float) -> str | None:
    """Return the first account ``ranked`` says may call ``model``."""
    return next((a for a, s in logins.ranked(model, now=now) if s is None), None)


def window(
    logins: Logins,
    account: str | None,
    model: str,
    *,
    now: float,
) -> Spent | None:
    """Return what ``ranked`` says keeps ``account`` from ``model``."""
    name = account or "default"
    return dict(logins.ranked(model, now=now, first=name)).get(name)


class Bound(Protocol):
    """An IPv4 socket, whose address typeshed leaves untyped."""

    def getsockname(self) -> tuple[str, int]: ...


def refuse_browsers(monkeypatch: pytest.MonkeyPatch, *modules: ModuleType) -> None:
    """Fail the test that opens a browser through any of ``modules``.

    A test that means to open one patches ``open_browser`` after this; one that
    does not, or code that stops honoring ``headless``, must not reach the user's
    real browser.

    Args:
      monkeypatch: The test's.
      modules: Each module that calls ``open_browser``.

    """

    def refused(url: str) -> bool:
        raise AssertionError(f"A test opened a real browser at {url}.")

    for module in modules:
        monkeypatch.setattr(module, "open_browser", refused)


def paused_backoff(
    monkeypatch: pytest.MonkeyPatch,
    *clocked: ModuleType,
) -> list[float]:
    """Make each retry's wait pass at once on a clock of its own; return the waits.

    The account gate and the retries read one clock, which only the waits move,
    so a hold the gate keeps lifts when it is waited out; jitter is off, so each
    wait is the one advised.

    Args:
      monkeypatch: The test's.
      clocked: Further modules whose ``time`` reads the same clock -- a wire that
        dates a reset the gate then holds to.

    Returns:
      waits: Each wait, in seconds, in the order waited.

    """
    waits: list[float] = []
    clock = [1_000_000.0]

    async def sleep(delay_sec: float) -> None:
        waits.append(delay_sec)
        clock[0] += delay_sec

    now = SimpleNamespace(time=lambda: clock[0])
    paused = SimpleNamespace(sleep=sleep, to_thread=asyncio.to_thread)
    monkeypatch.setattr(retry, "asyncio", paused)
    monkeypatch.setattr(gate, "asyncio", paused)
    for module in (retry, gate, *clocked):
        monkeypatch.setattr(module, "time", now)
    monkeypatch.setattr(
        retry,
        "ACCOUNT_POLICY",
        replace(retry.ACCOUNT_POLICY, jitter_sec=0.0),
    )
    return waits


def bound_port(bound: Bound) -> int:
    """Return the port ``bound`` listens on."""
    return bound.getsockname()[1]


def free_port() -> int:
    """Return a loopback port nothing listens on, as of the call."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return bound_port(probe)


async def visit(port: int, target: str) -> int:
    """Return the status a GET of ``target`` on ``port`` answers with."""
    return await send(
        port,
        f"GET {target} HTTP/1.1\r\nhost: localhost\r\n\r\n".encode(),
    )


async def send(port: int, request: bytes) -> int:
    """Send ``request`` to ``port`` as it is; return the status of the answer.

    Args:
      port: A loopback port.
      request: The raw request, malformed or not.

    Returns:
      status: The answer's HTTP status code.

    """
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        writer.write(request)
        await writer.drain()
        return int((await reader.readline()).split()[1])
    finally:
        writer.close()
        await writer.wait_closed()
