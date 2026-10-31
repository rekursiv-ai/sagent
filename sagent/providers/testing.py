"""What sign-in tests share: a free loopback port, and a browser's visit to one.

Typical usage example:

  port = free_port()
  assert await visit(port, "/callback?code=c&state=s") == 200
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import asyncio
import socket


if TYPE_CHECKING:
    from sagent.providers.login import Logins, Spent


__all__ = [
    "first_live",
    "free_port",
    "port_of",
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


def port_of(bound: Bound) -> int:
    """Return the port ``bound`` listens on."""
    return bound.getsockname()[1]


def free_port() -> int:
    """Return a loopback port nothing listens on, as of the call."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return port_of(probe)


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
    writer.write(request)
    await writer.drain()
    status = int((await reader.readline()).split()[1])
    writer.close()
    await writer.wait_closed()
    return status
