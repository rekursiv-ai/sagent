"""What providers share: one HTTP client, process pools, one tool server.

Each opens on first use and closes with the ``SharedResources`` holding it.

Typical usage example:

  async with SharedResources() as shared:
      anthropic = Anthropic(resources=shared)
      cli = AnthropicCLI(resources=shared)    # The same pools and tool server.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, override

import asyncio
import contextlib

from sagent.lib.posix.child import Child
from sagent.providers.tool_server import McpToolServer
from sagent.types import AsyncResource


if TYPE_CHECKING:
    from asyncio.subprocess import Process
    from collections.abc import Mapping, Sequence

    import httpx2
else:
    from wrapt import lazy_import

    httpx2 = lazy_import("httpx2")


__all__ = [
    "Closable",
    "SharedResources",
    "SubprocessPool",
    "close_all",
]


class SubprocessPool(AsyncResource):
    """Processes by command line, each with a spare started ahead of the next.

    Args:
      grace_sec: How long a released process has, after TERM, before KILL.

    """

    def __init__(self, *, grace_sec: float = 5.0) -> None:
        self._grace_sec = grace_sec
        self._live: set[Process] = set()
        self._spares: dict[
            tuple[tuple[str, ...], tuple[tuple[str, str], ...]],
            asyncio.Task[Process],
        ] = {}

    async def acquire(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
    ) -> Process:
        """Return a process running ``argv`` in ``env``, and start its successor.

        Args:
          argv: The command line.
          env: The environment.

        Returns:
          process: The spare started earlier for this command line, if it is
            still running, else one started now.

        """
        key = (tuple(argv), tuple(sorted(env.items())))
        spare = self._spares.pop(key, None)
        process = None
        if spare is not None:
            with contextlib.suppress(OSError):
                process = await spare
        if process is None or process.returncode is not None:
            if process is not None:
                await self.release(process)
            process = await self._spawn(argv, env=env)
        # Starting ``claude`` takes seconds; the next acquire of this command
        # line, a new conversation, takes this one instead of waiting.
        self._spares[key] = asyncio.create_task(self._spawn(argv, env=env))
        return process

    async def release(self, process: Process) -> None:
        """Stop ``process`` and the jobs it started: politely, then by force.

        Args:
          process: A process from ``acquire``.

        """
        self._live.discard(process)
        if process.stdin is not None:
            process.stdin.close()
        await Child(process=process, group=True, grace_sec=self._grace_sec).stop()

    @override
    async def aclose(self) -> None:
        # Awaited rather than cancelled: a spawn cancelled midway can leave a
        # process no one holds.
        spares = self._spares
        self._spares = {}
        for spare in spares.values():
            with contextlib.suppress(OSError):
                await spare
        for process in tuple(self._live):
            with contextlib.suppress(ProcessLookupError):
                await self.release(process)

    async def _spawn(self, argv: Sequence[str], *, env: Mapping[str, str]) -> Process:
        child = await Child.start(
            *argv,
            group=True,
            grace_sec=self._grace_sec,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=env,
            # Claude's stream-json lines carry whole tool inputs and outputs;
            # asyncio's 64 KiB default raises mid-turn on the first large one.
            limit=16 * 1024 * 1024,
        )
        self._live.add(child.process)
        return child.process


class SharedResources(AsyncResource):
    """An HTTP client, process pools, and a tool server, each opened on first use."""

    def __init__(self) -> None:
        self._http: httpx2.AsyncClient | None = None
        self._pools: dict[str, SubprocessPool] = {}
        self._tool_server: McpToolServer | None = None

    @property
    def http(self) -> httpx2.AsyncClient:
        """Return the HTTP client."""
        if self._http is None:
            # One stalled Wi-Fi connect outlasts httpx's 5s default; the read
            # timeout is the stream's idle allowance, not the turn's length.
            self._http = httpx2.AsyncClient(
                timeout=httpx2.Timeout(600.0, connect=30.0),
            )
        return self._http

    def process_pool(self, name: str) -> SubprocessPool:
        """Return the pool of processes named ``name``."""
        pool = self._pools.get(name)
        if pool is None:
            pool = self._pools[name] = SubprocessPool()
        return pool

    @property
    def tool_server(self) -> McpToolServer:
        """Return the MCP server that offers tools to CLIs."""
        if self._tool_server is None:
            self._tool_server = McpToolServer()
        return self._tool_server

    @override
    async def aclose(self) -> None:
        pools = self._pools
        self._pools = {}
        tool_server, self._tool_server = self._tool_server, None
        http, self._http = self._http, None
        await close_all(
            [*pools.values(), *(c for c in (tool_server, http) if c is not None)],
        )


class Closable(Protocol):
    """Something that releases what it holds when closed."""

    async def aclose(self) -> None:
        """Release what this holds."""
        ...


async def close_all(resources: Sequence[Closable]) -> None:
    """Close each of ``resources`` in order, even after one fails to.

    Args:
      resources: What to close.

    Raises:
      BaseException: What a close raised, once the rest are closed; a later
        failure carries the earlier ones as its context.

    """
    if not resources:
        return
    try:
        await resources[0].aclose()
    finally:
        await close_all(resources[1:])
