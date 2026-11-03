#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 -P "$0" "$@"
One client session's tools, run as sagent on this machine runs them.

A tool server starts one per client session, as ``python -m`` of this module,
and speaks to it on a socket the host inherits as ``--fd``: ``Call`` and
``Cancel`` in, ``Tools`` then each ``Result`` out. What a tool prints goes to
stdout and stderr, never into the lines. Each client agent has its own working
directory and seen files, as an agent here does; a new context window of it
starts afresh. The socket's end cancels every call, each cleaning up as it
would here.

Typical usage example:

  python -m sagent.host.session --fd 5 --tools Bash,Read --spill /state/spill
  await serve(reader, writer, tools={"Bash": Bash()}, new_context=ToolContext)
'''
# fmt: on

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import argparse
import asyncio
import contextlib
import socket

from wesearch.fetch import FetchSession

from sagent.agent.team import Team
from sagent.host.wire import (
    MAX_LINE_BYTES,
    Call,
    Cancel,
    Result,
    Tools,
    ToolSpec,
    encoded,
    received,
)
from sagent.providers.catalog import ModelCatalog
from sagent.runtime.exceptions import describe
from sagent.runtime.types import Markers
from sagent.session.types import UncategorizedToolResult
from sagent.tools.builtin import builders
from sagent.tools.remote import RemoteSession, saved
from sagent.tools.types import ToolContext


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from sagent.session.types import AnyToolResult
    from sagent.tools.types import Tool


__all__ = [
    "main",
    "serve",
]


async def serve(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    tools: Mapping[str, Tool],
    new_context: Callable[[str], ToolContext],
) -> None:
    """Say which tools are served, then run each call ``reader`` asks for.

    Returns once ``reader`` ends, every call it left running cancelled.

    Args:
      reader: The server's lines: ``Call`` and ``Cancel``.
      writer: Where ``Tools``, then each ``Result``, go.
      tools: What is served, by name.
      new_context: A fresh state for the agent of a label: at home, no file seen.

    """
    writer.write(
        encoded(Tools(tools={n: ToolSpec.from_tool(t) for n, t in tools.items()})),
    )
    calls = _Calls(writer=writer, tools=tools, new_context=new_context)
    try:
        while (message := await received(reader)) is not None:
            if isinstance(message, Call):
                calls.start(message)
            elif isinstance(message, Cancel):
                calls.cancel(message.id)
    finally:
        await calls.close()


def main(argv: Sequence[str] | None = None) -> int:
    """Run one session's tools on the inherited socket until it ends.

    Args:
      argv: The flags; ``None`` for the command line's.

    Returns:
      code: The process exit code.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast("Flags", parser.parse_args(argv))
    asyncio.run(_main(flags))
    return 0


class Flags(Protocol):
    """Parsed command-line flags."""

    fd: int
    tools: str
    spill: Path


class _Calls:
    """The calls running, and each agent's state, for one session."""

    def __init__(
        self,
        *,
        writer: asyncio.StreamWriter,
        tools: Mapping[str, Tool],
        new_context: Callable[[str], ToolContext],
    ) -> None:
        self._writer = writer
        self._tools = tools
        self._new_context = new_context
        self._agents: dict[str, _Agent] = {}
        self._running: dict[str, asyncio.Task[None]] = {}

    # A task cancelled before its first step never enters ``_run``: the callback
    # answers it, so every call gets a result.
    def start(self, call: Call) -> None:
        """Run ``call``; its result is written when it ends."""
        task = self._running[call.id] = asyncio.create_task(self._run(call))
        task.add_done_callback(partial(self._finished, call.id))

    def _finished(self, call_id: str, task: asyncio.Task[None]) -> None:
        self._running.pop(call_id, None)
        if task.cancelled():
            self._writer.write(encoded(Result(id=call_id, record=_cancelled())))

    def cancel(self, call_id: str) -> None:
        """Stop the call ``call_id``, if it runs; its result says it was cancelled."""
        task = self._running.get(call_id)
        if task is not None:
            task.cancel()

    async def close(self) -> None:
        """Cancel every call, and wait for each to clean up."""
        tasks = list(self._running.values())
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def _context(self, call: Call) -> ToolContext:
        """Return the state of ``call``'s agent; a new window's starts afresh."""
        kept = self._agents.get(call.agent)
        if kept is None or kept.window != call.window:
            kept = _Agent(window=call.window, context=self._new_context(call.agent))
            self._agents[call.agent] = kept
        return kept.context

    async def _run(self, call: Call) -> None:
        record = await _result(
            self._tools.get(call.tool),
            self._context(call),
            call=call,
        )
        self._writer.write(encoded(Result(id=call.id, record=record)))


@dataclass(frozen=True, slots=True, kw_only=True)
class _Agent:
    """A client agent's state here, for one context window of it."""

    window: str
    context: ToolContext


async def _result(
    tool: Tool | None,
    context: ToolContext,
    *,
    call: Call,
) -> AnyToolResult:
    """Return what ``tool`` did, a failure or a cancel as the runtime records it."""
    if tool is None:
        return UncategorizedToolResult(
            call_id="",
            content=f"Unknown tool: {call.tool}",
            is_error=True,
        )
    try:
        return await tool.call(context, call.arguments)
    except asyncio.CancelledError:
        return _cancelled()
    except Exception as exc:  # noqa: BLE001 -- A tool's failure is its call's result, as the runtime records it.
        return UncategorizedToolResult(call_id="", content=describe(exc), is_error=True)


def _cancelled() -> UncategorizedToolResult:
    """Return a cancelled call's result, as the runtime records one."""
    return UncategorizedToolResult(
        call_id="",
        content=Markers().cancelled,
        is_error=True,
        stub="cancelled",
    )


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument(
        "--fd",
        type=int,
        required=True,
        help="The inherited socket the server speaks on.",
    )
    parser.add_argument(
        "--tools",
        required=True,
        help="The tools served, comma-separated.",
    )
    parser.add_argument(
        "--spill",
        type=Path,
        required=True,
        help="Where a tool writes the whole of an output it cut.",
    )


async def _main(flags: Flags) -> None:
    """Serve the tools ``flags`` names on the socket ``flags.fd``."""
    home = Path.cwd()
    reader, writer = await asyncio.open_unix_connection(
        sock=socket.socket(fileno=flags.fd),
        limit=MAX_LINE_BYTES,
    )
    async with (
        ModelCatalog() as catalog,
        Team(catalog=catalog) as team,
        FetchSession() as fetch,
        RemoteSession(saved()) as remote,
    ):
        built = builders(cwd=home, team=team, remote=remote)
        await serve(
            reader,
            writer,
            tools={name: built[name]() for name in flags.tools.split(",") if name},
            new_context=partial(
                _new_context,
                home=home,
                spill_dir=flags.spill,
                fetch_session=fetch,
            ),
        )
    writer.close()


def _new_context(
    label: str,
    *,
    home: Path,
    spill_dir: Path,
    fetch_session: FetchSession,
) -> ToolContext:
    return ToolContext(
        cwd=home,
        label=label,
        spill_dir=spill_dir,
        fetch_session=fetch_session,
    )


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
