"""Other machines' tool servers as actions: ``remote add``, ``del``, ``list``.

Each is reached as ``sagent remote VERB ARGS`` and as ``/remote.VERB ARGS``.
What they change, the saved strings, the next session reads; a session already
open keeps the servers it started with, as it keeps its MCP servers.

Typical usage example:

  owners = [..., (("remote",), RemoteSettings())]
  sagent remote add sagent://gpu1:7777/SHA256:9f3a.../tok... --tools Bash,Read
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import asyncio
import contextlib

from treekle import json_dumps

from sagent.host.strings import ConnectionString
from sagent.lib.files.owner_only import write_private
from sagent.lib.userdirs import config_dir
from sagent.settings.flags import ActionParser
from sagent.tools.remote import RemoteSession, SavedRemote, saved


if TYPE_CHECKING:
    from collections.abc import Awaitable, Sequence

    import argparse

    from sagent.settings.types import Setting


__all__ = [
    "RemoteSettings",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class RemoteSettings:
    """Exports what keeps the strings of other machines' servers.

    Args:
      connect_sec: How long ``list`` waits for each server to answer.

    """

    connect_sec: float = 3.0

    def settings(self) -> list[Setting]:
        """Return each action.

        Returns:
          settings: ``add``, ``del``, ``list``.

        """
        verbs: Sequence[tuple[str, str, _Handler]] = (
            ("add", "save a server's connection string", self._add),
            ("del", "forget a server's string", self._del),
            ("list", "show the saved servers and whether each answers", self._list),
        )
        return [
            _parser(verb, description=described).setting(
                run,
                name=(verb,),
                help_short=described,
            )
            for verb, described, run in verbs
        ]

    async def _add(self, parsed: argparse.Namespace) -> str:
        flags = cast("_Added", parsed)
        added = ConnectionString.from_url(flags.url)
        tools = tuple(t.strip() for t in flags.tools.split(",") if t.strip()) or None
        # Saved again, a server's string replaces the one it had at that address.
        kept = [r for r in saved() if _where(r) != _where(SavedRemote(url=added.url))]
        _write([*kept, SavedRemote(url=added.url, tools=tools)])
        return f"Saved {added.host}:{added.port}; sessions started from now reach it."

    async def _del(self, parsed: argparse.Namespace) -> str:
        flags = cast("_Named", parsed)
        kept = saved()
        left = [r for r in kept if _where(r)[0] != flags.name]
        if len(left) == len(kept):
            names = ", ".join(_where(r)[0] for r in kept) or "none"
            raise ValueError(f"No saved server {flags.name!r}; saved: {names}")
        _write(left)
        return f"Forgot {flags.name}; sessions started from now go without it."

    async def _list(self, parsed: argparse.Namespace) -> str:
        del parsed
        kept = saved()
        if not kept:
            return "No servers. Save one with `sagent remote add`; see its --help."
        async with RemoteSession(kept, retry_sec=self.connect_sec) as session:
            # Each answers, or fails to, within ``connect_sec``; one still trying
            # then is listed as down.
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(self.connect_sec):
                    for endpoint in session.endpoints().values():
                        await endpoint.until(lambda e: e.tried)
            named = session.endpoints()
            return "\n".join(
                f"{name}  {e.string.host}:{e.string.port}  "
                + (
                    f"up: {', '.join(e.tools) or 'no tools'}"
                    if e.up
                    else f"down: {e.error}"
                )
                for name, e in named.items()
            )


type _Handler = Callable[[argparse.Namespace], Awaitable[str]]
"""A verb's work, given what its parser read."""


class _Added(Protocol):
    url: str
    tools: str


class _Named(Protocol):
    name: str


def _parser(verb: str, *, description: str) -> ActionParser:
    """Return how ``verb`` reads its arguments."""
    parser = ActionParser(
        ("remote", verb),
        description=f"{description[:1].upper()}{description[1:]}.",
    )
    if verb == "add":
        parser.add_argument(
            "url",
            metavar="SAGENT_URL",
            help="the string `sagent host add` printed there",
        )
        parser.add_argument(
            "--tools",
            default="",
            metavar="T,T,...",
            help="the tools to ask for (default: every one the string grants)",
        )
    if verb == "del":
        parser.add_argument("name", metavar="HOST", help="the server's host, as saved")
    return parser


def _where(remote: SavedRemote) -> tuple[str, int]:
    string = ConnectionString.from_url(remote.url)
    return string.host, string.port


def _write(kept: Sequence[SavedRemote]) -> None:
    write_private(
        config_dir() / "rekursiv-ai" / "secrets" / "sagent" / "remote.json",
        json_dumps(list(kept), target=list[SavedRemote]),
    )
