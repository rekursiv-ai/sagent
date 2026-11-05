"""The tool server as actions: ``sagent host`` runs it; ``add``, ``edit``, ``del``, ``list``.

Each is reached as ``sagent host VERB ARGS`` and as ``/host.VERB ARGS``, read by
one parser that is also the help. The verbs work with the server up or down.

Typical usage example:

  owners = [..., (("host",), HostSettings(dirs=dirs, show=print))]
  sagent host --listen 0.0.0.0:7777 --tools Bash,Read
  sagent host add --tools Read
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol, cast

import re

from treekle import json_dumps

from sagent.host import daemon, strings
from sagent.host.server import DEFAULT_TOOLS
from sagent.host.strings import Address, Listing, ServeDirs
from sagent.host.wire import DaemonControl
from sagent.settings.flags import ActionParser


if TYPE_CHECKING:
    from collections.abc import Awaitable, Sequence

    import argparse

    from sagent.settings.types import Setting


__all__ = [
    "HostSettings",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class HostSettings:
    """Exports what runs and manages this machine's tool server.

    Args:
      dirs: Where the server keeps its secrets and state.
      show: Shows the user what a long-running action is doing.

    """

    dirs: ServeDirs
    show: Callable[[str], object]

    def settings(self) -> list[Setting]:
        """Return the server, then each verb.

        Returns:
          settings: ``host`` itself, then ``add``, ``edit``, ``del``, ``list``.

        """
        verbs: Sequence[tuple[tuple[str, ...], str, _Handler]] = (
            ((), "serve this machine's tools to other machines", self._serve),
            (("add",), "make a connection string", self._add),
            (("edit",), "change the tools a string grants", self._edit),
            (("del",), "revoke a string, ending its sessions", self._del),
            (("list",), "show the strings, and who is connected", self._list),
        )
        return [
            _parser(name, description=described).setting(
                run,
                name=name,
                help_short=described,
            )
            for name, described, run in verbs
        ]

    async def _serve(self, parsed: argparse.Namespace) -> str:
        flags = cast("_Served", parsed)
        listen = (
            ()
            if flags.disable
            else tuple(
                str(Address.from_text(a)) for a in flags.listen or ("0.0.0.0:7777",)
            )
        )
        tools = _names(flags.tools)
        ttl_sec = _seconds(flags.ttl)
        _ = await daemon.ensure(self.dirs)
        status = await daemon.control(
            self.dirs,
            DaemonControl(listen=listen, tools=tools, ttl_sec=ttl_sec),
        )
        self.show(
            f"Serving {', '.join(tools)} on {', '.join(status.listen)}, as pid {status.pid};"
            " make a string with `sagent host add`.",
        )
        return ""

    async def _add(self, parsed: argparse.Namespace) -> str:
        flags = cast("_Granted", parsed)
        made = await strings.add(
            self.dirs,
            tools=_names(flags.tools) if flags.tools else None,
            peer_messaging=flags.peer_messaging == "on",
        )
        return "\n".join(s.url for s in made)

    async def _edit(self, parsed: argparse.Namespace) -> str:
        flags = cast("_Edited", parsed)
        if not flags.tools and flags.peer_messaging is None:
            raise ValueError("--tools or --peer-messaging is required.")
        await strings.edit(
            self.dirs,
            flags.token_id,
            tools=_names(flags.tools) if flags.tools else None,
            peer_messaging=None
            if flags.peer_messaging is None
            else flags.peer_messaging == "on",
        )
        return f"{flags.token_id} now grants {flags.tools}."

    async def _del(self, parsed: argparse.Namespace) -> str:
        flags = cast("_Named", parsed)
        ended = await strings.remove(self.dirs, flags.token_id)
        sessions = ", ".join(f"{r.client} ({r.session_id})" for r in ended) or "none"
        return f"Revoked {flags.token_id}; sessions ended: {sessions}."

    async def _list(self, parsed: argparse.Namespace) -> str:
        flags = cast("_Listed", parsed)
        found = strings.listing(self.dirs)
        if flags.token:
            found = Listing(
                grants=tuple(g for g in found.grants if g.id == flags.token),
                sessions=None
                if found.sessions is None
                else tuple(s for s in found.sessions if s.token_id == flags.token),
            )
        if flags.json:
            return json_dumps(found, target=Listing, indent=2)
        return _tables(found)


type _Handler = Callable[[argparse.Namespace], Awaitable[str]]
"""A verb's work, given what its parser read."""


class _Tooled(Protocol):
    tools: str


class _Served(_Tooled, Protocol):
    listen: list[str]
    ttl: str
    disable: bool


class _Granted(_Tooled, Protocol):
    peer_messaging: str | None


class _Named(Protocol):
    token_id: str


class _Edited(_Named, _Granted, Protocol):
    """What ``edit`` reads: an id and the grants to change."""


class _Listed(Protocol):
    token: str
    json: bool


def _parser(name: tuple[str, ...], *, description: str) -> ActionParser:
    """Return how ``sagent host`` and each verb read their arguments."""
    parser = ActionParser(
        ("host", *name),
        description=f"{description[:1].upper()}{description[1:]}.",
    )
    if name == ():
        parser.add_argument(
            "--disable",
            action="store_true",
            help="disable remote TLS access; keep local messaging running",
        )
        parser.add_argument(
            "--listen",
            action="append",
            metavar="IP:PORT",
            help="where to listen; repeatable (default: 0.0.0.0:7777)",
        )
        parser.add_argument(
            "--tools",
            default=",".join(DEFAULT_TOOLS),
            metavar="T,T,...",
            help=f"the tools served (default: {','.join(DEFAULT_TOOLS)})",
        )
        parser.add_argument(
            "--ttl",
            default="24h",
            metavar="DURATION",
            help="how long a disconnected session is kept, as 90s, 30m, 24h, 7d"
            " (default: 24h)",
        )
    if name in (("edit",), ("del",)):
        parser.add_argument("token_id", metavar="TOKEN_ID", help="the string's id")
    if name in (("add",), ("edit",)):
        parser.add_argument(
            "--tools",
            default="",
            metavar="T,T,...",
            help="the tools it grants, of those served"
            + ("" if name == ("edit",) else " (default: all)"),
        )
        parser.add_argument(
            "--peer-messaging",
            choices=("on", "off"),
            default=None,
            help="grant cross-session discovery and messaging (default: off)",
        )
    if name == ("list",):
        parser.add_argument(
            "--token",
            default="",
            metavar="TOKEN_ID",
            help="show one string",
        )
        parser.add_argument("--json", action="store_true", help="print JSON")
    return parser


def _names(tools: str) -> tuple[str, ...]:
    return tuple(t.strip() for t in tools.split(",") if t.strip())


def _seconds(duration: str) -> float:
    """Return ``90s``, ``30m``, ``24h`` or ``7d`` in seconds."""
    found = re.fullmatch(r"(\d+(?:\.\d+)?)([smhd])", duration.strip())
    if found is None:
        raise ValueError(f"{duration!r} is no duration, such as 90s, 30m, 24h, 7d")
    unit = {"s": 1, "m": 60, "h": 3_600, "d": 86_400}[found[2]]
    return float(found[1]) * unit


def _tables(found: Listing) -> str:
    """Return the strings, then the sessions, as aligned tables."""
    grant_rows = [
        (g.id, ",".join(g.tools) if g.tools is not None else "all", g.added)
        for g in found.grants
    ]
    lines = ["STRINGS", *_aligned([("ID", "TOOLS", "ADDED"), *grant_rows])]
    lines += ["", "SESSIONS"]
    if found.sessions is None:
        lines.append("(server not running)")
        return "\n".join(lines)
    session_rows = [
        (
            s.token_id,
            s.session_id[:12],
            s.client,
            s.ip,
            f"{'up since' if s.connected else 'gone since'} {_clock(s.since)}",
            str(s.calls),
        )
        for s in found.sessions
    ]
    lines += _aligned(
        [("TOKEN", "SESSION", "CLIENT", "IP", "STATUS", "CALLS"), *session_rows],
    )
    return "\n".join(lines)


def _aligned(rows: Sequence[tuple[str, ...]]) -> list[str]:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return [
        "  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)).rstrip()
        for row in rows
    ]


def _clock(at: float) -> str:
    return datetime.fromtimestamp(at, UTC).astimezone().strftime("%Y-%m-%d %H:%M")
