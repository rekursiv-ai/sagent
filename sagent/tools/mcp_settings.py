"""MCP servers as actions: list, get, add, add-json, remove, login, logout.

Each is reached as ``sagent mcp VERB ARGS`` and as ``/mcp.VERB ARGS``, with
the arguments Claude Code's ``claude mcp`` takes, read by one parser that is
also the help. What they change, ``mcp.json`` and saved sign-ins, the next
session reads; a session already open keeps the servers it started with.

Typical usage example:

  owners = [..., (("mcp",), McpSettings(cwd=Path.cwd(), show=print))]
  sagent mcp add --transport http linear https://mcp.linear.app/mcp
  /mcp.add my-server -e API_KEY=xxx -- npx my-mcp-server
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import argparse
import os

from treekle import PlainTree, json_loads

from sagent.providers.login import headless_wanted
from sagent.settings.exceptions import SettingError
from sagent.settings.flags import ActionParser
from sagent.settings.types import Setting
from sagent.tools.mcp import (
    McpServer,
    McpSession,
    add_server,
    login,
    logout,
    remove_server,
    servers,
)


if TYPE_CHECKING:
    from collections.abc import Awaitable, Mapping, Sequence

    from sagent.tools.mcp import McpScope


__all__ = [
    "McpSettings",
    "add_config_arguments",
]


def add_config_arguments(parser: argparse.ArgumentParser) -> list[argparse.Action]:
    """Register ``--mcp-config`` and ``--strict-mcp-config`` on ``parser``.

    A session and the actions that read servers take the same two flags.

    Args:
      parser: The parser to add them to.

    Returns:
      actions: The two flags, as ``add_argument`` returned them.

    """
    return [
        parser.add_argument(
            "--mcp-config",
            action="append",
            default=[],
            metavar="PATH|JSON",
            help="MCP servers to add, as an mcpServers file or its JSON;"
            " repeatable. Read after the user's and the project's mcp.json.",
        ),
        parser.add_argument(
            "--strict-mcp-config",
            action="store_true",
            help="Read MCP servers only from --mcp-config.",
        ),
    ]


@dataclass(frozen=True, slots=True, kw_only=True)
class McpSettings:
    """Exports what manages MCP servers: each an action, at the prompt and the shell.

    Args:
      cwd: The working directory, which names its project's ``mcp.json``.
      show: Shows the user a sign-in's page and what to do with it.

    """

    cwd: Path
    show: Callable[[str], object]

    def settings(self) -> list[Setting]:
        """Return each action, in the order ``claude mcp`` lists them.

        Returns:
          settings: ``list``, ``get``, ``add``, ``add-json``, ``remove``,
            ``login``, ``logout``.

        """
        verbs: Sequence[tuple[str, str, _Handler]] = (
            ("list", "list the MCP servers and whether each connects", self._list),
            ("get", "show one MCP server and whether it connects", self._get),
            ("add", "add an MCP server: a command to run, or a URL", self._add),
            ("add-json", "add an MCP server from its JSON entry", self._add_json),
            ("remove", "remove an MCP server, and its sign-in", self._remove),
            ("login", "sign in to a remote MCP server with OAuth", self._login),
            ("logout", "forget a remote MCP server's sign-in", self._logout),
        )
        settings: list[Setting] = []
        for verb, described, run in verbs:
            parser = _parser(verb, description=described)
            settings.append(
                Setting(
                    name=(verb,),
                    help_short=described,
                    help_long=parser.format_help(),
                    usage=parser.format_usage().partition(verb)[2].strip(),
                    apply=partial(_run, parser, run),
                    cli=True,
                    flags=parser.options,
                ),
            )
        return settings

    async def _list(self, parsed: argparse.Namespace) -> str:
        found = self._servers(cast(_Configured, parsed))
        if not found:
            return "No MCP servers. Add one with `sagent mcp add`; see its --help."
        return "\n".join(await _rows(found))

    async def _get(self, parsed: argparse.Namespace) -> str:
        flags = cast(_Named, parsed)
        server = _named(self._servers(flags), flags.name)
        (row,) = await _rows([server])
        kind = "url" if server.transport == "http" else "command"
        described = [f"  {server.description}"] if server.description else []
        return "\n".join([row, f"  {kind}: {server.target}", *described])

    async def _add(self, parsed: argparse.Namespace) -> str:
        flags = cast(_Added, parsed)
        entry = _entry(flags)
        await add_server(flags.name, entry, cwd=self.cwd, scope=flags.scope)
        return _added(flags.name, flags.scope)

    async def _add_json(self, parsed: argparse.Namespace) -> str:
        flags = cast(_AddedJson, parsed)
        try:
            entry = json_loads(flags.json, dict[str, PlainTree])
        except ValueError as e:
            raise SettingError(f"{flags.json!r} is no JSON object: {e}") from e
        await add_server(flags.name, entry, cwd=self.cwd, scope=flags.scope)
        return _added(flags.name, flags.scope)

    async def _remove(self, parsed: argparse.Namespace) -> str:
        flags = cast(_Removed, parsed)
        scope = await remove_server(flags.name, cwd=self.cwd, scope=flags.scope)
        return (
            f"Removed MCP server {flags.name!r} from {scope}; sessions started from"
            " now go without it."
        )

    async def _login(self, parsed: argparse.Namespace) -> str:
        flags = cast(_LoggedIn, parsed)
        server = _named(self._servers(flags), flags.name)
        headless = headless_wanted(os.environ, given=flags.headless)
        await login(server, show=self.show, headless=headless)
        return f"Signed in to MCP server {flags.name!r}."

    async def _logout(self, parsed: argparse.Namespace) -> str:
        flags = cast(_Named, parsed)
        if await logout(_named(self._servers(flags), flags.name)):
            return f"Signed out of MCP server {flags.name!r}."
        return f"MCP server {flags.name!r} had no sign-in."

    def _servers(self, flags: _Configured) -> list[McpServer]:
        return servers(self.cwd, flags=flags.mcp_config, strict=flags.strict_mcp_config)


type _Handler = Callable[[argparse.Namespace], Awaitable[str]]
"""A verb's work, given what its parser read."""


class _Configured(Protocol):
    """What ``list`` reads: the config flags."""

    mcp_config: list[str]
    strict_mcp_config: bool


class _Named(_Configured, Protocol):
    """What ``get`` and ``logout`` read."""

    name: str


class _LoggedIn(_Named, Protocol):
    """What ``login`` reads."""

    headless: bool | None


class _Added(Protocol):
    """What ``add`` reads."""

    name: str
    target: str
    args: list[str]
    transport: str
    env: list[str]
    header: list[str]
    scope: McpScope


class _AddedJson(Protocol):
    """What ``add-json`` reads."""

    name: str
    json: str
    scope: McpScope


class _Removed(Protocol):
    """What ``remove`` reads."""

    name: str
    scope: McpScope | None


async def _run(parser: ActionParser, run: _Handler, args: str) -> str:
    """Read ``args`` with ``parser``, then run; return its help when asked."""
    parsed = parser.read(args)
    if parsed is None:
        return parser.format_help()
    return await run(parsed)


def _parser(verb: str, *, description: str) -> ActionParser:
    """Return how ``verb`` reads its arguments, as ``claude mcp VERB`` does."""
    parser = ActionParser(
        ("mcp", verb),
        description=f"{description[:1].upper()}{description[1:]}.",
    )
    if verb != "list":
        parser.add_argument("name", help="the server's name")
    if verb == "add":
        parser.add_argument(
            "target",
            metavar="commandOrUrl",
            help="what to run, or its URL",
        )
        parser.add_argument("args", nargs="*", help="the command's arguments, after --")
        parser.add_argument(
            "-t",
            "--transport",
            choices=("stdio", "http"),
            default="stdio",
            help="stdio runs a command; http reaches a URL (default: stdio)",
        )
        parser.add_argument(
            "-e",
            "--env",
            action="append",
            default=[],
            metavar="KEY=value",
            help="an environment variable for a stdio server; repeatable",
        )
        parser.add_argument(
            "-H",
            "--header",
            action="append",
            default=[],
            metavar="'Key: value'",
            help="a header for an http server; repeatable",
        )
    if verb == "add-json":
        parser.add_argument("json", help="its mcpServers entry, as JSON")
    if verb in ("add", "add-json", "remove"):
        parser.add_argument(
            "-s",
            "--scope",
            choices=("local", "user"),
            default="local" if verb != "remove" else None,
            help="local: this project's mcp.json; user: every project's"
            + (" (default: local)" if verb != "remove" else " (default: either)"),
        )
    if verb in ("list", "get", "login", "logout"):
        add_config_arguments(parser)
    if verb == "login":
        parser.add_argument(
            "--headless",
            action=argparse.BooleanOptionalAction,
            help="print the URL and take the address pasted back; default: over"
            " ssh, or on Linux with no display",
        )
    return parser


def _entry(flags: _Added) -> Mapping[str, PlainTree]:
    """Return the ``mcpServers`` entry ``add``'s flags describe."""
    if flags.transport == "http":
        if flags.env:
            raise SettingError("-e/--env is for a stdio server")
        if flags.args:
            raise SettingError("an http server takes no arguments")
        entry: dict[str, PlainTree] = {"url": flags.target}
        if flags.header:
            entry["headers"] = dict(
                _pair(h, sep=":", form="'Key: value'", flag="-H") for h in flags.header
            )
        return entry
    if flags.header:
        raise SettingError("-H/--header is for an http server")
    entry = {"command": flags.target}
    if flags.args:
        entry["args"] = list(flags.args)
    if flags.env:
        entry["env"] = dict(
            _pair(e, sep="=", form="KEY=value", flag="-e") for e in flags.env
        )
    return entry


def _pair(given: str, *, sep: str, form: str, flag: str) -> tuple[str, str]:
    key, found, value = given.partition(sep)
    if not found or not key.strip():
        raise SettingError(f"{flag} takes {form}, not {given!r}")
    return key.strip(), value.strip()


def _named(found: Sequence[McpServer], name: str) -> McpServer:
    for server in found:
        if server.name == name:
            return server
    names = ", ".join(s.name for s in found) or "none"
    raise SettingError(f"No MCP server {name!r}; configured: {names}")


async def _rows(found: Sequence[McpServer]) -> list[str]:
    """Connect to each of ``found``; return a line apiece saying how it went."""
    async with McpSession(found) as session:
        status = session.status()
    cells = [(s.name, s.transport, s.loading) for s in found]
    widths = [max(len(row[i]) for row in cells) for i in range(3)]
    return [
        "  ".join(
            [*(c.ljust(w) for c, w in zip(row, widths, strict=True)), status[s.name]],
        )
        for row, s in zip(cells, found, strict=True)
    ]


def _added(name: str, scope: McpScope) -> str:
    return f"Added MCP server {name!r} to {scope}; sessions started from now use it."
