"""Connection strings: who may use a tool server, and which of its tools.

A string is ``sagent://HOST:PORT/SHA256:FINGERPRINT/TOKEN``. The fingerprint
pins the server's certificate; the token is 128 random bits, of which the
server keeps only the SHA-256. Whoever holds a string acts as the server's
user on its machine, as an ssh key does.

What ``sagent host add``, ``edit``, ``del`` and ``list`` do. Each works with the
server up or down: they write ``strings.json`` under a lock, and a running
server reads it again when it changes.

::

    <secrets>/                  key.pem, cert.pem, strings.json   (Grant per string)
    <state>/                    server.json  (ServerConfig, as last run)
                                server.pid   (locked by the running server)
                                sessions.json (SessionRow per client session)
                                server.jsonl (what happened, one line each)

Typical usage example:

  dirs = ServeDirs(secrets=..., state=...)
  made = await add(dirs, tools=("Read",))     # One string per address served.
  listing(dirs).grants                        # (Grant(id="tok_51c8", ...),)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast, override
from urllib import parse

import hashlib
import ipaddress
import secrets
import socket

from treekle import json_dumps, json_loads

from sagent.host.tls import fingerprint
from sagent.lib.files.owner_only import write_private
from sagent.lib.posix.pids import locked
from sagent.providers.login import file_lock


if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager


__all__ = [
    "Address",
    "ConnectionString",
    "Grant",
    "Listing",
    "ServeDirs",
    "ServerConfig",
    "SessionRow",
    "add",
    "configuration",
    "edit",
    "grants",
    "listing",
    "remove",
    "token_digest",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class Address:
    """Where a server listens: an IP, or a name, and a port."""

    host: str
    port: int

    @classmethod
    def from_text(cls, text: str) -> Address:
        """Return the address ``IP:PORT`` names, ``[V6]:PORT`` for IPv6.

        Args:
          text: Such as ``0.0.0.0:7777`` or ``[::1]:7777``.

        Returns:
          address: Its host, brackets stripped, and its port.

        Raises:
          ValueError: ``text`` gives no host, or no port in 0 to 65535; ``0``
            has the system pick one.

        """
        host, _, port = text.rpartition(":")
        host = host.removeprefix("[").removesuffix("]")
        if not host or not port.isdigit() or int(port) > 65_535:
            raise ValueError(f"{text!r} is no IP:PORT, such as 0.0.0.0:7777.")
        return cls(host=host, port=int(port))

    @override
    def __str__(self) -> str:
        return (
            f"[{self.host}]:{self.port}"
            if ":" in self.host
            else f"{self.host}:{self.port}"
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ConnectionString:
    """What a client needs to reach a server: where, its pin, and a token."""

    host: str
    port: int
    fingerprint: str
    """``SHA256:`` and the hex digest of the server certificate's DER bytes."""

    token: str

    @classmethod
    def from_url(cls, url: str) -> ConnectionString:
        """Return the string ``url`` spells.

        Args:
          url: As ``sagent host add`` printed it.

        Returns:
          string: Its host, port, pin and token.

        Raises:
          ValueError: ``url`` is no ``sagent://HOST:PORT/SHA256:FINGERPRINT/TOKEN``.

        """
        split = parse.urlsplit(url)
        parts = split.path.split("/")
        try:
            port = split.port
        except ValueError:
            port = None
        if (
            split.scheme != "sagent"
            or split.hostname in (None, "")
            or port is None
            or len(parts) != 3
            or not parts[1].startswith("SHA256:")
            or not parts[2]
        ):
            raise ValueError(
                f"{url!r} is no sagent://HOST:PORT/SHA256:FINGERPRINT/TOKEN.",
            )
        assert isinstance(split.hostname, str)
        return cls(host=split.hostname, port=port, fingerprint=parts[1], token=parts[2])

    @property
    def url(self) -> str:
        """Return the string as it is pasted: ``sagent://HOST:PORT/SHA256:.../TOKEN``."""
        where = Address(host=self.host, port=self.port)
        return f"sagent://{where}/{self.fingerprint}/{self.token}"


@dataclass(frozen=True, slots=True, kw_only=True)
class ServeDirs:
    """Where a tool server keeps what it must: its secrets apart from its state."""

    secrets: Path
    """The certificate, its key, and the strings' hashes."""

    state: Path
    """What it is doing: its config, pid, sessions and log."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Grant:
    """One string, as the server keeps it: its hash, never its token."""

    id: str
    """What ``edit`` and ``del`` name it by, such as ``tok_51c8``."""

    sha256: str
    tools: tuple[str, ...] | None
    """The tools it grants; ``None`` for every tool the server serves."""

    added: str
    """The day it was made, as ``YYYY-MM-DD``."""

    peer_messaging: bool = False
    """Whether this token may discover and message agents in other sessions."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ServerConfig:
    """What the server last ran with, which ``add`` mints strings for."""

    listen: tuple[Address, ...]
    tools: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionRow:
    """One client session, as the running server reports it."""

    token_id: str
    session_id: str
    client: str
    """The client's machine, as it calls itself."""

    ip: str
    connected: bool
    since: float
    """When it connected, or when it went, in seconds since the epoch."""

    calls: int
    """Calls running, or done and not acked."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Listing:
    """What ``sagent host list`` shows."""

    grants: tuple[Grant, ...]
    sessions: tuple[SessionRow, ...] | None
    """``None`` while no server runs."""


def token_digest(token: str) -> str:
    """Return the hash a server keeps of ``token``."""
    return hashlib.sha256(token.encode()).hexdigest()


async def add(
    dirs: ServeDirs,
    *,
    tools: Sequence[str] | None = None,
    peer_messaging: bool = False,
) -> list[ConnectionString]:
    """Mint a string; return it once per address the server listens at.

    Args:
      dirs: The server's.
      tools: What it grants, of the tools served; ``None`` for all of them.
      peer_messaging: Whether this token also grants cross-session messaging.

    Returns:
      strings: One per address, each with the same pin and token.

    Raises:
      ValueError: The server has never run, or does not serve a tool named.

    """
    config = configuration(dirs)
    _check(tools, config=config)
    token = secrets.token_urlsafe(16)
    async with _locked(dirs):
        kept = grants(dirs)
        taken = {g.id for g in kept}
        minted = f"tok_{secrets.token_hex(2)}"
        while minted in taken:
            minted = f"tok_{secrets.token_hex(2)}"
        grant = Grant(
            id=minted,
            sha256=token_digest(token),
            tools=tuple(tools) if tools is not None else None,
            added=datetime.now(UTC).date().isoformat(),
            peer_messaging=peer_messaging,
        )
        _write(dirs, [*kept, grant])
    pin = fingerprint(dirs.secrets)
    return [
        ConnectionString(host=host, port=address.port, fingerprint=pin, token=token)
        for address in config.listen
        for host in _advertised(address)
    ]


async def edit(
    dirs: ServeDirs,
    token_id: str,
    *,
    tools: Sequence[str] | None = None,
    peer_messaging: bool | None = None,
) -> None:
    """Grant the string ``token_id`` exactly ``tools``, from its next call on.

    Args:
      dirs: The server's.
      token_id: The string's id, such as ``tok_51c8``.
      tools: What it grants now; ``None`` leaves the tool grant unchanged.
      peer_messaging: Whether to grant cross-session messaging; ``None`` leaves it unchanged.

    Raises:
      ValueError: No string is ``token_id``, or a tool is not served.

    """
    _check(tools, config=configuration(dirs))
    async with _locked(dirs):
        kept = grants(dirs)
        _named(kept, token_id)
        _write(
            dirs,
            [
                replace(
                    g,
                    tools=tuple(tools) if tools is not None else g.tools,
                    peer_messaging=peer_messaging
                    if peer_messaging is not None
                    else g.peer_messaging,
                )
                if g.id == token_id
                else g
                for g in kept
            ],
        )


async def remove(dirs: ServeDirs, token_id: str) -> list[SessionRow]:
    """Revoke the string ``token_id``; return the sessions it held, which end now.

    Args:
      dirs: The server's.
      token_id: The string's id, such as ``tok_51c8``.

    Returns:
      sessions: Those the running server ends; none while it is down.

    Raises:
      ValueError: No string is ``token_id``.

    """
    async with _locked(dirs):
        kept = grants(dirs)
        _named(kept, token_id)
        _write(dirs, [g for g in kept if g.id != token_id])
    return [row for row in _sessions(dirs) if row.token_id == token_id]


def grants(dirs: ServeDirs) -> list[Grant]:
    """Return every string the server honors, oldest first."""
    path = dirs.secrets / "strings.json"
    return json_loads(path.read_text(), list[Grant]) if path.exists() else []


def listing(dirs: ServeDirs) -> Listing:
    """Return the strings, and the sessions while the server runs."""
    running = locked(dirs.state / "server.pid")
    return Listing(
        grants=tuple(grants(dirs)),
        sessions=tuple(_sessions(dirs)) if running else None,
    )


def configuration(
    dirs: ServeDirs,
    *,
    default: ServerConfig | None = None,
) -> ServerConfig:
    """Read the owner's saved listener and tool configuration.

    Args:
      dirs: Where the daemon keeps its state.
      default: Configuration for first startup, or ``None`` to require saved state.

    Returns:
      config: The saved configuration, or the supplied first-start default.

    Raises:
      ValueError: No configuration exists and no default was supplied.

    """
    path = dirs.state / "server.json"
    if not path.exists():
        if default is not None:
            return default
        raise ValueError("No server has listened here yet: run sagent host once first.")
    return json_loads(path.read_text(), ServerConfig)


def _check(tools: Sequence[str] | None, *, config: ServerConfig) -> None:
    unserved = sorted(set(tools or ()) - set(config.tools))
    if unserved:
        raise ValueError(
            f"The server does not serve {', '.join(unserved)}; it serves"
            f" {', '.join(config.tools)}.",
        )


def _named(kept: Sequence[Grant], token_id: str) -> None:
    if not any(g.id == token_id for g in kept):
        ids = ", ".join(g.id for g in kept) or "none"
        raise ValueError(f"No string {token_id!r}; strings: {ids}.")


def _locked(dirs: ServeDirs) -> AbstractAsyncContextManager[None]:
    return file_lock(dirs.secrets / "strings.json.lock")


def _write(dirs: ServeDirs, kept: Sequence[Grant]) -> None:
    write_private(
        dirs.secrets / "strings.json",
        json_dumps(kept, target=Sequence[Grant]),
    )


def _sessions(dirs: ServeDirs) -> list[SessionRow]:
    path = dirs.state / "sessions.json"
    return json_loads(path.read_text(), list[SessionRow]) if path.exists() else []


# A wildcard is no address a client can dial: the machine's name, then each
# address it resolves to, and the address its route out leaves by, stand in for
# it. The name alone may resolve to loopback, as Debian maps it to 127.0.1.1.
def _advertised(address: Address) -> list[str]:
    """Return what a client dials to reach ``address``."""
    if not ipaddress.ip_address(address.host).is_unspecified:
        return [address.host]
    name = socket.gethostname()
    found = {str(info[4][0]) for info in socket.getaddrinfo(name, None, socket.AF_INET)}
    found |= _routed_out()
    return [
        name,
        *sorted(ip for ip in found if not ipaddress.ip_address(ip).is_loopback),
    ]


# Connecting a UDP socket picks the route and sends nothing; 192.0.2.1 is a
# documentation address and 9 the discard port, so no packet could reach anyone
# if one were sent.
def _routed_out() -> set[str]:
    """Return the address this machine's route out leaves by; none with no route."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(("192.0.2.1", 9))
        except OSError:
            return set()
        return {cast(tuple[str, int], probe.getsockname())[0]}
