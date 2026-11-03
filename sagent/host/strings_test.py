"""Tests for connection strings: what ``sagent host add``, ``edit``, ``del`` and ``list`` do."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import fcntl
import os
import secrets
import socket

from treekle import json_dumps

import pytest

from sagent.host import strings
from sagent.host.strings import (
    Address,
    ConnectionString,
    ServeDirs,
    ServerConfig,
    SessionRow,
    token_digest,
)
from sagent.host.tls import fingerprint
from sagent.lib.files.owner_only import write_private


if TYPE_CHECKING:
    from pathlib import Path


_TOOLS = ("Bash", "Read", "Edit")


def _dirs(tmp_path: Path) -> ServeDirs:
    return ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")


def _served(dirs: ServeDirs, *listen: str) -> None:
    config = ServerConfig(
        listen=tuple(Address.from_text(a) for a in listen or ("127.0.0.1:7777",)),
        tools=_TOOLS,
    )
    write_private(dirs.state / "server.json", json_dumps(config, target=ServerConfig))


@pytest.mark.parametrize(
    ("text", "address"),
    [
        ("10.0.0.5:7777", Address(host="10.0.0.5", port=7777)),
        ("0.0.0.0:1", Address(host="0.0.0.0", port=1)),  # noqa: S104 -- Parsed, never bound.
        ("[::1]:7778", Address(host="::1", port=7778)),
    ],
)
def test_an_address_reads_as_written(text: str, address: Address) -> None:
    assert Address.from_text(text) == address
    assert str(address) == text


@pytest.mark.parametrize("text", ["7777", "host:", "host:port", "a:70000"])
def test_an_address_without_a_port_is_refused(text: str) -> None:
    with pytest.raises(ValueError, match="IP:PORT"):
        _ = Address.from_text(text)


def test_a_string_reads_back_from_its_url() -> None:
    token = secrets.token_urlsafe(16)
    given = ConnectionString(
        host="gpu1",
        port=7777,
        fingerprint="SHA256:ab",
        token=token,
    )
    assert given.url == f"sagent://gpu1:7777/SHA256:ab/{token}"
    assert ConnectionString.from_url(given.url) == given
    v6 = ConnectionString(host="::1", port=1, fingerprint="SHA256:ab", token=token)
    assert v6.url.startswith("sagent://[::1]:1/")
    assert ConnectionString.from_url(v6.url) == v6


@pytest.mark.parametrize(
    "url",
    [
        "http://gpu1:7777/SHA256:ab/t",
        "sagent://gpu1/SHA256:ab/t",
        "sagent://gpu1:7777/SHA256:ab",
        "sagent://gpu1:7777/MD5:ab/t",
        "sagent://gpu1:7777/SHA256:ab/t/more",
        "sagent://gpu1:99999/SHA256:ab/t",
    ],
)
def test_a_url_that_is_no_string_is_refused(url: str) -> None:
    with pytest.raises(ValueError, match="sagent://HOST:PORT/SHA256:FINGERPRINT/TOKEN"):
        _ = ConnectionString.from_url(url)


@pytest.mark.asyncio
async def test_add_needs_a_server_that_has_run(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run sagent host once first"):
        _ = await strings.add(_dirs(tmp_path))


@pytest.mark.asyncio
async def test_add_mints_a_string_per_address_and_keeps_only_its_hash(
    tmp_path: Path,
) -> None:
    dirs = _dirs(tmp_path)
    _served(dirs, "10.0.0.5:7777", "100.64.0.5:7778")
    made = await strings.add(dirs, tools=("Read",))
    assert [(s.host, s.port) for s in made] == [
        ("10.0.0.5", 7777),
        ("100.64.0.5", 7778),
    ]
    assert {s.token for s in made} == {made[0].token}
    assert {s.fingerprint for s in made} == {fingerprint(dirs.secrets)}
    assert len(made[0].token) >= 22
    (grant,) = strings.listing(dirs).grants
    assert grant.sha256 == token_digest(made[0].token)
    assert grant.tools == ("Read",)
    assert made[0].token not in (dirs.secrets / "strings.json").read_text()


@pytest.mark.asyncio
async def test_a_new_string_never_takes_a_kept_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dirs = _dirs(tmp_path)
    _served(dirs)
    drawn = iter(["aaaa", "aaaa", "bbbb"])

    def token_hex(nbytes: int) -> str:
        del nbytes
        return next(drawn)

    monkeypatch.setattr(secrets, "token_hex", token_hex)
    _ = await strings.add(dirs)
    _ = await strings.add(dirs)
    assert [g.id for g in strings.listing(dirs).grants] == ["tok_aaaa", "tok_bbbb"]


@pytest.mark.asyncio
async def test_add_without_tools_grants_every_tool_served(tmp_path: Path) -> None:
    dirs = _dirs(tmp_path)
    _served(dirs)
    _ = await strings.add(dirs)
    (grant,) = strings.listing(dirs).grants
    assert grant.tools is None


@pytest.mark.asyncio
async def test_add_refuses_a_tool_the_server_does_not_serve(tmp_path: Path) -> None:
    dirs = _dirs(tmp_path)
    _served(dirs)
    with pytest.raises(ValueError, match="WebSearch"):
        _ = await strings.add(dirs, tools=("Read", "WebSearch"))
    assert strings.listing(dirs).grants == ()


@pytest.mark.asyncio
async def test_a_wildcard_address_is_advertised_by_this_machine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def getaddrinfo(
        host: str,
        port: None,
        family: socket.AddressFamily,
    ) -> list[tuple[object, object, object, object, tuple[str, int]]]:
        assert (host, port, family) == ("gpu1", None, socket.AF_INET)
        return [
            (family, 0, 0, "", (ip, 0)) for ip in ("10.0.0.5", "127.0.1.1", "10.0.0.5")
        ]

    class _Routed:
        """A UDP socket whose route out leaves by ``192.168.1.175``."""

        def __init__(
            self,
            family: socket.AddressFamily,
            kind: socket.SocketKind,
        ) -> None:
            assert (family, kind) == (socket.AF_INET, socket.SOCK_DGRAM)

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *exc: object) -> None:
            del exc

        def connect(self, address: tuple[str, int]) -> None:
            assert address == ("192.0.2.1", 9)

        def getsockname(self) -> tuple[str, int]:
            return ("192.168.1.175", 40_000)

    monkeypatch.setattr(socket, "gethostname", lambda: "gpu1")
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(socket, "socket", _Routed)
    dirs = _dirs(tmp_path)
    _served(dirs, "0.0.0.0:7777")
    made = await strings.add(dirs)
    assert [(s.host, s.port) for s in made] == [
        ("gpu1", 7777),
        ("10.0.0.5", 7777),
        ("192.168.1.175", 7777),
    ]


@pytest.mark.asyncio
async def test_a_machine_with_no_route_out_advertises_what_its_name_resolves_to(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Unrouted:
        def __init__(
            self,
            family: socket.AddressFamily,
            kind: socket.SocketKind,
        ) -> None:
            del family, kind

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *exc: object) -> None:
            del exc

        def connect(self, address: tuple[str, int]) -> None:
            del address
            raise OSError("Network is unreachable")

    def unresolved(*args: object) -> list[object]:
        del args
        return []

    monkeypatch.setattr(socket, "gethostname", lambda: "gpu1")
    monkeypatch.setattr(socket, "getaddrinfo", unresolved)
    monkeypatch.setattr(socket, "socket", _Unrouted)
    dirs = _dirs(tmp_path)
    _served(dirs, "0.0.0.0:7777")
    assert [s.host for s in await strings.add(dirs)] == ["gpu1"]


@pytest.mark.asyncio
async def test_edit_changes_a_grant_and_del_revokes_it(tmp_path: Path) -> None:
    dirs = _dirs(tmp_path)
    _served(dirs)
    _ = await strings.add(dirs)
    _ = await strings.add(dirs)
    first, second = strings.listing(dirs).grants
    await strings.edit(dirs, first.id, tools=("Bash",))
    assert [g.tools for g in strings.listing(dirs).grants] == [("Bash",), None]
    _ = await strings.remove(dirs, first.id)
    assert strings.listing(dirs).grants == (second,)


@pytest.mark.asyncio
async def test_edit_and_del_refuse_an_unknown_id(tmp_path: Path) -> None:
    dirs = _dirs(tmp_path)
    _served(dirs)
    with pytest.raises(ValueError, match="tok_none"):
        await strings.edit(dirs, "tok_none", tools=("Bash",))
    with pytest.raises(ValueError, match="tok_none"):
        _ = await strings.remove(dirs, "tok_none")


@pytest.mark.asyncio
async def test_edit_refuses_a_tool_the_server_does_not_serve(tmp_path: Path) -> None:
    dirs = _dirs(tmp_path)
    _served(dirs)
    _ = await strings.add(dirs)
    (grant,) = strings.listing(dirs).grants
    with pytest.raises(ValueError, match="Train"):
        await strings.edit(dirs, grant.id, tools=("Train",))


def _session(token_id: str, session_id: str) -> SessionRow:
    return SessionRow(
        token_id=token_id,
        session_id=session_id,
        client="laptop",
        ip="100.64.0.2",
        connected=True,
        since=0.0,
        calls=0,
    )


@pytest.mark.asyncio
async def test_sessions_show_only_while_the_server_runs(tmp_path: Path) -> None:
    dirs = _dirs(tmp_path)
    _served(dirs)
    rows = (_session("tok_a", "1"), _session("tok_b", "2"))
    write_private(
        dirs.state / "sessions.json",
        json_dumps(rows, target=tuple[SessionRow, ...]),
    )
    assert strings.listing(dirs).sessions is None
    write_private(dirs.state / "server.pid", str(os.getpid()))
    fd = os.open(dirs.state / "server.pid", os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        assert strings.listing(dirs).sessions == rows
    finally:
        os.close(fd)


@pytest.mark.asyncio
async def test_del_says_which_sessions_it_drops(tmp_path: Path) -> None:
    dirs = _dirs(tmp_path)
    _served(dirs)
    _ = await strings.add(dirs)
    (grant,) = strings.listing(dirs).grants
    rows = (_session(grant.id, "1"), _session("tok_other", "2"))
    write_private(
        dirs.state / "sessions.json",
        json_dumps(rows, target=tuple[SessionRow, ...]),
    )
    assert await strings.remove(dirs, grant.id) == [rows[0]]


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
