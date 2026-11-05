"""Tests for the host actions: what ``sagent host VERB`` and ``/host.VERB`` run."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import asyncio
import contextlib
import re

from treekle import json_dumps, json_loads

import pytest

from sagent.host import strings
from sagent.host.daemon import control, fingerprint
from sagent.host.server import Server
from sagent.host.settings import HostSettings
from sagent.host.strings import Listing, ServeDirs, SessionRow
from sagent.host.wire import DaemonControl
from sagent.lib.files.owner_only import write_private
from sagent.lib.posix.sockets import socket_address
from sagent.settings.harvest import harvest


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

    from sagent.settings.types import Setting


def _dirs(tmp_path: Path) -> ServeDirs:
    return ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")


def _actions(dirs: ServeDirs, shown: list[str] | None = None) -> dict[str, Setting]:
    owner = HostSettings(dirs=dirs, show=(shown if shown is not None else []).append)
    return {".".join(s.name): s for s in harvest([(("host",), owner)])}


@contextlib.asynccontextmanager
async def _running(dirs: ServeDirs, shown: list[str]) -> AsyncGenerator[None]:
    owner = HostSettings(dirs=dirs, show=shown.append)
    serve = {".".join(s.name): s for s in harvest([(("host",), owner)])}["host"]
    async with Server(
        dirs,
        listen=[],
        local_socket=socket_address(dirs.state / "daemon.sock"),
        source_version=fingerprint(),
    ):
        async with asyncio.timeout(5):
            _ = await serve.apply("--listen 127.0.0.1:0 --tools Read --ttl 90s")
        yield


def test_every_verb_is_an_action_the_command_line_reaches(tmp_path: Path) -> None:
    actions = _actions(_dirs(tmp_path))
    assert list(actions) == ["host", "host.add", "host.edit", "host.del", "host.list"]
    for action in actions.values():
        assert action.cli is True
        assert action.value is None
        assert action.help_long.startswith(f"usage: sagent {' '.join(action.name)}")
    assert "--listen IP:PORT" in actions["host"].usage
    assert "TOKEN_ID" in actions["host.edit"].usage


@pytest.mark.asyncio
async def test_help_is_returned_not_run(tmp_path: Path) -> None:
    said = await _actions(_dirs(tmp_path))["host.add"].apply("--help")
    assert said.startswith("usage: sagent host add")


@pytest.mark.asyncio
async def test_a_server_serves_and_its_strings_are_managed(tmp_path: Path) -> None:
    dirs = _dirs(tmp_path)
    shown: list[str] = []
    async with _running(dirs, shown):
        (line,) = shown
        assert re.fullmatch(
            r"Serving Read on 127\.0\.0\.1:([1-9]\d*), as pid \d+; make a string"
            r" with `sagent host add`\.",
            line,
        )
        actions = _actions(dirs)
        url = await actions["host.add"].apply("")
        assert url.startswith(
            f"sagent://127.0.0.1:{line.split(':')[1].split(',')[0]}/SHA256:",
        )
        (grant,) = strings.listing(dirs).grants
        listed = await actions["host.list"].apply("")
        assert listed.splitlines()[:3] == [
            "STRINGS",
            "ID        TOOLS  ADDED",
            f"{grant.id}  all    {grant.added}",
        ]
        assert listed.splitlines()[-2:] == [
            "SESSIONS",
            "TOKEN  SESSION  CLIENT  IP  STATUS  CALLS",
        ]
        assert await actions["host.edit"].apply(f"{grant.id} --tools Read") == (
            f"{grant.id} now grants Read."
        )
        as_json = json_loads(await actions["host.list"].apply("--json"), Listing)
        assert as_json.grants[0].tools == ("Read",)
        row = SessionRow(
            token_id=grant.id,
            session_id="7f3a0c91e42b9999",
            client="desk",
            ip="100.64.0.2",
            connected=False,
            since=0.0,
            calls=2,
        )
        write_private(
            dirs.state / "sessions.json",
            json_dumps((row,), target=tuple[SessionRow, ...]),
        )
        gone = datetime.fromtimestamp(0, UTC).astimezone().strftime("%Y-%m-%d %H:%M")
        assert (await actions["host.list"].apply("")).splitlines()[-1] == (
            f"{grant.id}  7f3a0c91e42b  desk    100.64.0.2  gone since {gone}  2"
        )
        assert await actions["host.del"].apply(grant.id) == (
            f"Revoked {grant.id}; sessions ended: desk (7f3a0c91e42b9999)."
        )
    listed = await _actions(dirs)["host.list"].apply("")
    assert listed.splitlines()[-2:] == ["SESSIONS", "(server not running)"]


@pytest.mark.asyncio
async def test_list_filters_by_token(tmp_path: Path) -> None:
    dirs = _dirs(tmp_path)
    async with _running(dirs, []):
        actions = _actions(dirs)
        _ = await actions["host.add"].apply("")
        _ = await actions["host.add"].apply("")
        first, _second = strings.listing(dirs).grants
        found = json_loads(
            await actions["host.list"].apply(f"--token {first.id} --json"),
            Listing,
        )
        assert found.grants == (first,)
        assert found.sessions == ()


@pytest.mark.parametrize(
    ("verb", "args", "message"),
    [
        ("host.add", "", "run sagent host once first"),
        ("host.edit", "tok_none --tools Read", "run sagent host once first"),
        ("host.del", "tok_none", "No string 'tok_none'"),
        ("host.edit", "tok_none", "--tools or --peer-messaging is required"),
        ("host", "--ttl soon", "'soon' is no duration"),
    ],
)
@pytest.mark.asyncio
async def test_a_mistake_is_a_setting_error(
    tmp_path: Path,
    verb: str,
    args: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _ = await _actions(_dirs(tmp_path))[verb].apply(args)


@pytest.mark.asyncio
async def test_owner_disables_tls_without_stopping_the_daemon(tmp_path: Path) -> None:
    dirs = _dirs(tmp_path)
    async with _running(dirs, []):
        actions = _actions(dirs)
        assert await actions["host"].apply("--disable") == ""
        status = await control(dirs, DaemonControl())
        assert status.listen == ()
        assert not status.restarting


@pytest.mark.asyncio
async def test_connection_strings_have_an_explicit_editable_peer_grant(
    tmp_path: Path,
) -> None:
    dirs = _dirs(tmp_path)
    async with _running(dirs, []):
        actions = _actions(dirs)
        _ = await actions["host.add"].apply("--peer-messaging on")
        (grant,) = strings.listing(dirs).grants
        assert grant.peer_messaging
        _ = await actions["host.edit"].apply(f"{grant.id} --peer-messaging off")
        (edited,) = strings.listing(dirs).grants
        assert not edited.peer_messaging
        assert edited.tools == grant.tools


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
