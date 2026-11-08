"""Tests for the host actions: parsing, dispatch, filtering and presentation."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock, patch

import secrets

from treekle import json_loads

import pytest

from sagent.host import daemon, strings
from sagent.host.server import DEFAULT_TOOLS
from sagent.host.settings import HostSettings
from sagent.host.strings import (
    ConnectionString,
    Grant,
    Listing,
    ServeDirs,
    SessionRow,
)
from sagent.host.wire import DaemonControl, DaemonStatus
from sagent.settings.harvest import harvest


if TYPE_CHECKING:
    from pathlib import Path

    from sagent.settings.types import Setting


def _dirs(tmp_path: Path) -> ServeDirs:
    return ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")


def _actions(dirs: ServeDirs, shown: list[str] | None = None) -> dict[str, Setting]:
    owner = HostSettings(dirs=dirs, show=(shown if shown is not None else []).append)
    return {".".join(s.name): s for s in harvest([(("host",), owner)])}


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
    with patch.object(strings, "add", new_callable=AsyncMock) as add:
        said = await _actions(_dirs(tmp_path))["host.add"].apply("--help")
    assert said.startswith("usage: sagent host add")
    add.assert_not_awaited()


@pytest.mark.parametrize(
    ("args", "listen", "tools", "ttl_sec"),
    [
        ("", ("0.0.0.0:7777",), DEFAULT_TOOLS, 86_400),
        (
            '--listen [::1]:7778 --listen 127.0.0.1:0 --tools " Read, , Bash " --ttl 1.5m',
            ("[::1]:7778", "127.0.0.1:0"),
            ("Read", "Bash"),
            90,
        ),
        ("--disable --listen invalid --tools Read --ttl 2d", (), ("Read",), 172_800),
    ],
)
@pytest.mark.asyncio
async def test_serving_dispatches_configuration_and_shows_the_returned_address(
    tmp_path: Path,
    args: str,
    listen: tuple[str, ...],
    tools: tuple[str, ...],
    ttl_sec: float,
) -> None:
    dirs = _dirs(tmp_path)
    shown: list[str] = []
    status = DaemonStatus(pid=41, source_version="v", listen=("assigned:9000",))
    calls = Mock()
    ensure = AsyncMock(return_value=status)
    control = AsyncMock(return_value=status)
    calls.attach_mock(ensure, "ensure")
    calls.attach_mock(control, "control")
    with (
        patch.object(daemon, "ensure", ensure),
        patch.object(daemon, "control", control),
    ):
        assert await _actions(dirs, shown)["host"].apply(args) == ""
    assert calls.mock_calls == [
        ("ensure", (dirs,), {}),
        (
            "control",
            (dirs, DaemonControl(listen=listen, tools=tools, ttl_sec=ttl_sec)),
            {},
        ),
    ]
    assert shown == [
        (
            f"Serving {', '.join(tools)} on assigned:9000, as pid 41; make a string"
            " with `sagent host add`."
        ),
    ]


@pytest.mark.parametrize(
    ("args", "tools", "peer"),
    [
        ("", None, False),
        ('--tools "Read, Bash" --peer-messaging on', ("Read", "Bash"), True),
    ],
)
@pytest.mark.asyncio
async def test_add_forwards_grants_and_joins_every_connection_string(
    tmp_path: Path,
    args: str,
    tools: tuple[str, ...] | None,
    peer: bool,
) -> None:
    dirs = _dirs(tmp_path)
    token = secrets.token_urlsafe(16)
    made = (
        ConnectionString(host="one", port=7, fingerprint="SHA256:ab", token=token),
        ConnectionString(host="two", port=8, fingerprint="SHA256:ab", token=token),
    )
    with patch.object(strings, "add", new_callable=AsyncMock, return_value=made) as add:
        assert await _actions(dirs)["host.add"].apply(args) == (
            f"sagent://one:7/SHA256:ab/{token}\nsagent://two:8/SHA256:ab/{token}"
        )
    add.assert_awaited_once_with(dirs, tools=tools, peer_messaging=peer)


@pytest.mark.parametrize(
    ("args", "tools", "peer", "shown"),
    [
        ('--tools "Read, Bash"', ("Read", "Bash"), None, "Read, Bash"),
        ("--peer-messaging on", None, True, ""),
        ("--peer-messaging off", None, False, ""),
    ],
)
@pytest.mark.asyncio
async def test_edit_preserves_unspecified_grants(
    tmp_path: Path,
    args: str,
    tools: tuple[str, ...] | None,
    peer: bool | None,
    shown: str,
) -> None:
    dirs = _dirs(tmp_path)
    with patch.object(strings, "edit", new_callable=AsyncMock) as edit:
        assert await _actions(dirs)["host.edit"].apply(f"tok_one {args}") == (
            f"tok_one now grants {shown}."
        )
    edit.assert_awaited_once_with(dirs, "tok_one", tools=tools, peer_messaging=peer)


@pytest.mark.parametrize("connected", [False, True])
@pytest.mark.asyncio
async def test_listing_renders_grants_and_session_status(
    tmp_path: Path,
    connected: bool,
) -> None:
    grant = Grant(id="tok_a", sha256="hash", tools=None, added="today")
    row = SessionRow(
        token_id=grant.id,
        session_id="7f3a0c91e42b9999",
        client="desk",
        ip="100.64.0.2",
        connected=connected,
        since=0,
        calls=2,
    )
    dirs = _dirs(tmp_path)
    with patch.object(
        strings,
        "listing",
        return_value=Listing(grants=(grant,), sessions=(row,)),
    ) as listing:
        result = await _actions(dirs)["host.list"].apply("")
    listing.assert_called_once_with(dirs)
    clock = datetime.fromtimestamp(0, UTC).astimezone().strftime("%Y-%m-%d %H:%M")
    status = "up since" if connected else "gone since"
    assert result.splitlines() == [
        "STRINGS",
        "ID     TOOLS  ADDED",
        "tok_a  all    today",
        "",
        "SESSIONS",
        "TOKEN  SESSION       CLIENT  IP          STATUS"
        + (" " * (21 if connected else 23))
        + "CALLS",
        f"tok_a  7f3a0c91e42b  desk    100.64.0.2  {status} {clock}  2",
    ]


@pytest.mark.parametrize("running", [False, True])
@pytest.mark.asyncio
async def test_list_filters_both_grants_and_sessions_without_losing_stopped_state(
    tmp_path: Path,
    running: bool,
) -> None:
    first = Grant(id="tok_a", sha256="a", tools=("Read",), added="today")
    second = replace(first, id="tok_b", tools=())
    row = SessionRow(
        token_id=first.id,
        session_id="one",
        client="desk",
        ip="ip",
        connected=True,
        since=0,
        calls=1,
    )
    sessions = (row, replace(row, token_id=second.id)) if running else None
    with patch.object(
        strings,
        "listing",
        return_value=Listing(grants=(first, second), sessions=sessions),
    ):
        actions = _actions(_dirs(tmp_path))
        found = json_loads(
            await actions["host.list"].apply("--token tok_a --json"),
            Listing,
        )
        assert found == Listing(grants=(first,), sessions=(row,) if running else None)
        if not running:
            assert (await actions["host.list"].apply("")).splitlines()[-2:] == [
                "SESSIONS",
                "(server not running)",
            ]


@pytest.mark.parametrize(
    ("has_sessions", "token_id"),
    [(False, "tok_a"), (True, "tok_b")],
)
@pytest.mark.asyncio
async def test_delete_formats_the_sessions_ended(
    tmp_path: Path,
    has_sessions: bool,
    token_id: str,
) -> None:
    dirs = _dirs(tmp_path)
    row = SessionRow(
        token_id=token_id,
        session_id="one",
        client="desk",
        ip="ip",
        connected=False,
        since=0,
        calls=0,
    )
    ended = (
        (row, replace(row, session_id="two", client="laptop")) if has_sessions else ()
    )
    with patch.object(
        strings,
        "remove",
        new_callable=AsyncMock,
        return_value=ended,
    ) as remove:
        result = await _actions(dirs)["host.del"].apply(token_id)
    remove.assert_awaited_once_with(dirs, token_id)
    assert result == (
        f"Revoked {token_id}; sessions ended: "
        + ("desk (one), laptop (two)" if has_sessions else "none")
        + "."
    )


@pytest.mark.parametrize(
    ("verb", "args", "message"),
    [
        ("host.edit", "tok_none", "--tools or --peer-messaging is required"),
        ("host", "--ttl soon", "'soon' is no duration"),
        ("host.add", "--peer-messaging yes", "invalid choice"),
        ("host.edit", "tok_a --peer-messaging yes", "invalid choice"),
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


@pytest.mark.parametrize("verb", ["add", "edit", "remove"])
@pytest.mark.asyncio
async def test_storage_errors_reach_the_caller(tmp_path: Path, verb: str) -> None:
    error = ValueError("grant refused")
    action = {"add": "host.add", "edit": "host.edit", "remove": "host.del"}[verb]
    args = {"add": "", "edit": "tok_a --tools Read", "remove": "tok_a"}[verb]
    with (
        patch.object(strings, verb, new_callable=AsyncMock, side_effect=error),
        pytest.raises(ValueError, match="grant refused") as raised,
    ):
        _ = await _actions(_dirs(tmp_path))[action].apply(args)
    assert raised.value is error


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
