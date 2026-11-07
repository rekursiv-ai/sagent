"""Tests for the MCP actions: what ``sagent mcp VERB`` and ``/mcp.VERB`` run."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

from treekle import PlainTree, json_dumps, json_loads

import pytest

from sagent.lib.userdirs import data_dir
from sagent.session.projects import project_dir
from sagent.settings.exceptions import SettingError
from sagent.settings.harvest import harvest
from sagent.tools import mcp_settings
from sagent.tools.mcp import McpServer, servers
from sagent.tools.mcp_auth import McpTokenStorage
from sagent.tools.mcp_settings import McpSettings


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from sagent.settings.types import Setting


_VERBS = ["list", "get", "add", "add-json", "remove", "login", "logout"]


def _actions(tmp_path: Path, shown: list[str] | None = None) -> dict[str, Setting]:
    owner = McpSettings(cwd=tmp_path, show=(shown if shown is not None else []).append)
    return {".".join(s.name): s for s in harvest([(("mcp",), owner)])}


async def _run(tmp_path: Path, verb: str, args: str = "") -> str:
    return await _actions(tmp_path)[f"mcp.{verb}"].apply(args)


def _entries(path: Path) -> dict[str, PlainTree]:
    return json_loads(path.read_text(), dict[str, dict[str, PlainTree]])["mcpServers"]


def _local(tmp_path: Path) -> dict[str, PlainTree]:
    return _entries(project_dir(tmp_path) / "mcp.json")


def test_every_verb_is_an_action_the_command_line_reaches(tmp_path: Path) -> None:
    actions = _actions(tmp_path)
    assert list(actions) == [f"mcp.{verb}" for verb in _VERBS]
    for action in actions.values():
        assert action.cli is True
        assert action.value is None
        assert action.help_short
        assert action.help_long.startswith(f"usage: sagent {' '.join(action.name)}")


def test_each_verb_says_what_it_does_and_how_it_is_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COLUMNS", "300")
    actions = _actions(tmp_path)
    assert [(a.help_short, a.usage) for a in actions.values()] == [
        (
            "list the MCP servers and whether each connects",
            "[-h] [--mcp-config PATH|JSON] [--strict-mcp-config]",
        ),
        (
            "show one MCP server and whether it connects",
            "[-h] [--mcp-config PATH|JSON] [--strict-mcp-config] name",
        ),
        (
            "add an MCP server: a command to run, or a URL",
            (
                "[-h] [-t {stdio,http}] [-e KEY=value] [-H 'Key: value'] [-s {local,user}]"
                " name commandOrUrl [args ...]"
            ),
        ),
        (
            "add an MCP server from its JSON entry",
            "[-h] [-s {local,user}] name json",
        ),
        ("remove an MCP server, and its sign-in", "[-h] [-s {local,user}] name"),
        (
            "sign in to a remote MCP server with OAuth",
            (
                "[-h] [--mcp-config PATH|JSON] [--strict-mcp-config]"
                " [--headless | --no-headless] name"
            ),
        ),
        (
            "forget a remote MCP server's sign-in",
            "[-h] [--mcp-config PATH|JSON] [--strict-mcp-config] name",
        ),
    ]
    help_long = actions["mcp.add"].help_long
    assert "\nAdd an MCP server: a command to run, or a URL.\n" in help_long
    assert "the server's name" in help_long
    assert "the command's arguments, after --" in help_long
    assert "its mcpServers entry, as JSON" in actions["mcp.add-json"].help_long
    assert {"-t", "--transport", "-e", "--env", "-H", "--header"} <= actions[
        "mcp.add"
    ].flags


@pytest.mark.asyncio
@pytest.mark.parametrize("verb", _VERBS)
async def test_help_is_the_same_text_help_long_holds(tmp_path: Path, verb: str) -> None:
    action = _actions(tmp_path)[f"mcp.{verb}"]
    assert await action.apply("--help") == action.help_long


@pytest.mark.asyncio
async def test_add_http_writes_its_url_and_headers(tmp_path: Path) -> None:
    said = await _run(
        tmp_path,
        "add",
        "--transport http linear https://mcp.linear.app/mcp"
        " -H 'Authorization: Bearer x:y' --header X-Key:k",
    )
    assert said == (
        "Added MCP server 'linear' to local; sessions started from now use it."
    )
    assert _local(tmp_path) == {
        "linear": {
            "url": "https://mcp.linear.app/mcp",
            "headers": {"Authorization": "Bearer x:y", "X-Key": "k"},
        },
    }


@pytest.mark.asyncio
async def test_add_stdio_takes_its_command_after_the_separator(tmp_path: Path) -> None:
    await _run(tmp_path, "add", "-e API_KEY=xxx -e B=2 srv -- npx my-server --flag")
    assert _local(tmp_path) == {
        "srv": {
            "command": "npx",
            "args": ["my-server", "--flag"],
            "env": {"API_KEY": "xxx", "B": "2"},
        },
    }


@pytest.mark.asyncio
async def test_add_to_the_user_scope_writes_the_users_file(tmp_path: Path) -> None:
    said = await _run(tmp_path, "add", "-s user git uvx mcp-server-git")
    assert said.startswith("Added MCP server 'git' to user;")
    user = data_dir() / "rekursiv-ai" / "sagent" / "mcp.json"
    assert _entries(user) == {
        "git": {"command": "uvx", "args": ["mcp-server-git"]},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("args", "said"),
    [
        ("-t http web https://h -e A=1", "^-e/--env is for a stdio server$"),
        ("web uvx -H 'A: 1'", "^-H/--header is for an http server$"),
        ("-t http web https://h extra", "^an http server takes no arguments$"),
        ("web uvx -e NOEQUALS", "^-e takes KEY=value, not 'NOEQUALS'$"),
        ("-t http web https://h -H nocolon", "^-H takes 'Key: value', not 'nocolon'$"),
        ("-t sse web https://h", "invalid choice: 'sse'"),
        ("-s project web uvx", "invalid choice: 'project'"),
        ("a__b uvx", "MCP server name 'a__b'"),
    ],
)
async def test_add_refuses_what_it_cannot_write(
    tmp_path: Path,
    args: str,
    said: str,
) -> None:
    with pytest.raises(ValueError, match=said):
        await _run(tmp_path, "add", args)
    assert not (project_dir(tmp_path) / "mcp.json").exists()


@pytest.mark.asyncio
async def test_add_json_takes_an_entry_as_other_tools_write_it(tmp_path: Path) -> None:
    said = await _run(
        tmp_path,
        "add-json",
        """docs '{"serverUrl": "https://h/mcp", "loading": "on_demand"}' -s user""",
    )
    assert said.startswith("Added MCP server 'docs' to user;")
    user = data_dir() / "rekursiv-ai" / "sagent" / "mcp.json"
    assert _entries(user) == {
        "docs": {"serverUrl": "https://h/mcp", "loading": "on_demand"},
    }
    (server,) = servers(tmp_path)
    assert (server.name, server.loading) == ("docs", "on_demand")
    with pytest.raises(SettingError, match="is no JSON object"):
        await _run(tmp_path, "add-json", "x '[1]'")


@pytest.mark.asyncio
async def test_list_and_get_show_each_server_and_how_it_connected(
    tmp_path: Path,
) -> None:
    assert (await _run(tmp_path, "list")).startswith("No MCP servers. Add one with")
    await _run(tmp_path, "add", "good -- server-command --flag")
    await _run(
        tmp_path,
        "add-json",
        """off '{"url": "https://h/mcp", "disabled": true, "description": "docs"}'""",
    )
    good, off = servers(tmp_path)
    session = Mock(
        status=Mock(return_value={"good": "connected, 1 tool", "off": "disabled"}),
    )
    enter, exit_session = AsyncMock(return_value=session), AsyncMock()
    lifecycle = MagicMock(__aenter__=enter, __aexit__=exit_session)
    with patch.object(mcp_settings, "McpSession", return_value=lifecycle) as build:
        assert (await _run(tmp_path, "list")).splitlines() == [
            "good  stdio  every_request  connected, 1 tool",
            "off   http   every_request  disabled",
        ]
        assert (await _run(tmp_path, "get", "good")).splitlines() == [
            "good  stdio  every_request  connected, 1 tool",
            "  command: server-command --flag",
        ]
        assert (await _run(tmp_path, "get", "off")).splitlines() == [
            "off  http  every_request  disabled",
            "  url: https://h/mcp",
            "  docs",
        ]
    assert build.call_args_list == [call([good, off]), call([good]), call([off])]
    assert enter.await_count == exit_session.await_count == 3
    with pytest.raises(
        SettingError,
        match="No MCP server 'nope'; configured: good, off",
    ):
        await _run(tmp_path, "get", "nope")


@pytest.mark.asyncio
async def test_list_reads_the_config_flags_too(tmp_path: Path) -> None:
    await _run(tmp_path, "add", "git uvx")
    config = json_dumps(
        {"mcpServers": {"only": {"url": "https://h", "disabled": True}}},
    )
    session = Mock(status=Mock(return_value={"only": "disabled"}))
    lifecycle = MagicMock(
        __aenter__=AsyncMock(return_value=session),
        __aexit__=AsyncMock(),
    )
    with patch.object(mcp_settings, "McpSession", return_value=lifecycle) as build:
        said = await _run(
            tmp_path,
            "list",
            f"--strict-mcp-config --mcp-config '{config}'",
        )
    build.assert_called_once_with(servers(tmp_path, flags=[config], strict=True))
    assert said == "only  http  every_request  disabled"


@pytest.mark.asyncio
async def test_remove_and_logout_forget_what_they_name(tmp_path: Path) -> None:
    await _run(tmp_path, "add", "-t http web https://h/mcp")
    (web,) = servers(tmp_path)
    signed_in = McpTokenStorage(web.key).path
    signed_in.parent.mkdir(parents=True)
    signed_in.write_text("{}")
    assert await _run(tmp_path, "logout", "web") == "Signed out of MCP server 'web'."
    assert await _run(tmp_path, "logout", "web") == "MCP server 'web' had no sign-in."
    assert await _run(tmp_path, "remove", "web") == (
        "Removed MCP server 'web' from local; sessions started from now go without it."
    )
    with pytest.raises(ValueError, match="No MCP server 'web' in local or user"):
        await _run(tmp_path, "remove", "web")


@pytest.mark.asyncio
async def test_remove_takes_only_the_scope_named(tmp_path: Path) -> None:
    await _run(tmp_path, "add", "web uvx")
    await _run(tmp_path, "add", "-s user web uvx")
    said = await _run(tmp_path, "remove", "-s user web")
    assert said.startswith("Removed MCP server 'web' from user;")
    assert _local(tmp_path) == {"web": {"command": "uvx"}}
    assert _entries(data_dir() / "rekursiv-ai" / "sagent" / "mcp.json") == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(("args", "headless"), [("", None), ("--headless", True)])
async def test_login_signs_in_to_the_server_named(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    args: str,
    headless: bool | None,
) -> None:
    signed: list[tuple[str, bool]] = []
    shown: list[str] = []

    async def sign_in(
        server: McpServer,
        *,
        show: Callable[[str], object],
        headless: bool,
    ) -> None:
        show("visit")
        signed.append((server.name, headless))

    # At a desk: a display, and no ssh.
    monkeypatch.setenv("DISPLAY", ":0")
    for name in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(mcp_settings, "login", sign_in)
    await _run(tmp_path, "add", "-t http web https://h/mcp")
    said = await _actions(tmp_path, shown)["mcp.login"].apply(f"web {args}")
    assert said == "Signed in to MCP server 'web'."
    assert signed == [("web", headless is True)]
    assert shown == ["visit"]
    with pytest.raises(SettingError, match="No MCP server 'nope'"):
        await _run(tmp_path, "login", "nope")


@pytest.mark.asyncio
async def test_naming_a_server_where_there_are_none_says_so(tmp_path: Path) -> None:
    with pytest.raises(SettingError, match=r"^No MCP server 'x'; configured: none$"):
        await _run(tmp_path, "get", "x")


@pytest.mark.asyncio
async def test_login_says_why_a_local_server_cannot_sign_in(tmp_path: Path) -> None:
    await _run(tmp_path, "add", "git uvx")
    with pytest.raises(ValueError, match="MCP server 'git' runs locally"):
        await _run(tmp_path, "login", "git")


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
