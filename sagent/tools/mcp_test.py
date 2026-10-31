"""Tests for MCP servers' tools as sagent tools, against real in-process servers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import asyncio
import base64
import contextlib
import json
import logging
import re
import stat
import sys
import threading

from mcp import Client, StdioServerParameters
from mcp.client._memory import InMemoryTransport
from mcp.client.session_group import StreamableHttpParameters
from mcp.server.lowlevel import Server

import mcp.types as mcp_types
import pytest

from sagent.lib.userdirs import config_dir, data_dir, state_dir
from sagent.providers.tool_server import McpToolServer
from sagent.session.projects import project_dir
from sagent.session.types import Attachment
from sagent.tools import mcp
from sagent.tools.function import tool
from sagent.tools.mcp import (
    McpConnection,
    McpServer,
    McpServerTool,
    McpSession,
    McpTool,
    add_server,
    logout,
    remove_server,
    servers,
)
from sagent.tools.mcp_auth import McpTokenStorage
from sagent.tools.types import ToolContext


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from mcp.client._transport import TransportStreams
    from mcp.server.context import ServerRequestContext
    from treekle import PlainTree

    from sagent.tools.mcp import McpScope, McpToolLoading
    from sagent.tools.types import Tool


_PNG = b"\x89PNG\r\n\x1a\n"


def _write(path: Path, servers_: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mcpServers": servers_}))


def test_a_command_entry_runs_a_local_server(tmp_path: Path) -> None:
    _write(
        data_dir() / "rekursiv-ai" / "sagent" / "mcp.json",
        {
            "git": {
                "command": "uvx",
                "args": ["mcp-server-git"],
                "env": {"A": "1"},
                "cwd": "/w",
                "description": "The repo's history.",
            },
        },
    )
    (server,) = servers(tmp_path)
    assert server.name == "git"
    assert server.description == "The repo's history."
    assert server.parameters == StdioServerParameters(
        command="uvx",
        args=["mcp-server-git"],
        env={"A": "1"},
        cwd="/w",
    )


@pytest.mark.parametrize("key", ["url", "serverUrl"])
def test_a_url_entry_reaches_a_remote_server(tmp_path: Path, key: str) -> None:
    _write(
        data_dir() / "rekursiv-ai" / "sagent" / "mcp.json",
        {"docs": {key: "https://mcp.example/mcp", "headers": {"X-Key": "k"}}},
    )
    (server,) = servers(tmp_path)
    assert server.parameters == StreamableHttpParameters(
        url="https://mcp.example/mcp",
        headers={"X-Key": "k"},
    )


def test_variables_are_filled_from_the_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TOKEN", "t0")
    monkeypatch.delenv("UNSET", raising=False)
    _write(
        data_dir() / "rekursiv-ai" / "sagent" / "mcp.json",
        {
            "a": {
                "command": "${TOKEN}-run",
                "args": ["--x=${UNSET:-fallback}"],
                "env": {"T": "${TOKEN}"},
            },
            "b": {"url": "https://h/${TOKEN}", "headers": {"Auth": "Bearer ${TOKEN}"}},
        },
    )
    a, b = servers(tmp_path)
    assert a.parameters == StdioServerParameters(
        command="t0-run",
        args=["--x=fallback"],
        env={"T": "t0"},
    )
    assert b.parameters == StreamableHttpParameters(
        url="https://h/t0",
        headers={"Auth": "Bearer t0"},
    )


def test_an_unset_variable_with_no_default_names_itself(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("UNSET", raising=False)
    _write(
        data_dir() / "rekursiv-ai" / "sagent" / "mcp.json",
        {"a": {"command": "${UNSET}"}},
    )
    with pytest.raises(ValueError, match=r"MCP server 'a': \$UNSET is not set"):
        servers(tmp_path)


def test_the_project_file_replaces_the_user_entry_of_the_same_name(
    tmp_path: Path,
) -> None:
    _write(
        data_dir() / "rekursiv-ai" / "sagent" / "mcp.json",
        {"a": {"command": "user-a"}, "b": {"command": "user-b"}},
    )
    _write(project_dir(tmp_path) / "mcp.json", {"a": {"command": "project-a"}})
    found = {s.name: s.parameters for s in servers(tmp_path)}
    assert found == {
        "a": StdioServerParameters(command="project-a"),
        "b": StdioServerParameters(command="user-b"),
    }


def test_flags_win_and_strict_reads_only_them(tmp_path: Path) -> None:
    _write(
        data_dir() / "rekursiv-ai" / "sagent" / "mcp.json",
        {"a": {"command": "user-a"}, "b": {"command": "user-b"}},
    )
    flag_file = tmp_path / "extra.json"
    _write(flag_file, {"b": {"command": "file-b"}})
    inline = json.dumps({"mcpServers": {"c": {"command": "inline-c"}}})
    flags = [str(flag_file), inline]
    assert _commands(servers(tmp_path, flags=flags)) == {
        "a": "user-a",
        "b": "file-b",
        "c": "inline-c",
    }
    assert [s.name for s in servers(tmp_path, flags=flags, strict=True)] == ["b", "c"]


def test_no_file_is_no_servers(tmp_path: Path) -> None:
    assert servers(tmp_path) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope", "where"),
    [("user", "user"), ("local", "local")],
)
async def test_an_added_server_is_written_to_its_scopes_file_and_read_back(
    tmp_path: Path,
    scope: McpScope,
    where: str,
) -> None:
    await add_server(
        "linear",
        {"url": "https://mcp.linear.app/mcp", "loading": "on_demand"},
        cwd=tmp_path,
        scope=scope,
    )
    path = (
        data_dir() / "rekursiv-ai" / "sagent" / "mcp.json"
        if where == "user"
        else project_dir(tmp_path) / "mcp.json"
    )
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    (found,) = servers(tmp_path)
    assert (found.name, found.loading) == ("linear", "on_demand")


@pytest.mark.asyncio
async def test_adding_keeps_the_other_entries_and_refuses_a_name_taken(
    tmp_path: Path,
) -> None:
    await add_server(
        "git",
        {"command": "uvx", "args": ["mcp-server-git"]},
        cwd=tmp_path,
    )
    await add_server("docs", {"url": "https://h/mcp"}, cwd=tmp_path)
    assert [s.name for s in servers(tmp_path)] == ["git", "docs"]
    with pytest.raises(ValueError, match="MCP server 'git' is already in"):
        await add_server("git", {"command": "x"}, cwd=tmp_path)


def test_two_writers_at_once_keep_both_servers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each writer reads the file, then writes it; neither may write over the other."""
    both_read = threading.Barrier(2)
    read = mcp._file_config

    def slow_read(
        path: Path,
    ) -> tuple[dict[str, PlainTree], dict[str, dict[str, PlainTree]]]:
        config = read(path)
        # Holds the first reader until the second has read too, unless a lock
        # keeps the second waiting; then the first goes on alone.
        with contextlib.suppress(threading.BrokenBarrierError):
            both_read.wait(timeout=0.2)
        return config

    monkeypatch.setattr(mcp, "_file_config", slow_read)
    writers = [
        threading.Thread(
            target=asyncio.run,
            args=(add_server(name, {"command": "uvx"}, cwd=tmp_path),),
        )
        for name in ("git", "docs")
    ]
    for writer in writers:
        writer.start()
    for writer in writers:
        writer.join()
    assert sorted(s.name for s in servers(tmp_path)) == ["docs", "git"]


@pytest.mark.asyncio
async def test_an_entry_that_does_not_read_is_never_written(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="gives neither 'command' nor a URL"):
        await add_server("bad", {}, cwd=tmp_path)
    with pytest.raises(ValueError, match="MCP server name 'a__b'"):
        await add_server("a__b", {"command": "x"}, cwd=tmp_path)
    assert not (project_dir(tmp_path) / "mcp.json").exists()


@pytest.mark.asyncio
async def test_a_variable_is_kept_as_written_and_read_when_the_server_is(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SAGENT_TEST_TOKEN", raising=False)
    header = {"Authorization": "Bearer ${SAGENT_TEST_TOKEN}"}
    await add_server("gh", {"url": "https://h/mcp", "headers": header}, cwd=tmp_path)
    assert "${SAGENT_TEST_TOKEN}" in (project_dir(tmp_path) / "mcp.json").read_text()
    with pytest.raises(ValueError, match=r"\$SAGENT_TEST_TOKEN is not set"):
        servers(tmp_path)
    monkeypatch.setenv("SAGENT_TEST_TOKEN", "t")
    (server,) = servers(tmp_path)
    assert isinstance(server.parameters, StreamableHttpParameters)
    assert server.parameters.headers == {"Authorization": "Bearer t"}


@pytest.mark.asyncio
async def test_a_config_file_that_does_not_read_is_never_overwritten(
    tmp_path: Path,
) -> None:
    path = project_dir(tmp_path) / "mcp.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    with pytest.raises(ValueError, match=r"mcp\.json is no mcpServers JSON: "):
        await add_server("git", {"command": "uvx"}, cwd=tmp_path)
    assert path.read_text() == "{not json"


def test_a_server_is_known_by_its_name_and_where_it_is(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def key(name: str, entry: dict[str, object]) -> str:
        _write(project_dir(tmp_path) / "mcp.json", {name: entry})
        (server,) = servers(tmp_path)
        return server.key

    docs = key("docs", {"url": "https://a/mcp"})
    assert re.fullmatch(r"[0-9a-f]{16}", docs)
    assert key("docs", {"url": "https://a/mcp"}) == docs
    assert key("docs", {"url": "https://b/mcp"}) != docs
    assert key("wiki", {"url": "https://a/mcp"}) != docs
    assert key("docs", {"command": "uvx", "args": ["a"]}) != key(
        "docs",
        {"command": "uvx", "args": ["b"]},
    )
    monkeypatch.setenv("SAGENT_TEST_HOST", "a")
    written = key("docs", {"url": "https://${SAGENT_TEST_HOST}/mcp"})
    monkeypatch.setenv("SAGENT_TEST_HOST", "b")
    assert key("docs", {"url": "https://${SAGENT_TEST_HOST}/mcp"}) == written


@pytest.mark.asyncio
async def test_removing_keeps_a_sign_in_another_scope_still_uses(
    tmp_path: Path,
) -> None:
    await add_server("docs", {"url": "https://h/mcp"}, cwd=tmp_path, scope="user")
    await add_server("docs", {"url": "https://h/mcp"}, cwd=tmp_path)
    (server,) = servers(tmp_path)
    signed_in = McpTokenStorage(server.key).path
    signed_in.parent.mkdir(parents=True)
    signed_in.write_text("{}")
    assert await remove_server("docs", cwd=tmp_path) == "local"
    assert signed_in.exists()
    assert await remove_server("docs", cwd=tmp_path) == "user"
    assert not signed_in.exists()


@pytest.mark.asyncio
async def test_removing_finds_the_scope_unless_told_and_forgets_the_sign_in(
    tmp_path: Path,
) -> None:
    await add_server("web", {"url": "https://h/mcp"}, cwd=tmp_path, scope="user")
    await add_server("git", {"command": "uvx"}, cwd=tmp_path)
    (web,) = (s for s in servers(tmp_path) if s.name == "web")
    signed_in = McpTokenStorage(web.key).path
    signed_in.parent.mkdir(parents=True)
    signed_in.write_text("{}")
    assert await remove_server("web", cwd=tmp_path) == "user"
    assert not signed_in.exists()
    assert [s.name for s in servers(tmp_path)] == ["git"]
    with pytest.raises(ValueError, match=r"No MCP server 'git' in user"):
        await remove_server("git", cwd=tmp_path, scope="user")
    with pytest.raises(ValueError, match=r"No MCP server 'nope' in local or user"):
        await remove_server("nope", cwd=tmp_path)


@pytest.mark.asyncio
async def test_logging_out_forgets_only_the_sign_in(tmp_path: Path) -> None:
    await add_server("web", {"url": "https://h/mcp"}, cwd=tmp_path)
    (web,) = servers(tmp_path)
    signed_in = config_dir() / "rekursiv-ai" / "secrets" / "mcp" / f"{web.key}.json"
    signed_in.parent.mkdir(parents=True)
    signed_in.write_text("{}")
    assert logout(web) is True
    assert logout(web) is False
    assert [s.name for s in servers(tmp_path)] == ["web"]


@pytest.mark.parametrize("text", ["not json", '{"servers": {}}', '{"mcpServers": []}'])
def test_a_file_that_is_no_mcp_servers_json_names_itself(
    tmp_path: Path,
    text: str,
) -> None:
    path = tmp_path / "bad.json"
    path.write_text(text)
    with pytest.raises(ValueError, match=f"^{re.escape(str(path))} is no mcpServers"):
        servers(tmp_path, flags=[str(path)])


def test_filters_and_loading_are_read(tmp_path: Path) -> None:
    _write(
        data_dir() / "rekursiv-ai" / "sagent" / "mcp.json",
        {
            "gh": {
                "url": "https://gh/mcp",
                "enabled_tools": ["search_code", "get_file"],
                "disabled_tools": ["get_file"],
                "loading": "on_demand",
                "disabled": True,
            },
        },
    )
    (server,) = servers(tmp_path)
    assert server.enabled_tools == ("search_code", "get_file")
    assert server.disabled_tools == ("get_file",)
    assert server.loading == "on_demand"
    assert server.disabled is True


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"command": "x", "url": "https://h"}, "gives both 'command' and a URL"),
        ({}, "gives neither 'command' nor a URL"),
        ({"type": "sse", "url": "https://h"}, "transport 'sse' is not supported"),
        ({"command": "x", "loading": "lazy"}, ""),
    ],
)
def test_a_bad_entry_names_its_server(
    tmp_path: Path,
    entry: dict[str, object],
    message: str,
) -> None:
    _write(data_dir() / "rekursiv-ai" / "sagent" / "mcp.json", {"bad": entry})
    with pytest.raises(ValueError, match="MCP server 'bad'") as error:
        servers(tmp_path)
    assert message in str(error.value)


@pytest.mark.parametrize("name", ["has space", "dot.ted", "", "x" * 60, "a__b"])
def test_a_server_name_must_fit_in_a_tool_name(name: str) -> None:
    with pytest.raises(ValueError, match="MCP server name"):
        McpServer(name=name, parameters=StdioServerParameters(command="x"))


@pytest.mark.asyncio
async def test_an_every_request_server_offers_each_tool_by_its_own_name() -> None:
    async with _connected(_Backend()) as connection:
        tools = _by_name(connection.tools())
    assert list(tools) == _BACKEND_TOOLS
    echo = tools["mcp__srv__echo"]
    assert isinstance(echo, McpTool)
    assert echo.description == "Say the text back."
    assert echo.schema == _ECHO_SCHEMA
    assert echo.summary({"text": "hi"}) == "mcp__srv__echo"
    assert echo.serialize_key(_context(), {}) is None


@pytest.mark.asyncio
async def test_a_call_returns_text_images_and_errors(tmp_path: Path) -> None:
    context = ToolContext(cwd=tmp_path)
    async with _connected(_Backend()) as connection:
        tools = _by_name(connection.tools())
        echo = await tools["mcp__srv__echo"].call(context, {"text": "hi"})
        picture = await tools["mcp__srv__picture"].call(context, {})
        fail = await tools["mcp__srv__fail"].call(context, {})
    assert (echo.text(), echo.is_error, echo.call_id) == ("hi", None, "")
    assert picture.text() == "a picture"
    assert picture.attachments == (Attachment(mime_descriptor="image/png", data=_PNG),)
    assert (fail.text(), fail.is_error) == ("it broke", True)


@pytest.mark.asyncio
async def test_long_text_is_cut_to_what_one_result_holds(tmp_path: Path) -> None:
    context = ToolContext(cwd=tmp_path, max_result_chars=10)
    async with _connected(_Backend()) as connection:
        echo = _by_name(connection.tools())["mcp__srv__echo"]
        result = await echo.call(context, {"text": "x" * 30})
    assert result.text().startswith("x" * 10)
    assert "chars omitted" in result.text()


@pytest.mark.asyncio
async def test_enabled_then_disabled_tools_decide_what_is_offered() -> None:
    async with _connected(
        _Backend(),
        enabled_tools=("echo", "fail"),
        disabled_tools=("fail",),
    ) as connection:
        assert list(_by_name(connection.tools())) == ["mcp__srv__echo"]


@pytest.mark.asyncio
async def test_a_tool_whose_name_cannot_be_a_tool_name_is_skipped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    backend = _Backend(extra=("bad.name", "y" * 60))
    with caplog.at_level(logging.WARNING):
        async with _connected(backend) as connection:
            names = list(_by_name(connection.tools()))
    assert names == _BACKEND_TOOLS
    first, second = (r.getMessage() for r in caplog.records)
    assert first == (
        "MCP server 'srv': skipped tool 'bad.name'; 'mcp__srv__bad.name' is no"
        " tool name of at most 64 letters, digits, '_' or '-'."
    )
    assert f"'mcp__srv__{'y' * 60}'" in second


@pytest.mark.asyncio
async def test_an_on_demand_server_offers_one_tool_that_lists_its_tools() -> None:
    async with _connected(
        _Backend(),
        loading="on_demand",
        disabled_tools=("fail",),
    ) as connection:
        (tool,) = connection.tools()
    assert isinstance(tool, McpServerTool)
    assert tool.name == "mcp__srv"
    assert "- echo: Say the text back." in tool.description
    assert "- picture: Draw a picture." in tool.description
    assert "fail" not in tool.description
    assert _properties(tool) == {"tool", "arguments", "resource"}


@pytest.mark.asyncio
async def test_an_on_demand_tool_is_described_then_run(tmp_path: Path) -> None:
    context = ToolContext(cwd=tmp_path)
    async with _connected(_Backend(), loading="on_demand") as connection:
        (tool,) = connection.tools()
        described = await tool.call(context, {"tool": "echo"})
        ran = await tool.call(context, {"tool": "echo", "arguments": {"text": "hi"}})
        listed = await tool.call(context, {})
    assert json.loads(described.text()) == {
        "name": "echo",
        "description": "Say the text back.",
        "input_schema": _ECHO_SCHEMA,
    }
    assert ran.text() == "hi"
    assert listed.text() == tool.description
    assert tool.summary({"tool": "echo"}) == "mcp__srv echo"


@pytest.mark.asyncio
async def test_an_on_demand_server_refuses_a_tool_it_does_not_offer(
    tmp_path: Path,
) -> None:
    context = ToolContext(cwd=tmp_path)
    async with _connected(
        _Backend(),
        loading="on_demand",
        disabled_tools=("fail",),
    ) as connection:
        (tool,) = connection.tools()
        with pytest.raises(ValueError, match=r"No tool 'fail' on MCP server 'srv'"):
            await tool.call(context, {"tool": "fail", "arguments": {}})


@pytest.mark.asyncio
async def test_resources_are_read_through_the_server_tool(tmp_path: Path) -> None:
    context = ToolContext(cwd=tmp_path)
    async with _connected(_Backend(resources=True)) as connection:
        tools = _by_name(connection.tools())
        server_tool = tools["mcp__srv"]
        read = await server_tool.call(context, {"resource": "note://a"})
        with pytest.raises(ValueError, match="Unknown arguments: tool"):
            await server_tool.call(context, {"tool": "echo"})
    assert "mcp__srv__echo" in tools
    assert _properties(server_tool) == {"resource"}
    assert "- note://a: a note" in server_tool.description
    assert read.text() == "the note"


@pytest.mark.asyncio
async def test_a_tool_list_change_refreshes_the_on_demand_listing() -> None:
    backend = _Backend()
    async with _connected(backend, loading="on_demand") as connection:
        (tool,) = connection.tools()
        backend.extra = ("added",)
        await connection.on_message(object())
        ignored = connection.relisting
        await connection.on_message(mcp_types.ToolListChangedNotification())
        started = connection.relisting
        assert ignored is None
        assert started is not None
        await started
        assert "- added: Echo." in tool.description


@pytest.mark.asyncio
async def test_list_changes_that_overlap_relist_one_at_a_time() -> None:
    backend = _Backend()
    async with _connected(backend, loading="on_demand") as connection:
        (tool,) = connection.tools()
        await connection.on_message(mcp_types.ToolListChangedNotification())
        first = connection.relisting
        backend.extra = ("added",)
        await connection.on_message(mcp_types.ToolListChangedNotification())
        assert connection.relisting is first
        assert first is not None
        await first
        assert "- added: Echo." in tool.description
    # Both changes landed before the one relisting read the list, so it read both.
    assert backend.listed == 2


@pytest.mark.asyncio
async def test_a_list_change_during_a_relisting_lists_once_more() -> None:
    backend = _Backend()
    async with _connected(backend, loading="on_demand") as connection:
        (tool,) = connection.tools()

        async def changed_while_listing() -> None:
            backend.during_listing = None
            backend.extra = ("later",)
            await connection.on_message(mcp_types.ToolListChangedNotification())

        backend.during_listing = changed_while_listing
        await connection.on_message(mcp_types.ToolListChangedNotification())
        assert connection.relisting is not None
        await connection.relisting
        assert "- later: Echo." in tool.description
    assert backend.listed == 3


@pytest.mark.asyncio
async def test_a_failed_relisting_keeps_the_last_list(
    caplog: pytest.LogCaptureFixture,
) -> None:
    backend = _Backend()
    async with _connected(backend, loading="on_demand") as connection:
        (tool,) = connection.tools()
        backend.listing_fails = True
        await connection.on_message(mcp_types.ToolListChangedNotification())
        assert connection.relisting is not None
        with caplog.at_level(logging.WARNING):
            await connection.relisting
        assert "- echo: Say the text back." in tool.description
    assert "MCP server 'srv': relisting failed" in caplog.text


def test_a_connection_not_yet_attached_says_so() -> None:
    connection = McpConnection(
        McpServer(name="srv", parameters=StdioServerParameters(command="x")),
    )
    with pytest.raises(RuntimeError, match="MCP server 'srv' is not connected"):
        connection._open()


@pytest.mark.asyncio
async def test_every_kind_of_content_reaches_the_model(tmp_path: Path) -> None:
    context = ToolContext(cwd=tmp_path)
    async with _connected(_Backend()) as connection:
        tools = _by_name(connection.tools())
        mixed = await tools["mcp__srv__mixed"].call(context, {})
        data = await tools["mcp__srv__data"].call(context, {})
    assert mixed.text() == "embedded\nResource: file:///linked"
    assert mixed.attachments == (
        Attachment(mime_descriptor="audio/wav", data=b"RIFF"),
        Attachment(mime_descriptor="application/octet-stream", data=b"\x00\x01"),
    )
    assert json.loads(data.text()) == {"n": 1}


@pytest.mark.asyncio
async def test_an_on_demand_server_reads_resources_and_says_what_it_is(
    tmp_path: Path,
) -> None:
    context = ToolContext(cwd=tmp_path)
    async with _connected(
        _Backend(resources=True),
        loading="on_demand",
        description="Notes and echoes.",
    ) as connection:
        (tool,) = connection.tools()
        read = await tool.call(context, {"resource": "note://a"})
    assert read.text() == "the note"
    assert tool.description.splitlines()[:2] == [
        "The tools and resources of the MCP server `srv`.",
        "Notes and echoes.",
    ]
    assert tool.serialize_key(context, {}) is None


@pytest.mark.asyncio
async def test_a_session_connects_its_servers_and_skips_one_that_fails(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    script = tmp_path / "server.py"
    script.write_text(_STDIO_SERVER)
    good = McpServer(
        name="good",
        parameters=StdioServerParameters(command=sys.executable, args=[str(script)]),
    )
    missing = McpServer(
        name="missing",
        parameters=StdioServerParameters(command=str(tmp_path / "absent")),
    )
    off = McpServer(name="off", parameters=good.parameters, disabled=True)
    with caplog.at_level(logging.WARNING):
        async with McpSession([good, missing, off]) as session:
            tools = _by_name(session.tools())
            result = await tools["mcp__good__shout"].call(_context(), {"text": "hi"})
    assert list(tools) == ["mcp__good__shout"]
    assert result.text() == "HI"
    status = session.status()
    assert list(status) == ["good", "missing", "off"]
    assert status["good"] == "connected, 1 tool"
    assert status["missing"].startswith("failed: ")
    assert status["off"] == "disabled"
    assert "MCP server 'missing'" in caplog.text
    assert "off" not in caplog.text
    log = state_dir() / "rekursiv-ai" / "sagent" / "mcp" / f"good-{good.key}.log"
    assert "server started" in log.read_text()
    assert stat.S_IMODE(log.stat().st_mode) == 0o600
    log.chmod(0o664)
    # A second session, as another process would run, adds to the log.
    async with McpSession([good]):
        pass
    assert log.read_text().count("server started") == 2
    assert stat.S_IMODE(log.stat().st_mode) == 0o600
    # Another project's ``good`` runs another command, so writes another log.
    other = McpServer(
        name="good",
        parameters=StdioServerParameters(
            command=sys.executable,
            args=[str(script), "--other"],
        ),
    )
    async with McpSession([other]):
        pass
    assert log.read_text().count("server started") == 2


@pytest.mark.asyncio
async def test_a_remote_server_is_reached_over_streamable_http() -> None:
    async with McpToolServer() as served:
        mount = await served.mount()
        mount.offer([tool(_greet, name="Greet")])
        remote = McpServer(
            name="web",
            parameters=StreamableHttpParameters(url=mount.url, headers={"X-Key": "k"}),
        )
        async with McpSession([remote]) as session:
            (greet,) = session.tools()
    assert greet.name == "mcp__web__Greet"
    assert greet.description.startswith("Say hello.")


@pytest.mark.asyncio
async def test_a_connection_that_fails_as_it_closes_is_logged(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(mcp, "_transport", _breaks_on_close)
    server = McpServer(name="srv", parameters=StdioServerParameters(command="x"))
    with caplog.at_level(logging.WARNING):
        async with McpSession([server]) as session:
            assert [t.name for t in session.tools()] == _BACKEND_TOOLS
    assert "MCP server 'srv' disconnected: pipe closed" in caplog.text


@pytest.mark.asyncio
async def test_a_server_slower_than_the_timeout_is_skipped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    slow = McpServer(
        name="slow",
        # Reads requests until its stdin closes, and answers none.
        parameters=StdioServerParameters(
            command="/bin/sh",
            args=["-c", "cat >/dev/null"],
        ),
    )
    with caplog.at_level(logging.WARNING):
        async with McpSession([slow], connect_timeout_sec=0.2) as session:
            assert session.tools() == []
            assert session.status() == {"slow": "failed: no answer within 0.2s"}
    assert "MCP server 'slow' failed to connect: no answer within 0.2s" in caplog.text


_BACKEND_TOOLS = [
    "mcp__srv__echo",
    "mcp__srv__picture",
    "mcp__srv__fail",
    "mcp__srv__mixed",
    "mcp__srv__data",
]


_ECHO_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
}


_STDIO_SERVER = """\
import sys

from mcp.server.mcpserver import MCPServer

app = MCPServer("good")


@app.tool()
def shout(text: str) -> str:
    \"\"\"Say the text loudly.\"\"\"
    return text.upper()


print("server started", file=sys.stderr, flush=True)
app.run()
"""


@dataclass(slots=True, kw_only=True)
class _Backend:
    """An MCP server's tools and resources, served in-process."""

    extra: Sequence[str] = ()
    """More tools, each echoing its text."""

    resources: bool = False
    listing_fails: bool = False
    listed: int = 0
    """How many times the tools were listed."""

    during_listing: Callable[[], Awaitable[None]] | None = None
    """Run as the tools are listed, before the list is read."""

    def server(self) -> Server[object]:
        return Server(
            "srv",
            on_list_tools=self.list_tools,
            on_call_tool=self.call_tool,
            on_list_resources=self.list_resources if self.resources else None,
            on_read_resource=self.read_resource if self.resources else None,
        )

    async def list_tools(
        self,
        ctx: ServerRequestContext[object],
        params: mcp_types.PaginatedRequestParams | None,
    ) -> mcp_types.ListToolsResult:
        del ctx, params
        self.listed += 1
        if self.during_listing is not None:
            await self.during_listing()
        if self.listing_fails:
            raise RuntimeError("listing broke")
        described = {
            "echo": "Say the text back.",
            "picture": "Draw a picture.",
            "fail": "Always fail.",
            "mixed": "Return every kind of content.",
            "data": "Return structured content only.",
            **dict.fromkeys(self.extra, "Echo."),
        }
        return mcp_types.ListToolsResult(
            tools=[
                mcp_types.Tool(name=name, description=text, input_schema=_ECHO_SCHEMA)
                for name, text in described.items()
            ],
        )

    async def call_tool(
        self,
        ctx: ServerRequestContext[object],
        params: mcp_types.CallToolRequestParams,
    ) -> mcp_types.CallToolResult:
        del ctx
        if params.name == "picture":
            return mcp_types.CallToolResult(
                content=[
                    mcp_types.TextContent(text="a picture"),
                    mcp_types.ImageContent(
                        data=base64.b64encode(_PNG).decode(),
                        mime_type="image/png",
                    ),
                ],
            )
        if params.name == "fail":
            return mcp_types.CallToolResult(
                content=[mcp_types.TextContent(text="it broke")],
                is_error=True,
            )
        if params.name == "data":
            return mcp_types.CallToolResult(content=[], structured_content={"n": 1})
        if params.name == "mixed":
            return mcp_types.CallToolResult(
                content=[
                    mcp_types.EmbeddedResource(
                        type="resource",
                        resource=mcp_types.TextResourceContents(
                            uri="file:///a",
                            text="embedded",
                        ),
                    ),
                    mcp_types.AudioContent(
                        data=base64.b64encode(b"RIFF").decode(),
                        mime_type="audio/wav",
                    ),
                    mcp_types.EmbeddedResource(
                        type="resource",
                        resource=mcp_types.BlobResourceContents(
                            uri="file:///b",
                            blob=base64.b64encode(b"\x00\x01").decode(),
                        ),
                    ),
                    mcp_types.ResourceLink(
                        type="resource_link",
                        name="linked",
                        uri="file:///linked",
                    ),
                ],
            )
        # The SDK types a call's arguments as ``dict[str, Any]``.
        arguments = cast(dict[str, object], params.arguments or {})
        text = str(arguments.get("text", ""))
        return mcp_types.CallToolResult(content=[mcp_types.TextContent(text=text)])

    async def list_resources(
        self,
        ctx: ServerRequestContext[object],
        params: mcp_types.PaginatedRequestParams | None,
    ) -> mcp_types.ListResourcesResult:
        del ctx, params
        return mcp_types.ListResourcesResult(
            resources=[mcp_types.Resource(name="a note", uri="note://a")],
        )

    async def read_resource(
        self,
        ctx: ServerRequestContext[object],
        params: mcp_types.ReadResourceRequestParams,
    ) -> mcp_types.ReadResourceResult:
        del ctx
        return mcp_types.ReadResourceResult(
            contents=[mcp_types.TextResourceContents(uri=params.uri, text="the note")],
        )


@asynccontextmanager
async def _connected(
    backend: _Backend,
    *,
    loading: McpToolLoading = "every_request",
    enabled_tools: tuple[str, ...] | None = None,
    disabled_tools: tuple[str, ...] = (),
    description: str = "",
) -> AsyncGenerator[McpConnection]:
    """Yield a connection to ``backend``, configured as ``srv``."""
    connection = McpConnection(
        McpServer(
            name="srv",
            parameters=StdioServerParameters(command="unused"),
            description=description,
            loading=loading,
            enabled_tools=enabled_tools,
            disabled_tools=disabled_tools,
        ),
    )
    async with Client(
        backend.server(),
        message_handler=connection.on_message,
    ) as client:
        await connection.attach(client)
        yield connection


def _by_name(tools: Sequence[Tool]) -> dict[str, Tool]:
    return {tool.name: tool for tool in tools}


def _properties(tool: Tool) -> set[str]:
    """Return the names of the arguments ``tool`` takes."""
    properties = tool.schema["properties"]
    assert isinstance(properties, Mapping)
    return set(properties)


def _commands(found: Sequence[McpServer]) -> dict[str, str]:
    """Return each local server's command, by name."""
    out: dict[str, str] = {}
    for server in found:
        assert isinstance(server.parameters, StdioServerParameters)
        out[server.name] = server.parameters.command
    return out


# It fails from inside a task group, as the SDK's own transports do.
@asynccontextmanager
async def _breaks_on_close(server: McpServer) -> AsyncGenerator[TransportStreams]:
    """Stand in for a transport to ``server``: the backend's, failing as it closes."""
    del server
    async with InMemoryTransport(_Backend().server()) as streams:
        yield streams
    raise ExceptionGroup("unhandled errors in a TaskGroup", [OSError("pipe closed")])


def _greet(name: str) -> str:
    """Say hello."""
    return f"hello {name}"


def _context() -> ToolContext:
    return ToolContext(cwd=Path.cwd())


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
