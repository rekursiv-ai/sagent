"""Tests for Claude through the ``claude`` CLI, against a stub ``claude`` on PATH."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, override

import asyncio
import io
import os
import re
import stat
import tempfile

from PIL import Image
from treekle import PlainTree, from_plain, json_dumps, json_loads

import httpx2
import pytest
import pytest_asyncio

from sagent.model.exceptions import (
    ContextOverflowError,
    DisconnectedError,
    RateLimitedError,
)
from sagent.model.types import Request, Response, TextChunk
from sagent.providers.anthropic import login, wire
from sagent.providers.anthropic.api import models
from sagent.providers.anthropic.cli import AnthropicCLI, AnthropicCLIModel
from sagent.providers.resources import SharedResources
from sagent.providers.types import Login
from sagent.session.types import (
    AgentToAgentMessage,
    AnyToolResult,
    AssistantMessage,
    Attachment,
    ContextClear,
    SystemMessage,
    Thinking,
    ToolCall,
    TurnContext,
    UncategorizedToolResult,
    UserMessage,
)
from sagent.tools.display import ToolDisplay
from sagent.tools.types import Tool
from sagent.types import AsyncResource


if TYPE_CHECKING:
    from asyncio.subprocess import Process
    from collections.abc import AsyncIterator, Callable, Sequence

    from sagent.providers.types import ProcessPool, ToolMount
    from sagent.session.types import SessionRecord
    from sagent.tools.types import ToolContext


# Answers each line it reads with the next scripted turn, numbered across every
# process, and records what it was started with once it is spoken to; a spare
# never spoken to records nothing.
_CLAUDE = """#!/bin/sh
dir=$(dirname "$0")
while IFS= read -r line; do
  if [ ! -e "$dir/argv.$$" ]; then
    printf '%s\\0' "$@" > "$dir/argv.$$"
    env -0 > "$dir/env.$$"
  fi
  printf '%s %s\\n' "$$" "$line" >> "$dir/said"
  n=$(cat "$dir/count" 2>/dev/null || echo 0)
  echo $((n + 1)) > "$dir/count"
  [ -e "$dir/turn.$n" ] || exit 3
  cat "$dir/turn.$n"
done
"""


@pytest.fixture(autouse=True)
def temp_in_tmp_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the files a model hands claude under this test's own directory."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))


@pytest.fixture
def claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Stub:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "claude"
    script.write_text(_CLAUDE)
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return _Stub(dir=bin_dir)


@pytest_asyncio.fixture
async def shared() -> AsyncIterator[_Resources]:
    async with _Resources() as resources:
        yield resources


def _model(shared: _Resources, *, name: str = "opus") -> AnthropicCLIModel:
    return AnthropicCLI(resources=shared)[name]


async def _stream(
    model: AnthropicCLIModel,
    request: Request,
) -> list[TextChunk | Response]:
    return [event async for event in model.stream(request)]


def _response(events: Sequence[TextChunk | Response]) -> Response:
    response = events[-1]
    assert isinstance(response, Response)
    return response


def _text(*lines: Sequence[Mapping[str, object]]) -> list[str]:
    """Return the text block of each user line."""
    texts: list[str] = []
    for content in lines:
        (block,) = content
        texts.append(from_plain(block["text"], str))
    return texts


# ---- the process environment -------------------------------------------------


@pytest.mark.asyncio
async def test_claude_starts_without_other_accounts_and_without_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-other")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.example")
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock")
    monkeypatch.setenv("SAGENT_TEST_KEPT", "kept")
    for name in ("DISABLE_TELEMETRY", "DISABLE_ERROR_REPORTING", "DISABLE_AUTOUPDATER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("DISABLE_AUTO_COMPACT", raising=False)
    env = await AnthropicCLI().environment()
    assert not [key for key in env if key.startswith("ANTHROPIC_")]
    assert "CLAUDE_CODE_USE_BEDROCK" not in env
    assert "AWS_BEARER_TOKEN_BEDROCK" not in env
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in env
    assert env["SAGENT_TEST_KEPT"] == "kept"
    assert {
        key: env[key]
        for key in (
            "DISABLE_TELEMETRY",
            "DISABLE_ERROR_REPORTING",
            "DISABLE_AUTOUPDATER",
            "DISABLE_AUTO_COMPACT",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
            "CLAUDE_CODE_SKIP_PROMPT_HISTORY",
        )
    } == dict.fromkeys(
        (
            "DISABLE_TELEMETRY",
            "DISABLE_ERROR_REPORTING",
            "DISABLE_AUTOUPDATER",
            "DISABLE_AUTO_COMPACT",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
            "CLAUDE_CODE_SKIP_PROMPT_HISTORY",
        ),
        "1",
    )
    # A held call waits on the caller's tool, which may run for a day.
    assert env["MCP_TOOL_TIMEOUT"] == "86400000"


@pytest.mark.asyncio
async def test_a_named_login_reaches_claude_as_a_token(tmp_path: Path) -> None:
    oauth = {"accessToken": "at-work", "expiresAt": 10**15}
    credentials = tmp_path / ".credentials-work.json"
    credentials.write_text(json_dumps({"claudeAiOauth": oauth}))
    env = await AnthropicCLI(credentials=credentials).environment()
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == oauth["accessToken"]


# ---- a turn ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_turn_starts_claude_with_our_tools_only(
    claude: _Stub,
    shared: _Resources,
) -> None:
    claude.script(_said("Hello"))
    model = _model(shared)
    request = Request(
        system=("Be terse.", "Be kind."),
        records=(UserMessage(content="hi"),),
        tools=(_Echo(),),
    )
    events = await _stream(model, request)
    assert events[0] == TextChunk(text="Hello")
    response = _response(events)
    assert response.records == (AssistantMessage(content="Hello"),)
    assert response.stop_reason == "end_turn"
    ((argv, said),) = claude.processes()
    flags = {
        flag: argv[i + 1] for i, flag in enumerate(argv[:-1]) if flag.startswith("--")
    }
    system = Path(flags["--system-prompt-file"])
    mcp = Path(flags["--mcp-config"])
    assert argv == [
        "--print",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--include-partial-messages",
        "--verbose",
        "--model",
        "claude-opus-5-5",
        "--system-prompt-file",
        str(system),
        "--no-session-persistence",
        "--setting-sources",
        "",
        "--strict-mcp-config",
        "--mcp-config",
        str(mcp),
        # Its own tools off; ours arrive over MCP.
        "--tools",
        "",
        "--disable-slash-commands",
        "--permission-mode",
        "bypassPermissions",
    ]
    # Any local user can read a process's arguments: the mount's URL is the
    # only key to our tools, and a system prompt may be long or private.
    assert [arg for arg in argv if "mount-token" in arg or "Be kind." in arg] == []
    assert _read(system) == "Be terse.\n\nBe kind."
    assert json_loads(_read(mcp), dict[str, object]) == {
        "mcpServers": {"sagent": {"type": "http", "url": shared.server.mounted.url}},
    }
    assert re.fullmatch(r"system-[0-9a-f]{16}", system.name)
    assert re.fullmatch(r"mcp-[0-9a-f]{16}", mcp.name)
    assert system.parent == mcp.parent
    assert system.parent.name.startswith("sagent-claude-")
    assert _mode(system.parent) == 0o700
    assert _text(*said) == ["User: hi"]
    assert shared.server.mounted.offered == [(_Echo(),)]
    await model.aclose()
    assert not system.parent.exists()
    assert set(shared.pools) == {"claude"}  # Started and stopped in one pool.


@pytest.mark.asyncio
async def test_a_new_user_line_reaches_the_running_claude(
    claude: _Stub,
    shared: _Resources,
) -> None:
    claude.script(_said("Hello"), _said("Again"))
    model = _model(shared)
    first = Request(records=(UserMessage(content="hi"),))
    response = _response(await _stream(model, first))
    second = Request(
        records=(
            *first.records,
            *response.records,
            UserMessage(content="more"),
            AgentToAgentMessage(sender="lead", content="ping"),
            AgentToAgentMessage(content="unsigned"),
            AgentToAgentMessage(),
            UserMessage(),
            TurnContext(),
        ),
    )
    events = await _stream(model, second)
    assert _response(events).records == (AssistantMessage(content="Again"),)
    ((_, said),) = claude.processes()
    assert _text(*said) == ["User: hi", "more\n\n[from lead]: ping\n\nunsigned"]


@pytest.mark.asyncio
async def test_a_tool_call_is_held_until_its_result_is_answered(
    claude: _Stub,
    shared: _Resources,
) -> None:
    claude.script((*_calls("toolu_1", "Echo", {"text": "x"}), *_said("Done")))
    model = _model(shared)
    first = Request(records=(UserMessage(content="echo x"),), tools=(_Echo(),))
    response = _response(await _stream(model, first))
    call = ToolCall(call_id="toolu_1", name="Echo", arguments={"text": "x"})
    # The model calls our tool under the MCP server's name.
    assert response.records == (call,)
    assert response.stop_reason == "tool_use"
    result = UncategorizedToolResult(call_id="toolu_1", content="x")
    second = Request(records=(*first.records, call, result), tools=(_Echo(),))
    events = await _stream(model, second)
    assert _response(events).records == (AssistantMessage(content="Done"),)
    assert shared.server.mounted.answered == [(call, result)]
    # The answer continues the turn; claude reads no new line for it.
    ((_, said),) = claude.processes()
    assert _text(*said) == ["User: echo x"]


@pytest.mark.asyncio
async def test_notes_that_came_with_tool_results_reach_the_next_line_once(
    claude: _Stub,
    shared: _Resources,
) -> None:
    claude.script(
        (
            *_calls("toolu_1", "Echo", {}),
            *_calls("toolu_2", "Echo", {}),
            *_said("Done"),
        ),
        _said("Noted"),
        _said("Again"),
    )
    model = _model(shared)
    records: tuple[SessionRecord, ...] = (UserMessage(content="go"),)
    records += _response(await _stream(model, Request(records=records))).records
    for call_id, note in (("toolu_1", "One changed."), ("toolu_2", "Two changed.")):
        records += (
            UncategorizedToolResult(call_id=call_id, content="ok"),
            SystemMessage(content=note),
        )
        records += _response(await _stream(model, Request(records=records))).records
    for line in ("next", "again"):
        records += (UserMessage(content=line),)
        records += _response(await _stream(model, Request(records=records))).records
    ((_, said),) = claude.processes()
    assert _text(*said) == [
        "User: go",
        (
            "<system-reminder>\nOne changed.\n</system-reminder>\n\n"
            "<system-reminder>\nTwo changed.\n</system-reminder>\n\nnext"
        ),
        "again",
    ]


@pytest.mark.asyncio
async def test_a_tool_results_images_are_shrunk_before_answering(
    claude: _Stub,
    shared: _Resources,
) -> None:
    claude.script((*_calls("toolu_1", "Echo", {}), *_said("Seen")))
    model = _model(shared)
    first = Request(records=(UserMessage(content="look"),))
    (call,) = _response(await _stream(model, first)).records
    wide = Attachment(mime_descriptor="image/png", data=_png(2_600, 1))
    pdf = Attachment(mime_descriptor="application/pdf", data=b"%PDF-1.4")
    result = UncategorizedToolResult(call_id="toolu_1", attachments=(wide, pdf))
    await _stream(model, Request(records=(*first.records, call, result)))
    ((_, answered),) = shared.server.mounted.answered
    shrunk, kept = answered.attachments
    assert kept is pdf  # Only images are shrunk.
    assert shrunk.mime_descriptor == "image/webp"
    assert Image.open(io.BytesIO(shrunk.data)).size == (2_576, 1)


# ---- what starts a fresh claude ----------------------------------------------


@pytest.mark.parametrize(
    "change",
    ["system", "tools", "schema", "environment", "history"],
)
@pytest.mark.asyncio
async def test_a_change_the_process_cannot_hear_starts_a_fresh_claude(
    claude: _Stub,
    shared: _Resources,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    claude.script(_said("Hello"), _said("Again"))
    model = _model(shared)
    tools = (_Echo(),)
    first = Request(system=("S",), records=(UserMessage(content="hi"),), tools=tools)
    response = _response(await _stream(model, first))
    history: tuple[SessionRecord, ...] = (
        (UserMessage(content="hey"), *response.records)
        if change == "history"
        else (*first.records, *response.records)
    )
    if change == "tools":
        tools = ()
    if change == "schema":
        tools = (_Echo(schema=MappingProxyType({"type": "object", "required": []})),)
    if change == "environment":
        monkeypatch.setenv("SAGENT_TEST_ACCOUNT", "other")
    second = Request(
        system=("T",) if change == "system" else first.system,
        tools=tools,
        records=(*history, UserMessage(content="more")),
    )
    await _stream(model, second)
    started = claude.processes()
    assert len(started) == 2
    greeting = "hey" if change == "history" else "hi"
    assert _text(*started[1][1]) == [
        f"User: {greeting}\n\nAssistant: Hello\n\nUser: more",
    ]


@pytest.mark.asyncio
async def test_results_for_only_some_held_calls_start_a_fresh_claude(
    claude: _Stub,
    shared: _Resources,
) -> None:
    """Claude waits on every call it made; a turn answering some cannot go on."""
    two_calls = _streamed(
        {"type": "message_start", "message": {"usage": {}}},
        *(
            event
            for index, call_id in enumerate(("toolu_1", "toolu_2"))
            for event in (
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": call_id,
                        "name": "mcp__sagent__Echo",
                    },
                },
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "input_json_delta", "partial_json": "{}"},
                },
            )
        ),
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {}},
    )
    claude.script(two_calls, _said("Recovered"))
    model = _model(shared)
    first = Request(records=(UserMessage(content="go"),))
    calls = _response(await _stream(model, first)).records
    assert [c.call_id for c in calls if isinstance(c, ToolCall)] == [
        "toolu_1",
        "toolu_2",
    ]
    result = UncategorizedToolResult(call_id="toolu_1", content="one")
    await _stream(model, Request(records=(*first.records, *calls, result)))
    assert shared.server.mounted.answered == []
    assert len(claude.processes()) == 2


@pytest.mark.asyncio
async def test_a_stream_event_with_no_event_is_skipped(
    claude: _Stub,
    shared: _Resources,
) -> None:
    claude.script([{"type": "stream_event"}, {"type": "system"}, *_said("Hello")])
    events = await _stream(
        _model(shared),
        Request(records=(UserMessage(content="hi"),)),
    )
    assert _response(events).records == (AssistantMessage(content="Hello"),)


@pytest.mark.asyncio
async def test_a_call_that_never_reached_the_mount_starts_a_fresh_claude(
    claude: _Stub,
    shared: _Resources,
) -> None:
    claude.script(_calls("toolu_1", "Echo", {"text": "x"}), _said("Recovered"))
    shared.server.mounted.unanswerable = True
    model = _model(shared)
    first = Request(records=(UserMessage(content="echo x"),))
    (call,) = _response(await _stream(model, first)).records
    result = UncategorizedToolResult(call_id="toolu_1", content="x")
    events = await _stream(model, Request(records=(*first.records, call, result)))
    assert _response(events).records == (AssistantMessage(content="Recovered"),)
    started = claude.processes()
    assert len(started) == 2
    assert _text(*started[1][1]) == [
        (
            "User: echo x\n\n"
            'Assistant called Echo as toolu_1: {"text": "x"}\n\n'
            "toolu_1 returned: x"
        ),
    ]


@pytest.mark.asyncio
async def test_a_fresh_claude_hears_the_conversation_as_a_transcript(
    claude: _Stub,
    shared: _Resources,
) -> None:
    claude.script(_said("Hello"))
    records = (
        ContextClear(
            summary="We fixed the bug.",
            history=(UserMessage(content="a"),),
        ),
        ContextClear(history=(AssistantMessage(content="b"),)),
        AgentToAgentMessage(sender="lead", content="status?"),
        AgentToAgentMessage(content="unsigned"),
        SystemMessage(content="Saved."),
        SystemMessage(content=None),
        TurnContext(cwd="/work"),
        Thinking(content="hidden", encrypted="sig"),
        UserMessage(content="", attachments=(_png_attachment(),)),
        UserMessage(content="q"),
    )
    await _stream(_model(shared), Request(records=records))
    ((_, said),) = claude.processes()
    (content,) = said
    assert [block["type"] for block in content] == ["text", "image"]
    assert content[0]["text"] == "\n\n".join(
        [
            "Summary of the conversation so far:\nWe fixed the bug.",
            "User: a",
            "Assistant: b",
            "Agent lead: status?",
            "Agent: unsigned",
            "<system-reminder>\nSaved.\n</system-reminder>",
            "<system-reminder>\nWorking directory: /work\n</system-reminder>",
            "User: q",
        ],
    )


# ---- failures ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_claude_exiting_mid_turn_is_a_disconnect(
    claude: _Stub,
    shared: _Resources,
) -> None:
    claude.script()  # The stub exits on the first line, as a crash would.
    model = _model(shared)
    with pytest.raises(DisconnectedError, match=r"^claude exited before its answer$"):
        await _stream(model, Request(records=(UserMessage(content="hi"),)))


@pytest.mark.parametrize(
    ("result", "error", "message"),
    [
        (
            {"result": "Too long.", "terminal_reason": "blocking_limit"},
            ContextOverflowError,
            "Too long.",
        ),
        (
            {"result": "Cut off.", "terminal_reason": "aborted_streaming"},
            DisconnectedError,
            "Cut off.",
        ),
        (
            {"result": "Slow down.", "api_error_status": 429},
            RateLimitedError,
            "Slow down.",
        ),
        ({"result": "It failed."}, RuntimeError, "It failed."),
        ({"errors": ["boom"]}, RuntimeError, "['boom']"),
        ({}, RuntimeError, "{'type': 'result', 'is_error': True}"),
    ],
)
@pytest.mark.asyncio
async def test_a_failed_turn_raises_what_it_means(
    claude: _Stub,
    shared: _Resources,
    result: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    claude.script([{"type": "result", "is_error": True, **result}])
    with pytest.raises(error) as raised:
        await _stream(_model(shared), Request(records=(UserMessage(content="hi"),)))
    assert type(raised.value) is error
    assert str(raised.value) == message


def _rate_limit_event(status: str, *, resets_at: int, kind: str) -> dict[str, object]:
    """Return a ``rate_limit_event`` as ``claude`` 2.1.309 writes it."""
    return {
        "type": "rate_limit_event",
        "rate_limit_info": {
            "status": status,
            "resetsAt": resets_at,
            "rateLimitType": kind,
            "overageStatus": "rejected",
            "isUsingOverage": False,
        },
    }


_SPENT = {
    "type": "result",
    "is_error": True,
    "api_error_status": 429,
    "api_error": "usage_limit_reached",
    "terminal_reason": "api_error",
    "result": "You've hit your weekly limit · resets Oct 31, 6am (America/Los_Angeles)",
}


@pytest.mark.asyncio
async def test_a_spent_window_waits_for_its_reset(
    claude: _Stub,
    shared: _Resources,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI says when the window resets; it was dropped, so the wait was a guess."""
    monkeypatch.setattr(wire.time, "time", lambda: 1_793_285_478.0)
    claude.script(
        [
            _rate_limit_event("rejected", resets_at=1_793_451_600, kind="seven_day"),
            _SPENT,
        ],
    )
    with pytest.raises(RateLimitedError) as raised:
        await _stream(_model(shared), Request(records=(UserMessage(content="hi"),)))
    assert raised.value.retry_after_sec == 166_122.0
    assert raised.value.limit == "weekly"
    assert str(raised.value) == _SPENT["result"]


@pytest.mark.asyncio
async def test_a_window_still_allowed_says_nothing_of_the_wait(
    claude: _Stub,
    shared: _Resources,
) -> None:
    claude.script(
        [
            _rate_limit_event("allowed", resets_at=1_793_451_600, kind="five_hour"),
            {
                "type": "result",
                "is_error": True,
                "api_error_status": 429,
                "result": "x",
            },
        ],
    )
    with pytest.raises(RateLimitedError) as raised:
        await _stream(_model(shared), Request(records=(UserMessage(content="hi"),)))
    assert raised.value.retry_after_sec == 0.0
    assert raised.value.limit == ""


@pytest.mark.asyncio
async def test_a_failed_turn_leaves_the_next_to_a_fresh_claude(
    claude: _Stub,
    shared: _Resources,
) -> None:
    claude.script([{"type": "result", "is_error": True}], _said("Hello"))
    model = _model(shared)
    request = Request(records=(UserMessage(content="hi"),))
    with pytest.raises(RuntimeError):
        await _stream(model, request)
    await _stream(model, request)
    assert len(claude.processes()) == 2


# ---- the rest ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_closing_the_model_stops_claude_and_unmounts(
    claude: _Stub,
    shared: _Resources,
) -> None:
    claude.script(_said("Hello"))
    model = _model(shared)
    await _stream(model, Request(records=(UserMessage(content="hi"),)))
    (pid,) = claude.pids()
    before = shared.server.mounted.cancelled
    await model.aclose()
    assert shared.server.mounted.cancelled > before  # Its waiting calls fail.
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert shared.server.unmounted == [shared.server.mounted]


@pytest.mark.parametrize("with_reply", [False, True], ids=["same", "with-reply"])
@pytest.mark.asyncio
async def test_a_request_with_no_new_line_starts_a_fresh_claude(
    claude: _Stub,
    shared: _Resources,
    with_reply: bool,
) -> None:
    """A claude that already answered has nothing to answer anew."""
    claude.script(_said("Hello"), _said("Hello again"))
    model = _model(shared)
    first = Request(records=(UserMessage(content="hi"),))
    reply = _response(await _stream(model, first)).records
    again = Request(records=(*first.records, *reply)) if with_reply else first
    events = await _stream(model, again)
    assert _response(events).records == (AssistantMessage(content="Hello again"),)
    replay = "User: hi\n\nAssistant: Hello" if with_reply else "User: hi"
    assert [_text(*said) for _, said in claude.processes()] == [
        ["User: hi"],
        [replay],
    ]


@pytest.mark.parametrize(
    ("piped", "missing"),
    [("stdout", "stdin"), ("stdin", "stdout")],
)
@pytest.mark.asyncio
async def test_a_pool_process_without_both_pipes_is_refused(
    piped: str,
    missing: str,
) -> None:
    async with _Resources(pool=_Unpiped(piped=piped)) as unpiped:
        model = _model(unpiped)
        with pytest.raises(ValueError, match=f"^claude was started without {missing}$"):
            await _stream(model, Request(records=(UserMessage(content="hi"),)))


@pytest.mark.asyncio
async def test_closing_a_model_that_never_streamed_unmounts_nothing(
    shared: _Resources,
) -> None:
    await _model(shared).aclose()
    assert shared.server.unmounted == []
    assert shared.server.mounted.cancelled == 0


@pytest.mark.asyncio
async def test_a_provider_closes_only_resources_it_made() -> None:
    own = AnthropicCLI()
    client = own.resources.http
    await own.aclose()
    assert client.is_closed
    async with _Resources() as resources:
        client = resources.http
        await AnthropicCLI(resources=resources).aclose()
        assert not client.is_closed


@pytest.mark.asyncio
async def test_login_signs_claude_in_to_the_named_account(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(login, "_browser_code", _answered)
    grant = {"access_token": "at-new", "refresh_token": "rt", "expires_in": 3600}
    replies = {
        "https://platform.claude.com/v1/oauth/token": httpx2.Response(
            200,
            content=json_dumps(grant).encode(),
        ),
        "https://api.anthropic.com/api/oauth/profile": httpx2.Response(500),
    }
    client = httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda request: replies[str(request.url)]),
    )
    async with _Resources(http=client) as resources:
        cli = AnthropicCLI(resources=resources, credentials=tmp_path / "work.json")
        await cli.login()
        env = await cli.environment()
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == grant["access_token"]


@pytest.mark.asyncio
@pytest.mark.parametrize("headless", [True, False])
async def test_login_passes_headless_to_the_sign_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    headless: bool,
) -> None:
    seen: list[bool] = []

    async def sign_in(self: object, *, headless: bool = False) -> None:
        del self
        seen.append(headless)

    monkeypatch.setattr(login.ClaudeLogin, "sign_in", sign_in)
    await AnthropicCLI(credentials=tmp_path / "c.json").login(headless=headless)
    assert seen == [headless]


@pytest.mark.asyncio
async def test_count_estimates_from_the_requests_length() -> None:
    model = AnthropicCLI()["opus"]
    request = Request(system=("S",), records=(UserMessage(content="hi " * 100),))
    body, _ = wire.encode(
        request,
        capability=model.capability,
        settings=model.settings,
        system=wire.system_blocks(request.system, settings=model.settings),
    )
    assert await model.count(request) == int(len(json_dumps(body)) / 2.38)


@pytest.mark.asyncio
async def test_count_bills_an_image_by_its_pixels_not_its_bytes() -> None:
    model = AnthropicCLI()["opus"]
    picture = Attachment(mime_descriptor="image/png", data=_png(300, 300))
    bare = Request(records=(UserMessage(content="look"),))
    attached = Request(records=(UserMessage(content="look", attachments=(picture,)),))

    added = await model.count(attached) - await model.count(bare)

    assert added == model.attachment_tokens(picture) == 300 * 300 // 750


def test_the_cli_offers_no_effort_and_manages_its_own_context() -> None:
    row = models("anthropic-cli")["opus"]
    assert row.thinking_effort == {"none"}
    assert row.thinking_budget == {"none", "auto"}
    assert row.thinking_output == {"none", "text"}
    assert row.manage_context_server_side == {True}
    assert row.service_tier == {"auto", "default"}
    assert AnthropicCLI().usage == ()
    assert AnthropicCLI().name == "anthropic-cli"
    assert isinstance(AnthropicCLI(), Login)
    assert list(AnthropicCLI()) == list(models("anthropic-cli"))


def test_the_cli_model_takes_offered_settings_and_refuses_the_rest() -> None:
    model = AnthropicCLI()["opus"]
    model.settings = replace(model.settings, thinking_budget="auto")
    assert model.settings.thinking_budget == "auto"
    with pytest.raises(ValueError, match="thinking_effort='high' is not offered"):
        model.settings = replace(model.settings, thinking_effort="high")
    assert model.settings.thinking_effort == "none"


@dataclass(slots=True, kw_only=True)
class _Stub:
    """The stub ``claude``'s directory: its script, its turns, and its records."""

    dir: Path

    def script(self, *turns: Sequence[Mapping[str, object]]) -> None:
        """Answer the n-th line any stub reads with the n-th turn's events."""
        for n, events in enumerate(turns):
            lines = "".join(f"{json_dumps(event)}\n" for event in events)
            (self.dir / f"turn.{n}").write_text(lines)

    def pids(self) -> list[int]:
        """Return each stub spoken to, in the order it was first spoken to."""
        return list(dict.fromkeys(pid for pid, _ in self._said()))

    def processes(self) -> list[tuple[list[str], list[list[dict[str, object]]]]]:
        """Return each stub's arguments and the content of each line it read."""
        said = self._said()
        return [
            (
                (self.dir / f"argv.{pid}").read_text().split("\0")[:-1],
                [content for speaker, content in said if speaker == pid],
            )
            for pid in self.pids()
        ]

    def _said(self) -> list[tuple[int, list[dict[str, object]]]]:
        path = self.dir / "said"
        lines = path.read_text().splitlines() if path.exists() else []
        said: list[tuple[int, list[dict[str, object]]]] = []
        for line in lines:
            pid, _, raw = line.partition(" ")
            message = json_loads(raw, dict[str, object])
            assert message["type"] == "user"
            body = from_plain(message["message"], dict[str, object])
            assert body["role"] == "user"
            content = from_plain(body["content"], list[dict[str, object]])
            said.append((int(pid), content))
        return said


@dataclass(slots=True, kw_only=True)
class _Mount:
    """Where the stub's tool calls would wait; it answers them all at once."""

    url: str = "http://127.0.0.1:9/mount-token"
    offered: list[tuple[Tool, ...]] = field(default_factory=list[tuple["Tool", ...]])
    answered: list[tuple[ToolCall, AnyToolResult]] = field(
        default_factory=list[tuple[ToolCall, "AnyToolResult"]],
    )
    unanswerable: bool = False
    """Whether every call is one the CLI never made, as for a tool it lacks."""

    cancelled: int = 0

    def offer(self, tools: Sequence[Tool]) -> None:
        self.offered.append(tuple(tools))

    async def called(self) -> ToolCall:
        raise AssertionError("The claude transport reads its calls from its output.")

    async def answer(self, call: ToolCall, result: AnyToolResult) -> None:
        if self.unanswerable:
            raise TimeoutError
        self.answered.append((call, result))

    def cancel(self) -> None:
        self.cancelled += 1


class _ToolServer(AsyncResource):
    """One mount, so a test can read what reached it."""

    def __init__(self) -> None:
        self.mounted = _Mount()
        self.unmounted: list[ToolMount] = []

    async def mount(self) -> _Mount:
        return self.mounted

    async def unmount(self, mount: ToolMount) -> None:
        self.unmounted.append(mount)


class _Resources(AsyncResource):
    """Real process pools, beside a tool server that runs no MCP."""

    def __init__(
        self,
        *,
        http: httpx2.AsyncClient | None = None,
        pool: ProcessPool | None = None,
    ) -> None:
        self._shared = SharedResources()
        self._client = http
        self._pool = pool
        self.server = _ToolServer()
        self.pools: set[str] = set()

    @property
    def http(self) -> httpx2.AsyncClient:
        return self._shared.http if self._client is None else self._client

    def process_pool(self, name: str) -> ProcessPool:
        self.pools.add(name)
        return self._shared.process_pool(name) if self._pool is None else self._pool

    @property
    def tool_server(self) -> _ToolServer:
        return self.server

    @override
    async def aclose(self) -> None:
        await self._shared.aclose()
        if self._client is not None:
            await self._client.aclose()


class _Unpiped(AsyncResource):
    """Starts each process with only its ``piped`` stream connected to us."""

    def __init__(self, *, piped: str) -> None:
        self._piped = piped

    async def acquire(self, argv: Sequence[str], *, env: Mapping[str, str]) -> Process:
        del argv, env
        stdin = (
            asyncio.subprocess.PIPE
            if self._piped == "stdin"
            else asyncio.subprocess.DEVNULL
        )
        stdout = (
            asyncio.subprocess.PIPE
            if self._piped == "stdout"
            else asyncio.subprocess.DEVNULL
        )
        return await asyncio.create_subprocess_exec("cat", stdin=stdin, stdout=stdout)

    async def release(self, process: Process) -> None:
        if process.returncode is None:
            process.kill()
        await process.wait()


@dataclass(frozen=True, slots=True, kw_only=True)
class _Echo:
    """A tool the model may call; the caller, not claude, would run it."""

    name: str = "Echo"
    description: str = "Say the text back."
    schema: Mapping[str, PlainTree] = field(
        default_factory=lambda: MappingProxyType({"type": "object"}),
    )
    display: ToolDisplay = field(default_factory=ToolDisplay)

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        del arguments
        return self.name

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        del context, arguments
        return None

    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> UncategorizedToolResult:
        del context
        return UncategorizedToolResult(call_id="", content=str(arguments.get("text")))


def _streamed(*events: Mapping[str, object]) -> list[dict[str, object]]:
    return [{"type": "stream_event", "event": dict(event)} for event in events]


def _said(text: str) -> list[dict[str, object]]:
    """Return the output of a turn in which the model says ``text``."""
    return [
        *_streamed(
            {"type": "message_start", "message": {"usage": {"input_tokens": 3}}},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            },
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {},
            },
            {"type": "message_stop"},
        ),
        {"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}},
        {"type": "result", "is_error": False, "result": text},
    ]


def _calls(
    call_id: str,
    name: str,
    arguments: Mapping[str, object],
) -> list[dict[str, object]]:
    """Return the output of a message in which the model calls our tool ``name``."""
    return _streamed(
        {"type": "message_start", "message": {"usage": {"input_tokens": 3}}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": call_id,
                "name": f"mcp__sagent__{name}",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "input_json_delta",
                "partial_json": json_dumps(arguments),
            },
        },
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {}},
        {"type": "message_stop"},
    )


async def _answered(
    challenge: str,
    state: str,
    *,
    show: Callable[[str], object],
) -> tuple[str, str]:
    """Stand in for the browser: it brings back a code at once."""
    del challenge, state, show
    return "the-code", "http://localhost:1/callback"


def _read(path: Path) -> str:
    return path.read_text()


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _png(width: int, height: int) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (width, height), "red").save(out, format="PNG")
    return out.getvalue()


def _png_attachment() -> Attachment:
    return Attachment(mime_descriptor="image/png", data=_png(2, 2))


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
