"""Tests for Claude through the ``claude`` CLI, against a stub ``claude`` on PATH."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, override
from unittest.mock import AsyncMock

import asyncio
import io
import os
import re
import stat
import tempfile
import time

from PIL import Image
from treekle import PlainTree, ReadError, from_plain, json_dumps, json_loads

import pytest
import pytest_asyncio

from sagent.journal.types import (
    AgentMessage,
    AssistantMessage,
    Attachment,
    ContextClear,
    SessionContext,
    SystemMessage,
    Thinking,
    TokenUsage,
    ToolCall,
    ToolMessage,
    UncategorizedToolResult,
    UserMessage,
)
from sagent.model.exceptions import (
    AuthError,
    ContextOverflowError,
    DisconnectedError,
    RateLimitedError,
)
from sagent.model.types import Request, Response, TextChunk
from sagent.providers import cli_login
from sagent.providers.anthropic import cli, wire
from sagent.providers.anthropic.cli import AnthropicCLI, AnthropicCLIModel
from sagent.providers.login import Logins
from sagent.providers.resources import SharedResources
from sagent.providers.testing import refuse_browsers
from sagent.providers.types import HasLogin
from sagent.tools.display import ToolDisplay
from sagent.tools.types import (
    Tool,
    ToolOutput,
    ToolStreamingResult,
    whole_output,
)
from sagent.types import AsyncResource


if TYPE_CHECKING:
    from asyncio.subprocess import Process
    from collections.abc import AsyncIterator, Awaitable, Sequence

    import httpx2

    from sagent.journal.types import SessionRecord
    from sagent.model.capability import ModelSettingName
    from sagent.providers.types import ProcessPool, ToolMount
    from sagent.tools.types import ToolContext


@pytest.fixture(autouse=True)
def no_real_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail a test that opens the user's browser, as a sign-in not headless would."""
    refuse_browsers(monkeypatch, cli_login)


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
async def test_claude_keeps_the_login_sagent_keeps_whatever_this_process_says(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/elsewhere")
    monkeypatch.setenv("CLAUDE_SECURESTORAGE_CONFIG_DIR", "/elsewhere")
    env = await AnthropicCLI().environment()
    assert "CLAUDE_CONFIG_DIR" not in env
    assert "CLAUDE_SECURESTORAGE_CONFIG_DIR" not in env


@pytest.mark.asyncio
async def test_a_named_login_is_claude_s_own_in_its_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``claude`` renews it there, so a renewal never restarts it."""
    _keep_logins_in(tmp_path, monkeypatch)
    cli = AnthropicCLI(account="work")
    path = cli.logins.path("work")
    path.parent.mkdir()
    path.write_text(json_dumps({"claudeAiOauth": {"accessToken": "at-1"}}))
    env = await cli.environment()
    assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path / ".claude-work")
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in env
    path.write_text(json_dumps({"claudeAiOauth": {"accessToken": "at-2"}}))
    assert await cli.environment() == env


def test_its_accounts_are_the_logins_saved_beside_claude_s(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _keep_logins_in(tmp_path, monkeypatch)
    for name in (".claude", ".claude-work"):
        (tmp_path / name).mkdir()
        (tmp_path / name / ".credentials.json").write_text("{}")
    assert AnthropicCLI().logins.accounts() == ("work",)


def _keep_logins_in(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``claude``'s default login, and so every account's, live in ``home``."""
    monkeypatch.setattr(
        AnthropicCLI,
        "default_login_path",
        staticmethod(lambda: home / ".claude" / ".credentials.json"),
    )


@pytest.mark.asyncio
async def test_a_turn_starts_claude_with_our_tools_only(
    claude: _Stub,
    shared: _Resources,
) -> None:
    claude.script(_reply_output("Hello"))
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
        "--thinking-display",
        "summarized",
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
        # opus-5.5 starts where Claude Code starts it.
        "--effort",
        "medium",
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
    claude.script(_reply_output("Hello"), _reply_output("Again"))
    model = _model(shared)
    first = Request(records=(UserMessage(content="hi"),))
    response = _response(await _stream(model, first))
    second = Request(
        records=(
            *first.records,
            *response.records,
            UserMessage(content="more"),
            AgentMessage(sender="lead", content="ping"),
            AgentMessage(content="unsigned"),
            AgentMessage(),
            UserMessage(),
            SessionContext(),
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
    claude.script((*_calls("toolu_1", "Echo", {"text": "x"}), *_reply_output("Done")))
    model = _model(shared)
    first = Request(records=(UserMessage(content="echo x"),), tools=(_Echo(),))
    response = _response(await _stream(model, first))
    call = ToolCall(call_id="toolu_1", name="Echo", arguments={"text": "x"})
    # The model calls our tool under the MCP server's name.
    assert response.records == (call,)
    assert response.stop_reason == "tool_use"
    result = ToolMessage(
        call_id="toolu_1",
        state="done",
        stdout=("x",),
        result=UncategorizedToolResult(),
    )
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
            *_reply_output("Done"),
        ),
        _reply_output("Noted"),
        _reply_output("Again"),
    )
    model = _model(shared)
    records: tuple[SessionRecord, ...] = (UserMessage(content="go"),)
    records += _response(await _stream(model, Request(records=records))).records
    for call_id, note in (("toolu_1", "One changed."), ("toolu_2", "Two changed.")):
        records += (
            ToolMessage(
                call_id=call_id,
                state="done",
                stdout=("ok",),
                result=UncategorizedToolResult(),
            ),
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
    claude.script((*_calls("toolu_1", "Echo", {}), *_reply_output("Seen")))
    model = _model(shared)
    first = Request(records=(UserMessage(content="look"),))
    (call,) = _response(await _stream(model, first)).records
    wide = Attachment(mime_descriptor="image/png", data=_png(2_600, 1))
    pdf = Attachment(mime_descriptor="application/pdf", data=b"%PDF-1.4")
    result = ToolMessage(
        call_id="toolu_1",
        state="done",
        stdout=(
            wide,
            pdf,
        ),
        result=UncategorizedToolResult(),
    )
    await _stream(model, Request(records=(*first.records, call, result)))
    ((_, answered),) = shared.server.mounted.answered
    shrunk, kept = answered.attachments
    assert kept is pdf  # Only images are shrunk.
    assert shrunk.mime_descriptor == "image/webp"
    assert Image.open(io.BytesIO(shrunk.data)).size == (2_576, 1)


@pytest.mark.parametrize(
    "change",
    ["system", "tools", "schema", "environment", "history", "display"],
)
@pytest.mark.asyncio
async def test_a_change_the_process_cannot_hear_starts_a_fresh_claude(
    claude: _Stub,
    shared: _Resources,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    claude.script(_reply_output("Hello"), _reply_output("Again"))
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
    if change == "display":
        model.settings = replace(model.settings, thinking_output="none")
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
    # Asked for summaries, claude returns them; told nothing, it returns none.
    assert ("--thinking-display" in started[1][0]) == (change != "display")


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
    claude.script(two_calls, _reply_output("Recovered"))
    model = _model(shared)
    first = Request(records=(UserMessage(content="go"),))
    calls = _response(await _stream(model, first)).records
    assert [c.call_id for c in calls if isinstance(c, ToolCall)] == [
        "toolu_1",
        "toolu_2",
    ]
    result = ToolMessage(
        call_id="toolu_1",
        state="done",
        stdout=("one",),
        result=UncategorizedToolResult(),
    )
    await _stream(model, Request(records=(*first.records, *calls, result)))
    assert shared.server.mounted.answered == []
    assert len(claude.processes()) == 2


@pytest.mark.asyncio
async def test_a_stream_event_with_no_event_is_skipped(
    claude: _Stub,
    shared: _Resources,
) -> None:
    claude.script(
        [{"type": "stream_event"}, {"type": "system"}, *_reply_output("Hello")],
    )
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
    claude.script(_calls("toolu_1", "Echo", {"text": "x"}), _reply_output("Recovered"))
    shared.server.mounted.unanswerable = True
    model = _model(shared)
    first = Request(records=(UserMessage(content="echo x"),))
    (call,) = _response(await _stream(model, first)).records
    result = ToolMessage(
        call_id="toolu_1",
        state="done",
        stdout=("x",),
        result=UncategorizedToolResult(),
    )
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
    claude.script(_reply_output("Hello"))
    records = (
        ContextClear(
            summary="We fixed the bug.",
            history=(UserMessage(content="a"),),
        ),
        ContextClear(history=(AssistantMessage(content="b"),)),
        # Messages that say nothing add no line.
        AssistantMessage(content=""),
        AgentMessage(sender="lead", content=""),
        AgentMessage(
            sender="lead",
            content="status?",
            attachments=(
                Attachment(mime_descriptor="application/pdf", data=b"%PDF-1.4"),
            ),
        ),
        AgentMessage(content="unsigned"),
        SystemMessage(content="Saved."),
        SystemMessage(content=None),
        SessionContext(cwd=Path("/work")),
        Thinking(content="hidden", encrypted="sig"),
        UserMessage(content="", attachments=(_png_attachment(),)),
        UserMessage(content="q"),
    )
    await _stream(_model(shared), Request(records=records))
    ((_, said),) = claude.processes()
    (content,) = said
    # What each line attaches follows the words, in the lines' order.
    assert [block["type"] for block in content] == ["text", "document", "image"]
    assert content[0]["text"] == "\n\n".join(
        [
            "Summary of the conversation so far:\nWe fixed the bug.",
            "User: a",
            "Assistant: b",
            "User: [from lead]: status?",
            "User: unsigned",
            "<system-reminder>\nSaved.\n</system-reminder>",
            "<system-reminder>\nWorking directory: /work\n</system-reminder>",
            "User: q",
        ],
    )


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
    monkeypatch.setattr(time, "time", lambda: 1_793_285_478.0)
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
async def test_a_throttle_while_the_window_allows_waits_no_reset(
    claude: _Stub,
    shared: _Resources,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An allowed window names its reset too; a 429 under it is a throttle."""
    monkeypatch.setattr(time, "time", lambda: 1_793_285_478.0)
    claude.script(
        [
            _rate_limit_event("allowed", resets_at=1_793_451_600, kind="seven_day"),
            _SPENT,
        ],
    )
    with pytest.raises(RateLimitedError) as raised:
        await _stream(_model(shared), Request(records=(UserMessage(content="hi"),)))
    assert (raised.value.retry_after_sec, raised.value.limit) == (0.0, "")


@pytest.mark.parametrize(
    ("info", "headers"),
    [
        (
            {"status": "rejected", "rateLimitType": "seven_day", "resetsAt": 7},
            {
                "anthropic-ratelimit-unified-status": "rejected",
                "anthropic-ratelimit-unified-representative-claim": "seven_day",
                "anthropic-ratelimit-unified-reset": "7",
            },
        ),
        (
            {},
            {
                "anthropic-ratelimit-unified-status": "",
                "anthropic-ratelimit-unified-representative-claim": "",
            },
        ),
    ],
    ids=["spent", "silent"],
)
def test_a_rate_limit_event_reads_as_the_api_s_headers(
    info: dict[str, object],
    headers: dict[str, str],
) -> None:
    """Spelled as the API sends them, so ``wire`` reads both alike."""
    event: dict[str, object] = {"type": "rate_limit_event", "rate_limit_info": info}
    assert cli._unified_headers(event) == headers


@pytest.mark.asyncio
async def test_a_rate_limit_event_that_says_nothing_says_no_window(
    claude: _Stub,
    shared: _Resources,
) -> None:
    claude.script([{"type": "rate_limit_event"}, _SPENT])
    with pytest.raises(RateLimitedError) as raised:
        await _stream(_model(shared), Request(records=(UserMessage(content="hi"),)))
    assert raised.value.limit == ""


@pytest.mark.asyncio
async def test_a_line_that_is_not_an_object_is_an_error(
    claude: _Stub,
    shared: _Resources,
) -> None:
    (claude.dir / "turn.0").write_text("[1]\n")
    with pytest.raises(ReadError):
        await _stream(_model(shared), Request(records=(UserMessage(content="hi"),)))


@pytest.mark.asyncio
async def test_claude_quiet_for_ten_minutes_has_dropped_the_turn(
    claude: _Stub,
    shared: _Resources,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claude.script(_reply_output("Hello"))
    real = asyncio.wait_for
    waits: list[float | None] = []

    async def recorded[T](awaitable: Awaitable[T], **limits: float | None) -> T:
        waits.append(limits["timeout"])
        return await real(awaitable, **limits)

    monkeypatch.setattr(asyncio, "wait_for", recorded)
    await _stream(_model(shared), Request(records=(UserMessage(content="hi"),)))
    assert waits
    assert set(waits) == {600.0}


@pytest.mark.parametrize(
    "line",
    [UserMessage, AgentMessage],
    ids=["a person's", "another agent's"],
)
@pytest.mark.asyncio
async def test_a_new_lines_images_and_pdfs_reach_the_running_claude(
    claude: _Stub,
    shared: _Resources,
    line: type[UserMessage | AgentMessage],
) -> None:
    claude.script(_reply_output("Hello"), _reply_output("A dot"))
    model = _model(shared)
    first = Request(records=(UserMessage(content="hi"),))
    reply = _response(await _stream(model, first)).records
    pdf = Attachment(mime_descriptor="application/pdf", data=b"%PDF-1.4")
    seen = line(content="see", attachments=(_png_attachment(), pdf))
    await _stream(model, Request(records=(*first.records, *reply, seen)))
    ((_, said),) = claude.processes()
    assert [block["type"] for block in said[1]] == ["text", "image", "document"]


def test_a_model_that_takes_no_effort_is_told_none(shared: _Resources) -> None:
    sonnet = _model(shared, name="sonnet-4.5")
    assert sonnet.settings.thinking_effort == "none"
    assert "--effort" not in sonnet.command(Request(), url="http://127.0.0.1:9/m")
    opus = _model(shared)
    command = opus.command(Request(), url="http://127.0.0.1:9/m")
    assert command[command.index("--effort") + 1] == opus.settings.thinking_effort


@pytest.mark.asyncio
async def test_count_takes_records_that_attach_nothing() -> None:
    model = AnthropicCLI()["opus"]
    said = Request(records=(UserMessage(content="hi"),))
    answered = Request(
        records=(*said.records, TokenUsage(), AssistantMessage(content="hello")),
    )
    assert await model.count(answered) > await model.count(said)


@pytest.mark.asyncio
async def test_a_spent_window_with_no_reset_is_still_named(
    claude: _Stub,
    shared: _Resources,
) -> None:
    event = _rate_limit_event("rejected", resets_at=0, kind="five_hour")
    info = event["rate_limit_info"]
    assert isinstance(info, dict)
    del info["resetsAt"]
    claude.script([event, _SPENT])
    with pytest.raises(RateLimitedError) as raised:
        await _stream(_model(shared), Request(records=(UserMessage(content="hi"),)))
    assert (raised.value.retry_after_sec, raised.value.limit) == (0.0, "5-hour")


@pytest.mark.asyncio
async def test_a_rate_limit_through_claude_holds_the_login_for_every_caller(
    claude: _Stub,
    shared: _Resources,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI bills the subscription's login: its 429 holds that model there too."""
    monkeypatch.setattr(time, "time", lambda: 1_793_285_478.0)
    claude.script(
        [
            _rate_limit_event("rejected", resets_at=1_793_451_600, kind="seven_day"),
            _SPENT,
        ],
    )
    with pytest.raises(RateLimitedError):
        await _stream(_model(shared), Request(records=(UserMessage(content="hi"),)))
    model = _model(shared).capability.model_id
    held = shared.gate.states()[f"claude@default/{model}"]
    assert held.held_until == 1_793_451_600.0


@pytest.mark.asyncio
async def test_a_quota_recheck_reaches_claude_despite_an_old_hold(
    claude: _Stub,
    shared: _Resources,
) -> None:
    claude.script(_reply_output("restored"))
    model = _model(shared)
    key = f"claude@default/{model.capability.model_id}"
    shared.gate.hold(key, until=time.time() + 432_000.0)
    request = Request(records=(UserMessage(content="hi"),))
    with pytest.raises(RateLimitedError, match="cooling down"):
        await _stream(model, request)
    response = _response(await _stream(model, replace(request, recheck_quota=True)))
    assert response.records == (AssistantMessage(content="restored"),)
    assert shared.gate.states()[key].held_until == 0.0
    await model.aclose()


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
    claude.script([{"type": "result", "is_error": True}], _reply_output("Hello"))
    model = _model(shared)
    request = Request(records=(UserMessage(content="hi"),))
    with pytest.raises(RuntimeError):
        await _stream(model, request)
    await _stream(model, request)
    assert len(claude.processes()) == 2


@pytest.mark.asyncio
async def test_closing_the_model_stops_claude_and_unmounts(
    claude: _Stub,
    shared: _Resources,
) -> None:
    claude.script(_reply_output("Hello"))
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
    claude.script(_reply_output("Hello"), _reply_output("Hello again"))
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


def test_the_default_login_is_the_file_claude_keeps_in_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    assert AnthropicCLI.default_login_path() == (
        tmp_path / ".claude" / ".credentials.json"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("given", "headless"),
    [({"headless": True}, True), ({"headless": False}, False), ({}, False)],
    ids=["headless", "browser", "default"],
)
async def test_login_is_claude_s_own_sign_in_to_the_named_account(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    given: dict[str, bool],
    headless: bool,
) -> None:
    """``claude`` opens its own browser, at its own loopback server, unless headless."""
    signed_in = AsyncMock()
    added = AsyncMock()
    monkeypatch.setattr(cli, "cli_sign_in", signed_in)
    monkeypatch.setattr(Logins, "added", added)
    _keep_logins_in(tmp_path, monkeypatch)
    provider = AnthropicCLI(account="work")
    await provider.login(**given)
    env = await provider.environment()
    assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path / ".claude-work")
    signed_in.assert_awaited_once_with(
        "claude",
        "auth",
        "login",
        "--claudeai",
        env=env | ({"BROWSER": "true"} if headless else {}),
        browser=False,
        grace_sec=1.0,
    )
    added.assert_awaited_once_with("work")


@pytest.mark.asyncio
async def test_a_failed_login_adds_no_account(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refused = AuthError("claude sign-in failed: no")
    added = AsyncMock()
    monkeypatch.setattr(cli, "cli_sign_in", AsyncMock(side_effect=refused))
    monkeypatch.setattr(Logins, "added", added)
    _keep_logins_in(tmp_path, monkeypatch)
    with pytest.raises(AuthError) as raised:
        await AnthropicCLI(account="work").login()
    assert raised.value is refused
    added.assert_not_awaited()


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


@pytest.mark.asyncio
async def test_count_bills_a_tool_results_image_by_its_pixels_not_its_bytes() -> None:
    model = AnthropicCLI()["opus"]
    picture = Attachment(mime_descriptor="image/png", data=_png(300, 300))
    added = await model.count(_answered("seen", picture)) - await model.count(
        _answered("seen"),
    )
    assert added == model.attachment_tokens(picture) == 300 * 300 // 750


def _answered(*stdout: str | Attachment) -> Request:
    """Return a request whose one tool call was answered with ``stdout``."""
    call = ToolCall(call_id="toolu_1", name="Read", arguments={})
    result = ToolMessage(
        call_id="toolu_1",
        state="done",
        stdout=stdout,
        result=UncategorizedToolResult(),
    )
    return Request(records=(UserMessage(content="look"), call, result))


@pytest.mark.parametrize(
    ("model", "budgets"),
    [
        ("opus", {"auto"}),
        ("opus-5.0", {"none", "auto"}),
        ("opus-4.6", {"none", "auto", "fixed"}),
    ],
)
def test_the_cli_offers_what_claude_takes_and_no_tier(
    model: str,
    budgets: set[str],
) -> None:
    """``claude`` takes effort, a thinking budget, a display, and a cache lifetime.

    Its tier and its context are its own. Unasked, on the subscription, it wrote
    every token of a measured call to the one-hour cache (2.1.309, 2026-11-16).
    """
    row = AnthropicCLI.models()[model]
    assert row.thinking_budget == budgets
    assert row.thinking_output == {"none", "text"}
    assert row.default.thinking_output == "text"
    assert row.manage_context_server_side == {True}
    assert row.service_tier == {"auto"}
    assert row.cache_ttl_sec == {300.0, 3600.0}
    assert row.default.cache_ttl_sec == 3600.0
    assert AnthropicCLI().usage == ()
    assert AnthropicCLI().name == "anthropic-cli"
    assert isinstance(AnthropicCLI(), HasLogin)
    assert list(AnthropicCLI()) == list(AnthropicCLI.models())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "changed", "told"),
    [
        ("opus", (), {"CLAUDE_CODE_PROMPT_CACHE_TTL": "1h"}),
        ("opus", (("cache_ttl_sec", 300.0),), {"CLAUDE_CODE_PROMPT_CACHE_TTL": "5m"}),
        (
            "opus-5.0",
            (("thinking_budget", "none"),),
            {"CLAUDE_CODE_PROMPT_CACHE_TTL": "1h", "CLAUDE_CODE_DISABLE_THINKING": "1"},
        ),
        (
            "opus-4.6",
            (("thinking_budget", "fixed"),),
            {
                "CLAUDE_CODE_PROMPT_CACHE_TTL": "1h",
                "CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING": "1",
                "MAX_THINKING_TOKENS": "64000",
            },
        ),
        (
            "sonnet-4.5",
            (),
            {"CLAUDE_CODE_PROMPT_CACHE_TTL": "1h", "MAX_THINKING_TOKENS": "32000"},
        ),
    ],
    ids=["auto", "5m", "no-thinking", "fixed-over-adaptive", "fixed"],
)
async def test_claude_is_told_the_budget_and_cache_lifetime_chosen(
    model: str,
    changed: tuple[tuple[ModelSettingName, object], ...],
    told: dict[str, str],
) -> None:
    """As the API is asked them: half the reply room for a fixed budget."""
    chosen = AnthropicCLI()[model]
    settings = chosen.settings
    for name, value in changed:
        settings = settings.with_value(name, value)
    chosen.settings = settings
    env = await chosen.environment()
    knobs = (
        "CLAUDE_CODE_PROMPT_CACHE_TTL",
        "CLAUDE_CODE_DISABLE_THINKING",
        "CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING",
        "MAX_THINKING_TOKENS",
    )
    assert {key: env[key] for key in knobs if key in env} == told
    assert env["DISABLE_AUTO_COMPACT"] == "1"


def test_the_cli_model_takes_offered_settings_and_refuses_the_rest() -> None:
    model = AnthropicCLI()["opus"]
    model.settings = replace(model.settings, thinking_effort="max")
    assert model.settings.thinking_effort == "max"
    with pytest.raises(ValueError, match="thinking_output='redacted' is not offered"):
        model.settings = replace(model.settings, thinking_output="redacted")
    assert model.settings.thinking_output == "text"


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
        return list(dict.fromkeys(pid for pid, _ in self._user_turns()))

    def processes(self) -> list[tuple[list[str], list[list[dict[str, object]]]]]:
        """Return each stub's arguments and the content of each line it read."""
        turns = self._user_turns()
        return [
            (
                (self.dir / f"argv.{pid}").read_text().split("\0")[:-1],
                [content for speaker, content in turns if speaker == pid],
            )
            for pid in self.pids()
        ]

    def _user_turns(self) -> list[tuple[int, list[dict[str, object]]]]:
        path = self.dir / "said"
        lines = path.read_text().splitlines() if path.exists() else []
        turns: list[tuple[int, list[dict[str, object]]]] = []
        for line in lines:
            pid, _, raw = line.partition(" ")
            message = json_loads(raw, dict[str, object])
            assert message["type"] == "user"
            body = from_plain(message["message"], dict[str, object])
            assert body["role"] == "user"
            content = from_plain(body["content"], list[dict[str, object]])
            turns.append((int(pid), content))
        return turns


@dataclass(slots=True, kw_only=True)
class _Mount:
    """Where the stub's tool calls would wait; it answers them all at once."""

    url: str = "http://127.0.0.1:9/mount-token"
    offered: list[tuple[Tool, ...]] = field(
        default_factory=list[tuple["Tool", ...]],
    )
    answered: list[tuple[ToolCall, ToolMessage]] = field(
        default_factory=list[tuple[ToolCall, ToolMessage]],
    )
    unanswerable: bool = False
    """Whether every call is one the CLI never made, as for a tool it lacks."""

    cancelled: int = 0

    def offer(self, tools: Sequence[Tool]) -> None:
        self.offered.append(tuple(tools))

    async def called(self) -> ToolCall:
        raise AssertionError("The claude transport reads its calls from its output.")

    async def answer(self, call: ToolCall, result: ToolMessage) -> None:
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
        self.gate = self._shared.gate
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

    def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> ToolStreamingResult[UncategorizedToolResult, object, object]:
        return whole_output(self._run(context, arguments))

    async def _run(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> ToolOutput[UncategorizedToolResult, object, object]:
        del context
        return ToolOutput(
            stdout=(str(arguments.get("text")),),
            result=UncategorizedToolResult(),
        )


def _streamed(*events: Mapping[str, object]) -> list[dict[str, object]]:
    return [{"type": "stream_event", "event": dict(event)} for event in events]


def _reply_output(text: str) -> list[dict[str, object]]:
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


def _read(path: Path) -> str:
    return path.read_text()


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _png(width: int, height: int) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (width, height), "red").save(out, format="PNG")
    return out.getvalue()


def test_a_person_who_is_not_the_session_s_user_is_named_as_an_agent_is() -> None:
    bob = UserMessage(sender="bob", content="hi")

    assert cli._prompt_line(bob) == "[from user bob]: hi"
    # A fresh process hears the line as a running one does: one label, one way.
    assert cli._transcript((bob, UserMessage(content="mine"))) == [
        "User: [from user bob]: hi",
        "User: mine",
    ]


def _png_attachment() -> Attachment:
    return Attachment(mime_descriptor="image/png", data=_png(2, 2))


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
