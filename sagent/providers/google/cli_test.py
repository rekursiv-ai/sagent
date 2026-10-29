"""Tests for Gemini through ``agy``, against a stub ``agy`` on PATH.

The stub speaks whatever stream-json lines a test scripts: each line it reads plays
the next numbered turn file, counted across every process, and ``WAIT name`` in a
turn waits for a file our fake tool server writes when a held call is answered.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Self, override

import asyncio
import io
import os
import stat
import tempfile

from PIL import Image
from treekle import PlainTree, json_dumps, json_loads

import pytest
import pytest_asyncio

from sagent.model.cost import TokenCount
from sagent.model.exceptions import (
    AuthError,
    DisconnectedError,
    RateLimitedError,
)
from sagent.model.types import Request, Response, TextChunk
from sagent.providers.google.cli import GoogleCLI, GoogleCLIModel
from sagent.providers.resources import SharedResources
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
from sagent.types import AsyncResource


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    import httpx2

    from sagent.model.capability import ThinkingEffort
    from sagent.providers.types import ProcessPool, ToolMount
    from sagent.tools.types import Tool, ToolContext


_AGY = r"""#!/bin/sh
dir=$(dirname "$0")
trap 'exit 0' TERM
while IFS= read -r line; do
  if [ ! -e "$dir/argv.$$" ]; then
    printf '%s\n' "$@" > "$dir/argv.$$"
    env > "$dir/env.$$"
    pwd > "$dir/cwd.$$"
    cp -r "$HOME/.gemini" "$dir/gemini.$$"
  fi
  printf '%s %s\n' "$$" "$line" >> "$dir/said"
  n=$(cat "$dir/count" 2>/dev/null || echo 0)
  echo $((n + 1)) > "$dir/count"
  [ -e "$dir/turn.$n" ] || continue
  while IFS= read -r out; do
    case "$out" in
      "WAIT "*) f="$dir/${out#WAIT }"; while [ ! -e "$f" ]; do sleep 0.005; done ;;
      EXIT) exit 0 ;;
      *) printf '%s\n' "$out" ;;
    esac
  done < "$dir/turn.$n"
done
"""

_INIT = {"event": "init", "conversation_id": "c1", "init": {"tools": ["view_file"]}}


@pytest.fixture(autouse=True)
def private_temp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every agy home under ``tmp_path``, not the shared temp directory."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))


@pytest.fixture
def stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Stub:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "agy"
    script.write_text(_AGY)
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return _Stub(dir=bin_dir)


@pytest_asyncio.fixture
async def shared(stub: _Stub) -> AsyncIterator[_Resources]:
    async with _Resources(stub=stub) as resources:
        yield resources


# ---- the process ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_agy_runs_as_our_agent_in_a_home_of_its_own(
    shared: _Resources,
    stub: _Stub,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Our prompt is its agent's, our tools its only ones; its login is its own."""
    monkeypatch.setenv("GEMINI_API_KEY", "leak")
    monkeypatch.setenv("KEEP_ME", "1")
    stub.script([_INIT, *_said("hi", tokens=(10, 2))])
    provider = GoogleCLI(resources=shared)
    model = provider["gemini-flash"]
    request = Request(
        system=("Be terse.", "", "Rules."),
        records=_HELLO,
        tools=(_Echo(),),
    )
    await _stream(model, request)

    assert stub.only("argv").splitlines() == [
        "--agent",
        "sagent",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--disable-slash-commands",
        "--model",
        "gemini-3.8-flash-high",
    ]
    env = dict(
        line.split("=", 1) for line in stub.only("env").splitlines() if "=" in line
    )
    home = Path(env["HOME"])
    assert (env["KEEP_ME"], "GEMINI_API_KEY" in env, env["USERPROFILE"]) == (
        "1",
        False,
        env["HOME"],
    )
    assert home.name.startswith("sagent-agy-")
    assert stub.only("cwd") == str(home / "workdir")
    gemini = stub.copied()
    # ``agy`` lists an MCP tool by name alone; without its schema the model
    # guesses the arguments, or spends a call reading the schema's file.
    assert (gemini / "config" / "agents" / "sagent" / "agent.md").read_text() == (
        "---\nname: sagent\ndescription: Answers as sagent asks, with sagent's tools.\n"
        "mainAgent: true\ntools: []\n---\nBe terse.\n\nRules.\n\n"
        "# Tools\n\n"
        "Each tool is on the MCP server `sagent`; call it through call_mcp_tool,"
        " with arguments as its input schema says. A relative path resolves"
        " against sagent's working directory, not the workspace agy names.\n\n"
        "## Echo\n\nSay the text back.\n\n"
        'Input schema: {"type":"object","properties":{"text":{"type":"string"}}}\n'
    )
    assert json_loads((gemini / "config" / "mcp_config.json").read_text()) == {
        "mcpServers": {"sagent": {"serverUrl": shared.mount.url}},
    }
    assert json_loads((gemini / "antigravity-cli" / "settings.json").read_text()) == {
        "permissions": {"allow": ["mcp(sagent/*)"]},
        "showTips": False,
        "showFeedbackSurvey": False,
    }
    await provider.aclose()
    assert not await asyncio.to_thread(home.exists)


@pytest.mark.asyncio
@pytest.mark.parametrize("effort", ["low", "medium", "high"])
async def test_the_effort_is_the_model_id_s_suffix(
    shared: _Resources,
    stub: _Stub,
    effort: ThinkingEffort,
) -> None:
    """``agy`` refuses ``--effort`` beside a suffixed id: the suffix is the effort."""
    stub.script([_INIT, *_said("hi")])
    model = GoogleCLI(resources=shared)["gemini-flash"]
    model.settings = replace(model.settings, thinking_effort=effort)
    await _stream(model, Request(records=_HELLO))
    assert stub.only("argv").splitlines()[-1] == f"gemini-3.8-flash-{effort}"
    await model.provider.aclose()


# ---- turns ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_first_turn_tells_the_conversation_and_streams_the_reply(
    shared: _Resources,
    stub: _Stub,
) -> None:
    stub.script(
        [
            _INIT,
            _step("user_input"),
            _step("agent_response", text="good ", state="ACTIVE"),
            _step("agent_response", text="day", usage=(30, 5, 20)),
            _step("checkpoint", usage=(7, 1, 0)),
            _result(),
        ],
    )
    model = GoogleCLI(resources=shared)["gemini-flash"]
    records = (
        UserMessage(content="first"),
        AssistantMessage(content="ok"),
        SystemMessage(content="mind the tests"),
        UserMessage(content="second"),
    )
    events = await _stream(model, Request(records=records))

    assert stub.said() == [
        {
            "event": "user",
            "message": {
                "content": "User: first\n\nAssistant: ok\n\n"
                "<system-reminder>\nmind the tests\n</system-reminder>\n\nUser: second",
            },
        },
    ]
    assert events[:-1] == [TextChunk(text="good "), TextChunk(text="day")]
    response = _response(events)
    assert response.records == (AssistantMessage(content="good day"),)
    assert response.stop_reason == "end_turn"
    # Each step's usage, summed: input, then output, then what the cache served.
    assert response.tokens == TokenCount(request=37, response=6, cache_read=20)
    await model.provider.aclose()


@pytest.mark.asyncio
async def test_the_next_user_message_goes_to_the_same_agy(
    shared: _Resources,
    stub: _Stub,
) -> None:
    stub.script([_INIT, *_said("one")], _said("two"))
    model = GoogleCLI(resources=shared)["gemini-flash"]
    first = _response(await _stream(model, Request(records=_HELLO)))
    records = (*_HELLO, *first.records, UserMessage(content="and now?"))
    second = _response(await _stream(model, Request(records=records)))

    assert second.records == (AssistantMessage(content="two"),)
    assert len(stub.processes()) == 1
    assert stub.said()[-1] == {"event": "user", "message": {"content": "and now?"}}
    await model.provider.aclose()


@pytest.mark.asyncio
async def test_a_call_of_our_tools_is_held_and_its_result_continues_the_turn(
    shared: _Resources,
    stub: _Stub,
) -> None:
    call = ToolCall(call_id="c1", name="Echo", arguments={"text": "x"})
    shared.mount.calls.put_nowait(call)
    stub.script(
        [
            _INIT,
            _step("agent_response", text="calling", usage=(10, 3, 0)),
            _step("tool", state="ACTIVE", tool="call_mcp_tool"),
            "WAIT answered.0",
            _step("tool", tool="call_mcp_tool"),
            _step("agent_response", text="done", usage=(15, 2, 10)),
            _result(),
        ],
    )
    model = GoogleCLI(resources=shared)["gemini-flash"]
    request = Request(records=_HELLO, tools=(_Echo(),))
    held = _response(await _stream(model, request))
    assert held.records == (AssistantMessage(content="calling"), call)
    # Each stream bills its own steps, so the last reply's count is one call's.
    assert (held.stop_reason, held.tokens) == (
        "tool_use",
        TokenCount(request=10, response=3),
    )

    result = UncategorizedToolResult(call_id="c1", content="x")
    records = (*_HELLO, *held.records, result)
    done = _response(await _stream(model, Request(records=records, tools=(_Echo(),))))

    assert shared.mount.answered == [(call, result)]
    assert done.records == (AssistantMessage(content="done"),)
    assert done.tokens == TokenCount(request=15, response=2, cache_read=10)
    assert len(stub.processes()) == 1
    assert shared.mount.offered[-1] == ("Echo",)
    await model.provider.aclose()


@pytest.mark.asyncio
async def test_another_system_prompt_starts_a_fresh_agy(
    shared: _Resources,
    stub: _Stub,
) -> None:
    stub.script([_INIT, *_said("one")], [_INIT, *_said("two")])
    model = GoogleCLI(resources=shared)["gemini-flash"]
    first = _response(await _stream(model, Request(system=("A",), records=_HELLO)))
    records = (*_HELLO, *first.records, UserMessage(content="again"))
    await _stream(model, Request(system=("B",), records=records))

    assert len(stub.processes()) == 2
    assert stub.said()[-1]["message"] == {
        "content": "User: hello\n\nAssistant: one\n\nUser: again",
    }
    await model.provider.aclose()


@pytest.mark.asyncio
async def test_a_fresh_agy_hears_the_whole_conversation_as_a_transcript(
    shared: _Resources,
    stub: _Stub,
) -> None:
    stub.script([_INIT, *_said("ok")])
    model = GoogleCLI(resources=shared)["gemini-flash"]
    records = (
        ContextClear(summary="earlier", history=(UserMessage(content="before"),)),
        ContextClear(history=(UserMessage(content="kept"),)),
        AgentToAgentMessage(content="from a", sender="lead"),
        AgentToAgentMessage(content="anon"),
        Thinking(content="hmm"),
        AssistantMessage(content="ok"),
        ToolCall(call_id="c0", name="Echo", arguments={"text": "x"}),
        UncategorizedToolResult(call_id="c0", content="x"),
        TurnContext(),
        TurnContext(cwd="/w"),
        UserMessage(content="go"),
    )
    await _stream(model, Request(records=records))

    assert stub.said()[0]["message"] == {
        "content": "Summary of the conversation so far:\nearlier\n\n"
        "User: before\n\nUser: kept\n\nAgent lead: from a\n\nAgent: anon\n\n"
        "Assistant: ok\n\n"
        'Assistant called Echo as c0: {"text": "x"}\n\nc0 returned: x\n\n'
        "<system-reminder>\nWorking directory: /w\n</system-reminder>\n\n"
        "User: go",
    }
    await model.provider.aclose()


@pytest.mark.asyncio
async def test_the_next_turn_says_each_new_message_its_own_way(
    shared: _Resources,
    stub: _Stub,
) -> None:
    stub.script([_INIT, *_said("one")], _said("two"))
    model = GoogleCLI(resources=shared)["gemini-flash"]
    first = _response(await _stream(model, Request(records=_HELLO)))
    records = (
        *_HELLO,
        *first.records,
        AgentToAgentMessage(content="hi", sender="lead"),
        AgentToAgentMessage(content="plain"),
        TurnContext(),
        TurnContext(cwd="/w"),
        UserMessage(content="go"),
    )
    await _stream(model, Request(records=records))

    assert stub.said()[-1]["message"] == {
        "content": "[from lead]: hi\n\nplain\n\n"
        "<system-reminder>\nWorking directory: /w\n</system-reminder>\n\ngo",
    }
    assert len(stub.processes()) == 1
    await model.provider.aclose()


@pytest.mark.asyncio
async def test_a_history_it_never_said_starts_a_fresh_agy(
    shared: _Resources,
    stub: _Stub,
) -> None:
    """Only results it waits for, or new user lines, go to a running ``agy``."""
    stub.script([_INIT, *_said("one")], [_INIT, *_said("two")])
    model = GoogleCLI(resources=shared)["gemini-flash"]
    first = _response(await _stream(model, Request(records=_HELLO)))
    records = (
        *_HELLO,
        *first.records,
        AssistantMessage(content="put in"),
        UserMessage(content="again"),
    )
    await _stream(model, Request(records=records))

    assert len(stub.processes()) == 2
    assert stub.said()[-1]["message"] == {
        "content": "User: hello\n\nAssistant: one\n\nAssistant: put in\n\nUser: again",
    }
    await model.provider.aclose()


@pytest.mark.asyncio
async def test_a_call_no_longer_waiting_starts_a_fresh_agy(
    shared: _Resources,
    stub: _Stub,
) -> None:
    """A held call the server gave up on cannot take its result; a new agy hears it."""
    call = ToolCall(call_id="c1", name="Echo", arguments={"text": "x"})
    shared.mount.calls.put_nowait(call)
    stub.script(
        [_INIT, _step("tool", state="ACTIVE", tool="call_mcp_tool"), "WAIT never"],
        [_INIT, *_said("redone")],
    )
    model = GoogleCLI(resources=shared)["gemini-flash"]
    request = Request(records=_HELLO, tools=(_Echo(),))
    held = _response(await _stream(model, request))
    shared.mount.late = True

    result = UncategorizedToolResult(call_id="c1", content="x")
    records = (*_HELLO, *held.records, result)
    done = _response(await _stream(model, Request(records=records, tools=(_Echo(),))))

    assert done.records == (AssistantMessage(content="redone"),)
    assert len(stub.processes()) == 2
    assert stub.said()[-1]["message"] == {
        "content": 'User: hello\n\nAssistant called Echo as c1: {"text": "x"}\n\n'
        "c1 returned: x",
    }
    await model.provider.aclose()


@pytest.mark.asyncio
async def test_a_line_that_is_no_json_is_passed_over(
    shared: _Resources,
    stub: _Stub,
) -> None:
    stub.script([_INIT, "warning: not json", *_said("hi")])
    model = GoogleCLI(resources=shared)["gemini-flash"]
    response = _response(await _stream(model, Request(records=_HELLO)))
    assert response.records == (AssistantMessage(content="hi"),)
    await model.provider.aclose()


@pytest.mark.asyncio
async def test_an_image_is_named_not_sent(shared: _Resources, stub: _Stub) -> None:
    """``agy`` takes text alone; the model is told an image was left out."""
    stub.script([_INIT, *_said("ok")])
    model = GoogleCLI(resources=shared)["gemini-flash"]
    image = Attachment(mime_descriptor="image/png", data=_png(2, 2))
    await _stream(
        model,
        Request(records=(UserMessage(content="look", attachments=(image,)),)),
    )
    assert stub.said()[0]["message"] == {
        "content": "User: look\n[an image/png attachment, which agy cannot send]",
    }
    await model.provider.aclose()


# ---- failures ------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "kind"),
    [
        ("authentication required", AuthError),
        ("RESOURCE_EXHAUSTED: quota exceeded", RateLimitedError),
        ("model is not recognized", RuntimeError),
    ],
)
async def test_a_failed_turn_raises_what_agy_says(
    shared: _Resources,
    stub: _Stub,
    error: str,
    kind: type[Exception],
) -> None:
    stub.script([_INIT, _result(status="ERROR", error=error)])
    model = GoogleCLI(resources=shared)["gemini-flash"]
    with pytest.raises(kind, match=error):
        await _stream(model, Request(records=_HELLO))
    await model.provider.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "wait_sec"),
    [
        # As agy 1.2.2 ended a turn on a spent window (antigravity-cli#1018).
        (
            (
                "API error (attempt 3): RESOURCE_EXHAUSTED (code 429): Individual quota"
                " reached. Please upgrade your subscription to increase your limits."
                " Resets in 2h39m52s."
            ),
            2 * 3_600 + 39 * 60 + 52,
        ),
        (
            (
                "You have exhausted your capacity on this model."
                " Your quota will reset after 155h19m17s."
            ),
            155 * 3_600 + 19 * 60 + 17,
        ),
        ("RESOURCE_EXHAUSTED (code 429): Resets in 45s.", 45),
        ("RESOURCE_EXHAUSTED: quota exceeded", 0),
    ],
)
async def test_a_spent_quota_waits_for_the_reset_agy_names(
    shared: _Resources,
    stub: _Stub,
    error: str,
    wait_sec: int,
) -> None:
    """Agy says when the quota resets; dropped, the wait was a backoff's guess."""
    stub.script([_INIT, _result(status="ERROR", error=error)])
    model = GoogleCLI(resources=shared)["gemini-flash"]
    with pytest.raises(RateLimitedError) as raised:
        await _stream(model, Request(records=_HELLO))
    assert raised.value.retry_after_sec == wait_sec
    await model.provider.aclose()


@pytest.mark.asyncio
async def test_agy_exiting_mid_turn_is_a_disconnect(
    shared: _Resources,
    stub: _Stub,
) -> None:
    stub.script([_INIT, _step("agent_response", text="half", state="ACTIVE"), "EXIT"])
    model = GoogleCLI(resources=shared)["gemini-flash"]
    with pytest.raises(DisconnectedError, match="agy exited"):
        await _stream(model, Request(records=_HELLO))
    await model.provider.aclose()


@pytest.mark.asyncio
async def test_agy_quiet_too_long_is_a_disconnect(
    shared: _Resources,
    stub: _Stub,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(GoogleCLIModel, "quiet_sec", 0.05)
    stub.script([_INIT, "WAIT never"])
    model = GoogleCLI(resources=shared)["gemini-flash"]
    with pytest.raises(DisconnectedError, match=r"^agy went quiet for 0\.05 seconds$"):
        await _stream(model, Request(records=_HELLO))
    await model.provider.aclose()


# ---- the provider --------------------------------------------------------------


def test_the_provider_serves_what_agy_offers() -> None:
    """``agy models`` listed 3.8 Flash alone on 2026-10-27, at three efforts."""
    provider = GoogleCLI()
    assert (GoogleCLI.section, provider.name, provider.usage) == (
        "google-cli",
        "google-cli",
        (),
    )
    assert GoogleCLI(name="work").name == "work"
    assert list(provider) == ["gemini-flash-3.8"]
    for role in ("default", "best", "everyday", "utility"):
        assert provider[role].capability.model_id == "gemini-flash-3.8"
    assert provider["gemini-flash"].capability.thinking_effort == frozenset(
        {"low", "medium", "high"},
    )


def test_an_attachment_costs_only_the_line_naming_it() -> None:
    model = GoogleCLI()["gemini-flash"]
    picture = Attachment(mime_descriptor="image/png", data=_png(1600, 800))
    named = "[an image/png attachment, which agy cannot send]"
    assert model.attachment_tokens(picture) == int(
        len(named) / model.capability.approx_chars_per_token,
    )


@pytest.mark.asyncio
async def test_a_request_counts_as_the_text_agy_is_sent() -> None:
    model = GoogleCLI()["gemini-flash"]
    picture = Attachment(mime_descriptor="image/png", data=_png(1600, 800))
    plain = Request(system=("S",), records=(UserMessage(content="look"),))
    pictured = Request(
        system=("S",),
        records=(UserMessage(content="look", attachments=(picture,)),),
    )

    grown = await model.count(pictured) - await model.count(plain)

    assert 0 < grown <= model.attachment_tokens(picture) + 1


@pytest.mark.asyncio
async def test_a_provider_given_no_resources_closes_its_own() -> None:
    provider = GoogleCLI()
    client = provider.resources.http
    await provider.aclose()
    assert client.is_closed


# ---- helpers -------------------------------------------------------------------


_HELLO: tuple[UserMessage, ...] = (UserMessage(content="hello"),)


def _step(
    kind: str,
    *,
    text: str = "",
    state: str = "DONE",
    usage: tuple[int, int, int] | None = None,
    tool: str = "",
) -> dict[str, object]:
    """Return one ``step_update``; ``usage`` is input, output, and cache-read tokens."""
    step: dict[str, object] = {
        "conversation_id": "c1",
        "state": state,
        "step_type": kind,
    }
    if text:
        step["text_delta"] = text
    if tool:
        step["tool_name"] = tool
    if usage is not None:
        step["usage"] = {
            "input_tokens": usage[0],
            "output_tokens": usage[1],
            "cache_read_tokens": usage[2],
        }
    return {"event": "step_update", "step_update": step}


def _result(*, status: str = "SUCCESS", error: str = "") -> dict[str, object]:
    result: dict[str, object] = {"conversation_id": "c1", "status": status}
    if error:
        result["error"] = error
    # Cumulative over the session: never what one turn cost.
    result["usage"] = {"input_tokens": 999_999, "output_tokens": 999_999}
    return {"event": "result", "result": result}


def _said(text: str, *, tokens: tuple[int, int] = (1, 1)) -> list[object]:
    """Return the lines of a turn in which the model says ``text``."""
    return [
        _step("user_input"),
        _step("agent_response", text=text, usage=(*tokens, 0)),
        _result(),
    ]


async def _stream(
    model: GoogleCLIModel,
    request: Request,
) -> list[TextChunk | Response]:
    return [event async for event in model.stream(request)]


def _response(events: Sequence[TextChunk | Response]) -> Response:
    response = events[-1]
    assert isinstance(response, Response)
    return response


def _png(width: int, height: int) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (width, height), "red").save(out, format="PNG")
    return out.getvalue()


@dataclass(slots=True, kw_only=True)
class _Stub:
    """The directory the stub ``agy`` runs from, and what it recorded there."""

    dir: Path
    next_turn: int = 0

    def script(self, *turns: Sequence[object]) -> None:
        """Write each turn, as the lines the stub prints for the next line it reads."""
        for turn in turns:
            lines = [
                line if isinstance(line, str) else json_dumps(line) for line in turn
            ]
            (self.dir / f"turn.{self.next_turn}").write_text("\n".join(lines) + "\n")
            self.next_turn += 1

    def said(self) -> list[dict[str, object]]:
        path = self.dir / "said"
        lines = path.read_text().splitlines() if path.exists() else []
        return [json_loads(line.partition(" ")[2], dict[str, object]) for line in lines]

    def processes(self) -> list[Path]:
        return sorted(self.dir.glob("argv.*"))

    def only(self, kind: str) -> str:
        (path,) = self.dir.glob(f"{kind}.*")
        return (
            path.read_text().removesuffix("\n") if kind == "cwd" else path.read_text()
        )

    def copied(self) -> Path:
        """Return the copy of the ``.gemini`` the only process started with."""
        (path,) = self.dir.glob("gemini.*")
        return path


@dataclass(slots=True, kw_only=True)
class _Mount:
    """Tools offered to the stub; a call answered writes the file it waits for."""

    stub: _Stub
    url: str = "http://127.0.0.1:1/mount"
    calls: asyncio.Queue[ToolCall] = field(default_factory=asyncio.Queue[ToolCall])
    offered: list[tuple[str, ...]] = field(default_factory=list[tuple[str, ...]])
    answered: list[tuple[ToolCall, AnyToolResult]] = field(
        default_factory=list[tuple[ToolCall, "AnyToolResult"]],
    )
    late: bool = False
    """Whether a held call has stopped waiting, so answering it times out."""

    def offer(self, tools: Sequence[Tool]) -> None:
        self.offered.append(tuple(tool.name for tool in tools))

    async def called(self) -> ToolCall:
        return await self.calls.get()

    async def answer(self, call: ToolCall, result: AnyToolResult) -> None:
        if self.late:
            raise TimeoutError
        (self.stub.dir / f"answered.{len(self.answered)}").touch()
        self.answered.append((call, result))

    def cancel(self) -> None:
        return


class _ToolServer(AsyncResource):
    """One mount, so a test can read what reached it."""

    def __init__(self, *, mounted: _Mount) -> None:
        self.mounted = mounted

    async def mount(self) -> _Mount:
        return self.mounted

    async def unmount(self, mount: ToolMount) -> None:
        del mount


class _Resources(AsyncResource):
    """Real process pools, beside a tool server that runs no MCP."""

    def __init__(self, *, stub: _Stub) -> None:
        self._shared = SharedResources()
        self.mount = _Mount(stub=stub)
        self.server = _ToolServer(mounted=self.mount)

    @property
    def http(self) -> httpx2.AsyncClient:
        return self._shared.http

    def process_pool(self, name: str) -> ProcessPool:
        assert name == "agy"
        return self._shared.process_pool(name)

    @property
    def tool_server(self) -> _ToolServer:
        return self.server

    @override
    async def aclose(self) -> None:
        await self._shared.aclose()

    @override
    async def __aenter__(self) -> Self:
        return self


@dataclass(frozen=True, slots=True, kw_only=True)
class _Echo:
    """A tool the model may call; the caller, not agy, would run it."""

    name: str = "Echo"
    description: str = "Say the text back."
    schema: Mapping[str, PlainTree] = field(
        default_factory=lambda: MappingProxyType(
            {"type": "object", "properties": {"text": {"type": "string"}}},
        ),
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
    ) -> AnyToolResult:
        del context, arguments
        return UncategorizedToolResult(call_id="")


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
