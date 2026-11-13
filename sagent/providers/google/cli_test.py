"""Tests for Gemini's startup, conversation reuse, and in-memory stream parsing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from functools import partial
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock, Mock, call

import asyncio
import os
import stat
import tempfile

from treekle import PlainTree, json_dumps, json_loads

import pytest
import pytest_asyncio

from sagent.lib.posix.child import Child
from sagent.model.cost import (
    PriceCatalog,
    PriceCondition,
    TokenCost,
    TokenCount,
    TokenPrice,
)
from sagent.model.exceptions import (
    AuthError,
    DisconnectedError,
    RateLimitedError,
)
from sagent.model.types import Request, Response, TextChunk
from sagent.providers.google import cli
from sagent.providers.google.cli import GoogleCLI, GoogleCLIModel
from sagent.session.types import (
    AgentToAgentMessage,
    AssistantMessage,
    Attachment,
    ContextClear,
    SystemMessage,
    Thinking,
    ToolCall,
    ToolResult,
    TurnContext,
    UncategorizedToolResult,
    UserMessage,
)
from sagent.tools.display import ToolDisplay
from sagent.tools.types import ToolOutput, ToolStreamingResult, whole_output


if TYPE_CHECKING:
    from asyncio.subprocess import Process
    from collections.abc import AsyncIterator, Awaitable, Callable

    from sagent.model.capability import ThinkingEffort
    from sagent.providers.types import ToolMount
    from sagent.session.types import SessionRecord
    from sagent.tools.types import ToolContext


_INIT = {"event": "init", "conversation_id": "c1", "init": {"tools": ["view_file"]}}


@pytest.fixture(autouse=True)
def private_temp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every agy home under ``tmp_path``, not the shared temp directory."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))


@pytest_asyncio.fixture
async def agy_stream(
    tmp_path: Path,
) -> AsyncIterator[tuple[cli._Agy, asyncio.StreamReader]]:
    stdout = asyncio.StreamReader()
    agy = cli._Agy(
        Mock(stdout=stdout),
        home=tmp_path,
        started_with=(),
        quiet_sec=GoogleCLIModel.quiet_sec,
    )
    try:
        yield agy, stdout
    finally:
        tasks: list[Awaitable[object]] = [
            task for task in (agy._line, agy._call) if task is not None
        ]
        agy.close()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("headless", [True, False])
@pytest.mark.parametrize(
    "status",
    [0, 7, None],
    ids=["success", "failure", "cancelled"],
)
async def test_native_login_waits_in_the_child_context_before_reporting_status(
    monkeypatch: pytest.MonkeyPatch,
    headless: bool,
    status: int | None,
) -> None:
    waiting = asyncio.Event()
    finished = asyncio.Future[int]()

    async def wait() -> int:
        waiting.set()
        return await finished

    waited = AsyncMock(side_effect=wait)
    child = Mock(process=Mock(wait=waited))
    entered = AsyncMock(return_value=child)
    exited = AsyncMock(return_value=None)
    started = AsyncMock(
        return_value=MagicMock(__aenter__=entered, __aexit__=exited),
    )
    monkeypatch.setattr(Child, "start", started)
    provider = GoogleCLI(resources=Mock())
    running = asyncio.create_task(provider.login(headless=headless))
    try:
        await waiting.wait()
        started.assert_awaited_once_with("agy", "acp", "--login")
        entered.assert_awaited_once_with()
        waited.assert_awaited_once_with()
        exited.assert_not_awaited()
        assert not running.done()
        if status is None:
            running.cancel()
            with pytest.raises(asyncio.CancelledError) as raised:
                await running
            exited.assert_awaited_once()
            assert exited.await_args is not None
            assert exited.await_args.args[:2] == (asyncio.CancelledError, raised.value)
        else:
            finished.set_result(status)
            if status:
                with pytest.raises(
                    AuthError,
                    match=r"^agy sign-in failed \(exit 7\)\.$",
                ):
                    await running
            else:
                await running
            exited.assert_awaited_once_with(None, None, None)
    finally:
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)


@pytest.mark.asyncio
async def test_agy_runs_as_our_agent_in_a_home_of_its_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Our prompt is its agent's, our tools its only ones; its login is its own."""
    monkeypatch.setenv("GEMINI_API_KEY", "leak")
    monkeypatch.setenv("GOOGLE_API_KEY", "leak")
    monkeypatch.setenv("KEEP_ME", "1")
    drained = AsyncMock()
    written = Mock()
    process = Mock(stdin=Mock(spec=asyncio.StreamWriter, write=written, drain=drained))
    acquired = AsyncMock(return_value=process)
    released = AsyncMock()
    pool = Mock(acquire=acquired, release=released)
    pools = Mock(return_value=pool)
    mount = Mock(url="http://127.0.0.1:1/mount")
    mounted = AsyncMock(return_value=mount)
    unmounted = AsyncMock()
    resources = Mock(
        process_pool=pools,
        tool_server=Mock(mount=mounted, unmount=unmounted),
    )
    provider = GoogleCLI(resources=resources)
    model = provider["gemini-flash"]
    request = Request(
        system=("Be terse.", "", "Rules."),
        records=(
            UserMessage(content="first"),
            AssistantMessage(content="ok"),
            SystemMessage(content="mind the tests"),
            UserMessage(content="second"),
        ),
        tools=(_Echo(),),
    )
    try:
        agy = await model._start(request, started_with=("selected",))
        home = agy.home
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in {"GEMINI_API_KEY", "GOOGLE_API_KEY"}
        }
        env |= {"HOME": str(home), "USERPROFILE": str(home)}
        acquired.assert_awaited_once_with(
            [
                "/bin/sh",
                "-c",
                'cd "$0" && exec "$@"',
                str(home / "workdir"),
                "agy",
                "--agent",
                "sagent",
                "--input-format",
                "stream-json",
                "--output-format",
                "stream-json",
                "--disable-slash-commands",
                "--model",
                "gemini-3.8-flash-high",
            ],
            env=env,
        )
        mounted.assert_awaited_once_with()
        assert pools.call_args_list == [call("agy")]
        assert agy.process is process
        assert agy.started_with == ("selected",)
        assert provider._started == {model}
        assert home.name.startswith("sagent-agy-")
        written.assert_called_once_with(
            json_dumps(
                {
                    "event": "user",
                    "message": {
                        "content": "User: first\n\nAssistant: ok\n\n"
                        "<system-reminder>\nmind the tests\n</system-reminder>\n\nUser: second",
                    },
                },
            ).encode()
            + b"\n",
        )
        drained.assert_awaited_once_with()
        gemini = home / ".gemini"
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
            "mcpServers": {"sagent": {"serverUrl": "http://127.0.0.1:1/mount"}},
        }
        assert json_loads(
            (gemini / "antigravity-cli" / "settings.json").read_text(),
        ) == {
            "permissions": {"allow": ["mcp(sagent/*)"]},
            "showTips": False,
            "showFeedbackSurvey": False,
        }
    finally:
        await provider.aclose()
    released.assert_awaited_once_with(process)
    unmounted.assert_awaited_once_with(mount)
    assert not await asyncio.to_thread(home.exists)


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_the_effort_is_the_model_id_s_suffix(effort: ThinkingEffort) -> None:
    """``agy`` refuses ``--effort`` beside a suffixed id: the suffix is the effort."""
    model = GoogleCLI(resources=Mock())["gemini-flash"]
    model.settings = replace(model.settings, thinking_effort=effort)
    assert model.command(Path("/workspace"))[-2:] == [
        "--model",
        f"gemini-3.8-flash-{effort}",
    ]


@pytest.mark.asyncio
async def test_a_turn_streams_the_reply_and_sums_its_step_usage(
    agy_stream: tuple[cli._Agy, asyncio.StreamReader],
) -> None:
    agy, stdout = agy_stream
    lines = [
        _INIT,
        _step("user_input"),
        _step("agent_response", text="good ", state="ACTIVE"),
        _step("agent_response", text="day", usage=(30, 5, 20)),
        _step("checkpoint", usage=(7, 1, 0)),
        _result(),
    ]
    stdout.feed_data("".join(json_dumps(line) + "\n" for line in lines).encode())
    stdout.feed_eof()
    turn = cli._Turn()

    assert [chunk async for chunk in agy.read(turn, mount=Mock())] == [
        TextChunk(text="good "),
        TextChunk(text="day"),
    ]
    response = turn.response(capability=GoogleCLI.models()["gemini-flash-3.8"])
    assert response.records == (AssistantMessage(content="good day"),)
    assert response.stop_reason == "end_turn"
    # Each step's usage, summed: input, then output, then what the cache served.
    assert response.tokens == TokenCount(request=37, response=6, cache_read=20)


@pytest.mark.parametrize(
    "usage",
    [
        pytest.param({"output_tokens": 2}, id="missing-input-tokens"),
        pytest.param(
            {"input_tokens": None, "output_tokens": 2},
            id="null-input-tokens",
        ),
    ],
)
def test_missing_or_null_input_tokens_count_as_zero(
    usage: dict[str, int | None],
) -> None:
    agy = cli._Agy(
        Mock(),
        home=Path("/nowhere"),
        started_with=(),
        quiet_sec=GoogleCLIModel.quiet_sec,
    )
    turn = cli._Turn()
    assert agy._step(
        {"step_type": "agent_response", "text_delta": "hi", "usage": usage},
        turn=turn,
    ) == TextChunk(text="hi")
    assert turn.text == ["hi"]
    assert agy._spent == TokenCount(request=0, response=2)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("held", "eligible", "worn", "answered", "restart"),
    [
        (False, True, False, True, False),
        (True, True, False, True, False),
        (True, True, False, False, True),
        (True, False, False, True, True),
        (True, True, True, True, False),
        (False, True, True, True, True),
    ],
    ids=[
        "new-message",
        "held-result",
        "late-result",
        "mixed-message",
        "held-worn",
        "worn",
    ],
)
async def test_stream_reuses_only_a_conversation_that_can_continue(
    monkeypatch: pytest.MonkeyPatch,
    *,
    held: bool,
    eligible: bool,
    worn: bool,
    answered: bool,
    restart: bool,
) -> None:
    result = ToolResult(
        call_id="c1",
        state="done",
        stdout=("x",),
        result=UncategorizedToolResult(),
    )
    addition = (result,) if held else (UserMessage(content="and now?"),)
    unseen = Mock(return_value=addition if eligible else None)
    answered_call = AsyncMock(return_value=answered)
    prompted = AsyncMock()
    remembered = Mock()
    previous_read = Mock(side_effect=_reply)
    fresh_read = Mock(side_effect=_reply)
    old = Mock(
        held=(ToolCall(call_id="c1", name="Echo"),) if held else (),
        unseen=unseen,
        answer=answered_call,
        prompt=prompted,
        read=previous_read,
        saw=remembered,
    )
    fresh_remembered = Mock()
    fresh = Mock(read=fresh_read, saw=fresh_remembered)
    offered = Mock()
    mount = Mock(offer=offered)
    model = GoogleCLI(resources=Mock())["gemini-flash"]
    model._agy = old
    model._mount = mount

    async def start(request: Request, *, started_with: tuple[str, ...]) -> Mock:
        del request, started_with
        model._agy = fresh
        return fresh

    started = AsyncMock(side_effect=start)
    worn_out = Mock(return_value=worn)
    counted = AsyncMock(return_value=17)
    monkeypatch.setattr(model, "_start", started)
    monkeypatch.setattr(model, "_worn", worn_out)
    monkeypatch.setattr(model, "count", counted)
    request = Request(records=(*_HELLO, *addition), tools=(_Echo(),))

    chunk, response = [event async for event in model._stream(request)]

    assert chunk == TextChunk(text="reply")
    assert isinstance(response, Response)
    assert response.records == (AssistantMessage(content="reply"),)
    assert response.tokens == TokenCount(request=15, response=2, cache_read=10)
    assert response.stop_reason == "end_turn"
    identity = (cli._digest(request.tools), "gemini-3.8-flash-high")
    unseen.assert_called_once_with(request.records, started_with=identity)
    offered.assert_called_once_with(request.tools)
    assert started.await_args_list == (
        [call(request, started_with=identity)] if restart else []
    )
    assert answered_call.await_args_list == (
        [call(addition, mount=mount)] if held and eligible else []
    )
    assert prompted.await_args_list == (
        [call("and now?")] if not held and not restart else []
    )
    assert worn_out.call_args_list == ([call(old)] if eligible and not held else [])
    counted.assert_awaited_once_with(request)
    assert previous_read.call_count == (0 if restart else 1)
    assert fresh_read.call_count == (1 if restart else 0)
    selected = fresh_read if restart else previous_read
    assert selected.call_args is not None
    assert selected.call_args.kwargs == {"mount": mount}
    assert model._agy is (fresh if restart else old)
    expected = [call(request.records, response=response, context_tokens=17)]
    assert remembered.call_args_list == ([] if restart else expected)
    assert fresh_remembered.call_args_list == (expected if restart else [])


@pytest.mark.asyncio
async def test_stdout_keeps_streaming_while_a_tool_call_has_not_arrived(
    agy_stream: tuple[cli._Agy, asyncio.StreamReader],
) -> None:
    agy, stdout = agy_stream
    calls = asyncio.Queue[ToolCall]()
    received = AsyncMock(side_effect=calls.get)
    mount = Mock(called=received)
    lines = [
        _step("tool", state="ACTIVE", tool="call_mcp_tool"),
        _step("agent_response", text="waiting", usage=(7, 2, 0)),
    ]
    stdout.feed_data("".join(json_dumps(line) + "\n" for line in lines).encode())
    turn = cli._Turn()
    stream = agy.read(turn, mount=mount)
    assert await anext(stream) == TextChunk(text="waiting")
    received.assert_awaited_once_with()
    assert agy._call is not None
    assert not agy._call.done()
    call = ToolCall(call_id="c1", name="Echo", arguments={"text": "x"})
    calls.put_nowait(call)
    assert [chunk async for chunk in stream] == []
    assert agy._spent == TokenCount()

    response = turn.response(capability=GoogleCLI.models()["gemini-flash-3.8"])
    assert response.records == (AssistantMessage(content="waiting"), call)
    assert response.stop_reason == "tool_use"
    assert response.tokens == TokenCount(request=7, response=2)
    stdout.feed_data(json_dumps(_result()).encode() + b"\n")
    assert await agy._next(mount) == _result()
    assert agy._call is None
    received.assert_awaited_once_with()


def test_another_system_prompt_cannot_reuse_the_conversation() -> None:
    agy = cli._Agy(
        Mock(),
        home=Path("/nowhere"),
        started_with=("A", "tools", "model"),
        quiet_sec=GoogleCLIModel.quiet_sec,
    )
    agy.saw(
        _HELLO,
        response=Response(
            records=(AssistantMessage(content="one"),),
            tokens=TokenCount(),
            spend=TokenCost(),
            stop_reason="end_turn",
        ),
        context_tokens=0,
    )
    records = (*agy.seen.records, UserMessage(content="again"))

    assert agy.unseen(records, started_with=("A", "tools", "model")) == [
        UserMessage(content="again"),
    ]
    assert agy.unseen(records, started_with=("B", "tools", "model")) is None


def test_a_fresh_agy_hears_the_whole_conversation_as_a_transcript() -> None:
    records = (
        ContextClear(summary="earlier", history=(UserMessage(content="before"),)),
        ContextClear(history=(UserMessage(content="kept"),)),
        AgentToAgentMessage(content="from a", sender="lead"),
        AgentToAgentMessage(content="anon"),
        Thinking(content="hmm"),
        AssistantMessage(content="ok"),
        ToolCall(call_id="c0", name="Echo", arguments={"text": "x"}),
        ToolResult(
            call_id="c0",
            state="done",
            stdout=("x",),
            result=UncategorizedToolResult(),
        ),
        TurnContext(),
        TurnContext(cwd="/w"),
        UserMessage(content="go"),
    )

    assert "\n\n".join(cli._transcript(records)) == (
        "Summary of the conversation so far:\nearlier\n\n"
        "User: before\n\nUser: kept\n\nAgent lead: from a\n\nAgent: anon\n\n"
        "Assistant: ok\n\n"
        'Assistant called Echo as c0: {"text": "x"}\n\nc0 returned: x\n\n'
        "<system-reminder>\nWorking directory: /w\n</system-reminder>\n\n"
        "User: go"
    )


def test_the_next_turn_says_each_new_message_its_own_way() -> None:
    records = (
        AgentToAgentMessage(content="hi", sender="lead"),
        AgentToAgentMessage(content="plain"),
        TurnContext(),
        TurnContext(cwd="/w"),
        UserMessage(content="go"),
    )

    assert cli._said_text(records) == (
        "[from lead]: hi\n\nplain\n\n"
        "<system-reminder>\nWorking directory: /w\n</system-reminder>\n\ngo"
    )
    agy = cli._Agy(
        cast("Process", object()),
        home=Path("/nowhere"),
        started_with=(),
        quiet_sec=1.0,
    )
    agy.saw(
        _HELLO,
        response=Response(
            records=(AssistantMessage(content="one"),),
            tokens=TokenCount(),
            spend=TokenCost(),
            stop_reason="end_turn",
        ),
        context_tokens=0,
    )
    assert agy.unseen((*agy.seen.records, *records), started_with=()) == list(records)


def test_a_history_it_never_said_cannot_reuse_the_conversation() -> None:
    """Only results it waits for, or new user lines, go to a running ``agy``."""
    agy = cli._Agy(
        Mock(),
        home=Path("/nowhere"),
        started_with=(),
        quiet_sec=GoogleCLIModel.quiet_sec,
    )
    agy.saw(
        _HELLO,
        response=Response(
            records=(AssistantMessage(content="one"),),
            tokens=TokenCount(),
            spend=TokenCost(),
            stop_reason="end_turn",
        ),
        context_tokens=0,
    )
    records = (
        *agy.seen.records,
        AssistantMessage(content="put in"),
        UserMessage(content="again"),
    )

    assert agy.unseen(records, started_with=()) is None


@pytest.mark.parametrize(
    ("additions", "accepted"),
    [
        (
            (
                ToolResult(
                    call_id="c1",
                    state="done",
                    stdout=("x",),
                    result=UncategorizedToolResult(),
                ),
            ),
            True,
        ),
        (
            (
                SystemMessage(content="note"),
                ToolResult(
                    call_id="c1",
                    state="done",
                    stdout=("x",),
                    result=UncategorizedToolResult(),
                ),
                TurnContext(cwd="/work"),
            ),
            True,
        ),
        (
            (
                ToolResult(
                    call_id="wrong",
                    state="done",
                    result=UncategorizedToolResult(),
                ),
            ),
            False,
        ),
        ((), False),
        (
            (
                ToolResult(
                    call_id="c1",
                    state="done",
                    result=UncategorizedToolResult(),
                ),
                ToolResult(
                    call_id="c1",
                    state="done",
                    result=UncategorizedToolResult(),
                ),
            ),
            False,
        ),
        (
            (
                ToolResult(
                    call_id="c1",
                    state="done",
                    result=UncategorizedToolResult(),
                ),
                UserMessage(content="and this"),
            ),
            False,
        ),
    ],
    ids=["exact", "notes", "wrong-call", "missing", "duplicate", "new-message"],
)
def test_a_held_turn_accepts_only_its_exact_results_and_notes(
    additions: tuple[SessionRecord, ...],
    accepted: bool,
) -> None:
    agy = cli._Agy(
        Mock(),
        home=Path("/nowhere"),
        started_with=(),
        quiet_sec=GoogleCLIModel.quiet_sec,
    )
    agy.saw(
        _HELLO,
        response=Response(
            records=(ToolCall(call_id="c1", name="Echo"),),
            tokens=TokenCount(),
            spend=TokenCost(),
            stop_reason="tool_use",
        ),
        context_tokens=0,
    )
    assert agy.unseen((*agy.seen.records, *additions), started_with=()) == (
        list(additions) if accepted else None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("line", ["warning: not json", "[1]"])
async def test_non_object_lines_are_passed_over(
    agy_stream: tuple[cli._Agy, asyncio.StreamReader],
    line: str,
) -> None:
    agy, stdout = agy_stream
    stdout.feed_data((line + "\n" + json_dumps(_result()) + "\n").encode())
    stdout.feed_eof()
    turn = cli._Turn()

    assert [chunk async for chunk in agy.read(turn, mount=Mock())] == []
    assert turn == cli._Turn()


def test_an_image_is_named_not_sent() -> None:
    """``agy`` takes text alone; the model is told an image was left out."""
    image = Attachment(mime_descriptor="image/png", data=b"not sent")
    assert cli._transcript((UserMessage(content="look", attachments=(image,)),)) == [
        "User: look\n[an image/png attachment, which agy cannot send]",
    ]


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
    agy_stream: tuple[cli._Agy, asyncio.StreamReader],
    error: str,
    kind: type[Exception],
) -> None:
    agy, stdout = agy_stream
    stdout.feed_data(json_dumps(_result(status="ERROR", error=error)).encode() + b"\n")
    stdout.feed_eof()
    with pytest.raises(kind, match=error):
        _ = [chunk async for chunk in agy.read(cli._Turn(), mount=Mock())]


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
        # Go spells a duration under a second in milliseconds.
        ("RESOURCE_EXHAUSTED (code 429): Resets in 500ms.", 0.5),
        ("RESOURCE_EXHAUSTED (code 429): Resets in 1m0.5s.", 60.5),
        ("RESOURCE_EXHAUSTED: quota exceeded", 0),
    ],
)
def test_a_spent_quota_waits_for_the_reset_agy_names(
    error: str,
    wait_sec: float,
) -> None:
    """Agy says when the quota resets; dropped, the wait was a backoff's guess."""
    failure = cli._failure({"status": "ERROR", "error": error})
    assert isinstance(failure, RateLimitedError)
    assert failure.retry_after_sec == wait_sec
    # Only a quota that names its reset is the login's, spent for every model.
    assert failure.limit == ("quota" if wait_sec else "")


@pytest.mark.asyncio
@pytest.mark.parametrize("recheck", [False, True])
async def test_stream_forwards_pacing_and_stops_a_failed_or_cancelled_turn(
    monkeypatch: pytest.MonkeyPatch,
    recheck: bool,
) -> None:
    failure = asyncio.CancelledError() if recheck else RateLimitedError("quota")

    async def read(turn: cli._Turn, *, mount: ToolMount) -> AsyncIterator[TextChunk]:
        del turn, mount
        yield TextChunk(text="partial")
        raise failure

    async def pace(
        operation: Callable[[], AsyncIterator[TextChunk | Response]],
        **kwargs: object,
    ) -> AsyncIterator[TextChunk | Response]:
        del kwargs
        async for event in operation():
            yield event

    gate = Mock()
    offered = Mock()
    mount = Mock(offer=offered)
    model = GoogleCLI(resources=Mock(gate=gate))["gemini-flash"]
    started = AsyncMock(return_value=Mock(read=read))
    stopped = AsyncMock()
    mounted = AsyncMock(return_value=mount)
    paced = Mock(side_effect=pace)
    monkeypatch.setattr(model, "_start", started)
    monkeypatch.setattr(model, "_stop", stopped)
    monkeypatch.setattr(model, "_mounted", mounted)
    monkeypatch.setattr(cli, "retried", paced)
    request = Request(
        system=("rules",),
        records=_HELLO,
        tools=(_Echo(),),
        recheck_quota=recheck,
    )
    stream = model.stream(request)

    assert await anext(stream) == TextChunk(text="partial")
    with pytest.raises(type(failure)) as raised:
        await anext(stream)
    assert raised.value is failure
    stopped.assert_awaited_once_with()
    started.assert_awaited_once_with(
        request,
        started_with=("rules", cli._digest(request.tools), "gemini-3.8-flash-high"),
    )
    offered.assert_called_once_with(request.tools)
    mounted.assert_awaited_once_with()
    paced.assert_called_once()
    assert paced.call_args is not None
    assert paced.call_args.kwargs == {
        "gate": gate,
        "key": "google-cli@default/gemini-flash-3.8",
        "retries": 0,
        "recheck_quota": recheck,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["", "half"])
async def test_agy_exiting_before_its_result_is_a_disconnect(
    agy_stream: tuple[cli._Agy, asyncio.StreamReader],
    text: str,
) -> None:
    agy, stdout = agy_stream
    if text:
        stdout.feed_data(
            (
                json_dumps(_step("agent_response", text=text, state="ACTIVE")) + "\n"
            ).encode(),
        )
    stdout.feed_eof()
    turn = cli._Turn()
    stream = agy.read(turn, mount=Mock())
    if text:
        assert await anext(stream) == TextChunk(text=text)
    with pytest.raises(DisconnectedError, match=r"^agy exited before its answer$"):
        await anext(stream)
    assert turn.text == ([text] if text else [])


@pytest.mark.asyncio
async def test_agy_quiet_too_long_is_a_disconnect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = asyncio.StreamReader()
    agy = cli._Agy(
        Mock(stdout=stdout),
        home=tmp_path,
        started_with=(),
        quiet_sec=GoogleCLIModel.quiet_sec,
    )
    line = agy._line = asyncio.create_task(stdout.readline())
    waited = AsyncMock(return_value=(set(), set()))
    monkeypatch.setattr(asyncio, "wait", waited)
    try:
        with pytest.raises(
            DisconnectedError,
            match=r"^agy went quiet for 600 seconds$",
        ):
            await agy._next(Mock())
        assert agy._line is line
        waited.assert_awaited_once_with(
            {line},
            timeout=600.0,
            return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        agy.close()
        await asyncio.gather(line, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("late", [False, True])
async def test_answer_forwards_only_results_in_record_order(late: bool) -> None:
    first = ToolCall(call_id="first", name="Echo")
    second = ToolCall(call_id="second", name="Echo")
    first_result = ToolResult(
        call_id="first",
        state="done",
        stdout=("one",),
        result=UncategorizedToolResult(),
    )
    second_result = ToolResult(
        call_id="second",
        state="done",
        stdout=("two",),
        result=UncategorizedToolResult(),
    )
    agy = cli._Agy(
        Mock(),
        home=Path("/nowhere"),
        started_with=(),
        quiet_sec=1.0,
    )
    agy.saw(
        _HELLO,
        response=Response(
            records=(first, second),
            tokens=TokenCount(),
            spend=TokenCost(),
            stop_reason="tool_use",
        ),
        context_tokens=0,
    )
    answered = AsyncMock(side_effect=TimeoutError if late else None)

    assert await agy.answer(
        (
            SystemMessage(content="note"),
            second_result,
            TurnContext(cwd="/work"),
            first_result,
        ),
        mount=Mock(answer=answered),
    ) is (not late)

    expected = [call(second, second_result), call(first, first_result)]
    assert answered.await_args_list == (expected[:1] if late else expected)


def test_a_tool_schema_changes_the_session_identity() -> None:
    original = _Echo()
    wider = _Echo(schema=MappingProxyType({"type": "object"}))

    assert cli._digest((original,)) == cli._digest((_Echo(),))
    assert cli._digest((original,)) != cli._digest((wider,))


@pytest.mark.asyncio
async def test_steps_that_say_nothing_or_are_no_reply_add_no_text(
    agy_stream: tuple[cli._Agy, asyncio.StreamReader],
) -> None:
    agy, stdout = agy_stream
    lines = [
        _INIT,
        _step("agent_response", state="ACTIVE"),
        _step("planner", text="a plan"),
        # A tool step that is done asks nothing of our server.
        _step("tool", tool="view_file"),
        {
            "event": "step_update",
            "step_update": {
                "step_type": "checkpoint",
                "usage": {"input_tokens": 4},
            },
        },
        _step("user_input"),
        _step("agent_response", text="hi", usage=(1, 1, 0)),
        _result(),
    ]
    stdout.feed_data("".join(json_dumps(line) + "\n" for line in lines).encode())
    stdout.feed_eof()
    called = AsyncMock()
    answered = AsyncMock()
    turn = cli._Turn()

    assert [
        chunk
        async for chunk in agy.read(
            turn,
            mount=Mock(called=called, answer=answered),
        )
    ] == [TextChunk(text="hi")]
    assert turn.tokens == TokenCount(request=5, response=1)
    called.assert_not_awaited()
    answered.assert_not_awaited()


@pytest.mark.asyncio
async def test_closing_lets_go_of_held_calls_and_tolerates_a_home_gone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    released = AsyncMock()
    pools = Mock(return_value=Mock(release=released))
    cancelled = Mock()
    mount = Mock(cancel=cancelled)
    mounted = AsyncMock(return_value=mount)
    unmounted = AsyncMock()
    provider = GoogleCLI(
        resources=Mock(
            process_pool=pools,
            tool_server=Mock(mount=mounted, unmount=unmounted),
        ),
    )
    model = provider["gemini-flash"]
    await model.aclose()
    await model._mounted()
    process = Mock()
    agy = cli._Agy(
        process,
        home=tmp_path / "already-gone",
        started_with=(),
        quiet_sec=GoogleCLIModel.quiet_sec,
    )
    closed = Mock(wraps=agy.close)
    monkeypatch.setattr(agy, "close", closed)
    model._agy = agy
    provider.started(model)

    await provider.aclose()

    cancelled.assert_called_once_with()
    closed.assert_called_once_with()
    released.assert_awaited_once_with(process)
    pools.assert_called_once_with("agy")
    unmounted.assert_awaited_once_with(mount)
    assert model._agy is None
    assert model._mount is None


@pytest.mark.parametrize(
    ("turns", "context", "worn"),
    [
        (99, 0, False),
        (100, 0, True),
        # Half a 1,000-token window, and one token over it.
        (0, 500, False),
        (0, 501, True),
    ],
)
def test_an_agy_is_worn_after_its_turns_or_half_its_window(
    turns: int,
    context: int,
    worn: bool,
) -> None:
    model = GoogleCLI()["gemini-flash"]
    model._capability = replace(
        model.capability,
        limits=replace(model.capability.limits, max_request_tokens=1_000),
    )
    seen = SimpleNamespace(seen=SimpleNamespace(turns=turns, context_tokens=context))

    assert model._worn(cast("cli._Agy", seen)) is worn


def test_an_agy_counts_each_turn_it_finishes() -> None:
    agy = cli._Agy(
        # ``saw`` reads no process.
        cast("Process", object()),
        home=Path("/nowhere"),
        started_with=(),
        quiet_sec=1.0,
    )
    said = Response(
        records=(AssistantMessage(content="x"),),
        tokens=TokenCount(),
        spend=TokenCost(),
        stop_reason="end_turn",
    )
    calling = replace(said, records=(ToolCall(call_id="c", name="Echo"),))

    turns: list[int] = []
    for response in (said, calling, said, said):
        agy.saw(_HELLO, response=response, context_tokens=7)
        turns.append(agy.seen.turns)

    # A reply that stops at calls goes on in the next stream: it finishes no turn.
    assert turns == [1, 1, 2, 3]
    assert agy.seen.context_tokens == 7


@pytest.mark.parametrize(
    ("error", "kind", "message"),
    [
        (
            {"error": "Please sign in"},
            AuthError,
            "Please sign in; run agy once to sign in",
        ),
        ({"error": "resource_exhausted"}, RateLimitedError, "resource_exhausted"),
        ({"error": "HTTP 429"}, RateLimitedError, "HTTP 429"),
        ({"status": "CANCELLED"}, RuntimeError, "agy: CANCELLED"),
        ({}, RuntimeError, "agy: agy failed"),
    ],
)
def test_a_failure_is_what_agy_says_or_its_status(
    error: dict[str, object],
    kind: type[Exception],
    message: str,
) -> None:
    failure = cli._failure(error)

    assert type(failure) is kind
    assert str(failure) == message


def test_a_home_is_private_and_its_agent_has_no_tools_section_without_tools(
    tmp_path: Path,
) -> None:
    cli._furnish(tmp_path, system=("Be terse.",), tools=(), url="http://x")

    for directory in (
        tmp_path / "workdir",
        tmp_path / ".gemini" / "config" / "agents" / "sagent",
        tmp_path / ".gemini" / "antigravity-cli",
    ):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700, directory
    agent = tmp_path / ".gemini" / "config" / "agents" / "sagent" / "agent.md"
    assert agent.read_text().endswith("---\nBe terse.\n")


def test_a_tool_schema_keeps_its_characters(tmp_path: Path) -> None:
    accented = _Echo(schema=MappingProxyType({"title": "café"}))
    cli._furnish(tmp_path, system=(), tools=(accented,), url="http://x")

    agent = tmp_path / ".gemini" / "config" / "agents" / "sagent" / "agent.md"
    assert 'Input schema: {"title":"café"}' in agent.read_text()


@pytest.mark.parametrize(
    ("record", "said"),
    [
        (
            UserMessage(
                attachments=(Attachment(mime_descriptor="image/png", data=b""),),
            ),
            "\n[an image/png attachment, which agy cannot send]",
        ),
        (AgentToAgentMessage(sender="lead"), "[from lead]: "),
        (AgentToAgentMessage(), ""),
    ],
)
def test_a_message_with_no_words_says_none(record: UserMessage, said: str) -> None:
    assert cli._said(record) == said


@pytest.mark.asyncio
async def test_a_request_counts_as_its_parts_joined() -> None:
    model = GoogleCLI()["gemini-flash"]
    request = Request(system=("aaaa", "bbbb"), records=(UserMessage(content="cc"),))
    text = "aaaa\n\nbbbb\n\nUser: cc"

    assert await model.count(request) == int(
        len(text) / model.capability.approx_chars_per_token,
    )


def test_a_stream_is_costed_at_the_card_of_its_utc_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Evening:
        @classmethod
        def now(cls, tz: object = None) -> datetime:
            if tz is UTC:
                return datetime(2026, 10, 7, 1, tzinfo=UTC)
            return datetime(2026, 10, 6, 18)  # noqa: DTZ001 -- the local wall clock.

    capability = replace(
        GoogleCLI()["gemini-flash"].capability,
        prices=PriceCatalog(
            prices={
                PriceCondition(): TokenPrice(response=1.0),
                PriceCondition(effective_date=date(2026, 10, 7)): TokenPrice(
                    response=2.0,
                ),
            },
            window=1_000,
        ),
    )
    monkeypatch.setattr(cli, "datetime", _Evening)
    turn = cli._Turn(tokens=TokenCount(response=1_000_000))

    assert turn.response(capability=capability).spend.response == 2.0


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
    picture = Attachment(mime_descriptor="image/png", data=b"not sent")
    named = "[an image/png attachment, which agy cannot send]"
    assert model.attachment_tokens(picture) == int(
        len(named) / model.capability.approx_chars_per_token,
    )


@pytest.mark.asyncio
async def test_a_request_counts_as_the_text_agy_is_sent() -> None:
    model = GoogleCLI()["gemini-flash"]
    picture = Attachment(mime_descriptor="image/png", data=b"not sent")
    plain = Request(system=("S",), records=(UserMessage(content="look"),))
    pictured = Request(
        system=("S",),
        records=(UserMessage(content="look", attachments=(picture,)),),
    )

    grown = await model.count(pictured) - await model.count(plain)

    assert 0 < grown <= model.attachment_tokens(picture) + 1


@pytest.mark.asyncio
@pytest.mark.parametrize("owned", [False, True], ids=["shared", "owned"])
async def test_a_provider_closes_started_models_before_only_its_own_resources(
    monkeypatch: pytest.MonkeyPatch,
    owned: bool,
) -> None:
    closed: list[str] = []
    close_resources = AsyncMock(side_effect=partial(closed.append, "resources"))
    resources = Mock(aclose=close_resources)
    built = Mock(return_value=resources)
    monkeypatch.setattr(cli, "SharedResources", built)
    provider = GoogleCLI() if owned else GoogleCLI(resources=resources)
    close_model = AsyncMock(side_effect=partial(closed.append, "model"))
    model = Mock(spec=GoogleCLIModel, aclose=close_model)
    provider.started(model)

    assert provider.resources is resources
    assert built.call_args_list == ([call()] if owned else [])
    close_model.assert_not_awaited()
    close_resources.assert_not_awaited()

    await provider.aclose()

    close_model.assert_awaited_once_with()
    assert close_resources.await_args_list == ([call()] if owned else [])
    assert closed == (["model", "resources"] if owned else ["model"])


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


async def _reply(turn: cli._Turn, *, mount: ToolMount) -> AsyncIterator[TextChunk]:
    del mount
    turn.text.append("reply")
    turn.tokens = TokenCount(request=15, response=2, cache_read=10)
    yield TextChunk(text="reply")


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
        del context, arguments
        return ToolOutput(result=UncategorizedToolResult())


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
