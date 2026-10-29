"""Tests for slash commands: each changes the part of the agent it names."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING, override

import asyncio
import sys
import time

from treekle import PlainTree

import pytest

from sagent.agent.agent import Agent
from sagent.agent.team import Team
from sagent.model.capability import (
    ModelCapability,
    ModelLimits,
    ModelSettings,
    ThinkingEffort,
)
from sagent.model.cost import PriceCatalog
from sagent.model.exceptions import AuthError, UnknownModelError
from sagent.model.scripted import ScriptedModel, ScriptedProvider, call
from sagent.providers.catalog import ModelCatalog
from sagent.providers.table import ModelTable
from sagent.runtime import types
from sagent.session.recording import RecordingJournal
from sagent.session.types import (
    AgentToAgentMessage,
    AnyToolResult,
    UncategorizedToolResult,
    UserMessage,
)
from sagent.tools.display import DisplaySpec, ToolDisplay
from sagent.tui.prompt import Panes, Staged
from sagent.tui.render import Scrollback
from sagent.tui.settings.harvest import harvest
from sagent.tui.slash import Commands, headless_wanted


if TYPE_CHECKING:
    from pathlib import Path

    from sagent.providers.types import (
        Provider,
        ProviderFactory,
        ProviderSharedResources,
    )
    from sagent.tools.types import ToolContext


@dataclass(slots=True, kw_only=True)
class _Printer:
    """Records each call as ``(kind, text)``."""

    show_thinking: bool = True
    said: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])

    def line(self, text: str) -> None:
        self.said.append(("line", text))

    def note(self, text: str) -> None:
        self.said.append(("note", text))

    def user(self, text: str) -> None:
        self.said.append(("user", text))

    def agent(self, sender: str, recipient: str | None, text: str) -> None:
        to = "me" if recipient is None else recipient
        self.said.append(("agent", f"{sender}→{to}: {text}"))

    def markdown(self, text: str) -> None:
        self.said.append(("markdown", text))

    def thinking(self, text: str, *, opens: bool, closes: bool) -> None:
        del opens, closes
        self.said.append(("thinking", text))

    def call(self, label: str, *, display: ToolDisplay) -> None:
        del display
        self.said.append(("call", label))

    def output(self, text: str, *, display: ToolDisplay) -> None:
        del display
        self.said.append(("output", text))

    def diff(
        self,
        diff: str,
        *,
        path: str,
        lines: tuple[int, int] | None = None,
    ) -> None:
        del lines
        self.said.append(("diff", f"{path}\n{diff}"))

    def error(self, text: str) -> None:
        self.said.append(("error", text))

    def halt(self, text: str) -> None:
        self.said.append(("halt", text))


@dataclass(frozen=True, slots=True, kw_only=True)
class _Echo:
    name: str = "Echo"
    description: str = "Say the text back."
    schema: Mapping[str, PlainTree] = field(default_factory=dict[str, PlainTree])
    display: ToolDisplay = field(default_factory=ToolDisplay)

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        return f"Echo {arguments.get('text', '')}"

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
        del context
        return UncategorizedToolResult(call_id="", content=str(arguments.get("text")))


class _Plain:
    """A tool that is no dataclass, so its display cannot be replaced."""

    name = "Plain"
    description = "Nothing."
    display = ToolDisplay()

    @property
    def schema(self) -> Mapping[str, PlainTree]:
        schema: dict[str, PlainTree] = {}
        return schema

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        del arguments
        return "Plain"

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


def _windowed(
    model_id: str,
    window: int,
    *,
    efforts: frozenset[ThinkingEffort] = frozenset({"none"}),
) -> ModelCapability:
    """Return a model with a window and ``efforts``, at its defaults."""
    prices = PriceCatalog(window=window)
    return ModelCapability(
        model_id=model_id,
        limits=ModelLimits(max_request_tokens=window),
        prices=prices,
        thinking_effort=efforts,
        default=ModelSettings(context_size=prices["auto", "fullcontext"]),
    )


_DEEP = replace(
    _windowed("deep", 0, efforts=frozenset({"none", "low", "high"})),
    thinking_budget=frozenset({"none", "auto"}),
    thinking_output=frozenset({"none", "text"}),
)

_WIDE = _windowed("wide-1.0", 1_000_000, efforts=frozenset({"none", "high"}))


class _Picky(ScriptedProvider):
    """Serves only the names in ``serves``, refusing the rest as a vendor's table does."""

    def __init__(
        self,
        *,
        name: str,
        resources: ProviderSharedResources,
        serves: frozenset[str],
        capability: ModelCapability | None = None,
    ) -> None:
        super().__init__(name=name, resources=resources, capability=capability)
        self.serves = serves

    @override
    def __getitem__(self, model: str) -> ScriptedModel:
        if model not in self.serves:
            raise UnknownModelError(f"No model {model!r}.")
        return super().__getitem__(model)


class _Signing(_Picky):
    """A provider that signs in as ``account``, or fails to with ``error``."""

    def __init__(
        self,
        *,
        name: str,
        resources: ProviderSharedResources,
        account: str | None = None,
    ) -> None:
        super().__init__(
            name=name,
            resources=resources,
            serves=frozenset({"scripted", "signing"}),
        )
        self.account = account
        self.error: Exception | None = None
        self.signed_in = 0
        self.headless: list[bool] = []

    async def login(self, *, headless: bool = False) -> None:
        self.signed_in += 1
        self.headless.append(headless)
        if self.error is not None:
            raise self.error


def _wide(*, name: str, resources: ProviderSharedResources) -> Provider:
    return ScriptedProvider(
        name=name,
        resources=resources,
        models=ModelTable(rows=(_WIDE,)),
    )


def _per_account(account: str | None) -> dict[str, ProviderFactory]:
    """Return the desk's providers, ``signing`` signing in as ``account``."""
    return {
        "scripted": partial(_Picky, serves=frozenset({"scripted"})),
        "deep": partial(
            _Picky,
            serves=frozenset({"scripted", "deep"}),
            capability=_DEEP,
        ),
        "signing": partial(_Signing, account=account),
        "wide": _wide,
    }


@dataclass(slots=True, kw_only=True)
class _Desk:
    """An agent not yet running, so what a command pushes stays in its inbox."""

    catalog: ModelCatalog = field(
        default_factory=lambda: ModelCatalog(factories=_per_account),
    )
    printer: _Printer = field(default_factory=_Printer)
    panes: Panes = field(default_factory=Panes)
    provider: str = "scripted"
    team: Team | None = None
    agent: Agent = field(init=False)
    commands: Commands = field(init=False)

    def __post_init__(self) -> None:
        self.agent = Agent(
            model=self.catalog[self.provider][self.provider],
            tools=[_Echo(), _Plain()],
        )
        self.commands = Commands(
            self.agent,
            catalog=self.catalog,
            say=Scrollback(self.printer, tools=self.agent.tools).show,
            screen=self.printer,
            panes=self.panes,
            team=self.team,
        )

    def pushed(self, agent: Agent | None = None) -> list[types.RuntimeEvent]:
        return (agent or self.agent).runtime.inbox.drain_nowait()


def _teamed(*labels: str) -> tuple[_Desk, dict[str, Agent]]:
    catalog = ModelCatalog(factories=_per_account)
    team = Team(catalog=catalog)
    desk = _Desk(catalog=catalog, team=team)
    team.join(desk.agent)
    members: dict[str, Agent] = {}
    for label in labels:
        member = Agent(model=catalog["scripted"]["scripted"])
        team.join(member, label=label, parent="Agent", closing=AsyncExitStack())
        members[label] = member
    return desk, members


# ---- lines -----------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("line", ["", "   ", "\n"])
async def test_a_blank_line_does_nothing(line: str) -> None:
    desk = _Desk()
    assert await desk.commands(line) is False
    assert desk.pushed() == []
    assert desk.printer.said == []


@pytest.mark.asyncio
async def test_text_is_sent_to_the_agent_stripped() -> None:
    desk = _Desk()
    assert await desk.commands("  fix the bug \n") is False
    assert desk.pushed() == [UserMessage(content="fix the bug")]


@pytest.mark.asyncio
@pytest.mark.parametrize("line", ["/quit", "/exit", " /quit now"])
async def test_quit_and_exit_end_the_session(line: str) -> None:
    desk = _Desk()
    assert await desk.commands(line) is True
    assert desk.printer.said == []


@pytest.mark.asyncio
async def test_an_unknown_command_points_at_help() -> None:
    desk = _Desk()
    assert await desk.commands("/frobnicate x") is False
    assert desk.printer.said == [("error", "unknown command: /frobnicate; see /help")]


def _names(desk: _Desk) -> list[str]:
    return [".".join(s.name) for s in harvest(desk.commands.owners())]


def test_actions_come_first_then_the_model_the_tools_and_the_screen() -> None:
    names = _names(_Desk())
    assert names[:13] == [
        "quit",
        "exit",
        "help",
        "clear",
        "compact",
        "recompact",
        "halt",
        "kill",
        "send",
        "focus",
        "defer",
        "tasks",
        "login",
    ]
    assert names[13:15] == ["model", "model.provider"]
    assert names.index("tool.Echo.display.command.lines") < names.index(
        "tool.Plain.display.command.lines",
    )
    assert names[-1] == "screen.thinking"


@pytest.mark.asyncio
async def test_help_lists_every_setting_with_its_usage() -> None:
    desk = _Desk()
    await desk.commands("/help")
    ((kind, text),) = desk.printer.said
    rows = text.split("\n")
    assert kind == "note"
    assert rows[0] == "[/help] sagent commands"
    assert f"  {'/compact [hints]':<32s}compact history" in rows
    effort = f"  {'/model.thinking.effort VALUE':<32s}how hard the model reasons"
    assert effort in rows
    listed = [row.split()[0] for row in rows if row.startswith("  /")]
    assert listed == [f"/{name}" for name in _names(desk)]


@pytest.mark.asyncio
async def test_help_on_one_setting_says_all_it_knows() -> None:
    desk = _Desk()
    await desk.commands("/help eff")
    assert desk.printer.said == [
        (
            "note",
            (
                "[/help] /model.thinking.effort VALUE\nhow hard the model reasons"
                "\nnow: none (default)   (default, none)"
            ),
        ),
    ]
    await desk.commands("/help nothing-at-all")
    assert desk.printer.said[-1] == (
        "error",
        "[/help] no setting 'nothing-at-all'",
    )


@pytest.mark.asyncio
async def test_a_bare_setting_shows_its_value_and_a_bare_action_runs() -> None:
    desk = _Desk(provider="deep")
    await desk.commands("/effort")
    await desk.commands("/thinking")
    await desk.commands("/halt")
    assert desk.printer.said == [
        (
            "note",
            "[/model.thinking.effort] none (default)   (default, none, low, high)",
        ),
        ("note", "[/model.thinking] effort=none budget=none output=none"),
    ]
    assert desk.pushed() == [types.Halt()]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "line",
    [
        "/model.thinking.effort high",
        "/thinking.effort high",
        "/effort high",
        "/eff high",
    ],
)
async def test_a_setting_answers_to_its_name_its_last_segments_or_its_letters(
    line: str,
) -> None:
    desk = _Desk(provider="deep")
    await desk.commands(line)
    assert desk.agent.model.settings.thinking_effort == "high"
    assert desk.printer.said == [("note", "[/model.thinking.effort] high")]


@pytest.mark.asyncio
async def test_a_value_a_setting_refuses_changes_nothing() -> None:
    desk = _Desk(provider="deep")
    await desk.commands("/effort max")
    assert desk.agent.model.settings.thinking_effort == "none"
    assert desk.printer.said == [
        ("error", "[/model.thinking.effort] 'max' is not one of: none, low, high"),
    ]


@pytest.mark.asyncio
async def test_the_screen_shows_or_hides_reasoning_and_the_model_is_untouched() -> None:
    desk = _Desk(provider="deep")
    await desk.commands("/screen.thinking hide")
    assert desk.printer.show_thinking is False
    assert desk.agent.model.settings.thinking_output == "none"
    assert desk.printer.said == [("note", "[/screen.thinking] hide")]


# ---- session ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_clear_runs_as_a_task_and_empties_the_context() -> None:
    journal = RecordingJournal()
    journal.append(UserMessage(content="old"))
    provider = ScriptedProvider()
    printer = _Printer()
    async with Agent(model=provider["scripted"], journal=journal) as agent:
        commands = Commands(
            agent,
            catalog=ModelCatalog(factories={}),
            say=Scrollback(printer, tools=agent.tools).show,
            screen=printer,
            panes=Panes(),
        )
        assert list(agent.session) == [UserMessage(content="old")]
        await commands("/clear")
        assert printer.said == [("note", "[/clear] history cleared")]
        await asyncio.wait_for(asyncio.gather(*commands._tasks), timeout=5)
        assert list(agent.session) == []
    assert printer.said == [("note", "[/clear] history cleared")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("line", "note"),
    [
        ("/compact", "[/compact] queued"),
        ("/compact keep the API", "[/compact] queued (keep the API)"),
        ("/recompact  x ", "[/recompact] queued (x)"),
    ],
)
async def test_compact_queues_and_a_failure_prints_under_its_name(
    line: str,
    note: str,
) -> None:
    desk = _Desk()
    await desk.commands(line)
    await asyncio.gather(*desk.commands._tasks, return_exceptions=True)
    await asyncio.sleep(0)
    assert desk.printer.said == [
        ("note", note),
        (
            "error",
            "[/compact] NotImplementedError: This agent has no compactor or pruner.",
        ),
    ]
    assert desk.commands._tasks == set()


@dataclass(slots=True, kw_only=True)
class _Compactor:
    asked: list[str] = field(default_factory=list[str])

    def should_run(self) -> bool:
        return False

    async def run(self, instructions: str = "") -> None:
        self.asked.append(instructions)


@pytest.mark.asyncio
async def test_compact_hands_its_hints_to_the_compactor() -> None:
    desk = _Desk()
    compactor = _Compactor()
    desk.agent.session.compactor = compactor
    await desk.commands("/compact keep the API")
    await asyncio.gather(*desk.commands._tasks)
    assert compactor.asked == ["keep the API"]


@pytest.mark.asyncio
async def test_a_failed_clear_prints_under_its_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    desk = _Desk()

    async def broken() -> None:
        raise RuntimeError("no")

    monkeypatch.setattr(desk.agent.session, "clear", broken)
    await desk.commands("/clear")
    await asyncio.gather(*desk.commands._tasks, return_exceptions=True)
    assert desk.printer.said == [
        ("note", "[/clear] history cleared"),
        ("error", "[/clear] RuntimeError: no"),
    ]


@pytest.mark.asyncio
async def test_aclose_cancels_running_commands_quietly() -> None:
    desk = _Desk()
    await desk.commands("/clear")
    (task,) = desk.commands._tasks
    await desk.commands.aclose()
    assert task.cancelled()
    await asyncio.sleep(0)
    assert desk.printer.said == [("note", "[/clear] history cleared")]


# ---- runtime ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_bare_halt_halts_this_agent() -> None:
    desk = _Desk()
    await desk.commands("/halt")
    assert desk.pushed() == [types.Halt()]


@pytest.mark.asyncio
async def test_defer_waits_for_the_agent_to_idle() -> None:
    desk = _Desk()
    await desk.commands("/defer  later please ")
    await desk.commands("/defer")
    assert desk.pushed() == [types.UserDeferredMessage(text="later please")]
    assert desk.printer.said == [("error", "[/defer] needs TEXT")]


def _running(desk: _Desk, *call_ids: str, age_sec: float = 5.0) -> None:
    for call_id in call_ids:
        desk.agent.runtime._started[call_id] = (
            call("Echo", call_id, text=call_id),
            time.time() - age_sec,
        )


@pytest.mark.asyncio
async def test_kill_cancels_one_call_or_all_and_all_empties_the_panes() -> None:
    desk = _Desk()
    _running(desk, "c1", "c2")
    desk.panes.stage("queued", deferred=False)
    desk.panes.stage("later", deferred=True)
    await desk.commands("/kill c1")
    assert desk.pushed() == [types.Kill(call_id="c1")]
    assert desk.panes.queue == Staged(text="queued")
    await desk.commands("/kill all")
    assert desk.pushed() == [types.Kill(call_id="c1"), types.Kill(call_id="c2")]
    assert (desk.panes.queue, desk.panes.deferred) == (None, None)
    assert desk.printer.said == [
        ("note", "[/kill] cancelled c1"),
        ("note", "[/kill] cancelled all tool calls"),
    ]


@pytest.mark.asyncio
async def test_kill_needs_a_target_it_knows() -> None:
    desk = _Desk()
    await desk.commands("/kill")
    await desk.commands("/kill nope")
    assert desk.pushed() == []
    assert desk.printer.said == [
        ("error", "[/kill] needs an ID, a LABEL, or all"),
        ("error", "[/kill] no running call 'nope'; see /tasks"),
    ]


@pytest.mark.asyncio
async def test_kill_stops_a_persistent_child() -> None:
    desk, members = _teamed("reviewer")
    assert desk.team is not None
    await desk.commands("/kill reviewer")
    await asyncio.gather(*desk.commands._tasks)
    assert "reviewer" not in desk.team
    assert desk.printer.said == [("note", "[/kill] stopping reviewer")]
    del members


@pytest.mark.asyncio
async def test_tasks_lists_running_calls_and_persistent_children() -> None:
    desk, _ = _teamed("reviewer")
    _running(desk, "c1")
    await desk.commands("/tasks")
    ((kind, text),) = desk.printer.said
    assert kind == "note"
    head, job, child = text.split("\n")
    assert head == "[/tasks] 1 running tool call(s), 1 persistent agent(s)"
    assert job.split() == ["c1", "Echo", "running", "5s"]
    assert child.split() == ["reviewer", "persistent", "agent"]


@pytest.mark.asyncio
async def test_tasks_with_nothing_running_says_so() -> None:
    desk = _Desk()
    await desk.commands("/tasks")
    assert desk.printer.said == [
        ("note", "[/tasks] 0 running tool call(s), 0 persistent agent(s)"),
    ]


# ---- the team --------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("pattern", "halted"),
    [
        ("reviewer", ["reviewer"]),
        ("all", ["reviewer", "Agent_0", "Agent_1"]),
        ("Agent_*", ["Agent_0", "Agent_1"]),
        ("Agent_[1]", ["Agent_1"]),
        ("{reviewer, Agent_1,missing}", ["reviewer", "Agent_1"]),
        ("{Agent_0,reviewer}", ["reviewer", "Agent_0"]),
        ("/_0$|^rev/", ["reviewer", "Agent_0"]),
    ],
)
async def test_halt_reaches_the_agents_a_pattern_names(
    pattern: str,
    halted: list[str],
) -> None:
    desk, members = _teamed("reviewer", "Agent_0", "Agent_1")
    await desk.commands(f"/halt {pattern}")
    in_order = sorted(halted, key=list(members).index)
    assert [
        label for label, m in members.items() if desk.pushed(m) == [types.Halt()]
    ] == in_order
    assert desk.printer.said == [("note", f"[/halt] halted {', '.join(in_order)}")]
    assert desk.pushed() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("pattern", ["Agent", "all", "Ag*", "{Agent}", "/Ag/"])
async def test_no_pattern_reaches_this_agent(pattern: str) -> None:
    desk, _ = _teamed()
    await desk.commands(f"/halt {pattern}")
    assert desk.pushed() == []
    assert desk.printer.said == [
        ("error", f"[/halt] no agent {pattern!r}; agents: none"),
    ]


@pytest.mark.asyncio
async def test_an_unknown_target_names_the_agents_there_are() -> None:
    desk, _ = _teamed("b", "a")
    await desk.commands("/halt zed")
    assert desk.printer.said == [("error", "[/halt] no agent 'zed'; agents: a, b")]


@pytest.mark.asyncio
@pytest.mark.parametrize("pattern", ["{bb", "bb}"])
async def test_a_half_brace_is_a_label(pattern: str) -> None:
    """Read as braces, either would name ``b``, the label between them."""
    desk, members = _teamed("b")
    await desk.commands(f"/halt {pattern}")
    assert desk.pushed(members["b"]) == []
    assert desk.printer.said == [("error", f"[/halt] no agent {pattern!r}; agents: b")]


@pytest.mark.asyncio
async def test_a_lone_agent_has_no_team_to_reach() -> None:
    desk = _Desk()
    await desk.commands("/halt x")
    assert desk.printer.said == [("error", "[/halt] no agent 'x'; agents: none")]


@pytest.mark.asyncio
@pytest.mark.parametrize("pattern", ["/", "//"])
async def test_a_regex_of_nothing_is_a_label_not_everything(pattern: str) -> None:
    desk, _ = _teamed("a")
    await desk.commands(f"/halt {pattern}")
    assert desk.printer.said == [("error", f"[/halt] no agent {pattern!r}; agents: a")]


@pytest.mark.asyncio
async def test_a_bad_regex_says_why() -> None:
    desk, members = _teamed("a")
    await desk.commands("/halt /(/")
    ((kind, text),) = desk.printer.said
    assert (kind, text.split(":")[0]) == ("error", "[/halt] invalid target regex")
    assert desk.pushed(members["a"]) == []


@pytest.mark.asyncio
async def test_send_messages_each_agent_named() -> None:
    desk, members = _teamed("a", "b")
    await desk.commands("/send all  look at  this ")
    for label in ("a", "b"):
        assert desk.pushed(members[label]) == [
            AgentToAgentMessage(
                sender="user",
                recipient=label,
                content="look at  this",
            ),
        ]
    assert desk.printer.said == [("note", "[/send] user→a, b: look at  this")]


@pytest.mark.asyncio
@pytest.mark.parametrize("line", ["/send", "/send a", "/send a   "])
async def test_send_needs_a_target_and_text(line: str) -> None:
    desk, members = _teamed("a")
    await desk.commands(line)
    assert desk.printer.said == [("error", "[/send] needs AGENTS TEXT")]
    assert desk.pushed(members["a"]) == []


@pytest.mark.asyncio
async def test_send_runs_a_command_on_each_agent_named() -> None:
    desk, members = _teamed("a", "b")
    await desk.commands("/send all /halt")
    await desk.commands("/send a /effort")
    assert desk.pushed(members["a"]) == [types.Halt()]
    assert desk.pushed(members["b"]) == [types.Halt()]
    assert desk.pushed() == []
    assert desk.printer.said == [
        ("note", "[/model.thinking.effort] none (default)   (default, none)"),
    ]


@pytest.mark.asyncio
async def test_focus_refuses_with_no_screen_and_an_unknown_label() -> None:
    desk, _ = _teamed("a")
    await desk.commands("/focus a")
    shown: list[str] = []
    desk.commands.focus = shown.append
    await desk.commands("/focus nobody")
    await desk.commands("/focus a")
    assert shown == ["a"]
    assert desk.printer.said == [
        ("error", "[/focus] there is no screen to show another agent on"),
        ("error", "[/focus] no agent 'nobody'; agents: Agent, a"),
    ]


@pytest.mark.asyncio
async def test_focus_on_a_lone_agent_lists_no_agents() -> None:
    desk = _Desk()
    desk.commands.focus = list[str]().append
    await desk.commands("/focus a")
    assert desk.printer.said == [("error", "[/focus] no agent 'a'; agents: none")]


@pytest.mark.asyncio
async def test_a_command_sent_to_an_agent_can_still_focus() -> None:
    desk, _ = _teamed("a")
    shown: list[str] = []
    desk.commands.focus = shown.append
    await desk.commands("/send a /focus Agent")
    assert shown == ["Agent"]


@pytest.mark.asyncio
async def test_send_to_a_relation_reaches_its_agents_and_is_owed_a_reply() -> None:
    desk, members = _teamed("a", "b")
    assert desk.team is not None
    _ = desk.team.join(
        Agent(model=desk.catalog["scripted"]["scripted"]),
        label="a1",
        parent="a",
    )

    await desk.commands("/send children status?")
    await desk.commands("/send descendants /effort")

    for label in ("a", "b"):
        (message,) = desk.pushed(members[label])
        assert message == AgentToAgentMessage(
            sender="user",
            recipient=label,
            content="status?",
        )
        assert desk.team.owes(label) == {"user": "status?"}
    assert desk.printer.said[0] == ("note", "[/send] user→a, b: status?")
    assert len(desk.printer.said) == 4


@pytest.mark.asyncio
async def test_label_dot_setting_is_send_to_that_agent() -> None:
    desk, members = _teamed("a")
    await desk.commands("/a.halt")
    await desk.commands("/children.halt")
    await desk.commands("/nobody.halt")
    assert desk.pushed(members["a"]) == [types.Halt(), types.Halt()]
    assert desk.printer.said[-1] == (
        "error",
        "unknown command: /nobody.halt; see /help",
    )


@pytest.mark.asyncio
async def test_send_parent_from_a_root_starts_its_watcher_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    desk, _ = _teamed()
    team = desk.team
    assert team is not None
    async with team:
        await desk.commands("/send parent keep it honest")
        await desk.commands("/send parent and brief")
        await desk.commands("/parent.watcher off")
        await desk.commands("/parent.watcher")

        watcher = team["watcher"]
        assert team.resolve("parent", by="Agent") == ["watcher"]
        assert team.watching == {"watcher": False}
        inbox = [
            m.content or ""
            for m in watcher.runtime.inbox.drain_nowait()
            if isinstance(m, AgentToAgentMessage)
        ]
        assert inbox[0].startswith("<skill name='watch' source='builtin'>")
        assert inbox[1:] == ["keep it honest", "and brief"]
        assert watcher.system_prompt == desk.agent.system_prompt
        assert list(watcher.tools) == list(desk.agent.tools)
    assert desk.printer.said == [
        (
            "note",
            "[/send] started watcher, watching Agent\nuser→watcher: keep it honest",
        ),
        ("note", "[/send] user→watcher: and brief"),
        ("note", "[/watcher] off"),
        ("note", "[/watcher] off   (on, off)"),
    ]


@pytest.mark.asyncio
async def test_a_watcher_refuses_a_value_that_is_not_on_or_off(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    desk, _ = _teamed()
    team = desk.team
    assert team is not None
    async with team:
        await desk.commands("/send parent hi")
        await desk.commands("/parent.watcher maybe")
    assert desk.printer.said[-1] == (
        "error",
        "[/watcher] 'maybe' is not one of: on, off",
    )


@pytest.mark.asyncio
async def test_send_parent_from_a_child_reaches_its_spawner() -> None:
    desk, members = _teamed("a")
    commands = Commands(
        members["a"],
        catalog=desk.catalog,
        say=desk.commands.say,
        screen=desk.printer,
        panes=Panes(),
        team=desk.team,
    )
    await commands("/send parent done")
    assert [m.content for m in desk.pushed() if isinstance(m, AgentToAgentMessage)] == [
        "done",
    ]


def _efforts(account: str | None) -> dict[str, ProviderFactory]:
    """Return ``scripted``: ``all`` reasons up to max, ``some`` only to high."""
    del account

    def row(model_id: str, efforts: frozenset[ThinkingEffort]) -> ModelCapability:
        return _windowed(model_id, 200_000, efforts=efforts)

    table = ModelTable(
        rows=(
            row("scripted", frozenset({"none"})),
            row("all", frozenset({"none", "low", "high", "max"})),
            row("some", frozenset({"none", "low", "high"})),
        ),
        roles={"default": "all"},
    )
    return {"scripted": partial(ScriptedProvider, models=table)}


@dataclass(slots=True, kw_only=True)
class _View:
    show_thinking: bool = True


def _rolled(*labels: tuple[str, str]) -> tuple[_Desk, dict[str, _View]]:
    """Return a desk over a team of ``(label, parent)``, each with its own screen."""
    catalog = ModelCatalog(factories=_per_account)
    team = Team(catalog=catalog)
    screens: dict[str, _View] = {}
    desk = _Desk(catalog=catalog, team=team)
    team.join(desk.agent)
    for label, parent in labels:
        team.join(
            Agent(model=catalog["scripted"]["scripted"]),
            label=label,
            parent=parent,
        )

    def screen(label: str) -> _View:
        return screens.setdefault(label, _View())

    desk.commands = Commands(
        desk.agent,
        catalog=catalog,
        say=desk.commands.say,
        screen=screen("Agent"),
        panes=desk.panes,
        team=team,
        screens=screen,
    )
    return desk, screens


def _shown(screens: dict[str, _View]) -> dict[str, bool]:
    return {label: v.show_thinking for label, v in sorted(screens.items())}


@pytest.mark.asyncio
async def test_a_choice_rolls_down_to_each_child_that_made_none() -> None:
    desk, screens = _rolled(("a", "Agent"), ("b", "Agent"), ("a1", "a"))
    team = desk.team
    assert team is not None
    await desk.commands("/send b /screen.thinking show")

    await desk.commands("/screen.thinking hide")

    assert _shown(screens) == {"Agent": False, "a": False, "a1": False, "b": True}
    assert team.choices == {
        "b": {"screen.thinking": "show"},
        "Agent": {"screen.thinking": "hide"},
    }


@pytest.mark.asyncio
async def test_default_on_a_child_takes_its_parent_s_again() -> None:
    desk, screens = _rolled(("a", "Agent"), ("a1", "a"))
    team = desk.team
    assert team is not None
    await desk.commands("/screen.thinking hide")
    await desk.commands("/send a /screen.thinking show")
    assert _shown(screens) == {"Agent": False, "a": True, "a1": True}

    await desk.commands("/send a /screen.thinking default")

    assert _shown(screens) == {"Agent": False, "a": False, "a1": False}
    assert "a" not in team.choices or team.choices["a"] == {}


@pytest.mark.asyncio
async def test_default_on_a_root_is_a_fresh_one_and_rolls_down() -> None:
    desk, screens = _rolled(("a", "Agent"))
    team = desk.team
    assert team is not None
    await desk.commands("/screen.thinking hide")

    await desk.commands("/screen.thinking default")

    assert _shown(screens) == {"Agent": True, "a": True}
    assert team.choices["Agent"] == {}


@pytest.mark.asyncio
async def test_a_child_that_joins_later_inherits_once_its_commands_are_made() -> None:
    desk, screens = _rolled()
    team = desk.team
    assert team is not None
    await desk.commands("/screen.thinking hide")
    team.join(
        Agent(model=desk.catalog["scripted"]["scripted"]),
        label="late",
        parent="Agent",
    )

    _ = await desk.commands.commands_for(team, "late")

    assert _shown(screens) == {"Agent": False, "late": False}


@pytest.mark.asyncio
async def test_a_choice_a_child_cannot_take_says_why_and_stops_there() -> None:
    """A setting with no closest value, such as a screen's, cannot roll on."""
    desk, screens = _rolled(("a", "Agent"))
    child = await desk.commands.commands_for(
        desk.team or Team(catalog=desk.catalog),
        "a",
    )

    await child.take("screen.thinking", "maybe")
    await child.take("tool.Nope.display.output.lines", "1")

    assert _shown(screens) == {"Agent": True, "a": True}
    assert desk.printer.said == [
        (
            "error",
            "[a /screen.thinking] 'maybe' is not one of: default, show, hide",
        ),
    ]


@pytest.mark.asyncio
async def test_effort_rolls_to_the_closest_each_child_offers_and_on_down() -> None:
    catalog = ModelCatalog(factories=_efforts)
    team = Team(catalog=catalog)
    desk = _Desk(catalog=catalog, team=team)
    desk.agent.model = catalog["scripted"]["all"]
    team.join(desk.agent)
    for label, parent, model in (("a", "Agent", "some"), ("a1", "a", "all")):
        team.join(
            Agent(model=catalog["scripted"][model]),
            label=label,
            parent=parent,
        )

    await desk.commands("/effort max")

    assert [team[label].model.settings.thinking_effort for label in team] == [
        "max",
        "high",
        "max",
    ]
    assert [e for e in desk.printer.said if e[0] == "error"] == []


@pytest.mark.asyncio
async def test_send_without_a_team_has_no_one_to_reach() -> None:
    desk = _Desk()
    await desk.commands("/send all hi")
    assert desk.printer.said == [("error", "[/send] there are no other agents")]


@pytest.mark.asyncio
async def test_an_agent_outside_the_team_rolls_nothing() -> None:
    desk, _ = _rolled()
    outsider = Commands(
        Agent(model=desk.catalog["scripted"]["scripted"]),
        catalog=desk.catalog,
        say=desk.commands.say,
        screen=_View(),
        panes=Panes(),
        team=desk.team,
    )
    await outsider("/screen.thinking hide")
    await outsider("/screen.thinking default")
    await outsider.inherit()
    assert desk.team is not None
    assert "" not in desk.team.choices


@pytest.mark.asyncio
async def test_send_a_model_switch_changes_only_that_agent() -> None:
    desk, members = _teamed("a")
    await desk.commands("/send a /model deep/deep")
    assert members["a"].model.capability.model_id == "deep"
    assert desk.agent.model.capability.model_id == "scripted"


@pytest.mark.asyncio
async def test_send_quit_stops_that_agent() -> None:
    desk, members = _teamed("a")
    assert desk.team is not None
    await desk.commands("/send a /quit")
    await asyncio.gather(*desk.commands._tasks)
    assert "a" not in desk.team
    assert desk.printer.said == [("note", "[/send] stopping a")]
    del members


@pytest.mark.asyncio
async def test_commands_run_on_an_agent_close_with_these() -> None:
    desk, _ = _teamed("a")
    await desk.commands("/send a /clear")
    await desk.commands("/send a /clear")
    (child,) = desk.commands._children.values()
    tasks = set(child._tasks)
    assert len(tasks) == 2
    await desk.commands.aclose()
    assert all(task.cancelled() for task in tasks)


@pytest.mark.asyncio
async def test_a_command_run_on_an_agent_reaches_its_team_but_not_these_panes() -> None:
    desk, members = _teamed("a", "b")
    desk.panes.stage("mine", deferred=False)
    await desk.commands("/send a /halt b")
    await desk.commands("/send a /kill all")
    await desk.commands("/send a /halt a")
    assert desk.pushed(members["b"]) == [types.Halt()]
    assert desk.pushed(members["a"]) == []
    assert desk.panes.queue == Staged(text="mine")
    assert desk.printer.said == [
        ("note", "[/halt] halted b"),
        ("note", "[/kill] cancelled all tool calls"),
        ("error", "[/halt] no agent 'a'; agents: Agent, b"),
    ]


@pytest.mark.asyncio
async def test_send_to_no_one_says_so() -> None:
    desk, _ = _teamed("a")
    await desk.commands("/send zed hi")
    assert desk.printer.said == [("error", "[/send] no agent 'zed'; agents: a")]


@pytest.mark.asyncio
async def test_a_failed_stop_prints_under_its_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    desk, _ = _teamed("a")
    assert desk.team is not None

    async def broken(label: str) -> None:
        raise RuntimeError(f"stuck {label}")

    monkeypatch.setattr(desk.team, "stop", broken)
    await desk.commands("/kill a")
    await asyncio.gather(*desk.commands._tasks, return_exceptions=True)
    assert desk.printer.said == [
        ("note", "[/kill] stopping a"),
        ("error", "[/kill a] RuntimeError: stuck a"),
    ]


# ---- model -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_bare_model_names_what_each_name_resolved_to() -> None:
    desk = _Desk(provider="wide")
    await desk.commands("/model")
    await desk.commands("/provider")
    await desk.commands("/context")
    await desk.commands("/account")
    assert desk.printer.said == [
        (
            "note",
            "[/model] wide/wide-1.0+fullcontext+nothink",  # codespell:ignore nothink
        ),
        ("note", "[/model.provider] wide   (scripted, deep, signing, wide)"),
        ("note", "[/model.context] 1m (default)   (default, fullcontext)"),
        ("note", "[/model.provider.account] default   (default)"),
    ]


@pytest.mark.asyncio
async def test_model_switches_and_keeps_what_the_new_model_still_offers() -> None:
    desk = _Desk(provider="deep")
    model = desk.agent.model
    model.settings = replace(model.settings, thinking_effort="high")
    await desk.commands("/model deep")
    assert desk.agent.model.settings.thinking_effort == "high"
    await desk.commands("/model scripted/scripted")
    assert desk.agent.model.provider.name == "scripted"
    assert desk.agent.model.settings.thinking_effort == "none"
    assert desk.printer.said == [
        ("note", "[/model] deep/deep+fullcontext+highthink"),
        (
            "note",
            "[/model] scripted/scripted+fullcontext+nothink",  # codespell:ignore nothink
        ),
    ]


@pytest.mark.asyncio
async def test_a_bare_model_comes_from_whichever_provider_serves_it() -> None:
    desk = _Desk()
    await desk.commands("/model deep")
    assert desk.agent.model.provider.name == "deep"


@pytest.mark.asyncio
async def test_a_model_s_effort_tag_is_applied_after_its_settings_are_kept() -> None:
    desk = _Desk(provider="deep")
    await desk.commands("/model deep+high")
    assert desk.agent.model.settings.thinking_effort == "high"
    await desk.commands("/model deep+xhigh")
    assert desk.agent.model.settings.thinking_effort == "high"
    assert desk.printer.said[-1] == (
        "error",
        (
            "[/model] ValueError: thinking_effort='xhigh' is not offered by deep;"
            " allowed: 'high', 'low', 'none'"
        ),
    )


@pytest.mark.asyncio
async def test_context_picks_the_window_and_a_switch_keeps_it() -> None:
    desk = _Desk(provider="wide")
    await desk.commands("/context 200k")
    assert desk.agent.model.settings.context_size.tokens == 200_000
    await desk.commands("/model wide")
    assert desk.agent.model.settings.context_size.tokens == 200_000
    await desk.commands("/model wide+258k")
    assert desk.agent.model.settings.context_size.tokens == 258_000
    assert desk.printer.said == [
        ("note", "[/model.context] 200k"),
        ("note", "[/model] wide/wide-1.0+200k+nothink"),  # codespell:ignore nothink
        ("note", "[/model] wide/wide-1.0+258k+nothink"),  # codespell:ignore nothink
    ]


@pytest.mark.asyncio
async def test_provider_keeps_the_model_where_the_new_one_serves_it() -> None:
    desk = _Desk()
    await desk.commands("/provider deep")
    assert desk.agent.model.provider.name == "deep"
    assert (
        desk.printer.said
        == [
            (
                "note",
                "[/model.provider] deep/deep+fullcontext+nothink",  # codespell:ignore nothink
            ),
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("line", "error"),
    [
        ("/model nowhere/x", "[/model] UnknownModelError: No model 'nowhere/x'."),
        (
            "/provider nowhere",
            (
                "[/model.provider] UnknownProviderError: unknown provider"
                " 'nowhere'; known: scripted, deep, signing, wide"
            ),
        ),
        ("/model deep/nope", "[/model] UnknownModelError: No model 'nope'."),
    ],
)
async def test_an_unknown_model_leaves_the_model_alone(line: str, error: str) -> None:
    desk = _Desk()
    before = desk.agent.model
    await desk.commands(line)
    assert desk.agent.model is before
    assert desk.printer.said == [("error", error)]


# ---- thinking and effort ---------------------------------------------------


@pytest.mark.asyncio
async def test_thinking_sets_the_catalog_s_axes_and_the_model_shows_them() -> None:
    desk = _Desk(provider="deep")
    await desk.commands("/thinking budget=auto output=text")
    await desk.commands("/model")
    assert (
        desk.agent.model.settings.thinking_budget,
        desk.agent.model.settings.thinking_output,
    ) == ("auto", "text")
    assert (
        desk.printer.said
        == [
            ("note", "[/model.thinking] effort=none budget=auto output=text"),
            (
                "note",
                (
                    "[/model] deep/deep+fullcontext+nothink budget=auto output=text"  # codespell:ignore nothink
                ),
            ),
        ]
    )


# ---- tools -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_changes_a_tool_s_display_where_the_agent_reads_it() -> None:
    desk = _Desk()
    await desk.commands("/tool.Echo.display.output.lines 3,1")
    tool = desk.agent.tools["Echo"]
    assert tool.display.output == DisplaySpec(lines=(3, 1))
    assert tool.display.command == ToolDisplay().command
    assert desk.agent.runtime.tools["Echo"] is tool
    assert desk.printer.said == [("note", "[/tool.Echo.display.output.lines] 3,1")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("line", "error"),
    [
        (
            "/tool.Plain.display.output.lines 1,1",
            "[/tool.Plain.display.output.lines] Plain cannot change its display",
        ),
        (
            "/tool.Echo.display.output.lines x",
            (
                "[/tool.Echo.display.output.lines] lines must be HEAD,TAIL or all,"
                " not 'x'."
            ),
        ),
    ],
)
async def test_tool_says_what_it_cannot_change(line: str, error: str) -> None:
    desk = _Desk()
    before = desk.agent.tools["Echo"]
    await desk.commands(line)
    assert desk.agent.tools["Echo"] is before
    assert desk.printer.said == [("error", error)]


# ---- login -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_login_signs_in_and_sends_what_waited() -> None:
    desk = _Desk(provider="signing")
    model = desk.agent.model
    desk.panes.stage("after login", deferred=True)
    await desk.commands("/login")
    provider = desk.agent.model.provider
    assert isinstance(provider, _Signing)
    assert provider.signed_in == 1
    assert desk.agent.model is model, "a plain sign-in keeps the agent's model"
    assert desk.pushed() == [types.UserDeferredMessage(text="after login")]
    assert desk.panes.deferred is None
    assert desk.printer.said == [("note", "[/login] signing signed in")]


@pytest.mark.asyncio
async def test_login_retries_the_turn_a_failure_stopped() -> None:
    """A sign-in changes no model, so nothing else would send the turn again."""
    desk = _Desk(provider="signing")
    desk.agent.runtime.service_suspended_until = time.time() + 60.0
    await desk.commands("/login")
    assert desk.pushed() == [types.Retry()]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        OSError("no browser"),
        RuntimeError("denied"),
        ValueError("bad code"),
        AuthError("the code was for another sign-in"),
        EOFError("no input"),
    ],
)
async def test_a_failed_login_says_why_and_keeps_what_waits(
    error: Exception,
) -> None:

    desk = _Desk(provider="signing")
    provider = desk.agent.model.provider
    assert isinstance(provider, _Signing)
    provider.error = error
    desk.panes.stage("after login", deferred=True)
    await desk.commands("/login")
    assert desk.pushed() == []
    assert desk.panes.deferred == Staged(text="after login")
    assert desk.printer.said == [("error", f"[/login] {type(error).__name__}: {error}")]


@pytest.mark.asyncio
async def test_login_on_a_provider_without_one_says_so() -> None:
    desk = _Desk()
    before = desk.agent.model
    await desk.commands("/login")
    assert desk.printer.said == [
        ("error", "[/login] scripted has no interactive sign-in"),
    ]
    assert desk.agent.model is before


@pytest.mark.parametrize(
    ("environ", "platform", "wanted"),
    [
        ({"DISPLAY": ":0"}, "linux", False),
        ({"WAYLAND_DISPLAY": "wayland-0"}, "linux", False),
        ({}, "linux", True),
        ({"DISPLAY": ":0", "SSH_CONNECTION": "1 2 3 4"}, "linux", True),
        ({}, "darwin", False),
        ({"SSH_TTY": "/dev/pts/1"}, "darwin", True),
        ({"SSH_CLIENT": "1 2 3"}, "darwin", True),
    ],
    ids=["x11", "wayland", "no-display", "ssh", "mac", "mac-ssh-tty", "mac-ssh"],
)
def test_headless_is_wanted_over_ssh_or_where_no_screen_shows_a_browser(
    environ: dict[str, str],
    platform: str,
    wanted: bool,
) -> None:
    assert headless_wanted(environ, platform=platform) is wanted


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("line", "remote", "headless"),
    [
        ("/login", True, True),
        ("/login", False, False),
        ("/login headless=auto", True, True),
        ("/login headless=false", True, False),
        ("/login headless=true", False, True),
    ],
)
async def test_login_signs_in_headless_where_it_must_unless_told(
    monkeypatch: pytest.MonkeyPatch,
    line: str,
    remote: bool,
    headless: bool,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("DISPLAY", ":0")
    for name in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"):
        monkeypatch.delenv(name, raising=False)
    if remote:
        monkeypatch.setenv("SSH_CONNECTION", "1 2 3 4")
    desk = _Desk(provider="signing")
    await desk.commands(line)
    provider = desk.agent.model.provider
    assert isinstance(provider, _Signing)
    assert provider.headless == [headless]


@pytest.mark.asyncio
async def test_login_headless_signs_in_without_the_browser() -> None:
    desk = _Desk(provider="signing")
    await desk.commands("/login headless=true")
    provider = desk.agent.model.provider
    assert isinstance(provider, _Signing)
    assert provider.headless == [True]


@pytest.mark.asyncio
async def test_an_account_moves_the_agent_onto_it_and_login_signs_in_as_it() -> None:
    desk = _Desk(provider="signing")
    before = desk.agent.model
    await desk.commands("/account work")
    await desk.commands("/login")
    provider = desk.agent.model.provider
    assert desk.agent.model is not before
    assert desk.catalog.account == "work"
    assert provider is desk.catalog["signing"]
    assert isinstance(provider, _Signing)
    assert (provider.account, provider.signed_in) == ("work", 1)
    assert desk.printer.said == [
        ("note", "[/model.provider.account] work"),
        ("note", "[/login] signing signed in as work"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("line", ["/model signing/scripted", "/provider signing"])
async def test_a_model_switch_after_an_account_switch_stays_on_the_account(
    line: str,
) -> None:
    desk = _Desk(provider="signing")
    await desk.commands("/account work")

    await desk.commands(line)

    provider = desk.agent.model.provider
    assert isinstance(provider, _Signing)
    assert provider.account == "work"


@pytest.mark.asyncio
async def test_an_account_switch_reaches_what_the_team_spawns() -> None:
    catalog = ModelCatalog(factories=_per_account)
    team = Team(catalog=catalog)
    desk = _Desk(catalog=catalog, team=team, provider="signing")
    team.join(desk.agent)

    await desk.commands("/account work")

    provider = team.catalog["signing"]
    assert isinstance(provider, _Signing)
    assert provider.account == "work"


def _unbuilt_as_work(account: str | None) -> dict[str, ProviderFactory]:
    """Return the desk's providers, ``signing`` unbuildable as ``work``."""
    factories = _per_account(account)
    if account == "work":
        factories["signing"] = partial(_Picky, serves=frozenset())
    return factories


@pytest.mark.asyncio
async def test_an_account_the_model_cannot_move_onto_leaves_both_as_they_were() -> None:
    desk = _Desk(catalog=ModelCatalog(factories=_unbuilt_as_work), provider="signing")
    before = desk.agent.model

    await desk.commands("/account work")

    assert desk.agent.model is before
    assert desk.catalog.account is None
    assert desk.catalog["signing"] is before.provider
    assert desk.printer.said == [
        (
            "error",
            "[/model.provider.account] UnknownModelError: No model 'scripted'.",
        ),
    ]


@pytest.mark.asyncio
async def test_login_refuses_a_headless_that_is_no_boolean() -> None:
    desk = _Desk(provider="signing")
    await desk.commands("/login headless=maybe")
    assert desk.printer.said == [
        ("error", "[/login] headless='maybe': give auto, true, or false"),
    ]


@pytest.mark.asyncio
async def test_login_refuses_an_option_it_does_not_take() -> None:
    desk = _Desk(provider="signing")
    await desk.commands("/login headles=true")
    provider = desk.agent.model.provider
    assert isinstance(provider, _Signing)
    assert provider.signed_in == 0
    assert desk.printer.said == [("error", "[/login] unknown option: headles")]


@pytest.mark.asyncio
async def test_an_account_that_is_no_name_is_refused() -> None:
    desk = _Desk(provider="signing")
    await desk.commands("/account ../x")
    assert desk.catalog.account is None
    assert desk.printer.said == [
        (
            "error",
            (
                "[/model.provider.account] ValueError: account '../x' is not a name:"
                " use letters, digits, '_', '-'"
            ),
        ),
    ]


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
