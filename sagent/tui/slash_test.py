"""Tests for slash commands: each changes the part of the agent it names."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING, override
from unittest.mock import AsyncMock, MagicMock, Mock

import asyncio
import sys
import time

from treekle import PlainTree

import pytest

from sagent.agent.agent import Agent
from sagent.agent.team import Team
from sagent.host.settings import HostSettings
from sagent.lib.userdirs import config_dir
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
from sagent.providers.login import Logins
from sagent.providers.table import ModelTable
from sagent.runtime import types
from sagent.session.types import (
    AgentToAgentMessage,
    UncategorizedToolResult,
    UserMessage,
)
from sagent.settings.exceptions import SettingError
from sagent.settings.harvest import harvest
from sagent.settings.types import Setting
from sagent.tools.display import DisplaySpec, ToolDisplay
from sagent.tools.mcp_settings import McpSettings
from sagent.tools.types import ToolOutput, ToolStreamingResult, whole_output
from sagent.tui import slash
from sagent.tui.prompt import Panes, Staged
from sagent.tui.recording import RecordingPrinter
from sagent.tui.render import Scrollback
from sagent.tui.settings.runtime import RuntimeSettings
from sagent.tui.settings.screen import ScreenSettings
from sagent.tui.slash import Allowed, Commands, owners_without_agent
from sagent.tui.types import CommandOutput


if TYPE_CHECKING:
    from pathlib import Path

    from sagent.providers.types import (
        Provider,
        ProviderFactory,
        ProviderResources,
    )
    from sagent.settings.types import HasSettings
    from sagent.tools.types import ToolContext


@pytest.fixture(autouse=True)
def working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


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
        default=ModelSettings(context_size=prices["fullcontext"]),
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
        resources: ProviderResources,
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
        resources: ProviderResources,
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

    @property
    def logins(self) -> Logins:
        return Logins(config_dir() / "rekursiv-ai" / "secrets" / "signing.json")


def _wide(*, name: str, resources: ProviderResources) -> Provider:
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
    printer: RecordingPrinter = field(default_factory=RecordingPrinter)
    panes: Panes = field(default_factory=Panes)
    provider: str = "scripted"
    team: Team | None = None
    allowed: Allowed | None = None
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
            allowed=self.allowed,
        )

    def pushed(self, agent: Agent | None = None) -> list[types.RuntimeEvent]:
        return (agent or self.agent).runtime.inbox.drain_nowait()


def _teamed(
    *labels: str,
    allowed: Allowed | None = None,
) -> tuple[_Desk, dict[str, Agent]]:
    catalog = ModelCatalog(factories=_per_account)
    team = Team(catalog=catalog)
    desk = _Desk(catalog=catalog, team=team, allowed=allowed)
    team.join(desk.agent)
    members: dict[str, Agent] = {}
    for label in labels:
        member = Agent(model=catalog["scripted"]["scripted"])
        team.join(member, label=label, parent="Agent", closing=AsyncExitStack())
        members[label] = member
    return desk, members


@pytest.mark.asyncio
async def test_delegated_theme_commands_change_the_same_terminal() -> None:
    desk, _ = _teamed("kid")
    await desk.commands("/screen.theme light")
    await desk.commands("/send kid /screen.theme dark")
    assert desk.commands.theme.current.name == "dark"
    assert desk.team is not None
    assert "screen.theme" not in desk.team.choices.get("Agent", {})
    assert "screen.theme" not in desk.team.choices.get("kid", {})
    await desk.commands("/send kid /screen.theme default")
    assert desk.commands.theme.current.name == "default"
    await desk.commands.aclose()


@pytest.mark.asyncio
async def test_theme_edits_are_live_terminal_settings_not_inherited_choices() -> None:
    desk, _ = _teamed("kid")
    await desk.commands("/screen.theme paper")
    await desk.commands("/screen.theme.paper.thinking bold cyan")
    await desk.commands("/send kid /screen.theme.paper.status italic yellow")
    assert desk.commands.theme.current.name == "paper"
    assert desk.commands.theme.current.styles["thinking"].bold is True
    assert desk.commands.theme.current.styles["status"].italic is True
    assert desk.team is not None
    assert desk.team.choices == {}
    await desk.commands("/screen.theme.default.thinking italic magenta")
    assert desk.commands.theme.current.name == "user1"
    assert desk.printer.said[-1] == (
        "note",
        "[/screen.theme.default.thinking] user1.thinking: italic magenta",
    )
    await desk.commands.aclose()


@pytest.mark.asyncio
async def test_theme_edits_preserve_the_teams_independent_messaging_choice() -> None:
    desk, _ = _teamed("kid")
    assert desk.team is not None
    await desk.commands("/peer-messaging off")
    await desk.commands("/screen.theme paper")
    await desk.commands("/screen.theme.paper.status bold cyan")
    await desk.commands("/peer-messaging")
    assert desk.printer.said[-1] == (
        "note",
        "[/peer-messaging] off   (on, off, default)",
    )
    await desk.commands("/send kid /peer-messaging default")
    await desk.commands("/peer-messaging invalid")
    assert desk.printer.said[-1] == (
        "error",
        "[/peer-messaging] 'invalid' is not one of: on, off, default",
    )
    assert desk.commands.theme.current.name == "paper"
    assert desk.team.choices == {"Agent": {"peer-messaging": "off"}, "kid": {}}
    assert not desk.team.peer_messaging("kid")
    await desk.commands.aclose()


_ALLOWED = Allowed(
    names=frozenset({"help", "send", "model.thinking.*"}),
    refusal="is not available here",
)


@pytest.mark.asyncio
async def test_a_line_naming_what_is_not_allowed_is_refused_and_changes_nothing() -> (
    None
):
    desk, _ = _teamed("kid", allowed=_ALLOWED)
    await desk.commands("/screen.thinking show")
    await desk.commands("/send kid /screen.thinking show")
    await desk.commands("/kid.screen.thinking show")
    await desk.commands("/thinking.effort none")
    assert desk.printer.said == [
        ("error", "/screen.thinking is not available here"),
        ("error", "/screen.thinking is not available here"),
        ("error", "/screen.thinking is not available here"),
        ("note", "[/model.thinking.effort] none (default)"),
    ]
    assert not desk.printer.show_thinking
    await desk.commands.aclose()


@pytest.mark.asyncio
async def test_a_send_that_is_not_allowed_runs_nothing_on_the_agent_it_names() -> None:
    desk, _ = _teamed("kid", allowed=replace(_ALLOWED, names=frozenset({"help"})))
    await desk.commands("/kid.help")
    assert desk.printer.said == [("error", "/send is not available here")]
    await desk.commands.aclose()


def test_what_a_line_may_reach_is_only_what_is_allowed() -> None:
    reached = [".".join(s.name) for s in _Desk(allowed=_ALLOWED).commands.reachable()]
    assert reached == [
        "help",
        "send",
        "model.thinking.effort",
        "model.thinking.budget",
        "model.thinking.output",
    ]
    every = _Desk().commands
    assert [s.name for s in every.reachable()] == [
        s.name for s in harvest(every.owners())
    ]


@pytest.mark.asyncio
async def test_help_lists_only_what_a_line_may_reach() -> None:
    desk = _Desk(allowed=_ALLOWED)
    await desk.commands("/help")
    listed = [
        row.split()[0]
        for row in desk.printer.said[-1][1].split("\n")
        if row.startswith("  /")
    ]
    assert listed == [
        "/help",
        "/send",
        "/model.thinking.effort",
        "/model.thinking.budget",
        "/model.thinking.output",
    ]


@pytest.fixture
def action_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Commands, Mock, list[CommandOutput]]:
    runtime = Mock(tool_context=Mock(label="Agent"))
    said: list[CommandOutput] = []
    commands = Commands(
        Mock(spec=Agent, runtime=runtime),
        catalog=Mock(spec=ModelCatalog),
        say=said.append,
        screen=_View(),
        panes=Panes(),
    )
    monkeypatch.setattr(Commands, "owners", Mock(return_value=[((), commands)]))
    return commands, runtime, said


@pytest.mark.asyncio
@pytest.mark.parametrize("line", ["", "   ", "\n"])
async def test_a_blank_line_does_nothing(
    action_commands: tuple[Commands, Mock, list[CommandOutput]],
    line: str,
) -> None:
    commands, runtime, said = action_commands
    assert await commands(line) is False
    assert runtime.mock_calls == []
    assert said == []


@pytest.mark.asyncio
async def test_text_is_sent_to_the_agent_stripped(
    action_commands: tuple[Commands, Mock, list[CommandOutput]],
) -> None:
    commands, runtime, said = action_commands
    assert await commands("  fix the bug \n") is False
    assert runtime.mock_calls == [("push", (UserMessage(content="fix the bug"),), {})]
    assert said == []


@pytest.mark.asyncio
@pytest.mark.parametrize("line", ["/quit", "/exit", " /quit now"])
async def test_quit_and_exit_end_the_session(
    action_commands: tuple[Commands, Mock, list[CommandOutput]],
    line: str,
) -> None:
    commands, runtime, said = action_commands
    assert await commands(line) is True
    assert commands.done
    assert runtime.mock_calls == []
    assert said == []


@pytest.mark.asyncio
async def test_an_unknown_command_points_at_help(
    action_commands: tuple[Commands, Mock, list[CommandOutput]],
) -> None:
    commands, runtime, said = action_commands
    assert await commands("/frobnicate x") is False
    assert runtime.mock_calls == []
    assert said == [
        CommandOutput(tone="error", text="unknown command: /frobnicate; see /help"),
    ]


def _names(desk: _Desk) -> list[str]:
    return [".".join(s.name) for s in harvest(desk.commands.owners())]


def test_actions_come_first_then_the_model_the_tools_the_screen_and_mcp() -> None:
    names = _names(_Desk())
    assert names[:14] == [
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
        "login.precedence",
    ]
    assert names[14:16] == ["model", "model.provider"]
    assert names.index("tool.Echo.display.command.lines") < names.index(
        "tool.Plain.display.command.lines",
    )
    assert [n for n in names if not n.startswith("screen.theme.")][-17:] == [
        "screen.thinking",
        "screen.theme",
        "mcp.list",
        "mcp.get",
        "mcp.add",
        "mcp.add-json",
        "mcp.remove",
        "mcp.login",
        "mcp.logout",
        "host",
        "host.add",
        "host.edit",
        "host.del",
        "host.list",
        "remote.add",
        "remote.del",
        "remote.list",
    ]


def test_each_action_says_how_it_is_typed_and_what_it_does() -> None:
    assert [(a.name, a.usage, a.help_short) for a in Commands.actions()] == [
        (("quit",), "", "end the session"),
        (("exit",), "", "end the session"),
        (("help",), "[NAME]", "this list, or all one setting says"),
        (("clear",), "", "wipe context (the journal keeps it)"),
        (("compact",), "[hints]", "compact history"),
        (("recompact",), "[hints]", "compact history"),
        (("halt",), "[AGENTS]", "stop the model call (Ctrl+C), or the agents'"),
        (("kill",), "ID|LABEL|all", "cancel tool calls, or stop a persistent child"),
        (("send",), "AGENTS TEXT", "message agents, or run a /COMMAND on them"),
        (("focus",), "LABEL", "show another agent; what you type goes to it"),
        (("defer",), "TEXT", "send once the agent idles (Tab)"),
        (("tasks",), "", "list running tool calls"),
        (
            ("login",),
            "[--headless|--no-headless]",
            "sign in to the model's provider",
        ),
        (
            ("login", "precedence"),
            "[PROVIDER] ACCOUNT,...|off",
            "the order a provider's logins take the model, as quotas run out",
        ),
    ]


def test_the_actions_that_say_more_say_it() -> None:
    longer = {".".join(a.name): a.help_long for a in Commands.actions() if a.help_long}
    assert sorted(longer) == ["focus", "login", "login.precedence", "send"]
    assert longer["send"] == (
        "AGENTS: a label, parent, children, ancestors, descendants, all, a*glob,"
        " {a,b}, or /regex/. /send parent from a root with no parent starts its"
        " watcher."
    )
    assert longer["focus"] == "Left on an empty line picks one from the tree."
    assert longer["login.precedence"] == (
        "ACCOUNTs, comma separated; default is the default login. The first with"
        " quota left signs in, at each turn's start and when a call spends a"
        " window; an account chosen with /model.provider.account comes first. off"
        " orders default, then each account as added. PROVIDER is the model's"
        " when left out."
    )
    assert longer["login"] == slash._login_parser().format_help()


def test_the_login_action_reads_flags_at_the_prompt_and_the_shell() -> None:
    commands = Commands(
        Mock(spec=Agent, runtime=Mock(tool_context=Mock(label="Agent"))),
        catalog=Mock(spec=ModelCatalog),
        say=Mock(),
        screen=_View(),
        panes=Panes(),
    )
    settings = commands.settings()
    (login,) = (s for s in settings if s.name == ("login",))
    parser = slash._login_parser()
    assert (login.cli, login.flags) == (True, parser.options)
    assert login.help_long == parser.format_help()
    assert {"--headless", "--no-headless"} <= parser.options
    assert (
        "Sign in to the model's provider, as --provider, --account and"
        " --allow-providers say."
    ) in " ".join(parser.format_help().split())
    others = [s for s in settings if s.name != ("login",)]
    assert {(s.cli, s.flags) for s in others} == {(False, frozenset())}


def _setting(*, usage: str = "", said: str = "", fails: bool = False) -> Setting:
    async def apply(args: str) -> str:
        if fails:
            raise SettingError(f"no {args}")
        return said

    return Setting(name=("x",), help_short="what x is", usage=usage, apply=apply)


@pytest.mark.parametrize(
    ("usage", "row"),
    [
        ("", f"  {'/x':<32s}what x is"),
        ("u" * 28, f"  {'/x ' + 'u' * 28:<32s}what x is"),
        ("u" * 29, f"  /x {'u' * 29}\n  {'':<32s}what x is"),
    ],
)
def test_a_help_row_wraps_a_name_too_long_for_its_column(usage: str, row: str) -> None:
    assert slash._listed(_setting(usage=usage), lambda n: f"/{'.'.join(n)}") == row


def test_a_long_name_with_no_usage_ends_where_its_name_does() -> None:
    long = "/" + "n" * 40
    assert slash._listed(_setting(), lambda _: long) == (
        f"  {long}\n  {'':<32s}what x is"
    )


@pytest.mark.asyncio
async def test_a_child_keeps_the_choice_it_made_before_its_commands_were() -> None:
    desk, screens = _rolled()
    team = desk.team
    assert team is not None
    await desk.commands("/screen.thinking hide")
    team.join(
        Agent(model=desk.catalog["scripted"]["scripted"]),
        label="late",
        parent="Agent",
    )
    team.choose("late", "screen.thinking", "show")

    _ = await desk.commands.child_commands(team, "late")

    assert _shown(screens) == {"Agent": False, "late": True}


@pytest.mark.asyncio
async def test_applying_a_setting_says_what_it_did_or_why_not() -> None:
    said: list[CommandOutput] = []
    assert await slash.apply_setting(_setting(said="  done \n"), "a", say=said.append)
    assert not await slash.apply_setting(_setting(fails=True), "a", say=said.append)
    assert said == [
        CommandOutput(text="  done", tone="note", command="/x"),
        CommandOutput(text="no a", tone="error", command="/x"),
    ]


def test_help_on_a_setting_names_it_alone_and_says_more_only_if_it_has_more() -> None:
    settings = [
        _setting(),
        replace(_setting(), name=("more",), usage="VALUE", help_long="Longer help."),
    ]
    assert slash.help_text(settings, "x") == "/x\nwhat x is"
    assert slash.help_text(settings, "more").splitlines() == [
        "/more VALUE",
        "what x is",
        "Longer help.",
    ]


def test_help_on_a_dotted_group_lists_its_settings() -> None:
    settings = [
        replace(_setting(), name=("tool", "Echo", "second")),
        replace(_setting(), name=("tool", "Other", "first")),
        replace(_setting(), name=("tool", "Echo", "first")),
    ]
    assert slash.help_text(settings, "tool.Echo").splitlines() == [
        "tool.Echo commands",
        f"  {'/tool.Echo.second':<32s}what x is",
        f"  {'/tool.Echo.first':<32s}what x is",
    ]


def test_the_runtime_and_mcp_settings_act_on_this_agent(tmp_path: Path) -> None:
    desk = _Desk()
    owners = dict(desk.commands.owners())
    runtime = owners[("runtime",)]
    assert isinstance(runtime, RuntimeSettings)
    assert runtime.runtime is desk.agent.runtime
    mcp = owners[("mcp",)]
    assert isinstance(mcp, McpSettings)
    assert mcp.cwd == desk.agent.runtime.tool_context.origin
    _ = mcp.show("hi")
    assert desk.printer.said == [("note", "hi")]
    (_, alone), *_ = owners_without_agent(cwd=tmp_path, say=desk.commands.say)
    assert isinstance(alone, McpSettings)
    assert alone.cwd == tmp_path


@pytest.fixture
def help_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Commands, list[CommandOutput]]:
    said: list[CommandOutput] = []
    commands = Commands(
        Mock(spec=Agent, runtime=Mock(tool_context=Mock(label="Agent"))),
        catalog=Mock(spec=ModelCatalog),
        say=said.append,
        screen=_View(),
        panes=Panes(),
    )
    model = Mock(
        settings=Mock(
            return_value=[
                replace(
                    _setting(),
                    name=("thinking", "effort"),
                    usage="VALUE",
                    help_short="how hard the model reasons",
                    value=lambda: "none (default)",
                    options=lambda: ("default", "none"),
                ),
            ],
        ),
    )
    mcp = Mock(
        settings=Mock(
            return_value=[
                replace(_setting(), name=("list",), help_short="list servers"),
                replace(
                    _setting(),
                    name=("add",),
                    usage="NAME",
                    help_short="add a server",
                ),
            ],
        ),
    )
    owners: list[tuple[tuple[str, ...], HasSettings]] = [
        ((), commands),
        (("model",), model),
        (("mcp",), mcp),
    ]
    monkeypatch.setattr(Commands, "owners", Mock(return_value=owners))
    return commands, said


@pytest.mark.asyncio
async def test_help_lists_every_setting_with_its_usage(
    help_commands: tuple[Commands, list[CommandOutput]],
) -> None:
    commands, said = help_commands
    assert await commands("/help") is False
    (output,) = said
    assert (output.tone, output.command) == ("note", "/help")
    rows = output.text.split("\n")
    assert rows[0] == "sagent commands"
    assert f"  {'/compact [hints]':<32s}compact history" in rows
    assert f"  {'/model.thinking.effort VALUE':<32s}how hard the model reasons" in rows
    listed = [row.split()[0] for row in rows if row.startswith("  /")]
    assert listed == [
        *(f"/{'.'.join(action.name)}" for action in Commands.actions()),
        "/model.thinking.effort",
        "/mcp.list",
        "/mcp.add",
    ]


@pytest.mark.asyncio
async def test_help_on_one_setting_says_all_it_knows(
    help_commands: tuple[Commands, list[CommandOutput]],
) -> None:
    commands, said = help_commands
    assert await commands("/help eff") is False
    assert said == [
        CommandOutput(
            tone="note",
            command="/help",
            text=(
                "/model.thinking.effort VALUE\nhow hard the model reasons"
                "\nnow: none (default)   (default, none)"
            ),
        ),
    ]
    assert await commands("/help nothing-at-all") is False
    assert said[-1] == CommandOutput(
        tone="error",
        command="/help",
        text="no setting 'nothing-at-all'",
    )


def test_what_an_owner_with_no_agent_shows_is_said_as_a_note(tmp_path: Path) -> None:
    said: list[CommandOutput] = []
    owners = dict(owners_without_agent(cwd=tmp_path, say=said.append))
    assert list(owners) == [("mcp",), ("host",), ("remote",)]
    mcp, host = owners[("mcp",)], owners[("host",)]
    assert isinstance(mcp, McpSettings)
    assert isinstance(host, HostSettings)
    mcp.show("Open https://sign.in")
    host.show("Serving")
    assert said == [
        CommandOutput(text="Open https://sign.in", tone="note"),
        CommandOutput(text="Serving", tone="note"),
    ]


@pytest.mark.asyncio
async def test_help_on_a_group_lists_its_settings(
    help_commands: tuple[Commands, list[CommandOutput]],
) -> None:
    commands, said = help_commands
    assert await commands("/help mcp") is False
    assert said == [
        CommandOutput(
            tone="note",
            command="/help",
            text=(
                "mcp commands\n"
                f"  {'/mcp.list':<32s}list servers\n"
                f"  {'/mcp.add NAME':<32s}add a server"
            ),
        ),
    ]


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
    assert desk.printer.said == [("note", "[/screen.thinking] hide (default)")]


@pytest.mark.asyncio
async def test_clear_runs_the_session_clear_as_a_task(
    session_commands: tuple[Commands, AsyncMock, AsyncMock, list[CommandOutput]],
) -> None:
    commands, clear, compact, said = session_commands
    assert await commands("/clear") is False
    (task,) = commands._tasks
    clear.assert_not_awaited()
    assert said == [
        CommandOutput(tone="note", command="/clear", text="history cleared"),
    ]
    await task
    clear.assert_awaited_once_with()
    compact.assert_not_called()
    assert commands._tasks == set()
    assert said == [
        CommandOutput(tone="note", command="/clear", text="history cleared"),
    ]


@pytest.fixture
def session_commands(
    action_commands: tuple[Commands, Mock, list[CommandOutput]],
) -> tuple[Commands, AsyncMock, AsyncMock, list[CommandOutput]]:
    commands, runtime, said = action_commands
    clear, compact = AsyncMock(), AsyncMock()
    commands.agent = Mock(
        spec=Agent,
        runtime=runtime,
        session=Mock(clear=clear, compactor=Mock(run=compact)),
    )
    return commands, clear, compact, said


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("line", "note", "hints"),
    [
        ("/compact", "queued", ""),
        ("/compact keep the API", "queued (keep the API)", "keep the API"),
        ("/recompact  x ", "queued (x)", "x"),
    ],
)
async def test_compact_queues_and_a_failure_prints_under_its_name(
    session_commands: tuple[Commands, AsyncMock, AsyncMock, list[CommandOutput]],
    line: str,
    note: str,
    hints: str,
) -> None:
    commands, clear, compact, said = session_commands
    error = RuntimeError("cannot compact")
    compact.side_effect = error
    assert await commands(line) is False
    (task,) = commands._tasks
    assert said == [
        CommandOutput(tone="note", command=line.split(maxsplit=1)[0], text=note),
    ]
    await asyncio.gather(task, return_exceptions=True)
    compact.assert_awaited_once_with(hints)
    clear.assert_not_called()
    assert task.exception() is error
    assert said[-1] == CommandOutput(
        tone="error",
        command="/compact",
        text="RuntimeError: cannot compact",
    )
    assert len(said) == 2
    assert commands._tasks == set()


@pytest.mark.asyncio
async def test_compact_hands_its_hints_to_the_compactor(
    session_commands: tuple[Commands, AsyncMock, AsyncMock, list[CommandOutput]],
) -> None:
    commands, clear, compact, said = session_commands
    assert await commands("/compact keep the API") is False
    (task,) = commands._tasks
    await task
    compact.assert_awaited_once_with("keep the API")
    clear.assert_not_called()
    assert said == [
        CommandOutput(tone="note", command="/compact", text="queued (keep the API)"),
    ]
    assert commands._tasks == set()


@pytest.mark.asyncio
async def test_a_failed_clear_prints_under_its_name(
    session_commands: tuple[Commands, AsyncMock, AsyncMock, list[CommandOutput]],
) -> None:
    commands, clear, compact, said = session_commands
    error = RuntimeError("no")
    clear.side_effect = error
    assert await commands("/clear") is False
    (task,) = commands._tasks
    await asyncio.gather(task, return_exceptions=True)
    clear.assert_awaited_once_with()
    compact.assert_not_called()
    assert task.exception() is error
    assert commands._tasks == set()
    assert said == [
        CommandOutput(tone="note", command="/clear", text="history cleared"),
        CommandOutput(tone="error", command="/clear", text="RuntimeError: no"),
    ]


@pytest.mark.asyncio
async def test_aclose_cancels_running_commands_quietly(
    session_commands: tuple[Commands, AsyncMock, AsyncMock, list[CommandOutput]],
) -> None:
    commands, clear, compact, said = session_commands
    started = asyncio.Event()

    async def waiting() -> None:
        started.set()
        await asyncio.Future[None]()

    clear.side_effect = waiting
    try:
        assert await commands("/clear") is False
        (task,) = commands._tasks
        await started.wait()
        clear.assert_awaited_once_with()
    finally:
        await commands.aclose()
    assert task.cancelled()
    assert commands._tasks == set()
    compact.assert_not_called()
    assert said == [
        CommandOutput(tone="note", command="/clear", text="history cleared"),
    ]


@pytest.mark.asyncio
async def test_a_bare_halt_halts_this_agent(
    action_commands: tuple[Commands, Mock, list[CommandOutput]],
) -> None:
    commands, runtime, said = action_commands
    assert await commands("/halt") is False
    assert runtime.mock_calls == [("halt", (), {})]
    assert said == []


@pytest.mark.asyncio
async def test_defer_queues_text_and_refuses_an_empty_message(
    action_commands: tuple[Commands, Mock, list[CommandOutput]],
) -> None:
    commands, runtime, said = action_commands
    assert await commands("/defer  later please ") is False
    assert await commands("/defer") is False
    assert runtime.mock_calls == [
        ("push", (types.UserDeferredMessage(text="later please"),), {}),
    ]
    assert said == [CommandOutput(tone="error", command="/defer", text="needs TEXT")]


@pytest.mark.asyncio
async def test_kill_cancels_one_call_or_all_and_all_empties_the_panes(
    action_commands: tuple[Commands, Mock, list[CommandOutput]],
) -> None:
    commands, runtime, said = action_commands
    first, second = Mock(spec=types.Job), Mock(spec=types.Job)
    runtime.jobs = {"c1": first, "c2": second}
    commands.panes.stage("queued", deferred=False)
    commands.panes.stage("later", deferred=True)
    assert await commands("/kill c1") is False
    assert first.mock_calls == [("cancel", (), {})]
    assert second.mock_calls == []
    assert (commands.panes.queue, commands.panes.deferred) == (
        Staged(text="queued"),
        Staged(text="later"),
    )
    first.reset_mock()
    assert await commands("/kill all") is False
    assert first.mock_calls == []
    assert second.mock_calls == []
    assert (commands.panes.queue, commands.panes.deferred) == (None, None)
    assert said == [
        CommandOutput(tone="note", command="/kill", text="cancelled c1"),
        CommandOutput(tone="note", command="/kill", text="cancelled all tool calls"),
    ]
    assert runtime.mock_calls == [("push", (types.Kill(),), {})]


@pytest.mark.asyncio
async def test_kill_needs_a_target_it_knows(
    action_commands: tuple[Commands, Mock, list[CommandOutput]],
) -> None:
    commands, runtime, said = action_commands
    runtime.jobs = dict[str, types.Job]()
    assert await commands("/kill") is False
    assert await commands("/kill nope") is False
    assert runtime.mock_calls == []
    assert said == [
        CommandOutput(
            tone="error",
            command="/kill",
            text="needs an ID, a LABEL, or all",
        ),
        CommandOutput(
            tone="error",
            command="/kill",
            text="no running call 'nope'; see /tasks",
        ),
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
async def test_tasks_lists_running_calls_and_persistent_children(
    action_commands: tuple[Commands, Mock, list[CommandOutput]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands, runtime, said = action_commands
    monkeypatch.setattr(time, "time", lambda: 100.0)
    runtime.jobs = {
        "c2": Mock(
            spec=types.Job,
            call=call("Echo", "c2"),
            started=95.0,
            state="running",
        ),
        "c1": Mock(
            spec=types.Job,
            call=call("Read", "c1"),
            started=93.0,
            state="sleeping",
        ),
    }
    persistent = Mock(return_value=["reviewer", "writer"])
    commands.team = MagicMock(
        spec=Team,
        persistent=persistent,
        watched=Mock(return_value=None),
    )
    assert await commands("/tasks") is False
    (output,) = said
    assert (output.tone, output.command) == ("note", "/tasks")
    head, *rows = output.text.split("\n")
    assert head == "2 running tool call(s), 2 persistent agent(s)"
    assert [row.split() for row in rows] == [
        ["c2", "Echo", "running", "5s"],
        ["c1", "Read", "sleeping", "7s"],
        ["reviewer", "persistent", "agent"],
        ["writer", "persistent", "agent"],
    ]
    persistent.assert_called_once_with()
    assert runtime.mock_calls == []


@pytest.mark.asyncio
async def test_tasks_with_nothing_running_says_so(
    action_commands: tuple[Commands, Mock, list[CommandOutput]],
) -> None:
    commands, runtime, said = action_commands
    runtime.jobs = dict[str, types.Job]()
    assert await commands("/tasks") is False
    assert said == [
        CommandOutput(
            tone="note",
            command="/tasks",
            text="0 running tool call(s), 0 persistent agent(s)",
        ),
    ]
    assert runtime.mock_calls == []


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
    action_commands: tuple[Commands, Mock, list[CommandOutput]],
    pattern: str,
    halted: list[str],
) -> None:
    commands, runtime, said = action_commands
    runtimes = _target_runtimes(commands, "reviewer", "Agent_0", "Agent_1")
    assert await commands(f"/halt {pattern}") is False
    in_order = [label for label in runtimes if label in halted]
    for label, target in runtimes.items():
        assert target.mock_calls == ([("halt", (), {})] if label in halted else [])
    assert said == [
        CommandOutput(
            tone="note",
            command="/halt",
            text=f"halted {', '.join(in_order)}",
        ),
    ]
    assert runtime.mock_calls == []


def _target_runtimes(commands: Commands, *labels: str) -> dict[str, Mock]:
    runtimes = {label: Mock() for label in labels}
    members = {
        commands.agent.runtime.tool_context.label: commands.agent,
        **{
            label: Mock(spec=Agent, runtime=target)
            for label, target in runtimes.items()
        },
    }
    commands.team = MagicMock(
        spec=Team,
        __iter__=Mock(side_effect=members.__iter__),
        __contains__=Mock(side_effect=members.__contains__),
        __getitem__=Mock(side_effect=members.__getitem__),
        watched=Mock(return_value=None),
    )
    return runtimes


@pytest.mark.asyncio
@pytest.mark.parametrize("pattern", ["Agent", "all", "Ag*", "{Agent}", "/Ag/"])
async def test_no_pattern_reaches_this_agent(
    action_commands: tuple[Commands, Mock, list[CommandOutput]],
    pattern: str,
) -> None:
    commands, runtime, said = action_commands
    _target_runtimes(commands)
    assert await commands(f"/halt {pattern}") is False
    assert runtime.mock_calls == []
    assert said == [
        CommandOutput(
            tone="error",
            command="/halt",
            text=f"no agent {pattern!r}; agents: none",
        ),
    ]


@pytest.mark.asyncio
async def test_an_unknown_target_names_the_agents_there_are(
    action_commands: tuple[Commands, Mock, list[CommandOutput]],
) -> None:
    commands, runtime, said = action_commands
    targets = _target_runtimes(commands, "b", "a")
    assert await commands("/halt zed") is False
    assert said == [
        CommandOutput(
            tone="error",
            command="/halt",
            text="no agent 'zed'; agents: a, b",
        ),
    ]
    assert runtime.mock_calls == []
    assert all(target.mock_calls == [] for target in targets.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("pattern", ["{bb", "bb}"])
async def test_a_half_brace_is_a_label(
    action_commands: tuple[Commands, Mock, list[CommandOutput]],
    pattern: str,
) -> None:
    """Read as braces, either would name ``b``, the label between them."""
    commands, runtime, said = action_commands
    targets = _target_runtimes(commands, "b")
    assert await commands(f"/halt {pattern}") is False
    assert runtime.mock_calls == []
    assert targets["b"].mock_calls == []
    assert said == [
        CommandOutput(
            tone="error",
            command="/halt",
            text=f"no agent {pattern!r}; agents: b",
        ),
    ]


@pytest.mark.asyncio
async def test_a_lone_agent_has_no_team_to_reach(
    action_commands: tuple[Commands, Mock, list[CommandOutput]],
) -> None:
    commands, runtime, said = action_commands
    assert await commands("/halt x") is False
    assert runtime.mock_calls == []
    assert said == [
        CommandOutput(tone="error", command="/halt", text="no agent 'x'; agents: none"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("pattern", ["/", "//"])
async def test_a_regex_of_nothing_is_a_label_not_everything(
    action_commands: tuple[Commands, Mock, list[CommandOutput]],
    pattern: str,
) -> None:
    commands, runtime, said = action_commands
    targets = _target_runtimes(commands, "a")
    assert await commands(f"/halt {pattern}") is False
    assert said == [
        CommandOutput(
            tone="error",
            command="/halt",
            text=f"no agent {pattern!r}; agents: a",
        ),
    ]
    assert runtime.mock_calls == []
    assert targets["a"].mock_calls == []


@pytest.mark.asyncio
async def test_a_bad_regex_says_why(
    action_commands: tuple[Commands, Mock, list[CommandOutput]],
) -> None:
    commands, runtime, said = action_commands
    targets = _target_runtimes(commands, "a")
    assert await commands("/halt /(/") is False
    (output,) = said
    assert (output.tone, output.command) == ("error", "/halt")
    assert output.text.startswith("invalid target regex:")
    assert runtime.mock_calls == []
    assert targets["a"].mock_calls == []


@pytest.mark.asyncio
async def test_send_messages_each_agent_named(
    action_commands: tuple[Commands, Mock, list[CommandOutput]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands, runtime, said = action_commands
    targets = _target_runtimes(commands, "a", "b")
    sent = Mock()
    monkeypatch.setattr(commands.team, "send", sent)
    assert await commands("/send all  look at  this ") is False
    assert sent.call_args_list == [
        (("user", "a", "look at  this"), {}),
        (("user", "b", "look at  this"), {}),
    ]
    assert said == [
        CommandOutput(tone="note", command="/send", text="user→a, b: look at  this"),
    ]
    assert runtime.mock_calls == []
    assert all(target.mock_calls == [] for target in targets.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("line", ["/send", "/send a", "/send a   "])
async def test_send_needs_a_target_and_text(
    action_commands: tuple[Commands, Mock, list[CommandOutput]],
    monkeypatch: pytest.MonkeyPatch,
    line: str,
) -> None:
    commands, runtime, said = action_commands
    targets = _target_runtimes(commands, "a")
    sent = Mock()
    monkeypatch.setattr(commands.team, "send", sent)
    assert await commands(line) is False
    sent.assert_not_called()
    assert runtime.mock_calls == []
    assert targets["a"].mock_calls == []
    assert said == [
        CommandOutput(tone="error", command="/send", text="needs AGENTS TEXT"),
    ]


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
async def test_focus_refuses_with_no_screen_and_an_unknown_label(
    action_commands: tuple[Commands, Mock, list[CommandOutput]],
) -> None:
    commands, runtime, said = action_commands
    targets = _target_runtimes(commands, "a")
    assert await commands("/focus a") is False
    shown: list[str] = []
    commands.focus = shown.append
    assert await commands("/focus nobody") is False
    assert await commands("/focus a") is False
    assert shown == ["a"]
    assert runtime.mock_calls == []
    assert targets["a"].mock_calls == []
    assert said == [
        CommandOutput(
            tone="error",
            command="/focus",
            text="there is no screen to show another agent on",
        ),
        CommandOutput(
            tone="error",
            command="/focus",
            text="no agent 'nobody'; agents: Agent, a",
        ),
    ]


@pytest.mark.asyncio
async def test_focus_on_a_lone_agent_lists_no_agents(
    action_commands: tuple[Commands, Mock, list[CommandOutput]],
) -> None:
    commands, runtime, said = action_commands
    shown: list[str] = []
    commands.focus = shown.append
    assert await commands("/focus a") is False
    assert shown == []
    assert runtime.mock_calls == []
    assert said == [
        CommandOutput(
            tone="error",
            command="/focus",
            text="no agent 'a'; agents: none",
        ),
    ]


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
    # Only the first dot names the agent: the rest is the setting's own.
    await desk.commands("/a.model.thinking.effort")
    tone, said = desk.printer.said[-1]
    assert (tone, said.startswith("[/model.thinking.effort] ")) == ("note", True)


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
        await desk.commands("/parent.watcher on")
        await desk.commands("/parent.watcher")
        assert team.watching == {"watcher": True}
        await desk.commands("/parent.watcher off")
        await desk.commands("/parent.watcher")
        await desk.commands("/parent.help watcher")

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
        assert list(watcher.tools) == [*desk.agent.tools, "AgentSend"]
    assert desk.printer.said == [
        (
            "note",
            "[/send] started watcher, watching Agent\nuser→watcher: keep it honest",
        ),
        ("note", "[/send] user→watcher: and brief"),
        ("note", "[/watcher] on"),
        ("note", "[/watcher] on   (on, off)"),
        ("note", "[/watcher] off"),
        ("note", "[/watcher] off   (on, off)"),
        (
            "note",
            (
                "[/help] /watcher on|off\nwhether watcher is sent what Agent does"
                "\nnow: off   (on, off)"
            ),
        ),
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
async def test_a_choice_rolls_down_to_each_child_that_made_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def owners(commands: Commands) -> list[tuple[tuple[str, ...], HasSettings]]:
        return [((), commands), (("screen",), ScreenSettings(view=commands.screen))]

    monkeypatch.setattr(Commands, "owners", owners)
    desk, screens = _rolled(("a", "Agent"), ("b", "Agent"), ("c", "Agent"), ("a1", "a"))
    team = desk.team
    assert team is not None
    await desk.commands("/send b /screen.thinking show")

    await desk.commands("/screen.thinking hide")

    assert _shown(screens) == {
        "Agent": False,
        "a": False,
        "a1": False,
        "b": True,
        "c": False,
    }
    assert team.choices == {
        "b": {"screen.thinking": "show"},
        "Agent": {"screen.thinking": "hide"},
    }
    # Commands made by now take a later choice by its roll alone.
    await desk.commands("/screen.thinking show")
    await desk.commands("/send b /screen.thinking hide")
    assert _shown(screens) == {
        "Agent": True,
        "a": True,
        "a1": True,
        "b": False,
        "c": True,
    }


@pytest.mark.asyncio
async def test_default_on_a_child_takes_its_parent_s_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def owners(commands: Commands) -> list[tuple[tuple[str, ...], HasSettings]]:
        return [((), commands), (("screen",), ScreenSettings(view=commands.screen))]

    monkeypatch.setattr(Commands, "owners", owners)
    desk, screens = _rolled(("a", "Agent"), ("a1", "a"))
    team = desk.team
    assert team is not None
    # The parent's choice is not a fresh screen's, which hides.
    await desk.commands("/screen.thinking show")
    await desk.commands("/send a /screen.thinking hide")
    assert _shown(screens) == {"Agent": True, "a": False, "a1": False}

    await desk.commands("/send a /screen.thinking default")

    assert _shown(screens) == {"Agent": True, "a": True, "a1": True}
    assert "a" not in team.choices or team.choices["a"] == {}


@pytest.mark.asyncio
async def test_default_on_a_root_is_a_fresh_one_and_rolls_down() -> None:
    desk, screens = _rolled(("a", "Agent"))
    team = desk.team
    assert team is not None
    await desk.commands("/screen.thinking show")

    await desk.commands("/screen.thinking default")

    assert _shown(screens) == {"Agent": False, "a": False}
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

    _ = await desk.commands.child_commands(team, "late")

    assert _shown(screens) == {"Agent": False, "late": False}


@pytest.mark.asyncio
async def test_a_choice_can_be_taken_without_a_team_to_roll_it_to(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applied = AsyncMock(return_value="taken")
    owner = Mock(
        settings=Mock(
            return_value=[
                _setting(),
                Setting(
                    name=("choice",),
                    help_short="A choice.",
                    apply=applied,
                ),
            ],
        ),
    )
    owners = Mock(return_value=[((), owner)])
    monkeypatch.setattr(Commands, "owners", owners)
    said: list[CommandOutput] = []
    commands = Commands(
        Mock(spec=Agent),
        catalog=Mock(spec=ModelCatalog),
        say=said.append,
        screen=_View(),
        panes=Panes(),
    )

    await commands.take("choice", "chosen")

    applied.assert_awaited_once_with("chosen")
    assert commands.team is None
    assert said == []


@pytest.mark.asyncio
async def test_a_choice_a_child_cannot_take_says_why_and_stops_there() -> None:
    """A setting with no closest value, such as a screen's, cannot roll on."""
    desk, screens = _rolled(("a", "Agent"))
    child = await desk.commands.child_commands(
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
async def test_send_without_a_team_has_no_one_to_reach(
    action_commands: tuple[Commands, Mock, list[CommandOutput]],
) -> None:
    commands, runtime, said = action_commands
    assert await commands("/send all hi") is False
    assert runtime.mock_calls == []
    assert said == [
        CommandOutput(tone="error", command="/send", text="there are no other agents"),
    ]


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
    assert desk.pushed(members["a"]) == [types.Kill()]
    assert desk.panes.queue == Staged(text="mine")
    assert desk.printer.said == [
        ("note", "[/halt] halted b"),
        ("note", "[/kill] cancelled all tool calls"),
        ("error", "[/halt] no agent 'a'; agents: Agent, b"),
    ]


@pytest.mark.asyncio
async def test_send_to_no_one_says_so(
    action_commands: tuple[Commands, Mock, list[CommandOutput]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands, runtime, said = action_commands
    targets = _target_runtimes(commands, "a")
    sent = Mock()
    monkeypatch.setattr(commands.team, "send", sent)
    assert await commands("/send zed hi") is False
    sent.assert_not_called()
    assert runtime.mock_calls == []
    assert targets["a"].mock_calls == []
    assert said == [
        CommandOutput(tone="error", command="/send", text="no agent 'zed'; agents: a"),
    ]


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


@pytest.mark.asyncio
async def test_peer_participation_is_chosen_inherited_and_validated() -> None:
    desk, _ = _teamed("child")
    team = desk.team
    assert team is not None
    await desk.commands("/peer-messaging off")
    assert not team.peer_messaging("Agent")
    child = await desk.commands.child_commands(team, "child")
    await child("/peer-messaging")
    assert desk.printer.said[-1] == (
        "note",
        "[/peer-messaging] off   (on, off, default)",
    )
    await child("/peer-messaging on")
    assert team.peer_messaging("child")
    await child("/peer-messaging default")
    assert not team.peer_messaging("child")
    await child("/peer-messaging invalid")
    assert desk.printer.said[-1][0] == "error"
    await desk.commands("/peer-messaging default")
    assert team.peer_messaging("child")


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
async def test_login_precedence_orders_the_logins_of_the_model_s_provider() -> None:
    desk = _Desk(provider="signing")
    provider = desk.agent.model.provider
    assert isinstance(provider, _Signing)
    for account in ("default", "work", "home"):
        provider.logins.path(account).parent.mkdir(parents=True, exist_ok=True)
        provider.logins.path(account).write_text("{}")
    await desk.commands("/login.precedence")
    await desk.commands("/login.precedence work,default")
    assert provider.logins.precedence() == ("work", "default")
    await desk.commands("/login.precedence signing home")
    assert provider.logins.order() == ("home",)
    await desk.commands("/login.precedence off")
    assert provider.logins.precedence() is None
    assert desk.printer.said == [
        ("note", "[/login.precedence] default,home,work"),
        ("note", "[/login.precedence] signing: work,default"),
        ("note", "[/login.precedence] signing: home"),
        ("note", "[/login.precedence] signing: default,home,work"),
    ]


@pytest.mark.asyncio
async def test_login_precedence_refuses_a_provider_that_keeps_no_logins() -> None:
    desk = _Desk()
    await desk.commands("/login.precedence")
    await desk.commands("/login.precedence work")
    await desk.commands("/login.precedence a b c")
    assert desk.printer.said == [
        ("note", "[/login.precedence] "),
        ("error", "[/login.precedence] scripted keeps no logins to order"),
        ("error", "[/login.precedence] give [PROVIDER] ACCOUNT,... or [PROVIDER] off"),
    ]


@pytest.mark.asyncio
async def test_login_precedence_takes_a_provider_by_any_name_it_answers_to() -> None:
    desk = _Desk(provider="signing")
    provider = desk.agent.model.provider
    assert isinstance(provider, _Signing)
    for account in ("default", "work"):
        provider.logins.path(account).parent.mkdir(parents=True, exist_ok=True)
        provider.logins.path(account).write_text("{}")
    await desk.commands("/login.precedence sgn work")
    await desk.commands("/login.precedence nope work")
    assert provider.logins.precedence() == ("work",)
    assert desk.printer.said[0] == ("note", "[/login.precedence] signing: work")
    kind, text = desk.printer.said[1]
    assert kind == "error"
    assert "unknown provider 'nope'" in text


@pytest.mark.asyncio
async def test_login_precedence_reads_a_list_as_a_person_types_it() -> None:
    """A space after a comma is no provider, nor is an account a provider abbreviates."""
    desk = _Desk(provider="signing")
    provider = desk.agent.model.provider
    assert isinstance(provider, _Signing)
    for account in ("default", "work", "gs"):
        provider.logins.path(account).parent.mkdir(parents=True, exist_ok=True)
        provider.logins.path(account).write_text("{}")
    await desk.commands("/login.precedence work, default")
    assert provider.logins.precedence() == ("work", "default")
    await desk.commands("/login.precedence default ,work")
    assert provider.logins.precedence() == ("default", "work")
    await desk.commands("/login.precedence gs,work")
    assert provider.logins.precedence() == ("gs", "work")
    await desk.commands("/login.precedence zed,work")
    assert desk.printer.said[-1] == (
        "note",
        "[/login.precedence] signing: work (no login saved: zed)",
    )
    await desk.commands("/login.precedence zed,yo,work")
    assert desk.printer.said[-1][1].endswith("(no login saved: zed,yo)")
    # A comma after spaces still lists accounts; it names no provider.
    await desk.commands("/login.precedence work  ,default")
    assert provider.logins.precedence() == ("work", "default")


@pytest.mark.asyncio
async def test_login_precedence_says_saved_of_a_login_signed_out() -> None:
    """Its file is there; signed out, it is passed over for now, not missing."""
    desk = _Desk(provider="signing")
    provider = desk.agent.model.provider
    assert isinstance(provider, _Signing)
    for account in ("default", "work"):
        provider.logins.path(account).parent.mkdir(parents=True, exist_ok=True)
        provider.logins.path(account).write_text("{}")
    await provider.logins.sign_out("work")
    await desk.commands("/login.precedence work,default")
    assert desk.printer.said == [("note", "[/login.precedence] signing: default")]


@pytest.mark.asyncio
@pytest.mark.parametrize("line", ["work default", "off,work", "work,off"])
async def test_login_precedence_refuses_a_list_it_cannot_read(line: str) -> None:
    """Spaces with no comma, or ``off`` among accounts, would be read as something else."""
    desk = _Desk(provider="signing")
    provider = desk.agent.model.provider
    assert isinstance(provider, _Signing)
    await desk.commands(f"/login.precedence {line}")
    assert provider.logins.precedence() is None
    ((kind, text),) = desk.printer.said
    assert kind == "error"
    assert text.endswith("give [PROVIDER] ACCOUNT,... or [PROVIDER] off")


class _NativeSigning(_Picky):
    """A native sign-in with no managed credential files."""

    signed_in = 0

    async def login(self, *, headless: bool = False) -> None:
        del headless
        self.signed_in += 1


def _with_native(account: str | None) -> dict[str, ProviderFactory]:
    return {
        **_per_account(account),
        "native": partial(_NativeSigning, serves=frozenset({"native", "scripted"})),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("selected", "allowed"),
    [(None, True), ("default", True), ("work", False)],
)
async def test_native_login_refuses_named_accounts_before_authentication(
    selected: str | None,
    *,
    allowed: bool,
) -> None:
    desk = _Desk(
        provider="native",
        catalog=ModelCatalog(account=selected, factories=_with_native),
    )
    await desk.commands("/login --headless")
    provider = desk.catalog["native"]
    assert isinstance(provider, _NativeSigning)
    assert provider.signed_in == int(allowed)
    assert desk.catalog.account == (None if allowed else selected)
    assert desk.printer.said == [
        ("note", "[/login] native signed in")
        if allowed
        else (
            "error",
            "[/login] native supports only the default account; use /account default",
        ),
    ]


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("line", "remote", "headless"),
    [
        ("/login", True, True),
        ("/login", False, False),
        ("/login --no-headless", True, False),
        ("/login --headless", False, True),
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
    await desk.commands("/login --headless")
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
async def test_login_refuses_an_option_it_does_not_take() -> None:
    desk = _Desk(provider="signing")
    await desk.commands("/login --headles")
    provider = desk.agent.model.provider
    assert isinstance(provider, _Signing)
    assert provider.signed_in == 0
    assert desk.printer.said == [
        (
            "error",
            "[/login] unrecognized arguments: --headles; see sagent login --help",
        ),
    ]


@pytest.mark.asyncio
async def test_login_help_is_its_parsers() -> None:
    desk = _Desk(provider="signing")
    await desk.commands("/login --help")
    ((tone, said),) = desk.printer.said
    assert tone == "note"
    assert said.startswith("[/login] usage: sagent login [-h]")
    assert "[--headless | --no-headless]" in said
    provider = desk.agent.model.provider
    assert isinstance(provider, _Signing)
    assert provider.signed_in == 0


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
