"""Slash commands: every setting and action a user reaches, harvested from its owner.

``/NAME ARGS`` runs the ``Setting`` ``NAME`` matches: exactly, by its last
segments, or by its letters in order (``/effort``, ``/eff`` and
``/model.thinking.effort`` are one setting). A bare setting shows its value
and what it may take; a bare action runs. The owners, in the order a name
several match picks from:

::

    (actions)        quit, exit, help, clear, compact, halt, kill, send, ...
    model.*          the agent's model: AgentModel
    tool.NAME.*      each tool's display: ToolDisplaySettings
    runtime.*        when a call detaches, how a rate limit waits: RuntimeSettings
    screen.*         whether reasoning shows: ScreenSettings
    mcp.*            the MCP servers, added, removed, signed in: McpSettings

A compaction or a clear runs as a task, so the prompt stays live: ``/halt``
still reaches a long one.

A setting an agent chooses rolls down: each child that has not chosen it
takes it too, and passes it on. ``default`` on a child gives its choice up and
takes its parent's again; on a root it is a fresh model's.

``/LABEL.NAME ARGS`` is ``/send LABEL /NAME ARGS``, for a label or relation:
``/children.effort high``, ``/parent.watcher off``. ``/send parent`` from a
root with no parent starts its watcher first.

Typical usage example:

  commands = Commands(agent, catalog=catalog, say=scrollback.show, screen=printer,
                      panes=panes)
  done = await commands("/compact keep the API design")
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Literal, cast

import argparse
import asyncio
import fnmatch
import logging
import os
import re
import time

from sagent.providers.login import SIGN_IN_ERRORS, headless_wanted
from sagent.providers.types import HasLogin
from sagent.runtime import types
from sagent.session.types import UserMessage
from sagent.settings.exceptions import SettingError
from sagent.settings.flags import ActionParser
from sagent.settings.harvest import harvest
from sagent.settings.match import match
from sagent.settings.types import Setting
from sagent.tools.agent_spawn import AgentSpawn
from sagent.tools.mcp_settings import McpSettings
from sagent.tui.prompt import Panes
from sagent.tui.render import error_text
from sagent.tui.settings.model import AgentModel
from sagent.tui.settings.runtime import RuntimeSettings
from sagent.tui.settings.screen import ScreenSettings
from sagent.tui.settings.tool import ToolDisplaySettings
from sagent.tui.types import CommandOutput


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from sagent.agent.team import Team
    from sagent.agent.types import Agent
    from sagent.providers.types import ModelCatalog
    from sagent.settings.types import HasSettings
    from sagent.tui.types import ThinkingView


__all__ = [
    "Action",
    "Commands",
    "apply_setting",
    "help_text",
    "owners_without_agent",
]


logger = logging.getLogger(__name__)


# Relations ``/send`` and ``/LABEL.NAME`` resolve from the agent the commands act on.
_RELATIONS = frozenset({"parent", "children", "ancestors", "descendants"})

# Settings of one agent alone, never rolled down to its children.
_OWN = frozenset({("watcher",)})


class Commands:
    """Runs typed lines: a setting or action, or a message to the agent.

    Args:
      agent: What the commands change.
      catalog: Where the model's settings find models.
      say: Where each command says what it did.
      screen: Whether reasoning shows; ``/screen.thinking`` flips it.
      panes: What waits, emptied by ``/kill all``.
      team: The agents reachable by label; ``None`` for a lone agent.
      focus: Shows the agent a label names, for ``/focus``; ``None`` where
        there is no screen to show it.
      screens: Each agent's screen settings, by label; ``None`` shares ``screen``.

    """

    def __init__(
        self,
        agent: Agent,
        *,
        catalog: ModelCatalog,
        say: Callable[[CommandOutput], None],
        screen: ThinkingView,
        panes: Panes,
        team: Team | None = None,
        focus: Callable[[str], None] | None = None,
        screens: Callable[[str], ThinkingView] | None = None,
    ) -> None:
        self.agent = agent
        self.team = team
        self.catalog = catalog
        self.say = say
        self.screen = screen
        self.panes = panes
        self.focus = focus
        self.screens = screens
        self.done = False
        """Whether ``/quit`` or ``/exit`` ran."""
        self._tasks: set[asyncio.Task[None]] = set()
        self._children: dict[str, Commands] = {}

    def owners(self) -> list[tuple[tuple[str, ...], HasSettings]]:
        """Return what exports settings, each after where it is mounted, in order.

        Returns:
          owners: These actions, the model, each tool, the runtime, the screen,
            then the MCP servers.

        """
        tools = self.agent.runtime.tools
        return [
            ((), self),
            (("model",), AgentModel(agent=self.agent, catalog=self.catalog)),
            *(
                (("tool", name), ToolDisplaySettings(tools=tools, name=name))
                for name in tools
            ),
            (("runtime",), RuntimeSettings(runtime=self.agent.runtime)),
            (("screen",), ScreenSettings(view=self.screen)),
            *owners_without_agent(
                cwd=self.agent.runtime.tool_context.origin,
                say=self.say,
            ),
        ]

    def settings(self) -> list[Setting]:
        """Return the actions, first, so a short name reaches them before a setting.

        Returns:
          settings: Each of ``actions``, run on this agent, then ``watcher``.

        """
        return [
            Setting(
                name=a.name,
                help_short=a.help_short,
                help_long=a.help_long,
                usage=a.usage,
                value=None if a.value is None else partial(a.value, self),
                apply=partial(a.run, self),
                cli=a.cli,
                flags=a.flags,
            )
            for a in self.actions()
        ] + self._watching()

    # Listed with no agent, so the shell knows a command before it builds a model.
    # Each runs ``Commands``' own method, never a subclass's override: the shell lists
    # these before any instance exists.
    @classmethod
    def actions(cls) -> list[Action]:
        """Return each action ``quit`` to ``login``, as yet run on no agent.

        Returns:
          actions: Each action's name, help, and what runs it.

        """
        login = _login_parser()
        return [
            Action(name=("quit",), help_short="end the session", run=Commands._quit),
            Action(name=("exit",), help_short="end the session", run=Commands._quit),
            Action(
                name=("help",),
                help_short="this list, or all one setting says",
                usage="[NAME]",
                run=Commands._help,
            ),
            Action(
                name=("clear",),
                help_short="wipe context (the journal keeps it)",
                run=Commands._clear,
            ),
            Action(
                name=("compact",),
                help_short="compact history",
                usage="[hints]",
                run=Commands._compact,
            ),
            Action(
                name=("recompact",),
                help_short="compact history",
                usage="[hints]",
                run=Commands._compact,
            ),
            Action(
                name=("halt",),
                help_short="stop the model call (Ctrl+C), or the agents'",
                usage="[AGENTS]",
                run=Commands._halt,
            ),
            Action(
                name=("kill",),
                help_short="cancel tool calls, or stop a persistent child",
                usage="ID|LABEL|all",
                run=Commands._kill,
            ),
            Action(
                name=("send",),
                help_short="message agents, or run a /COMMAND on them",
                help_long="AGENTS: a label, parent, children, ancestors, descendants,"
                " all, a*glob, {a,b}, or /regex/. /send parent from a root with no"
                " parent starts its watcher.",
                usage="AGENTS TEXT",
                run=Commands._send,
            ),
            Action(
                name=("focus",),
                help_short="show another agent; what you type goes to it",
                help_long="Left on an empty line picks one from the tree.",
                usage="LABEL",
                run=Commands._focus,
            ),
            Action(
                name=("defer",),
                help_short="send once the agent idles (Tab)",
                usage="TEXT",
                run=Commands._defer,
            ),
            Action(
                name=("tasks",),
                help_short="list running tool calls",
                run=Commands._jobs,
            ),
            Action(
                name=("login",),
                help_short="sign in to the model's provider",
                help_long=login.format_help(),
                usage="[--headless|--no-headless]",
                run=Commands._login,
                cli=True,
                flags=login.options,
            ),
            Action(
                name=("login", "precedence"),
                help_short="the order a provider's logins take the model, as quotas run out",
                help_long="ACCOUNTs, comma separated; default is the default login."
                " The first with quota left signs in, at each turn's start and"
                " when a call spends a window; an account chosen with"
                " /model.provider.account comes first. off orders default, then"
                " each account as added. PROVIDER is the model's when left out.",
                usage="[PROVIDER] ACCOUNT,...|off",
                value=Commands._precedence_value,
                run=Commands._precedence,
            ),
        ]

    def _watching(self) -> list[Setting]:
        """Return ``watcher``, for a watcher: whether it is sent what it watches."""
        team, own = self.team, self.agent.runtime.tool_context.label
        if team is None or team.watched(own) is None:
            return []

        def value() -> str:
            return "on" if team.watching.get(own, False) else "off"

        async def apply(args: str) -> str:
            if args not in ("on", "off"):
                raise SettingError(f"{args!r} is not one of: on, off")
            team.watching[own] = args == "on"
            return value()

        return [
            Setting(
                name=("watcher",),
                help_short=f"whether {own} is sent what {team.watched(own)} does",
                usage="on|off",
                value=value,
                options=lambda: ("on", "off"),
                apply=apply,
            ),
        ]

    async def __call__(self, line: str) -> bool:
        """Run ``line``.

        Args:
          line: What was typed.

        Returns:
          quit: Whether the session should end.

        """
        text = line.strip()
        if not text:
            return False
        if not text.startswith("/"):
            self.agent.runtime.push(UserMessage(content=text))
            return False
        typed, _, args = text[1:].partition(" ")
        head, dot, rest = typed.partition(".")
        if (
            dot
            and rest
            and self.team is not None
            and (head in _RELATIONS or head in self.team)
        ):
            send = next(s for s in self.settings() if s.name == ("send",))
            await self._run(send, f"{head} /{rest} {args.strip()}".strip())
            return self.done
        setting = match(harvest(self.owners()), typed)
        if setting is None:
            self._say("error", f"unknown command: /{typed}; see /help")
            return self.done
        await self._run(setting, args.strip())
        return self.done

    async def aclose(self) -> None:
        """Cancel the commands still running, and those run on the team; wait for them."""
        for child in self._children.values():
            await child.aclose()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _run(self, setting: Setting, args: str) -> None:
        """Show ``setting``, or apply ``args`` to it; say what came of it."""
        key = ".".join(setting.name)
        shown = f"/{key}"
        if not args and setting.value is not None:
            self._say("note", _valued(setting.value, setting.options), command=shown)
            return
        rolls = setting.value is not None and setting.name not in _OWN
        given = self._given(key) if rolls and args == "default" else None
        chosen = args if given is None else given
        if not await apply_setting(setting, chosen, say=self.say):
            return
        if rolls and self.team is not None and self._label in self.team:
            self.team.choose(self._label, key, None if args == "default" else args)
            await self._roll(self.team, key, chosen)

    @property
    def _label(self) -> str:
        return self.agent.runtime.tool_context.label

    def _given(self, key: str) -> str | None:
        """Return what ``key`` is above this agent, if an ancestor chose it."""
        if self.team is None or self._label not in self.team:
            return None
        return self.team.inherited(self._label).get(key)

    async def _roll(self, team: Team, key: str, args: str) -> None:
        """Set ``key`` to ``args`` on each child that has not chosen it, and below."""
        for child in team.children(self._label):
            if key in team.choices.get(child, {}):
                continue
            await (await self.child_commands(team, child)).take(key, args)

    async def take(self, key: str, args: str) -> None:
        """Take ``args`` for ``key`` from an ancestor: apply it, then pass it on.

        Args:
          key: The setting's dotted name.
          args: What was typed above.

        """
        setting = next(
            (s for s in harvest(self.owners()) if ".".join(s.name) == key),
            None,
        )
        if setting is None:
            return
        take = setting.inherit if setting.inherit is not None else setting.apply
        try:
            _ = await take(args)
        except ValueError as e:
            self._say("error", error_text(e), command=f"{self._label} /{key}")
            return
        if self.team is not None:
            await self._roll(self.team, key, args)

    async def inherit(self) -> None:
        """Take each setting an ancestor chose and this agent has not."""
        team = self.team
        if team is None or self._label not in team:
            return
        mine = team.choices.get(self._label, {})
        for key, args in team.inherited(self._label).items():
            if key not in mine:
                await self.take(key, args)

    def _say(
        self,
        tone: Literal["line", "note", "error"],
        text: str,
        *,
        command: str = "",
    ) -> None:
        self.say(CommandOutput(text=text, tone=tone, command=command))

    def _spawn(self, name: str, work: Coroutine[object, object, None]) -> None:
        """Run ``work`` as a task; a failure prints under ``name``."""
        task = asyncio.create_task(work)
        self._tasks.add(task)

        def done(t: asyncio.Task[None]) -> None:
            self._tasks.discard(t)
            if not t.cancelled() and (e := t.exception()) is not None:
                self._say("error", error_text(e), command=name)

        task.add_done_callback(done)

    async def _quit(self, args: str) -> str:
        del args
        self.done = True
        return ""

    async def _help(self, args: str) -> str:
        return help_text(harvest(self.owners()), args)

    async def _clear(self, args: str) -> str:
        del args
        self._spawn("/clear", self.agent.session.clear())
        return "history cleared"

    async def _compact(self, args: str) -> str:
        self._spawn("/compact", self.agent.session.compactor.run(args))
        return f"queued{f' ({args})' if args else ''}"

    async def _halt(self, args: str) -> str:
        if not args:
            self.agent.runtime.halt()
            return ""
        targets = self._targets(args)
        for _, target in targets:
            target.runtime.halt()
        return f"halted {', '.join(label for label, _ in targets)}"

    async def _send(self, args: str) -> str:
        pattern, _, text = args.partition(" ")
        text = text.strip()
        if not text:
            raise SettingError("needs AGENTS TEXT")
        team = self.team
        if team is None:
            raise SettingError("there are no other agents")
        sent: list[str] = []
        said: list[str] = []
        if pattern == "parent":
            said.extend(await self._watcher())
        for label, _ in self._targets(pattern):
            if not text.startswith("/"):
                team.send("user", label, text)
                sent.append(label)
            elif await (await self.child_commands(team, label))(text):
                said.append(self._stop(team, label))
        return "\n".join(
            [*said, *([f"user→{', '.join(sent)}: {text}"] if sent else [])],
        )

    async def _watcher(self) -> list[str]:
        """Start this root's watcher, if it has no parent yet; say so."""
        team, own = self.team, self.agent.runtime.tool_context.label
        if team is None or team.above(own) is not None:
            return []
        label = await AgentSpawn(team=team).start_watcher(own)
        return [f"started {label}, watching {own}"]

    async def child_commands(self, team: Team, label: str) -> Commands:
        """Return the commands of ``team``'s agent ``label``, made once, inherited.

        Args:
          team: Where ``label`` is.
          label: The agent.

        Returns:
          commands: Its commands, which took up its ancestors' choices once made.

        """
        commands = self._children.get(label)
        if commands is None:
            commands = self._children[label] = Commands(
                team[label],
                catalog=self.catalog,
                say=self.say,
                screen=self.screens(label) if self.screens is not None else self.screen,
                panes=Panes(),
                team=team,
                focus=self.focus,
                screens=self.screens,
            )
            await commands.inherit()
        return commands

    async def _focus(self, args: str) -> str:
        if self.focus is None:
            raise SettingError("there is no screen to show another agent on")
        label = args.strip()
        team: Mapping[str, Agent] = self.team if self.team is not None else {}
        if label not in team:
            known = ", ".join(sorted(team)) or "none"
            raise SettingError(f"no agent {label!r}; agents: {known}")
        self.focus(label)
        return ""

    # Never this agent: a bare ``/halt`` halts it, and a message to itself would
    # wait on its own turn. Patterns are a label, ``all``, a glob, ``{a,b}``, or
    # ``/regex/``; a regex of two characters or fewer matches everything.
    def _targets(self, pattern: str) -> list[tuple[str, Agent]]:
        """Return the team's agents ``pattern`` names; raise if there are none."""
        team: Mapping[str, Agent] = self.team if self.team is not None else {}
        own = self.agent.runtime.tool_context.label
        labels = [label for label in team if label != own]
        if pattern == "all":
            matched = labels
        elif pattern in _RELATIONS and self.team is not None:
            matched = self.team.resolve(pattern, by=own)
        elif pattern.startswith("{") and pattern.endswith("}"):
            wanted = {part.strip() for part in pattern[1:-1].split(",")}
            matched = [label for label in labels if label in wanted]
        elif len(pattern) >= 3 and pattern.startswith("/") and pattern.endswith("/"):
            try:
                regex = re.compile(pattern[1:-1])
            except re.error as e:
                raise SettingError(f"invalid target regex: {e}") from e
            matched = [label for label in labels if regex.search(label)]
        else:
            # A plain label matches only itself; one holding *, ?, or [ reads as a glob.
            matched = [label for label in labels if fnmatch.fnmatchcase(label, pattern)]
        if not matched:
            known = ", ".join(sorted(labels)) or "none"
            raise SettingError(f"no agent {pattern!r}; agents: {known}")
        return [(label, team[label]) for label in matched]

    async def _kill(self, args: str) -> str:
        jobs = self.agent.runtime.jobs
        if not args:
            raise SettingError("needs an ID, a LABEL, or all")
        if args == "all":
            for job in jobs.values():
                job.cancel()
            self.panes.clear()
            return "cancelled all tool calls"
        if args in jobs:
            jobs[args].cancel()
            return f"cancelled {args}"
        if self.team is not None and args in self.team.persistent():
            return self._stop(self.team, args)
        raise SettingError(f"no running call {args!r}; see /tasks")

    def _stop(self, team: Team, label: str) -> str:
        """Stop the persistent child ``label`` of ``team``, as a task; say so."""
        self._spawn(f"/kill {label}", team.stop(label))
        return f"stopping {label}"

    async def _login(self, args: str) -> str:
        parser = _login_parser()
        parsed = parser.read(args)
        if parsed is None:
            return parser.format_help()
        provider = self.agent.model.provider
        if not isinstance(provider, HasLogin):
            raise SettingError(f"{provider.name} has no interactive sign-in")
        given = cast("bool | None", parsed.headless)
        headless = headless_wanted(os.environ, given=given)
        # A pasted code from another sign-in, or no answer at the prompt at all.
        try:
            await provider.login(headless=headless)
        except SIGN_IN_ERRORS as e:
            raise SettingError(error_text(e)) from e
        self.panes.send_deferred(self.agent.runtime)
        # The model is the same, so nothing else sends the turn a failure stopped.
        self.agent.runtime.retry()
        account = self.catalog.account
        return f"{provider.name} signed in{'' if account is None else f' as {account}'}"

    def _precedence_value(self) -> str:
        provider = self.agent.model.provider
        if not isinstance(provider, HasLogin):
            return ""
        return ",".join(provider.logins.order())

    # By shape alone, never by asking the catalog: an account a provider's name
    # abbreviates (``gs``) is still an account.
    async def _precedence(self, args: str) -> str:
        usage = SettingError("give [PROVIDER] ACCOUNT,... or [PROVIDER] off")
        first, _, rest = args.strip().partition(" ")
        named = bool(rest.strip()) and "," not in first and rest.lstrip()[:1] != ","
        listed = (rest if named else args).strip()
        accounts = tuple(a for a in (w.strip() for w in listed.split(",")) if a)
        if (
            not accounts
            or any(" " in a for a in accounts)
            or ("off" in accounts and accounts != ("off",))
        ):
            raise usage
        # ``work default``, its comma forgotten, names no provider: say how to write it.
        try:
            provider = (
                self.catalog[self.catalog.name(first)]
                if named
                else self.agent.model.provider
            )
        except ValueError as e:
            raise SettingError(f"{e}; {usage}") from e
        if not isinstance(provider, HasLogin):
            raise SettingError(f"{provider.name} keeps no logins to order")
        await provider.logins.prefer(None if accounts == ("off",) else accounts)
        order = provider.logins.order()
        saved = ("default", *provider.logins.accounts())
        unsaved = [a for a in accounts if a not in saved and a != "off"]
        missing = f" (no login saved: {','.join(unsaved)})" if unsaved else ""
        return f"{provider.name}: {','.join(order)}{missing}"

    async def _jobs(self, args: str) -> str:
        del args
        jobs = self.agent.runtime.jobs
        now = time.time()
        children = self.team.persistent() if self.team is not None else []
        return "\n".join(
            [
                f"{len(jobs)} running tool call(s), {len(children)} persistent agent(s)",
                *(
                    f"  {call_id:<28s}  {job.call.name:<16s}  {job.state:<9s}"
                    f"  {now - job.started:.0f}s"
                    for call_id, job in jobs.items()
                ),
                *(f"  {label:<28s}  persistent agent" for label in children),
            ],
        )

    async def _defer(self, args: str) -> str:
        if not args:
            raise SettingError("needs TEXT")
        self.agent.runtime.push(types.UserDeferredMessage(text=args))
        return ""


@dataclass(frozen=True, slots=True, kw_only=True)
class Action:
    """One of ``Commands``' own actions, before it is bound to an agent's commands."""

    name: tuple[str, ...]
    help_short: str
    help_long: str = ""
    usage: str = ""
    run: Callable[[Commands, str], Coroutine[object, object, str]]
    value: Callable[[Commands], str] | None = None
    cli: bool = False
    flags: frozenset[str] = frozenset()


def owners_without_agent(
    *,
    cwd: Path,
    say: Callable[[CommandOutput], None],
) -> list[tuple[tuple[str, ...], HasSettings]]:
    """Return the owners that need no agent, so run from the shell with no model.

    Args:
      cwd: The working directory, which names the project.
      say: Where they say what they do along the way, such as a sign-in's page.

    Returns:
      owners: Each owner, after where it is mounted.

    """
    return [(("mcp",), McpSettings(cwd=cwd, show=partial(_say_note, say)))]


async def apply_setting(
    setting: Setting,
    args: str,
    *,
    say: Callable[[CommandOutput], None],
) -> bool:
    """Apply ``args`` to ``setting``; say what it did, or why it could not.

    Args:
      setting: What was named.
      args: What followed its name.
      say: Where it is said, each saying ``/NAME`` as its command.

    Returns:
      done: Whether it applied.

    """
    shown = f"/{'.'.join(setting.name)}"
    try:
        said = await setting.apply(args)
    except ValueError as e:
        say(CommandOutput(text=error_text(e), tone="error", command=shown))
        return False
    if said:
        say(CommandOutput(text=said.rstrip(), tone="note", command=shown))
    return True


def help_text(
    settings: Sequence[Setting],
    args: str,
    *,
    spell: Callable[[tuple[str, ...]], str] = lambda name: f"/{'.'.join(name)}",
) -> str:
    """Return what ``/help ARGS`` says: every setting, a group's, or all of one.

    Args:
      settings: Every setting, in order.
      args: Nothing, a group's name such as ``mcp``, or a setting's name.
      spell: How a setting's name is typed where the help shows: ``/mcp.add``
        at the prompt, ``sagent mcp add`` at the shell.

    Returns:
      text: The list, or the one setting's usage, help, and value.

    Raises:
      SettingError: ``args`` names nothing.

    """
    if not args:
        return "\n".join(["sagent commands", *(_listed(s, spell) for s in settings)])
    parts = tuple(args.split("."))
    under = [s for s in settings if s.name[: len(parts)] == parts and s.name != parts]
    if under and not any(s.name == parts for s in settings):
        return "\n".join([f"{args} commands", *(_listed(s, spell) for s in under)])
    setting = match(settings, args)
    if setting is None:
        raise SettingError(f"no setting {args!r}")
    value = setting.value
    return "\n".join(
        [
            f"{spell(setting.name)} {setting.usage}".rstrip(),
            setting.help_short,
            *([setting.help_long] if setting.help_long else []),
            *([] if value is None else [f"now: {_valued(value, setting.options)}"]),
        ],
    )


def _say_note(say: Callable[[CommandOutput], None], text: str) -> None:
    say(CommandOutput(text=text, tone="note"))


def _login_parser() -> ActionParser:
    """Return how ``login`` reads its arguments, at the prompt or the shell."""
    parser = ActionParser(
        ("login",),
        description="Sign in to the model's provider, as --provider and --account say.",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        help="print the URL and take the code pasted back; default: over ssh, or"
        " on Linux with no display",
    )
    return parser


def _valued(
    value: Callable[[], str],
    options: Callable[[], Sequence[str]],
) -> str:
    """Return a setting's value, then what it may take: ``high   (none, high)``."""
    offered = ", ".join(options())
    return value() + (f"   ({offered})" if offered else "")


def _listed(setting: Setting, spell: Callable[[tuple[str, ...]], str]) -> str:
    """Return ``setting``'s row in ``/help``: its name and usage, then what it is."""
    head = f"{spell(setting.name)} {setting.usage}".rstrip()
    return (
        f"  {head:<32s}{setting.help_short}"
        if len(head) < 32
        else f"  {head}\n  {'':<32s}{setting.help_short}"
    )
