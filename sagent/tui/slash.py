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

from typing import TYPE_CHECKING, Literal

import asyncio
import fnmatch
import logging
import os
import re
import sys
import time

from sagent.model.exceptions import AuthError
from sagent.providers.types import Login
from sagent.runtime import types
from sagent.session.types import UserMessage
from sagent.tools.agent_spawn import AgentSpawn
from sagent.tui.prompt import Panes
from sagent.tui.render import error_text
from sagent.tui.settings.harvest import harvest
from sagent.tui.settings.match import match
from sagent.tui.settings.model import AgentModel
from sagent.tui.settings.runtime import RuntimeSettings
from sagent.tui.settings.screen import ScreenSettings
from sagent.tui.settings.tool import ToolDisplaySettings
from sagent.tui.settings.types import Setting, SettingError
from sagent.tui.types import CommandOutput


if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Mapping, Sequence

    from sagent.agent.team import Team
    from sagent.agent.types import Agent
    from sagent.providers.types import ModelCatalog
    from sagent.tui.settings.types import HasSettings
    from sagent.tui.types import ThinkingView


__all__ = [
    "Commands",
    "headless_wanted",
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
          owners: These actions, the model, each tool, the runtime, then the screen.

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
        ]

    def settings(self) -> list[Setting]:
        """Return the actions, first, so a short name reaches them before a setting.

        Returns:
          settings: Each action, ``quit`` to ``login``.

        """
        return [
            Setting(name=("quit",), help_short="end the session", apply=self._quit),
            Setting(name=("exit",), help_short="end the session", apply=self._quit),
            Setting(
                name=("help",),
                help_short="this list, or all one setting says",
                usage="[NAME]",
                apply=self._help,
            ),
            Setting(
                name=("clear",),
                help_short="wipe context (the journal keeps it)",
                apply=self._clear,
            ),
            Setting(
                name=("compact",),
                help_short="compact history",
                usage="[hints]",
                apply=self._compact,
            ),
            Setting(
                name=("recompact",),
                help_short="compact history",
                usage="[hints]",
                apply=self._compact,
            ),
            Setting(
                name=("halt",),
                help_short="stop the model call (Ctrl+C), or the agents'",
                usage="[AGENTS]",
                apply=self._halt,
            ),
            Setting(
                name=("kill",),
                help_short="cancel tool calls, or stop a persistent child",
                usage="ID|LABEL|all",
                apply=self._kill,
            ),
            Setting(
                name=("send",),
                help_short="message agents, or run a /COMMAND on them",
                help_long="AGENTS: a label, parent, children, ancestors, descendants,"
                " all, a*glob, {a,b}, or /regex/. /send parent from a root with no"
                " parent starts its watcher.",
                usage="AGENTS TEXT",
                apply=self._send,
            ),
            Setting(
                name=("focus",),
                help_short="show another agent; what you type goes to it",
                help_long="Left on an empty line picks one from the tree.",
                usage="LABEL",
                apply=self._focus,
            ),
            Setting(
                name=("defer",),
                help_short="send once the agent idles (Tab)",
                usage="TEXT",
                apply=self._defer,
            ),
            Setting(
                name=("tasks",),
                help_short="list running tool calls",
                apply=self._jobs,
            ),
            Setting(
                name=("login",),
                help_short="sign in to the model's provider",
                usage="[headless=auto|true|false]",
                apply=self._login,
            ),
            *self._watching(),
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
            self._say("note", f"[{shown}] {_valued(setting.value, setting.options)}")
            return
        rolls = setting.value is not None and setting.name not in _OWN
        given = self._given(key) if rolls and args == "default" else None
        try:
            said = await setting.apply(args if given is None else given)
        except ValueError as e:
            self._say("error", f"[{shown}] {error_text(e)}")
            return
        if rolls and self.team is not None and self._label in self.team:
            self.team.choose(self._label, key, None if args == "default" else args)
            await self._roll(self.team, key, args if given is None else given)
        if said:
            self._say("note", f"[{shown}] {said}")

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
            await (await self.commands_for(team, child)).take(key, args)

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
            self._say("error", f"[{self._label} /{key}] {error_text(e)}")
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

    def _say(self, tone: Literal["line", "note", "error"], text: str) -> None:
        self.say(CommandOutput(text=text, tone=tone))

    def _spawn(self, name: str, work: Coroutine[object, object, None]) -> None:
        """Run ``work`` as a task; a failure prints under ``name``."""
        task = asyncio.create_task(work)
        self._tasks.add(task)

        def done(t: asyncio.Task[None]) -> None:
            self._tasks.discard(t)
            if not t.cancelled() and (e := t.exception()) is not None:
                self._say("error", f"[{name}] {error_text(e)}")

        task.add_done_callback(done)

    async def _quit(self, args: str) -> str:
        del args
        self.done = True
        return ""

    async def _help(self, args: str) -> str:
        settings = harvest(self.owners())
        if not args:
            return "\n".join(["sagent commands", *map(_listed, settings)])
        setting = match(settings, args)
        if setting is None:
            raise SettingError(f"no setting {args!r}")
        value = setting.value
        return "\n".join(
            [
                f"/{'.'.join(setting.name)} {setting.usage}".rstrip(),
                setting.help_short,
                *([setting.help_long] if setting.help_long else []),
                *([] if value is None else [f"now: {_valued(value, setting.options)}"]),
            ],
        )

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
            elif await (await self.commands_for(team, label))(text):
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

    async def commands_for(self, team: Team, label: str) -> Commands:
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
        options: dict[str, str] = {}
        for word in args.split():
            key, _, value = word.partition("=")
            options[key] = value
        unknown = set(options) - {"headless"}
        if unknown:
            raise SettingError(f"unknown option: {', '.join(sorted(unknown))}")
        flag = options.get("headless", "auto").lower()
        if flag not in ("auto", "true", "false"):
            raise SettingError(f"headless={flag!r}: give auto, true, or false")
        provider = self.agent.model.provider
        if not isinstance(provider, Login):
            raise SettingError(f"{provider.name} has no interactive sign-in")
        headless = headless_wanted(os.environ) if flag == "auto" else flag == "true"
        # A pasted code from another sign-in, or no answer at the prompt at all.
        try:
            await provider.login(headless=headless)
        except (AuthError, EOFError, OSError, RuntimeError, ValueError) as e:
            raise SettingError(error_text(e)) from e
        self.panes.send_deferred(self.agent.runtime)
        # The model is the same, so nothing else sends the turn a failure stopped.
        self.agent.runtime.retry()
        account = self.catalog.account
        return f"{provider.name} signed in{'' if account is None else f' as {account}'}"

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


def headless_wanted(
    environ: Mapping[str, str],
    *,
    platform: str = sys.platform,
) -> bool:
    """Return whether a sign-in should print its URL rather than open a browser.

    Over ssh, a browser opens on the far machine's screen, not the user's; on
    Linux with no display, there is none to open. macOS always has a screen.

    Args:
      environ: The environment, such as ``os.environ``.
      platform: ``sys.platform``.

    Returns:
      headless: Whether no browser the user can see would open.

    """
    if any(
        environ.get(name, "") != ""
        for name in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY")
    ):
        return True
    return platform.startswith("linux") and all(
        environ.get(name, "") == "" for name in ("DISPLAY", "WAYLAND_DISPLAY")
    )


def _valued(
    value: Callable[[], str],
    options: Callable[[], Sequence[str]],
) -> str:
    """Return a setting's value, then what it may take: ``high   (none, high)``."""
    offered = ", ".join(options())
    return value() + (f"   ({offered})" if offered else "")


def _listed(setting: Setting) -> str:
    """Return ``setting``'s row in ``/help``: its name and usage, then what it is."""
    head = f"/{'.'.join(setting.name)} {setting.usage}".rstrip()
    return (
        f"  {head:<32s}{setting.help_short}"
        if len(head) < 32
        else f"  {head}\n  {'':<32s}{setting.help_short}"
    )
