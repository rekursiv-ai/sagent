"""Headless daemon adapter for Sagent's native user command surface.

The command semantics remain in :mod:`sagent.tui.slash`. This module supplies
the non-terminal callbacks that the existing command engine needs and projects
its native ``Setting`` metadata over the daemon RPC protocol. It must not
parse, validate, or reimplement command arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sagent.settings.harvest import harvest
from sagent.tui.prompt import Panes
from sagent.tui.slash import Commands
from sagent.tui.types import ThinkingView


if TYPE_CHECKING:
    from collections.abc import Callable

    from sagent.agent.agent import Agent
    from sagent.agent.team import Team
    from sagent.providers.types import ModelCatalog
    from sagent.tui.types import CommandOutput


__all__ = ["SagentCommandService"]

# Terminal lifecycle commands that do not apply to an external RPC daemon:
_TERMINAL_ONLY_COMMANDS = frozenset({
    "quit",  # terminal lifecycle; daemon lifecycle is controlled via shutdown RPC
    "exit",  # terminal lifecycle
})


@dataclass(slots=True)
class _DaemonThinkingView(ThinkingView):
    """Real per-agent screen state for Sagent's native ``screen`` settings.

    Headless clients do not have a terminal printer, but the setting still has a real
    meaning: the client decides whether to display structured thinking events.
    Keeping this state here lets the native ``ScreenSettings`` implementation
    remain authoritative without inventing a second settings model.
    """

    _show_thinking: bool = False

    @property
    def show_thinking(self) -> bool:
        """Whether the client should display thinking events."""
        return self._show_thinking

    @show_thinking.setter
    def show_thinking(self, show: bool) -> None:
        self._show_thinking = show


class SagentCommandService:
    """Adapt Sagent's existing ``Commands`` engine to the daemon transport."""

    def __init__(
        self,
        *,
        agent: Agent,
        team: Team,
        catalog: ModelCatalog,
        output: Callable[[str, CommandOutput], None],
        focus: Callable[[str], None],
        hidden_commands: frozenset[str] = _TERMINAL_ONLY_COMMANDS,
    ) -> None:
        self._team = team
        self._catalog = catalog
        self._output = output
        self._focus = focus
        self._hidden_commands = hidden_commands
        self._screens: dict[str, _DaemonThinkingView] = {}
        self._commands = Commands(
            agent,
            catalog=catalog,
            say=lambda result: output(agent.runtime.tool_context.label, result),
            screen=self._screen(agent.runtime.tool_context.label),
            panes=Panes(),
            team=team,
            focus=focus,
            screens=self._screen,
        )

    def _screen(self, label: str) -> _DaemonThinkingView:
        return self._screens.setdefault(label, _DaemonThinkingView())

    async def _commands_for(self, label: str) -> Commands:
        if label not in self._team:
            raise ValueError(f"unknown agent {label!r}")
        if label == self._commands.agent.runtime.tool_context.label:
            return self._commands
        commands = await self._commands.commands_for(self._team, label)
        # ``Commands.commands_for`` intentionally inherits the parent's output
        # callback for the TUI. Frontends need the selected agent on each
        # notification so command feedback stays attached to the right view.
        commands.say = lambda result, agent_label=label: self._output(agent_label, result)
        return commands

    async def descriptors(self, label: str) -> list[dict[str, object]]:
        """Return live metadata harvested from Sagent's native command owners."""
        commands = await self._commands_for(label)
        descriptors: list[dict[str, object]] = []
        for setting in harvest(commands.owners()):
            name = ".".join(setting.name)
            if name in self._hidden_commands:
                continue
            options = tuple(setting.options())
            descriptors.append(
                {
                    "name": name,
                    "kind": "action" if setting.value is None else "setting",
                    "scope": setting.name[0] if setting.name[0] in {
                        "model",
                        "runtime",
                        "screen",
                        "tool",
                        "mcp",
                    } else "session",
                    "helpShort": setting.help_short,
                    "helpLong": setting.help_long,
                    "usage": setting.usage,
                    "options": [option for option in options if isinstance(option, str)],
                    "readable": setting.value is not None,
                },
            )
        return descriptors

    async def execute_line(self, *, label: str, line: str) -> bool:
        """Run a complete slash command through Sagent's native parser."""
        if not line.strip().startswith("/"):
            raise ValueError("command line must start with '/'")
        commands = await self._commands_for(label)
        return await commands(line)

    async def execute(
        self,
        *,
        label: str,
        command_name: str,
        arguments_text: str,
    ) -> bool:
        """Run one command through Sagent's native parser and executor."""
        if not command_name or any(char.isspace() for char in command_name):
            raise ValueError("commandName must be a non-empty command name")
        line = f"/{command_name}"
        if arguments_text:
            line = f"{line} {arguments_text}"
        return await self.execute_line(label=label, line=line)
