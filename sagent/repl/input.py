"""REPL input pump + ``InputSource`` abstraction.

The pump is a long-running coroutine spawned as a *hidden* background
task on the Agent. It loops on an :class:`InputSource`, parses each
line via :func:`repl.slash.parse_slash`, and dispatches the resulting
:class:`SlashAction` directly against the agent's public API.

The pump is intentionally NOT a Handler / Tool / runtime event. It
lives in ``agent._bg`` as a hidden entry so user-initiated abort
(``/halt`` / ``/kill``) can never tear down the input loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import asyncio
import logging
import time

from sagent.agent.background import BackgroundTaskEntry
from sagent.agent.runtime import (
    Clear,
    Compact,
    Recompact,
    UserMessage,
)
from sagent.lib.lazy_import import lazy_import
from sagent.repl.slash import (
    Clear as SlashClear,
    Compact as SlashCompact,
    Halt as SlashHalt,
    Help as SlashHelp,
    Kill as SlashKill,
    Login as SlashLogin,
    ModelSwitch as SlashModelSwitch,
    Quit as SlashQuit,
    Recompact as SlashRecompact,
    SlashAction,
    Tasks as SlashTasks,
    Text as SlashText,
    parse_slash,
)
from sagent.tools.core import agent_registry


# Cycle break: ``run_repl`` imports ``spawn_repl_pump`` from this module.
# Lazy module proxy so the dispatch helpers (do_switch_model / do_login /
# format_tasks) are reachable without re-introducing a top-level cycle.
_run_repl = lazy_import("sagent.repl.run_repl")
_render = lazy_import("sagent.repl.render")

if TYPE_CHECKING:
    from sagent.agent.agent import Agent
    from sagent.repl.render import Printer

logger = logging.getLogger(__name__)

__all__ = [
    "REPL_PUMP_KEY",
    "InputSource",
    "StubInputSource",
    "spawn_repl_pump",
]

# Stable key for the REPL pump entry in ``agent._bg``.
REPL_PUMP_KEY = "__repl_pump__"


class InputSource(Protocol):
    """Source of user input lines."""

    async def next_line(self) -> str | None:
        """Return the next line, or ``None`` to terminate the input loop."""
        ...


class StubInputSource:
    """In-process queue of pre-staged lines for tests."""

    def __init__(self, lines: list[str | None]) -> None:
        self._lines: list[str | None] = list(lines)

    async def next_line(self) -> str | None:
        """Return the next staged line, or ``None`` when the queue empties."""
        if not self._lines:
            return None
        return self._lines.pop(0)


def spawn_repl_pump(
    agent: Agent,
    source: InputSource,
    *,
    printer: Printer | None = None,
) -> asyncio.Task[None]:
    """Spawn the REPL input pump as a hidden background task.

    Args:
      agent: Agent to drive.
      source: Where lines come from (prompt-toolkit in production).
      printer: Optional sink for status echoes (``/help``, ``/tasks``,
          ``/login``, ``/model``).

    Returns:
      task: The running pump task.

    """
    task = asyncio.create_task(_input_pump(agent, source, printer))
    agent.register_background(
        REPL_PUMP_KEY,
        BackgroundTaskEntry(
            task=task,
            tool_name="repl-input",
            queue_id=REPL_PUMP_KEY,
            started=time.time(),
            hidden=True,
            kind="tool",
        ),
    )
    return task


async def _input_pump(
    agent: Agent,
    source: InputSource,
    printer: Printer | None,
) -> None:
    """Read lines from ``source`` and dispatch the parsed action."""
    while True:
        try:
            line = await source.next_line()
            if line is None:
                agent.shutdown(force=False)
                return
            action = parse_slash(line)
            if action is None:
                continue
            should_exit = await _dispatch(agent, action, printer)
            if should_exit:
                return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("REPL input pump raised; surfacing as error")
            if printer is not None:
                printer.write_tool_error(f"[input pump] {type(exc).__name__}: {exc}")


async def _dispatch(
    agent: Agent,
    action: SlashAction,
    printer: Printer | None,
) -> bool:
    """Dispatch one parsed slash action; return True to exit the pump."""
    if isinstance(action, SlashQuit):
        agent.shutdown(force=False)
        return True
    if isinstance(action, SlashHalt):
        if action.target in ("", agent.name):
            agent.halt()
        else:
            other = agent_registry.get(action.target)
            if isinstance(other, type(agent)):
                other.halt()
            elif printer is not None:
                printer.write_tool_error(f"[/halt] unknown agent: {action.target}")
        return False
    if isinstance(action, SlashKill):
        if action.target == "all":
            agent.kill_all_tools()
            if printer is not None:
                printer.write_line("[/kill] cancelled all tool tasks")
        else:
            agent.kill_tool(action.target)
            if printer is not None:
                printer.write_line(f"[/kill] cancelled {action.target}")
        return False
    if isinstance(action, SlashClear):
        agent.runtime.inbox.push_back(Clear())
        if printer is not None:
            printer.write_line("[/clear] history cleared")
        return False
    if isinstance(action, SlashCompact):
        agent.runtime.inbox.push_back(Compact(args=action.args))
        if printer is not None:
            note = f" ({action.args})" if action.args else ""
            printer.write_line(f"[/compact] queued{note}")
        return False
    if isinstance(action, SlashRecompact):
        agent.runtime.inbox.push_back(Recompact(args=action.args))
        if printer is not None:
            note = f" ({action.args})" if action.args else ""
            printer.write_line(f"[/recompact] queued{note}")
        return False
    if isinstance(action, SlashModelSwitch):
        _run_repl.do_switch_model(agent, action.args, printer)
        return False
    if isinstance(action, SlashLogin):
        _run_repl.do_login(agent, printer)
        return False
    if isinstance(action, SlashHelp):
        if printer is not None:
            printer.write_line(_render.HELP_TEXT)
        return False
    if isinstance(action, SlashTasks):
        if printer is not None:
            printer.write_line(_run_repl.format_tasks(agent))
        return False
    if isinstance(action, SlashText):
        agent.runtime.inbox.push_back(UserMessage(text=action.content))
        return False
    # Remaining variant: Unknown -- surface the parse error.
    if printer is not None:
        printer.write_tool_error(action.text)
    return False
