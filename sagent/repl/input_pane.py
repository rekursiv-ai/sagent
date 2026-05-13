"""REPL input zone: bar rendering + pump + ``InputSource`` abstraction.

This module owns everything about the *input* zone of the REPL --
the bar where the user types, the optional dim ``queued_input``
preview rendered just above it, and the pump that consumes
submitted lines.

Pump
~~~~

A long-running coroutine spawned as a *hidden* background task on
the Agent. It loops on an :class:`InputSource`, parses each line
via :func:`repl.slash.parse_slash`, and dispatches the resulting
:class:`SlashAction` directly against the agent's public API. The
pump is intentionally NOT a Handler / Tool / runtime event; it
lives in ``agent._bg`` as a hidden entry so user-initiated abort
(``/halt`` / ``/kill``) can never tear down the input loop.

Input-pane rendering
~~~~~~~~~~~~~~~~~~~~

:func:`render_input_pane` builds the prompt-toolkit ``FormattedText``
for the input zone: an optional dim ``queued_input_pane`` preview
line above the prompt sigil, then the ``> `` sigil itself. Backed by
a REPL-local ``queued_input: list[str]`` buffer of texts the user
typed while the agent was busy; the tail is shown as the preview,
Up-arrow lifts it back, and the runtime's
``make_queued_input_clearer`` observer empties the buffer once the
runtime has committed user input to history.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, override

import asyncio
import logging
import time

from prompt_toolkit.formatted_text import FormattedText
from rich.text import Text

from sagent.agent.background import BackgroundTaskEntry
from sagent.agent.runtime import (
    Clear,
    Compact,
    Recompact,
    UserMessage,
)
from sagent.lib.lazy_import import lazy_import
from sagent.repl.slash import (
    QUIT_WORDS,
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
    from prompt_toolkit import PromptSession
    from rich.console import Console

    from sagent.agent.agent import Agent
    from sagent.repl.render import Printer

logger = logging.getLogger(__name__)

__all__ = [
    "REPL_PUMP_KEY",
    "InputSource",
    "PromptToolkitInputSource",
    "StubInputSource",
    "render_input_pane",
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
        # Empty target halts self; non-empty target looks up the live
        # registry label (which may have a ``_N`` suffix when default
        # names collide). Comparing against ``agent.name`` here would
        # halt the current agent on ``/halt Agent`` even if the live
        # registry key is ``Agent_2`` -- a target/label mismatch.
        if not action.target:
            agent.halt()
            return False
        other = agent_registry.get(action.target)
        if other is None:
            if printer is not None:
                printer.write_tool_error(f"[/halt] unknown agent: {action.target}")
            return False
        other.halt()
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


class PromptToolkitInputSource(InputSource):
    """Async input source backed by a :class:`prompt_toolkit.PromptSession`.

    The ``queued_input`` field mirrors the REPL-local buffer of texts
    the user typed while the agent was busy. The runtime's
    ``GatedDeque`` doesn't support tag-based peek / pop, so the REPL
    maintains this list itself; the dim preview rendered by
    :func:`render_input_pane` shows the tail and ``Up`` lifts it back.
    """

    queued_input: list[str]
    """List of texts the user typed while the agent was busy. New input
    appends; ``Up`` pops the latest; :func:`render_input_pane` previews
    the tail."""

    def __init__(
        self,
        session: PromptSession[str],
        *,
        queued_input: list[str],
        console: Console | None = None,
    ) -> None:
        self._session = session
        self.queued_input = queued_input
        self._console = console

    @override
    async def next_line(self) -> str | None:
        """Return the next line, or ``None`` to terminate the input loop."""
        try:
            text = await self._session.prompt_async()
        except (EOFError, KeyboardInterrupt):
            self._surface_queued_input_on_quit()
            return None
        stripped = text.strip()
        if stripped.lower() in QUIT_WORDS:
            self._surface_queued_input_on_quit()
            return None
        return text

    def _surface_queued_input_on_quit(self) -> None:
        """Surface the tail of ``queued_input`` before the loop ends."""
        if not self.queued_input or self._console is None:
            return
        tail = self.queued_input[-1]
        preview = tail.replace("\n", " ")[:80]
        self._console.print(
            Text(f"[discarding queued message: {preview}]", style="dim yellow"),
        )
        self.queued_input.clear()


def render_input_pane(agent: Agent, queued_input: list[str]) -> FormattedText:
    r"""Build the input-pane ``FormattedText``: full queue + prompt sigil.

    Renders the entire staged queue (blocks joined by ``\\n\\n``) above
    the ``> `` prompt sigil whenever ``queued_input`` has entries.
    The queue is a staging draft: blocks accumulate as the user
    submits, are visible until committed, and can be lifted back into
    ``input_pane`` for editing via Up.

    Args:
      agent: Agent (currently unused; reserved for callers that want
          to gate rendering on additional state).
      queued_input: REPL-local staging buffer; each entry is one block.

    Returns:
      formatted: The input pane's formatted text.

    """
    del agent
    parts: list[tuple[str, str]] = []
    if queued_input:
        parts.append(("class:queued_input_pane", "\n\n".join(queued_input)))
        parts.append(("", "\n"))
    parts.append(("class:input_pane", "> "))
    return FormattedText(parts)
