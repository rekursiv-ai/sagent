r"""REPL input zone: bar rendering + pump + ``InputSource`` abstraction.

This module owns everything about the *input* zone of the REPL --
the bar where the user types, the dim ``queued_input_pane`` preview
rendered just above it when the staging queue is non-empty, and the
pump that consumes slash submissions.

Behavior contract: Up / Down navigation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Up / Down arrows traverse a virtual stack of (queue, history) without
losing user state. The first Up captures a snapshot of
``(queued_input, buffer)``; the final Down restores it.

The slot rows render top-to-bottom as ``history[-N]`` (oldest) through
``queue`` then ``input``. ``[DONE]`` means the slot has no content to
display at that step.

Case 1 -- queue is non-empty::

                  t=0    t=1     t=2     t=3    t=4
                         UP      UP      DN     DN
    history[-3]    d      c       b       c      d
    history[-2]    e      d       c       d      e
    history[-1]  [DNE]    e       d       e    [DNE]
    queue          f    [DNE]   [DNE]   [DNE]    f
    input          g      f       e       f      g

- First Up "edits" the queue: queue text joins on ``\n\n``, lands in
  buffer; queue slot empties. A snapshot of ``(queued_input, buffer)``
  is captured.
- Subsequent Ups walk older history into buffer; queue slot stays
  empty during navigation.
- Down reverses the walk; the final Down at the navigation boundary
  restores the snapshot (queue and buffer return to t=0 state).

Case 2 -- queue is empty::

                  t=0    t=1     t=2     t=3    t=4
                         UP      UP      DN     DN
    history[-3]    d      c       b       c      d
    history[-2]    e      d       c       d      e
    history[-1]    f      e       d       e      f
    input          g      f       e       f      g

- First Up walks ``history[-1]`` into buffer; snapshot captured.
- Subsequent Ups walk older history; queue slot stays empty (there was
  no queue to restore).
- Down reverses; final Down restores buffer to its pre-navigation value.

Enter
~~~~~

- No navigation active (``cursor == 0``): preempt-dispatch as today --
  push ``UserMessage`` straight to the runtime, cutting in line over any
  cohort/stream. ``queued_input`` is not touched. Snapshot discarded.
- Navigation active (``cursor > 0``): commit the buffer as a queued
  block. Case 1 appends to ``queued_input``; case 2 creates the queue
  from the buffer's content. The snapshot is discarded; ``cursor``
  returns to 0; the dim ``queued_input_pane`` redraws with the new
  state. The runtime sees the queued blocks at the next ``ModelIdle``
  via ``make_queued_input_committer``.
- Text ending in ``\``: backslash continuation -- replace trailing
  ``\`` with literal ``\n``, stay in buffer, do not dispatch.
- Empty buffer: nothing.
- Slash command: route through pump.

Tab
~~~

Tab stages the buffer in ``queued_input`` (REPL-local). No runtime
push. ``make_queued_input_committer`` in :mod:`repl.run_repl` commits
the joined queue as a single ``UserQueuedMessage`` on ``ModelIdle``.

``queued_input`` is purely a Tab-staging buffer plus the post-navigation
commit target. Up-arrow's lift is a true retract because nothing was
ever in the runtime to begin with.

Headless callers without a Tab key use the ``/defer <text>`` slash
command, which pushes ``UserQueuedMessage`` directly through the pump
-- not retractable, but a one-shot defer gesture is sufficient for
non-interactive contexts.

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
above the ``> `` prompt sigil whenever ``queued_input`` is
non-empty. The preview shows all staged blocks joined by ``\n\n``.
The runtime's ``make_queued_input_clearer`` observer empties
``queued_input`` once the runtime publishes the committed
``UserMessage`` event so the dim preview stops showing stale entries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, override

import asyncio
import logging
import time

from prompt_toolkit.formatted_text import FormattedText
from rich.text import Text

from sagent.agent.background import BackgroundTaskEntry
from sagent.lib.lazy_import import lazy_import
from sagent.repl.slash import (
    QUIT_WORDS,
    Clear as SlashClear,
    Compact as SlashCompact,
    Defer as SlashDefer,
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
from sagent.types.exceptions import (
    UserFacingError,
    log_exception_or_warning,
)
from sagent.types.history import UserMessage
from sagent.types.runtime import (
    Clear,
    Compact,
    Recompact,
    UserQueuedMessage,
)


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
        except Exception as exc:  # noqa: BLE001 -- pump catches any slash-handler exception; UserFacingError routed to warning, others to exception
            log_exception_or_warning(
                logger, "REPL input pump raised; surfacing as error", exc
            )
            if printer is not None:
                # Polished message for UserFacingError; ClassName prefix
                # for unexpected exceptions (helps the operator).
                detail = (
                    str(exc)
                    if isinstance(exc, UserFacingError)
                    else f"{type(exc).__name__}: {exc}"
                )
                printer.write_tool_error(f"[input pump] {detail}")


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
        await _run_repl.do_login(agent, printer)
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
    if isinstance(action, SlashDefer):
        agent.runtime.inbox.push_back(UserQueuedMessage(text=action.content))
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

    Renders all pending user content above the ``> `` prompt sigil
    (blocks joined by ``\n\n``). Two buffers contribute:

    - ``queued_input`` -- Tab-staged blocks, REPL-local. Up-arrow can
      lift these back into the input buffer for editing.
    - ``agent.runtime._mid_stream_queue`` -- Enter-mid-stream blocks,
      already committed to the runtime. Shown for visibility ("is my
      message queued?") but not retractable.

    Both are pending model consumption; rendering them together gives
    the user one canonical "what's queued" surface.

    Args:
      agent: Agent whose runtime ``_mid_stream_queue`` is consulted.
      queued_input: REPL-local Tab-staging buffer; each entry is one block.

    Returns:
      formatted: The input pane's formatted text.

    """
    blocks: list[str] = list(queued_input)
    blocks.extend(m.text for m in agent.runtime.pending_mid_stream())
    parts: list[tuple[str, str]] = []
    if blocks:
        parts.append(("class:queued_input_pane", "\n\n".join(blocks)))
        parts.append(("", "\n"))
    parts.append(("class:input_pane", "> "))
    return FormattedText(parts)
