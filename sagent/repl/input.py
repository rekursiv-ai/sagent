"""REPL input pump + ``InputSource`` abstraction.

The input pump is a long-running coroutine, spawned as a *hidden*
background task in ``agent.background_tasks``. It loops on an
``InputSource`` (real prompt-toolkit in production; ``StubInputSource``
in tests), classifies each line via :func:`repl.slash.parse_slash`, and
dispatches the resulting :class:`SlashAction` to ``agent.inbox``.

The pump is intentionally NOT a ``Handler``. Earlier versions made it a
``SpawnedHandler`` subscribed to ``text/x-bootstrap``, which lumped it
into ``agent.tasks`` alongside dispatch-step work; ``/abort`` then
indiscriminately cancelled it and deadlocked the terminal (no reader
on stdin, raw mode never restored). Moving it to ``background_tasks``
with ``hidden=True`` is the clean separation: dispatch-step tasks live
in ``agent.tasks`` (all abortable); long-running infra lives in
``background_tasks`` (cancelled only at agent shutdown).

Slash commands flow through the FIFO with typed text so user intent
order is preserved. Only urgent actions (``/clear``, ``/abort``) jump
the queue -- they exist to preempt in-flight model/tool work.

Command handlers (``/login``, ``/help``, ``/tasks``) live in
:mod:`repl.render` next to the other render-side handlers; they all
need a printer to surface output and are bundled by
:func:`repl.render.repl_handler_set`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import asyncio
import logging
import time

from sagent.custom_types import TextMessage
from sagent.lib.descriptors import QUIT_SENTINEL
from sagent.repl.slash import (
    QUIT_WORDS,
    SlashAction,
    dispatch,
    parse_slash,
)
from sagent.tools.background_task import BackgroundTaskEntry


if TYPE_CHECKING:
    from sagent.agent.agent import Agent
    from sagent.repl.render import Printer


logger = logging.getLogger(__name__)


__all__ = [
    "QUIT_WORDS",
    "InputSource",
    "StubInputSource",
    "spawn_repl_pump",
]


# Stable key for the REPL pump entry in ``agent.background_tasks``.
# Using a sentinel name (rather than a generated qid) makes the
# entry easy to look up and unit-test, and trivially distinct from
# user-scheduled background tools whose qids start with ``qid_``.
REPL_PUMP_KEY = "__repl_pump__"


class InputSource(Protocol):
    """Source of user input lines.

    Implementations include a real prompt-toolkit session and the
    in-process :class:`StubInputSource` used by tests.
    """

    async def next_line(self) -> str | None:
        """Return the next line, or ``None`` to terminate the input loop."""
        ...


class StubInputSource:
    """In-process queue of pre-staged lines for tests.

    Attributes:
      lines: Remaining lines to deliver, in order. ``None`` ends the loop.

    """

    def __init__(self, lines: list[str | None]) -> None:
        self._lines: list[str | None] = list(lines)

    async def next_line(self) -> str | None:
        if not self._lines:
            return None
        return self._lines.pop(0)


async def _input_pump(
    agent: Agent,
    source: InputSource,
    printer: Printer | None,
) -> None:
    """Read lines from ``source`` and post the parsed action to the inbox.

    Runs until the source returns ``None`` (EOF) or a ``quit`` action
    is dispatched. Either way: post ``text/x-quit`` to the inbox so
    the dispatch loop terminates cleanly. A pump-internal exception
    is logged and surfaced as ``text/x-error`` rather than killing
    the task silently -- otherwise the terminal accepts no further
    input with no signal as to why.
    """
    while True:
        try:
            line = await source.next_line()
            if line is None:
                _ = agent.inbox.put(TextMessage("", QUIT_SENTINEL))
                return
            action = parse_slash(line)
            if action is None:
                continue
            _apply(agent, action, printer)
            if action.quit:
                return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("REPL input pump raised; surfacing as error")
            _ = agent.inbox.put(
                TextMessage(
                    f"[input pump] {type(exc).__name__}: {exc}",
                    "text/x-error",
                ),
            )


def _apply(agent: Agent, action: SlashAction, printer: Printer | None) -> None:
    """Dispatch ``action`` and emit any echo line."""
    dispatch(agent, action)
    if action.echo is not None and printer is not None:
        printer.write_line(action.echo)


def spawn_repl_pump(
    agent: Agent,
    source: InputSource,
    *,
    printer: Printer | None = None,
) -> asyncio.Task[None]:
    """Spawn the REPL input pump as a hidden background task.

    The task lands in ``agent.background_tasks`` keyed by
    ``REPL_PUMP_KEY`` with ``hidden=True``: the ``BackgroundTask``
    tool's ``list`` operation filters it out, and ``/abort`` (which
    targets ``agent.tasks``) leaves it alone -- so user-initiated
    abort can never tear down the input loop.
    """
    task = asyncio.create_task(_input_pump(agent, source, printer))
    agent.background_tasks[REPL_PUMP_KEY] = BackgroundTaskEntry(
        task=task,
        tool_name="repl-input",
        queue_id=REPL_PUMP_KEY,
        started=time.time(),
        hidden=True,
    )
    return task
