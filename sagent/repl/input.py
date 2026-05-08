"""REPL input pump + ``LoginHandler`` + ``InputSource`` abstraction.

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
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, override

import asyncio
import time

from sagent import providers as _providers
from sagent.agent.handlers.base import InlineHandler
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
    from sagent.custom_types import Message
    from sagent.repl.render import Printer


__all__ = [
    "QUIT_WORDS",
    "InputSource",
    "LoginHandler",
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
    the dispatch loop terminates cleanly.
    """
    while True:
        line = await source.next_line()
        if line is None:
            _ = agent.inbox.put(TextMessage("", QUIT_SENTINEL))
            return
        action = parse_slash(line)
        if action is None:
            continue
        if action.descriptor == "text/x-user-message":
            # Pass the original (untrimmed) line through.
            _ = agent.inbox.put(TextMessage(line, "text/x-user-message"))
            continue
        _apply(agent, action, printer)
        if action.quit:
            return


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


_HELP_TEXT = """\
sagent commands

  /help                       this list
  /quit                       exit

  /clear                      wipe context (logs preserved on disk)
  /compact [hints]            compact history
  /uncompact [hints]          revert compaction

  /model    [args]            switch model
  /provider <name>            switch provider
  /login                      re-auth current provider

  /tasks                      list running work (agents + fg + bg)
  /break    [<label>|all]     cancel current step          (Ctrl+Z analog)
  /abort    [<label>|all]     cancel step + queue          (Ctrl+C analog)
                              "all" also kills background tasks\
"""


class HelpHandler(InlineHandler):
    """Print the slash-command reference on ``text/x-help-request``.

    The verbatim block lives in ``_HELP_TEXT`` so the parser, the
    handler, and the user are all reading the same canonical list.
    """

    descriptors: tuple[str, ...] = ("text/x-help-request",)

    def __init__(self, printer: Printer | None = None) -> None:
        self._printer = printer

    @override
    async def handle(self, agent: Agent, msg: Message) -> None:
        del agent, msg
        if self._printer is not None:
            self._printer.write_line(_HELP_TEXT)


class TasksHandler(InlineHandler):
    """Print live work on ``text/x-tasks-request``.

    Lists every registered agent in :data:`agent_registry` with its
    foreground task count and visible background task summaries.
    Hidden bg tasks (REPL pump, daemons) are filtered out -- they're
    infrastructure, not user-actionable. Foreground tasks are reported
    as a count rather than per-task because the in-flight registry
    keys by task identity (``id(task)``), which isn't meaningful to
    surface.
    """

    descriptors: tuple[str, ...] = ("text/x-tasks-request",)

    def __init__(self, printer: Printer | None = None) -> None:
        self._printer = printer

    @override
    async def handle(self, agent: Agent, msg: Message) -> None:
        del msg
        if self._printer is None:
            return
        # Lazy import to avoid a top-level cycle (tools.core <-> repl).
        from sagent.tools.core import agent_registry  # noqa: PLC0415

        lines: list[str] = []
        now = time.time()
        total_fg = 0
        total_bg = 0
        for label, other in agent_registry.items():
            visible_bg = [j for j in other.background_tasks.values() if not j.hidden]
            fg = len(other.tasks)
            bg = len(visible_bg)
            total_fg += fg
            total_bg += bg
            tag = " (self)" if other is agent else ""
            lines.append(f"  {label}{tag:<8s}  fg={fg} bg={bg}")
            for job in visible_bg:
                phase = (
                    "cancelled"
                    if job.task.cancelled()
                    else (
                        "completed"
                        if job.task.done()
                        else (
                            "sleeping"
                            if job.delay_sec > 0 and (now - job.started) < job.delay_sec
                            else "running"
                        )
                    )
                )
                lines.append(
                    f"    bg: {job.queue_id:<10s}  {job.tool_name:<16s}  "
                    f"{phase:<10s}  {now - job.started:.0f}s"
                )
        header = (
            f"sagent: {len(agent_registry)} agent(s), "
            f"{total_fg} foreground, {total_bg} background"
        )
        out = header + "\n" + "\n".join(lines) if lines else header
        self._printer.write_line(out)


class LoginHandler(InlineHandler):
    """Handle ``text/x-login-request`` by invoking the provider's ``login``.

    Slash-command parsing emits ``text/x-login-request`` so both REPL
    paths (active keybinding, idle prompt) take the same code path; the
    actual re-auth lives here.
    """

    descriptors: tuple[str, ...] = ("text/x-login-request",)

    def __init__(self, printer: Printer | None = None) -> None:
        self._printer = printer

    @override
    async def handle(self, agent: Agent, msg: Message) -> None:
        del msg
        spec = agent.model_spec
        if spec is None:
            self._write("[/login] agent has no model spec")
            return
        prov_cls = getattr(_providers, spec.provider, None)
        if prov_cls is None:
            self._write(f"[/login] unknown provider {spec.provider!r}")
            return
        login_fn = getattr(prov_cls, "login", None)
        if login_fn is None:
            self._write(f"[/login] {spec.provider} has no login method")
            return
        try:
            login_fn()
            self._write(f"[/login] {spec.provider} re-authenticated")
        except (RuntimeError, OSError, ValueError, TimeoutError) as exc:
            self._write(f"[/login] {exc}")

    def _write(self, line: str) -> None:
        if self._printer is not None:
            self._printer.write_line(line)
