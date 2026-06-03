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

- No navigation active (``cursor == 0``): send immediately when idle,
  during tool wait, or while awaiting user input. While a model is
  streaming, stage the text in the urgent queue; it is still editable
  with Up and fires at ``ModelResponseComplete`` before tool calls run.
- Navigation active (``cursor > 0``): commit the buffer back to the
  queued lanes, preserving urgent/deferred lane metadata from the
  snapshot. The snapshot is discarded; ``cursor`` returns to 0.
- Text ending in ``\``: backslash continuation -- replace trailing
  ``\`` with literal ``\n``, stay in buffer, do not dispatch.
- Empty buffer: nothing.
- Slash command: route through pump.

Tab
~~~

Tab stages the buffer in the deferred queue (REPL-local). The
``install_input_queue_committer`` observer commits deferred blocks as a
single ``UserQueuedMessage`` on ``AgentIdle``. If the runtime is already
awaiting user input, Tab commits immediately because no future
``AgentIdle`` will release the gate.

Urgent/deferred queues are local draft state. Up-arrow's lift is a true
retract because the queued text has not entered runtime history.

Headless callers without a Tab key use the ``/defer <text>`` slash
command, which pushes ``UserDeferredMessage`` directly through the pump
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
for the input zone: an optional dim queue preview above the ``> ``
prompt sigil whenever urgent/deferred local queues or runtime pending
mid-stream messages exist. Local queue entries are removed by the
keybinding/observer that commits or restores them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, assert_never, override

import asyncio
import fnmatch
import logging
import re
import time

from prompt_toolkit.formatted_text import FormattedText
from rich.text import Text

from sagent.agent.background import BackgroundTaskEntry
from sagent.lib.lazy_import import lazy_import
from sagent.repl.input_queues import InputQueues
from sagent.repl.slash import (
    QUIT_WORDS,
    Clear as SlashClear,
    Compact as SlashCompact,
    Defer as SlashDefer,
    Effort as SlashEffort,
    Halt as SlashHalt,
    Help as SlashHelp,
    Kill as SlashKill,
    Login as SlashLogin,
    ModelSwitch as SlashModelSwitch,
    Quit as SlashQuit,
    Recompact as SlashRecompact,
    Send as SlashSend,
    SlashAction,
    Tasks as SlashTasks,
    Text as SlashText,
    Thinking as SlashThinking,
    Unknown as SlashUnknown,
    parse_slash,
)
from sagent.tools.background_task import cancel_persistent_subagent
from sagent.tools.core import agent_registry
from sagent.types.exceptions import (
    UserFacingError,
    log_exception_or_warning,
    log_task_exception,
)
from sagent.types.runtime import (
    AgentSendMessage,
    Clear,
    Compact,
    Quit,
    Recompact,
    UserDeferredMessage,
    UserMessage,
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
    from sagent.agent.state import AgentLike
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

# Max characters of the discard-preview body. Long enough to be
# recognisable, short enough to stay on one console line.
_PREVIEW_CHARS = 80


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
    queues: InputQueues | None = None,
    printer: Printer | None = None,
) -> asyncio.Task[None]:
    """Spawn the REPL input pump as a hidden background task.

    Args:
      agent: Agent to drive.
      source: Where lines come from (prompt-toolkit in production).
      queues: Optional REPL-local queues to flush after slash recovery commands.
      printer: Optional sink for status echoes (``/help``, ``/tasks``,
          ``/login``, ``/model``).

    Returns:
      task: The running pump task.

    """
    task = asyncio.create_task(_input_pump(agent, source, queues, printer))
    task.add_done_callback(
        log_task_exception(logger, "REPL input pump crashed"),
    )
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
    queues: InputQueues | None,
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
            should_exit = await _dispatch(agent, action, printer, queues=queues)
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


def _dispatch_send(sender: Agent, action: SlashSend, printer: Printer | None) -> None:
    """Dispatch ``/send`` content to matching persistent subagents.

    Routed text lands as an ``AgentSendMessage`` attributed to ``sender``
    so the child sees the message as a parent-to-child handoff rather
    than anonymous human input -- mirroring the AgentSend tool path and
    keeping renderer/replay attribution bars (`repl/render.py`,
    `repl/replay.py`) consistent.
    """
    try:
        targets = _resolve_targets(action.target)
    except UserFacingError as exc:
        if printer is not None:
            printer.write_tool_error(f"[/send] {exc}")
        return
    if not targets:
        if printer is not None:
            printer.write_tool_error(f"[/send] no matching subagents: {action.target}")
        return
    source = sender.name or "user"
    for label in targets:
        target = agent_registry[label]
        if action.content.startswith("/"):
            _dispatch_target_control(target, action.content, printer, label=label)
        else:
            target.runtime.inbox.push_back(
                AgentSendMessage(source=source, text=action.content)
            )
            if printer is not None:
                printer.write_slash_block(f"[/send {label}] sent")


def _dispatch_target_control(
    target: AgentLike,
    body: str,
    printer: Printer | None,
    *,
    label: str,
) -> None:
    """Dispatch a slash command against one targeted subagent.

    Supported controls mirror the agent-local pump: ``/model``,
    ``/thinking``, ``/halt``, ``/quit``, ``/clear``, ``/compact``,
    ``/kill``. Anything else surfaces as an error.
    """
    action = parse_slash(body)
    if isinstance(action, SlashModelSwitch):
        _run_repl.do_switch_model(target, action.args, printer)
        return
    if isinstance(action, SlashThinking):
        _run_repl.do_switch_thinking(target, action.command, printer)
        return
    if isinstance(action, SlashEffort):
        _run_repl.do_switch_effort(target, action.value, printer)
        return
    if isinstance(action, SlashHalt):
        target.halt()
        return
    if isinstance(action, SlashQuit):
        target.runtime.inbox.push_back(Quit())
        return
    if isinstance(action, SlashClear):
        target.runtime.inbox.push_back(Clear())
        return
    if isinstance(action, SlashCompact):
        target.runtime.inbox.push_back(Compact(args=action.args))
        return
    if isinstance(action, SlashKill):
        if action.target == "all":
            target.kill_all_tools()
        else:
            target.kill_tool(action.target)
        return
    if printer is not None:
        printer.write_tool_error(f"[/send {label}] unsupported control: {body}")


def _resolve_targets(pattern: str) -> list[str]:
    """Resolve an exact, glob, brace-list, or regex subagent target."""
    labels = [
        label
        for label, agent in agent_registry.items()
        if _is_persistent_subagent(agent)
    ]
    if pattern.startswith("{") and pattern.endswith("}"):
        wanted = [part.strip() for part in pattern[1:-1].split(",") if part.strip()]
        return [label for label in wanted if label in labels]
    if pattern.startswith("/") and pattern.endswith("/") and len(pattern) >= 3:
        # ``len(pattern) >= 3`` rejects ``/`` and ``//``: an empty-body
        # regex matches every label and would silently fan ``/halt /``
        # out to every persistent subagent.
        try:
            regex = re.compile(pattern[1:-1])
        except re.error as exc:
            raise UserFacingError(f"invalid target regex: {exc}") from exc
        return [label for label in labels if regex.search(label)]
    if any(char in pattern for char in "*?["):
        return [label for label in labels if fnmatch.fnmatchcase(label, pattern)]
    return [pattern] if pattern in labels else []


def _is_persistent_subagent(agent: AgentLike) -> bool:
    """Return true when ``agent`` is a live persistent subagent."""
    return bool(getattr(agent, "_persistent", False))


def _dispatch_halt(
    agent: Agent,
    action: SlashHalt,
    printer: Printer | None,
) -> None:
    """Halt the current agent, every persistent subagent, or matching labels.

    ``/halt`` halts the current agent. ``/halt all`` mirrors ``/kill all``
    and halts every persistent subagent. Any other target is resolved
    against the persistent-subagent registry (exact, glob, brace-list,
    or regex).
    """
    if not action.target:
        agent.halt()
        return
    if action.target == "all":
        for label, target in agent_registry.items():
            if _is_persistent_subagent(target):
                target.halt()
                if printer is not None:
                    printer.write_slash_block(f"[/halt {label}] halted")
        return
    targets = _resolve_targets(action.target)
    if not targets:
        if printer is not None:
            printer.write_tool_error(f"[/halt] no matching subagents: {action.target}")
        return
    for label in targets:
        agent_registry[label].halt()
        if printer is not None:
            printer.write_slash_block(f"[/halt {label}] halted")


def _dispatch_kill(
    agent: Agent,
    action: SlashKill,
    printer: Printer | None,
    *,
    queues: InputQueues | None = None,
) -> None:
    """Cancel tool tasks or matching persistent subagents.

    ``/kill all`` also clears any REPL-local urgent/deferred staging so
    the user is not left with a "pending" pane that no longer matches
    their intent.
    """
    if action.target == "all":
        agent.kill_all_tools()
        if queues is not None:
            queues.clear()
        if printer is not None:
            printer.write_slash_block("[/kill] cancelled all tool tasks")
        return
    owner, sep, job_id = action.target.partition("/")
    if sep:
        target = agent_registry.get(owner)
        if target is None:
            if printer is not None:
                printer.write_tool_error(f"[/kill] unknown owner: {owner}")
            return
        target.kill_tool(job_id)
        if printer is not None:
            printer.write_slash_block(f"[/kill {owner}/{job_id}] cancelled")
        return
    targets = _resolve_targets(action.target)
    if targets:
        for label in targets:
            _kill_persistent_subagent(agent, label)
            if printer is not None:
                printer.write_slash_block(f"[/kill {label}] cancelled")
        return
    agent.kill_tool(action.target)
    if printer is not None:
        printer.write_slash_block(f"[/kill] cancelled {action.target}")


def _kill_persistent_subagent(agent: Agent, label: str) -> None:
    """Cancel one persistent subagent through the unified graceful path."""
    _ = cancel_persistent_subagent(agent, label)


async def _dispatch(
    agent: Agent,
    action: SlashAction,
    printer: Printer | None,
    *,
    queues: InputQueues | None = None,
) -> bool:
    """Dispatch one parsed slash action; return True to exit the pump.

    Exhaustive over :class:`SlashAction`; ``assert_never`` makes the
    type checker flag any newly added variant that forgets a handler.
    """
    match action:
        case SlashQuit():
            agent.shutdown(force=False)
            return True
        case SlashHalt():
            _dispatch_halt(agent, action, printer)
        case SlashKill():
            _dispatch_kill(agent, action, printer, queues=queues)
        case SlashClear():
            agent.runtime.inbox.push_back(Clear())
            if printer is not None:
                printer.write_slash_block("[/clear] history cleared")
        case SlashCompact(args=args):
            agent.runtime.inbox.push_back(Compact(args=args))
            if printer is not None:
                note = f" ({args})" if args else ""
                printer.write_slash_block(f"[/compact] queued{note}")
        case SlashRecompact(args=args):
            agent.runtime.inbox.push_back(Recompact(args=args))
            if printer is not None:
                note = f" ({args})" if args else ""
                printer.write_slash_block(f"[/recompact] queued{note}")
        case SlashModelSwitch(args=args):
            _run_repl.do_switch_model(agent, args, printer)
        case SlashThinking(command=command):
            _run_repl.do_switch_thinking(agent, command, printer)
        case SlashEffort(value=value):
            _run_repl.do_switch_effort(agent, value, printer)
        case SlashLogin():
            await _run_repl.do_login(agent, printer)
            if queues is not None:
                queues.commit_deferred_on_idle(agent)
        case SlashHelp():
            if printer is not None:
                printer.write_line(_render.HELP_TEXT)
        case SlashTasks():
            if printer is not None:
                printer.write_line(_run_repl.format_tasks(agent))
        case SlashText(content=content):
            agent.runtime.inbox.push_back(UserMessage(text=content))
        case SlashDefer(content=content):
            agent.runtime.inbox.push_back(UserDeferredMessage(text=content))
        case SlashSend():
            _dispatch_send(agent, action, printer)
        case SlashUnknown(text=text):
            if printer is not None:
                printer.write_tool_error(text)
        case _:
            assert_never(action)
    return False


class PromptToolkitInputSource(InputSource):
    """Async input source backed by a :class:`prompt_toolkit.PromptSession`.

    The ``queued_input`` field mirrors the REPL-local buffer of texts
    the user typed while the agent was busy. The runtime's
    ``GatedDeque`` doesn't support tag-based peek / pop, so the REPL
    maintains this list itself; the dim preview rendered by
    :func:`render_input_pane` shows the tail and ``Up`` lifts it back.
    """

    queues: InputQueues
    """REPL-local urgent/deferred queues."""

    def __init__(
        self,
        session: PromptSession[str],
        *,
        queues: InputQueues,
        console: Console | None = None,
    ) -> None:
        self._session = session
        self.queues = queues
        self._console = console

    @override
    async def next_line(self) -> str | None:
        """Return the next line, or ``None`` to terminate the input loop."""
        try:
            text = await self._session.prompt_async(set_exception_handler=False)
        except (EOFError, KeyboardInterrupt):
            self._surface_queued_input_on_quit()
            return None
        stripped = text.strip()
        if stripped.lower() in QUIT_WORDS:
            self._surface_queued_input_on_quit()
            return None
        return text

    def _surface_queued_input_on_quit(self) -> None:
        """Surface the tail of ``queued_input`` before the loop ends.

        Mentions the total block count and marks truncated previews
        with an ellipsis so the operator can tell at a glance that the
        single line they see represents more than what was discarded.
        """
        if not self.queues.has_any() or self._console is None:
            return
        total = len(self.queues.urgent) + len(self.queues.deferred)
        raw = self.queues.peek_tail_preview().replace("\n", " ")
        truncated = len(raw) > _PREVIEW_CHARS
        preview = raw[:_PREVIEW_CHARS] + ("…" if truncated else "")
        noun = "message" if total == 1 else "messages"
        self._console.print(
            Text(
                f"[discarding {total} queued {noun}: {preview}]",
                style="dim yellow",
            ),
        )
        self.queues.clear()


def render_input_pane(agent: Agent, queues: InputQueues) -> FormattedText:
    r"""Build the input-pane ``FormattedText``: full queue + prompt sigil.

    Renders all pending user content above the ``> `` prompt sigil
    (blocks joined by ``\n\n``). Two buffers contribute:

    - ``queues`` -- REPL-local urgent/deferred blocks. Up-arrow can lift
      these back into the input buffer for editing.
    - ``agent.runtime._mid_stream_queue`` -- externally submitted
      mid-stream blocks, already committed to the runtime. Shown for
      visibility when present.

    Both are pending model consumption; rendering them together gives
    the user one canonical "what's queued" surface.

    Args:
      agent: Agent whose runtime ``_mid_stream_queue`` is consulted.
      queues: REPL-local urgent/deferred input queues.

    Returns:
      formatted: The input pane's formatted text.

    """
    blocks = queues.render_blocks()
    # Only surface human-typed pending items. ``_mid_stream_queue``
    # also buffers ``AgentSendMessage`` arriving while the model is
    # streaming, but agent-to-agent payloads are runtime plumbing the
    # user can't edit or retract -- they don't belong in the queue
    # preview meant for staged human input.
    blocks.extend(
        f"pending: {m.text}"
        for m in agent.runtime.pending_mid_stream()
        if isinstance(m, UserMessage)
    )
    parts: list[tuple[str, str]] = []
    if blocks:
        parts.append(("class:queued_input_pane", "\n\n".join(blocks)))
        parts.append(("", "\n"))
    parts.append(("class:input_pane", "> "))
    return FormattedText(parts)
