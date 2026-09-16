"""``run_repl``: orchestrate an interactive REPL on top of an ``Agent``.

Builds a prompt-toolkit session, attaches one render observer to the
agent's observer list, spawns the input pump as a hidden background
task, and calls ``agent.serve_forever()``. Returns when the user
types ``/quit`` or sends EOF.

Important: the ``rich.Console`` is constructed INSIDE the
``patch_stdout`` context. ``patch_stdout`` swaps ``sys.stdout`` /
``sys.stderr`` for a proxy that routes writes above the prompt;
``rich.Console`` snapshots its file handle at construction. Building
the console outside the patch causes its writes to bypass the proxy.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import asyncio
import functools
import logging
import sys


if TYPE_CHECKING:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.patch_stdout import patch_stdout
    from prompt_toolkit.styles import Style as PTStyle
    from rich.console import Console
else:
    from wrapt import lazy_import

    PromptSession = lazy_import(
        "prompt_toolkit", "PromptSession"
    )  # ~80 ms; only run_repl uses it.
    AutoSuggestFromHistory = lazy_import(
        "prompt_toolkit.auto_suggest", "AutoSuggestFromHistory"
    )
    FileHistory = lazy_import("prompt_toolkit.history", "FileHistory")
    patch_stdout = lazy_import("prompt_toolkit.patch_stdout", "patch_stdout")
    PTStyle = lazy_import("prompt_toolkit.styles", "Style")
    Console = lazy_import("rich.console", "Console")  # ~60 ms; run_repl uses it.

from sagent.agent.session_io import unpersisted_session_error
from sagent.lib.userdirs import state_dir
from sagent.repl.console_pane import ConsolePrinter
from sagent.repl.input_pane import (
    REPL_PUMP_KEY,
    PromptToolkitInputSource,
    render_input_pane,
    spawn_repl_pump,
)
from sagent.repl.input_queues import InputQueues
from sagent.repl.keybindings import NavState, build_key_bindings
from sagent.repl.render import make_render_observer
from sagent.repl.replay import replay_messages
from sagent.repl.status_pane import render_status_pane
from sagent.tools.display import ToolDisplay, row_spec
from sagent.types.exceptions import log_exception_or_warning
from sagent.types.runtime import (
    AgentIdle,
    AssistantMessage,
    ClearComplete,
    RuntimeEvent,
)


if TYPE_CHECKING:
    from sagent.agent.agent import Agent

logger = logging.getLogger(__name__)


async def run_repl(
    agent: Agent,
    *,
    history: Path | None = None,
    show_thinking: bool = True,
) -> None:
    """Drive ``agent`` interactively until the user types ``/quit``.

    Args:
      agent: The agent to drive.
      history: Path to the input-history file. ``None`` -> the per-user
        state directory.
      show_thinking: Whether reasoning renders locally. Handed to the
        printer, which owns it: it decides what reaches this terminal,
        while ``settings.thinking_output`` decides what the model sends.

    """
    history_path = history or state_dir() / "rekursiv-ai" / "sagent" / "repl-history"
    # ``FileHistory.store_string`` opens with ``"ab"`` and never creates
    # parents. ``Buffer.append_to_history`` runs BEFORE ``buf.reset()`` in
    # ``_kb_submit``, so a missing directory makes Enter dispatch the
    # message and then raise -- leaving the typed text in the input pane.
    history_path.parent.mkdir(parents=True, exist_ok=True)
    style = PTStyle.from_dict(
        {
            "bottom-toolbar": "fg:ansibrightblack noreverse bg:default",
            "queued_input_pane": "fg:ansibrightblack",
            "input_pane": "bold",
        },
    )
    queues = InputQueues()
    nav = NavState()
    with patch_stdout(raw=True):
        console = Console(stderr=True)
        session: PromptSession[str] = PromptSession(
            functools.partial(render_input_pane, agent, queues),
            multiline=True,
            erase_when_done=True,
            history=FileHistory(str(history_path)),
            auto_suggest=AutoSuggestFromHistory(),
            bottom_toolbar=functools.partial(render_status_pane, agent),
            refresh_interval=0.2,
            key_bindings=build_key_bindings(agent, queues, nav),
            enable_open_in_editor=False,
            style=style,
        )
        printer = ConsolePrinter(console, show_thinking=show_thinking)
        render_observer = make_render_observer(
            printer,
            output_policy=lambda call_id: _tool_output_policy(agent, call_id),
        )
        # Installed before the ``try`` so each is unconditionally bound for the
        # ``finally``: neither call can raise, while replay and the title call
        # both can on real input (a corrupt tape, a closed terminal), which is
        # why those stay inside.
        agent.runtime.observers.append(render_observer)
        uninstall_committer = install_input_queue_committer(agent, queues)
        pump_task = spawn_repl_pump(
            agent,
            PromptToolkitInputSource(session, queues=queues, console=console),
            queues=queues,
            printer=printer,
        )
        try:
            replay_messages(agent, printer)
            if agent.status:
                printer.set_terminal_title(agent.status)
            elif agent.name:
                printer.set_terminal_title(agent.name)
            await agent.serve_forever()
        finally:
            uninstall_committer()
            if render_observer in agent.runtime.observers:
                agent.runtime.observers.remove(render_observer)
            agent.shutdown(force=True)
            bg_tasks = _background_tasks_for_repl_cancel(agent)
            for t in bg_tasks:
                _ = t.cancel()
            if bg_tasks:
                _ = await asyncio.gather(*bg_tasks, return_exceptions=True)
            _ = pump_task.cancel()
            try:
                await pump_task
            except asyncio.CancelledError:
                # Only OUR cancellation is expected here. If this task
                # is itself being cancelled, swallowing it would break
                # structured cancellation, so re-raise in that case.
                if not pump_task.cancelled():
                    raise
            except Exception as exc:  # noqa: BLE001 -- Pump shutdown must contain every handler failure while reporting it.
                log_exception_or_warning(
                    logger,
                    "REPL input pump raised during shutdown",
                    exc,
                )
            agent.cancel_background(REPL_PUMP_KEY)
    # A non-empty tape with no transcript on disk is silent data loss. The
    # runtime isolates observer exceptions (Runtime.publish), so a persistence
    # write failure cannot surface mid-turn; this end-of-session check is where
    # it is pulled. Emit a loud error and exit non-zero -- consistent with the
    # CLI's stderr-and-exit convention -- rather than print a resume hint for a
    # session that cannot be resumed.
    persistence_error = unpersisted_session_error(agent)
    if persistence_error is not None:
        _ = sys.stderr.write(f"FATAL: {persistence_error}\n")
        logger.error("session persistence failed: %s", persistence_error)
        sys.exit(1)
    if agent.session_dir is not None:
        _ = sys.stderr.write(
            "Resume this session with:\n"
            f"sagent --resume {agent.session_dir.name[:8]}  # this exact session (any prefix unique in this dir)\n"
            "sagent --continue         # most recent session in this dir\n"
            "sagent --resume           # interactive picker for this dir\n"
            "sagent --continue-all     # most recent session across all dirs\n"
            "sagent --resume-all       # interactive picker across all dirs\n",
        )


def install_input_queue_committer(
    agent: Agent,
    queues: InputQueues,
) -> Callable[[], None]:
    """Install the REPL-local queue committer; return its uninstall closure.

    Installs both halves of the committer:

    - Wraps ``agent.runtime.before_tool_spawn`` so urgent queue blocks
      flush as a ``UserMessage`` before the next tool spawns. The
      previous hook is preserved and called first; if it returns an
      event, that event wins and the queue stays put.
    - Appends an observer to ``agent.runtime.observers`` that commits
      urgent / deferred queues on each ``AgentIdle`` or ``ClearComplete``
      (the latter releases deferred input staged after a model self-clear,
      which never reaches ``AgentIdle`` because the ``Clear`` armed
      ``AWAIT_USER``).

    The caller invokes the returned uninstall in ``finally`` to detach
    the observer and restore the prior ``before_tool_spawn``. Without
    this, a re-entered ``run_repl`` on the same agent stacks observers
    and hook layers.

    Args:
      agent: Agent whose runtime gains the committer.
      queues: REPL-local urgent / deferred queues to flush.

    Returns:
      uninstall: Closure that reverses both install steps. Idempotent;
          safe to call multiple times.

    """
    previous_before_tool_spawn = agent.runtime.before_tool_spawn
    wrapped_before_tool_spawn = functools.partial(
        _before_tool_spawn,
        queues=queues,
        previous_before_tool_spawn=previous_before_tool_spawn,
    )
    agent.runtime.before_tool_spawn = wrapped_before_tool_spawn
    observer = _input_queue_committer_observer(agent, queues)
    agent.runtime.observers.append(observer)

    def uninstall() -> None:
        if observer in agent.runtime.observers:
            agent.runtime.observers.remove(observer)
        # Restore only when nothing downstream replaced our wrapper;
        # blindly assigning would clobber a later install that
        # legitimately owns the slot now.
        if agent.runtime.before_tool_spawn is wrapped_before_tool_spawn:
            agent.runtime.before_tool_spawn = previous_before_tool_spawn

    return uninstall


# Whether a result body renders is the TOOL's setting (e.g. ``--tool Bash.output=on``),
# so the renderer resolves the owning tool rather than carrying a display policy of its
# own.
def _tool_output_policy(agent: Agent, call_id: str) -> ToolDisplay:
    """Return the output policy for the tool behind ``call_id``."""
    name, _started = agent.tool_name_for_call(call_id)
    tool = agent.tools_map.get(name)
    return ToolDisplay() if tool is None else row_spec(tool)


# Subagents (either lifecycle) own their own ``serve_forever`` loop and must be stopped
# gracefully, never raw-cancelled from the REPL teardown -- so they are excluded here.
def _background_tasks_for_repl_cancel(agent: Agent) -> list[asyncio.Task[object]]:
    """Return unfinished REPL-owned background tasks safe to raw-cancel."""
    return [
        job.task
        for job in list(agent.background.values())
        if job.kind != "subagent" and not job.task.done()
    ]


# Module-private: production callers go through :func:`install_input_queue_committer`,
# which also installs the ``before_tool_spawn`` hook and returns the uninstall closure.
# Exposed for observer-only unit tests that exercise dispatch in isolation from the
# install / uninstall mechanics.
def _input_queue_committer_observer(
    agent: Agent,
    queues: InputQueues,
) -> Callable[[RuntimeEvent], None]:
    """Return the observer half of the queue committer."""
    return functools.partial(_commit_local_queues, agent=agent, queues=queues)


def _before_tool_spawn(
    message: AssistantMessage,
    *,
    queues: InputQueues,
    previous_before_tool_spawn: Callable[[AssistantMessage], RuntimeEvent | None]
    | None,
) -> RuntimeEvent | None:
    if previous_before_tool_spawn is not None:
        event = previous_before_tool_spawn(message)
        if event is not None:
            return event
    return queues.pop_queue_message()


def _commit_local_queues(
    event: RuntimeEvent,
    *,
    agent: Agent,
    queues: InputQueues,
) -> None:
    # ``ClearComplete`` flushes alongside ``AgentIdle``: a self-issued
    # ``Clear`` arms ``AWAIT_USER`` so ``_fully_drained`` stays False and
    # ``AgentIdle`` never publishes -- without this, deferred (Tab) input
    # staged after a model self-clear would wedge until Ctrl+D. ``Clear`` is
    # the only ``AWAIT_USER`` arm that publishes a distinguishing terminal
    # event (Halt / ModelResponseError do not), so the released input lands
    # exactly where a fresh user redirect would.
    if isinstance(event, (AgentIdle, ClearComplete)) and not queues.commit_queue(agent):
        queues.commit_deferred_on_idle(agent)
