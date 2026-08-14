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
from typing import TYPE_CHECKING, Final

import asyncio
import dataclasses
import functools
import logging
import shlex
import sys
import time

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style as PTStyle
from rich.console import Console

from sagent.agent.background import BackgroundTaskEntry
from sagent.agent.session_io import unpersisted_session_error
from sagent.agent.state import agent_registry
from sagent.lib.userdirs import state_dir
from sagent.providers import (
    infer_provider,
    supported_provider_options,
)
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
from sagent.thinking import ThinkingState, resolve_thinking_command
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
    from sagent.repl.render import Printer

logger = logging.getLogger(__name__)


async def run_repl(
    agent: Agent,
    *,
    history: Path | None = None,
) -> None:
    """Drive ``agent`` interactively until the user types ``/quit``.

    Args:
      agent: The agent to drive.
      history: Path to the input-history file. ``None`` -> the per-user
        state directory.

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
        printer = ConsolePrinter(console)
        render_observer = make_render_observer(
            printer,
            show_thinking=lambda: agent.show_thinking,
            output_policy=lambda call_id: _tool_output_policy(agent, call_id),
        )
        # Everything that mutates the runtime lives inside the ``try``: the
        # ``finally`` below is the only path that detaches the observer,
        # restores ``before_tool_spawn``, and cancels the pump. Replay and
        # the title call can both raise on real input (a corrupt tape, a
        # closed terminal), so installing outside would leak all three.
        uninstall_committer: Callable[[], None] | None = None
        pump_task: asyncio.Task[None] | None = None
        try:
            agent.runtime.observers.append(render_observer)
            uninstall_committer = install_input_queue_committer(agent, queues)
            pump_task = spawn_repl_pump(
                agent,
                PromptToolkitInputSource(session, queues=queues, console=console),
                queues=queues,
                printer=printer,
            )
            replay_messages(agent, printer)
            if agent.status:
                printer.set_terminal_title(agent.status)
            elif agent.name:
                printer.set_terminal_title(agent.name)
            await agent.serve_forever()
        finally:
            # Detach FIRST. These two lines are the whole reason the
            # ``try`` exists; anything ahead of them (shutdown, task
            # cancellation) can raise and would skip them, re-opening the
            # leak. Detaching early is safe -- the observer only renders.
            if uninstall_committer is not None:
                uninstall_committer()
            if render_observer in agent.runtime.observers:
                agent.runtime.observers.remove(render_observer)
            agent.shutdown(force=True)
            bg_tasks = _background_tasks_for_repl_cancel(agent)
            for t in bg_tasks:
                _ = t.cancel()
            if bg_tasks:
                _ = await asyncio.gather(*bg_tasks, return_exceptions=True)
            if pump_task is not None:
                _ = pump_task.cancel()
                try:
                    await pump_task
                except asyncio.CancelledError:
                    # Only OUR cancellation is expected here. If this task
                    # is itself being cancelled, swallowing it would break
                    # structured cancellation, so re-raise in that case.
                    if not pump_task.cancelled():
                        raise
                except Exception as exc:
                    log_exception_or_warning(
                        logger, "REPL input pump raised during shutdown", exc
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
            "sagent --resume-all       # interactive picker across all dirs\n"
        )


def _tool_output_policy(agent: Agent, call_id: str) -> ToolDisplay:
    """Return the output policy for the tool behind ``call_id``.

    Whether a result body renders is the TOOL's setting (e.g.
    ``--tool Bash.output=on``), so the renderer resolves the owning tool
    rather than carrying a display policy of its own.
    """
    name, _started = agent.tool_name_for_call(call_id)
    tool = agent.tools_map.get(name)
    return ToolDisplay() if tool is None else row_spec(tool)


def _background_tasks_for_repl_cancel(agent: Agent) -> list[asyncio.Task[object]]:
    """Return unfinished REPL-owned background tasks safe to raw-cancel.

    Subagents (either lifecycle) own their own ``serve_forever`` loop and
    must be stopped gracefully, never raw-cancelled from the REPL
    teardown -- so they are excluded here.
    """
    return [
        job.task
        for job in list(agent.background.values())
        if job.kind != "subagent" and not job.task.done()
    ]


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


def _input_queue_committer_observer(
    agent: Agent,
    queues: InputQueues,
) -> Callable[[RuntimeEvent], None]:
    """Return the observer half of the queue committer.

    Module-private: production callers go through
    :func:`install_input_queue_committer`, which also installs the
    ``before_tool_spawn`` hook and returns the uninstall closure.
    Exposed for observer-only unit tests that exercise dispatch in
    isolation from the install / uninstall mechanics.
    """
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


def do_switch_model(
    agent: Agent,
    args: str,
    printer: Printer | None,
) -> None:
    """Render a ``/model`` slash command against :meth:`Agent.change_model`.

    Pure REPL adapter: parses the slash syntax, delegates the swap to
    the Agent API, prints the resulting label or error.

    Args:
      agent: Agent to mutate.
      args: Trailing arguments after ``/model`` (parsed via shlex).
      printer: Optional sink for status messages.

    """
    spec = agent.model_recipe
    if spec is None:
        _write(printer, "[/model] agent has no model spec; cannot swap.")
        return
    try:
        tokens = shlex.split(args)
    except ValueError as e:
        _write(printer, f"[/model] parse error: {e}")
        return
    if not tokens:
        _write(
            printer,
            f"[/model] provider={spec.provider} auth={spec.auth} "
            f"model={spec.model_id} account={spec.account or 'default'}",
        )
        return
    parsed = _parse_model_args(tokens)
    if isinstance(parsed, str):
        _write(printer, parsed)
        return
    # Bare model_id on the SAME provider may imply a different provider
    # (e.g. ``/model gemini-3-pro`` while on Anthropic). Infer.
    prov_override = parsed.provider
    auth_override = parsed.auth
    if parsed.model_id and parsed.provider is None:
        inferred = infer_provider(parsed.model_id, spec.provider)
        if inferred is not None:
            prov_override, auth_override = inferred
    old_id = agent.model.spec.tagged_model_id
    try:
        target = agent.change_model(
            provider=prov_override,
            auth=auth_override,
            model_id=parsed.model_id,
            account=parsed.account if parsed.account_set else None,
        )
    except (ValueError, RuntimeError) as exc:
        _write(printer, f"[/model] {exc}")
        return
    if target.provider != spec.provider:
        label = f"{spec.provider}/{old_id} -> {target.provider}/{target.model_id}"
    else:
        label = f"{old_id} -> {target.model_id}"
    queued = " (queued)" if agent.work is not None else ""
    _write(printer, f"[/model] {label}{queued}")


def do_switch_thinking(agent: Agent, command: str, printer: Printer | None) -> None:
    """Render a ``/thinking`` slash command against agent/provider state.

    Args:
      agent: Agent to mutate.
      command: Full thinking state or partial command.
      printer: Optional sink for status messages.

    """
    current = agent.thinking_state or _infer_thinking_state(agent)
    if not command:
        valid = ", ".join(agent.model.spec.valid_thinking_states)
        _write(printer, f"[/thinking] {current}\n[/thinking] options: {valid}")
        return
    try:
        state = resolve_thinking_command(command, current)
    except ValueError as exc:
        _write(printer, f"[/thinking] {exc}")
        return
    valid = agent.model.spec.valid_thinking_states
    if state not in valid:
        options = ", ".join(valid)
        _write(
            printer,
            f"[/thinking] {state} not supported by {agent.model.spec.tagged_model_id!r};"
            f" options: {options}",
        )
        return
    if agent.thinking_state == state:
        _write(printer, f"[/thinking] {state}")
        return
    # Snapshot the thinking fields so the ``_rebuild_current_model``
    # failure branch can roll back transactionally.
    # ``set_thinking_state`` is a pure attribute mutation, so capturing
    # the pre-state once is sufficient.
    old_state = agent.thinking_state
    old_thinking = agent.thinking
    old_show = agent.show_thinking
    agent.set_thinking_state(state)
    # Only a redact-supporting provider needs a rebuild (redaction is a
    # provider-construction knob derived from the thinking state); on
    # other providers the state change is purely local.
    if _provider_supports_redact(agent) and not _rebuild_current_model(agent, printer):
        agent.restore_thinking_state(old_state, old_thinking, old_show)
        return
    _write(printer, f"[/thinking] {state}")


def do_switch_effort(agent: Agent, value: str, printer: Printer | None) -> None:
    """Render an ``/effort`` slash command against agent/model state.

    Bare ``/effort`` (empty ``value``) prints the current effort plus the
    model's valid options. A non-empty value sets the effort; ``off`` /
    ``unset`` clears it. ``none`` is NOT a clear alias -- some providers
    (OpenAI, self-hosted) accept a literal ``none`` effort, so clearing
    uses unambiguous words only. Invalid values error with the option
    list -- a rejected request, never a silent no-op.

    Args:
      agent: Agent to mutate.
      value: Effort value, ``""`` for status, or a clear alias.
      printer: Optional sink for status messages.

    """
    valid = agent.model.spec.supported_thinking_efforts
    if not value:
        current = agent.effort or "unset"
        options = ", ".join(valid) or "(none)"
        _write(printer, f"[/effort] {current}\n[/effort] options: {options}")
        return
    target = None if value in ("off", "unset") else value
    try:
        agent.effort = target
    except ValueError as exc:
        _write(printer, f"[/effort] {exc}")
        return
    _write(printer, f"[/effort] {agent.effort or 'unset'}")


def _infer_thinking_state(agent: Agent) -> ThinkingState:
    """Infer a canonical current state from legacy thinking/display fields."""
    if agent.thinking is None:
        return "off-hide"
    if agent.thinking == "enabled":
        return "on-show" if agent.show_thinking else "on-hide"
    return "adaptive-show" if agent.show_thinking else "adaptive-hide"


def _provider_supports_redact(agent: Agent) -> bool:
    """Return whether the current provider takes the redact-thinking option."""
    spec = agent.model_recipe
    if spec is None:
        return False
    try:
        return "redact_thinking" in supported_provider_options(spec.provider)
    except AttributeError:
        return False


def _rebuild_current_model(agent: Agent, printer: Printer | None) -> bool:
    """Queue a rebuild of the current provider/model with stored provider args."""
    spec = agent.model_recipe
    if spec is None:
        _write(printer, "[/thinking] agent has no model spec; cannot rebuild model")
        return False
    try:
        target = agent.change_model()
    except (ValueError, RuntimeError) as exc:
        _write(printer, f"[/thinking] {exc}")
        return False
    queued = " (queued)" if agent.work is not None else ""
    _write(printer, f"[/thinking] provider args updated for {target.model_id}{queued}")
    return True


async def do_login(agent: Agent, printer: Printer | None) -> None:
    """Render a ``/login`` slash command against :meth:`Agent.relogin`.

    Pure REPL adapter: delegates the re-auth flow to the Agent API,
    prints success or error.

    Args:
      agent: Agent whose provider should be re-authenticated.
      printer: Optional sink for status messages.

    """
    spec = agent.model_recipe
    if spec is None:
        _write(printer, "[/login] agent has no model spec")
        return
    try:
        await agent.relogin()
    except (ValueError, RuntimeError, OSError, TimeoutError) as exc:
        _write(printer, f"[/login] {exc}")
        return
    _write(printer, f"[/login] {spec.provider} re-authenticated")


def format_tasks(agent: Agent) -> str:
    """Format running fg/bg work across every registered agent.

    Args:
      agent: Agent used to mark the "(self)" row in the listing.

    Returns:
      summary: Multi-line summary header followed by one row per agent
          and one indented row per visible background job.

    """
    lines: list[str] = []
    now = time.time()
    total_fg = 0
    total_bg = 0
    for label, other in agent_registry.items():
        visible_bg = [j for j in other.background.values() if not j.hidden]
        # ``AgentLike`` doesn't expose ``work`` (a foreground convenience
        # on ``Agent``); derive the same condition from runtime state
        # the Protocol does promise.
        runtime = other.runtime
        fg_active = runtime.model_call is not None or runtime.compact_task is not None
        fg = 1 if fg_active else 0
        bg_n = len(visible_bg)
        total_fg += fg
        total_bg += bg_n
        tag = " (self)" if other is agent else ""
        lines.append(f"  {label}{tag:<8s}  fg={fg} bg={bg_n}")
        for job in visible_bg:
            if job.kind == "subagent":
                phase = _subagent_phase(job)
            else:
                phase = _generic_job_phase(job, now)
            lines.append(
                f"    bg: {label}/{job.queue_id:<10s}  {job.tool_name:<16s}  "
                f"{phase:<10s}  {now - job.started:.0f}s"
            )
    header = (
        f"sagent: {len(agent_registry)} agent(s), "
        f"{total_fg} foreground, {total_bg} background"
    )
    if lines:
        return header + "\n" + "\n".join(lines)
    return header


def _generic_job_phase(job: BackgroundTaskEntry, now: float) -> str:
    """Phase label for non-persistent-subagent bg jobs.

    ``"errored"`` distinguishes crashes from graceful ``"completed"``;
    parallels :func:`_subagent_phase`'s same distinction so both bg-row
    families surface failures the same way.
    """
    if job.task.cancelled():
        return "cancelled"
    if job.task.done():
        try:
            exc = job.task.exception()
        except (asyncio.CancelledError, asyncio.InvalidStateError):
            exc = None
        return "errored" if exc is not None else "completed"
    if job.delay_sec > 0 and (now - job.started) < job.delay_sec:
        return "sleeping"
    return "running"


def _subagent_phase(job: BackgroundTaskEntry) -> str:
    """Return a lifecycle label for a persistent-subagent bg-job row.

    Reads child runtime state directly -- safe because asyncio is
    single-threaded and ``format_tasks`` contains no ``await``.

    Args:
      job: The ``BackgroundTaskEntry`` for the persistent subagent.

    Returns:
      phase: One of ``"idle"``, ``"running"``, ``"compacting"``,
          ``"tool-wait"``, ``"gate-armed"``, ``"errored"``, or
          ``"stopped"``.

    """
    if job.task.done():
        # Distinguish crash from graceful exit so the operator can
        # tell whether a missing child was intentional.
        try:
            exc = job.task.exception()
        except (asyncio.CancelledError, asyncio.InvalidStateError):
            exc = None
        return "errored" if exc is not None else "stopped"
    child = agent_registry.get(job.queue_id)
    if child is None:
        return "running"
    rt = child.runtime
    if rt.model_call is not None:
        return "running"
    if rt.compact_task is not None:
        return "compacting"
    if rt.cohort:
        return "tool-wait"
    if rt.inbox.gate_armed:
        return "gate-armed"
    return "idle"


def _write(printer: Printer | None, line: str) -> None:
    """Forward ``line`` to ``printer.write_slash_block`` when a printer is wired.

    Slash-command output (``/model``, ``/thinking``, ``/login``,
    ``/tasks``) renders as machinery, not user text -- dim, no user
    bar -- so the operator can tell at a glance which lines are
    REPL infrastructure vs. agent dialogue.
    """
    if printer is not None:
        printer.write_slash_block(line)


_FLAG_PROVIDER: Final = ("--provider", "-p")
_FLAG_AUTH: Final = ("--auth", "-a")
_FLAG_ACCOUNT: Final = ("--account",)
_KV_KEYS = frozenset({"provider", "auth", "account", "model", "model_id"})


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _ParsedModelArgs:
    """Only the fields the user explicitly typed; rest stay ``None``.

    ``account_set`` disambiguates "user typed ``account=``" from "user
    didn't mention account at all." The typed value is preserved
    verbatim, including ``""`` and ``"default"``: those name the legacy
    unnamed account (``resolve_account`` / ``credentials_path`` in
    ``providers/lib/oauth.py`` collapse both onto the default file), and
    normalising them to ``None`` here would mean "inherit the current
    account" instead -- leaving no way to switch back off a named one.

    """

    provider: str | None = None
    auth: str | None = None
    account: str | None = None
    account_set: bool = False
    model_id: str | None = None


def _parse_model_args(tokens: list[str]) -> _ParsedModelArgs | str:
    """Parse ``/model`` tokens; return explicit fields, or an error string."""
    provider: str | None = None
    auth: str | None = None
    account: str | None = None
    account_set = False
    model_id: str | None = None
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in _FLAG_PROVIDER and i + 1 < len(tokens):
            provider = tokens[i + 1]
            i += 2
            continue
        if tok in _FLAG_AUTH and i + 1 < len(tokens):
            auth = tokens[i + 1]
            i += 2
            continue
        if tok in _FLAG_ACCOUNT and i + 1 < len(tokens):
            account = tokens[i + 1]
            account_set = True
            i += 2
            continue
        if "=" in tok and not tok.startswith("-"):
            key, value = tok.split("=", 1)
            if key not in _KV_KEYS:
                return f"[/model] unknown key: {key}"
            if key == "provider":
                provider = value
            elif key == "auth":
                auth = value
            elif key == "account":
                account = value
                account_set = True
            else:
                model_id = value
            i += 1
            continue
        if not tok.startswith("-"):
            model_id = tok
            i += 1
            continue
        return f"[/model] unknown flag: {tok}"
    if provider is None and auth is None and not account_set and model_id is None:
        return (
            "[/model] usage: /model [provider=P] [auth=A] [account=ACCT]"
            " [model=MODEL_ID]   (or --provider/--auth/--account flags,"
            " or a bare model_id, with option tags like"
            " claude-opus-4-8+1m+fast)"
        )
    return _ParsedModelArgs(
        provider=provider,
        auth=auth,
        account=account,
        account_set=account_set,
        model_id=model_id,
    )
