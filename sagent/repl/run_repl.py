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
import contextlib
import dataclasses
import functools
import inspect
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

from sagent.agent import runtime as agent_runtime
from sagent.providers import infer_provider
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
from sagent.tools.core import agent_registry
from sagent.types.exceptions import log_exception_or_warning
from sagent.types.runtime import (
    AssistantMessage,
    ModelIdle,
    RuntimeEvent,
    ToolResult,
    UserMessage,
)


if TYPE_CHECKING:
    from sagent.agent.agent import Agent
    from sagent.repl.render import Printer

logger = logging.getLogger(__name__)

_DEFAULT_HISTORY = Path.home() / ".sagent_history"


async def run_repl(
    agent: Agent,
    *,
    history: Path | None = None,
) -> None:
    """Drive ``agent`` interactively until the user types ``/quit``.

    Args:
      agent: The agent to drive.
      history: Path to the input-history file. ``None`` -> ``~/.sagent_history``.

    """
    history_path = history or _DEFAULT_HISTORY
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
        agent.runtime.observers.append(
            make_render_observer(printer, show_thinking=lambda: agent.show_thinking)
        )
        agent.runtime.observers.append(make_input_queue_committer(agent, queues))
        pump_task = spawn_repl_pump(
            agent,
            PromptToolkitInputSource(session, queues=queues, console=console),
            queues=queues,
            printer=printer,
        )
        replay_messages(agent, printer)
        _publish_startup_idle_if_settled(agent.runtime)
        if agent.status:
            printer.set_terminal_title(agent.status)
        elif agent.name:
            printer.set_terminal_title(agent.name)
        try:
            await agent.serve_forever()
        finally:
            agent.shutdown(force=True)
            bg_tasks = _background_tasks_for_repl_cancel(agent)
            for t in bg_tasks:
                _ = t.cancel()
            if bg_tasks:
                _ = await asyncio.gather(*bg_tasks, return_exceptions=True)
            _ = pump_task.cancel()
            try:
                with contextlib.suppress(asyncio.CancelledError):
                    await pump_task
            except Exception as exc:  # noqa: BLE001 -- pump shutdown catches any slash-handler exception; UserFacingError routed to warning, others to exception
                log_exception_or_warning(
                    logger, "REPL input pump raised during shutdown", exc
                )
            agent.cancel_background(REPL_PUMP_KEY)
    if agent.session_dir is not None:
        _ = sys.stderr.write(
            "Resume this session with:\n"
            f"sagent --resume {agent.session_dir.name[:8]}  # this exact session (any unique prefix works)\n"
            "sagent --continue         # most recent session in this dir\n"
            "sagent --resume           # interactive picker for this dir\n"
            "sagent --continue-all     # most recent session across all dirs\n"
            "sagent --resume-all       # interactive picker across all dirs\n"
        )


def _background_tasks_for_repl_cancel(agent: Agent) -> list[asyncio.Task[object]]:
    """Return unfinished REPL-owned background tasks safe to raw-cancel."""
    return [
        job.task
        for job in list(agent.background.values())
        if job.kind != "persistent_subagent" and not job.task.done()
    ]


def _publish_startup_idle_if_settled(runtime: agent_runtime.AgentRuntime) -> None:
    """Publish an initial idle edge when the REPL starts already settled."""
    if (
        runtime.model_call is None
        and runtime.compact_task is None
        and not runtime.cohort
        and not runtime.inbox.gate_armed
        and not _history_triggers_model_call(runtime)
    ):
        runtime.publish(ModelIdle())


def _history_triggers_model_call(runtime: agent_runtime.AgentRuntime) -> bool:
    """Return True when persisted history already needs a model turn."""
    messages = runtime.context().messages
    return bool(messages) and isinstance(messages[-1], (ToolResult, UserMessage))


def make_input_queue_committer(
    agent: Agent,
    queues: InputQueues,
) -> Callable[[RuntimeEvent], None]:
    """Observer that commits REPL-local queues at their lifecycle events."""
    previous_before_tool_spawn = agent.runtime.before_tool_spawn
    agent.runtime.before_tool_spawn = functools.partial(
        _before_tool_spawn,
        queues=queues,
        previous_before_tool_spawn=previous_before_tool_spawn,
    )
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
    return queues.pop_urgent_message()


def _commit_local_queues(
    event: RuntimeEvent,
    *,
    agent: Agent,
    queues: InputQueues,
) -> None:
    if isinstance(event, ModelIdle) and not queues.commit_urgent(agent):
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
    spec = agent.model_spec
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
    old_id = agent.model.model_id
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
    try:
        state = resolve_thinking_command(command, current)
    except ValueError as exc:
        _write(printer, f"[/thinking] {exc}")
        return
    if state != "off-hide" and not agent.model.supports_thinking:
        _write(
            printer,
            f"[/thinking] model {agent.model.model_id!r} does not support thinking",
        )
        return
    supports_redact = _provider_accepts_arg(agent, "redact_thinking")
    if state == "redact-hide" and not supports_redact:
        _write(
            printer, "[/thinking] current provider does not support redacted thinking"
        )
        return
    if agent.thinking_state == state and (
        supports_redact or "redact_thinking" not in agent.provider_args
    ):
        _write(printer, f"[/thinking] {state}")
        return
    old_state = agent.thinking_state
    old_thinking = agent.thinking
    old_show = agent.show_thinking
    old_redact = agent.provider_args.get("redact_thinking", None)
    agent.set_thinking_state(state)
    if not supports_redact:
        agent.clear_provider_arg("redact_thinking")
    if supports_redact and not _rebuild_current_model(agent, printer):
        _restore_thinking_state(agent, old_state, old_thinking, old_show)
        if old_redact is None:
            agent.clear_provider_arg("redact_thinking")
        else:
            agent.set_provider_arg("redact_thinking", old_redact)
        return
    _write(printer, f"[/thinking] {state}")


def _restore_thinking_state(
    agent: Agent,
    state: ThinkingState | None,
    thinking: str | None,
    show_thinking: bool,
) -> None:
    """Restore thinking fields after a failed provider rebuild."""
    if hasattr(agent, "_thinking_state"):
        agent._thinking_state = state  # noqa: SLF001 -- transactional rollback
        agent._thinking = thinking  # noqa: SLF001 -- transactional rollback
        agent._show_thinking = show_thinking  # noqa: SLF001 -- transactional rollback
        return
    object.__setattr__(agent, "thinking_state", state)
    object.__setattr__(agent, "thinking", thinking)
    object.__setattr__(agent, "show_thinking", show_thinking)


def _infer_thinking_state(agent: Agent) -> ThinkingState:
    """Infer a canonical current state from legacy thinking/display fields."""
    if agent.thinking is None:
        return "off-hide"
    if agent.thinking == "enabled":
        return "on-show" if agent.show_thinking else "on-hide"
    return "adaptive-show" if agent.show_thinking else "adaptive-hide"


def _provider_accepts_arg(agent: Agent, key: str) -> bool:
    """Return whether the current provider factory accepts ``key``."""
    spec = agent.model_spec
    if spec is None:
        return False
    providers_mod = sys.modules["sagent.providers"]
    cls = getattr(providers_mod, spec.provider, None)
    if cls is None:
        return False
    factory = getattr(cls, f"from_{spec.auth}", None)
    if factory is None:
        return False
    try:
        return key in inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return False


def _rebuild_current_model(agent: Agent, printer: Printer | None) -> bool:
    """Queue a rebuild of the current provider/model with stored provider args."""
    spec = agent.model_spec
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
    spec = agent.model_spec
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
        visible_bg = [
            j for j in getattr(other, "background", {}).values() if not j.hidden
        ]
        fg = 1 if getattr(other, "work", None) is not None else 0
        bg_n = len(visible_bg)
        total_fg += fg
        total_bg += bg_n
        tag = " (self)" if other is agent else ""
        lines.append(f"  {label}{tag:<8s}  fg={fg} bg={bg_n}")
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


def _write(printer: Printer | None, line: str) -> None:
    """Forward ``line`` to ``printer.write_line`` when a printer is wired."""
    if printer is not None:
        printer.write_line(line)


_FLAG_PROVIDER = ("--provider", "-p")
_FLAG_AUTH = ("--auth", "-a")
_FLAG_ACCOUNT = ("--account",)
_KV_KEYS = frozenset({"provider", "auth", "account", "model", "model_id"})


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _ParsedModelArgs:
    """Only the fields the user explicitly typed; rest stay ``None``.

    ``account_set`` disambiguates "user typed account=" (with empty or
    'default' → ``None``) from "user didn't mention account at all."

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
            " or a bare model_id)"
        )
    return _ParsedModelArgs(
        provider=provider,
        auth=auth,
        account=account,
        account_set=account_set,
        model_id=model_id,
    )
