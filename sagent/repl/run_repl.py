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

from sagent import providers
from sagent.agent.runtime import (
    AgentRuntime,
    ModelIdle,
    ModelSwitch,
    RuntimeEvent,
    UserMessage,
    UserQueuedMessage,
)
from sagent.custom_types import ModelSpec
from sagent.lib import last_models
from sagent.providers import build_provider, infer_provider
from sagent.repl.console_pane import ConsolePrinter
from sagent.repl.input_pane import (
    REPL_PUMP_KEY,
    PromptToolkitInputSource,
    render_input_pane,
    spawn_repl_pump,
)
from sagent.repl.keybindings import build_key_bindings
from sagent.repl.render import make_render_observer
from sagent.repl.replay import replay_messages
from sagent.repl.status_pane import render_status_pane
from sagent.tools.core import agent_registry


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
    queued_input: list[str] = []
    with patch_stdout(raw=True):
        console = Console(stderr=True)
        session: PromptSession[str] = PromptSession(
            functools.partial(render_input_pane, agent, queued_input),
            multiline=True,
            erase_when_done=True,
            history=FileHistory(str(history_path)),
            auto_suggest=AutoSuggestFromHistory(),
            bottom_toolbar=functools.partial(render_status_pane, agent),
            refresh_interval=0.2,
            key_bindings=build_key_bindings(agent, queued_input),
            enable_open_in_editor=False,
            style=style,
        )
        printer = ConsolePrinter(console)
        agent.runtime.observers.append(make_render_observer(printer))
        agent.runtime.observers.append(make_queued_input_clearer(queued_input))
        agent.runtime.observers.append(
            make_queued_input_committer(agent.runtime, queued_input)
        )
        pump_task = spawn_repl_pump(
            agent,
            PromptToolkitInputSource(
                session, queued_input=queued_input, console=console
            ),
            printer=printer,
        )
        replay_messages(agent, printer)
        if agent.status:
            printer.set_terminal_title(agent.status)
        elif agent.name:
            printer.set_terminal_title(agent.name)
        try:
            await agent.serve_forever()
        finally:
            agent.shutdown(force=True)
            bg_tasks = [
                job.task
                for job in list(agent.background.values())
                if not job.task.done()
            ]
            for t in bg_tasks:
                _ = t.cancel()
            if bg_tasks:
                _ = await asyncio.gather(*bg_tasks, return_exceptions=True)
            _ = pump_task.cancel()
            try:
                with contextlib.suppress(asyncio.CancelledError):
                    await pump_task
            except Exception:
                logger.exception("REPL input pump raised during shutdown")
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


def make_queued_input_clearer(
    queued_input: list[str],
) -> Callable[[RuntimeEvent], None]:
    """Observer that empties ``queued_input`` once the runtime accepts user input.

    ``queued_input`` is the REPL-local mirror of "lines the user typed
    while the agent was busy" -- it backs the dim preview in
    :func:`repl.input_pane.render_input_pane` and the Up-arrow edit-back in
    :func:`repl.keybindings._kb_up`. Entries are appended by
    ``_kb_submit`` but never removed except by Up or quit, so without
    this observer the dim preview shows a stale tail entry indefinitely
    after the runtime commits the user input to history.

    The runtime publishes a ``UserMessage`` event whenever it accepts
    user input -- one event per coalesced batch under the mid-stream
    buffer, one event per submission otherwise. In either shape, the
    event means everything currently in ``queued_input`` has been
    committed; a full clear is correct.

    Args:
      queued_input: REPL-local queued-text buffer to clear.

    Returns:
      observer: Callable suitable for ``agent.runtime.observers.append``.

    """

    def observer(event: RuntimeEvent) -> None:
        if isinstance(event, UserMessage):
            queued_input.clear()

    return observer


def make_queued_input_committer(
    runtime: AgentRuntime,
    queued_input: list[str],
) -> Callable[[RuntimeEvent], None]:
    r"""Observer that commits the staged ``queued_input`` on ``ModelIdle``.

    Under the staging model (per ``repl.keybindings``), text typed
    while the agent is busy accumulates in ``queued_input`` locally
    and is not dispatched to the runtime until the user explicitly
    commits (Enter on empty input) or the round chain settles. This
    observer handles the latter: when the runtime publishes
    ``ModelIdle`` -- the agent has finished its current round chain --
    the staged queue is pushed as a single ``UserQueuedMessage``
    joined by ``\\n\\n``. The runtime's own ``queued``-list drain
    then appends a ``UserMessage`` and the gate fires for the next
    round.

    Args:
      runtime: Runtime to push the ``UserQueuedMessage`` onto.
      queued_input: REPL-local staging buffer.

    Returns:
      observer: Callable suitable for ``runtime.observers.append``.

    """

    def observer(event: RuntimeEvent) -> None:
        if isinstance(event, ModelIdle) and queued_input:
            joined = "\n\n".join(queued_input)
            queued_input.clear()
            runtime.inbox.push_back(UserQueuedMessage(text=joined))

    return observer


def do_switch_model(
    agent: Agent,
    args: str,
    printer: Printer | None,
) -> None:
    """Apply a ``/model`` directive against ``agent.swap_model``.

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
    try:
        prov_name, auth, account, model_id = _resolve_model_target(parsed, spec)
    except AttributeError as exc:
        _write(printer, f"[/model] {exc}")
        return
    # Bare model_id on the SAME provider may imply a different provider
    # (e.g. ``/model gemini-3-pro`` while on Anthropic). Infer.
    if parsed.model_id and prov_name == spec.provider:
        inferred = infer_provider(model_id, prov_name)
        if inferred is not None:
            prov_name, auth = inferred
    try:
        provider = build_provider(prov_name, auth, account=account)
        new_model = provider.model(model_id)
    except (AttributeError, RuntimeError, ValueError) as exc:
        _write(printer, f"[/model] {exc}")
        return
    old_id = agent.model.model_id
    new_spec = dataclasses.replace(
        spec,
        provider=prov_name,
        auth=auth,
        model_id=new_model.model_id,
        account=account,
    )
    if prov_name != spec.provider:
        label = f"{spec.provider}/{old_id} -> {prov_name}/{new_model.model_id}"
    else:
        label = f"{old_id} -> {new_model.model_id}"
    # Queue the swap through the runtime inbox so it sequences with
    # any in-flight model call: the OLD model finishes its response
    # (and records its own cost) before the new one becomes active.
    agent.runtime.inbox.push_back(
        ModelSwitch(
            apply=lambda: agent.swap_model(new_model, spec=new_spec),
            label=label,
        ),
    )
    queued = " (queued)" if agent.work is not None else ""
    _write(printer, f"[/model] {label}{queued}")


def do_login(agent: Agent, printer: Printer | None) -> None:
    """Re-auth the agent's current provider via its ``login`` classmethod.

    Args:
      agent: Agent whose provider should be re-authenticated.
      printer: Optional sink for status messages.

    """
    spec = agent.model_spec
    if spec is None:
        _write(printer, "[/login] agent has no model spec")
        return
    prov_cls = getattr(providers, spec.provider, None)
    if prov_cls is None:
        _write(printer, f"[/login] unknown provider {spec.provider!r}")
        return
    login_fn = getattr(prov_cls, "login", None)
    if login_fn is None:
        _write(printer, f"[/login] {spec.provider} has no login method")
        return
    try:
        login_fn()
        _write(printer, f"[/login] {spec.provider} re-authenticated")
    except (RuntimeError, OSError, ValueError, TimeoutError) as exc:
        _write(printer, f"[/login] {exc}")


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
                f"    bg: {job.queue_id:<10s}  {job.tool_name:<16s}  "
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
                account = None if value in ("", "default") else value
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


def _resolve_model_target(
    parsed: _ParsedModelArgs,
    spec: ModelSpec,
) -> tuple[str, str, str | None, str]:
    """Layer parsed args onto ``spec``; fill model_id from cross-session memory.

    When the user switches provider but doesn't name a model, look up
    the last model used for that provider in
    ``~/.sagent/last-models.json``. Fall back to the provider class's
    ``DEFAULT_MODEL`` if this provider hasn't been used before.
    """
    prov_name = parsed.provider or spec.provider
    auth = parsed.auth or spec.auth
    account = parsed.account if parsed.account_set else spec.account
    if parsed.model_id is not None:
        model_id = parsed.model_id
    elif prov_name == spec.provider:
        model_id = spec.model_id
    else:
        model_id = last_models.get(prov_name) or _default_model_for(prov_name)
    return prov_name, auth, account, model_id


def _default_model_for(prov_name: str) -> str:
    """Return ``Provider.DEFAULT_MODEL`` for the named provider class."""
    cls = getattr(providers, prov_name, None)
    if cls is None:
        raise AttributeError(f"unknown provider: {prov_name!r}")
    default = getattr(cls, "DEFAULT_MODEL", None)
    if not isinstance(default, str) or not default:
        raise AttributeError(
            f"provider {prov_name!r} has no DEFAULT_MODEL",
        )
    return default
