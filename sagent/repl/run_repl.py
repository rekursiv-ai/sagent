"""``run_repl``: orchestrate an interactive REPL on top of ``Agent``.

Builds a prompt-toolkit session, attaches one render observer to the
agent's observer list, spawns the input pump as a hidden background
task, and calls ``agent.serve_forever()``. Returns when the user types
``/quit`` or sends EOF.

Important: the ``rich.Console`` is constructed INSIDE the
``patch_stdout`` context. ``patch_stdout`` swaps ``sys.stdout`` /
``sys.stderr`` for a proxy that routes writes above the prompt;
``rich.Console`` snapshots its file handle at construction. Building
the console outside the patch causes its writes to bypass the proxy.
"""

from __future__ import annotations

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
from sagent.providers import build_provider, infer_provider
from sagent.repl.console import ConsolePrinter
from sagent.repl.input import REPL_PUMP_KEY, spawn_repl_pump
from sagent.repl.keybindings import build_key_bindings
from sagent.repl.prompt import (
    PromptToolkitInputSource,
    dynamic_prompt,
)
from sagent.repl.render import make_render_observer
from sagent.repl.replay import replay_messages
from sagent.repl.toolbar import render_toolbar
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
    show_exception_stack: bool = True,
) -> None:
    """Drive ``agent`` interactively until the user types ``/quit``.

    Args:
      agent: The agent to drive.
      history: Path to the input-history file. ``None`` -> ``~/.sagent_history``.
      show_exception_stack: When True, render structured stack traces for
        error Messages that include them. Defaults to True.

    """
    history_path = history or _DEFAULT_HISTORY
    style = PTStyle.from_dict(
        {
            "bottom-toolbar": "fg:ansibrightblack noreverse bg:default",
            "queued": "fg:ansibrightblack",
            "prompt": "bold",
        },
    )
    with patch_stdout(raw=True):
        console = Console(stderr=True)
        session: PromptSession[str] = PromptSession(
            functools.partial(dynamic_prompt, agent),
            multiline=True,
            erase_when_done=True,
            history=FileHistory(str(history_path)),
            auto_suggest=AutoSuggestFromHistory(),
            bottom_toolbar=functools.partial(render_toolbar, agent),
            refresh_interval=0.2,
            key_bindings=build_key_bindings(agent),
            enable_open_in_editor=False,
            style=style,
        )
        printer = ConsolePrinter(console)
        agent.observers.append(
            make_render_observer(printer, show_exception_stack=show_exception_stack),
        )
        pump_task = spawn_repl_pump(
            agent,
            PromptToolkitInputSource(session, agent=agent, console=console),
            printer=printer,
        )
        replay_messages(agent, printer, show_exception_stack=show_exception_stack)
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
                await asyncio.gather(*bg_tasks, return_exceptions=True)
            _ = pump_task.cancel()
            try:
                with contextlib.suppress(asyncio.CancelledError):
                    await pump_task
            except Exception:
                logger.exception("REPL input pump raised during shutdown")
            _ = agent.background.pop(REPL_PUMP_KEY, None)
    if agent.session_dir is not None:
        sys.stderr.write(f"[session {agent.session_dir.name}]\n")


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
    parsed = _parse_model_args(
        tokens, spec.provider, spec.auth, spec.model_id, spec.account
    )
    if isinstance(parsed, str):
        _write(printer, parsed)
        return
    prov_name, auth, account, model_id = parsed
    if model_id and prov_name == spec.provider:
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
    agent.swap_model(
        new_model,
        spec=dataclasses.replace(
            spec,
            provider=prov_name,
            auth=auth,
            model_id=new_model.model_id,
            account=account,
        ),
    )
    if prov_name != spec.provider:
        label = f"{spec.provider}/{old_id} -> {prov_name}/{new_model.model_id}"
    else:
        label = f"{old_id} -> {new_model.model_id}"
    _write(printer, f"[/model] {label}")


def do_login(agent: Agent, printer: Printer | None) -> None:
    """Re-auth the agent's current provider via its ``login`` classmethod."""
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
    """Format running fg/bg work across every registered agent."""
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
    if printer is not None:
        printer.write_line(line)


_FLAG_PROVIDER = ("--provider", "-p")
_FLAG_AUTH = ("--auth", "-a")
_FLAG_ACCOUNT = ("--account",)
_KV_KEYS = frozenset({"provider", "auth", "account", "model", "model_id"})


def _parse_model_args(
    tokens: list[str],
    cur_provider: str,
    cur_auth: str,
    cur_model_id: str,
    cur_account: str | None,
) -> tuple[str, str, str | None, str] | str:
    """Parse ``/model`` arguments. Return (provider, auth, account, model) or error str."""
    prov_name = cur_provider
    auth = cur_auth
    account = cur_account
    model_id = cur_model_id
    nothing_supplied = True
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in _FLAG_PROVIDER and i + 1 < len(tokens):
            prov_name = tokens[i + 1]
            nothing_supplied = False
            i += 2
            continue
        if tok in _FLAG_AUTH and i + 1 < len(tokens):
            auth = tokens[i + 1]
            nothing_supplied = False
            i += 2
            continue
        if tok in _FLAG_ACCOUNT and i + 1 < len(tokens):
            account = tokens[i + 1]
            nothing_supplied = False
            i += 2
            continue
        if "=" in tok and not tok.startswith("-"):
            key, value = tok.split("=", 1)
            if key not in _KV_KEYS:
                return f"[/model] unknown key: {key}"
            if key == "provider":
                prov_name = value
            elif key == "auth":
                auth = value
            elif key == "account":
                account = None if value in ("", "default") else value
            else:
                model_id = value
            nothing_supplied = False
            i += 1
            continue
        if not tok.startswith("-"):
            model_id = tok
            nothing_supplied = False
            i += 1
            continue
        return f"[/model] unknown flag: {tok}"
    if nothing_supplied:
        return (
            "[/model] usage: /model [provider=P] [auth=A] [account=ACCT]"
            " [model=MODEL_ID]   (or --provider/--auth/--account flags,"
            " or a bare model_id)"
        )
    return prov_name, auth, account, model_id
