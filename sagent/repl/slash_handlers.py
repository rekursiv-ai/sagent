"""Pure REPL adapters dispatched by slash commands.

These are the pure REPL adapters slash commands dispatch to, kept below
``input_pane`` so the input pane can import them directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import asyncio
import dataclasses
import shlex
import time

from sagent.agent.background import BackgroundTaskEntry
from sagent.agent.state import agent_registry
from sagent.providers.providers import infer_provider
from sagent.repl.slash import Controllable
from sagent.thinking import (
    THINKING_COMMANDS,
    apply_thinking_command,
    describe_thinking,
    thinking_offered,
)
from sagent.types.capability import ThinkingEffort


if TYPE_CHECKING:
    from sagent.agent.agent import Agent
    from sagent.repl.render import Printer


def do_switch_model(
    agent: Controllable,
    args: str,
    printer: Printer | None,
) -> None:
    """Render a ``/model`` slash command against :meth:`Agent.change_model`.

    Pure REPL adapter: parses the slash syntax, delegates the swap to the
    Agent API, prints the resulting label or error.

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
    old_id = agent.model.tagged_model_id
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


def do_switch_thinking(
    agent: Controllable,
    command: str,
    printer: Printer | None,
) -> None:
    """Render a ``/thinking`` slash command against the model's settings.

    ``show`` / ``hide`` land on ``printer.show_thinking`` -- they change
    what reaches THIS screen and nothing on the wire; the budget and output
    words land on the model's settings, which validate them.

    Args:
      agent: Agent whose model carries the settings.
      command: One word from ``THINKING_COMMANDS``, or ``""`` for status.
      printer: Optional sink for status messages; also owns the display
          flag. ``None`` (headless) has no screen, so nothing to toggle.

    """
    settings = agent.model.settings
    shown = printer.show_thinking if printer is not None else False
    if not command:
        valid = ", ".join(_reachable_thinking_words(agent))
        current = describe_thinking(settings, show=shown)
        _write(printer, f"[/thinking] {current}\n[/thinking] options: {valid}")
        return
    try:
        shown = apply_thinking_command(command, settings, show=shown)
    except ValueError as exc:
        options = ", ".join(_reachable_thinking_words(agent))
        _write(printer, f"[/thinking] {exc}; options: {options}")
        return
    if printer is not None:
        printer.show_thinking = shown
    _write(printer, f"[/thinking] {describe_thinking(settings, show=shown)}")


def do_switch_effort(agent: Controllable, value: str, printer: Printer | None) -> None:
    """Render an ``/effort`` slash command against the model's settings.

    Bare ``/effort`` (empty ``value``) prints the current effort plus the
    model's valid options. ``off`` / ``unset`` are aliases for ``none``,
    the axis's own unset value. Invalid values error with the option
    list -- a rejected request, never a silent no-op.

    Args:
      agent: Agent whose model carries the settings.
      value: Effort value, ``""`` for status, or a clear alias.
      printer: Optional sink for status messages.

    """
    settings = agent.model.settings
    options = ", ".join(sorted(agent.model.capability.thinking_effort))
    if not value:
        _write(
            printer,
            f"[/effort] {settings.thinking_effort}\n[/effort] options: {options}",
        )
        return
    try:
        settings.thinking_effort = cast(
            ThinkingEffort,
            "none" if value in ("off", "unset") else value,
        )
    except ValueError:
        _write(printer, f"[/effort] {value!r} is not one of: {options}")
        return
    _write(printer, f"[/effort] {settings.thinking_effort}")


async def do_login(agent: Controllable, printer: Printer | None) -> None:
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
                f"{phase:<10s}  {now - job.started:.0f}s",
            )
    header = (
        f"sagent: {len(agent_registry)} agent(s), "
        f"{total_fg} foreground, {total_bg} background"
    )
    if lines:
        return header + "\n" + "\n".join(lines)
    return header


# ``"errored"`` distinguishes crashes from graceful ``"completed"``; parallels
# :func:`_subagent_phase`'s same distinction so both bg-row families surface failures
# the same way.
def _generic_job_phase(job: BackgroundTaskEntry, now: float) -> str:
    """Phase label for non-persistent-subagent bg jobs."""
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


# Reads child runtime state directly -- safe because asyncio is single-threaded and
# ``format_tasks`` contains no ``await``.
def _subagent_phase(job: BackgroundTaskEntry) -> str:
    """Return a lifecycle label for a persistent-subagent bg-job row."""
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


# Slash-command output (``/model``, ``/thinking``, ``/login``, ``/tasks``) renders as
# machinery, not user text -- dim, no user bar -- so the operator can tell at a glance
# which lines are REPL infrastructure vs. agent dialogue.
def _write(printer: Printer | None, line: str) -> None:
    """Forward ``line`` to ``printer.write_slash_block`` when a printer is wired."""
    if printer is not None:
        printer.write_slash_block(line)


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
        if tok in ("--provider", "-p") and i + 1 < len(tokens):
            provider = tokens[i + 1]
            i += 2
            continue
        if tok in ("--auth", "-a") and i + 1 < len(tokens):
            auth = tokens[i + 1]
            i += 2
            continue
        if tok == "--account" and i + 1 < len(tokens):
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


# Each word is checked by applying it, because a word names one axis and inherits the
# rest -- ``redact`` is reachable only when the model can both budget the reasoning and
# withhold its body.
def _reachable_thinking_words(agent: Controllable) -> tuple[str, ...]:
    """Return the ``/thinking`` words this model can actually honor."""
    settings = agent.model.settings
    return tuple(word for word in THINKING_COMMANDS if thinking_offered(word, settings))
