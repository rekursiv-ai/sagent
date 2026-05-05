"""REPL slash-command handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dataclasses
import shlex

from rich.text import Text

from sagent.providers import build_provider, infer_provider


if TYPE_CHECKING:
    from rich.console import Console

    from sagent.agent import Agent


def handle_slash_command(agent: Agent, console: Console, stripped: str) -> bool:
    """Handle an idle REPL slash command.

    Returns True when ``stripped`` was a recognized slash command.
    REPL-local commands are handled here directly. Context-affecting commands
    are enqueued and interpreted by ``Agent._drain_inbox`` so there is one
    source of truth for conversation-history mutation.
    """
    if stripped.startswith("/model"):
        handle_slash_model(agent, console, stripped[6:])
        return True
    if stripped == "/clear" or stripped.startswith("/clear "):
        handle_slash_clear(agent, console, stripped[6:])
        return True
    if stripped == "/login":
        handle_slash_login(agent, console)
        return True
    return False


def handle_slash_model(agent: Agent, console: Console, args_str: str) -> bool:
    """Handle `/model` -- show or swap the agent's model without an LLM call.

    Returns True if the command was handled (even on error).
    """
    try:
        tokens = shlex.split(args_str)
    except ValueError as e:
        console.print(Text(f"[/model] parse error: {e}", style="red"))
        return True
    spec = agent.model_spec
    if spec is None:
        console.print(Text("[/model] agent has no model spec", style="red"))
        return True
    prov_name = spec.provider
    auth = spec.auth
    account = spec.account
    model_id: str | None = None
    if not tokens:
        console.print(
            Text(
                (
                    "[/model] "
                    f"provider={spec.provider} "
                    f"auth={spec.auth} "
                    f"model={spec.model_id} "
                    f"account={spec.account or 'default'}"
                ),
                style="dim",
            )
        )
        return True
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("--provider", "-p") and i + 1 < len(tokens):
            prov_name = tokens[i + 1]
            i += 2
        elif tok in ("--auth", "-a") and i + 1 < len(tokens):
            auth = tokens[i + 1]
            i += 2
        elif tok == "--account" and i + 1 < len(tokens):
            account = tokens[i + 1]
            i += 2
        elif not tok.startswith("-"):
            model_id = tok
            i += 1
        else:
            console.print(Text(f"[/model] unknown flag: {tok}", style="red"))
            return True
    if not model_id and prov_name == spec.provider:
        console.print(
            Text(
                "[/model] usage: /model [--provider P] [--auth A] [model_id]",
                style="dim",
            )
        )
        return True
    if model_id and prov_name == spec.provider:
        inferred = infer_provider(model_id, prov_name)
        if inferred is not None:
            prov_name, auth = inferred
    try:
        prov = build_provider(prov_name, auth, account=account)
        new_model = prov.model(model_id)
    except (AttributeError, RuntimeError, ValueError) as exc:
        console.print(Text(f"[/model] {exc}", style="red"))
        return True
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
    label = f"{old_id} → {new_model.model_id}"
    if prov_name != spec.provider:
        label = f"{spec.provider}/{old_id} → {prov_name}/{new_model.model_id}"
    console.print(Text(f"[/model] {label}", style="green"))
    return True


def handle_slash_login(agent: Agent, console: Console) -> None:
    """Handle ``/login`` -- re-authenticate the current provider."""
    spec = agent.model_spec
    if spec is None:
        console.print(Text("[/login] agent has no model spec", style="red"))
        return
    prov_name = spec.provider
    try:
        prov_cls = getattr(
            __import__(
                "sagent.providers",
                fromlist=[prov_name],
            ),
            prov_name,
        )
    except AttributeError:
        console.print(Text(f"[/login] unknown provider {prov_name!r}", style="red"))
        return
    login_fn = getattr(prov_cls, "login", None)
    if login_fn is None:
        console.print(Text(f"[/login] {prov_name} has no login method", style="red"))
        return
    try:
        login_fn()
        console.print(Text(f"[/login] {prov_name} re-authenticated", style="green"))
    except (RuntimeError, OSError, ValueError, TimeoutError) as exc:
        console.print(Text(f"[/login] {exc}", style="red"))


def handle_slash_clear(agent: Agent, console: Console, args_str: str) -> bool:
    """Queue ``/clear`` for the Agent inbox drain."""
    reason = args_str.strip()
    command = "/clear" if not reason else f"/clear {reason}"
    agent.inbox.put_left(command)
    note = f" ({reason})" if reason else ""
    console.print(
        Text(
            f"[/clear] conversation clear queued{note}",
            style="green",
        )
    )
    return True
