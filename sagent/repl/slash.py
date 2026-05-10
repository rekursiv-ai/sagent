"""Slash-command parsing shared by the active and idle REPL paths.

Single source of truth for ``/clear``, ``/compact``, ``/recompact``,
``/model``, ``/provider``, ``/abort``, ``/login``, ``/quit``. Both
``keybindings._kb_submit`` (active path: a model call or tool batch is
running) and the REPL input pump (idle path: prompt has returned)
call :func:`parse_slash` and then dispatch the resulting
:class:`SlashAction` against ``agent.inbox``.

Keeping a single table avoids the v1 wart where the two paths drifted
on which commands they accepted (``/login`` was idle-only, ``/abort``
echoed differently, etc.).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import dataclasses

from sagent.custom_types import TextMessage
from sagent.lib.descriptors import QUIT_SENTINEL, TextDescriptor


if TYPE_CHECKING:
    from sagent.agent.agent import Agent


@dataclasses.dataclass(kw_only=True, frozen=True, slots=True)
class SlashAction:
    """One inbox message to post in response to a typed line.

    Attributes:
      descriptor: Inbox descriptor to post.
      content: Message content (typically the post-prefix arg string).
      urgent: True to ``put_left`` (preempt queued messages).
      echo: Optional confirmation line for the printer.
      quit: True if this action terminates the REPL loop.

    """

    descriptor: TextDescriptor
    content: str = ""
    urgent: bool = False
    echo: str | None = None
    quit: bool = False


# Quit phrases recognized by both REPL paths.
QUIT_WORDS: frozenset[str] = frozenset({"/quit"})


def parse_slash(line: str) -> SlashAction | None:
    """Translate a typed line into an inbox action, or ``None`` for plain text.

    Empty / whitespace-only input returns ``None`` (caller should
    ignore). Any unknown ``/foo`` returns a ``text/x-error`` action.
    Non-slash input returns a ``text/x-user-message`` action so a
    single dispatch site handles both branches.
    """
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.lower() in QUIT_WORDS:
        return SlashAction(descriptor=QUIT_SENTINEL, quit=True)
    if stripped == "/help":
        return SlashAction(descriptor="text/x-help-request")
    if stripped == "/tasks":
        return SlashAction(descriptor="text/x-tasks-request")
    # ``/clear`` is destructive and accepts no argument: any extra
    # text means the user is talking *about* the command, not invoking
    # it. (See bugs/post-mortem on the accidental-/clear incident.)
    if stripped == "/clear":
        return SlashAction(
            descriptor="text/x-clear-request",
            urgent=True,
            echo="[/clear] history cleared",
        )
    arg = _arg_after("/compact", stripped)
    if arg is not None:
        note = f" ({arg})" if arg else ""
        return SlashAction(
            descriptor="text/x-compact-request",
            content=arg,
            echo=f"[/compact] queued{note}",
        )
    arg = _arg_after("/recompact", stripped)
    if arg is not None:
        note = f" ({arg})" if arg else ""
        return SlashAction(
            descriptor="text/x-recompact-request",
            content=arg,
            echo=f"[/recompact] queued{note}",
        )
    arg = _arg_after("/model", stripped)
    if arg is not None:
        # ModelSwitchHandler echoes its own status; no echo here.
        return SlashAction(
            descriptor="text/x-model-switch-request",
            content=arg,
        )
    arg = _arg_after("/provider", stripped)
    if arg is not None:
        # Sugar for ``/model --provider PROV [model_id]``.
        return SlashAction(
            descriptor="text/x-model-switch-request",
            content=f"--provider {arg}" if arg else "",
        )
    # /break and /abort take an optional argument:
    #   no arg   - this agent only
    #   <label>  - that agent (cross-agent reach via inbox)
    #   all      - every registered agent
    # /break cancels the in-flight step (Ctrl+Z analog -- queue
    # survives so typed-ahead commands still run).
    # /abort cancels in-flight + drains the queue. With "all" it
    # additionally cancels visible background tasks (Ctrl+C analog).
    arg = _arg_after("/break", stripped)
    if arg is not None:
        scope = f" {arg}" if arg else ""
        return SlashAction(
            descriptor="text/x-break",
            content=arg,
            urgent=True,
            echo=f"[/break{scope}] cancelling current step",
        )
    arg = _arg_after("/abort", stripped)
    if arg is not None:
        scope = f" {arg}" if arg else ""
        return SlashAction(
            descriptor="text/x-abort",
            content=arg,
            urgent=True,
            echo=f"[/abort{scope}] cancelling step + queue",
        )
    if stripped == "/login":
        return SlashAction(descriptor="text/x-login-request")
    if stripped.startswith("/"):
        cmd = stripped.split(maxsplit=1)[0]
        return SlashAction(
            descriptor="text/x-error",
            content=(
                f"unknown command: {cmd}. Supported: "
                "/help /clear /compact /recompact /model /provider"
                " /login /tasks /break /abort /quit"
            ),
        )
    # Strip user input so the active and idle REPL paths post identical
    # inbox content for the same typed line. Without this both call sites
    # have to re-decide a normalization policy and they disagree.
    return SlashAction(descriptor="text/x-user-message", content=stripped)


def dispatch(agent: Agent, action: SlashAction) -> None:
    """Post ``action`` to ``agent.inbox`` according to its urgent flag."""
    msg = TextMessage(action.content, action.descriptor)
    if action.urgent:
        _ = agent.inbox.put_left(msg)
    else:
        _ = agent.inbox.put(msg)


def supported_descriptors() -> Iterable[str]:
    """Return the set of inbox descriptors the parser may emit.

    Used by handler registries that want to assert every emitted
    descriptor has a subscriber.
    """
    return {
        "text/x-clear-request",
        "text/x-help-request",
        "text/x-tasks-request",
        "text/x-compact-request",
        "text/x-recompact-request",
        "text/x-model-switch-request",
        "text/x-break",
        "text/x-abort",
        "text/x-login-request",
        "text/x-error",
        "text/x-user-message",
        "text/x-quit",
    }


def _arg_after(prefix: str, stripped: str) -> str | None:
    """Return the argument after ``prefix``, or ``None`` if it doesn't match.

    Matches the bare prefix (``/clear``) and the prefix with an arg
    (``/clear xyz``); rejects unrelated commands sharing a prefix
    (``/clearxyz``).
    """
    if stripped == prefix:
        return ""
    if stripped.startswith(prefix + " "):
        return stripped[len(prefix) + 1 :].strip()
    return None
