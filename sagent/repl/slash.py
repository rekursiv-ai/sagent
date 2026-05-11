"""Slash-command parsing for the v3 REPL.

The input pump (active and idle paths) parses each typed line into a
:class:`SlashAction` -- a tagged union of typed actions -- and dispatches
it directly against the agent's public API. No descriptor round-tripping
through the inbox.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Quit:
    """User typed ``/quit`` (or pressed Ctrl+D); end the REPL."""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Abort:
    """User typed ``/abort [<label>]``; cancel step + drain queue."""

    target: str = ""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class AbortAll:
    """User typed ``/abort all``; cancel + drain + kill visible bg jobs."""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Break:
    """User typed ``/break [<label>]``; cancel current step only."""

    target: str = ""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class BreakAll:
    """User typed ``/break all``; cancel current step on every agent."""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Clear:
    """User typed ``/clear``; wipe context."""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Compact:
    """User typed ``/compact [hints]``; preempt + run compaction."""

    args: str = ""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Recompact:
    """User typed ``/recompact [hints]``; reload + re-run last compaction."""

    args: str = ""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ModelSwitch:
    """User typed ``/model [args]``; reconfigure provider/model/auth."""

    args: str = ""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Login:
    """User typed ``/login``; re-auth current provider."""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Help:
    """User typed ``/help``; show command reference."""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Tasks:
    """User typed ``/tasks``; list running work across registered agents."""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Text:
    """User typed plain (non-slash) input; queue as a user message."""

    content: str


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Unknown:
    """User typed an unrecognized ``/foo``; surface as an error."""

    text: str


type SlashAction = (
    Quit
    | Abort
    | AbortAll
    | Break
    | BreakAll
    | Clear
    | Compact
    | Recompact
    | ModelSwitch
    | Login
    | Help
    | Tasks
    | Text
    | Unknown
)


# Quit phrases recognized by both REPL paths.
QUIT_WORDS: frozenset[str] = frozenset({"/quit"})

# Public list of supported commands; drives the unknown-command help line.
_SUPPORTED = (
    "/help /clear /compact /recompact /model /provider /login /tasks "
    "/break /abort /quit"
)


def parse_slash(line: str) -> SlashAction | None:
    """Translate a typed line into a :class:`SlashAction`.

    Args:
      line: Raw input line (may have trailing whitespace).

    Returns:
      action: One of the :class:`SlashAction` variants, or ``None`` for
          empty / whitespace-only input (the caller should ignore).

    """
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.lower() in QUIT_WORDS:
        return Quit()
    if stripped == "/help":
        return Help()
    if stripped == "/tasks":
        return Tasks()
    if stripped == "/clear":
        return Clear()
    if stripped == "/login":
        return Login()
    arg = _arg_after("/compact", stripped)
    if arg is not None:
        return Compact(args=arg)
    arg = _arg_after("/recompact", stripped)
    if arg is not None:
        return Recompact(args=arg)
    arg = _arg_after("/model", stripped)
    if arg is not None:
        return ModelSwitch(args=arg)
    arg = _arg_after("/provider", stripped)
    if arg is not None:
        return ModelSwitch(args=f"--provider {arg}" if arg else "")
    arg = _arg_after("/break", stripped)
    if arg is not None:
        if arg == "all":
            return BreakAll()
        return Break(target=arg)
    arg = _arg_after("/abort", stripped)
    if arg is not None:
        if arg == "all":
            return AbortAll()
        return Abort(target=arg)
    if stripped.startswith("/"):
        cmd = stripped.split(maxsplit=1)[0]
        return Unknown(text=f"unknown command: {cmd}. Supported: {_SUPPORTED}")
    return Text(content=stripped)


def _arg_after(prefix: str, stripped: str) -> str | None:
    """Return the argument after ``prefix``, or ``None`` if it doesn't match."""
    if stripped == prefix:
        return ""
    if stripped.startswith(prefix + " "):
        return stripped[len(prefix) + 1 :].strip()
    return None
