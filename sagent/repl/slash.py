"""Slash-command parsing for the REPL.

The input pump (active and idle paths) parses each typed line into a
:class:`SlashAction` -- a tagged union of typed actions -- and dispatches
it directly against the agent's public API. No descriptor round-tripping
through the inbox.
"""

from __future__ import annotations

import dataclasses

from sagent.thinking import THINKING_COMMANDS


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Quit:
    """User typed ``/quit`` or ``/exit`` (or pressed Ctrl+D); end the REPL."""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Halt:
    """User typed ``/halt [<label>]`` (or Ctrl+C); halt the round loop.

    Cancels the in-flight model call, expunges any zombie response,
    requeues drained items, and arms ``block_until_user`` on the inbox.
    Pre-existing tool tasks keep running (use :class:`Kill` to stop them).
    """

    target: str = ""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Kill:
    """User typed ``/kill <qid|all>``; cancel one or all outstanding tool tasks."""

    target: str  # queue-id or ``"all"``


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Clear:
    """User typed ``/clear``; wipe context."""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Compact:
    """User typed ``/compact [hints]``; preempt + run compaction."""

    args: str = ""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Recompact:
    """User typed ``/recompact [hints]``; alias for ``/compact [hints]``."""

    args: str = ""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ModelSwitch:
    """User typed ``/model [args]``; reconfigure provider/model/auth."""

    args: str = ""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Thinking:
    """User typed ``/thinking <state|partial>``; update thinking state."""

    command: str


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Tool:
    """User typed ``/tool NAME.key=value``; reconfigure a live tool."""

    spec: str


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Effort:
    """User typed ``/effort [value]``; bare shows status, value sets it."""

    value: str


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
    """User typed plain (non-slash) input; dispatch as a preempting ``UserMessage``."""

    content: str


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Defer:
    """User typed ``/defer <text>`` (or pressed Tab on a non-empty buffer);
    dispatch as a non-preempting ``UserDeferredMessage`` that drains at
    ``AgentIdle``.

    Lets the user inject content that should be processed *after* the
    agent's current round chain completes, without preempting in-flight
    work. Headless callers use the ``/defer`` form (no Tab key); TUI
    users get the Tab gesture.
    """

    content: str


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Send:
    """User typed ``/send <target> <message>``; route to subagent(s).

    Plain ``message`` text becomes a ``UserMessage`` on the target's
    inbox. A leading-slash ``message`` (``/halt``, ``/quit``, ``/clear``,
    ``/compact``, ``/kill``, ``/model``, ``/thinking``) is parsed and
    dispatched as a control action against the target through
    ``_dispatch_target_control`` instead.
    """

    target: str
    content: str


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Unknown:
    """User typed an unrecognized ``/foo``; surface as an error."""

    text: str


type SlashAction = (
    Quit
    | Halt
    | Kill
    | Clear
    | Compact
    | Recompact
    | ModelSwitch
    | Thinking
    | Tool
    | Effort
    | Login
    | Help
    | Tasks
    | Text
    | Defer
    | Send
    | Unknown
)

# Quit phrases recognized by both REPL paths.
QUIT_WORDS: frozenset[str] = frozenset({"/quit", "/exit"})


def parse_slash(line: str) -> SlashAction | None:
    """Translate a typed line into a :class:`SlashAction`.

    Quit verbs (``QUIT_WORDS``) match the *whole* stripped line; ``/quit
    foo`` falls through to :class:`Unknown` rather than quitting,
    because trailing tokens almost always reflect typo-paste at exit
    time and silently quitting on those would surprise the user.

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
    arg = _arg_after("/thinking", stripped)
    if arg is not None:
        if not arg or arg in THINKING_COMMANDS:
            return Thinking(command=arg)
        return Unknown(
            text="/thinking requires one of: " + ", ".join(THINKING_COMMANDS)
        )
    arg = _arg_after("/effort", stripped)
    if arg is not None:
        return Effort(value=arg)
    arg = _arg_after("/tool", stripped)
    if arg is not None:
        if not arg:
            return Unknown(text="/tool requires NAME.key=value")
        return Tool(spec=arg)
    arg = _arg_after("/halt", stripped)
    if arg is not None:
        return Halt(target=arg)
    arg = _arg_after("/kill", stripped)
    if arg is not None:
        if not arg:
            return Unknown(text="/kill requires <qid> or 'all'")
        return Kill(target=arg)
    arg = _arg_after("/defer", stripped)
    if arg is not None:
        if not arg:
            return Unknown(text="/defer requires <text>")
        return Defer(content=arg)
    arg = _arg_after("/send", stripped)
    if arg is not None:
        target, sep, content = arg.partition(" ")
        if not sep or not target or not content.strip():
            return Unknown(text="/send requires <target> <message>")
        return Send(target=target, content=content.strip())
    if stripped.startswith("/"):
        cmd = stripped.split(maxsplit=1)[0]
        # Public list of supported commands; drives the unknown-command help line.
        supported = (
            "/help /clear /compact /recompact /model /provider /thinking /effort"
            " /tool /login /tasks /halt /kill /defer /send /quit /exit"
        )
        return Unknown(text=f"unknown command: {cmd}. Supported: {supported}")
    return Text(content=stripped)


def _arg_after(prefix: str, stripped: str) -> str | None:
    """Return the argument after ``prefix``, or ``None`` if it doesn't match."""
    if stripped == prefix:
        return ""
    if stripped.startswith(prefix + " "):
        return stripped[len(prefix) + 1 :].strip()
    return None
