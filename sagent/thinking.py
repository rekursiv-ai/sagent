"""Canonical thinking-state parsing and expansion."""

from __future__ import annotations

from typing import Literal, cast


type ThinkingState = Literal[
    "adaptive-show",
    "adaptive-hide",
    "on-show",
    "on-hide",
    "off-hide",
    "redact-hide",
]
type ThinkingCommand = Literal[
    "adaptive-show",
    "adaptive-hide",
    "on-show",
    "on-hide",
    "off-hide",
    "redact-hide",
    "adaptive",
    "on",
    "off",
    "redact",
    "show",
    "hide",
]

THINKING_STATES: tuple[ThinkingState, ...] = (
    "adaptive-show",
    "adaptive-hide",
    "on-show",
    "on-hide",
    "off-hide",
    "redact-hide",
)
THINKING_COMMANDS: tuple[ThinkingCommand, ...] = (
    *THINKING_STATES,
    "adaptive",
    "on",
    "off",
    "redact",
    "show",
    "hide",
)


def resolve_thinking_command(
    command: str,
    current: ThinkingState | None = None,
) -> ThinkingState:
    """Resolve a full or partial thinking command to a canonical state.

    Args:
      command: Full state or partial command.
      current: Current state for live partial display changes. ``None`` uses
        startup defaults for CLI construction.

    Returns:
      state: Canonical thinking state.

    Raises:
      ValueError: If the command is unknown or invalid for the current state.

    """
    if command in THINKING_STATES:
        return cast(ThinkingState, command)  # pyright: ignore[reportUnnecessaryCast] -- ty does not narrow tuple membership to Literal.
    if command not in THINKING_COMMANDS:
        valid = ", ".join(THINKING_COMMANDS)
        raise ValueError(f"thinking must be one of: {valid}")
    typed_command = cast(ThinkingCommand, command)  # pyright: ignore[reportUnnecessaryCast] -- ty does not narrow tuple membership to Literal.
    if current is None:
        return _startup_state(typed_command)
    return _live_state(typed_command, current)


def request_thinking(state: ThinkingState) -> str | None:
    """Return the provider request thinking mode for ``state``."""
    if state.startswith("adaptive-") or state == "redact-hide":
        return "adaptive"
    if state.startswith("on-"):
        return "enabled"
    return None


def should_show_thinking(state: ThinkingState) -> bool:
    """Return whether readable thinking should render locally."""
    return state.endswith("-show")


def should_redact_thinking(state: ThinkingState) -> bool:
    """Return whether provider-side thinking redaction is requested."""
    return state == "redact-hide"


def _startup_state(command: ThinkingCommand) -> ThinkingState:
    """Expand partial startup commands without a live current state."""
    match command:
        case "adaptive":
            return "adaptive-hide"
        case "on":
            return "on-hide"
        case "off":
            return "off-hide"
        case "redact":
            return "redact-hide"
        case "show":
            return "adaptive-show"
        case "hide":
            return "adaptive-hide"
        case _:
            return command


def _live_state(command: ThinkingCommand, current: ThinkingState) -> ThinkingState:
    """Expand partial live commands against ``current``."""
    suffix = "show" if current.endswith("-show") else "hide"
    match command:
        case "adaptive":
            return cast(ThinkingState, f"adaptive-{suffix}")
        case "on":
            return cast(ThinkingState, f"on-{suffix}")
        case "off":
            return "off-hide"
        case "redact":
            return "redact-hide"
        case "hide":
            if current == "adaptive-show":
                return "adaptive-hide"
            if current == "on-show":
                return "on-hide"
            return current
        case "show":
            if current == "adaptive-hide":
                return "adaptive-show"
            if current == "on-hide":
                return "on-show"
            raise ValueError(f"cannot show thinking from state {current!r}")
        case _:
            return command
