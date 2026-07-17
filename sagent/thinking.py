"""Canonical thinking-state parsing and expansion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, assert_never, cast


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


@dataclass(frozen=True, slots=True, kw_only=True)
class ThinkingCapability:
    """A provider/model's thinking capability, orthogonal bits.

    The valid ``ThinkingState`` set is derived from these bits via
    :func:`valid_thinking_states`; providers declare the capability
    rather than enumerating states, so the state set can never drift
    from the wire behavior.

    Attributes:
      supports_thinking: Whether the model can reason at all. When
          ``False``, the only valid state is ``off-hide``.
      readable_text: Whether requested thinking returns readable text to
          the client. ``False`` for models that return a signed but empty
          thinking block -- the model reasons, but plaintext is never
          delivered (measured: opus-4-8) -- making every ``-show`` state
          unsatisfiable and thus invalid.
      supports_adaptive_mode: Whether the model accepts the ``adaptive``
          thinking request mode (the ``adaptive-*`` states, and
          ``redact-hide`` which also requests ``adaptive``). ``False`` for
          the 4-5 generation, which rejects ``thinking.type=adaptive``
          with HTTP 400 and requires ``enabled``.
      supports_enabled_mode: Whether the model accepts the ``enabled``
          thinking request mode (the ``on-*`` states). ``False`` for
          adaptive-only models that reject ``thinking.type=enabled`` with
          HTTP 400 (measured: opus-4-8, opus-4-7), making ``on-show`` /
          ``on-hide`` invalid.
      supports_redaction: Whether the provider exposes an explicit
          redacted-thinking mode (the ``redact-hide`` state). Only the
          Anthropic API transport advertises this; it also requires
          ``supports_adaptive_mode`` since redaction rides ``adaptive``.

    """

    supports_thinking: bool
    readable_text: bool = True
    supports_adaptive_mode: bool = True
    supports_enabled_mode: bool = True
    supports_redaction: bool = False


def valid_thinking_states(capability: ThinkingCapability) -> tuple[ThinkingState, ...]:
    """Return the ``ThinkingState`` values valid for ``capability``.

    ``off-hide`` is always valid (thinking off is universal). The
    ``adaptive-*`` states require ``adaptive`` mode; the ``on-*`` states
    require ``enabled`` mode; ``redact-hide`` requires both server-side
    redaction and ``adaptive`` mode (redaction rides ``adaptive``). A
    ``-show`` state is valid only when the model returns readable text,
    since a state that can never render is a failed request, not a no-op.

    Args:
      capability: The provider/model thinking capability bits.

    Returns:
      states: Valid canonical states, in canonical order.

    """
    if not capability.supports_thinking:
        return ("off-hide",)
    states: list[ThinkingState] = []
    for state in THINKING_STATES:
        if state == "off-hide":
            states.append(state)
            continue
        if state == "redact-hide":
            if capability.supports_redaction and capability.supports_adaptive_mode:
                states.append(state)
            continue
        if state.startswith("adaptive-") and not capability.supports_adaptive_mode:
            continue
        if state.startswith("on-") and not capability.supports_enabled_mode:
            continue
        if state.endswith("-show") and not capability.readable_text:
            continue
        states.append(state)
    return tuple(states)


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
        return command
    if command not in THINKING_COMMANDS:
        valid = ", ".join(THINKING_COMMANDS)
        raise ValueError(f"thinking must be one of: {valid}")
    typed_command = command
    if current is None:
        return _startup_state(typed_command)
    return _live_state(typed_command, current)


def request_thinking(state: ThinkingState) -> str | None:
    """Return the provider request thinking mode for ``state``.

    ``redact-hide`` requests the provider's ``"adaptive"`` mode (despite
    the user-facing name suggesting "off"). The provider returns
    redacted thinking blocks under adaptive when policy triggers, and we
    suppress local rendering separately via ``should_show_thinking``.
    """
    if state.startswith("adaptive-") or state == "redact-hide":
        return "adaptive"
    if state.startswith("on-"):
        return "enabled"
    return None


def thinking_mode_supported(mode: str | None, valid_states: tuple[str, ...]) -> bool:
    """Return whether wire ``mode`` is reachable given ``valid_states``.

    ``mode`` is the provider-facing thinking value (``"adaptive"`` /
    ``"enabled"`` / ``None``) carried by ``Agent._thinking``, which the
    legacy ``Agent.thinking`` setter writes without a canonical state.
    A swap must clear it when the new model exposes no valid state mapping
    to it (else the next request sends a rejected mode and 400s).
    ``None`` (thinking off) is always supported.

    Args:
      mode: Wire thinking mode, or ``None`` for off.
      valid_states: The model's ``valid_thinking_states``.

    Returns:
      supported: True when ``mode`` is ``None`` or some valid state maps
          to it.

    """
    if mode is None:
        return True
    return any(request_thinking(cast(ThinkingState, s)) == mode for s in valid_states)


def should_show_thinking(state: ThinkingState) -> bool:
    """Return whether readable thinking should render locally."""
    return state.endswith("-show")


def should_redact_thinking(state: ThinkingState) -> bool:
    """Return whether provider-side thinking redaction is requested."""
    return state == "redact-hide"


def _startup_state(command: ThinkingCommand) -> ThinkingState:
    """Expand partial startup commands without a live current state.

    Full states pass through; partial commands map to canonical hides
    (or adaptive-show for the bare ``show`` alias). ``assert_never``
    keeps the match exhaustive as ``ThinkingCommand`` grows.
    """
    match command:
        case (
            "adaptive-show"
            | "adaptive-hide"
            | "on-show"
            | "on-hide"
            | "off-hide"
            | "redact-hide"
        ):
            return command
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
            assert_never(command)


def _live_state(command: ThinkingCommand, current: ThinkingState) -> ThinkingState:
    """Expand partial live commands against ``current``.

    ``show`` and ``hide`` are symmetric: each raises when ``current`` is
    a fixed-display state (``off-hide`` / ``redact-hide``) for which the
    requested transition is impossible.
    """
    suffix = "show" if current.endswith("-show") else "hide"
    match command:
        case (
            "adaptive-show"
            | "adaptive-hide"
            | "on-show"
            | "on-hide"
            | "off-hide"
            | "redact-hide"
        ):
            return command
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
            raise ValueError(f"cannot hide thinking from state {current!r}")
        case "show":
            if current == "adaptive-hide":
                return "adaptive-show"
            if current == "on-hide":
                return "on-show"
            raise ValueError(f"cannot show thinking from state {current!r}")
        case _:
            assert_never(command)
