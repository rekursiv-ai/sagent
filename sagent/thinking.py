"""Parse ``/thinking`` words into the settings they select.

A thinking request is three independent facts: how much budget the model
gets, whether the reasoning comes back readable, and whether this client
renders it. The first two are :class:`ModelSettings` axes owned by the
model; the third belongs to whatever is drawing the screen. A fused enum
spelling all three was a fourth vocabulary that had to be mapped back onto
the axes on every read, and the mappers disagreed -- the ``/thinking`` gate
compared an enum against ``capability.thinking_budget`` and so never
rejected anything.
"""

from __future__ import annotations

from typing import Final

from sagent.types.capability import (
    ModelSettings,
    ThinkingBudget,
    ThinkingOutput,
)


__all__ = [
    "THINKING_COMMANDS",
    "apply_thinking_command",
    "describe_thinking",
    "thinking_offered",
]


THINKING_COMMANDS: Final[tuple[str, ...]] = (
    "adaptive",
    "on",
    "off",
    "redact",
    "show",
    "hide",
)
"""Every word ``/thinking`` accepts, for completion and error text."""


def apply_thinking_command(
    command: str, settings: ModelSettings, *, show: bool
) -> bool:
    """Apply one word to ``settings``; return the new display flag.

    Each word names ONE fact and leaves the others alone:
    ``adaptive``/``on``/``off`` set the budget, ``redact`` sets the output,
    ``show``/``hide`` set the display. Composing them is what makes "stop
    showing me, keep thinking" expressible.

    Both axes move together or neither does: a budget the model accepts
    paired with an output it rejects would otherwise leave the settings
    half-applied, in a state neither the caller nor the wire agreed to.

    Args:
      command: A single word from :data:`THINKING_COMMANDS`.
      settings: The model's settings, mutated in place.
      show: The display flag in effect.

    Returns:
      show: The display flag after ``command``.

    Raises:
      ValueError: ``command`` is unknown, or the model does not offer it.

    """
    budget, output, display = _axes(command, settings, show=show)
    if not thinking_offered(command, settings):
        raise ValueError(
            f"thinking {command!r} is not offered by"
            f" {settings.capability.model_id or 'this model'}"
        )
    settings.thinking_budget = budget
    settings.thinking_output = output
    return display


def thinking_offered(command: str, settings: ModelSettings) -> bool:
    """Whether ``command`` names a selection the model allows.

    Args:
      command: A single word from :data:`THINKING_COMMANDS`.
      settings: The selection in effect, supplying the untouched axes.

    Returns:
      offered: True when applying ``command`` would succeed.

    """
    budget, output, _ = _axes(command, settings, show=True)
    capability = settings.capability
    return budget in capability.thinking_budget and output in capability.thinking_output


def describe_thinking(settings: ModelSettings, *, show: bool) -> str:
    """Render the selection as the words that would reproduce it.

    Args:
      settings: The selection in effect.
      show: Whether this client renders the reasoning.

    Returns:
      words: One or two words from :data:`THINKING_COMMANDS`.

    """
    if settings.thinking_output == "redacted":
        return "redact"
    budget = {"none": "off", "auto": "adaptive", "fixed": "on"}[
        settings.thinking_budget
    ]
    if settings.thinking_budget == "none":
        return budget
    return f"{budget} {'show' if show else 'hide'}"


def _axes(
    command: str, settings: ModelSettings, *, show: bool
) -> tuple[ThinkingBudget, ThinkingOutput, bool]:
    """Resolve one word against the axes already selected."""
    match command:
        case "adaptive":
            return ("auto", "text", show)
        case "on":
            return ("fixed", "text", show)
        case "off":
            # The one word that pins display: there is no reasoning to show.
            return ("none", "none", False)
        case "redact":
            # Server-side redaction still spends budget; only the body is
            # withheld, so an off budget turns back on.
            budget = settings.thinking_budget
            return ("auto" if budget == "none" else budget, "redacted", False)
        case "show" | "hide":
            return (
                settings.thinking_budget,
                settings.thinking_output,
                command == "show",
            )
        case _:
            valid = ", ".join(THINKING_COMMANDS)
            raise ValueError(f"thinking must be one of: {valid}")
