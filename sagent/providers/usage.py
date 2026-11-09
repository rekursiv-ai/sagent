"""Native quota, subscription, and rate-limit state reported by providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Mapping

    from sagent.session.types import PlainTree


__all__ = [
    "ProviderUsage",
    "UsageReporter",
    "format_countdown",
    "format_percent",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class ProviderUsage:
    """A provider's native quota, subscription, or rate-limit state.

    Vendors expose usage in fundamentally different forms (used percentages,
    remaining fractions, RPM/TPM counts, credit balances, or local compute).
    This dataclass preserves the vendor's native metrics and provides clean,
    human-readable summaries for frontends.
    """

    provider: str
    """The provider's section name, such as ``openai-sub`` or ``acp``."""

    account: str | None = None
    """The named account or login, or ``None`` for the default."""

    summary: str = ""
    """A compact, native one-line summary suitable for status bars and pills."""

    details: tuple[str, ...] = ()
    """Native multi-line descriptions suitable for tooltips and dialogs."""

    raw: Mapping[str, PlainTree] = field(default_factory=dict)
    """Untouched vendor-reported metadata."""


class UsageReporter:
    """Default implementation of native usage methods for providers."""

    def native_usage(self) -> ProviderUsage | None:
        """Return the provider's native quota and rate-limit state, if known."""
        return None

    async def refresh_usage(self) -> ProviderUsage | None:
        """Fetch and return the freshest native quota state; default returns native_usage()."""
        return self.native_usage()


def format_countdown(seconds: float) -> str:
    """Return a duration until reset formatted as e.g. ``2h 15m``, ``45s``, or ``3d 4h``.

    Args:
      seconds: Seconds remaining until reset; values <= 0 return ``"now"``.

    Returns:
      text: Formatted human-readable duration.
    """
    total = int(max(seconds, 0))
    if total <= 0:
        return "now"
    days, rem = divmod(total, 86_400)
    hours, rem = divmod(rem, 3_600)
    minutes, secs = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours}h" if hours > 0 else f"{days}d"
    if hours > 0:
        return f"{hours}h {minutes}m" if minutes > 0 else f"{hours}h"
    if minutes > 0:
        return f"{minutes}m {secs}s" if secs > 0 else f"{minutes}m"
    return f"{secs}s"


def format_percent(ratio: float) -> str:
    """Return a fraction between 0.0 and 1.0 formatted as a percentage string (e.g. ``85.0%``)."""
    clamped = max(0.0, min(1.0, ratio))
    return f"{clamped * 100:.1f}%"
