"""The setting a typed name means: exact, then by its last segments, then fuzzily.

Typical usage example:

  setting = match(harvest(owners), "effort")    # model.thinking.effort
  setting = match(settings, "svctier")          # model.service_tier
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal


if TYPE_CHECKING:
    from collections.abc import Sequence

    from sagent.tui.settings.types import Setting


__all__ = [
    "Tier",
    "match",
]


type Tier = Literal[
    "exact",
    "suffix",
    "fuzzy",
]
"""How a typed name may match: the whole name, its last segments, or its letters."""


def match(
    settings: Sequence[Setting],
    typed: str,
    *,
    tiers: Sequence[Tier] = ("exact", "suffix", "fuzzy"),
) -> Setting | None:
    """Return the setting ``typed`` names, trying each tier in turn.

    Within a tier the first setting that matches wins, so the order settings
    were harvested in settles a name several share. A fuzzy match is the
    name's letters in order, from the start of one of its segments:
    ``svctier`` is ``service_tier``, ``thout`` is ``thinking.output``. An
    action, such as ``clear``, is never matched fuzzily: a slip of the keys
    must not run one.

    Args:
      settings: Every setting, in the order a tie goes to the first.
      typed: A dotted name, such as ``thinking.effort``.
      tiers: The ways it may match, in the order they are tried.

    Returns:
      setting: The one named; ``None`` when none is.

    """
    parts = tuple(typed.split("."))
    if not all(parts):
        return None
    for tier in tiers:
        for setting in settings:
            if tier == "fuzzy" and setting.value is None:
                continue
            if _matches(tier, setting.name, parts):
                return setting
    return None


def _matches(tier: Tier, name: tuple[str, ...], parts: tuple[str, ...]) -> bool:
    """Return whether ``parts`` names ``name`` as ``tier`` allows."""
    match tier:
        case "exact":
            return name == parts
        case "suffix":
            return name[-len(parts) :] == parts
        case "fuzzy":
            typed = ".".join(parts)
            return any(
                _abbreviates(typed, ".".join(name[i:])) for i in range(len(name))
            )


def _abbreviates(short: str, name: str) -> bool:
    """Return whether ``short`` is ``name``'s letters in order, starting at its first."""
    letters = iter(name)
    return short[0] == name[0] and all(c in letters for c in short)
