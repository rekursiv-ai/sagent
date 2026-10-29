"""Tests for matching what was typed to a setting: exact, then suffix, then fuzzy."""

from __future__ import annotations

import pytest

from sagent.tui.settings.match import Tier, match
from sagent.tui.settings.types import Setting


async def _ok(args: str) -> str:
    return args


def _settings(*names: str) -> list[Setting]:
    """Return a setting per name; one in ``_ACTIONS`` is an action, the rest values."""
    return [
        Setting(
            name=tuple(n.split(".")),
            help_short=n,
            apply=_ok,
            value=None if n in _ACTIONS else _shown,
        )
        for n in names
    ]


def _shown() -> str:
    return "v"


_ACTIONS = frozenset({"clear", "compact"})


_NAMES = (
    "clear",
    "compact",
    "model",
    "model.provider",
    "model.provider.account",
    "model.thinking",
    "model.thinking.effort",
    "model.thinking.output",
    "model.service_tier",
    "tool.Bash.display.output.lines",
    "screen.thinking",
)


@pytest.mark.parametrize(
    ("typed", "found"),
    [
        ("model.thinking.effort", "model.thinking.effort"),
        ("model", "model"),
        ("effort", "model.thinking.effort"),
        ("thinking.effort", "model.thinking.effort"),
        ("thinking", "model.thinking"),
        ("screen.thinking", "screen.thinking"),
        ("account", "model.provider.account"),
        ("provider", "model.provider"),
        ("output", "model.thinking.output"),
        ("eff", "model.thinking.effort"),
        ("svctier", "model.service_tier"),
        ("thout", "model.thinking.output"),
        ("clear", "clear"),
        ("Bash.output.lines", "tool.Bash.display.output.lines"),
    ],
)
def test_exact_then_suffix_then_fuzzy_and_the_first_hit_wins(
    typed: str,
    found: str,
) -> None:
    hit = match(_settings(*_NAMES), typed)
    assert hit is not None
    assert ".".join(hit.name) == found


@pytest.mark.parametrize("typed", ["", "zzz", "home/jvd", "model..effort", "x.model"])
def test_what_names_nothing_is_none(typed: str) -> None:
    assert match(_settings(*_NAMES), typed) is None


@pytest.mark.parametrize(
    ("tiers", "typed", "found"),
    [
        (("exact",), "effort", None),
        (("exact", "suffix"), "effort", "model.thinking.effort"),
        (("exact", "suffix"), "eff", None),
        (("fuzzy",), "model", "model"),
        (("fuzzy", "exact"), "thinking", "model.thinking"),
    ],
)
def test_the_caller_chooses_the_tiers_and_their_order(
    tiers: tuple[Tier, ...],
    typed: str,
    found: str | None,
) -> None:
    hit = match(_settings(*_NAMES), typed, tiers=tiers)
    assert (None if hit is None else ".".join(hit.name)) == found


@pytest.mark.parametrize("typed", ["c", "cmp", "cl"])
def test_an_action_runs_only_by_its_whole_name(typed: str) -> None:
    """A slip of the keys must not clear the window; a value only shows itself."""
    assert match(_settings("clear", "compact"), typed) is None


def test_an_earlier_tier_beats_an_earlier_setting() -> None:
    settings = _settings("clean", "lean")
    hit = match(settings, "lean")
    assert hit is not None
    assert hit.name == ("lean",)


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
