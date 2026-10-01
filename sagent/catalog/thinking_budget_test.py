"""Tests for the per-vendor ``thinking_budget`` wire tables."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, TypeAliasType, cast, get_args

import pytest

from sagent.catalog import dashscope, google
from sagent.types.capability import ThinkingEffort


if TYPE_CHECKING:
    from collections.abc import Callable


_EFFORTS: Final = cast(
    "tuple[ThinkingEffort, ...]",
    get_args(cast(TypeAliasType, ThinkingEffort.__value__)),
)
_BUDGETS: Final = {
    "min": "1024",
    "low": "4096",
    "medium": "8192",
    "high": "16384",
    "xhigh": "20480",
    "max": "24576",
}


@pytest.mark.parametrize("budget", [google.thinking_budget, dashscope.thinking_budget])
def test_budgets_grow_with_effort(budget: Callable[[ThinkingEffort], str]) -> None:
    """A higher effort must never send a smaller cap."""
    assert {
        effort: budget(effort) for effort in _EFFORTS if effort != "none"
    } == _BUDGETS


def test_every_effort_has_a_dashscope_budget() -> None:
    """DashScope spells "thinking off" as a zero cap."""
    assert dashscope.thinking_budget("none") == "0"
    assert {dashscope.thinking_budget(effort) for effort in _EFFORTS} >= {"0"}


def test_google_refuses_a_budget_for_no_thinking() -> None:
    """Gemini disables thinking by omitting ``thinkingConfig``, not a zero cap."""
    with pytest.raises(ValueError, match="omit thinkingConfig"):
        google.thinking_budget("none")


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
