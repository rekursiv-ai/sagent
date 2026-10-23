"""Cross-provider thinking budgets preserve effort ordering."""

from collections.abc import Callable
from typing import cast, get_args

import itertools

import pytest

from sagent.catalog import dashscope, google
from sagent.types.capability import ThinkingEffort


@pytest.mark.parametrize("budget", [google.thinking_budget, dashscope.thinking_budget])
def test_budgets_grow_with_effort(budget: Callable[[ThinkingEffort], str]) -> None:
    efforts = cast(tuple[ThinkingEffort, ...], get_args(ThinkingEffort.__value__))
    values = [int(budget(effort)) for effort in efforts if effort != "none"]
    assert all(lower < upper for lower, upper in itertools.pairwise(values))


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
