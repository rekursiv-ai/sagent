"""DashScope thinking-budget wire values."""

import pytest

from sagent.catalog import dashscope
from sagent.types.capability import ThinkingEffort


@pytest.mark.parametrize(
    ("effort", "budget"),
    [
        ("none", "0"),
        ("min", "1024"),
        ("low", "4096"),
        ("medium", "8192"),
        ("high", "16384"),
        ("xhigh", "20480"),
        ("max", "24576"),
    ],
)
def test_each_effort_sends_its_thinking_budget(
    effort: ThinkingEffort,
    budget: str,
) -> None:
    assert dashscope.thinking_budget(effort) == budget


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
