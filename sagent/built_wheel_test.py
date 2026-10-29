"""The sagent wheel, built and run outside the checkout."""

from __future__ import annotations

import pytest

from sagent.lib.testing.wheel import built_wheel
from sagent.scripts import check_wheel


pytestmark = pytest.mark.cli_uv


def test_the_cli_runs_from_the_wheel_alone() -> None:
    """What ``uvx sagent`` runs, with no torch to hide a module that needs it."""
    wheel = built_wheel("sagent", anchor=__file__)

    assert wheel.run(
        "import sys; sys.argv = ['sagent', '--help']; "
        "from sagent.tui.main import main; main()",
    ).startswith("usage: sagent")


def test_the_wheel_ships_every_module_and_the_console_script(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel = built_wheel("sagent", anchor=__file__)
    monkeypatch.chdir(wheel.project)

    assert check_wheel.expected_modules() - wheel.names() == frozenset()
    (entry_points,) = (
        name for name in wheel.names() if name.endswith(".dist-info/entry_points.txt")
    )
    assert "sagent = sagent.tui.main:main" in wheel.read(entry_points)


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
