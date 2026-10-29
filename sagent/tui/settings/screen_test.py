"""Tests for the screen as settings: whether it shows the model's reasoning."""

from __future__ import annotations

from dataclasses import dataclass

import re

import pytest

from sagent.tui.settings.harvest import harvest
from sagent.tui.settings.screen import ScreenSettings


@dataclass(slots=True, kw_only=True)
class _View:
    show_thinking: bool = True


@pytest.mark.asyncio
async def test_thinking_shows_or_hides_the_reasoning_without_touching_the_model() -> (
    None
):
    view = _View()
    (setting,) = harvest([(("screen",), ScreenSettings(view=view))])
    assert (setting.name, setting.cli) == (("screen", "thinking"), True)
    assert setting.value is not None
    assert (setting.value(), tuple(setting.options())) == (
        "show (default)",
        ("default", "show", "hide"),
    )
    assert await setting.apply("hide") == "hide"
    assert (view.show_thinking, setting.value()) == (False, "hide")
    message = "'off' is not one of: default, show, hide"
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        await setting.apply("off")
    assert setting.value() == "hide"
    assert await setting.apply("default") == "show (default)"
    assert setting.value() == "show (default)"


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
