"""Tests for the screen as settings: whether it shows the model's reasoning."""

from __future__ import annotations

from dataclasses import dataclass

import re

import pytest

from sagent.lib.userdirs import config_dir
from sagent.providers.catalog import ModelCatalog
from sagent.settings.flags import flags
from sagent.settings.harvest import harvest
from sagent.tui.main import _configure
from sagent.tui.settings.screen import ScreenSettings


@dataclass(slots=True, kw_only=True)
class _View:
    show_thinking: bool = False


@pytest.mark.asyncio
async def test_thinking_shows_or_hides_the_reasoning_without_touching_the_model() -> (
    None
):
    view = _View()
    setting = next(
        s
        for s in harvest([(("screen",), ScreenSettings(view=view))])
        if s.name == ("screen", "thinking")
    )
    assert (setting.name, setting.cli) == (("screen", "thinking"), True)
    assert setting.value is not None
    assert (setting.value(), tuple(setting.options())) == (
        "hide (default)",
        ("default", "show", "hide"),
    )
    assert await setting.apply("show") == "show"
    assert (view.show_thinking, setting.value()) == (True, "show")
    message = "'off' is not one of: default, show, hide"
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        await setting.apply("off")
    assert setting.value() == "show"
    # The session keeps the model's summaries; the screen hides them unless asked.
    assert await setting.apply("default") == "hide (default)"
    assert (view.show_thinking, setting.value()) == (False, "hide (default)")


@pytest.mark.asyncio
async def test_theme_uses_the_shared_setting_surface_and_restores_default() -> None:
    settings = {
        setting.name: setting
        for setting in harvest([(("screen",), ScreenSettings(view=_View()))])
    }
    assert {("screen", "thinking"), ("screen", "theme")} <= set(settings)
    setting = settings[("screen", "theme")]
    assert setting.cli
    assert setting.value is not None
    assert setting.value() == "default"
    assert set(setting.options()) == {"default", "terminal", "light", "dark"}
    assert await setting.apply("light") == "light"
    assert setting.value() == "light"
    with pytest.raises(ValueError, match="theme"):
        await setting.apply("../elsewhere")
    assert setting.value() == "light"
    assert await setting.apply("default") == "default"
    assert await setting.apply("terminal") == "terminal"
    assert tuple(setting.options()).count("default") == 1


@pytest.mark.asyncio
async def test_custom_theme_choices_and_help_use_the_shared_user_directory() -> None:
    path = config_dir() / "rekursiv-ai" / "sagent" / "themes" / "paper.yaml"
    path.parent.mkdir(parents=True, mode=0o700)
    path.write_text("preset: light\nstyles:\n  accent: 'bold magenta'\n")
    settings = ScreenSettings(view=_View())
    setting = next(s for s in settings.settings() if s.name == ("theme",))
    assert "paper" in setting.options()
    assert str(path.parent) in setting.help_long
    assert await setting.apply("paper") == "paper"
    accent = settings.theme.current.style("accent")
    assert accent.bold is True
    assert accent.color is not None
    assert accent.color.name == "magenta"


@pytest.mark.asyncio
async def test_theme_creation_and_role_edits_share_the_setting_surface() -> None:
    owner = ScreenSettings(view=_View())
    choose = next(s for s in owner.settings() if s.name == ("theme",))
    assert await choose.apply("paper") == "paper"
    setting = next(
        s for s in owner.settings() if s.name == ("theme", "paper", "thinking")
    )
    assert setting.cli
    assert setting.value is not None
    assert await setting.apply("bold cyan") == "paper.thinking: bold cyan"
    assert setting.value() == "bold cyan"
    assert owner.theme.current.styles["thinking"].bold is True
    restored = ScreenSettings(view=_View())
    assert restored.theme.current.name == "paper"
    assert restored.theme.current.styles == owner.theme.current.styles
    preset = next(
        s for s in owner.settings() if s.name == ("theme", "default", "thinking")
    )
    assert await preset.apply("italic red") == "user1.thinking: italic red"
    assert owner.theme.current.name == "user1"


@pytest.mark.asyncio
async def test_cli_theme_edits_use_the_same_values_and_persistence() -> None:
    owner = ScreenSettings(view=_View())
    owner.theme.select("paper")
    rest, given = flags(
        ["--screen.theme.paper.thinking", "bold cyan"],
        static=frozenset(),
    )
    assert rest == []
    await _configure(
        harvest([(("screen",), owner)]),
        catalog=ModelCatalog(factories={}),
        given=given,
    )
    assert owner.theme.current.styles["thinking"].bold is True
    restored = ScreenSettings(view=_View())
    assert restored.theme.current.name == "paper"
    assert restored.theme.current.styles == owner.theme.current.styles
    setting = next(
        s for s in owner.settings() if s.name == ("theme", "paper", "thinking")
    )
    assert await setting.apply("none") == "paper.thinking: none"
    assert str(owner.theme.current.styles["thinking"]) == "none"


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
