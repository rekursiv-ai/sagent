"""Screen settings: reasoning visibility and a shared semantic color theme.

What the model sends back (``model.thinking.output``) is the model's; whether
this screen draws it is the screen's. The model's default asks for readable
thinking so the session keeps it; the screen's default hides it.

Typical usage example:

  owners = [..., (("screen",), ScreenSettings(view=printer, theme=theme))]
  /screen.thinking show
  /screen.theme light
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

from sagent.lib.userdirs import config_dir
from sagent.settings.exceptions import SettingError
from sagent.settings.types import Setting
from sagent.tui.theme import ThemeState, load_theme, theme_choices
from sagent.tui.types import ThinkingView


__all__ = [
    "ScreenSettings",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class ScreenSettings:
    """Export reasoning visibility and the terminal's semantic theme."""

    view: ThinkingView
    theme: ThemeState = field(default_factory=ThemeState)

    def settings(self) -> list[Setting]:
        """Return the screen's settings.

        Returns:
          settings: Reasoning visibility and the shared color theme.

        """
        return [
            Setting(
                name=("thinking",),
                help_short="show or hide the model's reasoning on this screen",
                usage="show|hide",
                value=self._shown,
                options=lambda: ("default", "show", "hide"),
                apply=self._show,
                cli=True,
            ),
            Setting(
                name=("theme",),
                help_short="choose the terminal's color theme",
                help_long=(
                    "Live panes and new output change immediately; printed scrollback"
                    " keeps its colors. Custom themes: "
                    f"{config_dir() / 'rekursiv-ai' / 'sagent' / 'themes'}/NAME.yaml."
                    " Use preset: default (original appearance), terminal, light, or"
                    " dark, and styles: mapping semantic roles to Rich styles, e.g."
                    " user: 'white on #373737'. syntax: selects a Pygments theme;"
                    " source_tokens: maps token names to command/diff syntax styles."
                    " Omitted roles inherit the preset. A missing NAME copies the"
                    " current appearance. /screen.theme.NAME.ROLE STYLE saves and"
                    " selects an edit; bundled presets fork as user1, user2, ... ."
                    " The last selection is remembered for future sessions;"
                    " --screen.theme applies the same choice at startup."
                ),
                usage="NAME|default",
                value=self._theme_name,
                options=theme_choices,
                apply=self._theme,
                cli=True,
            ),
            *(
                Setting(
                    name=("theme", name, role),
                    help_short=f"save {name}'s {role} style (presets fork as userN)",
                    usage="STYLE",
                    value=partial(self._role, name, role),
                    apply=partial(self._edit, name, role),
                    cli=True,
                )
                for name in theme_choices()
                for role in self.theme.current.styles
            ),
        ]

    def _role(self, name: str, role: str) -> str:
        return str(load_theme(name).styles[role])

    async def _edit(self, name: str, role: str, args: str) -> str:
        self.theme.edit(name, role, args)
        return f"{self._theme_name()}.{role}: {self.theme.current.styles[role]}"

    def _theme_name(self) -> str:
        return self.theme.current.name

    async def _theme(self, args: str) -> str:
        self.theme.select(args)
        return self._theme_name()

    def _shown(self) -> str:
        return "show" if self.view.show_thinking else "hide (default)"

    async def _show(self, args: str) -> str:
        if args not in ("default", "show", "hide"):
            raise SettingError(f"{args!r} is not one of: default, show, hide")
        self.view.show_thinking = args == "show"
        return self._shown()
