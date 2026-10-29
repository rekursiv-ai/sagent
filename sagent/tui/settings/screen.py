"""The screen as settings: whether it shows the model's reasoning.

What the model sends back (``model.thinking.output``) is the model's; whether
this screen draws it is the screen's.

Typical usage example:

  owners = [..., (("screen",), ScreenSettings(view=printer))]
  /screen.thinking hide
"""

from __future__ import annotations

from dataclasses import dataclass

from sagent.tui.settings.types import Setting, SettingError
from sagent.tui.types import ThinkingView


__all__ = [
    "ScreenSettings",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class ScreenSettings:
    """Exports ``thinking``: whether ``view`` shows the model's reasoning."""

    view: ThinkingView

    def settings(self) -> list[Setting]:
        """Return the screen's one setting.

        Returns:
          settings: ``thinking``, show or hide.

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
        ]

    def _shown(self) -> str:
        return "show (default)" if self.view.show_thinking else "hide"

    async def _show(self, args: str) -> str:
        if args not in ("default", "show", "hide"):
            raise SettingError(f"{args!r} is not one of: default, show, hide")
        self.view.show_thinking = args != "hide"
        return self._shown()
