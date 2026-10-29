"""A tool's display as settings, each change replacing the tool the agent reads.

Typical usage example:

  owners = [..., (("tool", "Bash"), ToolDisplaySettings(tools=tools, name="Bash"))]
  /tool.Bash.display.output.lines 3,1
"""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass, fields
from functools import partial
from typing import TYPE_CHECKING, cast

import dataclasses

from sagent.tools.display import DisplaySpec, override
from sagent.tools.types import Tool
from sagent.tui.settings.types import Setting, SettingError


if TYPE_CHECKING:
    from sagent.tools.display import ToolDisplay


__all__ = [
    "ToolDisplaySettings",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolDisplaySettings:
    """Exports how the tool ``name`` in ``tools`` shows: ``display.*``."""

    tools: MutableMapping[str, Tool]
    """Where the agent reads its tools; a change replaces the tool here."""

    name: str

    def settings(self) -> list[Setting]:
        """Return each display setting: per part, its lines, width, overflow.

        Returns:
          settings: Each, named relative to where the tool is mounted.

        """
        return [
            *(
                self._spec(part, spec.name)
                for part in ("command", "output")
                for spec in fields(DisplaySpec)
            ),
            Setting(
                name=("display", "command_language"),
                help_short="the Pygments lexer for the input row; empty for none",
                usage="LEXER",
                value=partial(self._value, "command_language"),
                options=lambda: ("default",),
                apply=partial(self._apply, "command_language"),
                cli=True,
            ),
        ]

    def _spec(self, part: str, name: str) -> Setting:
        """Return the setting for ``part``'s ``DisplaySpec`` field ``name``."""
        match name:
            case "lines":
                help_short, usage = "lines kept from the head and tail", "HEAD,TAIL|all"
            case "width":
                help_short, usage = "widest line, in cells; 0 for the pane's", "CELLS"
            case _:
                help_short, usage = (
                    "whether a long line folds or is cut",
                    "fold|ellipsis",
                )
        return Setting(
            name=("display", part, name),
            help_short=f"the {part}'s {help_short}",
            usage=usage,
            value=partial(self._value, f"{part}.{name}"),
            options=lambda: (
                ("default", "fold", "ellipsis") if name == "overflow" else ("default",)
            ),
            apply=partial(self._apply, f"{part}.{name}"),
            cli=True,
        )

    def _default(self) -> ToolDisplay:
        """Return the display the tool is built with; its own when it is no dataclass."""
        tool = self.tools[self.name]
        if not dataclasses.is_dataclass(tool) or isinstance(tool, type):
            return tool.display
        declared = next(f for f in dataclasses.fields(tool) if f.name == "display")
        if declared.default is not dataclasses.MISSING:
            return cast("ToolDisplay", declared.default)
        if declared.default_factory is not dataclasses.MISSING:
            return cast("ToolDisplay", declared.default_factory())
        return tool.display

    def _value(self, path: str) -> str:
        shown = _text(self.tools[self.name].display, path)
        default = shown == _text(self._default(), path)
        return f"{shown} (default)" if default else shown

    async def _apply(self, path: str, args: str) -> str:
        tool = self.tools[self.name]
        if not dataclasses.is_dataclass(tool) or isinstance(tool, type):
            raise SettingError(f"{self.name} cannot change its display")
        wanted = _text(self._default(), path) if args == "default" else args
        try:
            display = override(tool.display, f"{path}={wanted}")
        except ValueError as e:
            raise SettingError(str(e)) from e
        self.tools[self.name] = dataclasses.replace(tool, display=display)
        return self._value(path)


def _text(display: ToolDisplay, path: str) -> str:
    """Return the setting ``path`` of ``display`` as ``override`` reads it back."""
    if path == "command_language":
        return display.command_language
    part, _, name = path.partition(".")
    spec = display.command if part == "command" else display.output
    match name:
        case "lines":
            return "all" if spec.lines is None else f"{spec.lines[0]},{spec.lines[1]}"
        case "width":
            return str(spec.width)
        case _:
            return spec.overflow
