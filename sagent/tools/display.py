r"""How a tool call shows in the terminal: its input rows and its output.

A long result shows a few lines from each end, with ``⋯ N lines ⋯`` between:
across 894 results over 20 lines, the line that mattered sat in the tail 16.9%
of the time and the head 6.7%, so keeping only the head loses it more often.

A spec only picks lines; the printer fits each kept line to its row, since only
it knows the pane and what leads the row.

Typical usage example:

  display = override(Bash().display, "output.lines=3,1")
  for line in display.output.keep(result.text().split("\\n")):
      print(line.marker if isinstance(line, Elided) else line)
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, Literal


if TYPE_CHECKING:
    from collections.abc import Sequence


__all__ = [
    "DisplaySpec",
    "Elided",
    "Overflow",
    "ToolDisplay",
    "override",
]


type Overflow = Literal[
    "fold",
    "ellipsis",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class DisplaySpec:
    """How much of a block of text the pane shows, and how wide."""

    lines: tuple[int, int] | None = (2, 2)
    """Lines kept from the head and the tail; ``(0, 0)`` hides it, ``None`` keeps all."""

    width: int = 0
    """Widest a line may be, in cells; ``0`` is the pane's width."""

    overflow: Overflow = "fold"
    """Whether a wider line continues below, or is cut with ``…``."""

    def __post_init__(self) -> None:
        """Reject a negative count, which would hide lines the marker does not count."""
        if self.width < 0 or (self.lines is not None and min(self.lines) < 0):
            raise ValueError(f"{self} has a negative count.")

    def keep[T](self, lines: Sequence[T]) -> list[T | Elided]:
        """Return the lines of a block that show, the rest counted between them.

        Lines are kept whole, never cut to rows, so a long line never shows
        as a fragment of itself.

        Args:
          lines: The block's lines, styled or not.

        Returns:
          kept: The head and tail ``self.lines`` keeps, with an ``Elided``
            between them when it drops any; empty when it is ``(0, 0)``.

        """
        if self.lines is None:
            return list(lines)
        head, tail = self.lines
        if head + tail == 0:
            return []
        if len(lines) <= head + tail:
            return list(lines)
        # Not ``lines[-tail:]``: with no tail, ``lines[-0:]`` is every line.
        return [
            *lines[:head],
            Elided(hidden=len(lines) - head - tail),
            *lines[len(lines) - tail :],
        ]


@dataclass(frozen=True, slots=True, kw_only=True)
class Elided:
    """The lines a ``DisplaySpec`` dropped, shown as ``⋯ N lines ⋯``."""

    hidden: int

    @property
    def marker(self) -> str:
        """Return the row that stands for the dropped lines."""
        return f"⋯ {self.hidden} lines ⋯"


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolDisplay:
    """How one tool's calls show: the input rows, then the output."""

    command: DisplaySpec = DisplaySpec(lines=None)
    """The input rows, such as a Bash command or an Edit receipt."""

    output: DisplaySpec = DisplaySpec(lines=(0, 0))
    """The result; hidden unless the tool shows it."""

    command_language: str = ""
    """The Pygments lexer for the input rows, such as ``"bash"``; ``""`` for none."""


def override(display: ToolDisplay, setting: str) -> ToolDisplay:
    """Return ``display`` with one setting changed, as a command line gives it.

    Args:
      display: The tool's display.
      setting: ``part.field=value``, such as ``output.lines=3,1``,
        ``output.lines=all``, ``command.overflow=ellipsis``, or
        ``command_language=bash``.

    Returns:
      display: ``display`` with that setting.

    Raises:
      ValueError: The setting names no field, or its value does not fit it.

    """
    path, equals, value = setting.partition("=")
    if not equals:
        raise ValueError(f"Setting {setting!r} lacks '='.")
    if path == "command_language":
        return replace(display, command_language=value)
    settable = {
        f"{part}.{f.name}": (part, f.name)
        for part in ("command", "output")
        for f in fields(DisplaySpec)
    }
    if path not in settable:
        raise ValueError(
            f"Unknown display setting {path!r}. Settable: command_language, and"
            " command or output with .lines, .width, or .overflow.",
        )
    part, name = settable[path]
    spec = display.command if part == "command" else display.output
    if name == "lines":
        changed = replace(spec, lines=_lines(value))
    elif name == "width":
        changed = replace(spec, width=_width(value))
    else:
        changed = replace(spec, overflow=_overflow(value))
    if part == "command":
        return replace(display, command=changed)
    return replace(display, output=changed)


def _lines(raw: str) -> tuple[int, int] | None:
    """Return ``HEAD,TAIL`` as a pair, or ``all`` as ``None``."""
    if raw == "all":
        return None
    head, comma, tail = raw.partition(",")
    if not comma or not head.isdigit() or not tail.isdigit():
        raise ValueError(f"lines must be HEAD,TAIL or all, not {raw!r}.")
    return int(head), int(tail)


def _width(raw: str) -> int:
    """Return a width in cells."""
    if not raw.isdigit():
        raise ValueError(f"width must be a whole number, not {raw!r}.")
    return int(raw)


def _overflow(raw: str) -> Overflow:
    """Return ``fold`` or ``ellipsis``."""
    if raw == "fold":
        return "fold"
    if raw == "ellipsis":
        return "ellipsis"
    raise ValueError(f"overflow must be fold or ellipsis, not {raw!r}.")
