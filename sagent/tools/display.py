"""How a tool call shows in the terminal: its input row and its output.

A long result shows a few lines from each end, with ``⋯ N lines ⋯`` between:
across 894 results over 20 lines, the line that mattered sat in the tail 16.9%
of the time and the head 6.7%, so keeping only the head loses it more often.

Typical usage example:

  display = override(Bash().display, "output.lines=3,1")
  for row in display.output.render(result.text(), width=80):
      print(row)
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, Literal


if TYPE_CHECKING:
    from rich.cells import chop_cells
else:
    from wrapt import lazy_import

    chop_cells = lazy_import("rich.cells", "chop_cells")  # 60 ms cold.


__all__ = [
    "DisplaySpec",
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

    def render(self, text: str, *, width: int) -> list[str]:
        """Return the rows to print for ``text``.

        Lines are kept whole, then each is fit to the width, so a long line
        never shows as a fragment.

        Args:
          text: The block, newlines and all.
          width: The pane's width, used when ``self.width`` is ``0``; ``0``
            for no limit.

        Returns:
          rows: What to print; empty when the block is hidden or blank.

        """
        if not text.strip() or self.lines == (0, 0):
            return []
        rows = text.rstrip("\n").split("\n")
        marker = -1
        if self.lines is not None and len(rows) > sum(self.lines):
            head, tail = self.lines
            hidden = len(rows) - head - tail
            rows = [
                *rows[:head],
                f"⋯ {hidden} lines ⋯",
                *(rows[-tail:] if tail else ()),
            ]
            marker = head
        fit = self.width or width
        return [
            part
            for i, row in enumerate(rows)
            for part in (
                [row] if i == marker else _fit(row, width=fit, overflow=self.overflow)
            )
        ]


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolDisplay:
    """How one tool's calls show: the ``⎿`` input row, then the output."""

    command: DisplaySpec = DisplaySpec(lines=None)
    """The input row, such as a Bash command or an Edit receipt."""

    output: DisplaySpec = DisplaySpec(lines=(0, 0))
    """The result; hidden unless the tool shows it."""

    command_language: str = ""
    """The Pygments lexer for the input row, such as ``"bash"``; ``""`` for none."""


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
    part, _, name = path.partition(".")
    if part not in {"command", "output"} or name not in {
        f.name for f in fields(DisplaySpec)
    }:
        raise ValueError(
            f"Unknown display setting {path!r}. Settable: command_language, and"
            " command or output with .lines, .width, or .overflow.",
        )
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


def _fit(line: str, *, width: int, overflow: Overflow) -> list[str]:
    """Return ``line`` as rows no wider than ``width`` cells."""
    if width <= 0:
        return [line]
    rows = chop_cells(line, width) or [""]
    if overflow == "fold" or len(rows) == 1:
        return rows
    # One cell for the mark: a line cut silently looks like a short one.
    return [(chop_cells(line, max(1, width - 1)) or [""])[0] + "…"]
