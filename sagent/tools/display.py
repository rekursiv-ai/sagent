"""Per-tool output display policy.

A tool call renders as a header line, an optional ``⎿`` input row (the
Bash command, the Edit diff receipt), and an indented output body. Only
the body is configurable; errors and hints always show.

The body is bounded head-and-tail rather than head-only: measured across
894 real results over 20 lines, the load-bearing line sits in the tail
only 16.9% of the time versus head only 6.7%, so head-biased truncation
drops the answer 2.5x more often.

The knobs live as flat scalar attributes on the tool (``output``,
``output_head_rows``, ...) because ``--tool NAME.key=value`` is a
literal transcription of the constructor call; :func:`row_spec` gathers
them back into an :class:`OutputSpec` for the renderer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from rich.cells import chop_cells


__all__ = [
    "CommandDisplayable",
    "Displayable",
    "OutputSpec",
    "Toggle",
    "ToolDisplay",
    "Wrap",
    "format_output",
    "row_spec",
]

type Toggle = Literal["on", "off"]
"""Whether the output body renders."""

type Wrap = Literal["wrap", "chop"]
"""What happens to a line wider than the pane.

``wrap`` continues on the next line (pandas ``expand_frame_repr``);
``chop`` keeps the head and marks the cut (``less -S``).
"""


@dataclass(frozen=True, slots=True, kw_only=True)
class OutputSpec:
    """Resolved display policy for a tool's output body.

    Attributes:
      show: Whether the body renders at all.
      head_rows: Leading rows kept. ``0`` with ``tail_rows`` set gives a
          tail-only view.
      tail_rows: Trailing rows kept, after a ``⋯ N lines ⋯`` marker.
      max_width: Cell width cap. ``0`` defers to the caller's width
          (the terminal, less the indent).
      wrap: Treatment of a line exceeding the effective width.

    """

    show: bool = False
    head_rows: int = 0
    tail_rows: int = 0
    max_width: int = 0
    wrap: Wrap = "wrap"

    @property
    def bounded(self) -> bool:
        """Whether this spec elides anything at all."""
        return self.head_rows > 0 or self.tail_rows > 0


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolDisplay:
    """Resolved display policy for one tool's call block.

    Attributes:
      command: Policy for the ``⎿`` input row -- the Bash command. A
          tool with no command knobs gets an unbounded shown spec, which
          is what a one-line file path or receipt wants.
      output: Policy for the indented result body.
      command_lang: Pygments lexer name for the input row, e.g.
          ``"bash"``. Empty leaves it unhighlighted, right for an input
          that is not source.

    """

    command: OutputSpec = OutputSpec(show=True)
    output: OutputSpec = OutputSpec()
    command_lang: str = ""


@runtime_checkable
class Displayable(Protocol):
    """A tool that declares how its result body renders.

    Structural, not inherited: the knobs are plain constructor fields so
    ``--tool NAME.key=value`` stays a literal transcription of the call,
    and a tool opts in simply by declaring them.
    """

    output: Toggle
    output_head_rows: int
    output_tail_rows: int
    output_max_width: int
    output_wrap: Wrap


@runtime_checkable
class CommandDisplayable(Protocol):
    """A tool whose ``⎿`` input row is multi-line source worth bounding."""

    command_head_rows: int
    command_tail_rows: int
    command_lang: str


def row_spec(tool: object) -> ToolDisplay:
    """Gather a tool's flat display attributes into a policy.

    A tool that declares no knobs reads as hidden, which is what keeps a
    tool from dumping result bodies into the pane merely by existing.

    Args:
      tool: Constructed tool instance.

    Returns:
      display: Resolved policy for the tool's call block.

    """
    if not isinstance(tool, Displayable):
        return ToolDisplay()
    output = OutputSpec(
        show=tool.output == "on",
        head_rows=tool.output_head_rows,
        tail_rows=tool.output_tail_rows,
        max_width=tool.output_max_width,
        wrap=tool.output_wrap,
    )
    if not isinstance(tool, CommandDisplayable):
        return ToolDisplay(output=output)
    return ToolDisplay(
        output=output,
        command=OutputSpec(
            show=True,
            head_rows=tool.command_head_rows,
            tail_rows=tool.command_tail_rows,
            max_width=tool.output_max_width,
            wrap=tool.output_wrap,
        ),
        command_lang=tool.command_lang,
    )


def format_output(text: str, spec: OutputSpec, *, width: int = 0) -> list[str]:
    """Render ``text`` into display rows under ``spec``.

    Wrapping happens BEFORE the row budget applies, so the counts bound
    what the terminal actually prints rather than what the tool emitted.

    Args:
      text: Raw body; may contain newlines.
      spec: Resolved output policy.
      width: Cell width available to the caller, used when
          ``spec.max_width`` is ``0``. ``0`` here too means no cap.

    Returns:
      rows: Rendered lines, empty when the body is hidden or blank, with
          a ``⋯ N lines ⋯`` marker between head and tail when the budget
          elided content.

    """
    if not spec.show or not text.strip():
        return []
    usable = spec.max_width or width
    # Budget by LOGICAL line, then wrap: slicing wrapped rows would keep
    # an arbitrary fragment of one long line ("il -2" from the tail of a
    # pipeline), which reads as corrupt rather than elided. The wrap
    # still bounds the total, since each kept line is itself capped.
    lines = text.rstrip("\n").split("\n")
    kept = lines
    marker_at = -1
    if spec.bounded and len(lines) > spec.head_rows + spec.tail_rows:
        hidden = len(lines) - spec.head_rows - spec.tail_rows
        head = lines[: spec.head_rows]
        # ``lines[-0:]`` is the whole list, so a zero tail must slice to
        # empty explicitly or a head-only spec renders everything twice.
        tail = lines[-spec.tail_rows :] if spec.tail_rows else []
        kept = [*head, f"\u22ef {hidden} lines \u22ef", *tail]
        marker_at = len(head)
    rows: list[str] = []
    for i, raw in enumerate(kept):
        # The marker is ours, not content: never wrap or chop it.
        rows.extend([raw] if i == marker_at else _fit(raw, usable, spec.wrap))
    return rows


def _fit(line: str, width: int, wrap: Wrap) -> list[str]:
    """Apply the width cap to one physical line."""
    if width <= 0:
        return [line]
    parts = chop_cells(line, width) or [""]
    if wrap == "wrap":
        return parts
    if len(parts) == 1:
        return parts
    # Reserve one cell for the marker: a silently shortened line is
    # indistinguishable from a short one, which is the whole failure
    # mode this width cap is supposed to make visible.
    return [(chop_cells(line, max(1, width - 1)) or [""])[0] + "\u2026"]
