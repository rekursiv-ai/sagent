"""``ConsolePrinter``: rich-backed implementation of :class:`Printer`.

Implements the full :class:`Printer` surface using rich primitives
plus v1's existing render helpers (``print_user_bar``, ``TightMarkdown``,
``render_diff_detail``, ``set_terminal_title``). Reuse rather than
duplicate -- the formatting logic is already correct and battle-tested.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import io
import re

from rich.console import Console
from rich.text import Text

from sagent.custom_types import (
    JsonMessage,
    MultipartMessage,
    TextMessage,
)
from sagent.repl.format import (
    format_elapsed,
    print_user_bar,
    set_terminal_title,
)
from sagent.repl.render_diff import render_diff_detail
from sagent.repl.tight_markdown import TightMarkdown


if TYPE_CHECKING:
    from collections.abc import Mapping

    from sagent.custom_types import Message


class ConsolePrinter:
    """Rich-backed :class:`Printer` implementation.

    Attributes:
      console: Underlying rich console (exposed for one-off rich
          renderers that need to emit Text/Panel/Syntax objects).

    """

    def __init__(self, console: Console) -> None:
        self.console = console

    def write_line(self, text: str) -> None:
        """Render a complete line; the console adds the newline.

        Markup interpretation is disabled so payloads like
        ``[/clear] history cleared`` render verbatim instead of being
        parsed as a closing rich-markup tag.
        """
        self.console.print(text, markup=False, highlight=False)

    def write_chunk(self, text: str) -> None:
        """Render a streaming partial without a newline."""
        self.console.out(text, end="", highlight=False)

    def write_markdown(self, text: str) -> None:
        """Render a stable markdown block as scrollback text.

        Mirrors v1: leading newline, then ``TightMarkdown(text)``.
        """
        self.console.print()
        self.console.print(TightMarkdown(text))

    def write_user_bar(self, text: str) -> None:
        """Render full-width dark-gray user-message bar."""
        print_user_bar(self.console, text)

    def write_tool_label(self, text: str) -> None:
        """Render dim multi-line tool-call label."""
        for line in (text or "").splitlines() or [""]:
            self.console.print(Text(f"  {line}", style="dim"))

    def write_tool_error(self, text: str) -> None:
        """Render red, indented tool-error line."""
        self.console.print(Text(f"    ✗ {text.strip()}", style="dim red"))

    def write_tool_summary(self, text: str) -> None:
        """Render the dim ``  ⎿ <summary>`` receipt line for a tool result.

        Mirrors the diff-header glyph (``⎿``) so tools without a
        diff (Read, Bash, Glob, Grep, ...) get the same visual
        cue that the call landed and what came back.
        """
        self.console.print(Text(f"    ⎿  {text.strip()}", style="dim"))

    def write_hint(self, text: str) -> None:
        """Render a dim yellow ``hint:`` line (bash-lint nudge surface)."""
        self.console.print(Text(f"    hint: {text}", style="dim yellow"))

    def write_interrupted(self) -> None:
        """Render the dim ``[interrupted]`` line for cancelled work."""
        self.console.print(Text("[interrupted]", style="dim"))

    def write_child_block(self, label: str, items: list[Message]) -> None:
        r"""Render a child agent's labeled block.

        Format: first line carries the label gutter (``Agent_N  :  ``),
        subsequent lines indent to align with the gutter width. Each
        item is rendered through the same paths the parent uses for
        its own output (Markdown for ``text/plain``, dim text for
        tool labels, the diff renderer for tool results, etc.) so
        child output looks identical to parent output up to the
        gutter prefix.

        Implementation: render each item to a captured ``Console``
        with reduced width (real width minus gutter), then split the
        captured ANSI string into lines, strip any blank-only lines
        at the start/end (Rich's ``write_markdown`` adds a leading
        blank, TightMarkdown a trailing one), prefix each line with
        the gutter, and wrap the whole thing in dim-baseline ANSI so
        the entire child block reads as background output. The dim
        attribute is re-applied after every Rich ``\x1b[0m`` reset so
        per-span styles don't drop it.
        """
        if not items:
            return
        gutter_width = _gutter_width(label)
        pfx = _gutter_prefix(label, gutter_width)
        indent = " " * gutter_width

        # Render into a sub-printer wrapping a Console that writes
        # to a buffer with reduced width. ConsolePrinter is the
        # natural sub-printer because the same code paths
        # (write_markdown, write_tool_error, write_diff, etc.) need
        # to fire here.
        buf = io.StringIO()
        # ``color_system`` is a Literal in rich's typing; the live
        # console exposes it as an arbitrary str. Pass through as
        # cast so the inner console matches the outer's color depth.
        inner_console = Console(
            file=buf,
            width=max(20, self.console.width - gutter_width),
            force_terminal=self.console.is_terminal,
            color_system=cast(
                "Literal['auto', 'standard', '256', 'truecolor', 'windows'] | None",
                self.console.color_system,
            ),
            highlight=False,
            soft_wrap=False,
        )
        inner = ConsolePrinter(inner_console)
        for item in items:
            _render_child_item(inner, item)

        captured = buf.getvalue()
        if not captured:
            return
        lines = _strip_blank_edges(captured.split("\n"))
        if not lines:
            return
        out: list[str] = []
        for i, line in enumerate(lines):
            prefix = pfx if i == 0 else indent
            out.append(_dim_baseline(prefix + line))
        # Bypass ``console.print`` so the embedded ANSI codes pass
        # through verbatim instead of being parsed as Rich markup.
        self.console.file.write("\n".join(out) + "\n")
        self.console.file.flush()

    def write_thinking(self, text: str) -> None:
        """Render italic dim 'Thinking' header followed by indented body."""
        self.console.print(Text("∴ Thinking", style="italic dim"))
        for line in (text or "").splitlines() or [""]:
            self.console.print(Text(f"  {line}", style="dim"))
        self.console.print()

    def write_diff(self, diff: str, file_path: str = "") -> None:
        """Render a unified diff with syntax highlighting."""
        render_diff_detail(self.console, diff, file_path=file_path)

    def set_terminal_title(self, text: str) -> None:
        """Write an OSC 0 title escape (no-op when stderr is not a TTY)."""
        set_terminal_title(text)


# ANSI dim attribute. ``\x1b[2m`` enables dim; ``\x1b[22m`` cancels
# only the dim attribute (vs. ``\x1b[0m`` which resets everything).
# We re-apply dim after every full reset so embedded per-span styles
# from Rich don't drop the dim baseline.
_ANSI_DIM = "\x1b[2m"
_ANSI_RESET = "\x1b[0m"
_ANSI_RESET_DIM = _ANSI_RESET + _ANSI_DIM
_ANSI_DIM_OFF = "\x1b[22m"
_ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")


def _strip_blank_edges(lines: list[str]) -> list[str]:
    """Drop leading/trailing blank lines, ignoring ANSI escape codes.

    Rich's ``write_markdown`` prefixes its output with a blank line
    for parent-output spacing; that blank doesn't belong inside a
    gutter-rendered child block. TightMarkdown likewise leaves a
    trailing blank line at the end of its render. Strip both edges.
    Internal blank lines (between Markdown blocks within the same
    captured string) are preserved -- they're meaningful spacing.
    """
    out = list(lines)
    while out and not _ANSI_PATTERN.sub("", out[0]).strip():
        _ = out.pop(0)
    while out and not _ANSI_PATTERN.sub("", out[-1]).strip():
        _ = out.pop()
    return out


def _dim_baseline(line: str) -> str:
    r"""Wrap a rendered ANSI line so dim is the baseline attribute.

    The line may contain Rich-emitted ANSI for color/bold/italic, with
    a trailing ``\x1b[0m`` resetting all attributes between spans. Each
    such reset would also drop our dim attribute, so replace each
    full reset with reset+dim. Bracket the line with a dim-on prefix
    and a dim-off suffix so the dim baseline doesn't leak past the
    end of the child block.
    """
    rebound = line.replace(_ANSI_RESET, _ANSI_RESET_DIM)
    return _ANSI_DIM + rebound + _ANSI_DIM_OFF


def _gutter_width(label: str) -> int:
    """Width of the gutter column for a given child label.

    Format is ``"<label>  :  "`` -- two spaces, colon, two spaces, so
    the gutter is ``len(label) + 5``. Held to a floor of 12 columns
    for visual consistency when labels are short (``Agent_0`` is the
    typical case at 12 chars exactly).
    """
    return max(12, len(label) + 5)


def _gutter_prefix(label: str, width: int) -> str:
    """Render the first-line gutter for ``label`` padded to ``width``."""
    pfx = f"{label}  :  "
    if len(pfx) < width:
        pfx = pfx.ljust(width)
    return pfx


def _render_child_item(printer: ConsolePrinter, item: Message) -> None:
    """Dispatch one child-event item to the appropriate printer method.

    Used by ``ConsolePrinter.write_child_block`` for each accumulated
    inner event. Rendering goes through the same code paths the
    parent uses for its own output -- the gutter prefix is applied
    around the entire block, not per item.
    """
    desc = item.descriptor
    if desc == "text/plain" and isinstance(item, TextMessage):
        printer.write_markdown(item.content)
    elif desc == "text/x-tool-label" and isinstance(item, TextMessage):
        printer.write_tool_label(item.content)
    elif desc == "text/x-thinking" and isinstance(item, TextMessage):
        printer.write_thinking(item.content)
    elif desc == "text/x-error" and isinstance(item, TextMessage):
        printer.write_tool_error(item.content)
    elif desc == "text/x-hint-tool-use-nudge" and isinstance(item, TextMessage):
        printer.write_hint(item.content)
    elif desc == "multipart/x-tool-result" and isinstance(item, MultipartMessage):
        # Lazy import: render.py imports console.py at module load.
        from sagent.repl.render import (  # noqa: PLC0415
            render_tool_result,
        )

        render_tool_result(printer, item)
    elif desc == "application/x-child-done" and isinstance(item, JsonMessage):
        data = cast("Mapping[str, float]", item.content)
        elapsed = data.get("elapsed", 0.0)
        tokens = int(data.get("model_response_tokens", 0))
        cost = data.get("cost_usd", 0.0)
        summary_parts: list[str] = [format_elapsed(elapsed)]
        if tokens:
            summary_parts.append(f"{tokens}↓")
        if cost > 0:
            summary_parts.append(f"${cost:.2f}")
        printer.console.print(
            Text(f"done [{' · '.join(summary_parts)}]", style="dim"),
        )
