"""``ConsolePrinter``: rich-backed implementation of :class:`Printer`.

Implements the full :class:`Printer` surface using rich primitives
plus v1's existing render helpers (``print_user_bar``, ``TightMarkdown``,
``render_diff_detail``, ``set_terminal_title``). Reuse rather than
duplicate -- the formatting logic is already correct and battle-tested.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Final, Literal, assert_never, cast

import io
import re

from rich.cells import chop_cells
from rich.console import Console
from rich.text import Text

from sagent.repl.format import (
    print_user_bar,
    set_terminal_title,
)
from sagent.repl.render import (
    ChildItem,
    error_text,
    render_tool_result,
    service_suspended_text,
)
from sagent.repl.render_diff import (
    highlight_source,
    render_diff_detail,
)
from sagent.repl.tight_markdown import TightMarkdown
from sagent.tools.display import (
    OutputSpec,
    ToolDisplay,
    format_output,
)
from sagent.types.runtime import (
    AgentSendMessage,
    AssistantMessage,
    ModelResponseError,
    ModelResponseThinking,
    ModelServiceSuspended,
    NoticeMessage,
    ToolLabel,
    ToolResult,
    UserMessage,
)


class ConsolePrinter:
    """Rich-backed :class:`Printer` implementation."""

    console: Console
    """Underlying rich console (exposed for one-off rich renderers that
    need to emit Text/Panel/Syntax objects)."""

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
        """Render a streaming partial without a newline.

        Skips Rich markdown / markup parsing; partials are emitted as
        raw ANSI-safe text so an unterminated ``[`` or backtick mid-
        stream doesn't trigger a parse error or visual reflow. The
        finalized block is re-rendered through :meth:`write_markdown`
        for proper formatting once the assistant turn closes.
        """
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

    def write_agent_bar(self, source: str, text: str) -> None:
        """Render an inter-agent message attributed to its source.

        Visually distinct from the user bar (which is the live human's
        input) so the reader can tell at a glance who said what. The
        ``[from <source>]:`` prefix is dim cyan to mark the
        attribution as machinery; the body is rendered with a hard
        ``reset`` so it doesn't inherit the prefix's dim attribute
        and reads at normal weight.
        """
        prefix = f"[from {source}]: "
        for line in (text or "").splitlines() or [""]:
            self.console.print(
                Text(prefix, style="dim cyan") + Text(line, style="reset")
            )

    def write_slash_block(self, text: str) -> None:
        """Render slash-command output as a tool-call-like block.

        Same dim-italic family as ``write_tool_label`` so the reader
        sees slash dispatches as machinery (not user text), but without
        the two-space indent -- slash output is at the top level of
        the user's interaction stream, not under a model turn.
        """
        for line in (text or "").splitlines() or [""]:
            self.console.print(Text(line, style="dim"))

    def write_tool_label(
        self,
        text: str,
        *,
        command: OutputSpec | None = None,
        lang: str = "",
    ) -> None:
        """Render a dim tool-call label, wrapped to the console width.

        A label whose header is followed by more lines is a tool
        reporting an INPUT -- Bash's command. Input rows carry the ``⎿``
        glyph, matching the diff receipt, so the one thing that is not
        output reads the same everywhere. Tools with a one-line label
        are unaffected.

        Tool ``summary`` implementations return their argument whole --
        the cap lives here, in the only place that knows the terminal
        width and the child-block gutter. Wrapping (rather than the
        per-tool character clamps this replaced) keeps a long command or
        query readable instead of ellipsized mid-token; the row budget
        bounds the pathological case (a heredoc, a pasted blob) so one
        call cannot flood scrollback before it even runs.

        Args:
          text: Header line, then zero or more input lines.
          command: Row budget for the input. ``None`` renders it whole,
              which is right for a one-line path or receipt.
          lang: Pygments lexer name for the input, e.g. ``"bash"``.

        """
        header, _, rest = (text or "").partition("\n")
        for line in _wrap_label(header, self.console.width - len(_CHILD_INDENT)):
            self.console.print(Text(f"{_CHILD_INDENT}{line}", style="dim"))
        if not rest:
            return
        spec = command or OutputSpec(show=True)
        rows = format_output(rest, spec, width=self.console.width - len(_INPUT_GLYPH))
        for i, row in enumerate(rows):
            marker = _INPUT_GLYPH if i == 0 else _OUTPUT_INDENT
            body = highlight_source(row, lang) if lang else Text(row)
            body.stylize("dim")
            self.console.print(Text(marker, style="dim") + body)

    def write_tool_error(self, text: str) -> None:
        """Render a red tool-error at the output indent (multi-line aware).

        Errors are output, so they sit at the output indent rather than
        in a column of their own; the ``✗`` glyph marks the first line
        and continuations align under its message. An all-blank body
        still renders a placeholder: silently swallowing the call would
        let upstream callers think the operator saw the failure.
        """
        lines = text.rstrip("\n").splitlines() or [text.rstrip("\n")]
        if not any(line.strip() for line in lines):
            lines = ["<no error message>"]
        self.console.print(Text(f"{_OUTPUT_INDENT}✗ {lines[0]}", style="dim red"))
        for line in lines[1:]:
            self.console.print(Text(f"{_OUTPUT_INDENT}  {line}", style="dim red"))

    def write_tool_summary(self, text: str) -> None:
        """Render a receipt line for a tool result.

        Shares the ``⎿`` input glyph: a receipt describes the call, not
        its output, so it belongs in the same column as the command.
        """
        self._write_input(text.strip())

    def write_tool_output(self, text: str) -> None:
        """Render a tool's result body, indented under its label.

        Output carries no glyph at all -- indentation alone separates it
        from the header, so a 20-line body does not become 20 lines of
        box-drawing noise.

        Wrapping happens here because this is the only layer that knows
        the pane width: printing an over-wide line verbatim lets Rich
        break it back to column 0, outdenting the continuation past the
        indent so it reads as top-level output.
        """
        width = self.console.width - len(_OUTPUT_INDENT)
        for raw in text.split("\n"):
            for line in _wrap_label(raw, width) or [""]:
                self.console.print(Text(f"{_OUTPUT_INDENT}{line}", style="dim"))

    def write_hint(self, text: str) -> None:
        """Render a dim yellow ``hint:`` line at the output indent."""
        for line in (text or "").split("\n"):
            self.console.print(
                Text(f"{_OUTPUT_INDENT}hint: {line}", style="dim yellow"),
            )

    def write_interrupted(self) -> None:
        """Render the dim ``[interrupted]`` line for cancelled work."""
        self.console.print(Text("[interrupted]", style="dim"))

    def write_dim_line(self, text: str) -> None:
        """Render a dim status line (compaction, etc.)."""
        self.console.print(Text(text, style="dim"))

    def write_halt(self, text: str) -> None:
        """Render a full-width red banner signaling agent-level halt."""
        width = max(20, self.console.width)
        bar = "─" * width
        self.console.print(Text(bar, style="red"))
        self.console.print(Text(text, style="bold red"))
        self.console.print(Text(bar, style="red"))

    def write_child_block(
        self,
        label: str,
        items: Sequence[ChildItem],
        *,
        output_policy: Callable[[str], ToolDisplay] | None = None,
    ) -> None:
        r"""Render a child agent's labeled block.

        Format: first line carries the label gutter (``Agent_N  :  ``),
        subsequent lines indent to align with the gutter width. Each
        item is rendered through the same paths the parent uses for
        its own output (Markdown for ``AssistantMessage`` text, dim
        text for ``ToolLabel``, the diff renderer for ``ToolResult``,
        etc.) so child output looks identical to parent output up to
        the gutter prefix.

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

        buf = io.StringIO()
        inner_console = Console(
            file=buf,
            # Never wider than what the gutter leaves: a fixed floor made
            # the composed line exceed the terminal on narrow panes, so
            # every child line wrapped raggedly in the user's console.
            width=_inner_width(self.console.width, gutter_width),
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
            _render_child_item(inner, item, output_policy=output_policy)

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

    def _write_input(self, row: str) -> None:
        """Render one input row under its tool header.

        Wrapping happens here, not in the tool: this is the only place
        that knows the pane width and the glyph width. A row too wide for
        the pane continues at the output indent, so the ``⎿`` marks the
        input once rather than repeating down the block.
        """
        width = self.console.width - len(_INPUT_GLYPH)
        for i, line in enumerate(_wrap_label(row, width) or [""]):
            marker = _INPUT_GLYPH if i == 0 else _OUTPUT_INDENT
            self.console.print(Text(f"{marker}{line}", style="dim"))


# ANSI dim attribute. ``\x1b[2m`` enables dim; ``\x1b[22m`` cancels
# only the dim attribute (vs. ``\x1b[0m`` which resets everything).
# We re-apply dim after every full reset so embedded per-span styles
# from Rich don't drop the dim baseline.
_ANSI_DIM: Final = "\x1b[2m"
_ANSI_RESET: Final = "\x1b[0m"
_ANSI_RESET_DIM = _ANSI_RESET + _ANSI_DIM
_ANSI_DIM_OFF: Final = "\x1b[22m"
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


# Two-space gutter shared by tool labels and child blocks.
_CHILD_INDENT: Final = "  "

# Marks a tool call's INPUT -- the Bash command, the Edit diff receipt.
# Output carries no glyph, so the one non-output row is the one that
# stands out. Both are five cells wide and both start from the header's
# own indent, so header, input, and body share one left edge.
_INPUT_GLYPH: Final = f"{_CHILD_INDENT}\u23bf  "
_OUTPUT_INDENT: Final = f"{_CHILD_INDENT}   "

# Rendered-line cap for one tool label. Labels carry the argument whole
# (a command, a query, a pasted prompt), so a heredoc or blob would
# otherwise push the surrounding scrollback off-screen before the tool
# has even run. Head-biased: the opening of a command identifies it.
_LABEL_MAX_LINES: Final = 12


def _wrap_label(text: str, width: int) -> list[str]:
    """Wrap ``text`` to ``width`` cells, capped at ``_LABEL_MAX_LINES``.

    Args:
      text: Raw label text; may contain newlines.
      width: Usable cell width after the indent.

    Returns:
      lines: Wrapped lines, with a trailing ``... (N more lines)`` marker
          when the cap elided content.

    """
    usable = max(1, width)
    out: list[str] = []
    for raw in (text or "").splitlines() or [""]:
        out.extend(chop_cells(raw, usable) or [""])
    if len(out) <= _LABEL_MAX_LINES:
        return out
    # The marker occupies one of the capped lines rather than sitting
    # past the cap, so the rendered block never exceeds the limit the
    # constant advertises.
    kept = _LABEL_MAX_LINES - 1
    return [*out[:kept], f"... ({len(out) - kept} more lines)"]


def _inner_width(outer_width: int, gutter_width: int) -> int:
    """Usable width for a child block's inner console.

    The composed line is ``gutter + inner``, so the inner console must
    never claim more than the terminal leaves. The prior fixed floor of
    20 columns did exactly that on a narrow pane: with a 14-column
    gutter, anything under 34 columns overran the terminal and every
    child line wrapped raggedly.

    Args:
      outer_width: Width of the enclosing console.
      gutter_width: Columns consumed by the child-label gutter.

    Returns:
      width: Inner console width, at least 1 and never overrunning.

    """
    return max(1, outer_width - gutter_width)


def _gutter_width(label: str) -> int:
    """Width of the gutter column for a given child label.

    Format is ``"  <label>  :  "`` -- leading 2-space indent matching
    parent tool labels, then ``len(label) + 5`` for ``"  :  "``. Held
    to a floor of 14 columns for visual consistency when labels are
    short (``Agent_0`` is the typical case at 14 chars exactly).
    """
    return max(14, len(label) + 5 + len(_CHILD_INDENT))


def _gutter_prefix(label: str, width: int) -> str:
    """Render the first-line gutter for ``label`` padded to ``width``."""
    pfx = f"{_CHILD_INDENT}{label}  :  "
    if len(pfx) < width:
        pfx = pfx.ljust(width)
    return pfx


def _render_child_item(
    printer: ConsolePrinter,
    item: ChildItem,
    *,
    output_policy: Callable[[str], ToolDisplay] | None = None,
) -> None:
    """Dispatch one child-block item to the appropriate printer method.

    Exhaustive over :data:`repl.render.ChildItem`; ``assert_never`` makes
    the type checker flag any new variant that forgets a branch here.

    ``output_policy`` resolves the child tool's display settings. Without
    it a subagent's Bash body never renders even with ``output=on``,
    because the missing policy reads as hidden.
    """
    match item:
        case AssistantMessage(text=text):
            if text:
                printer.write_markdown(text)
        case ToolLabel(text=text):
            display = output_policy(item.call_id) if output_policy is not None else None
            printer.write_tool_label(
                text,
                command=display.command if display else None,
                lang=display.command_lang if display else "",
            )
        case ModelResponseThinking(text=text):
            printer.write_thinking(text)
        case ModelServiceSuspended():
            printer.write_dim_line(service_suspended_text(item))
        case NoticeMessage(text=text):
            printer.write_dim_line(text)
        case ModelResponseError(exception=exc):
            printer.write_tool_error(error_text(exc))
        case ToolResult():
            render_tool_result(
                printer,
                item,
                output=(
                    output_policy(item.call_id).output
                    if output_policy is not None
                    else None
                ),
            )
        case AgentSendMessage(source=source, text=text):
            printer.write_agent_bar(source, text)
        case UserMessage(text=text):
            printer.write_user_bar(text)
        case _:
            assert_never(item)
