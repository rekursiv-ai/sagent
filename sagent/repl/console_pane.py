"""``ConsolePrinter``: rich-backed implementation of :class:`Printer`.

Implements the full :class:`Printer` surface using rich primitives
plus v1's existing render helpers (``print_user_bar``, ``TightMarkdown``,
``render_diff_detail``, ``set_terminal_title``). Reuse rather than
duplicate -- the formatting logic is already correct and battle-tested.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, assert_never, cast

import io
import re

from rich.console import Console
from rich.text import Text

from sagent.repl.format import (
    print_user_bar,
    set_terminal_title,
)
from sagent.repl.render import (
    ChildItem,
    render_tool_result,
    service_suspended_text,
)
from sagent.repl.render_diff import render_diff_detail
from sagent.repl.tight_markdown import TightMarkdown
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

    def write_tool_label(self, text: str) -> None:
        """Render dim multi-line tool-call label."""
        for line in (text or "").splitlines() or [""]:
            self.console.print(Text(f"  {line}", style="dim"))

    def write_tool_error(self, text: str) -> None:
        """Render red, indented tool-error (multi-line aware).

        First line gets the ``✗`` glyph; subsequent lines align with the
        message column so structured traces stay readable. An all-blank
        body still renders a placeholder line: silently swallowing the
        call would let upstream callers think the operator saw the
        failure.
        """
        lines = text.rstrip("\n").splitlines() or [text.rstrip("\n")]
        if not any(line.strip() for line in lines):
            lines = ["<no error message>"]
        self.console.print(Text(f"    ✗ {lines[0]}", style="dim red"))
        for line in lines[1:]:
            self.console.print(Text(f"      {line}", style="dim red"))

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

    def write_child_block(self, label: str, items: Sequence[ChildItem]) -> None:
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


_CHILD_INDENT = "  "


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


def _render_child_item(printer: ConsolePrinter, item: ChildItem) -> None:
    """Dispatch one child-block item to the appropriate printer method.

    Exhaustive over :data:`repl.render.ChildItem`; ``assert_never`` makes
    the type checker flag any new variant that forgets a branch here.
    """
    match item:
        case AssistantMessage(text=text):
            if text:
                printer.write_markdown(text)
        case ToolLabel(text=text):
            printer.write_tool_label(text)
        case ModelResponseThinking(text=text):
            printer.write_thinking(text)
        case ModelServiceSuspended():
            printer.write_dim_line(service_suspended_text(item))
        case NoticeMessage(text=text):
            printer.write_dim_line(text)
        case ModelResponseError(exception=exc):
            printer.write_tool_error(f"{type(exc).__name__}: {exc}")
        case ToolResult():
            render_tool_result(printer, item)
        case AgentSendMessage(source=source, text=text):
            printer.write_agent_bar(source, text)
        case UserMessage(text=text):
            printer.write_user_bar(text)
        case _:
            assert_never(item)
