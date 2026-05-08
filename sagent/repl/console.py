"""``ConsolePrinter``: rich-backed implementation of :class:`Printer`.

Implements the full :class:`Printer` surface using rich primitives
plus v1's existing render helpers (``print_user_bar``, ``TightMarkdown``,
``render_diff_detail``, ``set_terminal_title``). Reuse rather than
duplicate -- the formatting logic is already correct and battle-tested.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from rich.text import Text

from sagent.repl.format import (
    format_elapsed,
    print_user_bar,
    set_terminal_title,
)
from sagent.repl.render_diff import render_diff_detail
from sagent.repl.tight_markdown import TightMarkdown


if TYPE_CHECKING:
    from collections.abc import Mapping

    from rich.console import Console


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

    def write_hint(self, text: str) -> None:
        """Render a dim yellow ``hint:`` line (bash-lint nudge surface)."""
        self.console.print(Text(f"    hint: {text}", style="dim yellow"))

    def write_interrupted(self) -> None:
        """Render the dim ``[interrupted]`` line for cancelled work."""
        self.console.print(Text("[interrupted]", style="dim"))

    def write_child_event(self, label: str, descriptor: str, content: object) -> None:
        """Render one child-agent event with a labeled prefix.

        Mirrors v1's child-event rendering: tool labels and errors get
        their own prefixed lines; ``text/plain`` payloads stream as
        labeled dim lines.
        """
        pfx = f"[{label}]"
        if descriptor == "text/x-tool-label":
            self.console.print(Text(f"    {pfx} {content!s}", style="dim"))
        elif descriptor == "text/plain":
            for line in str(content).splitlines() or [""]:
                self.console.print(Text(f"    {pfx} {line}", style="dim"))
        elif descriptor == "text/x-thinking":
            self.console.print(Text(f"    {pfx} ∴ Thinking", style="italic dim"))
        elif descriptor == "text/x-error":
            self.console.print(
                Text(f"    {pfx} ✗ {str(content).strip()}", style="dim red"),
            )
        elif descriptor == "text/x-hint-tool-use-nudge":
            self.console.print(
                Text(f"    {pfx} hint: {content!s}", style="dim yellow"),
            )
        elif descriptor == "application/x-child-done":
            data = cast("Mapping[str, float]", content)
            elapsed = data.get("elapsed", 0.0)
            tokens = int(data.get("model_response_tokens", 0))
            cost = data.get("cost_usd", 0.0)
            summary_parts: list[str] = [format_elapsed(elapsed)]
            if tokens:
                summary_parts.append(f"{tokens}↓")
            if cost > 0:
                summary_parts.append(f"${cost:.2f}")
            summary = " · ".join(summary_parts)
            self.console.print(Text(f"    {pfx} done [{summary}]", style="dim"))

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
