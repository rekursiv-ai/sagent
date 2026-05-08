"""Render handlers + ``Printer`` abstraction for the v2 REPL.

The Printer Protocol covers every formatted output the REPL produces:
plain lines, streaming chunks, markdown, full-width user-bar, dim
tool labels, errors, thinking blocks, diffs. Each method has a
test-friendly recording variant on :class:`RecordingPrinter` and a
rich-backed implementation on :class:`ConsolePrinter` (in
``console.py``).

Handlers translate inbox descriptors into Printer calls. State that
spans multiple events (the streaming buffer, the "header printed"
flag) lives on the handler instance, not on the Printer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast, override

from sagent.agent.handlers.base import Handler, InlineHandler
from sagent.agent.handlers.model_switch import ModelSwitchHandler
from sagent.lib.message import get_queue_id
from sagent.repl.input import HelpHandler, LoginHandler, TasksHandler
from sagent.repl.render_diff import find_stable_boundary


if TYPE_CHECKING:
    from sagent.agent.agent import Agent
    from sagent.custom_types import Message


class Printer(Protocol):
    """Sink for REPL output, fully covering v1 rendering surface.

    Method semantics:

    - ``write_line(text)`` -- complete line, terminated by a newline.
    - ``write_chunk(text)`` -- streaming partial; no trailing newline.
    - ``write_markdown(text)`` -- render as Markdown (rich
      ``TightMarkdown`` in production; plain capture in tests).
    - ``write_user_bar(text)`` -- full-width dark-gray user message bar.
    - ``write_tool_label(text)`` -- dim multi-line tool-call label.
    - ``write_tool_error(text)`` -- red tool-error line.
    - ``write_thinking(text)`` -- italic dim "Thinking" preface + body.
    - ``write_diff(diff, file_path)`` -- syntax-highlighted unified diff.
    - ``set_terminal_title(text)`` -- OSC 0 title escape (no-op off TTY).

    Render handlers know what to format; the printer knows how to
    display.
    """

    def write_line(self, text: str) -> None: ...
    def write_chunk(self, text: str) -> None: ...
    def write_markdown(self, text: str) -> None: ...
    def write_user_bar(self, text: str) -> None: ...
    def write_tool_label(self, text: str) -> None: ...
    def write_tool_error(self, text: str) -> None: ...
    def write_hint(self, text: str) -> None: ...
    def write_thinking(self, text: str) -> None: ...
    def write_diff(self, diff: str, file_path: str = "") -> None: ...
    def write_interrupted(self) -> None: ...
    def write_child_event(
        self, label: str, descriptor: str, content: object
    ) -> None: ...
    def set_terminal_title(self, text: str) -> None: ...


class RecordingPrinter:
    """In-memory printer for tests.

    Each method appends to a per-method list. Tests assert on counts
    and contents, not on rendered glyphs.

    Attributes:
      lines: ``write_line`` payloads.
      chunks: ``write_chunk`` payloads.
      markdowns: ``write_markdown`` payloads.
      user_bars: ``write_user_bar`` payloads.
      tool_labels: ``write_tool_label`` payloads.
      tool_errors: ``write_tool_error`` payloads.
      hints: ``write_hint`` payloads.
      thinkings: ``write_thinking`` payloads.
      diffs: ``write_diff`` payloads.
      interruptions: count of ``write_interrupted`` calls.
      child_events: tuples of ``(label, descriptor, content)``.
      titles: ``set_terminal_title`` payloads.

    """

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.chunks: list[str] = []
        self.markdowns: list[str] = []
        self.user_bars: list[str] = []
        self.tool_labels: list[str] = []
        self.tool_errors: list[str] = []
        self.hints: list[str] = []
        self.thinkings: list[str] = []
        self.diffs: list[tuple[str, str]] = []
        self.interruptions: int = 0
        self.child_events: list[tuple[str, str, object]] = []
        self.titles: list[str] = []

    def write_line(self, text: str) -> None:
        self.lines.append(text)

    def write_chunk(self, text: str) -> None:
        self.chunks.append(text)

    def write_markdown(self, text: str) -> None:
        self.markdowns.append(text)

    def write_user_bar(self, text: str) -> None:
        self.user_bars.append(text)

    def write_tool_label(self, text: str) -> None:
        self.tool_labels.append(text)

    def write_tool_error(self, text: str) -> None:
        self.tool_errors.append(text)

    def write_hint(self, text: str) -> None:
        self.hints.append(text)

    def write_thinking(self, text: str) -> None:
        self.thinkings.append(text)

    def write_diff(self, diff: str, file_path: str = "") -> None:
        self.diffs.append((diff, file_path))

    def write_interrupted(self) -> None:
        self.interruptions += 1

    def write_child_event(self, label: str, descriptor: str, content: object) -> None:
        self.child_events.append((label, descriptor, content))

    def set_terminal_title(self, text: str) -> None:
        self.titles.append(text)

    @property
    def rendered_text(self) -> str:
        """Concatenated markdown payloads (the committed model output)."""
        return "".join(self.markdowns)


class RenderUserBar(InlineHandler):
    """Render a user message as a full-width bar."""

    descriptors: tuple[str, ...] = ("text/x-user-message",)

    def __init__(self, printer: Printer) -> None:
        self._printer = printer

    @override
    async def handle(self, agent: Agent, msg: Message) -> None:
        del agent
        self._printer.write_user_bar(str(msg.content))


class RenderStream(InlineHandler):
    """Buffer streaming chunks; render Markdown blocks at stable boundaries.

    Mirrors v1's incremental-markdown pattern: accumulate chunks,
    look for a complete-block boundary, render the stable prefix as
    Markdown to the scrollback, keep the unstable tail buffering. On
    ``text/x-stream-end`` flush whatever remains.
    """

    descriptors: tuple[str, ...] = (
        "text/plain",
        "text/x-stream-end",
    )

    def __init__(self, printer: Printer) -> None:
        self._printer = printer
        self._buf: str = ""

    @override
    async def handle(self, agent: Agent, msg: Message) -> None:
        del agent
        if msg.descriptor == "text/x-stream-end":
            self._flush()
            return
        self._buf += str(msg.content)
        boundary = find_stable_boundary(self._buf)
        if boundary <= 0:
            return
        stable = self._buf[:boundary].rstrip("\n")
        self._buf = self._buf[boundary:]
        if stable:
            self._printer.write_markdown(stable)

    def _flush(self) -> None:
        """Render any remaining buffer and reset state."""
        remaining = self._buf.rstrip("\n")
        self._buf = ""
        if remaining:
            self._printer.write_markdown(remaining)


class RenderToolLabel(InlineHandler):
    """Render dim tool-call label lines."""

    descriptors: tuple[str, ...] = ("text/x-tool-label",)

    def __init__(self, printer: Printer) -> None:
        self._printer = printer

    @override
    async def handle(self, agent: Agent, msg: Message) -> None:
        del agent
        self._printer.write_tool_label(str(msg.content))


class RenderToolResult(InlineHandler):
    """Render relevant parts of a ``multipart/x-tool-result``.

    Errors via ``write_tool_error``; diffs via ``write_diff``;
    bash-lint nudges (``text/x-hint-tool-use-nudge``) via the dim
    yellow ``hint:`` line that v1 surfaces. Plain text parts are
    skipped here -- they belong to history, not user feedback (the
    model already saw them).
    """

    descriptors: tuple[str, ...] = ("multipart/x-tool-result",)

    def __init__(self, printer: Printer) -> None:
        self._printer = printer

    @override
    async def handle(self, agent: Agent, msg: Message) -> None:
        del agent
        for part in cast("tuple[Message, ...]", msg.content):
            if part.descriptor == "text/x-error":
                self._printer.write_tool_error(str(part.content))
            elif part.descriptor == "text/x-diff" and part.content:
                self._printer.write_diff(str(part.content), get_queue_id(msg))
            elif part.descriptor == "text/x-hint-tool-use-nudge" and part.content:
                self._printer.write_hint(str(part.content))


class RenderError(InlineHandler):
    """Print error events via the tool-error sink (red, indented)."""

    descriptors: tuple[str, ...] = ("text/x-error",)

    def __init__(self, printer: Printer) -> None:
        self._printer = printer

    @override
    async def handle(self, agent: Agent, msg: Message) -> None:
        del agent
        self._printer.write_tool_error(str(msg.content))


class RenderInterrupted(InlineHandler):
    """Render the dim ``[interrupted]`` line for cancelled / aborted work."""

    descriptors: tuple[str, ...] = ("text/x-interrupted",)

    def __init__(self, printer: Printer) -> None:
        self._printer = printer

    @override
    async def handle(self, agent: Agent, msg: Message) -> None:
        del agent, msg
        self._printer.write_interrupted()


class RenderChildEvent(InlineHandler):
    """Render labeled child-agent events from ``multipart/x-child-event``.

    The envelope carries ``(label_msg, inner_msg, ...)`` where the first
    part is a ``text/plain`` label and the rest are zero-or-more inner
    events (tool labels, plain text, errors). Only display-relevant
    inner descriptors are rendered.
    """

    descriptors: tuple[str, ...] = ("multipart/x-child-event",)

    def __init__(self, printer: Printer) -> None:
        self._printer = printer

    @override
    async def handle(self, agent: Agent, msg: Message) -> None:
        del agent
        parts = cast("tuple[Message, ...]", msg.content)
        if len(parts) < 2:
            return
        label = str(parts[0].content)
        for inner in parts[1:]:
            self._printer.write_child_event(label, inner.descriptor, inner.content)


class RenderThinking(InlineHandler):
    """Render thinking blocks (italic dim header + body)."""

    descriptors: tuple[str, ...] = ("text/x-thinking",)

    def __init__(self, printer: Printer) -> None:
        self._printer = printer

    @override
    async def handle(self, agent: Agent, msg: Message) -> None:
        del agent
        self._printer.write_thinking(str(msg.content))


class RenderStatusTitle(InlineHandler):
    """Reflect ``text/x-status-update`` messages onto the terminal title."""

    descriptors: tuple[str, ...] = ("text/x-status-update",)

    def __init__(self, printer: Printer) -> None:
        self._printer = printer

    @override
    async def handle(self, agent: Agent, msg: Message) -> None:
        del agent
        self._printer.set_terminal_title(str(msg.content))


def repl_handler_set(printer: Printer) -> list[Handler]:
    """Return the standard render handler bundle bound to ``printer``.

    Includes ``ModelSwitchHandler`` (handles ``/model`` slash command)
    -- it lives here rather than in ``core_handlers`` because it
    needs a printer to surface status output.

    Args:
      printer: Printer that receives formatted output.

    Returns:
      handlers: Render handlers ready to register on ``Agent``.

    """
    return [
        RenderUserBar(printer),
        RenderStream(printer),
        RenderThinking(printer),
        RenderToolLabel(printer),
        RenderToolResult(printer),
        RenderError(printer),
        RenderInterrupted(printer),
        RenderChildEvent(printer),
        RenderStatusTitle(printer),
        ModelSwitchHandler(printer=printer),
        LoginHandler(printer=printer),
        HelpHandler(printer=printer),
        TasksHandler(printer=printer),
    ]
