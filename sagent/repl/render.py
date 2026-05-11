"""``Printer`` Protocol + render observer for the v3 REPL.

The agent publishes ``Event`` instances to its observer list. The REPL
attaches a single observer (``make_render_observer``) that translates
each event into ``Printer`` calls. State that spans multiple events
(the streaming markdown buffer, child-event interleaving) lives on the
observer's closure, not on the Printer.

The Printer Protocol covers every formatted output the REPL produces:
plain lines, streaming chunks, markdown, full-width user bar, dim tool
labels, errors, thinking blocks, diffs, child gutter blocks. Concrete
implementations live in :mod:`repl.console` (rich-backed) and on
:class:`RecordingPrinter` (test harness).
"""

from __future__ import annotations

from typing import Protocol, cast

import logging

from sagent.custom_types import (
    ChildDoneEvent,
    ChildEvent,
    ErrorEvent,
    Event,
    InterruptedEvent,
    Message,
    StatusUpdateEvent,
    StreamEndEvent,
    TextChunkEvent,
    TextMessage,
    ThinkingEvent,
    ToolLabelEvent,
    ToolResultEvent,
    UserBarEvent,
)
from sagent.lib.message import get_queue_id
from sagent.repl.render_diff import find_stable_boundary


logger = logging.getLogger(__name__)


class Printer(Protocol):
    """Sink for REPL output, fully covering the v2/v3 rendering surface."""

    def write_line(self, text: str) -> None: ...
    def write_chunk(self, text: str) -> None: ...
    def write_markdown(self, text: str) -> None: ...
    def write_user_bar(self, text: str) -> None: ...
    def write_tool_label(self, text: str) -> None: ...
    def write_tool_error(self, text: str) -> None: ...
    def write_tool_summary(self, text: str) -> None: ...
    def write_hint(self, text: str) -> None: ...
    def write_thinking(self, text: str) -> None: ...
    def write_diff(self, diff: str, file_path: str = "") -> None: ...
    def write_interrupted(self) -> None: ...
    def write_child_block(self, label: str, items: list[Message]) -> None: ...
    def set_terminal_title(self, text: str) -> None: ...


class RecordingPrinter:
    """In-memory ``Printer`` for tests.

    Each method appends to a per-method list. Tests assert on counts and
    contents, not on rendered glyphs.

    Attributes:
      lines: ``write_line`` payloads.
      chunks: ``write_chunk`` payloads.
      markdowns: ``write_markdown`` payloads.
      user_bars: ``write_user_bar`` payloads.
      tool_labels: ``write_tool_label`` payloads.
      tool_errors: ``write_tool_error`` payloads.
      tool_summaries: ``write_tool_summary`` payloads.
      hints: ``write_hint`` payloads.
      thinkings: ``write_thinking`` payloads.
      diffs: ``write_diff`` payloads.
      interruptions: count of ``write_interrupted`` calls.
      child_blocks: tuples of ``(label, items)``.
      titles: ``set_terminal_title`` payloads.

    """

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.chunks: list[str] = []
        self.markdowns: list[str] = []
        self.user_bars: list[str] = []
        self.tool_labels: list[str] = []
        self.tool_errors: list[str] = []
        self.tool_summaries: list[str] = []
        self.hints: list[str] = []
        self.thinkings: list[str] = []
        self.diffs: list[tuple[str, str]] = []
        self.interruptions: int = 0
        self.child_blocks: list[tuple[str, list[Message]]] = []
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

    def write_tool_summary(self, text: str) -> None:
        self.tool_summaries.append(text)

    def write_hint(self, text: str) -> None:
        self.hints.append(text)

    def write_thinking(self, text: str) -> None:
        self.thinkings.append(text)

    def write_diff(self, diff: str, file_path: str = "") -> None:
        self.diffs.append((diff, file_path))

    def write_interrupted(self) -> None:
        self.interruptions += 1

    def write_child_block(self, label: str, items: list[Message]) -> None:
        self.child_blocks.append((label, list(items)))

    def set_terminal_title(self, text: str) -> None:
        self.titles.append(text)

    @property
    def rendered_text(self) -> str:
        """Concatenated markdown payloads (the committed model output)."""
        return "".join(self.markdowns)


def render_tool_result(printer: Printer, msg: Message) -> None:
    """Render the user-facing parts of a ``multipart/x-tool-result`` Message.

    Errors via ``write_tool_error``; diffs via ``write_diff``;
    one-line success summaries via ``write_tool_summary``;
    bash-lint nudges (``text/x-hint-tool-use-nudge``) via ``write_hint``.

    Args:
      printer: Printer to receive formatted output.
      msg: Multipart tool-result message to render.

    """
    for part in cast("tuple[Message, ...]", msg.content):
        if part.descriptor == "text/x-error":
            printer.write_tool_error(str(part.content))
        elif part.descriptor == "text/x-diff" and part.content:
            printer.write_diff(str(part.content), get_queue_id(msg))
        elif part.descriptor == "text/x-tool-summary" and part.content:
            printer.write_tool_summary(str(part.content))
        elif part.descriptor == "text/x-hint-tool-use-nudge" and part.content:
            printer.write_hint(str(part.content))


def make_render_observer(printer: Printer) -> RenderObserver:
    """Return an ``Event``-consuming observer bound to ``printer``.

    Args:
      printer: Printer that receives formatted output.

    Returns:
      observer: Callable that the agent appends to ``self.observers``.

    """
    return RenderObserver(printer)


class RenderObserver:
    """Translate agent ``Event`` payloads into ``Printer`` calls.

    Holds the streaming markdown buffer (paragraph-boundary detection)
    and the per-child interleave buffer (one accumulator per child label,
    flushed at stable Markdown boundaries or atomic events).
    """

    def __init__(self, printer: Printer) -> None:
        self._printer = printer
        self._stream_buf: str = ""
        self._child_text: dict[str, str] = {}
        self._child_items: dict[str, list[Message]] = {}

    def __call__(self, event: Event) -> None:
        try:
            self._dispatch(event)
        except Exception as e:  # noqa: BLE001 -- display safety net
            self._printer.write_tool_error(
                f"render failed for {type(event).__name__}: {type(e).__name__}: {e}",
            )
            logger.debug("render observer failed", exc_info=True)

    def _dispatch(self, event: Event) -> None:
        if isinstance(event, UserBarEvent):
            self._printer.write_user_bar(event.text)
        elif isinstance(event, TextChunkEvent):
            self._feed_stream(event.text)
        elif isinstance(event, ThinkingEvent):
            self._printer.write_thinking(event.text)
        elif isinstance(event, ToolLabelEvent):
            self._printer.write_tool_label(event.text)
        elif isinstance(event, ToolResultEvent):
            render_tool_result(self._printer, event.msg)
        elif isinstance(event, StreamEndEvent):
            self._flush_stream()
        elif isinstance(event, ErrorEvent):
            self._printer.write_tool_error(event.text)
        elif isinstance(event, InterruptedEvent):
            self._printer.write_interrupted()
        elif isinstance(event, StatusUpdateEvent):
            self._printer.set_terminal_title(event.text)
        elif isinstance(event, ChildEvent):
            self._consume_child(event.label, event.inner)
        elif isinstance(event, ChildDoneEvent):
            self._flush_child(event.label)
        # TurnCompleteEvent: rendered via TextChunkEvent + StreamEndEvent; no extra output.

    def _feed_stream(self, chunk: str) -> None:
        """Buffer streaming text; flush stable Markdown blocks."""
        self._stream_buf += chunk
        boundary = find_stable_boundary(self._stream_buf)
        if boundary <= 0:
            return
        stable = self._stream_buf[:boundary].rstrip("\n")
        self._stream_buf = self._stream_buf[boundary:]
        if stable:
            self._printer.write_markdown(stable)

    def _flush_stream(self) -> None:
        """Render any remaining streaming text and reset the buffer."""
        remaining = self._stream_buf.rstrip("\n")
        self._stream_buf = ""
        if remaining:
            self._printer.write_markdown(remaining)

    def _consume_child(self, label: str, inner: Event) -> None:
        """Buffer one child event; flush at stable boundaries or atomic events."""
        for other in list(self._child_items):
            if other != label:
                self._emit_child(other)
        if isinstance(inner, TextChunkEvent):
            self._child_text[label] = self._child_text.get(label, "") + inner.text
            buf = self._child_text[label]
            boundary = find_stable_boundary(buf)
            if boundary <= 0:
                return
            stable = buf[:boundary].rstrip("\n")
            self._child_text[label] = buf[boundary:]
            if stable:
                self._child_items.setdefault(label, []).append(
                    TextMessage(stable, "text/plain"),
                )
                self._emit_child(label)
            return
        atomic = _child_atomic_message(inner)
        if atomic is None:
            return
        self._move_text_to_items(label)
        self._child_items.setdefault(label, []).append(atomic)
        self._emit_child(label)

    def _flush_child(self, label: str) -> None:
        """Force-flush a child label's buffered events."""
        self._move_text_to_items(label)
        self._emit_child(label)

    def _move_text_to_items(self, label: str) -> None:
        """Move accumulated streaming text into the items list."""
        text = self._child_text.pop(label, "").rstrip("\n")
        if text:
            self._child_items.setdefault(label, []).append(
                TextMessage(text, "text/plain"),
            )

    def _emit_child(self, label: str) -> None:
        """Emit pending items for ``label``; leave streaming text untouched."""
        items = self._child_items.pop(label, [])
        if items:
            self._printer.write_child_block(label, items)


def _child_atomic_message(inner: Event) -> Message | None:
    """Translate non-streaming child events to a Message for child-block rendering.

    Args:
      inner: Inner event from a ``ChildEvent``.

    Returns:
      msg: Message the printer's ``write_child_block`` knows how to render,
          or ``None`` if the event has no in-block representation.

    """
    if isinstance(inner, ToolLabelEvent):
        return TextMessage(inner.text, "text/x-tool-label")
    if isinstance(inner, ThinkingEvent):
        return TextMessage(inner.text, "text/x-thinking")
    if isinstance(inner, ErrorEvent):
        return TextMessage(inner.text, "text/x-error")
    if isinstance(inner, ToolResultEvent):
        return inner.msg
    return None


HELP_TEXT = """\
sagent commands

  /help                       this list
  /quit                       exit

  /clear                      wipe context (logs preserved on disk)
  /compact [hints]            compact history
  /recompact [hints]          re-run the most recent compaction

  /model    [args]            switch model
  /provider <name>            switch provider
  /login                      re-auth current provider

  /tasks                      list running work (agents + fg + bg)
  /break    [<label>|all]     cancel current step          (Ctrl+Z analog)
  /abort    [<label>|all]     cancel step + queue          (Ctrl+C analog)
                              "all" also kills background tasks\
"""
