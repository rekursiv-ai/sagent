"""``Printer`` Protocol + render observer for the REPL.

The runtime publishes ``RuntimeEvent`` instances to its observer list.
The REPL attaches a single observer (``make_render_observer``) that
translates each event into ``Printer`` calls. State that spans multiple
events (the streaming markdown buffer, child-event interleaving) lives
on the observer's closure, not on the Printer.

The Printer Protocol covers every formatted output the REPL produces:
plain lines, streaming chunks, markdown, full-width user bar, dim tool
labels, errors, thinking blocks, diffs, child gutter blocks. Concrete
implementations live in :mod:`repl.console_pane` (rich-backed) and on
:class:`RecordingPrinter` (test harness).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import logging
import time

from sagent.lib.durations import humanize_duration
from sagent.repl.render_diff import find_stable_boundary
from sagent.types.exceptions import (
    AuthRefreshError,
    ContextOverflowError,
    UserFacingError,
)
from sagent.types.runtime import (
    AssistantMessage,
    BudgetReset,
    ChildDoneEvent,
    ChildEvent,
    CompactComplete,
    CompactFailed,
    CompactStarted,
    ModelResponseCancelled,
    ModelResponseComplete,
    ModelResponseError,
    ModelResponsePartial,
    ModelResponseThinking,
    ModelServiceSuspended,
    ModelSwitchRejected,
    RuntimeEvent,
    StatusChanged,
    ToolLabel,
    ToolResult,
    ToolResultPartial,
    UserMessage,
)


logger = logging.getLogger(__name__)

HALT_MESSAGE = "agent halted — type to retry, or /login, /model, /quit"
HALT_MESSAGE_AUTH = (
    "agent halted — run /login to re-authenticate or /model to switch providers"
)
HALT_MESSAGE_CONTEXT = (
    "agent halted — run /compact <hints>, /clear, or /model to reduce context"
)


def service_suspended_text(event: ModelServiceSuspended) -> str:
    """Render concise service-suspension text for parent and child panes.

    Short waits (under a minute) get a relative-seconds display; longer
    waits show an absolute resume clock plus a humanized duration --
    the only stable representation across long parked intervals. The
    label distinguishes provider-advertised (``rate-limited``) from
    locally-chosen (``temporarily blocked``) waits.

    Args:
      event: Suspension event published by the runtime.

    Returns:
      text: ``[model service suspended: ...]`` single-line banner.

    """
    label = "rate-limited" if event.server_supplied else "temporarily blocked"
    remaining = event.retry_at - time.time()
    if remaining < 60.0:
        seconds = max(0, round(remaining))
        return f"[model service suspended: {label}; resumes in {seconds}s]"
    clock = time.strftime("%H:%M:%S", time.localtime(event.retry_at))
    return (
        f"[model service suspended: {label}; "
        f"resumes at {clock} (in {humanize_duration(remaining)})]"
    )


type ChildItem = ToolResult | UserMessage | AssistantMessage | ModelServiceSuspended


class Printer(Protocol):
    """Sink for REPL output, fully covering the rendering surface."""

    def write_line(self, text: str) -> None: ...
    def write_dim_line(self, text: str) -> None: ...
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
    def write_halt(self, text: str) -> None: ...
    def write_child_block(self, label: str, items: list[object]) -> None: ...
    def set_terminal_title(self, text: str) -> None: ...


class RecordingPrinter:
    """In-memory ``Printer`` for tests.

    Each method appends to a per-method list. Tests assert on counts and
    contents, not on rendered glyphs.
    """

    lines: list[str]
    """``write_line`` payloads."""

    dim_lines: list[str]
    """``write_dim_line`` payloads."""

    chunks: list[str]
    """``write_chunk`` payloads."""

    markdowns: list[str]
    """``write_markdown`` payloads."""

    user_bars: list[str]
    """``write_user_bar`` payloads."""

    tool_labels: list[str]
    """``write_tool_label`` payloads."""

    tool_errors: list[str]
    """``write_tool_error`` payloads."""

    tool_summaries: list[str]
    """``write_tool_summary`` payloads."""

    hints: list[str]
    """``write_hint`` payloads."""

    thinkings: list[str]
    """``write_thinking`` payloads."""

    diffs: list[tuple[str, str]]
    """``write_diff`` payloads (``(old, new)`` pairs)."""

    interruptions: int
    """Count of ``write_interrupted`` calls."""

    halts: list[str]
    """``write_halt`` payloads."""

    child_blocks: list[tuple[str, list[object]]]
    """Tuples of ``(label, items)``."""

    titles: list[str]
    """``set_terminal_title`` payloads."""

    def __init__(self) -> None:
        self.lines = []
        self.dim_lines = []
        self.chunks = []
        self.markdowns = []
        self.user_bars = []
        self.tool_labels = []
        self.tool_errors = []
        self.tool_summaries = []
        self.hints = []
        self.thinkings = []
        self.diffs = []
        self.interruptions = 0
        self.halts = []
        self.child_blocks = []
        self.titles = []

    def write_line(self, text: str) -> None:
        self.lines.append(text)

    def write_dim_line(self, text: str) -> None:
        self.dim_lines.append(text)

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

    def write_halt(self, text: str) -> None:
        self.halts.append(text)

    def write_child_block(self, label: str, items: list[object]) -> None:
        self.child_blocks.append((label, list(items)))

    def set_terminal_title(self, text: str) -> None:
        self.titles.append(text)

    @property
    def rendered_text(self) -> str:
        """Concatenated markdown payloads (the committed model output)."""
        return "".join(self.markdowns)


def render_tool_result(printer: Printer, result: ToolResult) -> None:
    """Render the user-facing parts of a :class:`ToolResult`.

    On error, only ``write_tool_error(content)`` fires. On success,
    diff / hint / summary fire in field order when populated.

    Args:
      printer: Printer to receive formatted output.
      result: Tool-result entry to render.

    """
    if result.is_error:
        printer.write_tool_error(result.content)
        return
    if result.diff:
        printer.write_diff(result.diff, result.diff_file_path)
    if result.hint:
        printer.write_hint(result.hint)
    if result.summary:
        printer.write_tool_summary(result.summary)


def make_render_observer(
    printer: Printer,
    *,
    show_thinking: Callable[[], bool] | None = None,
) -> RenderObserver:
    """Return a ``RuntimeEvent``-consuming observer bound to ``printer``.

    Args:
      printer: Printer that receives formatted output.
      show_thinking: Predicate controlling thinking display. ``None`` always shows.

    Returns:
      observer: Callable that the agent appends to ``self.observers``.

    """
    return RenderObserver(printer, show_thinking=show_thinking)


class RenderObserver:
    """Translate runtime ``RuntimeEvent`` payloads into ``Printer`` calls.

    Holds the streaming markdown buffer (paragraph-boundary detection)
    and the per-child interleave buffer (one accumulator per child label,
    flushed at stable Markdown boundaries or atomic events).
    """

    def __init__(
        self,
        printer: Printer,
        *,
        show_thinking: Callable[[], bool] | None = None,
    ) -> None:
        self._printer = printer
        self._show_thinking = show_thinking or (lambda: True)
        self._stream_buf: str = ""
        self._child_text: dict[str, str] = {}
        self._child_items: dict[str, list[object]] = {}

    def __call__(self, event: RuntimeEvent) -> None:
        try:
            self._dispatch(event)
        except Exception as e:  # noqa: BLE001 -- display safety net
            self._printer.write_tool_error(
                f"render failed for {type(event).__name__}: {type(e).__name__}: {e}",
            )
            logger.debug("render observer failed", exc_info=True)

    def _dispatch(self, event: RuntimeEvent) -> None:
        match event:
            case UserMessage(text=text):
                self._flush_stream()
                self._printer.write_user_bar(text)
            case ModelResponsePartial(text=text):
                self._feed_stream(text)
            case ModelResponseThinking(text=text):
                if self._show_thinking():
                    self._printer.write_thinking(text)
            case ModelResponseComplete():
                self._flush_stream()
            case ToolLabel(text=text):
                self._flush_stream()
                self._printer.write_tool_label(text)
            case ToolResult():
                render_tool_result(self._printer, event)
            case ToolResultPartial(text=text):
                self._printer.write_chunk(text)
            case ModelResponseCancelled():
                self._flush_stream()
                self._printer.write_interrupted()
            case ModelServiceSuspended():
                self._flush_stream()
                self._printer.write_dim_line(service_suspended_text(event))
            case ModelResponseError(exception=exc):
                self._flush_stream()
                # ``UserFacingError`` carries a polished, user-actionable
                # message; the class-name prefix would just add Python-
                # internals noise to text the user is supposed to read
                # and act on. Some user-facing errors further tailor
                # the halt banner: the generic "type to retry" misleads
                # when retrying cannot change the failed request.
                if isinstance(exc, UserFacingError):
                    self._printer.write_tool_error(str(exc))
                else:
                    self._printer.write_tool_error(f"{type(exc).__name__}: {exc}")
                if isinstance(exc, AuthRefreshError):
                    self._printer.write_halt(HALT_MESSAGE_AUTH)
                elif isinstance(exc, ContextOverflowError):
                    self._printer.write_halt(HALT_MESSAGE_CONTEXT)
                else:
                    self._printer.write_halt(HALT_MESSAGE)
            case ModelSwitchRejected(exception=exc):
                self._flush_stream()
                if isinstance(exc, UserFacingError):
                    self._printer.write_tool_error(str(exc))
                else:
                    self._printer.write_tool_error(f"{type(exc).__name__}: {exc}")
            case BudgetReset(
                model_id=model_id,
                prior_max_request_tokens=prior_in,
                new_max_request_tokens=new_in,
            ):
                self._flush_stream()
                self._printer.write_line(
                    f"[/model] budget reset to {model_id} defaults "
                    f"(max_request_tokens {prior_in:,} -> {new_in:,}); "
                    f"re-apply customised budget if needed."
                )
            case ChildEvent(label=label, inner=inner):
                self._consume_child(label, inner)
            case ChildDoneEvent(label=label):
                self._flush_child(label)
            case StatusChanged(text=text):
                self._printer.set_terminal_title(text)
            case CompactStarted():
                self._flush_stream()
                self._printer.write_dim_line("[compacting history…]")
            case CompactComplete(
                token_before=tokens_before,
                token_after=tokens_after,
                payload_entries=payload_entries,
                fallback_reason=reason,
                preserved_tail_count=count,
            ):
                self._flush_stream()
                if reason:
                    entry = "entry" if count == 1 else "entries"
                    self._printer.write_dim_line(
                        f"[compaction fallback: {reason}; preserved {count}"
                        f" tail {entry}]",
                    )
                else:
                    self._printer.write_dim_line(
                        f"[compaction complete: ~{tokens_before} →"
                        f" ~{tokens_after} tokens, {payload_entries} entries]",
                    )
            case CompactFailed(exception=exc):
                self._flush_stream()
                self._printer.write_dim_line(
                    f"[compaction failed: {type(exc).__name__}: {exc}]",
                )
            case _:
                pass

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

    def _consume_child(self, label: str, inner: RuntimeEvent) -> None:
        """Buffer one child event; flush at stable boundaries or atomic events."""
        for other in list(self._child_items):
            if other != label:
                self._emit_child(other)
        if isinstance(inner, ModelResponsePartial):
            buf = self._child_text.get(label, "") + inner.text
            boundary = find_stable_boundary(buf)
            if boundary <= 0:
                self._child_text[label] = buf
                return
            stable = buf[:boundary].rstrip("\n")
            self._child_text[label] = buf[boundary:]
            if stable:
                self._child_items.setdefault(label, []).append(
                    AssistantMessage(text=stable),
                )
                self._emit_child(label)
            return
        atomic = _child_atomic_item(inner)
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
                AssistantMessage(text=text),
            )

    def _emit_child(self, label: str) -> None:
        """Emit pending items for ``label``; leave streaming text untouched."""
        items = self._child_items.pop(label, [])
        if items:
            self._printer.write_child_block(label, items)


def _child_atomic_item(inner: RuntimeEvent) -> object | None:
    """Translate a non-streaming child event into a child-block item."""
    if isinstance(inner, ToolLabel):
        return inner
    if isinstance(inner, ModelResponseThinking):
        return inner
    if isinstance(inner, ToolResult):
        return inner
    if isinstance(inner, ModelServiceSuspended):
        return inner
    if isinstance(inner, UserMessage):
        return inner
    return None


HELP_TEXT = """\
sagent commands

  /help                       this list
  /quit, /exit                exit

  /clear                      wipe context (logs preserved on disk)
  /compact [hints]            compact history
  /recompact [hints]          alias for /compact

  /model    [args]            switch model
  /provider <name>            switch provider
  /thinking <state|partial>   adaptive/on/off/redact/show/hide
  /login                      re-auth current provider

  /tasks                      list running work (agents + fg + bg)
  /send     <target> <text>   send to subagent target: label, glob, {a,b}, /re/
  /halt     [<target>]        halt self or matching subagents (Ctrl+C)
  /kill     <qid|all|target>  cancel tool task(s) or matching subagents
  /defer    <text>            send as deferred (non-preempting); drains at ModelIdle\
"""
