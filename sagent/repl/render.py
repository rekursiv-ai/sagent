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

from collections.abc import Callable, Generator, Sequence
from typing import Final, Protocol

import contextlib
import contextvars
import logging
import re
import time

from sagent.lib.durations import humanize_duration
from sagent.repl.render_diff import find_stable_boundary
from sagent.tools.display import (
    OutputSpec,
    ToolDisplay,
    format_output,
)
from sagent.types.exceptions import (
    AuthRefreshError,
    ContextOverflowError,
    UserFacingError,
)
from sagent.types.runtime import (
    AgentSendDeferredMessage,
    AgentSendMessage,
    AgentSendQueuedMessage,
    AssistantMessage,
    BudgetReset,
    ChildDoneEvent,
    ChildEvent,
    CompactComplete,
    CompactFailed,
    CompactStarted,
    DetachedResult,
    ModelResponseCancelled,
    ModelResponseComplete,
    ModelResponseError,
    ModelResponsePartial,
    ModelResponseThinking,
    ModelServiceSuspended,
    ModelSwitchRejected,
    NoticeMessage,
    RuntimeEvent,
    SessionMessage,
    StatusChanged,
    ToolLabel,
    ToolResult,
    ToolResultKind,
    ToolResultPartial,
    UserMessage,
)


logger = logging.getLogger(__name__)

# Streaming text without a stable Markdown boundary is held in
# ``_stream_buf`` until the next paragraph break. A 100K+-char in-progress
# fenced block would let the buffer grow for the entire round; flush
# unconditionally past this cap so memory stays bounded.
_STREAM_BUF_FLUSH_BYTES = (
    64 * 1024
)  # config-globals: ignore -- stream-buf flush cap, display pref

# Live-observer dispatch is wrapped in a broad ``except`` so a renderer
# bug never tears down the agent loop. That same swallow hides real
# failures from unit tests that legitimately want the exception to
# surface. Tests flip this ``ContextVar`` to True via the
# :func:`strict_observer` context manager so dispatch failures re-raise
# instead of getting logged.
_strict_observer: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "repl_render_strict_observer", default=False
)


@contextlib.contextmanager
def strict_observer() -> Generator[None, None, None]:
    """Make ``RenderObserver.__call__`` re-raise dispatch failures.

    Yields:
      None: enters the strict scope for the duration of the ``with``.

    """
    token = _strict_observer.set(True)
    try:
        yield
    finally:
        _strict_observer.reset(token)


HALT_MESSAGE: Final = "agent halted -- type to retry, or /login, /model, /quit"
HALT_MESSAGE_AUTH: Final = (
    "agent halted -- run /login to re-authenticate or /model to switch providers"
)
HALT_MESSAGE_CONTEXT: Final = (
    "agent halted -- run /compact <hints>, /clear, or /model to reduce context"
)


_SUSPENSION_REASON_CHARS: Final = 120
"""Cap on the provider reason in the one-line suspension banner.

An unbounded message wraps across the pane and pushes the resume time
out of view -- the one thing the banner exists to show.
"""


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
    # The provider's own message is the only thing that distinguishes a
    # wait worth sitting through from one that never clears (e.g. "Usage
    # credits are required for fast mode."). Dropping it left the user
    # watching escalating backoffs with no way to learn why.
    reason = " ".join(event.error.message.split())
    if len(reason) > _SUSPENSION_REASON_CHARS:
        reason = reason[: _SUSPENSION_REASON_CHARS - 1].rstrip() + "\u2026"
    detail = f"{label}: {reason}" if reason else label
    remaining = event.retry_at - time.time()
    if remaining < 60.0:
        seconds = max(0, round(remaining))
        return f"[model service suspended: {detail}; resumes in {seconds}s]"
    clock = time.strftime("%H:%M:%S", time.localtime(event.retry_at))
    return (
        f"[model service suspended: {detail}; "
        f"resumes at {clock} (in {humanize_duration(remaining)})]"
    )


type ChildItem = (
    ToolLabel
    | ModelResponseThinking
    | ToolResult
    | ModelServiceSuspended
    | NoticeMessage
    | ModelResponseError
    | AgentSendMessage
    | UserMessage
    | AssistantMessage
)
"""Items accumulated under a child label for rendering in one block.

Mirrors what ``_child_atomic_item`` returns (atomic forwards from the
child runtime), plus the synthetic ``AssistantMessage`` the observer
materializes from streamed ``ModelResponsePartial`` text on flush.
"""


class Printer(Protocol):
    """Sink for REPL output, fully covering the rendering surface."""

    def write_line(self, text: str) -> None: ...
    def write_dim_line(self, text: str) -> None: ...
    def write_chunk(self, text: str) -> None: ...
    def write_markdown(self, text: str) -> None: ...
    def write_user_bar(self, text: str) -> None: ...
    def write_agent_bar(self, source: str, text: str) -> None: ...
    def write_slash_block(self, text: str) -> None: ...
    def write_tool_label(
        self,
        text: str,
        *,
        command: OutputSpec | None = None,
        lang: str = "",
    ) -> None: ...
    def write_tool_error(self, text: str) -> None: ...
    def write_tool_summary(self, text: str) -> None: ...
    def write_tool_output(self, text: str) -> None: ...
    def write_hint(self, text: str) -> None: ...
    def write_thinking(self, text: str) -> None: ...
    def write_diff(self, diff: str, file_path: str = "") -> None: ...
    def write_interrupted(self) -> None: ...
    def write_halt(self, text: str) -> None: ...
    def write_child_block(
        self,
        label: str,
        items: Sequence[ChildItem],
        *,
        output_policy: Callable[[str], ToolDisplay] | None = None,
    ) -> None: ...
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

    agent_bars: list[tuple[str, str]]
    """``write_agent_bar`` payloads as ``(source, text)`` tuples."""

    slash_blocks: list[str]
    """``write_slash_block`` payloads."""

    tool_labels: list[str]
    """``write_tool_label`` payloads."""

    tool_errors: list[str]
    """``write_tool_error`` payloads."""

    tool_summaries: list[str]
    """``write_tool_summary`` payloads."""

    tool_outputs: list[str]
    """``write_tool_output`` payloads."""

    hints: list[str]
    """``write_hint`` payloads."""

    thinkings: list[str]
    """``write_thinking`` payloads."""

    diffs: list[tuple[str, str]]
    """``write_diff`` payloads as ``(diff, file_path)`` tuples."""

    interruptions: int
    """Count of ``write_interrupted`` calls."""

    halts: list[str]
    """``write_halt`` payloads."""

    child_blocks: list[tuple[str, list[ChildItem]]]
    """Tuples of ``(label, items)``."""

    titles: list[str]
    """``set_terminal_title`` payloads."""

    def __init__(self) -> None:
        self.lines = []
        self.dim_lines = []
        self.chunks = []
        self.markdowns = []
        self.user_bars = []
        self.agent_bars = []
        self.slash_blocks = []
        self.tool_labels = []
        self.tool_errors = []
        self.tool_summaries = []
        self.tool_outputs = []
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

    def write_agent_bar(self, source: str, text: str) -> None:
        self.agent_bars.append((source, text))

    def write_slash_block(self, text: str) -> None:
        self.slash_blocks.append(text)

    def write_tool_label(
        self,
        text: str,
        *,
        command: OutputSpec | None = None,
        lang: str = "",
    ) -> None:
        del command, lang
        self.tool_labels.append(text)

    def write_tool_error(self, text: str) -> None:
        self.tool_errors.append(text)

    def write_tool_summary(self, text: str) -> None:
        self.tool_summaries.append(text)

    def write_tool_output(self, text: str) -> None:
        self.tool_outputs.append(text)

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

    def write_child_block(
        self,
        label: str,
        items: Sequence[ChildItem],
        *,
        output_policy: Callable[[str], ToolDisplay] | None = None,
    ) -> None:
        del output_policy
        self.child_blocks.append((label, list(items)))

    def set_terminal_title(self, text: str) -> None:
        self.titles.append(text)

    @property
    def rendered_text(self) -> str:
        """Concatenated markdown payloads (the committed model output)."""
        return "".join(self.markdowns)


def render_tool_result(
    printer: Printer,
    result: ToolResult,
    *,
    output: OutputSpec | None = None,
) -> None:
    """Render the user-facing parts of a :class:`ToolResult`.

    Errors and hints always render; the body renders only when the
    owning tool asked for it. A ``PENDING`` stub renders no body: it
    stands in for a result that arrives later as its own event, so
    displaying it would show the same call twice.

    Args:
      printer: Printer to receive formatted output.
      result: Tool-result entry to render.
      output: Display policy for the result body. ``None`` hides it, so
          a tool must opt in (``--tool Bash.output=on``); most tools'
          bodies are far longer than the pane should carry.

    """
    if result.is_error:
        printer.write_tool_error(strip_reminders(result.content))
        return
    if result.diff:
        printer.write_diff(result.diff, result.diff_file_path)
    if result.hint:
        printer.write_hint(result.hint)
    if result.summary:
        printer.write_tool_summary(result.summary)
    if result.kind is ToolResultKind.PENDING:
        return
    rows = format_output(strip_reminders(result.content), output or OutputSpec())
    if rows:
        printer.write_tool_output("\n".join(rows))


def error_text(exc: BaseException) -> str:
    """Render an exception for display.

    ``UserFacingError`` carries polished, user-actionable text; the
    class-name prefix would add Python-internals noise to a message the
    reader is meant to act on. Shared so the child-block path cannot
    drift from the parent's rule.
    """
    return (
        str(exc) if isinstance(exc, UserFacingError) else f"{type(exc).__name__}: {exc}"
    )


def strip_reminders(content: str) -> str:
    """Drop ``<system-reminder>`` blocks from user-facing text.

    The banner is model plumbing: a tool bakes it into ``content`` so the
    model reads it, and surfaces the same text on ``hint`` for the human.
    Rendering ``content`` verbatim therefore prints every nudge twice,
    the second time wrapped in a tag that means nothing to the reader.
    """
    return _REMINDER_RE.sub("", content).strip()


_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>\s*", re.DOTALL)


def make_render_observer(
    printer: Printer,
    *,
    show_thinking: Callable[[], bool] | None = None,
    output_policy: Callable[[str], ToolDisplay] | None = None,
) -> RenderObserver:
    """Return a ``RuntimeEvent``-consuming observer bound to ``printer``.

    Args:
      printer: Printer that receives formatted output.
      show_thinking: Predicate controlling thinking display. ``None`` always shows.
      output_policy: Maps a result's ``call_id`` to the ``output``
          :class:`OutputSpec` of its originating tool. ``None`` shows no
          result bodies -- the historical behaviour.

    Returns:
      observer: ``RenderObserver`` instance the agent appends to
          ``self.observers``; callable on each ``RuntimeEvent``.

    """
    return RenderObserver(
        printer, show_thinking=show_thinking, output_policy=output_policy
    )


def _no_output(call_id: str) -> ToolDisplay:
    """Default policy: render no result bodies."""
    del call_id
    return ToolDisplay()


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
        output_policy: Callable[[str], ToolDisplay] | None = None,
    ) -> None:
        self._printer = printer
        self._show_thinking = show_thinking or (lambda: True)
        self._output_policy: Callable[[str], ToolDisplay] = (
            output_policy if output_policy is not None else _no_output
        )
        self._stream_buf: str = ""
        self._child_text: dict[str, str] = {}
        self._child_items: dict[str, list[ChildItem]] = {}

    def __call__(self, event: RuntimeEvent) -> None:
        try:
            self._dispatch(event)
        except Exception:
            # Don't re-enter the printer from the safety net: if the
            # printer was the cause of the dispatch failure, calling
            # ``write_tool_error`` here would just raise again and
            # bury the original traceback. Log it instead so the
            # operator can inspect the failure offline. Tests that need
            # to see the failure use :func:`strict_observer` to flip
            # this branch into a re-raise.
            if _strict_observer.get():
                raise
            logger.exception("render observer failed for %s", type(event).__name__)

    def _dispatch(self, event: RuntimeEvent) -> None:
        # A ``hidden`` message reaches the model but not the human: flush any
        # pending stream so ordering holds, then render nothing for it.
        if isinstance(event, SessionMessage) and event.hidden:
            self._flush_stream()
            return
        match event:
            case UserMessage(text=text):
                self._flush_stream()
                self._printer.write_user_bar(text)
            case AgentSendMessage(source=source, text=text):
                self._flush_stream()
                self._printer.write_agent_bar(source, text)
            case ModelResponsePartial(text=text):
                self._feed_stream(text)
            case ModelResponseThinking(text=text):
                if self._show_thinking():
                    self._printer.write_thinking(text)
            case ModelResponseComplete():
                self._flush_stream()
            case ToolLabel(text=text):
                self._flush_stream()
                display = self._output_policy(event.call_id)
                self._printer.write_tool_label(
                    text, command=display.command, lang=display.command_lang
                )
            case ToolResult():
                render_tool_result(
                    self._printer,
                    event,
                    output=self._output_policy(event.call_id).output,
                )
            case DetachedResult(result=result):
                # A detached tool finally completed; the user has no
                # other visual cue for the late arrival, so render the
                # result through the same surface a sync ``ToolResult``
                # would use.
                self._flush_stream()
                render_tool_result(
                    self._printer,
                    result,
                    output=self._output_policy(result.call_id).output,
                )
            case ToolResultPartial(text=text):
                self._printer.write_chunk(text)
            case ModelResponseCancelled():
                self._flush_stream()
                self._printer.write_interrupted()
            case ModelServiceSuspended():
                self._flush_stream()
                self._printer.write_dim_line(service_suspended_text(event))
            case NoticeMessage(text=text):
                self._flush_stream()
                self._printer.write_dim_line(text)
            case ModelResponseError(exception=exc):
                self._flush_stream()
                # ``UserFacingError`` carries a polished, user-actionable
                # message; the class-name prefix would just add Python-
                # internals noise to text the user is supposed to read
                # and act on. Some user-facing errors further tailor
                # the halt banner: the generic "type to retry" misleads
                # when retrying cannot change the failed request.
                self._printer.write_tool_error(error_text(exc))
                if isinstance(exc, AuthRefreshError):
                    self._printer.write_halt(HALT_MESSAGE_AUTH)
                elif isinstance(exc, ContextOverflowError):
                    self._printer.write_halt(HALT_MESSAGE_CONTEXT)
                else:
                    self._printer.write_halt(HALT_MESSAGE)
            case ModelSwitchRejected(exception=exc):
                self._flush_stream()
                self._printer.write_tool_error(error_text(exc))
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
            case AgentSendQueuedMessage() | AgentSendDeferredMessage():
                # Silent by design: the visible event is the
                # ``AgentSendMessage`` that lands once the inbox gate
                # drains. Rendering the queued/deferred placeholder
                # would double-render the same payload from the user's
                # point of view.
                pass
            case _:
                pass

    def _feed_stream(self, chunk: str) -> None:
        """Buffer streaming text; flush stable Markdown blocks."""
        self._stream_buf += chunk
        boundary = find_stable_boundary(self._stream_buf)
        if boundary <= 0:
            # A long-running unclosed fence (no ``\n\n`` boundary inside
            # a code block) would otherwise let the buffer grow without
            # bound for the rest of the round. Flush at the cap so
            # memory stays bounded even if the model never emits a
            # closing fence.
            if len(self._stream_buf) > _STREAM_BUF_FLUSH_BYTES:
                self._flush_stream()
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
        """Buffer one child event; flush at stable boundaries or atomic events.

        Whenever the active child label changes, every *other* label's
        pending text and items are flushed first. Cost is O(num-other-
        children) per event -- intentional, not a hot path: the cross-
        child flush keeps slow children from rendering interleaved into
        the wrong slot, and the typical cohort fanout is small.
        """
        # ``ChildEvent`` may nest: a grandchild forwards through its
        # parent, which forwards to us as ``ChildEvent(label,
        # ChildEvent(...))``. Unwrap so the innermost runtime event is
        # what the rest of the dispatcher sees; otherwise grandchild
        # events fall through ``_child_atomic_item``'s ``None`` branch
        # and disappear from the UI.
        while isinstance(inner, ChildEvent):
            inner = inner.inner
        # When the active child label changes, flush any pending
        # streaming text from the previous label too -- otherwise a slow
        # child's partial output lingers until its ``ChildDoneEvent``
        # and renders in the wrong slot.
        for other in list(self._child_items.keys() | self._child_text.keys()):
            if other != label:
                self._move_text_to_items(other)
                self._emit_child(other)
        if isinstance(inner, ModelResponsePartial):
            buf = self._child_text.get(label, "") + inner.text
            boundary = find_stable_boundary(buf)
            if boundary <= 0:
                self._child_text[label] = buf
                # Mirror the parent ``_feed_stream`` size cap: a
                # long-running unclosed fence (no ``\n\n`` boundary)
                # would otherwise let ``_child_text[label]`` grow
                # without bound for the whole round.
                if len(buf) > _STREAM_BUF_FLUSH_BYTES:
                    self._move_text_to_items(label)
                    self._emit_child(label)
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
            self._printer.write_child_block(
                label, items, output_policy=self._output_policy
            )


def _child_atomic_item(inner: RuntimeEvent) -> ChildItem | None:
    """Translate a non-streaming child event into a child-block item.

    Any forwardable atomic child event renders in the child block; only
    ``ModelResponsePartial`` needs separate streaming handling (done in
    ``_consume_child`` before this is called). The accepted set must stay a
    superset of what ``agent_spawn`` always-forwards -- see
    ``test_every_always_forwarded_event_renders_a_child_block``.
    """
    if isinstance(inner, _CHILD_ITEM_TYPES):
        return inner
    return None


# Atomic child events that render in a child block (everything the forwarder
# may cross, minus the streamed ``ModelResponsePartial`` handled upstream).
_CHILD_ITEM_TYPES = (
    ToolLabel,
    ModelResponseThinking,
    ToolResult,
    ModelServiceSuspended,
    NoticeMessage,
    ModelResponseError,
    AgentSendMessage,
    UserMessage,
)


HELP_TEXT: Final = """\
sagent commands

  /help                       this list
  /quit, /exit                exit

  /clear                      wipe context (logs preserved on disk)
  /compact [hints]            compact history
  /recompact [hints]          alias for /compact

  /model    [args]            switch model; ids take option tags (+1m, +fast)
  /provider <name>            switch provider
  /thinking <state|partial>   adaptive/on/off/redact/show/hide
  /effort [level]             show or set effort; bare lists options
  /tool     NAME.key=value    reconfigure a tool, e.g. Bash.output=off
  /login                      re-auth current provider

  /tasks                      list running work (agents + fg + bg)
  /send     <target> <text>   send to subagent target: label, glob, {a,b}, /re/
  /halt     [<target>]        halt self or matching subagents (Ctrl+C)
  /kill     <qid|all|target>  cancel tool task(s) or matching subagents
  /defer    <text>            send as deferred (non-preempting); drains at AgentIdle\
"""
