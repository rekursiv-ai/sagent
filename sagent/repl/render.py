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

import time

from sagent import providers as _providers
from sagent.agent.handlers.base import Handler, InlineHandler
from sagent.agent.handlers.model_switch import ModelSwitchHandler
from sagent.custom_types import TextMessage
from sagent.lib.message import get_queue_id
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
    def write_tool_summary(self, text: str) -> None: ...
    def write_hint(self, text: str) -> None: ...
    def write_thinking(self, text: str) -> None: ...
    def write_diff(self, diff: str, file_path: str = "") -> None: ...
    def write_interrupted(self) -> None: ...
    def write_child_block(self, label: str, items: list[Message]) -> None: ...
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
      tool_summaries: ``write_tool_summary`` payloads.
      hints: ``write_hint`` payloads.
      thinkings: ``write_thinking`` payloads.
      diffs: ``write_diff`` payloads.
      interruptions: count of ``write_interrupted`` calls.
      child_blocks: tuples of ``(label, items)`` where each item is
          a ``Message`` from the child agent's event stream.
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


def render_tool_result(printer: Printer, msg: Message) -> None:
    """Render the user-facing parts of a ``multipart/x-tool-result``.

    Errors via ``write_tool_error``; diffs via ``write_diff``;
    one-line success summaries via ``write_tool_summary``;
    bash-lint nudges (``text/x-hint-tool-use-nudge``) via the dim
    yellow ``hint:`` line. Plain text parts are skipped -- they
    belong to history, not user feedback (the model already saw
    them).

    Tools opt into the dim ``  ⎿ <summary>`` receipt line by
    emitting a ``text/x-tool-summary`` part in their result
    multipart (success path only -- on error, the ``text/x-error``
    line tells the story; doubling up adds noise).

    Shared helper so child-event rendering and replay reuse the
    same formatting the parent's live ``RenderToolResult`` does.
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


class RenderToolResult(InlineHandler):
    """Render relevant parts of a parent agent's ``multipart/x-tool-result``."""

    descriptors: tuple[str, ...] = ("multipart/x-tool-result",)

    def __init__(self, printer: Printer) -> None:
        self._printer = printer

    @override
    async def handle(self, agent: Agent, msg: Message) -> None:
        del agent
        render_tool_result(self._printer, msg)


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
    """Buffer child-agent events; flush as labeled gutter blocks.

    Streaming text from a child arrives as many small ``text/plain``
    chunks (mid-token byte-boundary fragments). Rendering each chunk
    as its own line produces jagged output. Instead, accumulate
    text per child label and flush at:

    1. Stable Markdown boundary (paragraph break, code-block end)
       -- same heuristic ``RenderStream`` uses for the parent's text.
    2. Atomic event from the same child (tool label, tool result,
       thinking, error, hint, child-done) -- those render in the
       same labeled block.
    3. Different child labels flush only already-complete pending items,
       not unstable streaming text. Provider chunks are arbitrary
       fragments, so label-change flushes of raw text turn concurrent
       child streams into mid-word gutter lines.

    Flushed output goes through ``Printer.write_child_block(label,
    items)`` where items is the list of ``(descriptor, content)``
    pairs accumulated in this block. The printer handles the gutter
    rendering ("Agent_N  :  ..." on first line, indented thereafter).
    """

    descriptors: tuple[str, ...] = ("multipart/x-child-event",)

    def __init__(self, printer: Printer) -> None:
        self._printer = printer
        # Per-label accumulated streaming text (not yet at a boundary).
        self._text_buf: dict[str, str] = {}
        # Per-label rendered items pending flush. Each item is a
        # Message the printer knows how to render -- mixing text/plain
        # (Markdown chunks), tool labels, tool results, errors, and
        # the child-done summary into one block matches the user's
        # mental model of "what this child said in this turn."
        self._items: dict[str, list[Message]] = {}

    @override
    async def handle(self, agent: Agent, msg: Message) -> None:
        del agent
        parts = cast("tuple[Message, ...]", msg.content)
        if len(parts) < 2:
            return
        label = str(parts[0].content)
        for inner in parts[1:]:
            self._consume(label, inner)

    def _consume(self, label: str, inner: Message) -> None:
        # X-interleave: when a different label emits, flush the
        # other one's pending state so both children remain visible
        # in real-time.
        for other in list(self._items):
            if other != label:
                self._emit_pending(other)
        if inner.descriptor == "text/plain":
            # X-interleave is item-level only: complete paragraphs,
            # tool events, and errors may surface when another child
            # speaks, but raw provider chunks stay buffered until a
            # real text boundary. Label changes are scheduler
            # interleaving, not parse boundaries.
            self._text_buf[label] = self._text_buf.get(label, "") + str(inner.content)
            buf = self._text_buf[label]
            boundary = find_stable_boundary(buf)
            if boundary <= 0:
                return
            stable = buf[:boundary].rstrip("\n")
            # Keep the unstable tail in ``text_buf``; only the stable
            # prefix gets emitted now. ``_emit_pending`` is used here
            # (NOT ``_flush``) because ``_flush`` would also move
            # ``text_buf`` into items -- splitting a mid-paragraph
            # word across two blocks.
            self._text_buf[label] = buf[boundary:]
            if stable:
                self._items.setdefault(label, []).append(
                    TextMessage(stable, "text/plain"),
                )
                self._emit_pending(label)
            return
        # Atomic event. Flush any pending streaming text first (so
        # it appears above the atomic event in the same block),
        # then emit the block.
        self._move_text_to_items(label)
        self._items.setdefault(label, []).append(inner)
        self._emit_pending(label)

    def _move_text_to_items(self, label: str) -> None:
        """Move accumulated streaming text into the items list."""
        text = self._text_buf.pop(label, "").rstrip("\n")
        if text:
            self._items.setdefault(label, []).append(
                TextMessage(text, "text/plain"),
            )

    def _emit_pending(self, label: str) -> None:
        """Emit pending items for ``label``; leave ``text_buf`` untouched."""
        items = self._items.pop(label, [])
        if items:
            self._printer.write_child_block(label, items)

    def _flush(self, label: str) -> None:
        """Force-flush everything for ``label``: text_buf + items.

        Used when the buffered tail must surface even if no stable
        boundary has been reached -- atomic event arriving or end of
        stream.
        """
        self._move_text_to_items(label)
        self._emit_pending(label)


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


_HELP_TEXT = """\
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


class HelpHandler(InlineHandler):
    """Print the slash-command reference on ``text/x-help-request``.

    The verbatim block lives in ``_HELP_TEXT`` so the parser, the
    handler, and the user are all reading the same canonical list.
    """

    descriptors: tuple[str, ...] = ("text/x-help-request",)

    def __init__(self, printer: Printer | None = None) -> None:
        self._printer = printer

    @override
    async def handle(self, agent: Agent, msg: Message) -> None:
        del agent, msg
        if self._printer is not None:
            self._printer.write_line(_HELP_TEXT)


class TasksHandler(InlineHandler):
    """Print live work on ``text/x-tasks-request``.

    Lists every registered agent in :data:`agent_registry` with its
    foreground task count and visible background task summaries.
    Hidden bg tasks (REPL pump, daemons) are filtered out -- they're
    infrastructure, not user-actionable. Foreground tasks are reported
    as a count rather than per-task because the in-flight registry
    keys by task identity (``id(task)``), which isn't meaningful to
    surface.
    """

    descriptors: tuple[str, ...] = ("text/x-tasks-request",)

    def __init__(self, printer: Printer | None = None) -> None:
        self._printer = printer

    @override
    async def handle(self, agent: Agent, msg: Message) -> None:
        del msg
        if self._printer is None:
            return
        # Lazy import to avoid a top-level cycle (tools.core <-> repl).
        from sagent.tools.core import agent_registry  # noqa: PLC0415

        lines: list[str] = []
        now = time.time()
        total_fg = 0
        total_bg = 0
        for label, other in agent_registry.items():
            visible_bg = [j for j in other.background_tasks.values() if not j.hidden]
            fg = len(other.tasks)
            bg = len(visible_bg)
            total_fg += fg
            total_bg += bg
            tag = " (self)" if other is agent else ""
            lines.append(f"  {label}{tag:<8s}  fg={fg} bg={bg}")
            for job in visible_bg:
                phase = (
                    "cancelled"
                    if job.task.cancelled()
                    else (
                        "completed"
                        if job.task.done()
                        else (
                            "sleeping"
                            if job.delay_sec > 0 and (now - job.started) < job.delay_sec
                            else "running"
                        )
                    )
                )
                lines.append(
                    f"    bg: {job.queue_id:<10s}  {job.tool_name:<16s}  "
                    f"{phase:<10s}  {now - job.started:.0f}s"
                )
        header = (
            f"sagent: {len(agent_registry)} agent(s), "
            f"{total_fg} foreground, {total_bg} background"
        )
        out = header + "\n" + "\n".join(lines) if lines else header
        self._printer.write_line(out)


class LoginHandler(InlineHandler):
    """Handle ``text/x-login-request`` by invoking the provider's ``login``.

    Slash-command parsing emits ``text/x-login-request`` so both REPL
    paths (active keybinding, idle prompt) take the same code path; the
    actual re-auth lives here.
    """

    descriptors: tuple[str, ...] = ("text/x-login-request",)

    def __init__(self, printer: Printer | None = None) -> None:
        self._printer = printer

    @override
    async def handle(self, agent: Agent, msg: Message) -> None:
        del msg
        spec = agent.model_spec
        if spec is None:
            self._write("[/login] agent has no model spec")
            return
        prov_cls = getattr(_providers, spec.provider, None)
        if prov_cls is None:
            self._write(f"[/login] unknown provider {spec.provider!r}")
            return
        login_fn = getattr(prov_cls, "login", None)
        if login_fn is None:
            self._write(f"[/login] {spec.provider} has no login method")
            return
        try:
            login_fn()
            self._write(f"[/login] {spec.provider} re-authenticated")
        except (RuntimeError, OSError, ValueError, TimeoutError) as exc:
            self._write(f"[/login] {exc}")

    def _write(self, line: str) -> None:
        if self._printer is not None:
            self._printer.write_line(line)


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
