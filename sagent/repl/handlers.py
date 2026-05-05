"""Event handlers and replay for the REPL.

Descriptor contract: the ``descriptor`` field on a ``Message`` is the
authoritative type tag for its ``content``. Do NOT inspect ``content``
with ``isinstance`` to determine its type -- check ``descriptor`` instead.
For example, ``descriptor == "text/plain"`` means ``content`` is a ``str``;
use ``str(msg.content)`` only to satisfy the static type checker, never as
a runtime type guard.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, cast

from rich.text import Text

from sagent.agent import Agent
from sagent.custom_types import Message
from sagent.lib.descriptors import is_thinking, is_user_message
from sagent.lib.message import get_tool_name, thinking_text
from sagent.repl.custom_types import RenderState
from sagent.repl.render import (
    format_elapsed,
    print_user_bar,
    set_terminal_title,
)
from sagent.repl.render_diff import (
    find_stable_boundary,
    render_diff_detail,
)
from sagent.repl.tight_markdown import TightMarkdown


if TYPE_CHECKING:
    from rich.console import Console


def render_batch(batch: list[Message], render_frame: RenderState) -> None:
    """Dispatch a batch of events to their descriptor-specific handlers.

    Args:
      batch: Events to render.
      render_frame: Mutable render state for the current frame.

    """
    for event in batch:
        handler = _DISPATCH.get(event.descriptor)
        if handler is not None:
            handler(event, render_frame)


def replay_messages(agent: Agent, console: Console, out: Console) -> None:
    """Render persisted messages into scrollback on resume.

    Args:
      agent: Agent whose message history to replay.
      console: Console for tool labels and metadata.
      out: Console for markdown content and user bars.

    """
    messages = agent.messages
    if not messages:
        return
    for msg in messages:
        if is_user_message(msg.descriptor):
            content = (
                cast(str, msg.content)
                if msg.descriptor == "text/x-user-message"
                else ""
            )
            print_user_bar(out, content)
        elif msg.descriptor == "multipart/x-model-message":
            parts = cast(tuple[Message, ...], msg.content)
            for p in parts:
                if is_thinking(p.descriptor):
                    text = thinking_text(p)
                    if text:
                        _emit_thinking(out, text)
            text_content = "\n".join(
                cast(str, p.content) for p in parts if p.descriptor == "text/plain"
            )
            if text_content.strip():
                emit_markdown(out, text_content)
            for p in parts:
                if p.descriptor == "multipart/x-tool-call":
                    label = get_tool_name(p)
                    _emit_tool_label(console, label)
        elif msg.descriptor == "multipart/x-tool-result":
            parts = cast(tuple[Message, ...], msg.content)
            _render_tool_result_parts(parts, console)
    cost = float(agent.total_cost_usd)
    cost_str = f" · ${cost:.2f}" if cost > 0 else ""
    console.print(
        Text(
            f"── resumed · {len(messages)} messages{cost_str} ──",
            style="dim",
        ),
    )


def _flush_text_buf(render_frame: RenderState) -> None:
    """Render buffered text before a non-text transition (tool, error, etc)."""
    if not render_frame.buf.strip():
        return
    stable = render_frame.buf.rstrip("\n")
    render_frame.buf = ""
    if not render_frame.printed_header:
        render_frame.printed_header = True
    render_frame.out.print()
    emit_markdown(render_frame.out, stable)


def _flush_child_text(label: str, render_frame: RenderState) -> None:
    text = render_frame.child_bufs.pop(label, "").strip()
    if not text:
        return
    pfx = f"[{label}]"
    for line in text.splitlines():
        render_frame.console.print(Text(f"    {pfx} {line}", style="dim"))


def _handle_text(event: Message, render_frame: RenderState) -> None:
    text = cast(str, event.content)
    render_frame.buf += text
    boundary = find_stable_boundary(render_frame.buf)
    if boundary <= 0:
        return
    stable = render_frame.buf[:boundary].rstrip("\n")
    render_frame.buf = render_frame.buf[boundary:]
    if not stable:
        return
    if not render_frame.printed_header:
        render_frame.printed_header = True
    render_frame.out.print()
    emit_markdown(render_frame.out, stable)


def _handle_thinking(event: Message, render_frame: RenderState) -> None:
    text = cast(str, event.content)
    _emit_thinking(render_frame.out, text)


def _handle_injected_user(event: Message, render_frame: RenderState) -> None:
    text = cast(str, event.content)
    _flush_text_buf(render_frame)
    print_user_bar(render_frame.out, text)


def _handle_tool_call_msg(event: Message, render_frame: RenderState) -> None:
    _flush_text_buf(render_frame)
    label = cast(str, event.content)
    _emit_tool_label(render_frame.console, label)


def _render_tool_result_parts(
    parts: tuple[Message, ...],
    console: Console,
) -> None:
    for p in parts:
        if p.descriptor == "text/x-error":
            _emit_tool_error(console, cast(str, p.content))
        elif p.descriptor == "text/x-hint-tool-use-nudge" and p.content:
            _emit_bash_lint(console, cast(str, p.content))
        elif p.descriptor == "text/x-diff" and p.content:
            render_diff_detail(console, cast(str, p.content))


def _handle_tool_result_msg(event: Message, render_frame: RenderState) -> None:
    parts = cast(tuple[Message, ...], event.content)
    _render_tool_result_parts(parts, render_frame.console)


def _handle_done(event: Message, render_frame: RenderState) -> None:
    render_frame.done_event = event


def _handle_error(event: Message, render_frame: RenderState) -> None:
    text = cast(str, event.content)
    _flush_text_buf(render_frame)
    _emit_tool_error(render_frame.console, text)


def _handle_interrupted(event: Message, render_frame: RenderState) -> None:
    del event
    _flush_text_buf(render_frame)
    render_frame.console.print(Text("[interrupted]", style="dim"))


def _handle_child_event(event: Message, render_frame: RenderState) -> None:
    """Render a labeled child-agent event."""
    content = cast(tuple[Message, ...], event.content)
    if len(content) < 2:
        return
    label_msg, inner = content[0], content[1]
    label = cast(str, label_msg.content)
    pfx = f"[{label}]"

    if inner.descriptor == "text/x-tool-label":
        _flush_child_text(label, render_frame)
        desc = cast(str, inner.content)
        render_frame.console.print(Text(f"    {pfx} {desc}", style="dim"))
    elif inner.descriptor == "multipart/x-tool-result":
        _flush_child_text(label, render_frame)
        parts = cast(tuple[Message, ...], inner.content)
        for p in parts:
            if p.descriptor == "text/x-error":
                _emit_child_error(render_frame.console, pfx, cast(str, p.content))
            elif p.descriptor == "text/x-hint-tool-use-nudge" and p.content:
                render_frame.console.print(
                    Text(f"    {pfx} hint: {cast(str, p.content)}", style="dim yellow")
                )
    elif inner.descriptor == "text/plain":
        text = cast(str, inner.content)
        buf = render_frame.child_bufs.get(label, "") + text
        lines = buf.splitlines(keepends=True)
        render_frame.child_bufs[label] = ""
        for line in lines:
            if line.endswith("\n"):
                render_frame.console.print(
                    Text(f"    {pfx} {line.rstrip()}", style="dim")
                )
            else:
                render_frame.child_bufs[label] = line
    elif inner.descriptor == "text/x-thinking":
        _flush_child_text(label, render_frame)
        render_frame.console.print(Text(f"    {pfx} ∴ Thinking", style="italic dim"))
    elif inner.descriptor == "application/x-child-done":
        _flush_child_text(label, render_frame)
        data = cast(Mapping[str, float], inner.content)
        elapsed = data.get("elapsed", 0.0)
        tokens = int(data.get("model_response_tokens", 0))
        cost = data.get("cost_usd", 0.0)
        summary_parts: list[str] = [format_elapsed(elapsed)]
        if tokens:
            summary_parts.append(f"{tokens}↓")
        if cost > 0:
            summary_parts.append(f"${cost:.2f}")
        summary = " · ".join(summary_parts)
        render_frame.console.print(Text(f"    {pfx} done [{summary}]", style="dim"))
    elif inner.descriptor == "multipart/x-child-event":
        _flush_child_text(label, render_frame)
        _handle_child_event(inner, render_frame)


def flush_child_text_buffers(render_frame: RenderState) -> None:
    """Flush all pending child-agent text buffers.

    Args:
      render_frame: Mutable render state holding child buffers.

    """
    for label in list(render_frame.child_bufs):
        _flush_child_text(label, render_frame)


def _handle_status_changed(event: Message, _render_frame: RenderState) -> None:
    text = cast(str, event.content)
    set_terminal_title(text)


_DISPATCH: dict[str, Callable[[Message, RenderState], None]] = {
    "text/plain": _handle_text,
    "text/x-thinking": _handle_thinking,
    "text/x-error": _handle_error,
    "text/x-user-injected": _handle_injected_user,
    "text/x-interrupted": _handle_interrupted,
    "text/x-signal-user-input": _handle_injected_user,
    "text/x-signal-status-changed": _handle_status_changed,
    "text/x-tool-label": _handle_tool_call_msg,
    "multipart/x-tool-result": _handle_tool_result_msg,
    "multipart/x-child-event": _handle_child_event,
    "application/x-done": _handle_done,
}


def emit_markdown(out: Console, stable: str) -> None:
    out.print(TightMarkdown(stable))


def _emit_thinking(out: Console, text: str) -> None:
    out.print(Text("∴ Thinking", style="italic dim"))
    for line in text.splitlines() or [""]:
        out.print(Text(f"  {line}", style="dim"))
    out.print()


def _emit_tool_label(console: Console, label: str) -> None:
    lines = label.splitlines() or [""]
    header = lines[0]
    console.print(Text(f"  {header}", style="dim"))
    for line in lines[1:]:
        console.print(Text(f"  {line}", style="dim"))


def _emit_tool_error(console: Console, raw: str) -> None:
    console.print(Text(f"    ✗ {raw.strip()}", style="dim red"))


def _emit_bash_lint(console: Console, hint: str) -> None:
    console.print(Text(f"    hint: {hint}", style="dim yellow"))


def _emit_child_error(console: Console, pfx: str, raw: str) -> None:
    console.print(Text(f"    {pfx} ✗ {raw.strip()}", style="dim red"))
