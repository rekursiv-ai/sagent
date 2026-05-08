"""Render persisted messages into scrollback on resume.

Mirrors ``repl/handlers.py::replay_messages`` so sessions resumed
under the v2 REPL display the same scrollback the v1 REPL does.
Renders user messages as bars, model responses as markdown,
thinking blocks as the dim "Thinking" preface, tool labels as
dim lines, tool errors / diffs in their full formatting. Closes
with a single ``── resumed · N messages · $X ──`` footer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from rich.text import Text

from sagent.lib.descriptors import is_thinking, is_user_message
from sagent.lib.message import get_tool_name, thinking_text
from sagent.repl.format import print_user_bar
from sagent.repl.render_diff import render_diff_detail
from sagent.repl.tight_markdown import TightMarkdown


if TYPE_CHECKING:
    from rich.console import Console

    from sagent.agent.agent import Agent
    from sagent.custom_types import Message


def replay_messages(agent: Agent, console: Console) -> None:
    """Render persisted messages into scrollback.

    Args:
      agent: Agent whose ``history`` to replay.
      console: Console for all replayed output.

    """
    messages = agent.history
    if not messages:
        return
    for msg in messages:
        if is_user_message(msg.descriptor):
            content = (
                cast(str, msg.content)
                if msg.descriptor == "text/x-user-message"
                else ""
            )
            print_user_bar(console, content)
        elif msg.descriptor == "multipart/x-model-message":
            parts = cast(tuple["Message", ...], msg.content)
            for p in parts:
                if is_thinking(p.descriptor):
                    text = thinking_text(p)
                    if text:
                        _emit_thinking(console, text)
            text_content = "\n".join(
                cast(str, p.content) for p in parts if p.descriptor == "text/plain"
            )
            if text_content.strip():
                console.print()
                console.print(TightMarkdown(text_content))
            for p in parts:
                if p.descriptor == "multipart/x-tool-call":
                    label = get_tool_name(p)
                    console.print(Text(f"  {label}", style="dim"))
        elif msg.descriptor == "multipart/x-tool-result":
            parts = cast(tuple["Message", ...], msg.content)
            for p in parts:
                if p.descriptor == "text/x-error":
                    console.print(
                        Text(f"    ✗ {cast(str, p.content).strip()}", style="dim red")
                    )
                elif p.descriptor == "text/x-diff" and p.content:
                    render_diff_detail(console, cast(str, p.content))
    cost = float(agent.total_cost_usd)
    cost_str = f" · ${cost:.2f}" if cost > 0 else ""
    console.print(
        Text(
            f"── resumed · {len(messages)} messages{cost_str} ──",
            style="dim",
        ),
    )


def _emit_thinking(out: Console, text: str) -> None:
    out.print(Text("∴ Thinking", style="italic dim"))
    for line in text.splitlines() or [""]:
        out.print(Text(f"  {line}", style="dim"))
    out.print()
