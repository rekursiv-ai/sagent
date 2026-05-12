"""Render persisted history into scrollback on resume.

Resumed sessions display the same scrollback the live REPL produces:
user messages as bars, model responses as Markdown, thinking blocks
as the dim "Thinking" preface, tool labels via each tool's own
``summary(args)``, tool results via the shared :func:`render_tool_result`.
Closes with a single ``── resumed · N messages · $X ──`` footer.

Live and replay both go through ``Printer`` + ``render_tool_result``;
adding a new render concern lights up in both paths automatically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sagent.agent.runtime import (
    AssistantMessage,
    ToolResult,
    UserMessage,
)
from sagent.repl.render import render_tool_result


if TYPE_CHECKING:
    from sagent.agent.agent import Agent
    from sagent.repl.render import Printer


def replay_messages(agent: Agent, printer: Printer) -> None:
    """Render persisted history into scrollback.

    Args:
      agent: Agent whose ``history`` to replay.
      printer: Printer that receives all replayed output.

    """
    history = agent.history
    if not history:
        return
    tools = agent.tools_map
    for entry in history:
        match entry:
            case UserMessage(text=text):
                printer.write_user_bar(text)
            case AssistantMessage(
                text=text,
                thinking_blocks=blocks,
                tool_calls=calls,
            ):
                for block in blocks:
                    body = str(block.get("thinking") or block.get("text") or "")
                    if body:
                        printer.write_thinking(body)
                if text.strip():
                    printer.write_markdown(text)
                for tc in calls:
                    tool = tools.get(tc.name)
                    label = tool.summary(tc.args) if tool is not None else tc.name
                    printer.write_tool_label(label)
            case ToolResult():
                render_tool_result(printer, entry)
    cost = float(agent.total_cost_usd)
    cost_str = f" · ${cost:.2f}" if cost > 0 else ""
    printer.write_line(f"── resumed · {len(history)} messages{cost_str} ──")
