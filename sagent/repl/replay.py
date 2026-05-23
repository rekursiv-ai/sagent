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

from sagent.lib.compaction import MICROCOMPACTED_ARGS_KEY
from sagent.repl.render import render_tool_result
from sagent.types.history import (
    AssistantMessage,
    ToolCall,
    ToolResult,
    UserMessage,
)


if TYPE_CHECKING:
    from collections.abc import Mapping

    from sagent.agent.agent import Agent
    from sagent.repl.render import Printer
    from sagent.types.tools import Tool


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
                    printer.write_tool_label(_label_for_call(tc, tools))
            case ToolResult():
                render_tool_result(printer, entry)
    cost = float(agent.total_cost_usd)
    cost_str = f" · ${cost:.2f}" if cost > 0 else ""
    printer.write_line(f"── resumed · {len(history)} messages{cost_str} ──")


def _label_for_call(tc: ToolCall, tools: Mapping[str, Tool]) -> str:
    """Return the rendered label for ``tc``, honoring microcompacted stubs.

    Microcompaction preserves the original ``tool.summary(args)`` output
    inside ``args[MICROCOMPACTED_ARGS_KEY]`` so resume can still render
    the historical label even though the original args (``file_path``,
    ``cmd``, etc.) have been replaced. Without this check every
    microcompacted ``Read`` falls back to ``Read.summary``'s ``"?"``
    placeholder and the resumed scrollback loses every filename.
    """
    stored = tc.args.get(MICROCOMPACTED_ARGS_KEY) if tc.args else None
    if isinstance(stored, str) and stored:
        return stored
    tool = tools.get(tc.name)
    return tool.summary(tc.args) if tool is not None else tc.name
