"""Render persisted messages into scrollback on resume.

Resumed sessions display the same scrollback the live REPL produces:
user messages as bars, model responses as Markdown, thinking blocks
as the dim "Thinking" preface, tool labels via each tool's own
``summary(req)`` (so ``Bash cd . && git status`` shows up,
not the lowercase ``bash``), tool results via the shared
:func:`render_tool_result` so diffs / errors / hints / receipts
render the same way they do live. Closes with a single
``── resumed · N messages · $X ──`` footer.

Live and replay both go through ``Printer`` + ``render_tool_result``;
adding a new render concern (e.g. the ``⎿`` summary line on tool
results) lights up in both paths automatically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from sagent.agent.dispatch import tc_tool_id, tool_call_label
from sagent.lib.descriptors import is_thinking, is_user_message
from sagent.lib.message import thinking_text
from sagent.repl.render import render_tool_result


if TYPE_CHECKING:
    from sagent.agent.agent import Agent
    from sagent.custom_types import Message
    from sagent.repl.render import Printer


def replay_messages(agent: Agent, printer: Printer) -> None:
    """Render persisted messages into scrollback.

    Args:
      agent: Agent whose ``history`` to replay.
      printer: Printer that receives all replayed output.

    """
    messages = agent.history
    if not messages:
        return
    tools = agent.tools_map
    for msg in messages:
        if is_user_message(msg.descriptor):
            content = (
                cast(str, msg.content)
                if msg.descriptor == "text/x-user-message"
                else ""
            )
            printer.write_user_bar(content)
        elif msg.descriptor == "multipart/x-model-message":
            parts = cast(tuple["Message", ...], msg.content)
            for p in parts:
                if is_thinking(p.descriptor):
                    text = thinking_text(p)
                    if text:
                        printer.write_thinking(text)
            text_content = "\n".join(
                cast(str, p.content) for p in parts if p.descriptor == "text/plain"
            )
            if text_content.strip():
                printer.write_markdown(text_content)
            for p in parts:
                if p.descriptor == "multipart/x-tool-call":
                    tool = tools.get(tc_tool_id(p))
                    printer.write_tool_label(tool_call_label(tool, p))
        elif msg.descriptor == "multipart/x-tool-result":
            render_tool_result(printer, msg)
    cost = float(agent.total_cost_usd)
    cost_str = f" · ${cost:.2f}" if cost > 0 else ""
    printer.write_line(f"── resumed · {len(messages)} messages{cost_str} ──")
