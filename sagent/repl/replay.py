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
from sagent.repl.render import (
    make_render_observer,
    render_tool_result,
)
from sagent.types.runtime import (
    AgentSendMessage,
    AssistantMessage,
    ToolCall,
    ToolResult,
    UserMessage,
)
from sagent.types.tape import (
    ContextSplice,
    ReferrableTapeEvent,
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
    tape = agent.runtime.tape
    if not tape:
        return
    tools = agent.tools_map
    render_event = make_render_observer(
        printer,
        show_thinking=lambda: agent.show_thinking,
    )
    rendered_messages = 0
    for record in tape:
        if isinstance(record, ContextSplice):
            continue
        assert isinstance(record, ReferrableTapeEvent)
        entry = record.event
        match entry:
            case UserMessage(text=text):
                rendered_messages += 1
                printer.write_user_bar(text)
            case AgentSendMessage(source=source, text=text):
                rendered_messages += 1
                printer.write_agent_bar(source, text)
            case AssistantMessage(
                text=text,
                thinking_blocks=blocks,
                tool_calls=calls,
            ):
                rendered_messages += 1
                if agent.show_thinking:
                    for block in blocks:
                        body = str(block.get("thinking") or block.get("text") or "")
                        if body:
                            printer.write_thinking(body)
                if text.strip():
                    printer.write_markdown(text)
                for tc in calls:
                    printer.write_tool_label(_label_for_call(tc, tools))
            case ToolResult():
                rendered_messages += 1
                render_tool_result(printer, entry)
            case _:
                render_event(entry)
    cost = float(agent.total_cost_usd)
    cost_str = f" · ${cost:.2f}" if cost > 0 else ""
    printer.write_line(f"── resumed · {rendered_messages} messages{cost_str} ──")


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
