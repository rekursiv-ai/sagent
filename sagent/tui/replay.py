"""A resumed session, redrawn as the scrollback it would have had.

Each record of the window since the last clear goes through the scrollback's
own ``show``, the path a live session's events take.

Typical usage example:

  replay(agent, scrollback)       # Before the prompt opens.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sagent.session.types import (
    AgentToAgentMessage,
    AssistantMessage,
    Thinking,
    ToolCall,
    ToolResult,
    UserMessage,
)


if TYPE_CHECKING:
    from sagent.agent.types import Agent
    from sagent.tui.render import Scrollback


__all__ = [
    "replay",
]


def replay(agent: Agent, scrollback: Scrollback) -> None:
    """Show what the model sees of ``agent``'s session, then a footer.

    Args:
      agent: Whose session to redraw.
      scrollback: Where it goes; its printer's ``show_thinking`` decides reasoning.

    """
    records = list(agent.session)
    if not records:
        return
    before = scrollback.shown
    for record in records:
        match record:
            case (
                UserMessage()
                | AgentToAgentMessage()
                | AssistantMessage()
                | Thinking()
                | ToolCall()
                | ToolResult()
            ):
                scrollback.show(record)
            case _:
                pass
    scrollback.show_held()
    model = agent.model
    settings = model.settings
    parts = [
        "resumed",
        f"{scrollback.shown - before} messages",
        f"{model.provider.name}/{model.capability.model_id}",
        *(
            [f"thinking={settings.thinking_budget}"]
            if settings.thinking_budget != "none"
            else []
        ),
        *(
            [f"effort={settings.thinking_effort}"]
            if settings.thinking_effort != "none"
            else []
        ),
    ]
    scrollback.printer.line(f"── {' · '.join(parts)} ──")
