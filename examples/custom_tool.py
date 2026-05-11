"""Build an Agent with a small custom tool."""

from __future__ import annotations

import asyncio
import sys

from sagent.agent import Agent
from sagent.custom_types import Message, TextMessage
from sagent.lib.json import json_freeze
from sagent.lib.message import get_directive, response_text
from sagent.providers import Google


class CharacterCount:
    """Count characters in a string."""

    name = "CharacterCount"
    tool_id = "application/x-tool-character-count"
    description = "Count Unicode code points in a string."
    supports_microcompaction = False
    directive_schema = json_freeze(
        {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text whose characters should be counted.",
                },
            },
            "required": ["text"],
        }
    )

    def summary(self, msg: Message) -> str:
        """Return a compact status label for this invocation."""
        text = str(get_directive(msg).get("text", ""))
        return f"CharacterCount {len(text)} chars"

    def summary_result(self, result: Message) -> str | None:
        del result
        return None

    def prompt(self) -> str:
        """Return optional tool-specific system prompt text."""
        return ""

    async def run(self, msg: Message) -> Message:
        """Run the tool and return a conversation-visible result."""
        text = str(get_directive(msg).get("text", ""))
        return TextMessage(str(len(text)), "text/plain", parent_id=msg.id)


async def main() -> None:
    """Run the example agent."""
    agent = Agent(
        model=Google.from_env().model("gemini-3.1-pro-preview"),
        system="Use CharacterCount whenever exact string length matters.",
        tools=[CharacterCount()],
    )
    prompt = (
        "How many characters are in 'agentic systems'? Use the"
        " tool, then answer in one sentence."
    )
    async for _event in agent.run(TextMessage(prompt, "text/x-user-message")):
        pass
    for m in reversed(agent.history):
        if m.descriptor == "multipart/x-model-message":
            sys.stdout.write(f"{response_text(m)}\n")
            return


if __name__ == "__main__":
    asyncio.run(main())
