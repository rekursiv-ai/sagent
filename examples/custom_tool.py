"""Build an Agent with a small custom tool."""

from __future__ import annotations

from collections.abc import Mapping

import asyncio
import sys

from sagent.agent import Agent
from sagent.lib.custom_json import json_freeze
from sagent.providers import Google
from sagent.types.runtime import (
    AssistantMessage,
    ToolResult,
    UserMessage,
)


class CharacterCount:
    """Count characters in a string."""

    name = "CharacterCount"
    tool_id = "application/x-tool-character-count"
    clearable_results = True
    description = "Count Unicode code points in a string."
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

    def summary(self, args: Mapping[str, object]) -> str:
        """Return a compact status label for this invocation.

        Args:
          args: Tool invocation arguments.

        Returns:
          label: Single-line summary including the character count.

        """
        return f"CharacterCount {len(str(args.get('text', '')))} chars"

    def prompt(self) -> str:
        """Return optional tool-specific system prompt text.

        Returns:
          prompt: Empty string (no extra prompt section).

        """
        return ""

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        """Run in parallel: this tool has no shared resource."""
        del args
        return None

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        """Run the tool and return a ToolResult.

        Args:
          args: Tool invocation arguments containing ``text``.

        Returns:
          result: Tool result whose content is the character count.

        """
        return ToolResult(call_id="", content=str(len(str(args.get("text", "")))))


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
    async for _event in agent.run(UserMessage(text=prompt)):
        pass
    for m in reversed(agent.history):
        if isinstance(m, AssistantMessage) and m.text:
            sys.stdout.write(f"{m.text}\n")
            return


if __name__ == "__main__":
    asyncio.run(main())
