"""Create a tool from a plain Python function."""

from __future__ import annotations

import asyncio
import sys

from sagent.agent import Agent
from sagent.lib.json import json_freeze
from sagent.providers import Google
from sagent.tools import tool


@tool(name="WordCount")
def word_count(text: str) -> str:
    """Count whitespace-separated words in text."""
    return str(len(text.split()))


async def main() -> None:
    """Run an agent with the decorator-created tool."""
    agent = Agent(
        model=Google.from_env().model("gemini-2.5-flash"),
        system="Use WordCount whenever exact word counts matter.",
        tools=[word_count],
    )
    result = await agent.run(
        json_freeze(
            {
                "prompt": (
                    "How many words are in 'typed agents compose cleanly'?"
                    " Use the tool, then answer in one sentence."
                ),
            }
        )
    )
    sys.stdout.write(f"{result.content}\n")


if __name__ == "__main__":
    asyncio.run(main())
