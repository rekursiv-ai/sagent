"""Create a tool from a plain Python function."""

from __future__ import annotations

import asyncio
import sys

from sagent.agent import Agent
from sagent.agent.runtime import AssistantMessage, UserMessage
from sagent.providers import Google
from sagent.tools import tool


@tool(name="WordCount")
def word_count(text: str) -> str:
    """Count whitespace-separated words in text.

    Args:
      text: Input text to tokenize on whitespace.

    Returns:
      count: Word count as a decimal string.

    """
    return str(len(text.split()))


async def main() -> None:
    """Run an agent with the decorator-created tool."""
    agent = Agent(
        model=Google.from_env().model("gemini-3.1-pro-preview"),
        system="Use WordCount whenever exact word counts matter.",
        tools=[word_count],
    )
    prompt = (
        "How many words are in 'typed agents compose cleanly'?"
        " Use the tool, then answer in one sentence."
    )
    async for _event in agent.run(UserMessage(text=prompt)):
        pass
    for m in reversed(agent.history):
        if isinstance(m, AssistantMessage) and m.text:
            sys.stdout.write(f"{m.text}\n")
            return


if __name__ == "__main__":
    asyncio.run(main())
