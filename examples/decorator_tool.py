"""Create a tool from a plain Python function."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from typing import cast

import asyncio
import sys

from sagent.agent.agent import Agent
from sagent.providers import Google
from sagent.tools import tool
from sagent.types.runtime import AssistantMessage, UserMessage


@tool(name="WordCount")
def word_count(text: str) -> str:
    """Count whitespace-separated words in text.

    Args:
      text: Input text to tokenize on whitespace.

    Returns:
      count: Word count as a decimal string.

    """
    return str(len(text.split()))


async def main() -> int:
    """Run an agent with the decorator-created tool.

    Returns:
      status: Process exit code.

    """
    agent = Agent(
        model=Google.from_env().model("gemini-3.1-pro-preview"),
        system="Use WordCount whenever exact word counts matter.",
        tools=[word_count],
    )
    prompt = (
        "How many words are in 'typed agents compose cleanly'?"
        " Use the tool, then answer in one sentence."
    )
    async for _ in cast(
        AsyncGenerator[object, None],
        agent.run(UserMessage(text=prompt)),
    ):
        pass
    history = cast(Sequence[object], agent.history)
    for m in reversed(history):
        if isinstance(m, AssistantMessage) and m.text:
            sys.stdout.write(f"{m.text}\n")
            return 0
    return 0


if __name__ == "__main__":
    asyncio.run(main())
