"""Create a tool from a plain Python function."""

from __future__ import annotations

import asyncio
import sys

from sagent.agent import Agent
from sagent.custom_types import TextMessage
from sagent.lib.message import response_text
from sagent.providers import Google
from sagent.tools import tool


@tool(name="WordCount")
def word_count(text: str) -> str:
    """Count whitespace-separated words in text."""
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
    async for _event in agent.run(TextMessage(prompt, "text/x-user-message")):
        pass
    for m in reversed(agent.history):
        if m.descriptor == "multipart/x-model-message":
            sys.stdout.write(f"{response_text(m)}\n")
            return


if __name__ == "__main__":
    asyncio.run(main())
