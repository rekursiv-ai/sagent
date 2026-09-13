"""Spawn a reviewer sub-agent from a parent agent."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from typing import cast

import asyncio
import sys

from sagent.agent.agent import Agent
from sagent.providers import Google
from sagent.tools import AgentSpawn
from sagent.types.runtime import AssistantMessage, UserMessage


async def main() -> int:
    """Run the parent-with-reviewer-sub-agent example.

    Returns:
      status: Process exit code.

    """
    reviewer = AgentSpawn(
        system="You are a strict reviewer. Return only concrete issues.",
        tools=[],
        max_tool_call_rounds=1,
        max_depth=0,
    )
    agent = Agent(
        model=Google.from_env().model("gemini-3.1-pro-preview"),
        system=(
            "Draft the answer, then use AgentSpawn to get an independent review "
            "before returning the final version."
        ),
        tools=[reviewer],
    )
    prompt = (
        "Write a two-sentence explanation of why typed history entries "
        "help agent tool dispatch."
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
