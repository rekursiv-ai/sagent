"""Spawn a reviewer sub-agent from a parent agent."""

from __future__ import annotations

import asyncio
import sys

from sagent.agent import Agent
from sagent.providers import Google
from sagent.tools import AgentSpawn
from sagent.types.history import AssistantMessage, UserMessage


async def main() -> None:
    """Run the parent-with-reviewer-sub-agent example."""
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
    async for _event in agent.run(UserMessage(text=prompt)):
        pass
    for m in reversed(agent.history):
        if isinstance(m, AssistantMessage) and m.text:
            sys.stdout.write(f"{m.text}\n")
            return


if __name__ == "__main__":
    asyncio.run(main())
