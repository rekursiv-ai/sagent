"""Spawn a reviewer sub-agent from a parent agent."""

from __future__ import annotations

import asyncio
import sys

from sagent.agent import Agent
from sagent.lib.json import json_freeze
from sagent.providers import Google
from sagent.tools import AgentSpawn


async def main() -> None:
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
    result = await agent.run(
        json_freeze(
            {
                "prompt": (
                    "Write a two-sentence explanation of why typed Message "
                    "objects help agent tool dispatch."
                ),
            }
        )
    )
    sys.stdout.write(f"{result.content}\n")


if __name__ == "__main__":
    asyncio.run(main())
