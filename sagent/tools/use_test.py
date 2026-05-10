# ruff: noqa: T201
"""Integration test: agent with tool dispatch.

Verifies end-to-end: Anthropic provider → model → agent → tool
dispatch → final text response.

Run manually (hits real API):
    cd .
    uv run python -m sagent.tools.use_test
"""

from __future__ import annotations

import asyncio

from sagent.agent import Agent
from sagent.custom_types import TextMessage
from sagent.lib.message import response_text
from sagent.providers import Anthropic
from sagent.tools import Bash, Read


async def _run_and_get_text(agent: Agent, prompt: str) -> str:
    """Drive one turn of ``agent.run`` and return the final assistant text."""
    async for _event in agent.run(TextMessage(prompt, "text/x-user-message")):
        pass
    for m in reversed(agent.history):
        if m.descriptor == "multipart/x-model-message":
            return response_text(m) or "(no text)"
    return "(no text)"


async def main() -> None:
    """Run tool-use scenarios via the Agent abstraction."""
    provider = Anthropic.from_env()
    sonnet = provider.model("claude-sonnet-4-20250514")

    scientist = Agent(
        name="test-agent",
        description="Integration test agent.",
        model=sonnet,
        system="You are a helpful assistant. Be concise.",
        tools=[Bash(), Read()],
    )

    print("=== Test 1: Simple bash tool use ===")
    text = await _run_and_get_text(
        scientist,
        "What Python version is installed? Use bash.",
    )
    print(f"Response: {text}\n")

    print("=== Test 2: Read a file ===")
    text = await _run_and_get_text(
        scientist,
        "Read sagent/__init__.py and tell me what it exports.",
    )
    print(f"Response: {text}\n")

    print("=== Test 3: Multi-step ===")
    text = await _run_and_get_text(
        scientist,
        "List the Python files in sagent/ using bash, then read the shortest"
        " one. Summarize.",
    )
    print(f"Response: {text}\n")


if __name__ == "__main__":
    asyncio.run(main())
