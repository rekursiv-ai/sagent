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
from sagent.custom_types import Message, MultipartMessage, TextMessage
from sagent.lib.json import json_freeze
from sagent.providers import Anthropic
from sagent.tools import Bash, Read


def _text(msg: Message) -> str:
    if isinstance(msg, TextMessage):
        return msg.content
    if isinstance(msg, MultipartMessage):
        for p in msg.content:
            if isinstance(p, TextMessage) and p.descriptor == "text/plain":
                return p.content
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
    response = await scientist.run(
        json_freeze({"prompt": "What Python version is installed? Use bash."}),
    )
    print(f"Response: {_text(response)}\n")

    print("=== Test 2: Read a file ===")
    response = await scientist.run(
        json_freeze({"prompt": "Read sagent/__init__.py and tell me what it exports."}),
    )
    print(f"Response: {_text(response)}\n")

    print("=== Test 3: Multi-step ===")
    response = await scientist.run(
        json_freeze(
            {
                "prompt": "List the Python files in"
                " sagent/ using bash,"
                " then read the shortest one. Summarize."
            }
        ),
    )
    print(f"Response: {_text(response)}\n")


if __name__ == "__main__":
    asyncio.run(main())
