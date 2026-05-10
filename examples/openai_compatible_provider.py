"""Connect Sagent to a local OpenAI-compatible endpoint."""

from __future__ import annotations

from typing import ClassVar

import asyncio
import os
import sys

from sagent.agent import Agent
from sagent.custom_types import TextMessage
from sagent.lib.message import response_text
from sagent.providers.lib.cost import ModelProfile, Pricing
from sagent.providers.openai_compat import OpenAICompat


class LocalOpenAI(OpenAICompat):
    """Provider for a chat-completions-compatible local server."""

    DEFAULT_MODEL = os.environ.get("LOCAL_OPENAI_MODEL", "local-model")
    DEFAULT_UTILITY_MODEL = DEFAULT_MODEL
    ENV_VAR = "LOCAL_OPENAI_API_KEY"
    BASE_URL = os.environ.get("LOCAL_OPENAI_BASE_URL", "http://localhost:8000/v1")
    KNOWN_MODELS: ClassVar[dict[str, ModelProfile]] = {
        DEFAULT_MODEL: ModelProfile(
            max_request_tokens=128_000,
            max_response_tokens=8_192,
            pricing=Pricing(),
        ),
    }


async def main() -> None:
    """Run a single prompt against the local endpoint."""
    if not os.environ.get(LocalOpenAI.ENV_VAR):
        sys.stderr.write(f"Set {LocalOpenAI.ENV_VAR} before running this example.\n")
        sys.exit(1)
    agent = Agent(
        model=LocalOpenAI.from_env().model(),
        system="Answer concisely.",
        tools=[],
    )
    async for _event in agent.run(
        TextMessage("Say hello from Sagent.", "text/x-user-message"),
    ):
        pass
    for m in reversed(agent.history):
        if m.descriptor == "multipart/x-model-message":
            sys.stdout.write(f"{response_text(m)}\n")
            return


if __name__ == "__main__":
    asyncio.run(main())
