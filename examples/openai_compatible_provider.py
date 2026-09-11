"""Connect Sagent to a local OpenAI-compatible endpoint."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import ClassVar

import asyncio
import os
import sys

from sagent.agent import Agent
from sagent.providers.openai.compat import OpenAICompat
from sagent.types.capability import ModelCapability, ModelLimits
from sagent.types.cost import (
    PriceCatalog,
    PriceCatalogProduct,
    TokenPrice,
)
from sagent.types.runtime import AssistantMessage, UserMessage


class LocalOpenAI(OpenAICompat):
    """Provider for a chat-completions-compatible local server."""

    DEFAULT_MODEL = os.environ.get("LOCAL_OPENAI_MODEL", "local-model")
    DEFAULT_UTILITY_MODEL = DEFAULT_MODEL
    ENV_VAR = "LOCAL_OPENAI_API_KEY"
    BASE_URL = os.environ.get("LOCAL_OPENAI_BASE_URL", "http://localhost:8000/v1")
    CAPABILITIES: ClassVar[Mapping[str, ModelCapability]] = MappingProxyType(
        {
            DEFAULT_MODEL: ModelCapability(
                model_id=DEFAULT_MODEL,
                context=MappingProxyType(
                    {
                        "": ModelLimits(
                            max_request_tokens=128_000, max_response_tokens=8_192
                        )
                    }
                ),
                # A local server bills nothing, but a missing row would raise.
                prices=PriceCatalog({PriceCatalogProduct(): TokenPrice()}),
            )
        }
    )


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
    async for _event in agent.run(UserMessage(text="Say hello from Sagent.")):
        pass
    for m in reversed(agent.history):
        if isinstance(m, AssistantMessage) and m.text:
            sys.stdout.write(f"{m.text}\n")
            return


if __name__ == "__main__":
    asyncio.run(main())
