"""Run an Agent with a custom tool and a scripted model."""

from __future__ import annotations

from collections.abc import Callable

import asyncio
import sys

from sagent.agent import Agent
from sagent.custom_types import (
    ModelRequest,
    ModelResponse,
    MultipartMessage,
    Pricing,
    TextMessage,
    TokenCount,
)
from sagent.lib.json import json_freeze
from sagent.lib.message import tool_call_message
from sagent.tools import tool


@tool(name="Echo")
def echo(text: str) -> str:
    """Return the supplied text with a visible prefix."""
    return f"Echo said: {text}"


class ScriptedModel:
    """Minimal offline model used to demonstrate Sagent's model contract."""

    model_id = "scripted-offline"
    max_request_tokens = 16_384
    max_response_tokens = 1024
    supports_streaming = False
    supports_thinking = False
    supports_effort = False
    supports_cache_control = False
    supports_context_management = False
    supports_persistent_retry = False
    supports_account_auth = False
    max_image_dim = 0
    max_image_bytes = 0
    pricing = Pricing()

    def estimate_text_token_count(self, text: str) -> int:
        """Estimate text tokens with a deliberately simple offline heuristic."""
        return max(1, len(text) // 4)

    def estimate_image_token_count(self, data: bytes) -> int:
        """Return zero because this scripted model has no image support."""
        del data
        return 0

    def is_context_overflow(self, error: Exception) -> bool:
        """Return false because the scripted model never raises API errors."""
        del error
        return False

    async def buffer(self, request: ModelRequest) -> ModelResponse:
        """Return one tool call, then a final answer after the tool result."""
        if any(msg.descriptor == "multipart/x-tool-result" for msg in request.messages):
            return ModelResponse(
                content=MultipartMessage(
                    (TextMessage("Echo said: hello", "text/plain"),),
                    "multipart/x-model-message",
                ),
                tokens=TokenCount(input_tokens=12, output_tokens=4),
            )
        return ModelResponse(
            content=MultipartMessage(
                (
                    tool_call_message(
                        "echo-1",
                        "Echo",
                        json_freeze({"text": "hello"}),
                    ),
                ),
                "multipart/x-model-message",
            ),
            tokens=TokenCount(input_tokens=8, output_tokens=4),
            stop_reason="model_tool_use",
        )

    async def stream(
        self,
        request: ModelRequest,
        on_text: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        """Return the buffered response and optionally emit final text."""
        del on_thinking
        response = await self.buffer(request)
        if (
            on_text is not None
            and response.stop_reason == "model_finished"
            and isinstance(response.content, MultipartMessage)
        ):
            for part in response.content.content:
                if part.descriptor == "text/plain":
                    on_text(str(part.content))
        return response


async def run_example() -> str:
    """Run the offline custom-tool example.

    Returns:
      text: Final agent response text.

    """
    agent = Agent(
        model=ScriptedModel(),
        system="Use Echo when asked to repeat text.",
        tools=[echo],
        max_tool_call_rounds=3,
        thinking=None,
    )
    result = await agent.run(json_freeze({"prompt": "Echo hello."}))
    return str(result.content)


def main() -> None:
    """Run the example from the command line."""
    sys.stdout.write(f"{asyncio.run(run_example())}\n")


if __name__ == "__main__":
    main()
