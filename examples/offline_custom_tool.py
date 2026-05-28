"""Run an Agent with a custom tool and a scripted model."""

from __future__ import annotations

from collections.abc import Callable

import asyncio
import sys

from sagent.agent import Agent
from sagent.lib import token_count
from sagent.tools import tool
from sagent.types.model import (
    ModelRequest,
    ModelResponse,
    Pricing,
    TokenCount,
)
from sagent.types.runtime import (
    AssistantMessage,
    ToolCall,
    ToolResult,
    UserMessage,
)


@tool(name="Echo")
def echo(text: str) -> str:
    """Return the supplied text with a visible prefix.

    Args:
      text: Text to echo.

    Returns:
      echoed: ``text`` prefixed with ``"Echo said: "``.

    """
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
    valid_service_tiers: tuple[str, ...] = ()
    supports_context_management = False
    supports_persistent_retry = False
    supports_account_auth = False
    max_image_dim = 0
    max_image_bytes = 0
    pricing = Pricing()

    def approx_text_tokens(self, text: str) -> int:
        """Offline ``len(text) // 4`` heuristic; minimum 1."""
        return max(1, len(text) // 4)

    def approx_image_tokens(self, data: bytes) -> int:
        """Scripted model has no image support; returns zero."""
        del data
        return 0

    def approx_request_tokens(self, request: ModelRequest) -> int:
        """Walk-and-sum every wire-bearing surface of ``request``."""
        return token_count.approx_request_tokens(request, self)

    async def actual_text_tokens(self, text: str) -> int:
        """Offline model; delegates to ``approx``."""
        return self.approx_text_tokens(text)

    async def actual_image_tokens(self, data: bytes) -> int:
        """Offline model; delegates to ``approx``."""
        return self.approx_image_tokens(data)

    async def actual_request_tokens(self, request: ModelRequest) -> int:
        """Offline model; delegates to ``approx``."""
        return self.approx_request_tokens(request)

    def is_context_overflow(self, error: Exception) -> bool:
        """Return false because the scripted model never raises API errors.

        Args:
          error: Provider error to classify.

        Returns:
          overflow: Always ``False``.

        """
        del error
        return False

    def is_retryable_provider_error(self, error: Exception) -> bool:
        """Return false because the scripted model never raises API errors.

        Args:
          error: Provider error to classify.

        Returns:
          retryable: Always ``False``.

        """
        del error
        return False

    async def buffer(self, request: ModelRequest) -> ModelResponse:
        """Return one tool call, then a final answer after the tool result.

        Args:
          request: Model request containing the conversation history.

        Returns:
          response: Either a tool-call response or the final answer.

        """
        if any(isinstance(msg, ToolResult) for msg in request.messages):
            return ModelResponse(
                message=AssistantMessage(text="Echo said: hello"),
                tokens=TokenCount(input_tokens=12, output_tokens=4),
            )
        return ModelResponse(
            message=AssistantMessage(
                tool_calls=(
                    ToolCall(id="echo-1", name="Echo", args={"text": "hello"}),
                ),
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
        """Return the buffered response and optionally emit final text.

        Args:
          request: Model request containing the conversation history.
          on_text: Optional callback invoked with the final response text.
          on_thinking: Optional thinking callback (ignored).

        Returns:
          response: Buffered response identical to :meth:`buffer`.

        """
        del on_thinking
        response = await self.buffer(request)
        if on_text is not None and response.message.text:
            on_text(response.message.text)
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
    async for _event in agent.run(UserMessage(text="Echo hello.")):
        pass
    for m in reversed(agent.history):
        if isinstance(m, AssistantMessage) and m.text:
            return m.text
    return ""


def main() -> None:
    """Run the example from the command line."""
    sys.stdout.write(f"{asyncio.run(run_example())}\n")


if __name__ == "__main__":
    main()
