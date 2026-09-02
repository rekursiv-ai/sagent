"""Run an Agent with a custom tool and a scripted model."""

from __future__ import annotations

from collections.abc import Callable
from types import MappingProxyType

import asyncio
import sys

from sagent.agent import Agent
from sagent.lib import token_count
from sagent.tools import tool
from sagent.types.capability import (
    ModelCapability,
    ModelLimits,
    ModelSettings,
)
from sagent.types.cost import (
    PriceCatalog,
    PriceCatalogProduct,
    TokenCost,
    TokenCount,
    TokenPrice,
)
from sagent.types.model import (
    ModelRequest,
    ModelResponse,
    UsageSnapshot,
)
from sagent.types.runtime import (
    AssistantMessage,
    ModelResponsePartial,
    RuntimeEvent,
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
    """Minimal offline model used to demonstrate Sagent's model contract.

    What the model OFFERS is ``capability``; what this instance CHOSE is
    ``settings``. An implementation declares both, not a flag per question.
    """

    capability = ModelCapability(
        model_id="scripted-offline",
        context=MappingProxyType(
            {"": ModelLimits(max_request_tokens=16_384, max_response_tokens=1_024)}
        ),
        # An offline model bills nothing, but an empty catalog would raise.
        prices=PriceCatalog({PriceCatalogProduct(): TokenPrice()}),
    )
    settings = ModelSettings.narrowest(capability)

    @property
    def limits(self) -> ModelLimits:
        """Ceilings of the selected context tag."""
        return self.settings.limits

    @property
    def tagged_model_id(self) -> str:
        """Display id carrying its context tag."""
        return f"{self.capability.model_id}{self.settings.context}"

    def spend(self, tokens: TokenCount) -> TokenCost:
        """An offline model bills nothing."""
        return self.capability.prices[PriceCatalogProduct()] * tokens

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

    def usage_snapshot(self) -> UsageSnapshot | None:
        """Return ``None``; the offline scripted model has no rate limits."""
        return None

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
                tokens=TokenCount(request=12, response=4),
            )
        return ModelResponse(
            message=AssistantMessage(
                tool_calls=(
                    ToolCall(id="echo-1", name="Echo", args={"text": "hello"}),
                ),
            ),
            tokens=TokenCount(request=8, response=4),
            stop_reason="model_tool_use",
        )

    async def stream(
        self,
        request: ModelRequest,
        publish: Callable[[RuntimeEvent], None] | None = None,
    ) -> ModelResponse:
        """Return the buffered response and optionally emit final text.

        Args:
          request: Model request containing the conversation history.
          publish: Optional runtime event sink; the final response text
              is published as a ``ModelResponsePartial``.

        Returns:
          response: Buffered response identical to :meth:`buffer`.

        """
        response = await self.buffer(request)
        if publish is not None and response.message.text:
            publish(ModelResponsePartial(response.message.text))
        return response

    async def close(self) -> None:
        """No-op teardown; this scripted model holds no resources."""
        return


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
