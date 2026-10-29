"""OpenAI models over the Responses API, billed to an API key.

Typical usage example:

  async with OpenAI() as openai:                # reads OPENAI_API_KEY
      sol = openai["sol"]
      await sol.count(request)                   # exact, from the server
      async for event in sol.stream(request):    # TextChunk..., then Response
          ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

from treekle import from_plain

from sagent.providers.openai import responses
from sagent.providers.openai.compat import OpenAICompatible


if TYPE_CHECKING:
    from sagent.model.types import Request
    from sagent.providers.openai.compat import OpenAIModel, Reader


__all__ = [
    "OpenAI",
]


class OpenAI(OpenAICompatible):
    """OpenAI's models, billed to the API key in ``OPENAI_API_KEY``.

    Args:
      name: The name the catalog knows this provider by; the section by default.
      resources: What this shares with other providers; ``None`` makes and
        closes its own.
      api_key: The key; empty reads ``OPENAI_API_KEY``.
      base_url: The endpoint; empty uses OpenAI's.

    """

    section: ClassVar[str] = "openai"
    package: ClassVar[str] = "sagent.providers.openai"
    document: ClassVar[str] = "openai.yaml"
    url: ClassVar[str] = "https://api.openai.com/v1"
    key_variable: ClassVar[str] = "OPENAI_API_KEY"
    path: ClassVar[str] = "/responses"

    @override
    def encode(self, request: Request, *, model: OpenAIModel) -> dict[str, object]:
        return responses.encode(
            request,
            capability=model.capability,
            settings=model.settings,
            temperature=self.temperature,
        )

    @override
    def reader(self, *, model: OpenAIModel) -> Reader:
        return responses.StreamReader(
            capability=model.capability,
            settings=model.settings,
        )

    @override
    async def count(self, request: Request, *, model: OpenAIModel) -> int:
        """Return the tokens ``request`` would send ``model``, as the server counts them."""
        body = self.encode(request, model=model)
        # ``input_tokens`` rejects the generation knobs it has no use for.
        counted = {
            "model",
            "input",
            "instructions",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "reasoning",
        }
        reply = await self.reply(
            "/responses/input_tokens",
            body={key: value for key, value in body.items() if key in counted},
            headers=self.headers(request),
        )
        return from_plain(reply.get("input_tokens"), int)
