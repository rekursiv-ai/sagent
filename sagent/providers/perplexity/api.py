"""Perplexity Orchestrator's stateless Responses API subset."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, ClassVar, override

from treekle import from_plain

from sagent.providers import perplexity
from sagent.providers.openai.compat import ResponsesCompatible


if TYPE_CHECKING:
    from types import ModuleType

    from sagent.model.types import Request
    from sagent.providers.openai.compat import OpenAIModel


__all__ = ["Perplexity"]


class Perplexity(ResponsesCompatible):
    """Serve Orchestrator models using ``PERPLEXITY_API_KEY`` (private preview)."""

    section: ClassVar[str] = "perplexity"
    aliases: ClassVar[tuple[str, ...]] = ("pplx",)
    package: ClassVar[ModuleType] = perplexity
    url: ClassVar[str] = "https://api.perplexity.ai/orchestrator/v1"
    base_url_variable: ClassVar[str] = "PERPLEXITY_BASE_URL"
    key_variable: ClassVar[str] = "PERPLEXITY_API_KEY"
    temperature: ClassVar[float | None] = None

    @override
    def encode(self, request: Request, *, model: OpenAIModel) -> dict[str, object]:
        # Orchestrator rejects allowed_tools and strict:false, not just ignores them.
        tools = tuple(
            tool
            for tool in request.tools
            if request.allowed is None or tool.name in request.allowed
        )
        body = super().encode(replace(request, tools=tools, allowed=None), model=model)
        if tools:
            definitions = from_plain(body["tools"], list[dict[str, object]])
            for definition in definitions:
                definition.pop("strict", None)
            body["tools"] = definitions
        elif request.allowed is not None:
            body["tool_choice"] = "none"
        return body
