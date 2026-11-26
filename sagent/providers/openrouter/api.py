"""OpenRouter's Chat Completions API, sharing transport and reasoning replay."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

from sagent.providers import openrouter
from sagent.providers.openai.compat import OpenAICompatible


if TYPE_CHECKING:
    from types import ModuleType

    from sagent.model.types import Request
    from sagent.providers.openai.compat import OpenAIModel


__all__ = ["OpenRouter"]


class OpenRouter(OpenAICompatible):
    """Serve OpenRouter models using ``OPENROUTER_API_KEY``."""

    section: ClassVar[str] = "openrouter"
    aliases: ClassVar[tuple[str, ...]] = ("or",)
    package: ClassVar[ModuleType] = openrouter
    url: ClassVar[str] = "https://openrouter.ai/api/v1"
    base_url_variable: ClassVar[str] = "OPENROUTER_BASE_URL"
    key_variable: ClassVar[str] = "OPENROUTER_API_KEY"
    reasoning_field: ClassVar[str] = "reasoning"
    reasoning_details_field: ClassVar[str] = "reasoning_details"
    max_tokens_field: ClassVar[str] = "max_tokens"
    temperature: ClassVar[float | None] = None

    @override
    def encode(self, request: Request, *, model: OpenAIModel) -> dict[str, object]:
        body = super().encode(request, model=model)
        body.pop("reasoning_effort", None)
        if model.capability.thinking_effort != frozenset({"none"}):
            effort = model.settings.thinking_effort
            body["reasoning"] = {"effort": "minimal" if effort == "min" else effort}
        return body
