"""DeepSeek models through its OpenAI-compatible Chat Completions API."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

from sagent.providers import deepseek
from sagent.providers.openai.compat import OpenAICompatible


if TYPE_CHECKING:
    from types import ModuleType

    from sagent.model.types import Request
    from sagent.providers.openai.compat import OpenAIModel


__all__ = ["DeepSeek"]


class DeepSeek(OpenAICompatible):
    """Serve DeepSeek models using ``DEEPSEEK_API_KEY``."""

    section: ClassVar[str] = "deepseek"
    aliases: ClassVar[tuple[str, ...]] = ()
    package: ClassVar[ModuleType] = deepseek
    url: ClassVar[str] = "https://api.deepseek.com"
    key_variable: ClassVar[str] = "DEEPSEEK_API_KEY"
    base_url_variable: ClassVar[str] = "DEEPSEEK_BASE_URL"
    max_tokens_field: ClassVar[str] = "max_tokens"
    reasoning_field: ClassVar[str] = "reasoning_content"
    temperature: ClassVar[float | None] = None

    @override
    def encode(self, request: Request, *, model: OpenAIModel) -> dict[str, object]:
        body = super().encode(request, model=model)
        body["reasoning_effort"] = model.settings.thinking_effort
        return body
