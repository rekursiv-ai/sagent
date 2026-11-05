"""Fireworks models through its OpenAI-compatible Chat Completions API."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

from sagent.providers import fireworks
from sagent.providers.openai.compat import OpenAICompatible


if TYPE_CHECKING:
    from types import ModuleType

    from sagent.model.types import Request
    from sagent.providers.openai.compat import OpenAIModel


__all__ = ["Fireworks"]


class Fireworks(OpenAICompatible):
    """Serve Fireworks models using ``FIREWORKS_API_KEY``."""

    section: ClassVar[str] = "fireworks"
    aliases: ClassVar[tuple[str, ...]] = ()
    package: ClassVar[ModuleType] = fireworks
    url: ClassVar[str] = "https://api.fireworks.ai/inference/v1"
    key_variable: ClassVar[str] = "FIREWORKS_API_KEY"
    base_url_variable: ClassVar[str] = "FIREWORKS_BASE_URL"
    max_tokens_field: ClassVar[str] = "max_tokens"
    reasoning_field: ClassVar[str] = "reasoning_content"

    @override
    def encode(self, request: Request, *, model: OpenAIModel) -> dict[str, object]:
        body = super().encode(request, model=model)
        body["reasoning_history"] = "preserved"
        return body
