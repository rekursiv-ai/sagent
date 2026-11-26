"""Mistral's Chat API, preserving its native thinking content blocks."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

from sagent.providers import mistral
from sagent.providers.openai.compat import OpenAICompatible


if TYPE_CHECKING:
    from types import ModuleType

    from sagent.model.types import Request
    from sagent.providers.openai.compat import OpenAIModel


__all__ = ["Mistral"]


class Mistral(OpenAICompatible):
    """Serve Mistral models using ``MISTRAL_API_KEY``."""

    section: ClassVar[str] = "mistral"
    aliases: ClassVar[tuple[str, ...]] = ()
    package: ClassVar[ModuleType] = mistral
    url: ClassVar[str] = "https://api.mistral.ai/v1"
    base_url_variable: ClassVar[str] = "MISTRAL_BASE_URL"
    key_variable: ClassVar[str] = "MISTRAL_API_KEY"
    max_tokens_field: ClassVar[str] = "max_tokens"
    reasoning_content_blocks: ClassVar[bool] = True
    temperature: ClassVar[float | None] = None

    @override
    def encode(self, request: Request, *, model: OpenAIModel) -> dict[str, object]:
        body = super().encode(request, model=model)
        body.pop("stream_options", None)
        return body
