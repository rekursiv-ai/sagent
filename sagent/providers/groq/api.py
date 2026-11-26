"""Groq models through its OpenAI-compatible Chat Completions API."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

from sagent.providers import groq
from sagent.providers.openai.compat import OpenAICompatible


if TYPE_CHECKING:
    from types import ModuleType

    from sagent.model.types import Request
    from sagent.providers.openai.compat import OpenAIModel


__all__ = ["Groq"]


class Groq(OpenAICompatible):
    """Serve Groq models using ``GROQ_API_KEY``."""

    section: ClassVar[str] = "groq"
    aliases: ClassVar[tuple[str, ...]] = ()
    package: ClassVar[ModuleType] = groq
    url: ClassVar[str] = "https://api.groq.com/openai/v1"
    key_variable: ClassVar[str] = "GROQ_API_KEY"
    base_url_variable: ClassVar[str] = "GROQ_BASE_URL"
    max_tokens_field: ClassVar[str] = "max_completion_tokens"
    reasoning_field: ClassVar[str] = "reasoning"

    @override
    def encode(self, request: Request, *, model: OpenAIModel) -> dict[str, object]:
        body = super().encode(request, model=model)
        body["include_reasoning"] = True
        return body
