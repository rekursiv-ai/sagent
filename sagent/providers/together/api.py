"""Together models through its OpenAI-compatible Chat Completions API."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from sagent.providers import together
from sagent.providers.openai.compat import OpenAICompatible


if TYPE_CHECKING:
    from types import ModuleType


__all__ = ["Together"]


class Together(OpenAICompatible):
    """Serve Together models using ``TOGETHER_API_KEY``."""

    section: ClassVar[str] = "together"
    aliases: ClassVar[tuple[str, ...]] = ()
    package: ClassVar[ModuleType] = together
    url: ClassVar[str] = "https://api.together.ai/v1"
    key_variable: ClassVar[str] = "TOGETHER_API_KEY"
    base_url_variable: ClassVar[str] = "TOGETHER_BASE_URL"
    max_tokens_field: ClassVar[str] = "max_tokens"
    reasoning_field: ClassVar[str] = "reasoning"
