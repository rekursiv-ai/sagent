"""Cerebras models through its OpenAI-compatible Chat Completions API."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from sagent.providers import cerebras
from sagent.providers.openai.compat import OpenAICompatible


if TYPE_CHECKING:
    from types import ModuleType


__all__ = ["Cerebras"]


class Cerebras(OpenAICompatible):
    """Serve Cerebras models using ``CEREBRAS_API_KEY``."""

    section: ClassVar[str] = "cerebras"
    aliases: ClassVar[tuple[str, ...]] = ()
    package: ClassVar[ModuleType] = cerebras
    url: ClassVar[str] = "https://api.cerebras.ai/v1"
    key_variable: ClassVar[str] = "CEREBRAS_API_KEY"
    base_url_variable: ClassVar[str] = "CEREBRAS_BASE_URL"
    max_tokens_field: ClassVar[str] = "max_completion_tokens"
    reasoning_field: ClassVar[str] = "reasoning"
