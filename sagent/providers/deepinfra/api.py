"""DeepInfra models through its OpenAI-compatible Chat Completions API."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from sagent.providers import deepinfra
from sagent.providers.openai.compat import OpenAICompatible


if TYPE_CHECKING:
    from types import ModuleType


__all__ = ["DeepInfra"]


class DeepInfra(OpenAICompatible):
    """Serve DeepInfra models using ``DEEPINFRA_API_KEY``."""

    section: ClassVar[str] = "deepinfra"
    aliases: ClassVar[tuple[str, ...]] = ()
    package: ClassVar[ModuleType] = deepinfra
    url: ClassVar[str] = "https://api.deepinfra.com/v1/openai"
    key_variable: ClassVar[str] = "DEEPINFRA_API_KEY"
    base_url_variable: ClassVar[str] = "DEEPINFRA_BASE_URL"
    max_tokens_field: ClassVar[str] = "max_tokens"
    reasoning_field: ClassVar[str] = "reasoning_content"
