"""Local LM Studio models through Chat Completions.

Pass ``models=ModelTable(...)`` to select installed model identifiers and limits.
An explicit ``api_key`` supports servers with optional authentication enabled.
"""

from types import ModuleType
from typing import ClassVar

from sagent.providers import lmstudio
from sagent.providers.openai.compat import OpenAICompatible


__all__ = ["LMStudio"]


class LMStudio(OpenAICompatible):
    """Serve installed LM Studio models with optional bearer authentication."""

    section: ClassVar[str] = "lmstudio"
    aliases: ClassVar[tuple[str, ...]] = ("lm-studio",)
    package: ClassVar[ModuleType] = lmstudio
    url: ClassVar[str] = "http://localhost:1234/v1"
    base_url_variable: ClassVar[str] = "LMSTUDIO_BASE_URL"
    max_tokens_field: ClassVar[str] = "max_tokens"
