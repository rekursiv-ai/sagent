"""Local Ollama models through Chat Completions.

Pass ``models=ModelTable(...)`` for installed names, capabilities and limits.
The local server ignores API keys; an explicit key still supports a proxy.
"""

from types import ModuleType
from typing import ClassVar

from sagent.providers import ollama
from sagent.providers.openai.compat import OpenAICompatible


__all__ = ["Ollama"]


class Ollama(OpenAICompatible):
    """Serve installed Ollama models without requiring authentication."""

    section: ClassVar[str] = "ollama"
    aliases: ClassVar[tuple[str, ...]] = ()
    package: ClassVar[ModuleType] = ollama
    url: ClassVar[str] = "http://localhost:11434/v1"
    base_url_variable: ClassVar[str] = "OLLAMA_BASE_URL"
    max_tokens_field: ClassVar[str] = "max_tokens"
