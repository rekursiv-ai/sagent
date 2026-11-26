"""Z.ai models through its OpenAI-compatible Chat Completions API."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

from sagent.providers import zai
from sagent.providers.openai.compat import OpenAICompatible


if TYPE_CHECKING:
    from types import ModuleType

    from sagent.model.types import Request
    from sagent.providers.openai.compat import OpenAIModel


__all__ = ["ZAI"]


class ZAI(OpenAICompatible):
    """Serve Z.ai models using ``ZAI_API_KEY``."""

    section: ClassVar[str] = "zai"
    aliases: ClassVar[tuple[str, ...]] = ()
    package: ClassVar[ModuleType] = zai
    url: ClassVar[str] = "https://api.z.ai/api/paas/v4"
    key_variable: ClassVar[str] = "ZAI_API_KEY"
    base_url_variable: ClassVar[str] = "ZAI_BASE_URL"
    max_tokens_field: ClassVar[str] = "max_tokens"
    reasoning_field: ClassVar[str] = "reasoning_content"

    @override
    def encode(self, request: Request, *, model: OpenAIModel) -> dict[str, object]:
        body = super().encode(request, model=model)
        body["reasoning_effort"] = model.settings.thinking_effort
        body["thinking"] = {"type": "enabled", "clear_thinking": False}
        body["tool_stream"] = True
        return body
