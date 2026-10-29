"""Kimi through Moonshot AI's OpenAI-compatible endpoint, billed to an API key.

Typical usage example:

  async with Moonshot() as moonshot:             # reads MOONSHOT_API_KEY
      kimi = moonshot["default"]

  box = Moonshot(api_key="none", base_url="http://gpu-box:8000/v1")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

from sagent.providers.openai.compat import OpenAICompatible


if TYPE_CHECKING:
    from sagent.model.types import Request
    from sagent.providers.openai.compat import OpenAIModel


__all__ = [
    "Moonshot",
]


class Moonshot(OpenAICompatible):
    """Kimi models, billed to the API key in ``MOONSHOT_API_KEY``.

    Args:
      name: The name the catalog knows this provider by; the section by default.
      resources: What this shares with other providers; ``None`` makes and
        closes its own.
      api_key: The key; empty reads ``MOONSHOT_API_KEY``.
      base_url: The endpoint; empty uses ``api.moonshot.ai``.

    """

    section: ClassVar[str] = "moonshot"
    package: ClassVar[str] = "sagent.providers.moonshot"
    document: ClassVar[str] = "moonshot.yaml"
    url: ClassVar[str] = "https://api.moonshot.ai/v1"
    key_variable: ClassVar[str] = "MOONSHOT_API_KEY"
    reasoning_field: ClassVar[str] = "reasoning_content"
    # K3 fixes its sampler and says to omit it; K2.x says the temperature
    # cannot be changed and is not to be passed.
    temperature: ClassVar[float | None] = None

    @override
    def encode(self, request: Request, *, model: OpenAIModel) -> dict[str, object]:
        body = super().encode(request, model=model)
        capability, settings = model.capability, model.settings
        # The base folds ``max`` into OpenAI's ``high``; K3 takes its levels as
        # they are, and its row offers only levels the wire spells.
        if capability.effort_as_level:
            body["reasoning_effort"] = settings.thinking_effort
        # Only K2.6 has a switch; K3 and K2.7 Code are not to be sent one.
        if capability.thinking_budget == frozenset({"none", "auto"}):
            off = settings.thinking_budget == "none"
            body["thinking"] = {"type": "disabled" if off else "enabled"}
        return body
