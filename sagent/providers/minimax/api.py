"""MiniMax models through its OpenAI-compatible endpoint, billed to an API key.

Typical usage example:

  async with MiniMax() as minimax:               # reads MINIMAX_API_KEY
      m3 = minimax["default"]

  box = MiniMax(api_key="none", base_url="http://gpu-box:8000/v1")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

from sagent.providers.openai.compat import OpenAICompatible


if TYPE_CHECKING:
    from sagent.model.types import Request
    from sagent.providers.openai.compat import OpenAIModel


__all__ = [
    "MiniMax",
]


class MiniMax(OpenAICompatible):
    """MiniMax models, billed to the API key in ``MINIMAX_API_KEY``.

    Args:
      name: The name the catalog knows this provider by; the section by default.
      resources: What this shares with other providers; ``None`` makes and
        closes its own.
      api_key: The key; empty reads ``MINIMAX_API_KEY``.
      base_url: The endpoint; empty uses ``api.minimax.io``.

    """

    section: ClassVar[str] = "minimax"
    package: ClassVar[str] = "sagent.providers.minimax"
    document: ClassVar[str] = "minimax.yaml"
    url: ClassVar[str] = "https://api.minimax.io/v1"
    key_variable: ClassVar[str] = "MINIMAX_API_KEY"
    reasoning_field: ClassVar[str] = "reasoning_content"

    @override
    def encode(self, request: Request, *, model: OpenAIModel) -> dict[str, object]:
        body = super().encode(request, model=model)
        # Without it the reasoning stays inside ``content``, wrapped in
        # ``<think>`` tags, and would show as the answer.
        body["reasoning_split"] = True
        # Only M3 has a switch; M2.x takes ``disabled`` and ignores it, so its
        # rows offer nothing to send.
        if model.capability.thinking_budget != frozenset({"none"}):
            off = model.settings.thinking_budget == "none"
            body["thinking"] = {"type": "disabled" if off else "adaptive"}
        return body
