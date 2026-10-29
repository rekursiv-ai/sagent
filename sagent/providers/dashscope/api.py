"""Qwen through DashScope's OpenAI-compatible endpoint, billed to an API key.

Every thinking knob is read off the catalog row: ``thinking_budget`` toggles
``enable_thinking``, and the effort rides as ``reasoning_effort`` on rows that
take a level, or as a ``thinking_budget`` cap on rows that take tokens.

Typical usage example:

  async with DashScope() as dashscope:           # reads DASHSCOPE_API_KEY
      qwen = dashscope["default"]

  box = DashScope(api_key="none", base_url="http://gpu-box:8000/v1")  # vLLM
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

from sagent.providers.openai.compat import OpenAICompatible


if TYPE_CHECKING:
    from sagent.model.types import Request
    from sagent.providers.openai.compat import OpenAIModel


__all__ = [
    "DashScope",
]


class DashScope(OpenAICompatible):
    """Qwen models, billed to the API key in ``DASHSCOPE_API_KEY``.

    Args:
      name: The name the catalog knows this provider by; the section by default.
      resources: What this shares with other providers; ``None`` makes and
        closes its own.
      api_key: The key; empty reads ``DASHSCOPE_API_KEY``.
      base_url: The endpoint; empty uses the international one. Mainland China
        is ``https://dashscope.aliyuncs.com/compatible-mode/v1``.

    """

    section: ClassVar[str] = "dashscope"
    package: ClassVar[str] = "sagent.providers.dashscope"
    document: ClassVar[str] = "dashscope.yaml"
    url: ClassVar[str] = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    key_variable: ClassVar[str] = "DASHSCOPE_API_KEY"
    reasoning_field: ClassVar[str] = "reasoning_content"

    completion_cap: ClassVar[frozenset[str]] = frozenset(
        {
            "qwen-max-3.8",
            "qwen-flash-3.8",
            "qwen-max-3.7",
            "qwen-plus-3.7",
            "qwen-flash-3.7",
            "qwen-plus-3.6",
            "qwen-flash-3.6",
        },
    )
    """Rows the API page lists for ``max_completion_tokens``; the rest take ``max_tokens``.

    Absent: ``qwen3.8-27b`` and ``qwen3.8-2.4t-a95b`` are neither Max, Plus, nor
    Flash; ``qwen3.6-max-preview`` predates the Max 3.7 floor.
    """

    @override
    def encode(self, request: Request, *, model: OpenAIModel) -> dict[str, object]:
        body = super().encode(request, model=model)
        capability, settings = model.capability, model.settings
        # Spelled in OpenAI's words (``xhigh`` is ``high``), which DashScope
        # rejects or reads otherwise, so it is derived again from the row.
        body.pop("reasoning_effort", None)
        cap = body.pop("max_completion_tokens", body.pop("max_tokens", None))
        if cap is not None:
            field = (
                "max_completion_tokens"
                if capability.model_id in self.completion_cap
                else "max_tokens"
            )
            body[field] = cap
        # A row offering only ``none`` rejects the switch (``-instruct``,
        # ``-coder``); it is never sent one.
        if capability.thinking_budget == frozenset({"none"}):
            return body
        stopped = "none" in (settings.thinking_budget, settings.thinking_effort)
        body["enable_thinking"] = not stopped
        if stopped:
            return body
        # Qwen 3.8 fails when ``reasoning_effort`` and ``thinking_budget``
        # arrive together, so a level row never sends the cap.
        if capability.effort_as_level:
            body["reasoning_effort"] = settings.thinking_effort
        elif settings.thinking_budget == "fixed":
            body["thinking_budget"] = {
                "none": 0,
                "min": 1024,
                "low": 4096,
                "medium": 8192,
                "high": 16_384,
                "xhigh": 20_480,
                "max": 24_576,
            }[settings.thinking_effort]
        return body
