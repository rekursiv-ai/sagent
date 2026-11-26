"""xAI's stateless Responses API, with local token estimates."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from sagent.providers import xai
from sagent.providers.openai.compat import ResponsesCompatible


if TYPE_CHECKING:
    from types import ModuleType


__all__ = ["XAI"]


class XAI(ResponsesCompatible):
    """Serve Grok using ``XAI_API_KEY`` without an OpenAI counting endpoint.

    xAI's ``max_output_tokens`` caps visible text only, excluding reasoning and
    function calls; it does not enforce a total output-token budget.
    """

    section: ClassVar[str] = "xai"
    aliases: ClassVar[tuple[str, ...]] = ("grok",)
    package: ClassVar[ModuleType] = xai
    url: ClassVar[str] = "https://api.x.ai/v1"
    base_url_variable: ClassVar[str] = "XAI_BASE_URL"
    key_variable: ClassVar[str] = "XAI_API_KEY"
    temperature: ClassVar[float | None] = None
