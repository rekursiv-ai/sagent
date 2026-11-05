"""Hugging Face models through its OpenAI-compatible inference router."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from sagent.providers import huggingface
from sagent.providers.openai.compat import OpenAICompatible


if TYPE_CHECKING:
    from types import ModuleType


__all__ = ["HuggingFace"]


class HuggingFace(OpenAICompatible):
    """Route models through Hugging Face using ``HF_TOKEN``.

    The shipped GPT-OSS row pins Groq so its catalog price cannot silently
    drift when Hugging Face changes its automatic provider selection.
    """

    section: ClassVar[str] = "huggingface"
    aliases: ClassVar[tuple[str, ...]] = ("hf",)
    package: ClassVar[ModuleType] = huggingface
    url: ClassVar[str] = "https://router.huggingface.co/v1"
    key_variable: ClassVar[str] = "HF_TOKEN"
    base_url_variable: ClassVar[str] = "HF_BASE_URL"
    max_tokens_field: ClassVar[str] = "max_tokens"
    reasoning_field: ClassVar[str] = "reasoning"
