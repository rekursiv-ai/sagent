"""NVIDIA NIM models through its OpenAI-compatible Chat Completions API."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from sagent.providers import nvidia
from sagent.providers.openai.compat import OpenAICompatible


if TYPE_CHECKING:
    from types import ModuleType


__all__ = ["NVIDIA"]


class NVIDIA(OpenAICompatible):
    """Serve NVIDIA NIM using ``NVIDIA_API_KEY`` and caller-priced ``models``.

    No production token rates are published for the hosted trial endpoint, so
    the shipped table is empty. Supply a ``ModelTable`` with the deployment's
    actual prices through the inherited ``models`` constructor argument.
    """

    section: ClassVar[str] = "nvidia"
    aliases: ClassVar[tuple[str, ...]] = ("nim",)
    package: ClassVar[ModuleType] = nvidia
    url: ClassVar[str] = "https://integrate.api.nvidia.com/v1"
    key_variable: ClassVar[str] = "NVIDIA_API_KEY"
    base_url_variable: ClassVar[str] = "NVIDIA_BASE_URL"
    max_tokens_field: ClassVar[str] = "max_tokens"
    reasoning_field: ClassVar[str] = "reasoning_content"
    temperature: ClassVar[float | None] = None
