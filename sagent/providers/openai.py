"""OpenAI provider.

Usage::

    from sagent.providers import OpenAI

    provider = OpenAI.from_key("sk-...")
    # or: export OPENAI_API_KEY=sk-... and use OpenAI.from_env()
    gpt = provider.model("gpt-5.6-sol")
    response = await gpt.buffer(request)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar, Final, override

from sagent.providers import openai_catalog
from sagent.providers.openai_compat import (
    OpenAICompat,
    OpenAICompatModel,
)
from sagent.types.model import ModelCapability


__all__ = [
    "OpenAI",
]


class _OpenAIModel(OpenAICompatModel):
    """OpenAI chat-completions backend - o-series accepts effort."""

    @property
    @override
    def valid_latency_modes(self) -> tuple[str, ...]:
        """``latency="fast"`` maps to ``service_tier="priority"``.

        OpenAI has no separate fast-mode field: its fast path is the
        ``priority`` processing tier (lower queue latency at higher cost).
        This contrasts with Anthropic, whose fast mode is a distinct
        ``speed="fast"`` inference-acceleration field, orthogonal to its
        ``service_tier``. The cross-provider ``latency`` hint papers over
        that difference; see ``effective_service_tier``.
        """
        return ("fast",)


# Conservative compatibility caps retained for pre-GPT-5.6 profiles:
#   - ``max_image_dim`` = 2048: detail:high images are first scaled to fit
#     within a 2048x2048 square before tiling
#     (https://developers.openai.com/api/docs/guides/images-vision); pre-resizing
#     to this caps wire bytes without losing fidelity the model would keep.
#   - The 20 MB byte ceilings preserve their established local resize/compaction
#     behavior. They are not the current GPT-5.6 limits; those are declared
#     separately below.

# GPT-5.6 ``auto``/``original`` preserves source dimensions. Current image
# requirements publish a 512 MB TOTAL request-payload cap and no separate
# per-image byte cap, so ``0`` keeps each image unmodified while the request
# guard enforces the aggregate ceiling.
# https://developers.openai.com/api/docs/guides/images-vision

# Cheap-tier input window for models OpenAI prices in two tiers. For GPT-5.6,
# gpt-5.5, gpt-5.4, and their applicable variants, prompts above 272K input
# tokens bill at 2x input / 1.5x output for the whole session. The base profile
# caps the window here; a ``+1m`` model id opts into the full window.
# https://developers.openai.com/api/docs/models/gpt-5.6-sol
_TWO_TIER_TOKENS: Final = 272_000


class OpenAI(OpenAICompat):
    """OpenAI provider."""

    # Current frontier default for API-key users; ``+1m`` opts into the full
    # 1.05M window. Subscription auth clamps this back to its 272K contract, so
    # the suffix is a no-op on that path.
    DEFAULT_MODEL: ClassVar[str] = "gpt-5.6-sol+1m"
    DEFAULT_UTILITY_MODEL: ClassVar[str] = "gpt-5.4-mini"

    ENV_VAR: ClassVar[str] = "OPENAI_API_KEY"
    BASE_URL: ClassVar[str] = "https://api.openai.com/v1"

    # Model limits and pricing.
    # Limits: https://developers.openai.com/api/docs/models/<model>
    # Pricing: https://developers.openai.com/api/docs/pricing
    # Cross-ref: https://github.com/taylorwilsdon/llm-context-limits
    #
    CAPABILITIES: ClassVar[Mapping[str, ModelCapability]] = openai_catalog.CHAT_MODELS
    """Chat Completions rows: GPT-5.6 ``max`` downgrades to ``xhigh``."""

    TRANSPORT: ClassVar[ModelCapability] = openai_catalog.API
    """What this transport lets through; subclasses declare their own."""

    MODEL_CLASS: ClassVar[type[OpenAICompatModel]] = _OpenAIModel
