"""OpenAI provider.

Usage::

    from sagent.providers import OpenAI

    provider = OpenAI.from_key("sk-...")
    # or: export OPENAI_API_KEY=sk-... and use OpenAI.from_env()
    gpt = provider.model("gpt-5.5")
    response = await gpt.buffer(request)
"""

from __future__ import annotations

from typing import ClassVar, override

import dataclasses

from sagent.providers.lib.cost import ModelProfile, Pricing
from sagent.providers.openai_compat import (
    OpenAICompat,
    OpenAICompatModel,
)


__all__ = [
    "OpenAI",
]


class _OpenAIModel(OpenAICompatModel):
    """OpenAI chat-completions backend - o-series accepts effort."""

    @override
    def _is_effort_model(self, model_id: str) -> bool:
        """True for OpenAI reasoning models that accept ``reasoning_effort``."""
        return model_id.startswith(("o1", "o3", "o4", "gpt-5"))

    @property
    @override
    def valid_service_tiers(self) -> tuple[str, ...]:
        """OpenAI chat-completions accepts auto/default/flex/priority."""
        return ("auto", "default", "flex", "priority")

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


# Image / wire byte limits shared by every OpenAI model (verified Jun 2026):
#   - ``max_image_dim`` = 2048: detail:high images are first scaled to fit
#     within a 2048x2048 square before tiling
#     (https://developers.openai.com/api/docs/guides/images-vision); pre-resizing
#     to this caps wire bytes without losing fidelity the model would keep.
#   - ``max_image_bytes`` = 20 MB: documented per-image upload limit.
#   - ``max_request_bytes`` = 20 MB: conservative wire ceiling; OpenAI does not
#     publish a single Responses-API body limit, and the 20 MB per-image figure
#     is the binding practical constraint for image-bearing requests.
_IMAGE_DIM = 2048
_IMAGE_BYTES = 20 * 1024 * 1024
_REQUEST_BYTES = 20 * 1024 * 1024

# Cheap-tier input window for models OpenAI prices in two tiers. For gpt-5.5 /
# gpt-5.4 (and their -pro variants) prompts above 272K input tokens bill at 2x
# input / 1.5x output for the whole session, so the default profile caps the
# window here; a ``+1m`` model id opts into the full window (and the 2x tier).
# https://developers.openai.com/api/docs/models/gpt-5.5
_TWO_TIER_TOKENS = 272_000


def _with_window(profile: ModelProfile, max_request_tokens: int) -> ModelProfile:
    """Clone a profile with a different input-token window."""
    return dataclasses.replace(profile, max_request_tokens=max_request_tokens)


class OpenAI(OpenAICompat):
    """OpenAI provider."""

    # Current frontier default for API-key users; ``+1m`` opts into the full 1M
    # window (mirrors Anthropic's ``+1m`` default). Subscription auth clamps this
    # back to its 272K contract, so the suffix is a no-op on that path.
    DEFAULT_MODEL: ClassVar[str] = "gpt-5.5+1m"
    DEFAULT_UTILITY_MODEL: ClassVar[str] = "gpt-5.4-mini"

    ENV_VAR: ClassVar[str] = "OPENAI_API_KEY"
    BASE_URL: ClassVar[str] = "https://api.openai.com/v1"

    # Model limits and pricing.
    # Limits: https://developers.openai.com/api/docs/models/<model>
    # Pricing: https://developers.openai.com/api/docs/pricing
    # Cross-ref: https://github.com/taylorwilsdon/llm-context-limits
    #
    # To add a new model: look up its page at the limits URL above,
    # find the context window and max output tokens, then add a
    # ModelProfile entry here with the correct pricing from the
    # pricing URL.
    KNOWN_MODELS: ClassVar[dict[str, ModelProfile]] = {
        "gpt-5.5": ModelProfile(
            # Defaults to the 272K cheap tier; ``gpt-5.5+1m`` opts into the full
            # 1M window (billed at the 2x tier above 272K). See ``_TWO_TIER_TOKENS``.
            max_request_tokens=_TWO_TIER_TOKENS,
            max_response_tokens=128_000,
            pricing=Pricing(
                request=5.0,
                response=30.0,
                cache_read=0.5,
            ),
            max_image_dim=_IMAGE_DIM,
            max_image_bytes=_IMAGE_BYTES,
            max_request_bytes=_REQUEST_BYTES,
        ),
        "gpt-5.5-pro": ModelProfile(
            max_request_tokens=_TWO_TIER_TOKENS,
            max_response_tokens=128_000,
            pricing=Pricing(
                request=30.0,
                response=180.0,
            ),
            max_image_dim=_IMAGE_DIM,
            max_image_bytes=_IMAGE_BYTES,
            max_request_bytes=_REQUEST_BYTES,
        ),
        "gpt-5.4": ModelProfile(
            max_request_tokens=_TWO_TIER_TOKENS,
            max_response_tokens=128_000,
            pricing=Pricing(
                request=2.5,
                response=15.0,
                cache_read=0.25,
            ),
            max_image_dim=_IMAGE_DIM,
            max_image_bytes=_IMAGE_BYTES,
            max_request_bytes=_REQUEST_BYTES,
        ),
        "gpt-5.4-pro": ModelProfile(
            max_request_tokens=_TWO_TIER_TOKENS,
            max_response_tokens=128_000,
            pricing=Pricing(
                request=30.0,
                response=180.0,
            ),
            max_image_dim=_IMAGE_DIM,
            max_image_bytes=_IMAGE_BYTES,
            max_request_bytes=_REQUEST_BYTES,
        ),
        "gpt-5.4-mini": ModelProfile(
            max_request_tokens=400_000,
            max_response_tokens=128_000,
            pricing=Pricing(
                request=0.75,
                response=4.5,
                cache_read=0.075,
            ),
            max_image_dim=_IMAGE_DIM,
            max_image_bytes=_IMAGE_BYTES,
            max_request_bytes=_REQUEST_BYTES,
        ),
        "gpt-5.4-nano": ModelProfile(
            max_request_tokens=400_000,
            max_response_tokens=128_000,
            pricing=Pricing(
                request=0.20,
                response=1.25,
                cache_read=0.02,
            ),
            max_image_dim=_IMAGE_DIM,
            max_image_bytes=_IMAGE_BYTES,
            max_request_bytes=_REQUEST_BYTES,
        ),
        "gpt-5.3-codex": ModelProfile(
            max_request_tokens=400_000,
            max_response_tokens=128_000,
            pricing=Pricing(
                request=1.75,
                response=14.0,
                cache_read=0.175,
            ),
            max_image_dim=_IMAGE_DIM,
            max_image_bytes=_IMAGE_BYTES,
            max_request_bytes=_REQUEST_BYTES,
        ),
        "gpt-5.3-codex-spark": ModelProfile(
            max_request_tokens=400_000,
            max_response_tokens=128_000,
            pricing=Pricing(
                request=1.75,
                response=14.0,
                cache_read=0.175,
            ),
            max_image_dim=_IMAGE_DIM,
            max_image_bytes=_IMAGE_BYTES,
            max_request_bytes=_REQUEST_BYTES,
        ),
        "gpt-5.2": ModelProfile(
            max_request_tokens=400_000,
            max_response_tokens=128_000,
            pricing=Pricing(
                request=1.75,
                response=14.0,
                cache_read=0.175,
            ),
            max_image_dim=_IMAGE_DIM,
            max_image_bytes=_IMAGE_BYTES,
            max_request_bytes=_REQUEST_BYTES,
        ),
        "gpt-5.3-chat-latest": ModelProfile(
            max_request_tokens=128_000,
            max_response_tokens=16_384,
            pricing=Pricing(
                request=1.75,
                response=14.0,
                cache_read=0.175,
            ),
            max_image_dim=_IMAGE_DIM,
            max_image_bytes=_IMAGE_BYTES,
            max_request_bytes=_REQUEST_BYTES,
        ),
        "gpt-4.1": ModelProfile(
            max_request_tokens=1_047_576,
            max_response_tokens=32_768,
            pricing=Pricing(
                request=2.0,
                response=8.0,
                cache_read=0.5,
            ),
            max_image_dim=_IMAGE_DIM,
            max_image_bytes=_IMAGE_BYTES,
            max_request_bytes=_REQUEST_BYTES,
        ),
        "gpt-4.1-mini": ModelProfile(
            max_request_tokens=1_047_576,
            max_response_tokens=32_768,
            pricing=Pricing(
                request=0.4,
                response=1.6,
                cache_read=0.1,
            ),
            max_image_dim=_IMAGE_DIM,
            max_image_bytes=_IMAGE_BYTES,
            max_request_bytes=_REQUEST_BYTES,
        ),
        "gpt-4.1-nano": ModelProfile(
            max_request_tokens=1_047_576,
            max_response_tokens=32_768,
            pricing=Pricing(
                request=0.1,
                response=0.4,
                cache_read=0.025,
            ),
            max_image_dim=_IMAGE_DIM,
            max_image_bytes=_IMAGE_BYTES,
            max_request_bytes=_REQUEST_BYTES,
        ),
        "gpt-4o": ModelProfile(
            max_request_tokens=128_000,
            max_response_tokens=16_384,
            pricing=Pricing(
                request=2.5,
                response=10.0,
                cache_read=1.25,
            ),
            max_image_dim=_IMAGE_DIM,
            max_image_bytes=_IMAGE_BYTES,
            max_request_bytes=_REQUEST_BYTES,
        ),
        "gpt-4o-mini": ModelProfile(
            max_request_tokens=128_000,
            max_response_tokens=16_384,
            pricing=Pricing(
                request=0.15,
                response=0.6,
                cache_read=0.075,
            ),
            max_image_dim=_IMAGE_DIM,
            max_image_bytes=_IMAGE_BYTES,
            max_request_bytes=_REQUEST_BYTES,
        ),
        "gpt-4-turbo": ModelProfile(
            max_request_tokens=128_000,
            max_response_tokens=4_096,
            pricing=Pricing(
                request=10.0,
                response=30.0,
            ),
            max_image_dim=_IMAGE_DIM,
            max_image_bytes=_IMAGE_BYTES,
            max_request_bytes=_REQUEST_BYTES,
        ),
        "gpt-4": ModelProfile(
            max_request_tokens=8_192,
            max_response_tokens=8_192,
            pricing=Pricing(
                request=30.0,
                response=60.0,
            ),
            max_image_dim=_IMAGE_DIM,
            max_image_bytes=_IMAGE_BYTES,
            max_request_bytes=_REQUEST_BYTES,
        ),
        "o1": ModelProfile(
            max_request_tokens=200_000,
            max_response_tokens=100_000,
            pricing=Pricing(
                request=15.0,
                response=60.0,
                cache_read=7.5,
            ),
            max_image_dim=_IMAGE_DIM,
            max_image_bytes=_IMAGE_BYTES,
            max_request_bytes=_REQUEST_BYTES,
        ),
        "o1-mini": ModelProfile(
            max_request_tokens=128_000,
            max_response_tokens=65_536,
            pricing=Pricing(
                request=3.0,
                response=12.0,
                cache_read=1.5,
            ),
            max_image_dim=_IMAGE_DIM,
            max_image_bytes=_IMAGE_BYTES,
            max_request_bytes=_REQUEST_BYTES,
        ),
        "o3-mini": ModelProfile(
            max_request_tokens=200_000,
            max_response_tokens=100_000,
            pricing=Pricing(
                request=1.1,
                response=4.4,
                cache_read=0.55,
            ),
            max_image_dim=_IMAGE_DIM,
            max_image_bytes=_IMAGE_BYTES,
            max_request_bytes=_REQUEST_BYTES,
        ),
    }
    # Long-context (``+1m``) variants. For the two-tier models the base id caps at
    # ``_TWO_TIER_TOKENS`` and ``+1m`` opts into the model's full input window; for
    # gpt-4.1 (single flat price, no tier cliff) ``+1m`` is an alias of the base,
    # so both ids resolve to the same full window.
    KNOWN_MODELS.update(
        {
            "gpt-5.5+1m": _with_window(KNOWN_MODELS["gpt-5.5"], 1_000_000),
            "gpt-5.5-pro+1m": _with_window(KNOWN_MODELS["gpt-5.5-pro"], 1_050_000),
            "gpt-5.4+1m": _with_window(KNOWN_MODELS["gpt-5.4"], 1_050_000),
            "gpt-5.4-pro+1m": _with_window(KNOWN_MODELS["gpt-5.4-pro"], 1_050_000),
            "gpt-4.1+1m": KNOWN_MODELS["gpt-4.1"],
            "gpt-4.1-mini+1m": KNOWN_MODELS["gpt-4.1-mini"],
            "gpt-4.1-nano+1m": KNOWN_MODELS["gpt-4.1-nano"],
        }
    )
    MODEL_CLASS: ClassVar[type[OpenAICompatModel]] = _OpenAIModel
