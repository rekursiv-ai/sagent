"""Moonshot provider - OpenAI chat-completions compatible.

Usage::

    from sagent.providers import Moonshot

    provider = Moonshot.from_env()          # MOONSHOT_API_KEY
    model = provider.model()            # kimi-k2.6
    response = await model.buffer(request)

Self-hosted::

    provider = Moonshot.from_key("no-auth", base_url="http://gpu-box:8000/v1")

Moonshot streams reasoning text via ``reasoning_content`` (same as
DeepSeek/DashScope). Tool-calling uses the standard ``tool_calls`` block.
"""

from __future__ import annotations

from typing import ClassVar, Final

from sagent.providers.lib.cost import ModelProfile, Pricing
from sagent.providers.openai_compat import (
    OpenAICompat,
    OpenAICompatModel,
)


class _MoonshotModel(OpenAICompatModel):
    """Moonshot backend - surfaces ``reasoning_content`` as thinking."""

    _reasoning_field: ClassVar[str | None] = "reasoning_content"


# Moonshot/Kimi (OpenAI-compatible) publishes no per-image pixel or byte limit
# and no request-body byte ceiling; images are preprocessed server-side. Use the
# 0=unlimited sentinel rather than borrowing OpenAI's caps (verified Jun 2026;
# https://platform.kimi.ai/docs/guide/use-kimi-vision-model).
_IMAGE_DIM: Final = 0
_IMAGE_BYTES: Final = 0
_REQUEST_BYTES: Final = 0


class Moonshot(OpenAICompat):
    """Moonshot AI provider."""

    DEFAULT_MODEL: ClassVar[str] = "kimi-k2.6"
    ENV_VAR: ClassVar[str] = "MOONSHOT_API_KEY"
    BASE_URL: ClassVar[str] = "https://api.moonshot.ai/v1"
    # Model limits and pricing.
    # Source: https://platform.moonshot.cn/docs/api/chat
    # Cross-ref: https://github.com/taylorwilsdon/llm-context-limits
    #
    # To add a new model: check the Moonshot platform docs for the
    # model's context window and max output tokens.
    KNOWN_MODELS: ClassVar[dict[str, ModelProfile]] = {
        "kimi-k2.6": ModelProfile(
            max_request_tokens=256_000,
            max_response_tokens=96_000,
            pricing=Pricing(
                request=0.95,
                response=4.00,
                cache_read=0.16,
            ),
            max_image_dim=_IMAGE_DIM,
            max_image_bytes=_IMAGE_BYTES,
            max_request_bytes=_REQUEST_BYTES,
        ),
        "kimi-k2.5": ModelProfile(
            max_request_tokens=256_000,
            max_response_tokens=96_000,
            pricing=Pricing(
                request=0.60,
                response=3.00,
                cache_read=0.10,
            ),
            max_image_dim=_IMAGE_DIM,
            max_image_bytes=_IMAGE_BYTES,
            max_request_bytes=_REQUEST_BYTES,
        ),
        "kimi-k2-0905-preview": ModelProfile(
            max_request_tokens=256_000,
            max_response_tokens=32_768,
            pricing=Pricing(
                request=0.60,
                response=2.50,
                cache_read=0.15,
            ),
            max_image_dim=_IMAGE_DIM,
            max_image_bytes=_IMAGE_BYTES,
            max_request_bytes=_REQUEST_BYTES,
        ),
        "kimi-k2-0711-preview": ModelProfile(
            max_request_tokens=131_072,
            max_response_tokens=32_768,
            pricing=Pricing(
                request=0.60,
                response=2.50,
                cache_read=0.15,
            ),
            max_image_dim=_IMAGE_DIM,
            max_image_bytes=_IMAGE_BYTES,
            max_request_bytes=_REQUEST_BYTES,
        ),
        "kimi-k2-turbo-preview": ModelProfile(
            max_request_tokens=256_000,
            max_response_tokens=32_768,
            pricing=Pricing(
                request=1.20,
                response=5.00,
                cache_read=0.30,
            ),
            max_image_dim=_IMAGE_DIM,
            max_image_bytes=_IMAGE_BYTES,
            max_request_bytes=_REQUEST_BYTES,
        ),
        "moonshot-v1-8k": ModelProfile(
            max_request_tokens=8_000,
            max_response_tokens=16_384,
            pricing=Pricing(
                request=0.20,
                response=2.00,
            ),
            max_image_dim=_IMAGE_DIM,
            max_image_bytes=_IMAGE_BYTES,
            max_request_bytes=_REQUEST_BYTES,
        ),
        "moonshot-v1-32k": ModelProfile(
            max_request_tokens=32_000,
            max_response_tokens=16_384,
            pricing=Pricing(
                request=0.40,
                response=4.00,
            ),
            max_image_dim=_IMAGE_DIM,
            max_image_bytes=_IMAGE_BYTES,
            max_request_bytes=_REQUEST_BYTES,
        ),
        "moonshot-v1-128k": ModelProfile(
            max_request_tokens=128_000,
            max_response_tokens=16_384,
            pricing=Pricing(
                request=0.60,
                response=6.00,
            ),
            max_image_dim=_IMAGE_DIM,
            max_image_bytes=_IMAGE_BYTES,
            max_request_bytes=_REQUEST_BYTES,
        ),
    }
    MODEL_CLASS: ClassVar[type[OpenAICompatModel]] = _MoonshotModel
