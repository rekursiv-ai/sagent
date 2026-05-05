"""OpenAI provider.

Usage::

    from sagent.providers import OpenAI

    provider = OpenAI.from_key("sk-...")
    # or: export OPENAI_API_KEY=sk-... and use OpenAI.from_env()
    gpt = provider.model("gpt-4o")
    response = await gpt.buffer(request)
"""

from __future__ import annotations

from typing import ClassVar, override

from sagent.providers.lib.cost import ModelProfile, Pricing
from sagent.providers.openai_compat import (
    OpenAICompat,
    OpenAICompatModel,
    build_messages,
    parse_response,
)


__all__ = [
    "OpenAI",
    "build_messages",
    "parse_response",
]


class _OpenAIModel(OpenAICompatModel):
    """OpenAI chat-completions backend - o-series accepts effort."""

    @override
    def _is_effort_model(self, model_id: str) -> bool:
        return model_id.startswith(("o1", "o3", "o4", "gpt-5"))


class OpenAI(OpenAICompat):
    """OpenAI provider."""

    # Best value for API-key users ($2.50/$10 per 1M tok).
    DEFAULT_MODEL: ClassVar[str] = "gpt-4o"
    DEFAULT_UTILITY_MODEL: ClassVar[str] = "gpt-4o-mini"

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
            max_request_tokens=1_050_000,
            max_response_tokens=128_000,
            pricing=Pricing(
                request=5.0,
                response=30.0,
                cache_read=0.5,
            ),
        ),
        "gpt-5.5-pro": ModelProfile(
            max_request_tokens=1_050_000,
            max_response_tokens=128_000,
            pricing=Pricing(
                request=30.0,
                response=180.0,
            ),
        ),
        "gpt-5.4": ModelProfile(
            max_request_tokens=1_050_000,
            max_response_tokens=128_000,
            pricing=Pricing(
                request=2.5,
                response=15.0,
                cache_read=0.25,
            ),
        ),
        "gpt-5.4-pro": ModelProfile(
            max_request_tokens=1_050_000,
            max_response_tokens=128_000,
            pricing=Pricing(
                request=30.0,
                response=180.0,
            ),
        ),
        "gpt-5.4-mini": ModelProfile(
            max_request_tokens=400_000,
            max_response_tokens=128_000,
            pricing=Pricing(
                request=0.75,
                response=4.5,
                cache_read=0.075,
            ),
        ),
        "gpt-5.4-nano": ModelProfile(
            max_request_tokens=400_000,
            max_response_tokens=128_000,
            pricing=Pricing(
                request=0.20,
                response=1.25,
                cache_read=0.02,
            ),
        ),
        "gpt-5.3-codex": ModelProfile(
            max_request_tokens=400_000,
            max_response_tokens=128_000,
            pricing=Pricing(
                request=1.75,
                response=14.0,
                cache_read=0.175,
            ),
        ),
        "gpt-5.3-codex-spark": ModelProfile(
            max_request_tokens=400_000,
            max_response_tokens=128_000,
            pricing=Pricing(
                request=1.75,
                response=14.0,
                cache_read=0.175,
            ),
        ),
        "gpt-5.2": ModelProfile(
            max_request_tokens=400_000,
            max_response_tokens=128_000,
            pricing=Pricing(
                request=1.75,
                response=14.0,
                cache_read=0.175,
            ),
        ),
        "gpt-5.3-chat-latest": ModelProfile(
            max_request_tokens=128_000,
            max_response_tokens=16_384,
            pricing=Pricing(
                request=1.75,
                response=14.0,
                cache_read=0.175,
            ),
        ),
        "gpt-4.1": ModelProfile(
            max_request_tokens=1_047_576,
            max_response_tokens=32_768,
            pricing=Pricing(
                request=2.0,
                response=8.0,
                cache_read=0.5,
            ),
        ),
        "gpt-4.1-mini": ModelProfile(
            max_request_tokens=1_047_576,
            max_response_tokens=32_768,
            pricing=Pricing(
                request=0.4,
                response=1.6,
                cache_read=0.1,
            ),
        ),
        "gpt-4.1-nano": ModelProfile(
            max_request_tokens=1_047_576,
            max_response_tokens=32_768,
            pricing=Pricing(
                request=0.1,
                response=0.4,
                cache_read=0.025,
            ),
        ),
        "gpt-4o": ModelProfile(
            max_request_tokens=128_000,
            max_response_tokens=16_384,
            pricing=Pricing(
                request=2.5,
                response=10.0,
                cache_read=1.25,
            ),
        ),
        "gpt-4o-mini": ModelProfile(
            max_request_tokens=128_000,
            max_response_tokens=16_384,
            pricing=Pricing(
                request=0.15,
                response=0.6,
                cache_read=0.075,
            ),
        ),
        "gpt-4-turbo": ModelProfile(
            max_request_tokens=128_000,
            max_response_tokens=4_096,
            pricing=Pricing(
                request=10.0,
                response=30.0,
            ),
        ),
        "gpt-4": ModelProfile(
            max_request_tokens=8_192,
            max_response_tokens=8_192,
            pricing=Pricing(
                request=30.0,
                response=60.0,
            ),
        ),
        "o1": ModelProfile(
            max_request_tokens=200_000,
            max_response_tokens=100_000,
            pricing=Pricing(
                request=15.0,
                response=60.0,
                cache_read=7.5,
            ),
        ),
        "o1-mini": ModelProfile(
            max_request_tokens=128_000,
            max_response_tokens=65_536,
            pricing=Pricing(
                request=3.0,
                response=12.0,
                cache_read=1.5,
            ),
        ),
        "o3-mini": ModelProfile(
            max_request_tokens=200_000,
            max_response_tokens=100_000,
            pricing=Pricing(
                request=1.1,
                response=4.4,
                cache_read=0.55,
            ),
        ),
    }
    MODEL_CLASS: ClassVar[type[OpenAICompatModel]] = _OpenAIModel
