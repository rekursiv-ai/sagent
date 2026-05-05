"""MiniMax provider - OpenAI chat-completions compatible.

Usage::

    from sagent.providers import MiniMax

    provider = MiniMax.from_env()       # MINIMAX_API_KEY
    model = provider.model()            # MiniMax-M1
    response = await model.buffer(request)

Self-hosted::

    provider = MiniMax.from_key("empty", base_url="http://gpu-box:8000/v1")

MiniMax-M1 exposes long context (1M tokens) and lightning-attention
reasoning. Tool-calling uses the standard ``tool_calls`` block.
"""

from __future__ import annotations

from typing import ClassVar

from sagent.providers.lib.cost import ModelProfile, Pricing
from sagent.providers.openai_compat import (
    OpenAICompat,
    OpenAICompatModel,
)


class _MiniMaxModel(OpenAICompatModel):
    """MiniMax backend - reasoning surfaces via ``reasoning_content``."""

    _reasoning_field: ClassVar[str | None] = "reasoning_content"


class MiniMax(OpenAICompat):
    """MiniMax provider (api.minimax.io)."""

    DEFAULT_MODEL: ClassVar[str] = "MiniMax-M1"
    ENV_VAR: ClassVar[str] = "MINIMAX_API_KEY"
    BASE_URL: ClassVar[str] = "https://api.minimax.io/v1"
    # Model limits and pricing.
    # Source: https://platform.minimaxi.com/document/guides/chat-model/pro
    # Cross-ref: https://github.com/taylorwilsdon/llm-context-limits
    #
    # To add a new model: check the MiniMax platform docs for the
    # model's context window and max output tokens.
    KNOWN_MODELS: ClassVar[dict[str, ModelProfile]] = {
        "MiniMax-M1": ModelProfile(
            max_request_tokens=1_000_000,
            max_response_tokens=16_384,
            pricing=Pricing(
                request=0.40,
                response=2.20,
            ),
        ),
        "MiniMax-Text-01": ModelProfile(
            max_request_tokens=1_000_000,
            max_response_tokens=16_384,
            pricing=Pricing(
                request=0.20,
                response=1.10,
            ),
        ),
        "abab6.5s-chat": ModelProfile(
            max_request_tokens=245_000,
            max_response_tokens=16_384,
            pricing=Pricing(
                request=0.15,
                response=0.15,
            ),
        ),
        "abab6.5-chat": ModelProfile(
            max_request_tokens=245_000,
            max_response_tokens=16_384,
            pricing=Pricing(
                request=1.50,
                response=1.50,
            ),
        ),
    }
    MODEL_CLASS: ClassVar[type[OpenAICompatModel]] = _MiniMaxModel
