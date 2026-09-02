"""MiniMax provider - OpenAI chat-completions compatible.

Usage::

    from sagent.providers import MiniMax

    provider = MiniMax.from_env()       # MINIMAX_API_KEY
    model = provider.model()            # MiniMax-M2.7
    response = await model.buffer(request)

Self-hosted::

    provider = MiniMax.from_key("empty", base_url="http://gpu-box:8000/v1")

MiniMax-M2.7 exposes long context and reasoning traces. Tool-calling
uses the standard ``tool_calls`` block.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from sagent.catalog import minimax as minimax_catalog
from sagent.providers.openai.compat import (
    OpenAICompat,
    OpenAICompatModel,
)
from sagent.types.capability import ModelCapability


class _MiniMaxModel(OpenAICompatModel):
    """MiniMax backend - reasoning surfaces via ``reasoning_content``."""

    _reasoning_field: ClassVar[str | None] = "reasoning_content"


# MiniMax (OpenAI-compatible) publishes no per-image pixel or byte limit and no
# request-body byte ceiling; images are preprocessed server-side. Use the
# 0=unlimited sentinel rather than borrowing OpenAI's caps (verified Jun 2026;
# https://platform.minimax.io/docs/api-reference/text-openai-api).


class MiniMax(OpenAICompat):
    """MiniMax provider (api.minimax.io)."""

    DEFAULT_MODEL: ClassVar[str] = "MiniMax-M2.7"
    ENV_VAR: ClassVar[str] = "MINIMAX_API_KEY"
    BASE_URL: ClassVar[str] = "https://api.minimax.io/v1"
    # Model limits and pricing.
    # Source: https://platform.minimaxi.com/document/guides/chat-model/pro
    # Cross-ref: https://github.com/taylorwilsdon/llm-context-limits
    #
    # To add a new model: check the MiniMax platform docs for the
    # model's context window and max output tokens.
    CAPABILITIES: ClassVar[Mapping[str, ModelCapability]] = minimax_catalog.models()
    """Per-model capability; transport limits live on ``TRANSPORT``."""

    MODEL_CLASS: ClassVar[type[OpenAICompatModel]] = _MiniMaxModel
