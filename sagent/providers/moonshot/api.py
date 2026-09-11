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

from collections.abc import Mapping
from typing import ClassVar

from sagent.catalog import moonshot as moonshot_catalog
from sagent.providers.openai.compat import (
    OpenAICompat,
    OpenAICompatModel,
)
from sagent.types.capability import ModelCapability


class _MoonshotModel(OpenAICompatModel):
    """Moonshot backend - surfaces ``reasoning_content`` as thinking."""

    _reasoning_field: ClassVar[str | None] = "reasoning_content"


# Moonshot/Kimi (OpenAI-compatible) publishes no per-image pixel or byte limit
# and no request-body byte ceiling; images are preprocessed server-side. Use the
# 0=unlimited sentinel rather than borrowing OpenAI's caps (verified Jun 2026;
# https://platform.kimi.ai/docs/guide/use-kimi-vision-model).


class Moonshot(OpenAICompat):
    """Moonshot AI provider."""

    DEFAULT_MODEL: ClassVar[str] = "kimi-k2.6"
    # Cheapest row that keeps the default's 256k window: the ``moonshot-v1``
    # ids undercut it but cap at 8k-128k, and a utility model still has to
    # hold the conversation it is summarizing. Without this,
    # ``utility_model()`` falls back to the default and bills 1.6x.
    DEFAULT_UTILITY_MODEL: ClassVar[str] = "kimi-k2-0905-preview"
    ENV_VAR: ClassVar[str] = "MOONSHOT_API_KEY"
    BASE_URL: ClassVar[str] = "https://api.moonshot.ai/v1"
    # Model limits and pricing.
    # Source: https://platform.moonshot.cn/docs/api/chat
    # Cross-ref: https://github.com/taylorwilsdon/llm-context-limits
    #
    # To add a new model: check the Moonshot platform docs for the
    # model's context window and max output tokens.
    CAPABILITIES: ClassVar[Mapping[str, ModelCapability]] = moonshot_catalog.models()
    """Per-model capability; transport limits live on ``TRANSPORT``."""

    MODEL_CLASS: ClassVar[type[OpenAICompatModel]] = _MoonshotModel
