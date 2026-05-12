"""DashScope (Alibaba) provider - OpenAI compatible.

Usage::

    from sagent.providers import DashScope

    provider = DashScope.from_env()          # DASHSCOPE_API_KEY
    model = provider.model()            # qwen3.6-plus
    response = await model.buffer(request)

Self-hosted (vLLM/SGLang on localhost)::

    provider = DashScope.from_key("empty", base_url="http://gpu-box:8000/v1")

Qwen3 surfaces reasoning via ``reasoning_content`` when
``enable_thinking=true`` is passed. The hybrid thinking/non-thinking
models default to non-thinking; set ``effort`` on the request to
switch behavior (mapped to ``enable_thinking`` in the body).
"""

from __future__ import annotations

from typing import ClassVar, override

from sagent.custom_types import ModelRequest
from sagent.lib.json import MutableJSON
from sagent.providers.lib.cost import ModelProfile, Pricing
from sagent.providers.openai_compat import (
    OpenAICompat,
    OpenAICompatModel,
)


# Qwen3 "thinking" models always emit reasoning. Hybrid models toggle.
_THINKING_PREFIXES = (
    "qwen3-",
    "qwen-plus",
    "qwen-max",
    "qwq-",
    "qvq-",
)


class _DashScopeModel(OpenAICompatModel):
    """DashScope backend - reasoning_content, enable_thinking routing."""

    _reasoning_field: ClassVar[str | None] = "reasoning_content"

    @override
    def _is_effort_model(self, model_id: str) -> bool:
        """True for Qwen3/QwQ/QvQ models that accept ``enable_thinking``."""
        # Qwen3 / QwQ / QvQ accept enable_thinking; we translate
        # ``effort`` into that flag via _transform_body.
        return any(model_id.startswith(p) for p in _THINKING_PREFIXES)

    @override
    def _transform_body(
        self,
        body: MutableJSON,
        request: ModelRequest,
    ) -> MutableJSON:
        """Map OpenAI-style ``reasoning_effort`` to DashScope's ``enable_thinking``."""
        # DashScope rejects ``reasoning_effort``; map to ``enable_thinking``.
        effort = body.pop("reasoning_effort", None)
        if effort is not None:
            body["enable_thinking"] = effort != "none"
        # The *-thinking suffix models are always-on reasoning; don't send.
        if self._model_id.endswith(("-thinking", "-thinking-2507")):
            body.pop("enable_thinking", None)
        del request
        return body


class DashScope(OpenAICompat):
    """DashScope (Alibaba) provider."""

    DEFAULT_MODEL: ClassVar[str] = "qwen3.6-plus"
    ENV_VAR: ClassVar[str] = "DASHSCOPE_API_KEY"
    # International endpoint. For mainland China use
    # dashscope.aliyuncs.com via the ``base_url=`` override.
    BASE_URL: ClassVar[str] = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    # Model limits and pricing.
    # Source: https://help.aliyun.com/zh/model-studio/developer-reference/
    # Cross-ref: https://github.com/taylorwilsdon/llm-context-limits
    #
    # To add a new model: check the Alibaba Cloud Model Studio docs
    # for context window and max output tokens.
    KNOWN_MODELS: ClassVar[dict[str, ModelProfile]] = {
        "qwen3.6-max-preview": ModelProfile(
            max_request_tokens=262_144,
            max_response_tokens=65_536,
            pricing=Pricing(
                request=1.60,
                response=6.40,
            ),
        ),
        "qwen3.6-plus": ModelProfile(
            max_request_tokens=1_000_000,
            max_response_tokens=65_536,
            pricing=Pricing(
                request=0.50,
                response=3.00,
            ),
        ),
        "qwen3.6-flash": ModelProfile(
            max_request_tokens=1_000_000,
            max_response_tokens=65_536,
            pricing=Pricing(
                request=0.05,
                response=0.20,
            ),
        ),
        "qwen3-235b-a22b-instruct-2507": ModelProfile(
            max_request_tokens=262_144,
            max_response_tokens=65_536,
            pricing=Pricing(
                request=0.70,
                response=2.80,
            ),
        ),
        "qwen3-235b-a22b-thinking-2507": ModelProfile(
            max_request_tokens=262_144,
            max_response_tokens=32_768,
            pricing=Pricing(
                request=0.70,
                response=8.40,
            ),
        ),
        "qwen3-30b-a3b-instruct-2507": ModelProfile(
            max_request_tokens=262_144,
            max_response_tokens=65_536,
            pricing=Pricing(
                request=0.20,
                response=0.80,
            ),
        ),
        "qwen3-32b": ModelProfile(
            max_request_tokens=262_144,
            max_response_tokens=65_536,
            pricing=Pricing(
                request=0.40,
                response=1.20,
            ),
        ),
        "qwen3-coder-480b-a35b-instruct": ModelProfile(
            max_request_tokens=262_144,
            max_response_tokens=65_536,
            pricing=Pricing(
                request=1.00,
                response=5.00,
            ),
        ),
        "qwen-plus": ModelProfile(
            max_request_tokens=1_000_000,
            max_response_tokens=32_768,
            pricing=Pricing(
                request=0.40,
                response=1.20,
            ),
        ),
        "qwen-max": ModelProfile(
            max_request_tokens=262_144,
            max_response_tokens=65_536,
            pricing=Pricing(
                request=1.60,
                response=6.40,
            ),
        ),
        "qwen-turbo": ModelProfile(
            max_request_tokens=1_000_000,
            max_response_tokens=32_768,
            pricing=Pricing(
                request=0.05,
                response=0.20,
            ),
        ),
    }
    MODEL_CLASS: ClassVar[type[OpenAICompatModel]] = _DashScopeModel
