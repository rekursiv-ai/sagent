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

from typing import ClassVar, Final, override

from sagent.lib.custom_json import MutableJSON
from sagent.providers.lib.cost import ModelProfile, Pricing
from sagent.providers.openai_compat import (
    OpenAICompat,
    OpenAICompatModel,
)
from sagent.types.model import ModelRequest


# Qwen3 "thinking" models always emit reasoning. Hybrid models toggle.
# Prefixes are stored WITHOUT a trailing hyphen so they match both the
# hyphenated ids (``qwen3-32b``) and the dotted generation ids
# (``qwen3.6-plus``, the default). A bare ``qwen3-`` silently excluded the
# entire qwen3.6 family -- including ``DEFAULT_MODEL`` -- from the effort knob.
_THINKING_PREFIXES: Final = (
    "qwen3",
    "qwen-plus",
    "qwen-max",
    "qwq",
    "qvq",
)

# Non-reasoning variants that share a thinking prefix but reject the
# ``enable_thinking`` / ``thinking_budget`` knobs: the ``instruct`` and
# ``coder`` qwen3 ids (e.g. ``qwen3-235b-a22b-instruct-2507``,
# ``qwen3-coder-480b-a35b-instruct``). Matching the bare ``qwen3`` prefix is
# necessary to include the hybrid ``qwen3.6`` family, but these markers carve the
# pure non-reasoning models back out -- the same wire hazard the ``-thinking``
# suffix strip guards against, from the other direction. Markers match as whole
# hyphen-delimited SEGMENTS (not substrings), so a hypothetical ``qwen3-
# instructive-…`` id is NOT mis-flagged.
_NON_REASONING_MARKERS = frozenset({"instruct", "coder"})


def _is_non_reasoning_variant(model_id: str) -> bool:
    """True if ``model_id`` carries a non-reasoning marker as a hyphen segment."""
    return bool(_NON_REASONING_MARKERS.intersection(model_id.split("-")))


# Map sagent's effort levels onto Qwen's ``thinking_budget`` (max reasoning
# tokens). ``none`` is absent: it toggles ``enable_thinking=False`` instead, so
# no budget applies. Mirrors Google's per-level budget table so the effort knob
# drives reasoning depth rather than collapsing to an on/off bool.
_DASHSCOPE_THINKING_BUDGETS: Final = {
    "minimal": 1_024,
    "low": 4_096,
    "medium": 8_192,
    "high": 16_384,
    "xhigh": 24_576,
    "max": 32_768,
}


class _DashScopeModel(OpenAICompatModel):
    """DashScope backend - reasoning_content, enable_thinking routing."""

    _reasoning_field: ClassVar[str | None] = "reasoning_content"

    @override
    def _is_effort_model(self, model_id: str) -> bool:
        """True for Qwen3/QwQ/QvQ models that accept ``enable_thinking``.

        A thinking prefix is necessary but not sufficient: the ``-instruct`` /
        ``-coder`` qwen3 ids share the prefix yet are pure non-reasoning models
        that reject the toggle, so they are excluded here.
        """
        if _is_non_reasoning_variant(model_id):
            return False
        return any(model_id.startswith(p) for p in _THINKING_PREFIXES)

    @override
    def _transform_body(
        self,
        body: MutableJSON,
        request: ModelRequest,
    ) -> MutableJSON:
        """Map sagent's effort onto DashScope's thinking knobs.

        DashScope rejects ``reasoning_effort``; it exposes ``enable_thinking``
        (on/off) plus an optional ``thinking_budget`` reasoning-token cap. Read
        the RAW ``request.effort`` rather than the wire-mapped value the base
        already wrote into ``reasoning_effort`` -- the base maps ``none`` to
        ``minimal`` for OpenAI, which would otherwise read as "thinking on" and
        discard the level entirely.
        """
        body.pop("reasoning_effort", None)
        # Gate on the SAME predicate ``supports_effort`` exposes: a model that is
        # not an effort model (no thinking prefix, or a non-reasoning
        # ``-instruct``/``-coder`` variant, or one with no prefix at all like
        # ``qwen-turbo``) must never receive ``enable_thinking``/``thinking_budget``.
        # Using one predicate keeps this wire-side guard from drifting away from
        # ``_is_effort_model``.
        if not self._is_effort_model(self._model_id):
            return body
        effort = request.effort
        if effort is not None:
            body["enable_thinking"] = effort != "none"
            budget = _DASHSCOPE_THINKING_BUDGETS.get(effort)
            if budget is not None:
                body["thinking_budget"] = budget
        # The *-thinking suffix models are always-on reasoning: they reject the
        # ``enable_thinking`` toggle, and forwarding a ``thinking_budget`` they
        # may not accept is the same wire hazard -- drop both knobs.
        if self._model_id.endswith(("-thinking", "-thinking-2507")):
            body.pop("enable_thinking", None)
            body.pop("thinking_budget", None)
        return body


# DashScope/Qwen-VL preprocesses images server-side (Qwen resizes via
# min_pixels/max_pixels; object localization is robust 480-2560 px) and
# publishes no hard per-image pixel/byte reject the client must preempt, nor a
# request-body byte ceiling. Use the 0=unlimited sentinel rather than borrowing
# OpenAI's caps (verified Jun 2026;
# https://www.alibabacloud.com/help/en/model-studio/vision).
_IMAGE_DIM: Final = 0
_IMAGE_BYTES: Final = 0
_REQUEST_BYTES: Final = 0


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
            max_image_dim=_IMAGE_DIM,
            max_image_bytes=_IMAGE_BYTES,
            max_request_bytes=_REQUEST_BYTES,
        ),
        "qwen3.6-plus": ModelProfile(
            max_request_tokens=1_000_000,
            max_response_tokens=65_536,
            pricing=Pricing(
                request=0.50,
                response=3.00,
            ),
            max_image_dim=_IMAGE_DIM,
            max_image_bytes=_IMAGE_BYTES,
            max_request_bytes=_REQUEST_BYTES,
        ),
        "qwen3.6-flash": ModelProfile(
            max_request_tokens=1_000_000,
            max_response_tokens=65_536,
            pricing=Pricing(
                request=0.05,
                response=0.20,
            ),
            max_image_dim=_IMAGE_DIM,
            max_image_bytes=_IMAGE_BYTES,
            max_request_bytes=_REQUEST_BYTES,
        ),
        "qwen3-235b-a22b-instruct-2507": ModelProfile(
            max_request_tokens=262_144,
            max_response_tokens=65_536,
            pricing=Pricing(
                request=0.70,
                response=2.80,
            ),
            max_image_dim=_IMAGE_DIM,
            max_image_bytes=_IMAGE_BYTES,
            max_request_bytes=_REQUEST_BYTES,
        ),
        "qwen3-235b-a22b-thinking-2507": ModelProfile(
            max_request_tokens=262_144,
            max_response_tokens=32_768,
            pricing=Pricing(
                request=0.70,
                response=8.40,
            ),
            max_image_dim=_IMAGE_DIM,
            max_image_bytes=_IMAGE_BYTES,
            max_request_bytes=_REQUEST_BYTES,
        ),
        "qwen3-30b-a3b-instruct-2507": ModelProfile(
            max_request_tokens=262_144,
            max_response_tokens=65_536,
            pricing=Pricing(
                request=0.20,
                response=0.80,
            ),
            max_image_dim=_IMAGE_DIM,
            max_image_bytes=_IMAGE_BYTES,
            max_request_bytes=_REQUEST_BYTES,
        ),
        "qwen3-32b": ModelProfile(
            max_request_tokens=262_144,
            max_response_tokens=65_536,
            pricing=Pricing(
                request=0.40,
                response=1.20,
            ),
            max_image_dim=_IMAGE_DIM,
            max_image_bytes=_IMAGE_BYTES,
            max_request_bytes=_REQUEST_BYTES,
        ),
        "qwen3-coder-480b-a35b-instruct": ModelProfile(
            max_request_tokens=262_144,
            max_response_tokens=65_536,
            pricing=Pricing(
                request=1.00,
                response=5.00,
            ),
            max_image_dim=_IMAGE_DIM,
            max_image_bytes=_IMAGE_BYTES,
            max_request_bytes=_REQUEST_BYTES,
        ),
        "qwen-plus": ModelProfile(
            max_request_tokens=1_000_000,
            max_response_tokens=32_768,
            pricing=Pricing(
                request=0.40,
                response=1.20,
            ),
            max_image_dim=_IMAGE_DIM,
            max_image_bytes=_IMAGE_BYTES,
            max_request_bytes=_REQUEST_BYTES,
        ),
        "qwen-max": ModelProfile(
            max_request_tokens=262_144,
            max_response_tokens=65_536,
            pricing=Pricing(
                request=1.60,
                response=6.40,
            ),
            max_image_dim=_IMAGE_DIM,
            max_image_bytes=_IMAGE_BYTES,
            max_request_bytes=_REQUEST_BYTES,
        ),
        "qwen-turbo": ModelProfile(
            max_request_tokens=1_000_000,
            max_response_tokens=32_768,
            pricing=Pricing(
                request=0.05,
                response=0.20,
            ),
            max_image_dim=_IMAGE_DIM,
            max_image_bytes=_IMAGE_BYTES,
            max_request_bytes=_REQUEST_BYTES,
        ),
    }
    MODEL_CLASS: ClassVar[type[OpenAICompatModel]] = _DashScopeModel
